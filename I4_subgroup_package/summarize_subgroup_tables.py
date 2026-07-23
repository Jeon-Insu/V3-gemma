"""BMI / 60세 기준 서브그룹 eval 결과를 논문용 표 형식으로 정리 (I4 포함 모델)."""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_PREFIX = "20260622_053"
TRAIN_ROOT = ROOT / "train_results" / "train"
EVAL_ROOT = ROOT / "eval_results"
OUT_DIR = ROOT / "summary"

MODELS = [
    ("logistic_regression", "Logistic Regression"),
    ("xgboost", "XGBoost"),
    ("svm", "SVM"),
    ("randomforest", "Random Forest"),
]

SUBGROUPS = [
    ("BMI25_under_I4", "BMI < 25"),
    ("BMI25_upper_I4", "BMI ≥ 25"),
    ("age60_under_I4", "Age < 60"),
    ("age60_upper_I4", "Age ≥ 60"),
]

METRICS = ("AUC", "Accuracy", "Sensitivity", "Specificity", "Brier score")


def _latest_performance_summary(group_dir: Path) -> Path | None:
    files = sorted(group_dir.glob("performance_summary_*.csv"))
    return files[-1] if files else None


def _fmt_auc(value: float, ci: str) -> str:
    return f"{value:.3f} ({ci})"


def _fmt_dev_auc(formatted: str) -> str:
    return formatted.replace("-", "–")


def _load_dev_auc() -> dict[str, str]:
    out: dict[str, str] = {}
    for algo, _ in MODELS:
        algo_dir = TRAIN_ROOT / algo
        runs = sorted(
            d for d in algo_dir.iterdir()
            if d.is_dir() and d.name.startswith(TRAIN_PREFIX)
        )
        if not runs:
            raise FileNotFoundError(f"train run not found: {algo} / {TRAIN_PREFIX}")
        df = pd.read_csv(runs[-1] / "results.csv")
        out[algo] = _fmt_dev_auc(str(df.iloc[0]["AUC_formatted"]))
    return out


def _load_eval_metrics(summary_path: Path) -> dict[str, dict[str, float | str]]:
    df = pd.read_csv(summary_path)
    out: dict[str, dict[str, float | str]] = {}
    for algo, _ in MODELS:
        sub = df[df["Model"] == algo]
        if sub.empty:
            continue
        m: dict[str, float | str] = {}
        for metric in METRICS:
            row = sub[sub["Metric"] == metric]
            if row.empty:
                continue
            r = row.iloc[0]
            m[metric] = float(r["Value"])
            if metric == "AUC":
                m["AUC_CI"] = str(r["Bootstrap_CI_95"])
        out[algo] = m
    return out


def build_table(group_key: str, group_label: str, dev_auc: dict[str, str]) -> pd.DataFrame:
    summary = _latest_performance_summary(EVAL_ROOT / group_key)
    if summary is None:
        raise FileNotFoundError(f"no performance_summary in {group_key}")
    eval_m = _load_eval_metrics(summary)

    rows = []
    for algo, display in MODELS:
        ev = eval_m.get(algo, {})
        rows.append({
            "Model": display,
            "Development AUC (95% CI)": dev_auc[algo],
            "Evaluation AUC (95% CI)": _fmt_auc(ev["AUC"], ev["AUC_CI"])
            if "AUC" in ev else "",
            "Accuracy": round(ev.get("Accuracy", float("nan")), 2),
            "Sensitivity": round(ev.get("Sensitivity", float("nan")), 2),
            "Specificity": round(ev.get("Specificity", float("nan")), 2),
            "Brier Score": round(ev.get("Brier score", float("nan")), 3),
        })
    df = pd.DataFrame(rows)
    df.attrs["group_key"] = group_key
    df.attrs["group_label"] = group_label
    df.attrs["summary_path"] = str(summary)
    return df


def _best_indices(df: pd.DataFrame) -> dict[str, int]:
    """각 수치 컬럼에서 최고 성능 행 인덱스 (Brier는 최소)."""
    best: dict[str, int] = {}
    for col in ("Accuracy", "Sensitivity", "Specificity"):
        best[col] = int(df[col].idxmax())
    best["Brier Score"] = int(df["Brier Score"].idxmin())
    # Eval AUC: parse point estimate
    eval_auc = df["Evaluation AUC (95% CI)"].str.extract(r"^([\d.]+)")[0].astype(float)
    best["Evaluation AUC (95% CI)"] = int(eval_auc.idxmax())
    return best


def to_markdown_table(df: pd.DataFrame, title: str) -> str:
    best = _best_indices(df)
    lines = [f"### {title}", ""]
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for i, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if isinstance(val, float) and pd.isna(val):
                text = ""
            elif isinstance(val, float):
                text = f"{val:.3f}" if col == "Brier Score" else f"{val:.2f}"
            else:
                text = str(val)
            if best.get(col) == i:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dev_auc = _load_dev_auc()
    all_tables: list[pd.DataFrame] = []
    md_parts = [
        "# Subgroup performance tables (I4 models, eval 2026-07-03)",
        "",
        f"Development AUC: train CV from `{TRAIN_PREFIX}*` runs (19 features, n=60 train).",
        "",
    ]

    for group_key, group_label in SUBGROUPS:
        table = build_table(group_key, group_label, dev_auc)
        all_tables.append(table)
        csv_path = OUT_DIR / f"performance_table_{group_key}.csv"
        table.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
        md_parts.append(to_markdown_table(table, group_label))

    combined = []
    for (group_key, group_label), table in zip(SUBGROUPS, all_tables):
        t = table.copy()
        t.insert(0, "Subgroup", group_label)
        combined.append(t)
    combined_df = pd.concat(combined, ignore_index=True)
    combined_path = OUT_DIR / "subgroup_performance_tables_I4.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved {combined_path}")

    md_path = OUT_DIR / "subgroup_performance_tables_I4.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
