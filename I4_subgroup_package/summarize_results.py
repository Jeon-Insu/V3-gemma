"""train / eval 최신 run 결과를 모델별 종합 표로 정리."""
from __future__ import annotations

import os
import re
from datetime import datetime

import pandas as pd

ALGOS = (
    "randomforest", "xgboost", "svm",
    "logistic_regression", "neural_network",
)
PROJECT = "train"
TRAIN_ROOT = os.path.join("train_results", PROJECT)
EVAL_ROOT = os.path.join("eval_results", PROJECT)
KEY_METRICS = ("AUC", "Accuracy", "Sensitivity", "Specificity", "Brier score")


def _latest_run(base: str, algo: str) -> str | None:
    algo_dir = os.path.join(base, algo)
    if not os.path.isdir(algo_dir):
        return None
    runs = [
        d for d in os.listdir(algo_dir)
        if re.match(r"^\d{8}_\d{6}$", d)
        and os.path.isdir(os.path.join(algo_dir, d))
    ]
    if not runs:
        return None
    return os.path.join(algo_dir, sorted(runs)[-1])


def _read_perf_table(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "performance_table.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _read_train_cv(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "cv_auc": row.get("Mean_AUC"),
        "cv_ci_low": row.get("CI_Low"),
        "cv_ci_high": row.get("CI_High"),
        "cv_auc_formatted": row.get("AUC_formatted"),
        "n_features": row.get("n_features"),
        "brier_test_fold": row.get("Brier_Score"),
    }


def _count_test_samples(run_dir: str) -> int | None:
    path = os.path.join(run_dir, "test_set.csv")
    if os.path.exists(path):
        return len(pd.read_csv(path))
    path = os.path.join(run_dir, "test_predictions.csv")
    if os.path.exists(path):
        return len(pd.read_csv(path))
    return None


def build_summary() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        train_dir = _latest_run(TRAIN_ROOT, algo)
        eval_dir = _latest_run(EVAL_ROOT, algo)
        cv = _read_train_cv(train_dir) if train_dir else None
        train_perf = _read_perf_table(train_dir) if train_dir else None
        eval_perf = _read_perf_table(eval_dir) if eval_dir else None
        train_n = _count_test_samples(train_dir) if train_dir else None

        eval_n = None
        if eval_dir and os.path.exists(os.path.join(eval_dir, "test_predictions.csv")):
            eval_n = len(pd.read_csv(os.path.join(eval_dir, "test_predictions.csv")))

        for metric in KEY_METRICS:
            row = {
                "Model": algo,
                "Metric": metric,
            }
            if cv:
                row["Train_CV_AUC"] = cv["cv_auc"] if metric == "AUC" else None
                row["Train_CV_95CI"] = cv["cv_auc_formatted"] if metric == "AUC" else None
                row["Train_CV_CI_Low"] = cv["cv_ci_low"] if metric == "AUC" else None
                row["Train_CV_CI_High"] = cv["cv_ci_high"] if metric == "AUC" else None
                row["Train_run"] = os.path.basename(train_dir)
                row["n_features"] = cv["n_features"]
            if train_perf is not None:
                m = train_perf[train_perf["Metric"] == metric]
                if not m.empty:
                    row["Train_TestFold_Value"] = m.iloc[0]["Value"]
                    row["Train_TestFold_Bootstrap_CI"] = m.iloc[0]["Bootstrap_CI_95"]
                    row["Train_TestFold_n"] = train_n
            if eval_perf is not None:
                m = eval_perf[eval_perf["Metric"] == metric]
                if not m.empty:
                    row["Eval_Value"] = m.iloc[0]["Value"]
                    row["Eval_Bootstrap_CI"] = m.iloc[0]["Bootstrap_CI_95"]
                    row["Eval_n"] = eval_n
                    row["Eval_run"] = os.path.basename(eval_dir)
            rows.append(row)

    return pd.DataFrame(rows)


def build_wide_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """모델별 wide format (논문용)."""
    wide_rows = []
    for algo in ALGOS:
        sub = long_df[long_df["Model"] == algo]
        if sub.empty:
            continue
        r = {"Model": algo}
        meta = sub.iloc[0]
        r["n_features"] = meta.get("n_features")
        r["Train_run"] = meta.get("Train_run")
        r["Eval_run"] = meta.get("Eval_run")
        r["Train_TestFold_n"] = meta.get("Train_TestFold_n")
        r["Eval_n"] = meta.get("Eval_n")

        auc = sub[sub["Metric"] == "AUC"].iloc[0]
        r["Train_CV_AUC_95CI"] = auc.get("Train_CV_95CI")
        for metric in KEY_METRICS:
            mrow = sub[sub["Metric"] == metric].iloc[0]
            key = metric.replace(" ", "_")
            r[f"Train_TestFold_{key}"] = mrow.get("Train_TestFold_Value")
            r[f"Train_TestFold_{key}_CI"] = mrow.get("Train_TestFold_Bootstrap_CI")
            r[f"Eval_{key}"] = mrow.get("Eval_Value")
            r[f"Eval_{key}_CI"] = mrow.get("Eval_Bootstrap_CI")
        wide_rows.append(r)
    return pd.DataFrame(wide_rows)


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("summary", exist_ok=True)
    long_df = build_summary()
    wide_df = build_wide_summary(long_df)
    long_path = os.path.join("summary", f"train_eval_summary_{ts}.csv")
    wide_path = os.path.join("summary", f"train_eval_summary_wide_{ts}.csv")
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")
    print("\n--- Wide summary ---")
    print(wide_df.to_string(index=False))
