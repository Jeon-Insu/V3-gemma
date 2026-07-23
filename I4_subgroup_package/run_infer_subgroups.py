"""
testdataset 및 BMI/나이 분할 CSV 에 대해 infer.py 를 순차 실행.

1) abnormality 라벨 복원 (없을 경우 기존 eval 예측 또는 Final datasets 사용)
2) BMI/나이 분할 CSV 재생성
3) 각 CSV × 5개 모델 추론
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("PYTHON_EXE", sys.executable)
CONFIG = ROOT / "param_config.yaml"
EVAL_ROOT = ROOT / "eval_results"

SUBGROUPS = [
    ("testdataset.csv", "all"),
    ("testdataset_BMI25_under.csv", "BMI25_under"),
    ("testdataset_BMI25_upper.csv", "BMI25_upper"),
    ("testdataset_age60_under.csv", "age60_under"),
    ("testdataset_age60_upper.csv", "age60_upper"),
]

# abnormality 복원용 참조 (패키지에 포함된 all_I4 결과)
REF_PRED = (
    ROOT / "eval_results" / "all_I4" / "randomforest" / "20260703_193654" / "test_predictions.csv"
)


def restore_abnormality(csv_path: Path) -> Path:
    """id 기준으로 abnormality 컬럼 복원. 원본 수정 실패 시 *_labeled.csv 생성."""
    df = pd.read_csv(csv_path)
    if "abnormality" in df.columns and df["abnormality"].notna().all():
        return csv_path
    if not REF_PRED.exists():
        raise FileNotFoundError(f"라벨 복원용 예측 파일 없음: {REF_PRED}")

    base = pd.read_csv(ROOT / "testdataset.csv").reset_index().rename(columns={"index": "idx"})
    pred = pd.read_csv(REF_PRED)
    pred["idx"] = pred["original_row_number"] - 2
    base = base.merge(pred[["idx", "true_label"]], on="idx", how="left")
    label_map = dict(zip(base["id"].astype(str), base["true_label"].astype(int)))

    df["id"] = df["id"].astype(str)
    df["abnormality"] = df["id"].map(label_map)
    missing = df["abnormality"].isna().sum()
    if missing:
        raise ValueError(f"{csv_path.name}: abnormality 매칭 실패 {missing}건")

    try:
        df.to_csv(csv_path, index=False)
        print(f"  ✓ abnormality restored -> {csv_path.name}")
        return csv_path
    except PermissionError:
        labeled = csv_path.with_name(f"{csv_path.stem}_labeled.csv")
        df.to_csv(labeled, index=False)
        print(f"  ✓ abnormality restored -> {labeled.name} (원본 잠김)")
        return labeled


def refresh_splits() -> None:
    """BMI/나이 분할 CSV 재생성."""
    for script in ("split_testdataset_by_bmi.py", "split_testdataset_by_age.py"):
        path = ROOT / script
        print(f"\n>> {script}")
        subprocess.run([PYTHON, str(path)], cwd=ROOT, check=True)


def run_infer(csv_name: str, eval_label: str, weights_run_prefix: str | None = None) -> int:
    csv_path = ROOT / csv_name
    cmd = [
        PYTHON, str(ROOT / "infer.py"),
        "--config", str(CONFIG),
        "--csv", str(csv_path),
        "--weights_dir", str(ROOT / "train_results"),
        "--output_dir", str(EVAL_ROOT),
        "--project", "train",
        "--eval_label", eval_label,
    ]
    if weights_run_prefix:
        cmd.extend(["--weights_run_prefix", weights_run_prefix])
    print(f"\n{'=' * 60}")
    print(f"INFER: {csv_name}  ->  eval_label={eval_label}")
    print(f"{'=' * 60}")
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights_run_prefix", default=None,
                    help="가중치 run timestamp 접두사 (예: 20260622_053)")
    ap.add_argument("--eval_suffix", default="",
                    help="eval_label 접미사 (예: _I4 -> all_I4)")
    cli = ap.parse_args()

    labeled_paths: dict[str, Path] = {}
    print("Step 1: restore abnormality labels")
    labeled_paths["testdataset.csv"] = restore_abnormality(ROOT / "testdataset.csv")

    print("\nStep 2: refresh BMI/age split files")
    refresh_splits()

    for split_csv in [
        "testdataset_BMI25_under.csv",
        "testdataset_BMI25_upper.csv",
        "testdataset_age60_under.csv",
        "testdataset_age60_upper.csv",
    ]:
        labeled_paths[split_csv] = restore_abnormality(ROOT / split_csv)

    results = []
    for csv_name, eval_label in SUBGROUPS:
        csv_path = labeled_paths.get(csv_name, ROOT / csv_name)
        label = f"{eval_label}{cli.eval_suffix}"
        code = run_infer(csv_path.name, label, cli.weights_run_prefix)
        results.append((csv_name, label, code))

    print(f"\n{'=' * 60}")
    print("Inference batch summary")
    print(f"{'=' * 60}")
    ok = 0
    for csv_name, eval_label, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  [{status}] {csv_name} -> eval_results/{eval_label}/")
        if code == 0:
            ok += 1
    print(f"\n{ok}/{len(results)} completed")
    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
