"""
demographics Excel 의 연구번호·BMI 로 testdataset.csv 를 BMI 25 기준 분할 저장.

- BMI < 25  -> testdataset_BMI25_under.csv
- BMI >= 25 -> testdataset_BMI25_upper.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BMI_THRESHOLD = 25.0
DEFAULT_DEMO_XLSX = "demographics new old set 260626_나이BMI.xlsx"
DEFAULT_TEST_CSV = "testdataset.csv"
OUT_UNDER = "testdataset_BMI25_under.csv"
OUT_UPPER = "testdataset_BMI25_upper.csv"


def _find_columns(df: pd.DataFrame) -> tuple[str, str]:
    """연구번호·BMI 컬럼명 탐색 (한글 헤더 또는 위치 기반)."""
    id_col = bmi_col = None
    for col in df.columns:
        name = str(col).strip()
        if "연구번호" in name or name.lower() in ("id", "subject_id", "research_id"):
            id_col = col
        if name.upper() == "BMI" or "bmi" in name.lower():
            bmi_col = col
    if id_col is None and len(df.columns) >= 2:
        id_col = df.columns[1]
    if bmi_col is None and len(df.columns) >= 6:
        bmi_col = df.columns[5]
    if id_col is None or bmi_col is None:
        raise ValueError(f"연구번호/BMI 컬럼을 찾을 수 없습니다: {list(df.columns)}")
    return id_col, bmi_col


def split_testdataset_by_bmi(
    demo_path: str,
    test_path: str,
    out_under: str = OUT_UNDER,
    out_upper: str = OUT_UPPER,
    bmi_threshold: float = BMI_THRESHOLD,
    sheet_name: str | int = 0,
) -> dict:
    demo = pd.read_excel(demo_path, sheet_name=sheet_name)
    id_col, bmi_col = _find_columns(demo)
    demo = demo[[id_col, bmi_col]].copy()
    demo.columns = ["id", "BMI"]
    demo["id"] = demo["id"].astype(str).str.strip()
    demo["BMI"] = pd.to_numeric(demo["BMI"], errors="coerce")
    demo = demo.dropna(subset=["id", "BMI"]).drop_duplicates(subset=["id"], keep="first")

    test = pd.read_csv(test_path)
    if "id" not in test.columns:
        raise ValueError(f"'{test_path}' 에 id 컬럼이 없습니다.")
    test["id"] = test["id"].astype(str).str.strip()

    merged = test.merge(demo, on="id", how="inner", validate="m:1")
    missing = sorted(set(test["id"]) - set(demo["id"]))
    if missing:
        print(f"WARNING: BMI 매칭 실패 id ({len(missing)}): {missing}")

    under = merged[merged["BMI"] < bmi_threshold].drop(columns=["BMI"])
    upper = merged[merged["BMI"] >= bmi_threshold].drop(columns=["BMI"])

    under.to_csv(out_under, index=False)
    upper.to_csv(out_upper, index=False)

    return {
        "test_total": len(test),
        "matched": len(merged),
        "under_n": len(under),
        "upper_n": len(upper),
        "out_under": out_under,
        "out_upper": out_upper,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default=DEFAULT_DEMO_XLSX, help="demographics Excel 경로")
    parser.add_argument("--test", default=DEFAULT_TEST_CSV, help="testdataset CSV 경로")
    parser.add_argument("--out-under", default=OUT_UNDER)
    parser.add_argument("--out-upper", default=OUT_UPPER)
    parser.add_argument("--threshold", type=float, default=BMI_THRESHOLD)
    args = parser.parse_args()

    for path in (args.demo, args.test):
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

    info = split_testdataset_by_bmi(
        args.demo, args.test, args.out_under, args.out_upper, args.threshold
    )
    print(f"testdataset rows : {info['test_total']}")
    print(f"matched with BMI : {info['matched']}")
    print(f"BMI < {args.threshold}  -> {info['out_under']} ({info['under_n']} rows)")
    print(f"BMI >= {args.threshold} -> {info['out_upper']} ({info['upper_n']} rows)")


if __name__ == "__main__":
    main()
