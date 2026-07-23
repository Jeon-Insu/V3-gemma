"""eval_results 서브그룹별 AUC 요약 CSV 생성."""
from __future__ import annotations
import glob
import os
from pathlib import Path
import pandas as pd

rows = []
for f in sorted(glob.glob("eval_results/*/performance_summary_*.csv")):
    grp = Path(f).parent.name
    df = pd.read_csv(f)
    for _, r in df[df["Metric"] == "AUC"].iterrows():
        rows.append({
            "Group": grp,
            "Model": r["Model"],
            "AUC": r["Value"],
            "Bootstrap_CI_95": r["Bootstrap_CI_95"],
        })
out = pd.DataFrame(rows)
os.makedirs("summary", exist_ok=True)
out.to_csv("summary/subgroup_auc_summary.csv", index=False)
print("saved summary/subgroup_auc_summary.csv", len(out), "rows")
