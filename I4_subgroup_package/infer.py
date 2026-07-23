"""
inference / evaluation script for trained ML models in example/ml_projects.

- 입력 CSV 를 example/ml_projects 안의 가중치 파일로 평가
- 입력 파일명에 따라 자동으로 revision / revision_w_I4 매칭
- 결과는 원본과 동일한 폴더 구조(<project>/<algorithm>/<timestamp>/)로 저장
- AUROC, 95% CI (부트스트랩), 정확도/정밀도/재현율/F1, confusion matrix, ROC curve 모두 저장
- train.py 와 동일한 param_config.yaml 형식을 --config 로 받아서 실행 가능

Usage:
    # YAML config 사용 (train.py 와 동일 형식)
    python infer.py --config param_config.yaml
    python infer.py --config param_config.yaml --csv testdataset_woI4.csv
    python infer.py --config param_config_woI4.yaml

    # 기존 방식: CSV 직접 지정
    python infer.py --csv testdataset_woI4.csv
    python infer.py --csv traindataset_wI4.csv --output_dir results
    python infer.py --csv testdataset_wI4.csv --project revision_w_I4
    python infer.py --csv testdataset_woI4.csv --project revision
    python infer.py --csv testdataset_value.csv --project value
    python infer.py --csv testdataset.csv --project value

YAML config 에서 읽는 키:
    csv_name     -> --csv        (입력 CSV, CLI 인자로 덮어쓰기 가능)
    output_dir   -> --weights_dir (학습된 가중치가 있는 루트 디렉토리)
    project_name -> --project     (project 폴더 이름)
    알고리즘 블록 (randomforest / xgboost / svm / decisiontree) -> --algorithms
        (YAML 에 명시된 알고리즘 블록만 평가 대상으로 자동 추론)
    CLI 인자가 지정되면 config 보다 항상 우선합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Windows 콘솔(cp1252) 에서도 유니코드 출력이 가능하도록 UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from ml_analysis_utils import (
    DEFAULT_N_BOOTSTRAPS,
    bootstrap_metrics_ci,
    build_performance_table,
    compute_shap,
    compute_permutation_importance,
    save_calibration_analysis,
    save_performance_table,
    save_test_predictions,
)
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

TARGET_COL = "abnormality"
DEFAULT_WEIGHTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "eval_results"
RANDOM_STATE = 42
N_BOOTSTRAPS = DEFAULT_N_BOOTSTRAPS

PERFORMANCE_NOTE = (
    "Calibration was performed exploratively; with n=50, stable calibration "
    "interpretation is limited. This study evaluates a screening classifier "
    "(not risk prediction). Bootstrap CI is reported to demonstrate robustness."
)

# train.py 의 param_config.yaml 과 동일한 알고리즘 키
ALGO_KEYS = (
    "randomforest", "xgboost", "svm", "decisiontree",
    "logistic_regression", "neural_network",
)


# ────────────────────────────────────────────────────────────────────────────
# Config 로드
# ────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """YAML config 파일을 dict 로 로드. 매핑 형식이 아니면 예외."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# 모델 / 가중치 탐색
# ────────────────────────────────────────────────────────────────────────────
def find_models(weights_dir: str, run_prefix: str | None = None) -> dict:
    """example/ml_projects 안의 모든 학습 결과를 project → model_type 으로 인덱싱.

    두 가지 구조를 모두 지원:
      1) <weights_dir>/<project>/<algo>/<timestamp>/...   (기존 ml_projects 형태)
      2) <weights_dir>/<algo>/<timestamp>/...             (project_name 생략 형태)
    """
    models: dict = {}
    if not os.path.isdir(weights_dir):
        return models

    # (project, algo) 별로 후보 run_dir 들을 모은 뒤, 가장 최근 timestamp 만 채택
    candidates: dict[tuple[str, str], str] = {}

    for top in os.listdir(weights_dir):
        top_path = os.path.join(weights_dir, top)
        if not os.path.isdir(top_path):
            continue

        for child in os.listdir(top_path):
            child_path = os.path.join(top_path, child)
            if not os.path.isdir(child_path):
                continue
            if re.match(r"^\d{8}_\d{6}$", child):
                # 구조(2): top=algo, child=timestamp
                _maybe_update(candidates, "default", top, child_path, run_prefix)
            else:
                # 구조(1): top=project, child=algo — 그 아래 timestamp 가 있는지 확인
                for sub in os.listdir(child_path):
                    if re.match(r"^\d{8}_\d{6}$", sub):
                        _maybe_update(candidates, top, child, os.path.join(child_path, sub), run_prefix)

    for (project_name, model_type), run_dir in candidates.items():
        info = _extract_model_info(model_type, run_dir)
        if info is not None:
            models.setdefault(project_name, {})[model_type] = info
    return models


def _maybe_update(candidates: dict, project: str, algo: str, run_dir: str,
                  run_prefix: str | None = None):
    """candidates[(project, algo)] = run_dir — 더 최신 timestamp 면 교체.

    run_prefix 가 있으면 해당 접두사로 시작하는 run 만 후보에 포함.
    """
    key = (project, algo)
    ts = os.path.basename(run_dir)
    if run_prefix and not ts.startswith(run_prefix):
        return
    if key not in candidates or os.path.basename(candidates[key]) < ts:
        candidates[key] = run_dir


def _extract_model_info(model_type: str, run_dir: str) -> dict | None:
    """단일 run 디렉토리에서 weight / params / log 경로 추출."""
    log_path = os.path.join(run_dir, "analysis_log.txt")
    model_subdir = os.path.join(run_dir, model_type)
    params_path = os.path.join(model_subdir, "model_params.txt")

    if model_type in ("decision_tree", "decisiontree",
                      "logistic_regression", "neural_network",
                      "random_forest", "randomforest",
                      "svm", "xgboost"):
        pkl_path = os.path.join(model_subdir, "model.pkl")
        if not (os.path.exists(pkl_path) and os.path.exists(params_path)):
            return None
        return {
            "weight_path": pkl_path,
            "params_path": params_path,
            "log_path": log_path if os.path.exists(log_path) else None,
        }
    if model_type == "tabnet":
        if not os.path.isdir(model_subdir):
            return None
        zip_path = None
        for f in os.listdir(model_subdir):
            if f.startswith("tabnet_model") and f.endswith(".zip"):
                zip_path = os.path.join(model_subdir, f)
                break
        if zip_path is None or not os.path.exists(params_path):
            return None
        return {
            "weight_path": zip_path,
            "params_path": params_path,
            "log_path": log_path if os.path.exists(log_path) else None,
        }
    return None


# ────────────────────────────────────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────────────────────────────────────
def parse_model_params(params_path: str) -> dict:
    """model_params.txt 를 dict 로 파싱."""
    params: dict = {}
    if not os.path.exists(params_path):
        return params
    with open(params_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*(.+?)\s*$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            try:
                if val.lower() in ("true", "false"):
                    params[key] = val.lower() == "true"
                elif re.fullmatch(r"-?\d+", val):
                    params[key] = int(val)
                elif re.fullmatch(r"-?\d+\.\d+", val):
                    params[key] = float(val)
                else:
                    params[key] = val
            except Exception:
                params[key] = val
    return params


def load_model(model_type: str, weight_path: str, params: dict):
    """가중치 파일에서 모델 로드."""
    if model_type in ("decision_tree", "decisiontree",
                      "logistic_regression", "neural_network",
                      "random_forest", "randomforest",
                      "svm", "xgboost"):
        with open(weight_path, "rb") as f:
            return pickle.load(f)
    if model_type == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        clf = TabNetClassifier(
            n_d=int(params.get("n_d", 8)),
            n_a=int(params.get("n_a", 8)),
            n_steps=int(params.get("n_steps", 3)),
            gamma=float(params.get("gamma", 1.3)),
            seed=RANDOM_STATE,
            verbose=0,
        )
        clf.load_model(weight_path)
        return clf
    raise ValueError(f"Unsupported model_type: {model_type}")


def get_feature_names(model, model_type: str, log_path: str | None) -> list[str] | None:
    """모델이 학습될 때 사용한 feature 이름 목록 복원."""
    # 1) sklearn feature_names_in_
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        if names:
            return names
    # 2) analysis_log.txt 의 "Key variables: [...]"
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"Key variables:\s*(\[.*?\])", content)
        if m:
            try:
                return list(eval(m.group(1), {"__builtins__": {}}, {}))
            except Exception:
                pass
    return None


# ────────────────────────────────────────────────────────────────────────────
# 예측 / 메트릭
# ────────────────────────────────────────────────────────────────────────────
def predict_proba(model, X: pd.DataFrame, model_type: str) -> np.ndarray:
    if model_type == "tabnet":
        return model.predict_proba(X.values)[:, 1]
    return model.predict_proba(X)[:, 1]


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    boot = bootstrap_metrics_ci(
        y_true, y_pred_proba, n_bootstraps=N_BOOTSTRAPS, threshold=threshold,
    )
    performance_table = build_performance_table(boot)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "auc": boot["auc"]["value"],
        "ci_low": boot["auc"]["ci_low"],
        "ci_high": boot["auc"]["ci_high"],
        "brier_score": boot["brier_score"]["value"],
        "accuracy": boot["accuracy"]["value"],
        "accuracy_ci_low": boot["accuracy"]["ci_low"],
        "accuracy_ci_high": boot["accuracy"]["ci_high"],
        "precision": boot["precision"]["value"],
        "precision_ci_low": boot["precision"]["ci_low"],
        "precision_ci_high": boot["precision"]["ci_high"],
        "recall": boot["sensitivity"]["value"],
        "recall_ci_low": boot["sensitivity"]["ci_low"],
        "recall_ci_high": boot["sensitivity"]["ci_high"],
        "sensitivity": boot["sensitivity"]["value"],
        "sensitivity_ci_low": boot["sensitivity"]["ci_low"],
        "sensitivity_ci_high": boot["sensitivity"]["ci_high"],
        "specificity": boot["specificity"]["value"],
        "specificity_ci_low": boot["specificity"]["ci_low"],
        "specificity_ci_high": boot["specificity"]["ci_high"],
        "f1": boot["f1"]["value"],
        "f1_ci_low": boot["f1"]["ci_low"],
        "f1_ci_high": boot["f1"]["ci_high"],
        "brier_ci_low": boot["brier_score"]["ci_low"],
        "brier_ci_high": boot["brier_score"]["ci_high"],
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "performance_table": performance_table.to_dict(orient="records"),
        "n_samples": len(y_true),
        "n_bootstraps": N_BOOTSTRAPS,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
    }


# ────────────────────────────────────────────────────────────────────────────
# 시각화 / 파일 저장
# ────────────────────────────────────────────────────────────────────────────
def save_roc_curve(y_true, y_pred_proba, auc, ci_low, ci_high, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="red", linewidth=2,
             label=f"Model (AUC: {auc:.3f}, 95% CI: {ci_low:.3f}-{ci_high:.3f})")
    plt.fill_between(fpr, np.maximum(tpr - 0.05, 0), np.minimum(tpr + 0.05, 1),
                     alpha=0.15, color="red", label="±0.05 band")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Test Set Evaluation", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_data(y_true, y_pred_proba, auc, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    pd.DataFrame({
        "fold": 1,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "model": "Test evaluation",
    }).to_csv(output_path, index=False)


def save_predictions(df, y_true, y_pred_proba, feature_names, output_path: str):
    out = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_pred_proba})
    out["y_pred"] = (y_pred_proba >= 0.5).astype(int)
    for col in feature_names:
        if col in df.columns:
            out.insert(0, col, df[col].values)
    out.to_csv(output_path, index=False)


def write_log(path: str, project: str, model_type: str, csv_path: str,
              feature_names: list, metrics: dict, params: dict, weight_path: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"===== ML Analysis Project: {project} (Evaluation) =====\n")
        f.write(f"Algorithm: {model_type}\n")
        f.write(f"Created at: {ts}\n\n")
        f.write(f"[{ts}] ==================================================\n")
        f.write(f"[{ts}] Evaluation started\n")
        f.write(f"[{ts}] Model type: {model_type}\n")
        f.write(f"[{ts}] Weight file: {weight_path}\n")
        f.write(f"[{ts}] Input CSV: {csv_path}\n")
        f.write(f"[{ts}] Features ({len(feature_names)}): {feature_names}\n")
        f.write(f"[{ts}] Target: {TARGET_COL}\n")
        f.write(f"[{ts}] Sample size: {len(metrics['y_true'])}\n")
        f.write(f"[{ts}] Positive samples: {int(sum(metrics['y_true']))}\n")
        f.write(f"[{ts}] Model parameters: {params}\n")
        f.write(f"[{ts}] AUC: {metrics['auc']:.4f} "
                f"(95% Bootstrap CI: {metrics['ci_low']:.4f}-{metrics['ci_high']:.4f})\n")
        f.write(f"[{ts}] Accuracy: {metrics['accuracy']:.4f} "
                f"(95% Bootstrap CI: {metrics['accuracy_ci_low']:.4f}-{metrics['accuracy_ci_high']:.4f})\n")
        f.write(f"[{ts}] Sensitivity: {metrics['sensitivity']:.4f} "
                f"(95% Bootstrap CI: {metrics['sensitivity_ci_low']:.4f}-{metrics['sensitivity_ci_high']:.4f})\n")
        f.write(f"[{ts}] Specificity: {metrics['specificity']:.4f} "
                f"(95% Bootstrap CI: {metrics['specificity_ci_low']:.4f}-{metrics['specificity_ci_high']:.4f})\n")
        f.write(f"[{ts}] Brier score: {metrics['brier_score']:.4f} "
                f"(95% Bootstrap CI: {metrics['brier_ci_low']:.4f}-{metrics['brier_ci_high']:.4f})\n")
        f.write(f"[{ts}] Precision: {metrics['precision']:.4f}\n")
        f.write(f"[{ts}] F1: {metrics['f1']:.4f}\n")
        f.write(f"[{ts}] Confusion matrix [[TN, FP], [FN, TP]] = {metrics['confusion_matrix']}\n")
        f.write(f"[{ts}] Classification report:\n{metrics['classification_report']}")
        f.write(f"[{ts}] Evaluation completed\n")


def write_detailed_analysis(path: str, project: str, model_type: str,
                            feature_names: list, metrics: dict, params: dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"===== Extended ML Model Evaluation =====\n")
        f.write(f"Generated at: {ts}\n\n")
        f.write(f"Project: {project}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"AUC: {metrics['auc']:.4f} (95% Bootstrap CI: {metrics['ci_low']:.4f}-{metrics['ci_high']:.4f})\n")
        f.write(f"Brier score: {metrics['brier_score']:.4f} "
                f"(95% Bootstrap CI: {metrics['brier_ci_low']:.4f}-{metrics['brier_ci_high']:.4f})\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"Feature List: {feature_names}\n")
        f.write(f"Model Parameters: {params}\n\n")
        f.write("--- Performance Metrics (95% Bootstrap CI) ---\n")
        f.write(f"{'Metric':<16} {'Value':>8}  {'95% Bootstrap CI':>18}\n")
        f.write("-" * 46 + "\n")
        for row in metrics["performance_table"]:
            f.write(f"{row['Metric']:<16} {row['Value']:>8.2f}  {row['Bootstrap_CI_95']:>18}\n")
        f.write(f"\n{PERFORMANCE_NOTE}\n\n")
        f.write("--- Confusion Matrix ---\n")
        f.write(f"  TN={metrics['tn']}  FP={metrics['fp']}\n")
        f.write(f"  FN={metrics['fn']}  TP={metrics['tp']}\n\n")
        f.write("--- Classification Report ---\n")
        f.write(metrics['classification_report'])


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────
def detect_project(csv_name: str) -> str:
    name = csv_name.lower()
    if "wi4" in name:
        return "revision_w_I4"
    return "revision"


def copy_weight_file(src: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def evaluate_one(eval_label: str, model_type: str, model_info: dict, df: pd.DataFrame,
                 csv_path: str, output_root: str,
                 weights_project: str = "train") -> tuple[str | None, dict | None]:
    """단일 (eval_label, model_type) 평가 후 결과 디렉토리 경로 반환."""
    try:
        params = parse_model_params(model_info["params_path"])
        print(f"  ▸ Loading {model_type} from {model_info['weight_path']}")
        model = load_model(model_type, model_info["weight_path"], params)

        feature_names = get_feature_names(model, model_type, model_info["log_path"])
        if feature_names is None:
            print(f"  ⚠ Could not infer feature names; using all non-target columns")
            feature_names = [c for c in df.columns if c != TARGET_COL]

        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            print(f"  ✗ SKIP: input CSV is missing required features: {missing}")
            return None, None
        if len(feature_names) == 0:
            print(f"  ✗ SKIP: no features available")
            return None, None

        X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_true = df[TARGET_COL].astype(int).values

        y_pred_proba = predict_proba(model, X, model_type)
        metrics = compute_metrics(y_true, y_pred_proba)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(output_root, eval_label, model_type, ts)
        model_subdir = os.path.join(out_dir, model_type)
        os.makedirs(model_subdir, exist_ok=True)

        def _log(msg):
            print(f"  {msg}")

        # results.csv  (원본 스키마와 동일)
        pd.DataFrame([{
            "Model": "Test only",
            "Mean_AUC": metrics["auc"],
            "CI_Low": metrics["ci_low"],
            "CI_High": metrics["ci_high"],
            "AUC_formatted": f"{metrics['auc']:.3f} ({metrics['ci_low']:.3f}-{metrics['ci_high']:.3f})",
            "Brier_Score": metrics["brier_score"],
            "n_features": len(feature_names),
        }]).to_csv(os.path.join(out_dir, "results.csv"), index=False)

        performance_df = pd.DataFrame(metrics["performance_table"])
        save_performance_table(
            performance_df, out_dir,
            model_type=model_type,
            n_samples=metrics["n_samples"],
            n_bootstraps=metrics["n_bootstraps"],
            note=PERFORMANCE_NOTE,
            log_func=_log,
        )

        # ROC curve + 데이터
        save_roc_curve(y_true, y_pred_proba, metrics["auc"],
                       metrics["ci_low"], metrics["ci_high"],
                       os.path.join(out_dir, "roc_curve.png"))
        save_roc_data(y_true, y_pred_proba, metrics["auc"],
                      os.path.join(out_dir, "roc_data.csv"))

        # Calibration plot + Brier score (모든 알고리즘 공통)
        save_calibration_analysis(y_true, y_pred_proba, out_dir, log_func=_log)

        # 샘플별 예측
        save_predictions(df, y_true, y_pred_proba, feature_names,
                         os.path.join(out_dir, "predictions.csv"))

        # ml_analysis_app.py 호환 test_predictions.csv
        y_pred_class = (y_pred_proba >= 0.5).astype(int)
        test_pred_df = pd.DataFrame({
            "original_row_number": X.index + 2,
            "true_label": y_true,
            "predicted_label": y_pred_class,
            "predicted_probability": y_pred_proba,
        })
        test_pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
        print(f"  ✓ test_predictions saved ({len(test_pred_df)} samples)")

        # ────── Extended analysis (SHAP / Permutation) ──────
        compute_shap(model, model_type, X, feature_names, out_dir, log_func=_log)
        compute_permutation_importance(
            model, model_type, X, pd.Series(y_true, name=TARGET_COL),
            out_dir, log_func=_log,
        )

        # 메트릭 JSON
        serializable = {k: v for k, v in metrics.items()
                        if k not in ("y_true", "y_pred_proba")}
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        # 로그
        write_log(os.path.join(out_dir, "analysis_log.txt"),
                  weights_project, model_type, csv_path, feature_names, metrics, params,
                  model_info["weight_path"])
        write_detailed_analysis(os.path.join(out_dir, "detailed_analysis.txt"),
                                weights_project, model_type, feature_names, metrics, params)

        # 모델 weight / params (원본 구조 유지)
        copy_weight_file(model_info["weight_path"], model_subdir)
        with open(os.path.join(model_subdir, "model_params.txt"), "w", encoding="utf-8") as f:
            f.write(f"Model Type: {model_type}\nParameters:\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")

        print(f"  ✓ AUC={metrics['auc']:.4f} "
              f"(95% Bootstrap CI {metrics['ci_low']:.4f}-{metrics['ci_high']:.4f}) "
              f"Sens={metrics['sensitivity']:.4f} "
              f"Spec={metrics['specificity']:.4f} "
              f"Brier={metrics['brier_score']:.4f}")
        return out_dir, metrics

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="param_config YAML 파일 경로 (train.py 와 동일 형식). "
                             "지정 시 csv_name / output_dir (weights_dir) / project_name / "
                             "알고리즘 블록을 자동으로 읽음. CLI 인자는 config 보다 항상 우선")
    parser.add_argument("--csv", default=None,
                        help="입력 CSV 경로 (config에 csv_name 이 있으면 생략 가능)")
    parser.add_argument("--weights_dir", default=None,
                        help="가중치 루트 디렉토리 (config의 output_dir 사용 가능, 기본: results)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="결과 루트 디렉토리 (default: eval_results)")
    parser.add_argument("--project", default=None,
                        help="가중치 project 이름 (config 의 project_name, 기본: train)")
    parser.add_argument("--eval_label", default=None,
                        help="결과 저장 서브폴더 (예: BMI25_under, all). "
                             "미지정 시 --project 와 동일")
    parser.add_argument("--algorithms", nargs="*", default=None,
                        help="특정 알고리즘만 평가 (예: --algorithms xgboost random_forest). "
                             "config 사용 시 YAML 의 알고리즘 블록에서 자동 추론")
    parser.add_argument("--weights_run_prefix", default=None,
                        help="가중치 run timestamp 접두사 (예: 20260622_053). "
                             "지정 시 해당 접두사로 시작하는 run 중 최신 것을 사용")
    args = parser.parse_args()

    # ────── YAML config 로드 + CLI 기본값 채우기 ──────
    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(f"Config loaded: {args.config}")
        # CLI 인자가 명시되지 않은 항목은 config 값으로 채움
        if args.csv is None and "csv_name" in cfg:
            args.csv = cfg["csv_name"]
        if args.weights_dir is None and "output_dir" in cfg:
            args.weights_dir = cfg["output_dir"]
        if args.project is None and "project_name" in cfg:
            args.project = cfg["project_name"]
        if args.algorithms is None:
            algos = [k for k in ALGO_KEYS if k in cfg]
            if algos:
                args.algorithms = algos

    # ────── 필수 인자 / 경로 검증 ──────
    if args.csv is None:
        print("ERROR: --csv 또는 --config 의 csv_name 중 하나는 반드시 필요합니다.")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"ERROR: input CSV not found: {args.csv}")
        sys.exit(1)

    if args.weights_dir is None:
        args.weights_dir = DEFAULT_WEIGHTS_DIR

    df = pd.read_csv(args.csv)
    if TARGET_COL not in df.columns:
        print(f"ERROR: target column '{TARGET_COL}' missing in {args.csv}")
        sys.exit(1)

    weights_project = args.project
    if weights_project is None and cfg.get("project_name"):
        weights_project = cfg["project_name"]
    if weights_project is None:
        weights_project = detect_project(os.path.basename(args.csv))

    eval_label = args.eval_label or weights_project
    print(f"Weights project: {weights_project}")
    print(f"Eval label     : {eval_label}")
    print(f"Input CSV: {args.csv}  shape={df.shape}")
    print(f"Weights dir: {args.weights_dir}")
    print(f"Output dir : {args.output_dir}")
    if args.algorithms:
        print(f"Algorithms  : {args.algorithms}")
    print()

    if args.weights_run_prefix:
        print(f"Weights prefix: {args.weights_run_prefix}")
    all_models = find_models(args.weights_dir, run_prefix=args.weights_run_prefix)
    if weights_project not in all_models:
        # "default" 키(구조2, project_name 미지정)로 저장된 가중치를 폴백으로 시도
        if "default" in all_models:
            weights_project = "default"
        else:
            print(f"ERROR: project '{weights_project}' not found in {args.weights_dir}")
            print(f"Available projects: {list(all_models.keys())}")
            sys.exit(1)

    project_models = all_models[weights_project]
    if args.algorithms:
        project_models = {k: v for k, v in project_models.items() if k in args.algorithms}
        if not project_models:
            print(f"ERROR: none of {args.algorithms} available in project '{weights_project}'")
            print(f"Available in '{weights_project}': {list(all_models[weights_project].keys())}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dirs: list[str] = []
    summary_rows: list[dict] = []
    for model_type, model_info in project_models.items():
        print(f"[{eval_label}/{model_type}]")
        out, metrics = evaluate_one(
            eval_label, model_type, model_info, df, args.csv, args.output_dir,
            weights_project=weights_project,
        )
        if out and metrics:
            out_dirs.append(out)
            for row in metrics["performance_table"]:
                summary_rows.append({"Model": model_type, **row})
        print()

    if summary_rows:
        summary_dir = os.path.join(args.output_dir, eval_label)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"performance_summary_{run_ts}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Combined performance summary: {summary_path}")

    print("=" * 60)
    print(f"✓ {len(out_dirs)} evaluation(s) completed")
    for d in out_dirs:
        print(f"  - {d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
