"""
ML 모델 학습 스크립트 (CLI 버전)

- param_config.yaml 을 입력으로 받아서 학습을 실행
- example/ml_analysis_app.py 의 학습 파이프라인
  (CV 평가, ROC curve, 모델 저장) 을 그대로 차용
- 결과는 <output_dir>/<algorithm>/<timestamp>/ 아래에 저장

Usage:
    python train.py --config param_config.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_analysis_utils import (
    compute_shap,
    compute_permutation_importance,
    save_detailed_analysis,
    save_test_predictions,
    save_train_test_split,
)

# Windows 콘솔(cp1252) 에서도 유니코드 출력이 가능하도록 UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DEFAULT_RANDOM_STATE = RANDOM_STATE

# config 의 알고리즘 키 ↔ 사람이 읽는 이름
ALGO_KEYS = (
    "randomforest", "xgboost", "svm", "decisiontree",
    "logistic_regression", "neural_network",
)


# ────────────────────────────────────────────────────────────────────────────
# Config 로드 / 검증
# ────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """YAML config 를 로드하고 필수 키를 검증."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required = ["csv_name", "input_columns", "label_column", "output_dir"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Config is missing required key: '{k}'")

    if not isinstance(cfg["input_columns"], list) or not cfg["input_columns"]:
        raise ValueError("'input_columns' must be a non-empty list")

    found = [a for a in ALGO_KEYS if a in cfg and cfg[a] is not None]
    if not found:
        raise ValueError(
            f"Config must contain at least one algorithm block: {list(ALGO_KEYS)}"
        )
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# 모델 팩토리 (ml_analysis_app.py 의 get_model 참고)
# ────────────────────────────────────────────────────────────────────────────
def get_model(model_type: str, params: dict | None = None, random_state: int = DEFAULT_RANDOM_STATE):
    base_params = {
        "randomforest": {"n_jobs": -1, "verbose": 0, "random_state": random_state},
        "xgboost": {"eval_metric": "logloss", "n_jobs": -1, "verbosity": 0, "random_state": random_state},
        "svm": {"probability": True, "random_state": random_state, "kernel": "rbf"},
        "decisiontree": {"random_state": random_state},
        "logistic_regression": {"max_iter": 1000, "random_state": random_state},
        "neural_network": {"max_iter": 500, "random_state": random_state, "early_stopping": True},
    }
    merged = {**base_params.get(model_type, {}), **(params or {})}
    if model_type == "neural_network" and "hidden_layer_sizes" in merged:
        merged["hidden_layer_sizes"] = tuple(merged["hidden_layer_sizes"])
    registry = {
        "randomforest": RandomForestClassifier,
        "xgboost": xgb.XGBClassifier,
        "svm": SVC,
        "decisiontree": DecisionTreeClassifier,
        "logistic_regression": LogisticRegression,
        "neural_network": MLPClassifier,
    }
    if model_type not in registry:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return registry[model_type](**merged)


# ────────────────────────────────────────────────────────────────────────────
# 데이터 전처리 (ml_analysis_app.py 의 preprocess_data 참고)
# ────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple[pd.DataFrame, dict]:
    """선택 컬럼에서 결측/비숫자 행을 제거하고 숫자형으로 강제 변환."""
    selected = feature_cols + [target_col]
    df_sub = df[selected].copy()
    original = df_sub.shape[0]

    na_by_column = df_sub.isna().sum()
    na_by_column = na_by_column[na_by_column > 0].to_dict()

    df_clean = df_sub.dropna()
    after_na = df_clean.shape[0]

    # 비숫자 행 인덱스 수집
    non_numeric_idx: set = set()
    for col in selected:
        coerced = pd.to_numeric(df_clean[col], errors="coerce")
        mask = coerced.isna()
        if mask.any():
            non_numeric_idx.update(df_clean[mask].index.tolist())
    if non_numeric_idx:
        df_clean = df_clean.drop(list(non_numeric_idx))

    # 숫자형 변환 + 안전망
    for col in selected:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean = df_clean.dropna()

    info = {
        "original_rows": original,
        "after_na_removal": after_na,
        "after_non_numeric_removal": df_clean.shape[0],
        "na_removed": original - after_na,
        "non_numeric_removed": after_na - df_clean.shape[0],
        "total_removed": original - df_clean.shape[0],
        "processed_columns": selected,
        "na_by_column": na_by_column,
    }
    return df_clean, info


# ────────────────────────────────────────────────────────────────────────────
# CV AUC 평가 (ml_analysis_app.py 의 cross_val_auc 참고)
# ────────────────────────────────────────────────────────────────────────────
def cross_val_auc(X, y, model_type, params, n_splits, random_state):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = get_model(model_type, params, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_pred))
    mean_auc = float(np.mean(aucs))
    ci_low, ci_high = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    return mean_auc, (ci_low, ci_high), aucs


# ────────────────────────────────────────────────────────────────────────────
# ROC 데이터 (ml_analysis_app.py 의 ROC 생성 로직 참고)
# ────────────────────────────────────────────────────────────────────────────
def collect_roc_data(X, y, model_type, params, n_splits, random_state, model_name):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = get_model(model_type, params, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc_score(y_test, y_pred))
    return {
        "base_fpr": base_fpr,
        "best": {"tprs": tprs, "aucs": aucs},
        "best_model_name": model_name,
    }


def create_roc_plot_matplotlib(roc_data: dict) -> bytes:
    """ml_analysis_app.py 의 create_roc_plot_matplotlib 그대로 차용."""
    tprs = np.array(roc_data["best"]["tprs"])
    aucs = np.array(roc_data["best"]["aucs"])
    base_fpr = roc_data["base_fpr"]
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = aucs.mean()
    ci_low = mean_auc - 1.96 * aucs.std() / np.sqrt(len(aucs))
    ci_high = mean_auc + 1.96 * aucs.std() / np.sqrt(len(aucs))
    tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr / np.sqrt(len(tprs)), 1)
    tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr / np.sqrt(len(tprs)), 0)

    plt.figure(figsize=(10, 8))
    plt.fill_between(base_fpr, tpr_lower, tpr_upper, alpha=0.2, color="red", label="95% Confidence Interval")
    plt.plot(base_fpr, mean_tpr, color="red", linewidth=2,
             label=f"Model (AUC: {mean_auc:.3f}, 95% CI: {ci_low:.3f}-{ci_high:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Model with 95% Confidence Intervals", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def create_roc_data_csv(roc_data: dict) -> str:
    """ml_analysis_app.py 의 create_roc_data_csv 그대로 차용."""
    tprs = roc_data["best"]["tprs"]
    aucs = roc_data["best"]["aucs"]
    base_fpr = roc_data["base_fpr"]
    model_name = roc_data["best_model_name"]
    rows = []
    for i, (tpr, auc) in enumerate(zip(tprs, aucs)):
        for fpr_val, tpr_val in zip(base_fpr, tpr):
            rows.append({"fold": i + 1, "auc": auc, "fpr": fpr_val, "tpr": tpr_val, "model": model_name})
    return pd.DataFrame(rows).to_csv(index=False)


# ────────────────────────────────────────────────────────────────────────────
# 단일 알고리즘 학습 + 결과 저장
# ────────────────────────────────────────────────────────────────────────────
def train_algorithm(algo_key, params, X, y, output_dir, project_name, n_splits, random_state,
                    csv_name, feature_cols, target_col, preproc_info):
    """한 알고리즘에 대해 CV 평가 + 최종 모델 학습 + 결과 저장."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir if not project_name else os.path.join(output_dir, project_name)
    run_dir = os.path.join(base, algo_key, ts)
    algo_subdir = os.path.join(run_dir, algo_key)
    os.makedirs(algo_subdir, exist_ok=True)

    log_path = os.path.join(run_dir, "analysis_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"===== ML Training Project =====\n")
        f.write(f"Algorithm: {algo_key}\n")
        f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ==================================================\n")

    def log(msg):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {msg}\n")

    log("Analysis started")
    log(f"Model type: {algo_key}")
    log(f"Input CSV: {csv_name}")
    log(f"Key variables: {feature_cols}")
    log(f"Explore variables: []")
    log(f"Target variable: {target_col}")
    log(f"CV folds: {n_splits}")
    log(f"Random state: {random_state}")
    log(f"Starting data preprocessing...")
    log(f"Preprocessing complete: {preproc_info}")
    log(f"Evaluating 1 model combinations")
    log("Evaluating model 1/1: Key only")

    # CV 평가
    model_name = "Key only"
    mean_auc, (ci_low, ci_high), aucs = cross_val_auc(
        X, y, algo_key, params, n_splits, random_state
    )
    log(f"  Result: AUC={mean_auc:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})")
    log(f"Best model: {model_name}")
    log(f"Best AUC: {mean_auc:.4f}")

    auc_formatted = f"{mean_auc:.3f} ({ci_low:.3f}-{ci_high:.3f})"

    # results.csv (원본 스키마와 동일)
    results_df = pd.DataFrame([{
        "Model": model_name,
        "Mean_AUC": mean_auc,
        "CI_Low": ci_low,
        "CI_High": ci_high,
        "AUC_formatted": auc_formatted,
        "n_features": X.shape[1],
    }])
    results_df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    # ROC curve + 데이터
    roc_data = collect_roc_data(X, y, algo_key, params, n_splits, random_state, model_name)
    with open(os.path.join(run_dir, "roc_data.csv"), "w", encoding="utf-8") as f:
        f.write(create_roc_data_csv(roc_data))
    with open(os.path.join(run_dir, "roc_curve.png"), "wb") as f:
        f.write(create_roc_plot_matplotlib(roc_data))

    # 최종 모델 학습 (전체 데이터) + 저장
    log("Training final model on full data...")
    final_model = get_model(algo_key, params, random_state)
    final_model.fit(X, y)

    with open(os.path.join(algo_subdir, "model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    with open(os.path.join(algo_subdir, "model_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model Type: {algo_key}\nParameters:\n")
        for k, v in (params or {}).items():
            f.write(f"  {k}: {v}\n")

    log(f"Model weights saved to {algo_subdir}")

    # ────── Extended analysis (example/ml_analysis_app.py 차용) ──────
    log("Starting extended analysis...")

    # 1. CV 마지막 fold 기준 train/test 분리
    log("Splitting train/test (last fold)...")
    X_train, X_test, y_train, y_test = save_train_test_split(
        X, y, n_splits, run_dir, log_func=log
    )
    log("Train/test sets saved")

    # 2. test_predictions (마지막 fold 에서 재학습한 모델의 예측)
    log("Generating test predictions...")
    pred_model = get_model(algo_key, params, random_state)
    pred_result = save_test_predictions(
        pred_model, algo_key, X_train, X_test, y_train, y_test, run_dir, log_func=log
    )
    if pred_result is not None:
        _, brier = pred_result
        results_df["Brier_Score"] = brier
        results_df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
        log(f"Brier score (test set): {brier:.4f}")

    # 3. SHAP (전체 데이터로 학습한 final_model 사용)
    compute_shap(final_model, algo_key, X, feature_cols, run_dir, log_func=log)

    # 4. Permutation importance (마지막 fold 의 test set 기준)
    perm_model = get_model(algo_key, params, random_state)
    if algo_key == "tabnet":
        # tabnet 은 numpy 입출력 + fit 옵션이 달라 별도 처리
        perm_model.fit(
            X_train.values, y_train.values,
            max_epochs=100, patience=15, batch_size=256,
            virtual_batch_size=128,
        )
        compute_permutation_importance(
            perm_model, algo_key, X_test, y_test, run_dir, log_func=log
        )
    else:
        perm_model.fit(X_train, y_train)
        compute_permutation_importance(
            perm_model, algo_key, X_test, y_test, run_dir, log_func=log
        )

    # 5. Detailed analysis
    save_detailed_analysis(
        run_dir, project_name or "training", algo_key, params,
        feature_cols, X.shape[1], mean_auc, ci_low, ci_high, auc_formatted,
        n_splits, y, preproc_info, results_df,
    )
    log("Detailed analysis info saved")

    log("Saving analysis results...")
    log("All results saved successfully")
    log("Analysis completed")
    return run_dir


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="param_config YAML 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_name = cfg["csv_name"]
    feature_cols = cfg["input_columns"]
    target_col = cfg["label_column"]
    output_dir = cfg["output_dir"]
    project_name = cfg.get("project_name") or None
    n_splits = int(cfg.get("n_splits", 5))
    random_state = int(cfg.get("random_state", RANDOM_STATE))

    print(f"Config loaded: {args.config}")
    print(f"  CSV           : {csv_name}")
    print(f"  Features      : {len(feature_cols)} columns")
    print(f"  Target        : {target_col}")
    print(f"  Output dir    : {output_dir}")
    if project_name:
        print(f"  Project name  : {project_name}")
    else:
        print(f"  Project name  : (none — using flat output_dir/<algo>/<ts>/)")
    print(f"  CV folds      : {n_splits}")
    print(f"  Random state  : {random_state}")
    algos = [k for k in ALGO_KEYS if k in cfg]
    print(f"  Algorithms    : {algos}")
    print()

    # 데이터 로드 + 검증
    if not os.path.exists(csv_name):
        print(f"ERROR: CSV not found: {csv_name}")
        sys.exit(1)
    df = pd.read_csv(csv_name)
    print(f"Loaded {csv_name}: {df.shape}")

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        print(f"ERROR: columns missing in CSV: {missing}")
        sys.exit(1)

    df_clean, preproc_info = preprocess(df, feature_cols, target_col)
    print(f"Preprocessing: {preproc_info['original_rows']} -> "
          f"{preproc_info['after_non_numeric_removal']} rows "
          f"(removed {preproc_info['total_removed']})")

    if df_clean.shape[0] < 10:
        print(f"ERROR: too few rows after preprocessing ({df_clean.shape[0]})")
        if preproc_info.get("na_by_column"):
            print("  Columns with missing values (rows dropped by dropna):")
            for col, cnt in preproc_info["na_by_column"].items():
                print(f"    - {col}: {cnt}/{preproc_info['original_rows']} rows")
            print("  Hint: check CSV header/column names (e.g. duplicate I2 -> empty I2 column).")
        sys.exit(1)

    y = df_clean[target_col].astype(int)
    X = df_clean[feature_cols]

    if len(y.unique()) != 2:
        print(f"ERROR: target must be binary (found {len(y.unique())} unique values: {sorted(y.unique())})")
        sys.exit(1)
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        print(f"ERROR: each class must have at least 2 samples (pos={y.sum()}, neg={len(y)-y.sum()})")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 알고리즘 학습
    run_dirs = []
    for key in algos:
        print(f"\n{'=' * 60}")
        print(f"Training: {key}")
        print(f"{'=' * 60}")
        params = cfg[key] or {}
        try:
            run_dir = train_algorithm(
                key, params, X, y, output_dir, project_name,
                n_splits, random_state, csv_name, feature_cols, target_col, preproc_info,
            )
            run_dirs.append(run_dir)
            print(f"  ✓ done -> {run_dir}")
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"✓ {len(run_dirs)}/{len(algos)} algorithm(s) trained successfully")
    for d in run_dirs:
        print(f"  - {d}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
