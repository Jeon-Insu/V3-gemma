"""
ML 분석 유틸리티 (train.py / infer.py 공용).

example/ml_analysis_app.py 의 SHAP / Permutation / Detailed analysis 로직을
CLI 환경에 맞게 차용한 함수 모음.

- compute_shap(): 모델별 SHAP explainer 로 summary plot + values + importance 저장
- compute_permutation_importance(): sklearn permutation_importance 로 CSV/PNG 저장
- save_train_test_split(): CV 마지막 fold 기준 train/test 분리
- save_test_predictions(): test set 에 대한 예측값 저장
- save_detailed_analysis(): detailed_analysis.txt 생성
- save_calibration_analysis(): Brier score + calibration plot/data 저장
- bootstrap_metrics_ci(): 성능 지표별 95% Bootstrap CI
- save_performance_table(): Metric / Value / Bootstrap CI 표 저장
"""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

DEFAULT_N_BOOTSTRAPS = 2000

METRIC_LABELS = {
    "auc": "AUC",
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "brier_score": "Brier score",
}

DEFAULT_METRIC_ORDER = (
    "auc", "accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score",
)

# Windows 콘솔(cp1252) 에서도 유니코드 출력이 가능하도록 UTF-8 강제
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# SHAP / permutation 결과를 모델 단독 attribute 로 캐시하기 위한 prefix
SHAP_TREE = {"xgboost", "randomforest", "decisiontree"}
SHAP_LINEAR = {"logistic_regression"}
SHAP_KERNEL = {"svm", "neural_network", "tabnet"}


def _calibration_n_bins(n_samples: int, requested: int = 10) -> int:
    """샘플 수가 bin 수보다 작을 때만 조정 (기본 n_bins=10)."""
    return min(requested, max(2, n_samples))


def compute_brier_score(y_true, y_pred_proba) -> float:
    """Brier score (0=perfect, 1=worst)."""
    return float(brier_score_loss(y_true, y_pred_proba))


def save_calibration_plot(y_true, y_pred_proba, output_path: str,
                          n_bins: int = 10,
                          title: str = "Calibration Plot") -> float:
    """Reliability diagram 저장. 반환값: Brier score."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    brier_score = compute_brier_score(y_true, y_pred_proba)
    n_bins = _calibration_n_bins(len(y_true), n_bins)
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return brier_score


def save_calibration_data(y_true, y_pred_proba, output_path: str, n_bins: int = 10):
    """Calibration curve bin 데이터를 CSV 로 저장."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n_bins = _calibration_n_bins(len(y_true), n_bins)
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    pd.DataFrame({
        "predicted_probability": prob_pred,
        "observed_probability": prob_true,
        "brier_score": compute_brier_score(y_true, y_pred_proba),
    }).to_csv(output_path, index=False)


def save_calibration_analysis(y_true, y_pred_proba, save_dir: str,
                              n_bins: int = 10, log_func=None) -> float:
    """Brier score 계산 + calibration plot/data 저장."""
    if log_func is None:
        log_func = print
    brier = save_calibration_plot(
        y_true, y_pred_proba,
        os.path.join(save_dir, "calibration_plot.png"),
        n_bins=n_bins,
    )
    save_calibration_data(
        y_true, y_pred_proba,
        os.path.join(save_dir, "calibration_data.csv"),
        n_bins=n_bins,
    )
    log_func(f"  ✓ Calibration saved (Brier score={brier:.4f})")
    return brier


# ────────────────────────────────────────────────────────────────────────────
# Bootstrap CI / Performance table
# ────────────────────────────────────────────────────────────────────────────
def format_bootstrap_ci(ci_low: float, ci_high: float) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "NA"
    return f"{ci_low:.2f}–{ci_high:.2f}"


def _compute_scalar_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict | None:
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    cm = confusion_matrix(y_true, y_pred)
    if cm.size != 4:
        return None
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
    }


def bootstrap_metrics_ci(
    y_true,
    y_pred_proba,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict:
    """지표별 point estimate + 95% Bootstrap CI 반환."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    point = _compute_scalar_metrics(y_true, y_pred_proba, threshold)
    if point is None:
        raise ValueError("Cannot compute metrics: need binary labels and valid predictions")

    boot_samples = {k: [] for k in point}
    rng = np.random.RandomState(seed)
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        sampled = _compute_scalar_metrics(y_true[idx], y_pred_proba[idx], threshold)
        if sampled is None:
            continue
        for key, val in sampled.items():
            boot_samples[key].append(val)

    results = {}
    for key, value in point.items():
        samples = np.array(boot_samples[key])
        if len(samples) == 0:
            ci_low, ci_high = float("nan"), float("nan")
        else:
            ci_low = float(np.percentile(samples, 100 * alpha / 2))
            ci_high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        results[key] = {"value": value, "ci_low": ci_low, "ci_high": ci_high}
    return results


def build_performance_table(
    boot_results: dict,
    metric_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    metric_order = metric_order or DEFAULT_METRIC_ORDER
    rows = []
    for key in metric_order:
        if key not in boot_results:
            continue
        item = boot_results[key]
        rows.append({
            "Metric": METRIC_LABELS.get(key, key),
            "Value": round(item["value"], 4),
            "CI_Low": round(item["ci_low"], 4) if not np.isnan(item["ci_low"]) else None,
            "CI_High": round(item["ci_high"], 4) if not np.isnan(item["ci_high"]) else None,
            "Bootstrap_CI_95": format_bootstrap_ci(item["ci_low"], item["ci_high"]),
        })
    return pd.DataFrame(rows)


def save_performance_table(
    table_df: pd.DataFrame,
    save_dir: str,
    model_type: str | None = None,
    n_samples: int | None = None,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    note: str | None = None,
    log_func=None,
) -> str:
    """Metric / Value / 95% Bootstrap CI 표를 CSV + TXT 로 저장."""
    if log_func is None:
        log_func = print
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "performance_table.csv")
    txt_path = os.path.join(save_dir, "performance_table.txt")
    table_df.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Performance Metrics with 95% Bootstrap CI\n")
        if model_type:
            f.write(f"Model: {model_type}\n")
        if n_samples is not None:
            f.write(f"n = {n_samples}\n")
        f.write(f"Bootstrap resamples: {n_bootstraps}\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"{'Metric':<16} {'Value':>8}  {'95% Bootstrap CI':>18}\n")
        f.write("-" * 46 + "\n")
        for _, row in table_df.iterrows():
            f.write(
                f"{row['Metric']:<16} {row['Value']:>8.2f}  "
                f"{row['Bootstrap_CI_95']:>18}\n"
            )
        if note:
            f.write(f"\nNote:\n{note}\n")

    log_func(f"  ✓ Performance table saved ({csv_path})")
    return csv_path


def save_bootstrap_performance(
    y_true,
    y_pred_proba,
    save_dir: str,
    model_type: str | None = None,
    threshold: float = 0.5,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = 42,
    note: str | None = None,
    log_func=None,
) -> pd.DataFrame:
    """Bootstrap CI 계산 후 performance table 저장."""
    boot = bootstrap_metrics_ci(
        y_true, y_pred_proba,
        n_bootstraps=n_bootstraps, seed=seed, threshold=threshold,
    )
    table = build_performance_table(boot)
    save_performance_table(
        table, save_dir,
        model_type=model_type,
        n_samples=len(y_true),
        n_bootstraps=n_bootstraps,
        note=note,
        log_func=log_func,
    )
    return table


# ────────────────────────────────────────────────────────────────────────────
# SHAP 분석
# ────────────────────────────────────────────────────────────────────────────
def _select_class1_shap(sv):
    """shap_values 가 list[neg, pos] 또는 (n, features, 2) 일 때 class1 만 골라냄."""
    if isinstance(sv, list):
        return sv[1]
    arr = np.asarray(sv)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    return arr


def compute_shap(model_obj, model_type, X, feature_names, save_dir, log_func=None):
    """SHAP 분석을 수행하고 summary plot / values / importance 를 저장.

    Args:
        model_obj: 학습 완료된 모델 (sklearn / xgboost / tabnet)
        model_type: 모델 키 (xgboost, randomforest, ...)
        X: feature DataFrame
        feature_names: list[str]
        save_dir: 결과 저장 폴더
        log_func: 메시지 로깅용 콜백 (선택)
    """
    if log_func is None:
        log_func = print
    try:
        import shap  # noqa: F401
    except ImportError:
        log_func("  SHAP skipped (shap not installed)")
        return

    log_func("  Starting SHAP analysis...")
    try:
        if model_type in SHAP_TREE:
            explainer = shap.TreeExplainer(model_obj)
            sv = explainer.shap_values(X)
            shap_values = _select_class1_shap(sv)
            shap_data = X
        elif model_type in SHAP_LINEAR:
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.LinearExplainer(model_obj, background)
            shap_values = explainer.shap_values(X)
            shap_data = X
        elif model_type in SHAP_KERNEL:
            background = shap.sample(X, min(50, len(X)))
            sample_size = min(100, len(X))
            explainer = shap.KernelExplainer(
                lambda x: model_obj.predict_proba(x)[:, 1], background
            )
            shap_values = explainer.shap_values(X.iloc[:sample_size].values)
            shap_data = X.iloc[:sample_size]
        else:
            log_func(f"  SHAP skipped (unsupported model_type: {model_type})")
            return

        # summary plot
        plt.figure(figsize=(10, 8))
        if model_type == "tabnet":
            shap.summary_plot(
                shap_values, shap_data.values,
                feature_names=list(feature_names), show=False,
            )
        else:
            shap.summary_plot(shap_values, shap_data, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=300)
        plt.close()

        # values CSV
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(os.path.join(save_dir, "shap_values.csv"), index=False)

        # importance CSV (mean |SHAP|)
        shap_importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(os.path.join(save_dir, "shap_importance.csv"), index=False)

        log_func(f"  ✓ SHAP saved for {model_type}")
    except Exception as e:
        log_func(f"  ✗ SHAP error: {e}")
        traceback.print_exc()


# ────────────────────────────────────────────────────────────────────────────
# Permutation Feature Importance
# ────────────────────────────────────────────────────────────────────────────
def _custom_auc_scorer(model, X, y):
    """neural_network / tabnet 처럼 'predict' 가 라벨 반환하는 모델용 scorer."""
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values
    y_pred = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)


def compute_permutation_importance(model_obj, model_type, X_test, y_test,
                                  save_dir, n_repeats=30, log_func=None):
    """sklearn permutation_importance 로 feature importance 계산 후 저장."""
    if log_func is None:
        log_func = print
    log_func("  Starting Permutation Importance...")
    try:
        if model_type in ("neural_network", "tabnet"):
            perm = permutation_importance(
                model_obj, X_test.values if hasattr(X_test, "values") else X_test,
                y_test.values if hasattr(y_test, "values") else y_test,
                n_repeats=n_repeats, random_state=42,
                scoring=_custom_auc_scorer,
            )
        else:
            perm = permutation_importance(
                model_obj, X_test, y_test,
                n_repeats=n_repeats, random_state=42, scoring="roc_auc",
            )

        importances = perm.importances_mean
        stds = perm.importances_std
        features = list(X_test.columns)
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(8, 5))
        plt.bar(range(len(features)), importances[sorted_idx],
                yerr=stds[sorted_idx], align="center")
        plt.xticks(range(len(features)),
                   np.array(features)[sorted_idx], rotation=45, ha="right")
        plt.ylabel("Mean Importance (AUC decrease)")
        plt.title("Permutation Feature Importance (Test Set)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "permutation_importance.png"), dpi=300)
        plt.close()

        pd.DataFrame({
            "feature": np.array(features)[sorted_idx],
            "importance_mean": importances[sorted_idx],
            "importance_std": stds[sorted_idx],
        }).to_csv(os.path.join(save_dir, "permutation_importance.csv"), index=False)

        log_func("  ✓ Permutation importance saved")
    except Exception as e:
        log_func(f"  ✗ Permutation error: {e}")
        traceback.print_exc()


# ────────────────────────────────────────────────────────────────────────────
# CV 마지막 fold 기준 train/test 분리
# ────────────────────────────────────────────────────────────────────────────
def save_train_test_split(X, y, n_splits, save_dir, log_func=None):
    """StratifiedKFold 의 마지막 fold 로 train/test 분리 후 CSV 저장.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if log_func is None:
        log_func = print
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, test_idx = None, None
    for tr, te in kf.split(X, y):
        train_idx, test_idx = tr, te  # 마지막 fold 채택
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(save_dir, "train_set.csv"), index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(save_dir, "test_set.csv"), index=False
    )
    return X_train, X_test, y_train, y_test


# ────────────────────────────────────────────────────────────────────────────
# Test set 예측 결과 저장
# ────────────────────────────────────────────────────────────────────────────
def save_test_predictions(model_obj, model_type, X_train, X_test, y_train, y_test,
                          save_dir, log_func=None):
    """Test set 에 대해 학습 후 예측값/실측값을 CSV 로 저장."""
    if log_func is None:
        log_func = print
    try:
        # tabnet 은 numpy 입출력 필요
        if model_type == "tabnet":
            model_obj.fit(
                X_train.values, y_train.values,
                max_epochs=100, patience=15, batch_size=256,
                virtual_batch_size=128,
            )
            y_pred_proba = model_obj.predict_proba(X_test.values)[:, 1]
            y_pred_class = model_obj.predict(X_test.values)
        else:
            model_obj.fit(X_train, y_train)
            y_pred_proba = model_obj.predict_proba(X_test)[:, 1]
            y_pred_class = model_obj.predict(X_test)

        inference_df = pd.DataFrame({
            "original_row_number": X_test.index + 2,  # +2: 헤더(1) + 0-index 보정
            "true_label": y_test.values,
            "predicted_label": y_pred_class,
            "predicted_probability": y_pred_proba,
        })
        inference_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

        acc = (inference_df["true_label"] == inference_df["predicted_label"]).mean()
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = save_calibration_analysis(y_test, y_pred_proba, save_dir, log_func=log_func)
        save_bootstrap_performance(
            y_test.values, y_pred_proba, save_dir,
            model_type=model_type, log_func=log_func,
        )
        log_func(f"  ✓ test_predictions saved ({len(inference_df)} samples, "
                 f"Acc={acc:.4f}, AUC={auc:.4f}, Brier={brier:.4f})")
        return inference_df, brier
    except Exception as e:
        log_func(f"  ✗ test_predictions error: {e}")
        traceback.print_exc()
        return None


# ────────────────────────────────────────────────────────────────────────────
# Detailed analysis TXT
# ────────────────────────────────────────────────────────────────────────────
def save_detailed_analysis(save_dir, project, model_type, params,
                           feature_names, n_features, mean_auc, ci_low, ci_high,
                           auc_formatted, n_splits, y, preprocessing_info,
                           results_df):
    """example/ml_analysis_app.py 의 detailed_analysis.txt 형식 차용."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(save_dir, "detailed_analysis.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== Extended ML Model Analysis =====\n")
        f.write(f"Generated at: {ts}\n\n")
        f.write(f"Project: {project}\n")
        f.write(f"Best Model: Key only\n")
        f.write(f"AUC (CV Mean): {mean_auc:.4f}\n")
        f.write(f"AUC (95% CI): {auc_formatted}\n")
        f.write(f"Number of Features: {n_features}\n")
        f.write(f"Feature List: {list(feature_names)}\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("Model Parameters:\n")
        for k, v in (params or {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("Cross-Validation: StratifiedKFold\n")
        f.write(f"  n_splits: {n_splits}\n")
        f.write(f"  random_state: 42\n\n")
        f.write("Data Info:\n")
        f.write(f"  Total samples: {len(y)}\n")
        f.write(f"  Positive class: {int(y.sum())}\n")
        f.write(f"  Negative class: {int((1 - y).sum())}\n")
        f.write(f"  Target variable: {y.name if hasattr(y, 'name') else 'target'}\n")
        f.write("\n")
        f.write("Preprocessing Info:\n")
        for k, v in (preprocessing_info or {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("All model candidates evaluated:\n")
        for _, row in results_df.iterrows():
            f.write(f"  {row['Model']}: AUC={row['Mean_AUC']:.4f}, "
                    f"Features={row['n_features']}\n")
    return path
