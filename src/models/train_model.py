"""
Train an XGBoost fraud classifier on engineered transaction features, with
full MLflow experiment tracking (local file-based tracking store — no
tracking server required).

Logs:
  - params: model hyperparameters, train/test split sizes, fraud rate
  - metrics: AUC-ROC, average precision (PR-AUC), precision/recall/F1 @ 0.5,
    precision/recall at a business-tuned threshold
  - artifacts: the trained model (via mlflow.xgboost.log_model), a
    feature-importance plot data CSV, and the fitted category frequency maps

Usage:
    python src/models/train_model.py --data data/transactions.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.pandas_features import (  # noqa: E402
    engineer_features,
    get_feature_matrix,
    load_transactions,
)

# --- Backend selection: prefer real xgboost/mlflow/scikit-learn when the
# package is installed; otherwise fall back to the from-scratch, numpy-only
# implementations in src/models/gbm.py, src/utils/mlflow_lite.py and
# src/utils/metrics_lite.py. Both code paths expose the same call surface,
# so the rest of this script does not need to know which backend is active.
# See README "What Actually Ran" for which branch executed in this repo.
try:
    import xgboost as xgb  # type: ignore

    _XGB_BACKEND = "xgboost"

    def _make_model(**params):
        params.pop("gamma", None)
        params.pop("min_child_weight", None)
        return xgb.XGBClassifier(**params)

except ImportError:
    from src.models import gbm  # noqa: E402

    _XGB_BACKEND = "gbm_lite (numpy, xgboost-compatible)"

    def _make_model(**params):
        return gbm.GradientBoostedTreesClassifier(**params)

try:
    import mlflow  # type: ignore
    import mlflow.xgboost  # type: ignore

    _MLFLOW_BACKEND = "mlflow"
except ImportError:
    from src.utils import mlflow_lite as mlflow  # noqa: E402

    _MLFLOW_BACKEND = "mlflow_lite (file-based shim, MLflow-compatible directory layout)"

try:
    from sklearn.metrics import (  # type: ignore
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split  # type: ignore

    _SKLEARN_BACKEND = "scikit-learn"
except ImportError:
    from src.utils.metrics_lite import (  # noqa: E402
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        train_test_split,
    )

    _SKLEARN_BACKEND = "metrics_lite (numpy, scikit-learn-compatible)"

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "data", "transactions.csv")
MLFLOW_TRACKING_DIR = os.path.join(REPO_ROOT, "mlruns")
EXPERIMENT_NAME = "fraud-detection"


def _best_threshold_for_precision(y_true, y_scores, min_precision: float = 0.5) -> float:
    """Find the score threshold that maximizes recall subject to precision >= min_precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    best_thresh, best_recall = 0.5, 0.0
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_thresh = t
    return float(best_thresh)


def train(
    data_path: str = DEFAULT_DATA_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    register_model_name: str | None = "fraud-xgboost",
) -> dict:
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    raw = load_transactions(data_path)
    engineered, freq_maps = engineer_features(raw)

    X, y = get_feature_matrix(engineered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Class imbalance handling: weight positive (fraud) class inversely to its frequency.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "objective": "binary:logistic",
        "random_state": random_state,
        "n_jobs": -1,
    }

    print(f"[backends] model={_XGB_BACKEND} | tracking={_MLFLOW_BACKEND} | metrics={_SKLEARN_BACKEND}")

    with mlflow.start_run(run_name=f"xgboost-{int(time.time())}") as run:
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_rows_total", len(engineered))
        mlflow.log_param("n_rows_train", len(X_train))
        mlflow.log_param("n_rows_test", len(X_test))
        mlflow.log_param("fraud_rate_total", float(y.mean()))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("model_backend", _XGB_BACKEND)
        for k, v in params.items():
            if k != "scale_pos_weight":
                mlflow.log_param(k, v)

        model = _make_model(**params)
        t0 = time.time()
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        train_seconds = time.time() - t0

        y_scores = model.predict_proba(X_test)[:, 1]
        y_pred_05 = (y_scores >= 0.5).astype(int)

        y_test_arr = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

        auc_roc = roc_auc_score(y_test, y_scores)
        auc_pr = average_precision_score(y_test, y_scores)
        precision_05 = precision_score(y_test, y_pred_05, zero_division=0)
        recall_05 = recall_score(y_test, y_pred_05, zero_division=0)
        f1_05 = f1_score(y_test, y_pred_05, zero_division=0)

        business_thresh = _best_threshold_for_precision(y_test_arr, y_scores, min_precision=0.5)
        y_pred_biz = (y_scores >= business_thresh).astype(int)
        precision_biz = precision_score(y_test, y_pred_biz, zero_division=0)
        recall_biz = recall_score(y_test, y_pred_biz, zero_division=0)

        mlflow.log_metric("train_seconds", train_seconds)
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.log_metric("auc_pr", auc_pr)
        mlflow.log_metric("precision_at_0.5", precision_05)
        mlflow.log_metric("recall_at_0.5", recall_05)
        mlflow.log_metric("f1_at_0.5", f1_05)
        mlflow.log_metric("business_threshold", business_thresh)
        mlflow.log_metric("precision_at_business_threshold", precision_biz)
        mlflow.log_metric("recall_at_business_threshold", recall_biz)

        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        os.makedirs(MODELS_DIR, exist_ok=True)
        importance_path = os.path.join(MODELS_DIR, "feature_importance.csv")
        importances.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        freq_maps_path = os.path.join(MODELS_DIR, "freq_maps.json")
        with open(freq_maps_path, "w") as f:
            json.dump(freq_maps, f, indent=2)
        mlflow.log_artifact(freq_maps_path)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=register_model_name if register_model_name else None,
        )

        model_path = os.path.join(MODELS_DIR, "fraud_xgboost.json")
        model.save_model(model_path)

        threshold_path = os.path.join(MODELS_DIR, "serving_config.json")
        with open(threshold_path, "w") as f:
            json.dump({
                "scoring_threshold": business_thresh,
                "feature_columns": list(X.columns),
                "run_id": run.info.run_id,
            }, f, indent=2)

        results = {
            "run_id": run.info.run_id,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision_at_0.5": precision_05,
            "recall_at_0.5": recall_05,
            "f1_at_0.5": f1_05,
            "business_threshold": business_thresh,
            "precision_at_business_threshold": precision_biz,
            "recall_at_business_threshold": recall_biz,
            "n_rows_train": len(X_train),
            "n_rows_test": len(X_test),
            "model_path": model_path,
        }

        print(json.dumps(results, indent=2))
        return results


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost fraud model with MLflow tracking.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registry registration.")
    args = parser.parse_args()

    train(
        data_path=args.data,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        register_model_name=None if args.no_register else "fraud-xgboost",
    )


if __name__ == "__main__":
    main()
