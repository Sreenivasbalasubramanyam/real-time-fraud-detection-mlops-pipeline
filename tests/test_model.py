import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generate_synthetic_transactions import generate_transactions
from src.features.pandas_features import engineer_features, get_feature_matrix

try:
    import xgboost as xgb  # noqa: F401

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

from src.models import gbm


@pytest.fixture(scope="module")
def engineered_data():
    raw = generate_transactions(n_rows=4000, fraud_rate=0.05, seed=3)
    engineered, freq_maps = engineer_features(raw)
    X, y = get_feature_matrix(engineered)
    return X, y


def test_gbm_lite_fits_and_predicts_reasonable_auc(engineered_data):
    """The from-scratch numpy gradient boosting model should learn a real
    signal from the synthetic features (much better than random = 0.5 AUC).
    """
    from src.utils.metrics_lite import roc_auc_score, train_test_split

    X, y = engineered_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    model = gbm.GradientBoostedTreesClassifier(
        n_estimators=60, max_depth=4, learning_rate=0.2, scale_pos_weight=scale_pos_weight, random_state=0
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    auc = roc_auc_score(y_test, proba[:, 1])
    assert auc > 0.75, f"expected model to learn real signal, got AUC={auc:.3f}"


def test_gbm_lite_save_and_load_round_trip(engineered_data):
    X, y = engineered_data
    model = gbm.GradientBoostedTreesClassifier(n_estimators=10, max_depth=3, random_state=1)
    model.fit(X, y)
    preds_before = model.predict_proba(X)[:, 1]

    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "model.json")
        model.save_model(path)
        assert os.path.exists(path)

        loaded = gbm.GradientBoostedTreesClassifier()
        loaded.load_model(path)
        preds_after = loaded.predict_proba(X)[:, 1]

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-9)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_feature_importances_sum_to_one(engineered_data):
    X, y = engineered_data
    model = gbm.GradientBoostedTreesClassifier(n_estimators=20, max_depth=3, random_state=2)
    model.fit(X, y)
    importances = model.feature_importances_
    assert len(importances) == X.shape[1]
    assert np.isclose(importances.sum(), 1.0, atol=1e-6)
    assert (importances >= 0).all()


@pytest.mark.skipif(not _HAS_XGBOOST, reason="real xgboost not installed in this environment")
def test_real_xgboost_also_learns_signal(engineered_data):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = engineered_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc > 0.75


def test_train_model_end_to_end(tmp_path):
    """Full train_model.train() run against a small synthetic dataset, writing
    MLflow(-lite) artifacts and the model file to a temp models/mlruns dir.
    """
    from src.models import train_model

    data_path = tmp_path / "transactions.csv"
    df = generate_transactions(n_rows=3000, fraud_rate=0.04, seed=9)
    df.to_csv(data_path, index=False)

    original_models_dir = train_model.MODELS_DIR
    original_tracking_dir = train_model.MLFLOW_TRACKING_DIR
    try:
        train_model.MODELS_DIR = str(tmp_path / "models")
        train_model.MLFLOW_TRACKING_DIR = str(tmp_path / "mlruns")

        results = train_model.train(
            data_path=str(data_path),
            n_estimators=20,
            max_depth=3,
            register_model_name=None,
        )

        assert "run_id" in results
        assert 0.0 <= results["auc_roc"] <= 1.0
        assert 0.0 <= results["auc_pr"] <= 1.0
        assert os.path.exists(results["model_path"])

        # an MLflow(-lite)-style run directory should exist under the temp tracking dir
        run_dirs = list((tmp_path / "mlruns").glob("*/*/meta.yaml"))
        assert len(run_dirs) >= 1
    finally:
        train_model.MODELS_DIR = original_models_dir
        train_model.MLFLOW_TRACKING_DIR = original_tracking_dir
