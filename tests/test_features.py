import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generate_synthetic_transactions import generate_transactions
from src.features.pandas_features import (
    LABEL_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    add_amount_features,
    add_categorical_frequency_features,
    add_risk_flags,
    add_temporal_features,
    add_velocity_features,
    engineer_features,
    get_feature_matrix,
)


@pytest.fixture(scope="module")
def raw_transactions():
    return generate_transactions(n_rows=2000, fraud_rate=0.05, seed=1)


def test_generate_transactions_shape_and_columns(raw_transactions):
    assert len(raw_transactions) == 2000
    expected_cols = {
        "transaction_id", "timestamp", "amount", "merchant_category", "country",
        "distance_from_home_km", "velocity_1h", "account_age_days", "cvv_match",
        "is_new_device", "is_new_shipping_address", "is_fraud", "card_id",
    }
    assert expected_cols.issubset(set(raw_transactions.columns))


def test_fraud_rate_is_class_imbalanced(raw_transactions):
    rate = raw_transactions["is_fraud"].mean()
    # requested fraud_rate=0.05 but exact count rounding can shift it slightly
    assert 0.03 < rate < 0.08


def test_add_amount_features_adds_expected_columns(raw_transactions):
    df = add_amount_features(raw_transactions)
    assert "log_amount" in df.columns
    assert "amount_zscore_by_card" in df.columns
    assert np.isfinite(df["log_amount"]).all()
    assert not df["amount_zscore_by_card"].isna().any()


def test_add_temporal_features(raw_transactions):
    df = add_temporal_features(raw_transactions)
    assert df["hour_of_day"].between(0, 23).all()
    assert set(df["is_night"].unique()).issubset({0, 1})


def test_add_velocity_features_no_lookahead(raw_transactions):
    df = add_velocity_features(raw_transactions)
    # the very first transaction for any card must have zero prior transactions
    first_per_card = df.sort_values("timestamp").groupby("card_id").first()
    assert (first_per_card["txn_count_last_24h"] == 0).all()
    assert (first_per_card["amount_sum_last_24h"] == 0).all()
    assert (df["txn_count_last_24h"] >= 0).all()


def test_add_categorical_frequency_features_reuses_maps(raw_transactions):
    df1, maps = add_categorical_frequency_features(raw_transactions)
    assert "merchant_category_freq" in df1.columns
    assert "country_freq" in df1.columns
    assert set(maps.keys()) == {"merchant_category", "country"}

    # applying fitted maps to a fresh (unseen-category) batch should not error
    # and unseen categories should map to 0
    fresh = raw_transactions.copy()
    fresh.loc[0, "merchant_category"] = "totally_new_category"
    df2, _ = add_categorical_frequency_features(fresh, freq_maps=maps)
    assert df2.loc[0, "merchant_category_freq"] == 0.0


def test_add_risk_flags_casts_to_int(raw_transactions):
    df = add_risk_flags(raw_transactions)
    for col in ["cvv_match", "is_new_device", "is_new_shipping_address"]:
        assert df[col].dtype.kind in "iu"


def test_engineer_features_full_pipeline(raw_transactions):
    engineered, freq_maps = engineer_features(raw_transactions)
    for col in NUMERIC_FEATURE_COLUMNS:
        assert col in engineered.columns, f"missing feature column {col}"
    assert LABEL_COLUMN in engineered.columns
    assert len(engineered) == len(raw_transactions)
    assert not engineered[NUMERIC_FEATURE_COLUMNS].isna().any().any()


def test_get_feature_matrix_shapes(raw_transactions):
    engineered, _ = engineer_features(raw_transactions)
    X, y = get_feature_matrix(engineered)
    assert list(X.columns) == NUMERIC_FEATURE_COLUMNS
    assert len(X) == len(y) == len(raw_transactions)
    assert set(y.unique()).issubset({0, 1})


def test_engineered_features_are_deterministic(raw_transactions):
    e1, _ = engineer_features(raw_transactions)
    e2, _ = engineer_features(raw_transactions)
    pd.testing.assert_frame_equal(e1, e2)
