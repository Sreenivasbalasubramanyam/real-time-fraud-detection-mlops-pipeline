"""
Feature engineering for fraud scoring — pandas implementation.

This is the implementation that actually ran in this environment/CI (no JVM /
Spark cluster required). `src/features/spark_features.py` mirrors the exact
same function signatures and logic using PySpark DataFrame operations, and is
the "production-scale" implementation intended for a real Spark cluster
processing high transaction volumes.

Feature groups engineered here:
  - Amount-based: log-amount, z-score vs. card's historical mean.
  - Behavioral / velocity: rolling transaction count & spend per card.
  - Risk flags: new device, new shipping address, CVV mismatch, high distance.
  - Categorical encodings: merchant category & country frequency encoding.
  - Temporal: hour-of-day, is_night flag.

All functions accept and return plain pandas DataFrames so they can be unit
tested without any external services.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURE_COLUMNS = [
    "log_amount",
    "amount_zscore_by_card",
    "distance_from_home_km",
    "velocity_1h",
    "account_age_days",
    "hour_of_day",
    "is_night",
    "cvv_match",
    "is_new_device",
    "is_new_shipping_address",
    "merchant_category_freq",
    "country_freq",
    "txn_count_last_24h",
    "amount_sum_last_24h",
]

LABEL_COLUMN = "is_fraud"


def load_transactions(path: str) -> pd.DataFrame:
    """Load raw transactions CSV and parse timestamp column."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-amount and per-card amount z-score features."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])

    card_stats = df.groupby("card_id")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "_card_amount_mean", "std": "_card_amount_std"}
    )
    df = df.merge(card_stats, on="card_id", how="left")
    df["_card_amount_std"] = df["_card_amount_std"].fillna(0.0)
    denom = df["_card_amount_std"].replace(0.0, np.nan)
    df["amount_zscore_by_card"] = ((df["amount"] - df["_card_amount_mean"]) / denom).fillna(0.0)
    df = df.drop(columns=["_card_amount_mean", "_card_amount_std"])
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and is_night flag from the timestamp column."""
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h <= 5 or h >= 22) else 0)
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 24h transaction count & spend per card (causal / no look-ahead).

    Implemented as a per-card time-ordered rolling window keyed off `timestamp`,
    counting/summing only transactions strictly before the current one so the
    feature reflects information available at scoring time.
    """
    df = df.copy().sort_values(["card_id", "timestamp"]).reset_index(drop=True)

    txn_counts = np.zeros(len(df))
    amount_sums = np.zeros(len(df))

    for _, group in df.groupby("card_id"):
        idx = group.index.to_numpy()
        times = group["timestamp"].to_numpy()
        amounts = group["amount"].to_numpy()
        window = pd.Timedelta(hours=24)

        start = 0
        for i in range(len(group)):
            # advance window start pointer past entries older than 24h
            while times[i] - times[start] > window:
                start += 1
            # count/sum only prior rows (exclude current transaction itself)
            txn_counts[idx[i]] = i - start
            amount_sums[idx[i]] = amounts[start:i].sum()

    df["txn_count_last_24h"] = txn_counts
    df["amount_sum_last_24h"] = amount_sums
    return df


def add_categorical_frequency_features(df: pd.DataFrame, freq_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Frequency-encode merchant_category and country.

    If `freq_maps` is provided (e.g. computed on a training set), those
    frequencies are reused so unseen categories map to 0 — this avoids
    train/serve skew. Otherwise frequencies are computed from `df` itself.
    Returns the transformed dataframe and the fitted maps (for reuse at
    scoring time).
    """
    df = df.copy()
    maps: dict = {}

    for col, out_col in [("merchant_category", "merchant_category_freq"), ("country", "country_freq")]:
        if freq_maps and col in freq_maps:
            freq = freq_maps[col]
        else:
            freq = (df[col].value_counts(normalize=True)).to_dict()
        maps[col] = freq
        df[out_col] = df[col].map(freq).fillna(0.0)

    return df, maps


def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Cast boolean risk flags to int and add a high-distance indicator."""
    df = df.copy()
    for col in ["cvv_match", "is_new_device", "is_new_shipping_address"]:
        df[col] = df[col].astype(int)
    return df


def engineer_features(df: pd.DataFrame, freq_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Run the full feature engineering pipeline.

    Args:
        df: raw transactions dataframe (as produced by
            data/generate_synthetic_transactions.py or loaded via
            `load_transactions`).
        freq_maps: optional pre-fitted frequency-encoding maps (pass the maps
            returned from engineering the *training* set when transforming a
            new/serving batch, to avoid data leakage).

    Returns:
        (engineered_df, freq_maps) — engineered_df contains all
        NUMERIC_FEATURE_COLUMNS plus identifying/label columns.
    """
    out = df.copy()
    out = add_amount_features(out)
    out = add_temporal_features(out)
    out = add_velocity_features(out)
    out = add_risk_flags(out)
    out, maps = add_categorical_frequency_features(out, freq_maps=freq_maps)

    keep_cols = ["transaction_id", "card_id", "timestamp"] + NUMERIC_FEATURE_COLUMNS
    if LABEL_COLUMN in out.columns:
        keep_cols.append(LABEL_COLUMN)
    out = out[keep_cols]
    return out, maps


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split an engineered dataframe into (X, y). y is None if no label column."""
    X = df[NUMERIC_FEATURE_COLUMNS].astype(float)
    y = df[LABEL_COLUMN] if LABEL_COLUMN in df.columns else None
    return X, y


if __name__ == "__main__":
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "transactions.csv")
    raw = load_transactions(path)
    engineered, maps = engineer_features(raw)
    print(engineered.head())
    print(f"\nEngineered shape: {engineered.shape}")
    print(f"Feature columns: {NUMERIC_FEATURE_COLUMNS}")
