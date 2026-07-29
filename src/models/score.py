"""
Load the trained XGBoost fraud model and score incoming transaction messages.

Used by `src/streaming/consumer.py` to turn each streamed transaction JSON
message into a fraud risk score + decision, mirroring how a real-time
scoring service would consume from a Kafka topic and call a model.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import xgboost as xgb  # type: ignore

    def _new_model():
        return xgb.XGBClassifier()

except ImportError:
    from src.models import gbm  # noqa: E402

    def _new_model():
        return gbm.GradientBoostedTreesClassifier()

from src.features.pandas_features import (  # noqa: E402
    NUMERIC_FEATURE_COLUMNS,
    add_amount_features,
    add_categorical_frequency_features,
    add_risk_flags,
    add_temporal_features,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "fraud_xgboost.json")
DEFAULT_CONFIG_PATH = os.path.join(MODELS_DIR, "serving_config.json")
DEFAULT_FREQ_MAPS_PATH = os.path.join(MODELS_DIR, "freq_maps.json")


@dataclass
class ScoringResult:
    transaction_id: str
    card_id: str
    fraud_probability: float
    is_flagged: bool
    threshold_used: float
    raw_transaction: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "transaction_id": self.transaction_id,
            "card_id": self.card_id,
            "fraud_probability": round(self.fraud_probability, 6),
            "is_flagged": self.is_flagged,
            "threshold_used": self.threshold_used,
        })


class FraudScorer:
    """Loads a trained model + serving config once, then scores transactions
    one at a time (as a streaming consumer would) or in small batches.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        freq_maps_path: str = DEFAULT_FREQ_MAPS_PATH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run src/models/train_model.py first."
            )
        self.model = _new_model()
        self.model.load_model(model_path)

        with open(config_path) as f:
            self.config = json.load(f)
        self.threshold = self.config["scoring_threshold"]
        self.feature_columns = self.config["feature_columns"]

        with open(freq_maps_path) as f:
            self.freq_maps = json.load(f)

    def _featurize_single(self, txn: dict[str, Any]) -> pd.DataFrame:
        """Turn a single raw transaction dict (as it arrives from the streaming
        layer) into a one-row engineered feature frame. Velocity features
        (txn_count_last_24h / amount_sum_last_24h) default to 0 for a single
        streamed event with no historical window materialized; in production
        these would be computed from a feature store / stateful stream join.
        """
        df = pd.DataFrame([txn])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["amount_zscore_by_card"] = 0.0  # no per-card history available online
        df = add_amount_features(df.drop(columns=["amount_zscore_by_card"]))
        df = add_temporal_features(df)
        df["txn_count_last_24h"] = txn.get("velocity_1h", 0)
        df["amount_sum_last_24h"] = df["amount"]
        df = add_risk_flags(df)
        df, _ = add_categorical_frequency_features(df, freq_maps=self.freq_maps)
        return df

    def score_transaction(self, txn: dict[str, Any]) -> ScoringResult:
        features_df = self._featurize_single(txn)
        X = features_df[self.feature_columns].astype(float)
        proba = float(self.model.predict_proba(X)[0, 1])
        return ScoringResult(
            transaction_id=txn.get("transaction_id", "unknown"),
            card_id=txn.get("card_id", "unknown"),
            fraud_probability=proba,
            is_flagged=proba >= self.threshold,
            threshold_used=self.threshold,
            raw_transaction=txn,
        )

    def score_batch(self, transactions: list[dict[str, Any]]) -> list[ScoringResult]:
        return [self.score_transaction(t) for t in transactions]


if __name__ == "__main__":
    # Quick manual smoke test using a couple of hand-built transactions.
    scorer = FraudScorer()
    sample_legit = {
        "transaction_id": "demo-legit-1",
        "card_id": "123456",
        "timestamp": "2026-01-05T14:22:00",
        "amount": 42.50,
        "merchant_category": "grocery",
        "country": "US",
        "distance_from_home_km": 3.2,
        "velocity_1h": 0,
        "account_age_days": 900,
        "cvv_match": True,
        "is_new_device": False,
        "is_new_shipping_address": False,
    }
    sample_fraud = {
        "transaction_id": "demo-fraud-1",
        "card_id": "999999",
        "timestamp": "2026-01-05T03:14:00",
        "amount": 2450.00,
        "merchant_category": "cash_advance",
        "country": "NG",
        "distance_from_home_km": 4200.0,
        "velocity_1h": 6,
        "account_age_days": 2,
        "cvv_match": False,
        "is_new_device": True,
        "is_new_shipping_address": True,
    }
    for txn in [sample_legit, sample_fraud]:
        result = scorer.score_transaction(txn)
        print(result.to_json())
