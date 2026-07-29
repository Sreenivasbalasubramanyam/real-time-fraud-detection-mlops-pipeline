"""
Synthetic payment transaction generator.

Produces a class-imbalanced dataset (~0.5-1.5% fraud rate, similar to real-world
card-not-present fraud rates) with realistic-looking features: amount, merchant
category, geography mismatch, device/velocity signals, and time-of-day patterns.

Usage:
    python data/generate_synthetic_transactions.py --n-rows 50000 --fraud-rate 0.012
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "restaurant", "fuel",
    "online_retail", "utilities", "entertainment", "jewelry", "cash_advance",
]

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "BR", "NG", "RU", "CN"]

# Countries flagged as higher base-risk purely for synthetic realism (not a real claim).
HIGH_RISK_COUNTRIES = {"NG", "RU"}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_transactions(n_rows: int = 50_000, fraud_rate: float = 0.012, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic transactions dataset with a binary `is_fraud` label.

    The generative process intentionally creates overlapping-but-separable
    distributions between fraud and non-fraud classes so that a model trained
    on these features has a real (non-trivial) learning problem, similar to
    production fraud data.
    """
    rng = _rng(seed)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    start_time = datetime(2026, 1, 1)

    def make_block(n: int, fraud: bool) -> pd.DataFrame:
        # Transaction amount: legit purchases are log-normal & modest;
        # fraud skews toward either very small "card testing" or large cash-outs.
        if fraud:
            small_test = rng.lognormal(mean=1.0, sigma=0.4, size=n)      # card testing
            large_cashout = rng.lognormal(mean=6.2, sigma=0.7, size=n)    # cash-out
            mask = rng.random(n) < 0.35
            amount = np.where(mask, small_test, large_cashout)
        else:
            amount = rng.lognormal(mean=3.6, sigma=0.9, size=n)

        amount = np.clip(amount, 0.5, 25_000)

        # Hour of day: fraud disproportionately happens overnight (local merchant time).
        if fraud:
            hour = rng.choice(range(24), size=n, p=_night_weighted_probs())
        else:
            hour = rng.choice(range(24), size=n, p=_day_weighted_probs())

        merchant_category = rng.choice(
            MERCHANT_CATEGORIES,
            size=n,
            p=_merchant_probs(fraud),
        )

        country = rng.choice(COUNTRIES, size=n, p=_country_probs(fraud))

        # Distance between billing address and transaction location (km).
        if fraud:
            distance_km = rng.exponential(scale=900, size=n)
        else:
            distance_km = rng.exponential(scale=25, size=n)
        distance_km = np.clip(distance_km, 0, 20_000)

        # Velocity: number of transactions by this card in the last hour.
        if fraud:
            velocity_1h = rng.poisson(lam=4.5, size=n)
        else:
            velocity_1h = rng.poisson(lam=0.4, size=n)

        # Account age in days (fraud rings often use fresh/synthetic identities).
        if fraud:
            account_age_days = rng.exponential(scale=30, size=n)
        else:
            account_age_days = rng.exponential(scale=500, size=n)
        account_age_days = np.clip(account_age_days, 0, 5000)

        # Whether the CVV matched (fraud has a higher, but not total, mismatch rate).
        cvv_match = rng.random(n) > (0.35 if fraud else 0.02)

        # New device / new IP flags.
        is_new_device = rng.random(n) < (0.7 if fraud else 0.05)
        is_new_shipping_address = rng.random(n) < (0.55 if fraud else 0.04)

        # Random timestamps spread across ~120 days.
        day_offset = rng.integers(0, 120, size=n)
        minute_offset = rng.integers(0, 60, size=n)
        timestamps = [
            start_time + timedelta(days=int(d), hours=int(h), minutes=int(m))
            for d, h, m in zip(day_offset, hour, minute_offset)
        ]

        df = pd.DataFrame({
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "timestamp": timestamps,
            "amount": np.round(amount, 2),
            "merchant_category": merchant_category,
            "country": country,
            "distance_from_home_km": np.round(distance_km, 2),
            "velocity_1h": velocity_1h,
            "account_age_days": np.round(account_age_days, 1),
            "cvv_match": cvv_match,
            "is_new_device": is_new_device,
            "is_new_shipping_address": is_new_shipping_address,
            "is_fraud": int(fraud),
        })
        return df

    fraud_df = make_block(n_fraud, fraud=True)
    legit_df = make_block(n_legit, fraud=False)

    full = pd.concat([legit_df, fraud_df], ignore_index=True)
    full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    full["card_id"] = rng.integers(100000, 999999, size=len(full)).astype(str)
    full = full.sort_values("timestamp").reset_index(drop=True)
    return full


def _night_weighted_probs() -> np.ndarray:
    weights = np.array([3 if (h >= 0 and h <= 5) or h >= 22 else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


def _day_weighted_probs() -> np.ndarray:
    weights = np.array([0.3 if (h >= 0 and h <= 5) else (2.0 if 9 <= h <= 21 else 1.0) for h in range(24)], dtype=float)
    return weights / weights.sum()


def _merchant_probs(fraud: bool) -> np.ndarray:
    base = np.array([0.22, 0.10, 0.08, 0.16, 0.12, 0.14, 0.08, 0.06, 0.02, 0.02])
    if fraud:
        # Fraud skews toward electronics, online retail, jewelry, cash advance.
        skew = np.array([0.05, 0.20, 0.06, 0.05, 0.03, 0.24, 0.02, 0.05, 0.14, 0.16])
        probs = 0.35 * base + 0.65 * skew
    else:
        probs = base
    return probs / probs.sum()


def _country_probs(fraud: bool) -> np.ndarray:
    n = len(COUNTRIES)
    base = np.full(n, 1.0 / n)
    if fraud:
        boosted = np.array([0.10, 0.06, 0.08, 0.07, 0.06, 0.09, 0.07, 0.22, 0.20, 0.05])
        return boosted / boosted.sum()
    return base


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transaction data.")
    parser.add_argument("--n-rows", type=int, default=50_000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "transactions.csv"),
    )
    args = parser.parse_args()

    df = generate_transactions(n_rows=args.n_rows, fraud_rate=args.fraud_rate, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%} ({df['is_fraud'].sum():,} fraud / {len(df):,} total)")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
