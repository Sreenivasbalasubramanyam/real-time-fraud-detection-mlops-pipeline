"""
Helper script (used for local demo/testing of the drift monitor) that
generates a "reference" batch and two "current" batches: one with the same
generative process (no drift expected) and one with an injected distribution
shift (amount inflation + higher account-takeover velocity), so
`drift_monitor.py` / `retrain_trigger.py` can be demonstrated in both the
"stable" and "drifted" cases.

Usage:
    python data/generate_drift_batches.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generate_synthetic_transactions import generate_transactions

OUT_DIR = os.path.dirname(__file__)


def main():
    reference = generate_transactions(n_rows=6000, fraud_rate=0.012, seed=100)
    reference.to_csv(os.path.join(OUT_DIR, "reference_batch.csv"), index=False)

    current_stable = generate_transactions(n_rows=6000, fraud_rate=0.012, seed=200)
    current_stable.to_csv(os.path.join(OUT_DIR, "current_batch_stable.csv"), index=False)

    current_drifted = generate_transactions(n_rows=6000, fraud_rate=0.012, seed=300)
    rng = np.random.default_rng(999)
    # Simulate a real-world drift scenario: a merchant pricing change inflates
    # amounts, and a rise in account-takeover-style fraud increases velocity
    # and new-device usage broadly (not just among fraud-labeled rows).
    current_drifted["amount"] = current_drifted["amount"] * rng.uniform(2.0, 3.0, len(current_drifted))
    current_drifted["velocity_1h"] = current_drifted["velocity_1h"] + rng.poisson(3, len(current_drifted))
    current_drifted["distance_from_home_km"] = current_drifted["distance_from_home_km"] * rng.uniform(1.5, 2.5, len(current_drifted))
    current_drifted.to_csv(os.path.join(OUT_DIR, "current_batch_drifted.csv"), index=False)

    print("Wrote reference_batch.csv, current_batch_stable.csv, current_batch_drifted.csv")
    print("reference fraud rate:", reference["is_fraud"].mean())
    print("current_stable fraud rate:", current_stable["is_fraud"].mean())
    print("current_drifted fraud rate:", current_drifted["is_fraud"].mean())


if __name__ == "__main__":
    main()
