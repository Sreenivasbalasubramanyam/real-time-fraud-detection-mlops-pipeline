"""
Data / feature drift monitoring for the fraud model, using two standard,
industry-common statistical tests:

  1. Population Stability Index (PSI) — bins a reference distribution into
     deciles, then measures how much the current distribution's mass has
     shifted across those same bins. Rule-of-thumb thresholds (used widely
     in credit-risk / fraud modeling):
         PSI < 0.1              -> no significant shift
         0.1 <= PSI < 0.25       -> moderate shift, investigate
         PSI >= 0.25             -> significant shift, retrain likely needed

  2. Two-sample Kolmogorov-Smirnov (KS) test — the maximum distance between
     the reference and current empirical CDFs, with a distribution-free
     critical-value approximation for the p-value (Marsaglia/Kolmogorov
     asymptotic formula), so no external stats package is required.

Both are computed per numeric feature and aggregated into a single
DriftReport, which `retrain_trigger.py` consumes to decide whether to
kick off a retraining run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25
KS_ALPHA = 0.05  # significance level for the KS test p-value


def population_stability_index(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI between a reference and current 1-D numeric distribution.

    Bin edges are derived from the reference distribution's quantiles so each
    reference bin starts with ~equal mass; the current distribution is then
    binned into those same edges.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # Degenerate distribution (near-constant); fall back to min/max split.
        edges = np.linspace(reference.min(), reference.max() + 1e-9, n_bins + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1)
    cur_pct = cur_counts / max(len(current), 1)

    # Laplace smoothing to avoid log(0) / division by zero for empty bins.
    eps = 1e-4
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _ks_p_value(d_stat: float, n1: int, n2: int) -> float:
    """Asymptotic two-sided KS p-value via the Kolmogorov distribution
    (Marsaglia-Marshall-Wilk style series approximation). Matches
    scipy.stats.ks_2samp's asymptotic method closely for reasonably-sized n.
    """
    n_eff = (n1 * n2) / (n1 + n2)
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * d_stat
    if lam < 0.2:
        return 1.0

    total = 0.0
    for k in range(1, 101):
        term = ((-1) ** (k - 1)) * np.exp(-2 * (lam ** 2) * (k ** 2))
        total += term
    p = 2 * total
    return float(np.clip(p, 0.0, 1.0))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test statistic and asymptotic p-value."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    n1, n2 = len(reference), len(current)

    all_values = np.concatenate([reference, current])
    all_values.sort()

    cdf1 = np.searchsorted(reference, all_values, side="right") / n1
    cdf2 = np.searchsorted(current, all_values, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    p_value = _ks_p_value(d_stat, n1, n2)
    return d_stat, p_value


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    psi_flag: str  # "stable" | "moderate" | "significant"
    ks_flag: bool   # True if p_value < KS_ALPHA (distributions significantly differ)

    @property
    def is_drifted(self) -> bool:
        return self.psi_flag == "significant" or self.ks_flag


@dataclass
class DriftReport:
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    n_reference: int = 0
    n_current: int = 0

    @property
    def n_drifted_features(self) -> int:
        return sum(1 for r in self.feature_results if r.is_drifted)

    @property
    def overall_drift_detected(self) -> bool:
        return self.n_drifted_features > 0

    @property
    def max_psi(self) -> float:
        return max((r.psi for r in self.feature_results), default=0.0)

    def to_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "overall_drift_detected": self.overall_drift_detected,
            "n_drifted_features": self.n_drifted_features,
            "max_psi": self.max_psi,
            "features": [
                {
                    "feature": r.feature,
                    "psi": round(r.psi, 5),
                    "psi_flag": r.psi_flag,
                    "ks_statistic": round(r.ks_statistic, 5),
                    "ks_p_value": round(r.ks_p_value, 5),
                    "ks_flag": r.ks_flag,
                    "is_drifted": r.is_drifted,
                }
                for r in self.feature_results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            f"Drift report: {self.n_drifted_features}/{len(self.feature_results)} features drifted "
            f"(reference n={self.n_reference}, current n={self.n_current})",
        ]
        for r in sorted(self.feature_results, key=lambda x: -x.psi):
            marker = "DRIFT" if r.is_drifted else "ok"
            lines.append(
                f"  [{marker:5s}] {r.feature:28s} PSI={r.psi:7.4f} ({r.psi_flag:11s})  "
                f"KS={r.ks_statistic:6.4f} p={r.ks_p_value:.4f}"
            )
        return "\n".join(lines)


def _psi_flag(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "stable"


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    n_bins: int = 10,
) -> DriftReport:
    """Run PSI + KS-test drift checks for every column in `feature_columns`
    that is present in both dataframes.
    """
    results = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_vals = reference_df[col].to_numpy(dtype=float)
        cur_vals = current_df[col].to_numpy(dtype=float)

        psi = population_stability_index(ref_vals, cur_vals, n_bins=n_bins)
        ks_stat, ks_p = ks_test(ref_vals, cur_vals)

        results.append(FeatureDriftResult(
            feature=col,
            psi=psi,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            psi_flag=_psi_flag(psi),
            ks_flag=bool(ks_p < KS_ALPHA),
        ))

    return DriftReport(feature_results=results, n_reference=len(reference_df), n_current=len(current_df))


if __name__ == "__main__":
    import argparse
    import os

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.features.pandas_features import NUMERIC_FEATURE_COLUMNS, engineer_features, load_transactions

    parser = argparse.ArgumentParser(description="Run PSI/KS drift check between two transaction batches.")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--current", type=str, required=True)
    args = parser.parse_args()

    ref_raw = load_transactions(args.reference)
    cur_raw = load_transactions(args.current)
    ref_eng, freq_maps = engineer_features(ref_raw)
    cur_eng, _ = engineer_features(cur_raw, freq_maps=freq_maps)

    report = check_drift(ref_eng, cur_eng, NUMERIC_FEATURE_COLUMNS)
    print(report.summary())
    print()
    print(report.to_json())
