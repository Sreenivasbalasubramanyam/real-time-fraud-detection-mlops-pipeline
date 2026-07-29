"""
Automated retraining trigger.

Checks two independent conditions and retrains (re-invokes
`src/models/train_model.py`) if either is breached:

  1. DRIFT breach — `drift_monitor.check_drift()` reports `overall_drift_detected`
     (i.e. any feature crosses the PSI "significant" threshold or fails the
     KS test at alpha=0.05).
  2. METRIC breach — the most recent MLflow(-lite) run's logged `auc_pr`
     (precision-recall AUC, the primary metric for an imbalanced fraud
     problem) has dropped below a configured floor, signalling model decay.

This mirrors a production MLOps retraining trigger: a scheduled job (cron /
Airflow DAG / GitHub Action) would run this script periodically against the
latest reference/production feature snapshots and the latest model's
tracked metrics, and re-train automatically when either signal fires.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.pandas_features import NUMERIC_FEATURE_COLUMNS, engineer_features, load_transactions  # noqa: E402
from src.monitoring.drift_monitor import check_drift  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MLFLOW_TRACKING_DIR = os.path.join(REPO_ROOT, "mlruns")
MIN_ACCEPTABLE_AUC_PR = 0.55  # business floor: below this, the model is considered decayed


def _latest_run_metric(metric_name: str) -> float | None:
    """Read the most recent run's metric value directly from the mlruns/
    file store (works whether it was written by real MLflow or mlflow_lite,
    since both use the same on-disk format: metrics/<name> = "<ts> <value> <step>"
    lines, most recent line last).
    """
    pattern = os.path.join(MLFLOW_TRACKING_DIR, "*", "*", "meta.yaml")
    run_dirs = []
    for meta_path in glob.glob(pattern):
        run_dir = os.path.dirname(meta_path)
        mtime = os.path.getmtime(meta_path)
        run_dirs.append((mtime, run_dir))

    if not run_dirs:
        return None

    run_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_run_dir = run_dirs[0][1]
    metric_path = os.path.join(latest_run_dir, "metrics", metric_name)
    if not os.path.exists(metric_path):
        return None

    with open(metric_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    # format: "<timestamp_ms> <value> <step>"
    parts = last_line.split()
    return float(parts[1])


def evaluate_triggers(
    reference_path: str,
    current_path: str,
    min_auc_pr: float = MIN_ACCEPTABLE_AUC_PR,
) -> dict:
    ref_raw = load_transactions(reference_path)
    cur_raw = load_transactions(current_path)
    ref_eng, freq_maps = engineer_features(ref_raw)
    cur_eng, _ = engineer_features(cur_raw, freq_maps=freq_maps)

    drift_report = check_drift(ref_eng, cur_eng, NUMERIC_FEATURE_COLUMNS)
    drift_breach = drift_report.overall_drift_detected

    latest_auc_pr = _latest_run_metric("auc_pr")
    metric_breach = (latest_auc_pr is not None) and (latest_auc_pr < min_auc_pr)

    should_retrain = drift_breach or metric_breach

    return {
        "drift_breach": drift_breach,
        "n_drifted_features": drift_report.n_drifted_features,
        "max_psi": drift_report.max_psi,
        "latest_auc_pr": latest_auc_pr,
        "min_acceptable_auc_pr": min_auc_pr,
        "metric_breach": metric_breach,
        "should_retrain": should_retrain,
        "drift_report": drift_report.to_dict(),
    }


def maybe_retrain(
    reference_path: str,
    current_path: str,
    min_auc_pr: float = MIN_ACCEPTABLE_AUC_PR,
    dry_run: bool = False,
) -> dict:
    decision = evaluate_triggers(reference_path, current_path, min_auc_pr=min_auc_pr)

    print("=== Retraining Trigger Evaluation ===")
    print(f"Drift breach   : {decision['drift_breach']}  "
          f"({decision['n_drifted_features']} feature(s) drifted, max PSI={decision['max_psi']:.4f})")
    print(f"Metric breach  : {decision['metric_breach']}  "
          f"(latest auc_pr={decision['latest_auc_pr']}, floor={decision['min_acceptable_auc_pr']})")
    print(f"=> should_retrain = {decision['should_retrain']}")

    if decision["should_retrain"] and not dry_run:
        print("\nThreshold breached — triggering src/models/train_model.py on the current batch...")
        train_script = os.path.join(REPO_ROOT, "src", "models", "train_model.py")
        result = subprocess.run(
            [sys.executable, train_script, "--data", current_path],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
        decision["retrain_triggered"] = True
        decision["retrain_returncode"] = result.returncode
    else:
        decision["retrain_triggered"] = False
        if not decision["should_retrain"]:
            print("\nNo threshold breached — skipping retraining.")
        else:
            print("\n[dry-run] Would have triggered retraining, but --dry-run was set.")

    return decision


def main():
    parser = argparse.ArgumentParser(description="Check drift/metric thresholds and retrain if breached.")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference (baseline) transactions CSV")
    parser.add_argument("--current", type=str, required=True, help="Path to current (production) transactions CSV")
    parser.add_argument("--min-auc-pr", type=float, default=MIN_ACCEPTABLE_AUC_PR)
    parser.add_argument("--dry-run", action="store_true", help="Evaluate triggers but never actually retrain")
    args = parser.parse_args()

    decision = maybe_retrain(
        reference_path=args.reference,
        current_path=args.current,
        min_auc_pr=args.min_auc_pr,
        dry_run=args.dry_run,
    )
    print("\n--- Full decision JSON ---")
    print(json.dumps({k: v for k, v in decision.items() if k != "drift_report"}, indent=2, default=str))


if __name__ == "__main__":
    main()
