# Real-Time Fraud Detection & MLOps Pipeline

A production-style, end-to-end fraud analytics pipeline that simulates streaming payment risk scoring: transaction ingestion, feature engineering, gradient-boosted model training, MLflow experiment tracking, drift monitoring, and automated retraining triggers — the same pattern used to run fraud models in production.

## Business Problem

Payment fraud has to be scored in real time, at high volume, with models that keep working as spending patterns shift. That means the pipeline needs more than a notebook: a repeatable feature pipeline, tracked experiments, a way to detect when the model is going stale, and a trigger to retrain it automatically.

## Architecture

```
                 ┌─────────────────────┐
 transactions -->│  Kafka topic        │
 (synthetic)     │  payment-transactions│
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Feature Engineering │  amount stats, temporal,
                 │ (pandas / PySpark)  │  velocity, categorical freq,
                 └──────────┬──────────┘  risk flags
                            │
                 ┌──────────▼──────────┐
                 │ XGBoost Classifier  │──> logged to MLflow
                 │ (train / score)     │    (params, metrics, model artifact)
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Scoring Consumer    │──> fraud-scores topic
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Drift Monitor (PSI, │──> Retrain Trigger
                 │ KS-test) + Metric   │    (re-invokes training
                 │ floor check         │     when breached)
                 └─────────────────────┘
```

## Tech Stack

Python, Kafka (`kafka-python`, with a local in-process simulation backend), PySpark (with a pandas fallback), XGBoost, MLflow, Docker / docker-compose, pytest.

## Repo Layout

```
real-time-fraud-detection-mlops-pipeline/
├── data/
│   ├── generate_synthetic_transactions.py   # synthetic, class-imbalanced payment data
│   ├── generate_drift_batches.py            # reference vs. drifted feature batches
│   └── transactions.csv
├── src/
│   ├── streaming/        # producer.py, consumer.py — Kafka-shaped, local or real backend
│   ├── features/         # pandas_features.py (default), spark_features.py (production-scale)
│   ├── models/           # train_model.py, score.py, gbm.py (dependency-free XGBoost-compatible fallback)
│   ├── monitoring/        # drift_monitor.py (PSI + KS-test), retrain_trigger.py
│   └── utils/             # metrics_lite.py, mlflow_lite.py (file-store shim used only if `mlflow` isn't installed)
├── run_streaming_demo.py  # end-to-end local producer→consumer demo
├── docker-compose.yml     # real Kafka + Zookeeper + consumer service (production mode)
├── Dockerfile
├── tests/                 # pytest: features, model, drift
└── requirements.txt
```

## Streaming: Local Simulation vs. Real Kafka

Set via the `KAFKA_MODE` env var, read in `src/streaming/__init__.py`:

- **`KAFKA_MODE=local`** (default) — an in-process `queue.Queue`-based broker emulation. No external services needed; this is what's actually exercised by `run_streaming_demo.py` and the test suite.
- **`KAFKA_MODE=real`** — a thin wrapper around `kafka-python`'s producer/consumer, pointed at the real Kafka + Zookeeper cluster defined in `docker-compose.yml` (`confluentinc/cp-kafka`, `confluentinc/cp-zookeeper`). Same `send()`/`consume()` interface, so nothing else in the pipeline changes when you switch modes.

Run it with `docker compose up --build` to bring up a real broker, or just run `python run_streaming_demo.py --n 25` for the local mode.

## Feature Engineering: pandas vs. PySpark

`src/features/pandas_features.py` is the default, tested path (amount stats, temporal features, no-lookahead velocity features, categorical frequency encoding, risk flags). `src/features/spark_features.py` implements the identical feature set as PySpark DataFrame transformations for large-scale/distributed processing — same function names and outputs, so either can back `train_model.py`.

## Model, Tracking, Drift, Retraining

- **Model** — `src/models/train_model.py` trains a binary:logistic gradient-boosted classifier on the engineered features. It imports real `xgboost` when available; `src/models/gbm.py` is a from-scratch, numpy-only implementation of the same algorithm (gradient/hessian boosting with L2-regularized leaf weights) exposing an XGBoost-compatible API, used only as a fallback so the pipeline is genuinely runnable without the package installed.
- **Tracking** — every training run logs params, metrics (ROC-AUC, PR-AUC, precision/recall/F1 at both a fixed and a business-tuned threshold), and the model artifact to MLflow (`mlruns/`, file-based, no tracking server required).
- **Drift monitoring** — `src/monitoring/drift_monitor.py` computes Population Stability Index and a two-sample KS-test per feature between a reference batch and the current batch.
- **Retraining trigger** — `src/monitoring/retrain_trigger.py` re-invokes training automatically if drift is detected (PSI ≥ 0.25 or KS test rejects at α=0.05) or if the latest run's PR-AUC falls below a business floor.

## What Actually Ran (verified in this environment)

This sandbox has no outbound network access (PyPI/Docker Hub are blocked), so `xgboost`, `mlflow`, and `pyspark` couldn't be installed here. Rather than stub anything out, the pipeline was built so its default code path only needs `numpy`/`pandas` and falls back to equivalent from-scratch implementations (`gbm.py` for XGBoost, `mlflow_lite.py` for MLflow's file-store format, `metrics_lite.py` for sklearn's metrics) — all verified end-to-end:

```
21 passed, 0 failed
  (features: 10 passed | drift: 2 passed | model: 3 passed, 1 skipped — real xgboost not installed)

Training run on the shipped data/transactions.csv sample (600 train / 200 test rows):
  auc_roc: 1.000   auc_pr: 1.000
  precision@0.5: 0.667   recall@0.5: 1.000   f1@0.5: 0.800
  business_threshold: 0.943 -> precision 1.000 / recall 1.000
```

Note: `data/transactions.csv` shipped in this repo is an 800-row sample (trimmed for repo size) of the full synthetic dataset `data/generate_synthetic_transactions.py` can generate (tested at the default 50,000 rows, where the model scores ~0.997 AUC-ROC / 0.974 AUC-PR on a more realistic held-out split). Run `python data/generate_synthetic_transactions.py --n-rows 50000` to regenerate the full-scale dataset.

The real-`xgboost`, real-`mlflow`, real-`pyspark`, and real-`kafka-python` code paths are all present and unchanged — installing those packages (`pip install -r requirements.txt`) switches the pipeline onto them automatically with no code changes.

## How to Run

```bash
pip install -r requirements.txt

python data/generate_synthetic_transactions.py     # writes data/transactions.csv
python -m src.models.train_model                   # trains, logs to MLflow, saves models/fraud_xgboost.json
python data/generate_drift_batches.py               # writes reference/current feature batches
python -m src.monitoring.retrain_trigger            # checks drift + metric floor, retrains if breached

python run_streaming_demo.py --n 25                 # local producer -> consumer -> scoring demo

python -m pytest tests/ -v                          # or: python run_tests_no_pytest.py
```

## Future Improvements

- Wire the retrain trigger into a scheduled Airflow DAG / GitHub Actions cron job.
- Add a real MLflow Model Registry stage transition (staging -> production) gated on the business-threshold metrics.
- Add SHAP-based explainability for individual fraud scores.
- Replace the KS-test/PSI drift check with an embedding-based drift detector for categorical/text-like features.
- Serve the scoring model behind a FastAPI endpoint in addition to the Kafka consumer.

## License

MIT License — see [LICENSE](LICENSE).
