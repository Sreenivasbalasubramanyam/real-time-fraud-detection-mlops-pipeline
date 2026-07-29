"""
Minimal file-based experiment tracker that mirrors MLflow's on-disk tracking
store layout and exposes the same call pattern used by real `mlflow`:

    mlflow.set_tracking_uri(...)
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=...) as run:
        mlflow.log_param(k, v)
        mlflow.log_metric(k, v)
        mlflow.log_artifact(path)
        mlflow.xgboost.log_model(model, artifact_path="model", ...)

WHY THIS FILE EXISTS: `train_model.py` and `retrain_trigger.py` are written
against the real `mlflow` API. In this sandbox, outbound network access to
PyPI was blocked, so `pip install mlflow` could not run (see README "What
Actually Ran"). Rather than skip experiment tracking, this module
re-implements MLflow's local file store format directly:

    mlruns/<experiment_id>/meta.yaml
    mlruns/<experiment_id>/<run_id>/meta.yaml
    mlruns/<experiment_id>/<run_id>/params/<key>        (file content = value)
    mlruns/<experiment_id>/<run_id>/metrics/<key>       (file content = "<ts> <value> <step>")
    mlruns/<experiment_id>/<run_id>/tags/<key>
    mlruns/<experiment_id>/<run_id>/artifacts/...

This is the *actual* directory structure the real MLflow file store uses, so
runs produced by this shim are inspectable the same way (`ls mlruns/`, `cat
mlruns/<exp>/<run>/metrics/auc_roc`), and a drop-in swap to real `mlflow`
(when available) requires no changes to calling code — see the try/except
import in `train_model.py`.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

_TRACKING_DIR = "mlruns"
_ACTIVE_EXPERIMENT = {"id": "0", "name": "Default"}
_ACTIVE_RUN = {"run": None}


@dataclass
class RunInfo:
    run_id: str
    experiment_id: str
    run_name: str
    start_time: int


@dataclass
class ActiveRun:
    info: RunInfo
    _dir: str
    metric_steps: dict = field(default_factory=dict)


def set_tracking_uri(uri: str):
    global _TRACKING_DIR
    _TRACKING_DIR = uri.replace("file:", "", 1) if uri.startswith("file:") else uri
    os.makedirs(_TRACKING_DIR, exist_ok=True)


def _experiments_index() -> dict:
    path = os.path.join(_TRACKING_DIR, "_experiments_index.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_experiments_index(idx: dict):
    path = os.path.join(_TRACKING_DIR, "_experiments_index.json")
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)


def set_experiment(name: str):
    os.makedirs(_TRACKING_DIR, exist_ok=True)
    idx = _experiments_index()
    if name not in idx:
        exp_id = str(len(idx) + 1)
        idx[name] = exp_id
        _save_experiments_index(idx)
        exp_dir = os.path.join(_TRACKING_DIR, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "meta.yaml"), "w") as f:
            f.write(f"experiment_id: {exp_id}\nname: {name}\nartifact_location: {os.path.abspath(exp_dir)}\nlifecycle_stage: active\n")
    _ACTIVE_EXPERIMENT["id"] = idx[name]
    _ACTIVE_EXPERIMENT["name"] = name
    return idx[name]


@contextmanager
def start_run(run_name: str | None = None):
    exp_id = _ACTIVE_EXPERIMENT["id"]
    run_id = uuid.uuid4().hex
    run_name = run_name or f"run-{run_id[:8]}"
    run_dir = os.path.join(_TRACKING_DIR, exp_id, run_id)
    for sub in ("params", "metrics", "tags", "artifacts"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    start_time = int(time.time() * 1000)
    info = RunInfo(run_id=run_id, experiment_id=exp_id, run_name=run_name, start_time=start_time)
    with open(os.path.join(run_dir, "meta.yaml"), "w") as f:
        f.write(
            f"run_id: {run_id}\n"
            f"experiment_id: {exp_id}\n"
            f"run_name: {run_name}\n"
            f"status: RUNNING\n"
            f"start_time: {start_time}\n"
            f"artifact_uri: {os.path.abspath(os.path.join(run_dir, 'artifacts'))}\n"
        )
    with open(os.path.join(run_dir, "tags", "mlflow.runName"), "w") as f:
        f.write(run_name)

    active = ActiveRun(info=info, _dir=run_dir)
    _ACTIVE_RUN["run"] = active
    try:
        yield active
        status = "FINISHED"
    except Exception:
        status = "FAILED"
        raise
    finally:
        end_time = int(time.time() * 1000)
        meta_path = os.path.join(run_dir, "meta.yaml")
        with open(meta_path) as f:
            content = f.read()
        content = content.replace("status: RUNNING", f"status: {status}")
        content += f"end_time: {end_time}\n"
        with open(meta_path, "w") as f:
            f.write(content)
        _ACTIVE_RUN["run"] = None


def _current_run_dir() -> str:
    active = _ACTIVE_RUN["run"]
    if active is None:
        raise RuntimeError("No active MLflow(-lite) run. Call within `with start_run():`.")
    return active._dir


def log_param(key: str, value):
    run_dir = _current_run_dir()
    with open(os.path.join(run_dir, "params", key), "w") as f:
        f.write(str(value))


def log_metric(key: str, value: float, step: int = 0):
    run_dir = _current_run_dir()
    ts = int(time.time() * 1000)
    path = os.path.join(run_dir, "metrics", key)
    with open(path, "a") as f:
        f.write(f"{ts} {value} {step}\n")


def log_artifact(local_path: str):
    import shutil
    run_dir = _current_run_dir()
    dest_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(local_path, os.path.join(dest_dir, os.path.basename(local_path)))


def log_dict(dictionary: dict, artifact_file: str):
    run_dir = _current_run_dir()
    dest_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, artifact_file), "w") as f:
        json.dump(dictionary, f, indent=2)


class _XGBoostModule:
    """Stand-in for `mlflow.xgboost` — saves the model + a registry entry."""

    @staticmethod
    def log_model(model, artifact_path: str = "model", registered_model_name: str | None = None):
        run_dir = _current_run_dir()
        model_dir = os.path.join(run_dir, "artifacts", artifact_path)
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "model.json")
        model.save_model(model_file)
        with open(os.path.join(model_dir, "MLmodel"), "w") as f:
            f.write(f"artifact_path: {artifact_path}\nflavor: gbm_lite (xgboost-compatible)\n")

        if registered_model_name:
            registry_path = os.path.join(_TRACKING_DIR, "_model_registry.json")
            registry = {}
            if os.path.exists(registry_path):
                with open(registry_path) as f:
                    registry = json.load(f)
            versions = registry.setdefault(registered_model_name, [])
            active = _ACTIVE_RUN["run"]
            versions.append({
                "version": len(versions) + 1,
                "run_id": active.info.run_id,
                "source": model_dir,
            })
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)


xgboost = _XGBoostModule()
