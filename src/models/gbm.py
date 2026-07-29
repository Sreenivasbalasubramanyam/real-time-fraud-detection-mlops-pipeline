"""
Minimal, dependency-free (numpy-only) gradient-boosted trees classifier,
implementing the same core algorithm XGBoost uses: additive regression trees
fit to the first- and second-order gradients (gradient/hessian) of the
logistic loss, with L2 leaf-weight regularization.

WHY THIS FILE EXISTS: this repo's reference implementation is real XGBoost
(see `train_model.py`, which imports `xgboost` when available). In this
sandbox environment, outbound network access to PyPI was blocked (see
README "What Actually Ran"), so `pip install xgboost` was not possible.
Rather than stub the model out, this module implements real gradient
boosting from first principles so the training/evaluation/MLflow-logging
pipeline is genuinely exercised end-to-end. It exposes an XGBoost-compatible
surface (`fit`, `predict_proba`, `save_model`, `load_model`,
`feature_importances_`) so `train_model.py` and `score.py` do not need to
know which backend is active — see the import shim at the top of
`train_model.py`.

Algorithm (binary:logistic objective, matching XGBoost's default):
  For each boosting round:
    1. p = sigmoid(F(x))                      (current model's predicted prob)
    2. g_i = p_i - y_i                        (gradient of log loss)
    3. h_i = p_i * (1 - p_i)                  (hessian of log loss)
    4. Fit a regression tree to (g_i, h_i) via greedy best-split search that
       maximizes the standard XGBoost gain formula, with L2 leaf reg (lambda)
       and a minimum child weight.
    5. Leaf value = -sum(g) / (sum(h) + lambda)
    6. F(x) += learning_rate * tree(x)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class _TreeNode:
    is_leaf: bool = True
    value: float = 0.0
    feature_index: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    def to_dict(self) -> dict:
        d = {"is_leaf": bool(self.is_leaf), "value": float(self.value)}
        if not self.is_leaf:
            d["feature_index"] = int(self.feature_index)
            d["threshold"] = float(self.threshold)
            d["left"] = self.left.to_dict()
            d["right"] = self.right.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "_TreeNode":
        node = _TreeNode(is_leaf=d["is_leaf"], value=d["value"])
        if not node.is_leaf:
            node.feature_index = d["feature_index"]
            node.threshold = d["threshold"]
            node.left = _TreeNode.from_dict(d["left"])
            node.right = _TreeNode.from_dict(d["right"])
        return node


class _RegressionTree:
    """A single CART regression tree fit to gradients/hessians (XGBoost-style)."""

    def __init__(self, max_depth: int = 5, min_child_weight: float = 1.0,
                 reg_lambda: float = 1.0, gamma: float = 0.0,
                 colsample: float = 1.0, rng: np.random.Generator | None = None):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.colsample = colsample
        self.rng = rng or np.random.default_rng(0)
        self.root: _TreeNode | None = None
        self.feature_gain: np.ndarray | None = None

    def _leaf_value(self, g: np.ndarray, h: np.ndarray) -> float:
        return -g.sum() / (h.sum() + self.reg_lambda)

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, n_features_total: int):
        self.feature_gain = np.zeros(n_features_total)
        self.root = self._build(X, g, h, depth=0)
        return self

    def _best_split_for_feature(self, col: np.ndarray, g: np.ndarray, h: np.ndarray):
        """O(n log n) exact-greedy split search for one feature: sort rows by
        feature value, then scan cumulative gradient/hessian sums once to
        evaluate every possible split point (the standard XGBoost exact-greedy
        trick), instead of re-masking the full array per candidate threshold.
        """
        order = np.argsort(col, kind="mergesort")
        col_sorted = col[order]
        g_sorted = g[order]
        h_sorted = h[order]

        g_cum = np.cumsum(g_sorted)
        h_cum = np.cumsum(h_sorted)
        g_total, h_total = g_cum[-1], h_cum[-1]

        # Only consider split points where the feature value actually changes
        # (splitting between two rows with an identical value is meaningless).
        distinct = np.where(np.diff(col_sorted) != 0)[0]
        if len(distinct) == 0:
            return -np.inf, None

        gl = g_cum[distinct]
        hl = h_cum[distinct]
        gr = g_total - gl
        hr = h_total - hl

        valid = (hl >= self.min_child_weight) & (hr >= self.min_child_weight)
        if not valid.any():
            return -np.inf, None

        gains = 0.5 * (
            (gl ** 2) / (hl + self.reg_lambda)
            + (gr ** 2) / (hr + self.reg_lambda)
            - ((g_total ** 2) / (h_total + self.reg_lambda))
        ) - self.gamma
        gains = np.where(valid, gains, -np.inf)

        best_local_idx = int(np.argmax(gains))
        best_gain = float(gains[best_local_idx])
        if not np.isfinite(best_gain):
            return -np.inf, None

        split_pos = distinct[best_local_idx]
        threshold = (col_sorted[split_pos] + col_sorted[split_pos + 1]) / 2.0
        return best_gain, threshold

    def _build(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> _TreeNode:
        node = _TreeNode(is_leaf=True, value=self._leaf_value(g, h))
        if depth >= self.max_depth or len(g) < 2 * self.min_child_weight:
            return node

        n_features = X.shape[1]
        if self.colsample < 1.0:
            k = max(1, int(round(n_features * self.colsample)))
            candidate_features = self.rng.choice(n_features, size=k, replace=False)
        else:
            candidate_features = range(n_features)

        best_gain = 0.0
        best_feat, best_thresh = -1, None

        for f in candidate_features:
            col = X[:, f]
            if col.max() == col.min():
                continue
            gain, threshold = self._best_split_for_feature(col, g, h)
            if gain > best_gain:
                best_gain = gain
                best_feat, best_thresh = f, threshold

        if best_feat == -1 or best_gain <= 0:
            return node

        best_mask = X[:, best_feat] <= best_thresh

        self.feature_gain[best_feat] += best_gain
        node = _TreeNode(is_leaf=False, feature_index=best_feat, threshold=best_thresh)
        node.left = self._build(X[best_mask], g[best_mask], h[best_mask], depth + 1)
        node.right = self._build(X[~best_mask], g[~best_mask], h[~best_mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction: recurses on row-masks instead of a per-row
        Python loop, so a full pass over X touches each tree node once
        (rather than each tree node once per row).
        """
        out = np.zeros(len(X))
        self._predict_recursive(self.root, X, np.arange(len(X)), out)
        return out

    def _predict_recursive(self, node: _TreeNode, X: np.ndarray, row_idx: np.ndarray, out: np.ndarray):
        if len(row_idx) == 0:
            return
        if node.is_leaf:
            out[row_idx] = node.value
            return
        col = X[row_idx, node.feature_index]
        left_mask = col <= node.threshold
        self._predict_recursive(node.left, X, row_idx[left_mask], out)
        self._predict_recursive(node.right, X, row_idx[~left_mask], out)


class GradientBoostedTreesClassifier:
    """XGBoost-compatible-surface classifier: fit/predict_proba/save_model/load_model.

    Accepts the same constructor keyword names used in train_model.py
    (n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    scale_pos_weight, reg_lambda, random_state) so it's a drop-in fallback.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        scale_pos_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        random_state: int = 42,
        **_ignored,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.random_state = random_state

        self.trees: list[_RegressionTree] = []
        self.base_score: float = 0.0
        self.feature_names_: list[str] | None = None
        self.n_features_: int | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y, eval_set=None, verbose: bool = False):
        X_arr = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        y_arr = y.to_numpy(dtype=float) if hasattr(y, "to_numpy") else np.asarray(y, dtype=float)
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X_arr.shape[1])]
        self.n_features_ = X_arr.shape[1]

        n = len(y_arr)
        # sample weights implement scale_pos_weight (upweight the positive/fraud class)
        sample_weight = np.where(y_arr == 1, self.scale_pos_weight, 1.0)

        p0 = np.clip(y_arr.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(p0 / (1 - p0)))  # logit of base rate
        F = np.full(n, self.base_score)

        self.trees = []
        self._feature_importance_raw = np.zeros(self.n_features_)

        for round_i in range(self.n_estimators):
            p = _sigmoid(F)
            g = (p - y_arr) * sample_weight
            h = (p * (1 - p)) * sample_weight

            if self.subsample < 1.0:
                idx = self._rng.choice(n, size=int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)

            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                colsample=self.colsample_bytree,
                rng=self._rng,
            )
            tree.fit(X_arr[idx], g[idx], h[idx], n_features_total=self.n_features_)
            update = tree.predict(X_arr)
            F += self.learning_rate * update
            self.trees.append(tree)
            self._feature_importance_raw += tree.feature_gain

            if verbose and (round_i + 1) % 25 == 0:
                loss = -np.mean(y_arr * np.log(np.clip(p, 1e-9, 1)) + (1 - y_arr) * np.log(np.clip(1 - p, 1e-9, 1)))
                print(f"[round {round_i + 1}/{self.n_estimators}] logloss={loss:.5f}")

        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        total = self._feature_importance_raw.sum()
        if total <= 0:
            return np.ones(self.n_features_) / self.n_features_
        return self._feature_importance_raw / total

    def _raw_score(self, X) -> np.ndarray:
        X_arr = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
        F = np.full(len(X_arr), self.base_score)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X_arr)
        return F

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self._raw_score(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save_model(self, path: str):
        payload = {
            "backend": "gbm_lite_v1",
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "scale_pos_weight": self.scale_pos_weight,
                "reg_lambda": self.reg_lambda,
                "gamma": self.gamma,
                "min_child_weight": self.min_child_weight,
                "random_state": self.random_state,
            },
            "base_score": self.base_score,
            "feature_names_": self.feature_names_,
            "n_features_": self.n_features_,
            "feature_importance_raw": self._feature_importance_raw.tolist(),
            "trees": [t.root.to_dict() for t in self.trees],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    def load_model(self, path: str):
        with open(path) as f:
            payload = json.load(f)
        p = payload["params"]
        self.n_estimators = p["n_estimators"]
        self.max_depth = p["max_depth"]
        self.learning_rate = p["learning_rate"]
        self.subsample = p["subsample"]
        self.colsample_bytree = p["colsample_bytree"]
        self.scale_pos_weight = p["scale_pos_weight"]
        self.reg_lambda = p["reg_lambda"]
        self.gamma = p["gamma"]
        self.min_child_weight = p["min_child_weight"]
        self.random_state = p["random_state"]

        self.base_score = payload["base_score"]
        self.feature_names_ = payload["feature_names_"]
        self.n_features_ = payload["n_features_"]
        self._feature_importance_raw = np.array(payload["feature_importance_raw"])

        self.trees = []
        for tree_dict in payload["trees"]:
            tree = _RegressionTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
            tree.root = _TreeNode.from_dict(tree_dict)
            self.trees.append(tree)
        return self
