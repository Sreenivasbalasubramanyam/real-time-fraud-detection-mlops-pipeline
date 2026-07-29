"""
Classification metrics implemented from scratch in numpy.

WHY THIS FILE EXISTS: `train_model.py` is written against scikit-learn's
`sklearn.metrics` (roc_auc_score, average_precision_score, precision_score,
recall_score, f1_score, precision_recall_curve, train_test_split). In this
sandbox, outbound network access to PyPI was blocked so `pip install
scikit-learn` could not run (see README "What Actually Ran"). These are
standard, well-defined formulas, reimplemented here directly so the training
script has no non-stdlib/non-numpy/non-pandas dependency. `train_model.py`
prefers real scikit-learn when importable and falls back to this module
otherwise — see the try/except import there.
"""

from __future__ import annotations

import numpy as np


def train_test_split(X, y, test_size: float = 0.25, random_state: int = 42, stratify=None):
    """Stratified (if `stratify` given) train/test split, pandas- or numpy-friendly."""
    n = len(y)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)

    if stratify is not None:
        strat = stratify.to_numpy() if hasattr(stratify, "to_numpy") else np.asarray(stratify)
        test_idx_parts = []
        train_idx_parts = []
        for cls in np.unique(strat):
            cls_idx = idx[strat == cls]
            rng.shuffle(cls_idx)
            n_test = int(round(len(cls_idx) * test_size))
            test_idx_parts.append(cls_idx[:n_test])
            train_idx_parts.append(cls_idx[n_test:])
        test_idx = np.concatenate(test_idx_parts)
        train_idx = np.concatenate(train_idx_parts)
        rng.shuffle(test_idx)
        rng.shuffle(train_idx)
    else:
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_test = int(round(n * test_size))
        test_idx = shuffled[:n_test]
        train_idx = shuffled[n_test:]

    def _select(obj, indices):
        if hasattr(obj, "iloc"):
            return obj.iloc[indices].reset_index(drop=True)
        return np.asarray(obj)[indices]

    return _select(X, train_idx), _select(X, test_idx), _select(y, train_idx), _select(y, test_idx)


def roc_auc_score(y_true, y_score) -> float:
    """AUC via the Mann-Whitney U statistic (rank-based), equivalent to sklearn's."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty(len(y_score))
    sorted_scores = y_score[order]
    # average ranks for ties
    i = 0
    rank = 1
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def precision_recall_curve(y_true, y_score):
    """Precision/recall at every distinct score threshold, in
    non-decreasing-recall order (equivalently, decreasing-threshold order).

    Returns (precision, recall, thresholds) where precision[i]/recall[i] is
    the precision/recall obtained by predicting "positive" for every sample
    with score >= thresholds[i]. `thresholds` is strictly increasing recall
    order's corresponding score cutoffs (same length as precision/recall —
    no extra boundary point is appended, keeping the three arrays
    same-length and unambiguous for downstream threshold selection).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)  # descending score => as we walk forward, recall only grows
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    n_pos = y_true.sum()

    # Keep only the last cumulative row for each distinct score value (ties
    # at the same score share one threshold / one (precision, recall) point).
    distinct_idx = np.where(np.diff(y_score_sorted, append=np.nan) != 0)[0]

    tps = tps[distinct_idx]
    fps = fps[distinct_idx]
    thresholds = y_score_sorted[distinct_idx]

    precision = np.divide(tps, tps + fps, out=np.ones_like(tps, dtype=float), where=(tps + fps) > 0)
    recall = tps / n_pos if n_pos > 0 else np.zeros_like(tps, dtype=float)

    # thresholds is currently in descending order (matching descending score);
    # recall is non-decreasing in this same order. Both are already internally
    # consistent (index i always refers to the same threshold across all
    # three arrays), which is all downstream callers rely on.
    return precision, recall, thresholds


def average_precision_score(y_true, y_score) -> float:
    """Area under the precision-recall curve via the step-function (AP) formula:
        AP = sum_n (R_n - R_{n-1}) * P_n
    walked in increasing-recall order, with R_0 := 0.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # precision/recall are indexed in non-decreasing-recall order already.
    ap = np.sum(np.diff(recall, prepend=0.0) * precision)
    return float(ap)


def precision_score(y_true, y_pred, zero_division: float = 0.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    if tp + fp == 0:
        return float(zero_division)
    return float(tp / (tp + fp))


def recall_score(y_true, y_pred, zero_division: float = 0.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp + fn == 0:
        return float(zero_division)
    return float(tp / (tp + fn))


def f1_score(y_true, y_pred, zero_division: float = 0.0) -> float:
    p = precision_score(y_true, y_pred, zero_division=zero_division)
    r = recall_score(y_true, y_pred, zero_division=zero_division)
    if p + r == 0:
        return float(zero_division)
    return float(2 * p * r / (p + r))
