"""Classification metrics for isolated SLR.

Everything accepts plain NumPy arrays so it is framework-agnostic; in a training
loop, detach/convert your torch tensors with ``.cpu().numpy()`` first.

Reported for the benchmark:
  * Top-1 / Top-5 accuracy            (WLASL standard)
  * mean per-class (macro) accuracy   (WLASL standard; robust to class imbalance)
  * macro + weighted precision/recall/F1
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _as_arrays(logits_or_probs, targets):
    scores = np.asarray(logits_or_probs)
    targets = np.asarray(targets).astype(np.int64)
    if scores.ndim == 1:  # already predicted labels
        return None, scores.astype(np.int64), targets
    preds = scores.argmax(axis=1).astype(np.int64)
    return scores, preds, targets


def topk_accuracy(logits, targets, ks: Sequence[int] = (1, 5)) -> dict:
    """Top-k accuracy for each k. Requires full score matrix (N, C)."""
    scores = np.asarray(logits)
    targets = np.asarray(targets).astype(np.int64)
    if scores.ndim != 2:
        raise ValueError("topk_accuracy needs a (N, C) score matrix.")
    n, c = scores.shape
    out = {}
    # argsort descending once; slice per k.
    order = np.argsort(-scores, axis=1)
    for k in ks:
        kk = min(k, c)
        topk = order[:, :kk]
        correct = (topk == targets[:, None]).any(axis=1)
        out[f"top{k}"] = float(correct.mean()) if n else 0.0
    return out


def mean_per_class_accuracy(preds, targets, num_classes: Optional[int] = None) -> float:
    """Mean of per-class recall (a.k.a. balanced accuracy / WLASL mean-class acc)."""
    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    classes = (
        np.arange(num_classes)
        if num_classes is not None
        else np.unique(targets)
    )
    accs = []
    for cls in classes:
        mask = targets == cls
        if mask.sum() == 0:
            continue  # class absent from this eval set
        accs.append((preds[mask] == cls).mean())
    return float(np.mean(accs)) if accs else 0.0


def precision_recall_f1(preds, targets, num_classes: Optional[int] = None) -> dict:
    """Macro and weighted precision/recall/F1 via scikit-learn."""
    from sklearn.metrics import precision_recall_fscore_support

    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    labels = list(range(num_classes)) if num_classes is not None else None

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        targets, preds, average="macro", labels=labels, zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", labels=labels, zero_division=0
    )
    return {
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
    }


def confusion(preds, targets, num_classes: Optional[int] = None) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    labels = list(range(num_classes)) if num_classes is not None else None
    return confusion_matrix(
        np.asarray(targets), np.asarray(preds), labels=labels
    )


def compute_metrics(
    logits,
    targets,
    num_classes: Optional[int] = None,
    ks: Sequence[int] = (1, 5),
) -> dict:
    """One-call metric bundle for evaluation.

    ``logits`` may be a (N, C) score matrix (enables top-k) or a 1-D array of
    predicted labels (top-k skipped). Returns a flat dict ready for logging.
    """
    scores, preds, targets = _as_arrays(logits, targets)
    if num_classes is None and scores is not None:
        num_classes = scores.shape[1]

    out: dict = {}
    if scores is not None:
        out.update(topk_accuracy(scores, targets, ks))
    else:
        out["top1"] = float((preds == targets).mean()) if len(targets) else 0.0

    out["mean_class_acc"] = mean_per_class_accuracy(preds, targets, num_classes)
    out.update(precision_recall_f1(preds, targets, num_classes))
    return out
