"""Classification metrics computation."""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    adjusted_mutual_info_score,
    v_measure_score,
)
from .base import HeadContext


def find_best_threshold(y_true, scores, n_thresholds=100) -> Tuple[float, float, float]:
    """
    Find the best classification threshold based on F1 score.
    
    Args:
        y_true: True labels
        scores: Predicted scores
        n_thresholds: Number of thresholds to test
        
    Returns:
        Tuple of (best_threshold, best_f1, best_accuracy)
    """
    y = np.array(y_true, dtype=float)
    s = np.array(scores, dtype=float)

    # Filter valid indices
    mask_valid = (~np.isnan(y)) & np.isfinite(y) & np.isfinite(s)
    if not mask_valid.any():
        return 0.5, 0.0, 0.0

    y = y[mask_valid]
    s = s[mask_valid]

    # Convert -1/1 labels to 0/1
    uniq = set(np.unique(y))
    if uniq == {-1.0, 1.0}:
        y = (y == 1.0).astype(int)
    else:
        y = y.astype(int)

    # Determine threshold range
    thr_min, thr_max = np.min(s), np.max(s)
    if thr_min == thr_max:
        return float(thr_min), 0.0, accuracy_score(y, (s >= thr_min).astype(int))

    thresholds = np.linspace(0, 1, n_thresholds)

    best_f1, best_acc, best_thr = 0.0, 0.0, thresholds[0]
    for thr in thresholds:
        y_pred = (s >= thr).astype(int)
        try:
            f1 = f1_score(y, y_pred)
        except ValueError:
            f1 = 0.0
        acc = accuracy_score(y, y_pred)
        if f1 > best_f1:
            best_f1, best_acc, best_thr = f1, acc, thr

    return float(best_thr), float(best_f1), float(best_acc)


def compute_binary_metrics(ctx: HeadContext) -> Dict[str, float]:
    """
    Compute metrics for binary classification tasks.
    
    Args:
        ctx: Head context containing scores and labels
        
    Returns:
        Dictionary of computed metrics
    """
    if ctx.scores is None or ctx.labels.size == 0:
        return {}
    
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    
    scores = ctx.scores[mask].astype(float)
    truths = ctx.labels[mask].astype(float).flatten()
    
    if scores.size == 0:
        return {}

    y_true = truths.astype(int)
    best_thr, best_f1, best_acc = find_best_threshold(y_true, scores)
    
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    
    mse = mean_squared_error(y_true, scores)
    
    return {
        f"{ctx.head}_best-threshold": float(best_thr),
        f"{ctx.head}_best-accuracy": float(best_acc),
        f"{ctx.head}_best-f1": float(best_f1),
        f"{ctx.head}_auc": float(auc),
        f"{ctx.head}_mse": float(mse),
    }


def compute_infonce_metrics(ctx: HeadContext) -> Dict[str, float]:
    """
    Compute metrics for InfoNCE (contrastive learning) tasks.
    
    Args:
        ctx: Head context containing embeddings, labels, and centroids
        
    Returns:
        Dictionary of computed metrics
    """
    if ctx.embeddings is None or ctx.centroids is None or not ctx.centroids:
        return {}
    
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    
    embeddings = ctx.embeddings
    if embeddings.ndim != 2:
        return {}
    if mask.shape[0] != embeddings.shape[0]:
        return {}
    
    emb_subset = embeddings[mask]
    if emb_subset.shape[0] == 0:
        return {}
    
    labels = ctx.labels[mask] if ctx.labels.size else np.array([], dtype=int)
    labels = np.asarray(labels)
    
    if labels.ndim > 1:
        labels = labels.reshape(labels.shape[0], -1)
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = labels.flatten()
    labels = labels.astype(int, copy=False)

    centroid_labels = np.array(list(ctx.centroids.keys()), dtype=int)
    centroid_vectors = np.asarray([ctx.centroids[label] for label in centroid_labels], dtype=float)
    
    if centroid_vectors.ndim != 2 or centroid_vectors.shape[0] == 0:
        return {}
    if centroid_vectors.shape[1] != emb_subset.shape[1]:
        return {}

    # Compute nearest centroid for each embedding
    diff = emb_subset[:, None, :] - centroid_vectors[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    y_pred = centroid_labels[nearest_idx]
    
    if labels.size == 0:
        return {}

    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
    ami = adjusted_mutual_info_score(labels, y_pred)
    v_measure = v_measure_score(labels, y_pred)
    
    return {
        f"{ctx.head}_knn_accuracy": float(acc),
        f"{ctx.head}_knn_macro_f1": float(f1),
        f"{ctx.head}_cluster_ami": float(ami),
        f"{ctx.head}_cluster_v_measure": float(v_measure),
    }
