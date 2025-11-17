"""Contrastive learning metrics computation."""
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from .base import HeadContext


def compute_contrastive_metrics(ctx: HeadContext) -> Dict[str, float]:
    """
    Compute metrics for contrastive learning tasks (triplet-based).
    
    Args:
        ctx: Head context containing positive/negative scores and anchor probabilities
        
    Returns:
        Dictionary of computed metrics
    """
    if ctx.pos_scores is None or ctx.neg_scores is None:
        return {}
    
    mask = ctx.active_mask
    if mask.size == 0 or not mask.any():
        return {}
    
    pos_scores = ctx.pos_scores[mask].astype(float)
    neg_scores = ctx.neg_scores[mask].astype(float)
    metrics: Dict[str, float] = {}
    
    if pos_scores.size > 0 and neg_scores.size > 0:
        trip_acc = float(np.mean(pos_scores > neg_scores))
        avg_pos = float(np.mean(pos_scores))
        avg_neg = float(np.mean(neg_scores))
        metrics[f"{ctx.head}_triplet_accuracy"] = trip_acc
        metrics[f"{ctx.head}_avg_positive_similarity"] = avg_pos
        metrics[f"{ctx.head}_avg_negative_similarity"] = avg_neg

    if ctx.anchor_probs is not None and ctx.labels.size > 0:
        probs = ctx.anchor_probs[mask].astype(float)
        y_true = ctx.labels[mask].astype(int)
        if probs.ndim == 2 and y_true.size > 0:
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            metrics[f"{ctx.head}_anchor_accuracy"] = float(acc)
            metrics[f"{ctx.head}_anchor_macro_f1"] = float(f1)
    
    return metrics
