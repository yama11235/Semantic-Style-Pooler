"""Regression metrics computation."""
from typing import Dict
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from .base import HeadContext


def compute_regression_metrics(ctx: HeadContext) -> Dict[str, float]:
    """
    Compute metrics for regression tasks.
    
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

    mse = mean_squared_error(truths, scores)
    pearson = pearsonr(scores, truths)[0] if scores.size > 1 else 0.0
    
    if scores.size > 1:
        spearman = spearmanr(scores, truths).correlation
    else:
        spearman = 0.0
    
    return {
        f"{ctx.head}_mse": float(mse),
        f"{ctx.head}_single-pearson": float(pearson),
        f"{ctx.head}_single-spearman": float(spearman),
    }
