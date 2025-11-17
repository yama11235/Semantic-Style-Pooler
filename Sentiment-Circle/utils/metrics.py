"""
Metrics module - backward compatibility wrapper.

This module maintains backward compatibility with the original metrics.py.
The actual implementation has been refactored into sub-modules.
"""
from .metrics import compute_metrics, OBJECTIVE_HANDLERS
from .metrics.base import HeadContext, flatten_floats, to_float_array, build_head_vector, select_centroids
from .metrics.regression import compute_regression_metrics
from .metrics.classification import compute_binary_metrics, compute_infonce_metrics, find_best_threshold
from .metrics.contrastive import compute_contrastive_metrics

# Expose all functions for backward compatibility
__all__ = [
    "compute_metrics",
    "OBJECTIVE_HANDLERS",
    "HeadContext",
    "flatten_floats",
    "to_float_array", 
    "build_head_vector",
    "select_centroids",
    "compute_regression_metrics",
    "compute_binary_metrics",
    "compute_infonce_metrics",
    "compute_contrastive_metrics",
    "find_best_threshold",
]
