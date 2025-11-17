"""
Backward compatibility wrapper for dataset_preprocessing.

This module has been moved to utils.data.preprocessing.
"""
from utils.data.preprocessing import (
    scale_to_range,
    parse_dict,
    get_preprocessing_function,
    batch_get_preprocessing_function,
)

__all__ = [
    "scale_to_range",
    "parse_dict",
    "get_preprocessing_function",
    "batch_get_preprocessing_function",
]
