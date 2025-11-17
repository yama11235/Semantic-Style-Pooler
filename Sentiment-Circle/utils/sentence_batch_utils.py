"""
Backward compatibility wrapper for sentence_batch_utils.

This module has been moved to utils.data.batch_utils.
"""
from utils.data.batch_utils import (
    extract_unique_strings,
    flatten_strings,
    compute_sentence_partitions,
    BatchPartitioner,
    tokenize_optional_sentences,
)

__all__ = [
    "extract_unique_strings",
    "flatten_strings",
    "compute_sentence_partitions",
    "BatchPartitioner",
    "tokenize_optional_sentences",
]
