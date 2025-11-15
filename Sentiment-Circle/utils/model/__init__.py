"""Model subpackage that hosts encoder architectures and classifier utilities."""

from __future__ import annotations

from .pooler import Pooler
from .sentence_paths import (
    SentencePath,
    SingleSentencePath,
    PairSentencePath,
    TripletSentencePath,
)
from .classifier_strategies import (
    calculate_similarity,
    calculate_pos_neg_similarity,
    _ClassifierStrategy,
    _DefaultClassifierStrategy,
    _ContrastiveLogitStrategy,
)

__all__ = [
    "Pooler",
    "SentencePath",
    "SingleSentencePath",
    "PairSentencePath",
    "TripletSentencePath",
    "calculate_similarity",
    "calculate_pos_neg_similarity",
    "_ClassifierStrategy",
    "_DefaultClassifierStrategy",
    "_ContrastiveLogitStrategy",
]
