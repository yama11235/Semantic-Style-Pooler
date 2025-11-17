"""
Backward compatibility wrapper for head_objectives.

This module has been moved to utils.training.objectives.
"""
from utils.training.objectives import (
    HeadObjective,
    InfoNCEObjective,
    AngleNCEObjective,
    RegressionObjective,
    BinaryClassificationObjective,
    ContrastiveLogitObjective,
)

__all__ = [
    "HeadObjective",
    "InfoNCEObjective",
    "AngleNCEObjective",
    "RegressionObjective",
    "BinaryClassificationObjective",
    "ContrastiveLogitObjective",
]
