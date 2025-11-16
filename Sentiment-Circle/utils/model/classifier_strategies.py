"""Strategies and helpers for classifier heads."""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from modeling_encoders import BiEncoderForClassification

__all__ = [
    "calculate_similarity",
    "calculate_pos_neg_similarity",
    "_ClassifierStrategy",
    "_DefaultClassifierStrategy",
    "_ContrastiveLogitStrategy",
]


def calculate_similarity(name: str, output1: torch.Tensor, output2: torch.Tensor, classifier_configs: Dict[str, dict]) -> torch.Tensor:
    """Calculate similarity (or distance) between two tensors based on classifier config."""
    config = classifier_configs[name]
    distance_type = config["distance"]
    objective = config.get("objective", None)

    if distance_type == "cosine":
        similarity = torch.nn.functional.cosine_similarity(output1, output2, dim=1).to(output1.dtype)
        if objective == "binary_classification":
            similarity = torch.sigmoid(similarity).to(output1.dtype)
    elif distance_type == "euclidean":
        similarity = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
    elif distance_type == "dot_product":
        similarity = torch.sum(output1 * output2, dim=1)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")

    return similarity


def calculate_pos_neg_similarity(
    name: str,
    output1: torch.Tensor,
    output2: torch.Tensor,
    output3: torch.Tensor,
    classifier_configs: Dict[str, dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    config = classifier_configs[name]
    dist = config["distance"]

    if dist == "cosine":
        pos_sim = F.cosine_similarity(output1, output2, dim=1).to(output1.dtype)
        neg_sim = F.cosine_similarity(output1, output3, dim=1).to(output1.dtype)
    elif dist == "euclidean":
        pos_sim = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        neg_sim = torch.sqrt(torch.sum((output1 - output3) ** 2, dim=1))
    elif dist == "dot_product":
        pos_sim = torch.sum(output1 * output2, dim=1).to(output1.dtype)
        neg_sim = torch.sum(output1 * output3, dim=1).to(output1.dtype)
    else:
        raise ValueError(f"Unknown distance type: {dist}")

    return pos_sim, neg_sim


class _ClassifierStrategy:
    def single(
        self,
        model: "BiEncoderForClassification",
        name: str,
        classifier: nn.Module,
        features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {}

    def pair(
        self,
        model: "BiEncoderForClassification",
        name: str,
        classifier: nn.Module,
        features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {}

    def triplet(
        self,
        model: "BiEncoderForClassification",
        name: str,
        classifier: nn.Module,
        features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {}

    def classify(
        self,
        model: "BiEncoderForClassification",
        name: str,
        classifier: nn.Module,
        features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {}


class _DefaultClassifierStrategy(_ClassifierStrategy):
    def single(self, model, name, classifier, features):
        seq, mask = features[0]
        embedding = classifier.encode(seq, mask).to(seq.dtype)
        return {name: embedding}

    def pair(self, model, name, classifier, features):
        seq1, mask1 = features[0]
        seq2, mask2 = features[1]
        output1 = classifier(seq1, mask1)
        output2 = classifier(seq2, mask2)
        similarity = calculate_similarity(name, output1, output2, model.classifier_configs)
        return {name: similarity}

    def triplet(self, model, name, classifier, features):
        seq1, mask1 = features[0]
        seq2, mask2 = features[1]
        seq3, mask3 = features[2]
        output1 = classifier(seq1, mask1)
        output2 = classifier(seq2, mask2)
        output3 = classifier(seq3, mask3)
        pos_similarity, neg_similarity = calculate_pos_neg_similarity(
            name, output1, output2, output3, model.classifier_configs
        )
        return {
            f"{name}_pos_similarity": pos_similarity,
            f"{name}_neg_similarity": neg_similarity,
        }


class _ContrastiveLogitStrategy(_ClassifierStrategy):
    def single(self, model, name, classifier, features):
        seq, mask = features[0]
        embedding = classifier.encode(seq, mask).to(seq.dtype)
        return {name: embedding}

    def pair(self, model, name, classifier, features):
        seq1, mask1 = features[0]
        seq2, mask2 = features[1]
        output1 = classifier.encode(seq1, mask1)
        output2 = classifier.encode(seq2, mask2)
        similarity = calculate_similarity(name, output1, output2, model.classifier_configs)
        return {name: similarity}

    def triplet(self, model, name, classifier, features):
        seq1, mask1 = features[0]
        seq2, mask2 = features[1]
        seq3, mask3 = features[2]
        output1, prob1 = classifier(seq1, mask1)
        output2, prob2 = classifier(seq2, mask2)
        output3, prob3 = classifier(seq3, mask3)
        pos_similarity, neg_similarity = calculate_pos_neg_similarity(
            name, output1, output2, output3, model.classifier_configs
        )
        return {
            f"{name}_pos_similarity": pos_similarity,
            f"{name}_neg_similarity": neg_similarity,
            f"{name}_anchor_prob": prob1,
            f"{name}_positive_prob": prob2,
            f"{name}_negative_prob": prob3,
        }

    def classify(self, model, name, classifier, features):
        seq, mask = features[0]
        _, prob = classifier(seq, mask)
        return {f"{name}_prob": prob}
