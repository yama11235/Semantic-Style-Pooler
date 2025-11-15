from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from loss_function import compute_info_nce_loss


class HeadObjective(ABC):
    """Base class providing a uniform interface for head-specific losses."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config

    # ---- Single sentence objectives -------------------------------------------------
    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None

    # ---- Pair objectives ------------------------------------------------------------
    def compute_pair(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None

    # ---- Triplet objectives ---------------------------------------------------------
    def compute_triplet(
        self,
        trainer: Any,
        outputs: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None


class InfoNCEObjective(HeadObjective):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.tau = float(config.get("tau", 1.0))
        self.max_pos_pairs = int(config.get("inbatch_positive_pairs", 0))
        self.max_neg_pairs = int(config.get("inbatch_negative_pairs", 0))

    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if embeddings.numel() == 0:
            return None
        tau = max(self.tau, 1e-6)
        return compute_info_nce_loss(
            embeddings=embeddings,
            labels=labels.long(),
            tau=tau,
            max_pos_pairs=max(self.max_pos_pairs, 0),
            max_neg_pairs=max(self.max_neg_pairs, 0),
        )


class AngleNCEObjective(InfoNCEObjective):
    """Placeholder for AngleNCE (kept for completeness)."""

    def compute_single(
        self,
        trainer: Any,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # AngleNCE is not implemented in the original code either.
        return None


class RegressionObjective(HeadObjective):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.loss_fn = nn.MSELoss()

    def compute_pair(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if outputs.numel() == 0:
            return None
        return self.loss_fn(outputs.float(), labels.float())


class BinaryClassificationObjective(HeadObjective):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        pos_weight = config.get("pos_weight")
        if pos_weight is not None:
            weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_pair(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if outputs.numel() == 0:
            return None
        outputs = outputs.float().view_as(labels.float())
        return self.loss_fn(outputs, labels.float())


class ContrastiveLogitObjective(HeadObjective):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.alpha = float(config.get("alpha", 1.0))
        self.gamma = float(config.get("gamma", 1.0))
        self.margin = float(config.get("margin", 0.0))
        self.loss_type = config.get("loss_type", "triplet")
        self.tau = float(config.get("tau", 1.0))

    def compute_triplet(
        self,
        trainer: Any,
        outputs: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if mask.numel() == 0 or mask.sum() == 0:
            return None

        prob_key = f"{self.name}_anchor_prob"
        pos_key = f"{self.name}_pos_similarity"
        neg_key = f"{self.name}_neg_similarity"

        if prob_key not in outputs or pos_key not in outputs or neg_key not in outputs:
            return None

        logits = outputs[prob_key][mask].to(torch.float32)
        targets = labels.to(logits.device, dtype=torch.long)
        if logits.numel() == 0:
            return None

        total_loss = self.gamma * F.cross_entropy(logits, targets)

        pos = outputs[pos_key][mask]
        neg = outputs[neg_key][mask]
        if pos is None or neg is None or pos.numel() == 0 or neg.numel() == 0:
            return total_loss

        if self.loss_type == "infoNCE":
            pos = pos.view(-1, 1)
            neg = neg.view(pos.size(0), -1)
            logits = torch.cat([pos, neg], dim=1) / max(self.tau, 1e-6)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            total_loss = total_loss + self.alpha * F.cross_entropy(logits, labels)
        else:
            if self.margin > 0:
                triplet = F.relu(neg - pos + self.margin).mean()
            else:
                triplet = F.relu(pos - neg - self.margin).mean()
            total_loss = total_loss + self.alpha * triplet

        return total_loss


__all__ = [
    "HeadObjective",
    "InfoNCEObjective",
    "AngleNCEObjective",
    "RegressionObjective",
    "BinaryClassificationObjective",
    "ContrastiveLogitObjective",
]
