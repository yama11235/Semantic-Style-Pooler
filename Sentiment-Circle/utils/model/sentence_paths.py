"""Utilities for handling different sentence path configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from sentence_batch_utils import BatchPartitioner

if TYPE_CHECKING:  # pragma: no cover
    from modeling_encoders import BiEncoderForClassification

__all__ = [
    "SentencePath",
    "SingleSentencePath",
    "PairSentencePath",
    "TripletSentencePath",
]


def _sentence_kwargs(batch: Dict[str, torch.Tensor], suffix: str = "") -> Dict[str, Optional[torch.Tensor]]:
    keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "inputs_embeds",
    ]
    payload: Dict[str, Optional[torch.Tensor]] = {}
    for key in keys:
        name = key if not suffix else f"{key}_{suffix}"
        payload[key] = batch.get(name)
    return payload


class SentencePath:
    kind = ""

    def __init__(self, model: "BiEncoderForClassification") -> None:
        self.model = model

    def run_full(self, batch: Dict[str, torch.Tensor], extra_kwargs: Dict[str, object]) -> Dict[str, torch.Tensor]:
        return self._forward(batch, extra_kwargs)

    def run_partition(
        self,
        batch: Dict[str, torch.Tensor],
        partitioner: BatchPartitioner,
        device: torch.device,
        extra_kwargs: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        subset = partitioner.slice(batch, self.kind, device=device)
        if not subset:
            return {}
        return self._forward(subset, extra_kwargs)

    def _forward(self, batch: Dict[str, torch.Tensor], extra_kwargs: Dict[str, object]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class SingleSentencePath(SentencePath):
    kind = "single"

    def _forward(self, batch: Dict[str, torch.Tensor], extra_kwargs: Dict[str, object]) -> Dict[str, torch.Tensor]:
        args = _sentence_kwargs(batch)
        return self.model.encode(**args)


class PairSentencePath(SentencePath):
    kind = "pair"

    def _forward(self, batch: Dict[str, torch.Tensor], extra_kwargs: Dict[str, object]) -> Dict[str, torch.Tensor]:
        first = _sentence_kwargs(batch)
        second = _sentence_kwargs(batch, "2")
        return self.model._forward_binary(
            first["input_ids"],
            first["attention_mask"],
            first["token_type_ids"],
            first["position_ids"],
            first["head_mask"],
            first["inputs_embeds"],
            second["input_ids"],
            second["attention_mask"],
            second["token_type_ids"],
            second["position_ids"],
            second["head_mask"],
            second["inputs_embeds"],
            **extra_kwargs,
        )


class TripletSentencePath(SentencePath):
    kind = "triplet"

    def _forward(self, batch: Dict[str, torch.Tensor], extra_kwargs: Dict[str, object]) -> Dict[str, torch.Tensor]:
        first = _sentence_kwargs(batch)
        second = _sentence_kwargs(batch, "2")
        third = _sentence_kwargs(batch, "3")
        return self.model.triplet_encode(
            first["input_ids"],
            first["attention_mask"],
            first["token_type_ids"],
            first["position_ids"],
            first["head_mask"],
            first["inputs_embeds"],
            second["input_ids"],
            second["attention_mask"],
            second["token_type_ids"],
            second["position_ids"],
            second["head_mask"],
            second["inputs_embeds"],
            third["input_ids"],
            third["attention_mask"],
            third["token_type_ids"],
            third["position_ids"],
            third["head_mask"],
            third["inputs_embeds"],
            **extra_kwargs,
        )
