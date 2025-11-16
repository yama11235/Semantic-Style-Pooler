"""Pooling utilities for sentence embeddings."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Pooler"]


class Pooler(nn.Module):
    """Parameter-free poolers to get the sentence embedding."""

    def __init__(self, pooler_type: str):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
            "last",
            "max",
        ], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask: torch.Tensor, outputs, target_layer: int = -1):
        if isinstance(outputs, torch.Tensor):
            last_hidden = outputs
            hidden_states = (outputs,)
        else:
            last_hidden = outputs.last_hidden_state
            hidden_states = outputs.hidden_states or (last_hidden,)

        dtype = last_hidden.dtype

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        if self.pooler_type == "avg":
            return (
                (hidden_states[target_layer] * attention_mask.unsqueeze(-1)).sum(1)
                / attention_mask.sum(-1).unsqueeze(-1)
            ).to(dtype)
        if self.pooler_type == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked_hidden = hidden_states[target_layer].masked_fill(
                ~mask, torch.finfo(hidden_states[target_layer].dtype).min
            )
            pooled_result, _ = masked_hidden.max(dim=1)
            return pooled_result.to(dtype)
        if self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result.to(dtype)
        if self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result.to(dtype)
        if self.pooler_type == "last":
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
            return hidden_states[target_layer][batch_idx, lengths].to(dtype)

        raise NotImplementedError
