"""
Loss function computation - refactored version.

This module maintains backward compatibility with the original loss_function.py.
The implementation has been refactored into sub-modules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from utils.training.loss_helpers import (
    prepare_head_batch,
    head_mask,
    finite_mask,
    apply_valid_to_mask,
    accumulate_loss,
)
from utils.training.info_nce_loss import compute_info_nce_loss

TensorDict = Dict[str, torch.Tensor]
LossResult = Tuple[Optional[torch.Tensor], TensorDict]


def compute_single_loss(
    trainer: Any,
    model: torch.nn.Module,
    inputs: Optional[Dict[str, Any]],
    active_heads: List[str],
    device: torch.device,
) -> LossResult:
    """Compute loss for single sentence tasks."""
    if not inputs:
        return None, {}

    outputs = model(**inputs)
    loss: Optional[torch.Tensor] = None
    labels = inputs.get("labels")
    batch_active_heads = inputs.get("active_heads")

    for head in active_heads:
        objective = trainer.head_objectives.get(head)
        assert objective is not None, f"Objective for head '{head}' is not defined."
        head_output = outputs.get(head)
        if head_output is None:
            raise KeyError(f"Model outputs missing head '{head}'")
        prepared = prepare_head_batch(batch_active_heads, head, labels, device, trainer.head2idx)
        if prepared is None:
            continue
        mask, valid, subset_labels = prepared
        filtered_embeddings = head_output[mask]
        filtered_embeddings = filtered_embeddings[valid]
        filtered_labels = subset_labels[valid]
        head_loss = objective.compute_single(trainer, filtered_embeddings, filtered_labels)
        loss = accumulate_loss(loss, head_loss)

    return loss, outputs


def compute_pair_loss(
    trainer: Any,
    model: torch.nn.Module,
    inputs: Optional[Dict[str, Any]],
    active_heads: List[str],
    device: torch.device,
) -> LossResult:
    """Compute loss for sentence pair tasks."""
    if not inputs:
        return None, {}

    outputs = model(**inputs)
    loss: Optional[torch.Tensor] = None
    labels = inputs.get("labels")
    batch_active_heads = inputs.get("active_heads")

    for head in active_heads:
        objective = trainer.head_objectives.get(head)
        assert objective is not None, f"Objective for head '{head}' is not defined."
        tensor = outputs.get(head)
        if tensor is None:
            raise KeyError(f"Model outputs missing head '{head}'")
        prepared = prepare_head_batch(batch_active_heads, head, labels, device, trainer.head2idx)
        if prepared is None:
            continue
        mask, valid, subset_labels = prepared
        filtered_output = tensor[mask]
        filtered_output = filtered_output[valid]
        filtered_labels = subset_labels[valid]
        head_loss = objective.compute_pair(filtered_output, filtered_labels)
        loss = accumulate_loss(loss, head_loss)

    return loss, outputs


def compute_triplet_loss(
    trainer: Any,
    model: torch.nn.Module,
    inputs: Optional[Dict[str, Any]],
    active_heads: List[str],
    device: torch.device,
) -> LossResult:
    """Compute loss for triplet tasks."""
    if not inputs:
        return None, {}

    outputs = model(**inputs)
    loss: Optional[torch.Tensor] = None
    labels = inputs.get("labels")
    batch_active_heads = inputs.get("active_heads")

    for head in active_heads:
        objective = trainer.head_objectives.get(head)
        assert objective is not None, f"Objective for head '{head}' is not defined."
        prepared = prepare_head_batch(batch_active_heads, head, labels, device, trainer.head2idx)
        if prepared is None:
            continue
        mask, valid, subset_labels = prepared
        refined_mask = apply_valid_to_mask(mask, valid)
        filtered_labels = subset_labels[valid]
        head_loss = objective.compute_triplet(trainer, outputs, refined_mask, filtered_labels)
        loss = accumulate_loss(loss, head_loss)

    return loss, outputs


def compute_pair_correlation_penalty(
    trainer: Any,
    outputs: TensorDict,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute correlation penalty for sentence pair outputs."""
    if not outputs:
        return None

    penalty: Optional[torch.Tensor] = None
    for hi, row in trainer.corr_labels.items():
        for hj, target_corr in row.items():
            if hi not in outputs or hj not in outputs:
                continue
            xi = outputs[hi].float()
            xj = outputs[hj].float()
            if xi.numel() <= 1 or xj.numel() <= 1:
                continue
            corrmat = torch.corrcoef(torch.stack([xi, xj], dim=0))
            rho = corrmat[0, 1]
            weights = trainer.corr_weights.get(hi, {}).get(hj, 1.0)
            tgt = torch.tensor(target_corr, device=device, dtype=rho.dtype)
            penalty = accumulate_loss(penalty, weights * F.mse_loss(rho, tgt))

    return penalty


def compute_triplet_correlation_penalty(
    trainer: Any,
    outputs: TensorDict,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute correlation penalty for triplet outputs."""
    if not outputs:
        return None

    penalty: Optional[torch.Tensor] = None
    for hi, row in trainer.corr_labels.items():
        for hj, target_corr in row.items():
            pos_i = outputs.get(f"{hi}_pos_similarity")
            neg_i = outputs.get(f"{hi}_neg_similarity")
            pos_j = outputs.get(f"{hj}_pos_similarity")
            neg_j = outputs.get(f"{hj}_neg_similarity")
            if pos_i is None or neg_i is None or pos_j is None or neg_j is None:
                continue
            vi = torch.cat([pos_i, neg_i], dim=0).float()
            vj = torch.cat([pos_j, neg_j], dim=0).float()
            if vi.numel() <= 1 or vj.numel() <= 1:
                continue
            corrmat = torch.corrcoef(torch.stack([vi, vj], dim=0))
            rho = corrmat[0, 1]
            weights = trainer.corr_weights.get(hi, {}).get(hj, 1.0)
            tgt = torch.tensor(target_corr, device=device, dtype=rho.dtype)
            penalty = accumulate_loss(penalty, weights * F.mse_loss(rho, tgt))

    return penalty


def fill_missing_output_keys(
    trainer: Any,
    outputs: TensorDict,
    batch_size: int,
    device: torch.device,
) -> None:
    """Fill missing keys in model outputs with NaN tensors."""
    for head, cfg in trainer.classifier_configs.items():
        if head not in outputs:
            outputs[head] = torch.full((batch_size,), float("nan"), device=device)

        for suffix in ("pos_similarity", "neg_similarity"):
            key = f"{head}_{suffix}"
            if key not in outputs:
                outputs[key] = torch.full((batch_size,), float("nan"), device=device)

        if cfg.get("objective") == "contrastive_logit":
            for suffix in ("anchor_prob", "positive_prob", "negative_prob"):
                key = f"{head}_{suffix}"
                if key not in outputs:
                    nclass = cfg.get("output_dim")
                    if nclass is None:
                        raise ValueError(f"Missing num_labels in config for {head}")
                    shape = (batch_size, nclass)
                    outputs[key] = torch.full(
                        shape,
                        float("nan"),
                        dtype=torch.float32,
                        device=device,
                    )

        for suffix in ("_pos_similarity", "_neg_similarity", ""):
            key = f"original{suffix}"
            if key not in outputs:
                outputs[key] = torch.full((batch_size,), float("nan"), device=device)


__all__ = [
    "compute_single_loss",
    "compute_pair_loss",
    "compute_triplet_loss",
    "compute_pair_correlation_penalty",
    "compute_triplet_correlation_penalty",
    "fill_missing_output_keys",
    "prepare_head_batch",
    "head_mask",
    "finite_mask",
    "apply_valid_to_mask",
    "accumulate_loss",
    "compute_info_nce_loss",
]
