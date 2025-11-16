from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.sentence_batch_utils import flatten_strings

TensorDict = Dict[str, torch.Tensor]
LossResult = Tuple[Optional[torch.Tensor], TensorDict]


def compute_single_loss(
    trainer: Any,
    model: torch.nn.Module,
    inputs: Optional[Dict[str, Any]],
    active_heads: List[str],
    device: torch.device,
) -> LossResult:
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


def prepare_head_batch(
    batch_active_heads: Any,
    head: str,
    labels: Optional[torch.Tensor],
    device: torch.device,
    head2idx: Dict[str, int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if labels is None:
        return None

    mask = head_mask(batch_active_heads, head, device, head2idx)
    if mask is None or mask.sum() == 0:
        return None

    subset_labels = labels[mask].flatten()
    valid = finite_mask(subset_labels)
    if valid.sum() == 0:
        return None

    return mask, valid, subset_labels


def head_mask(
    active_heads_field: Any,
    head: str,
    device: torch.device,
    head2idx: Dict[str, int],
) -> Optional[torch.Tensor]:
    if active_heads_field is None:
        return None
    if isinstance(active_heads_field, torch.Tensor):
        if active_heads_field.dtype == torch.bool:
            return active_heads_field.to(device)
        if active_heads_field.dtype in (torch.int32, torch.int64, torch.long):
            idx = head2idx.get(head)
            if idx is None:
                return None
            return (active_heads_field.to(device) == idx)
    heads = flatten_strings(active_heads_field)
    if not heads:
        return None
    mask_list = [h == head for h in heads]
    if not any(mask_list):
        return None
    return torch.tensor(mask_list, device=device, dtype=torch.bool)


def finite_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype.is_floating_point:
        return torch.isfinite(tensor)
    return torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)


def apply_valid_to_mask(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    refined_mask = mask.clone()
    refined_mask[mask] = valid
    return refined_mask


def accumulate_loss(
    current: Optional[torch.Tensor],
    new_loss: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if new_loss is None:
        return current
    if current is None:
        return new_loss
    return current + new_loss


def compute_info_nce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    max_pos_pairs: int,
    max_neg_pairs: int,
) -> Optional[torch.Tensor]:
    if embeddings.size(0) < 2:
        return None

    device = embeddings.device
    eps = 1e-8

    z = F.normalize(embeddings.float(), dim=1)
    B = z.size(0)

    lbls = labels.to(device=device, dtype=torch.long).view(-1)

    sim = z @ z.t()
    logits = sim / max(tau, eps)

    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & (~self_mask)
    neg_mask = (~pos_mask) & (~self_mask)

    if not pos_mask.any():
        return None

    if (max_pos_pairs is None or max_pos_pairs <= 0) and (max_neg_pairs is None or max_neg_pairs <= 0):
        exp_logits = torch.exp(logits) * (~self_mask)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
        numer = (exp_logits * pos_mask).sum(dim=1, keepdim=True)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return None
        log_prob = torch.log(numer.clamp_min(eps)) - torch.log(denom)
        return -(log_prob.squeeze(1)[valid]).mean()

    losses: List[torch.Tensor] = []
    for i in range(B):
        pos_idx = torch.nonzero(pos_mask[i], as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue
        neg_idx = torch.nonzero(neg_mask[i], as_tuple=True)[0]

        if max_pos_pairs is not None and max_pos_pairs > 0 and pos_idx.numel() > max_pos_pairs:
            perm = torch.randperm(pos_idx.numel(), device=device)[:max_pos_pairs]
            pos_idx = pos_idx[perm]
        if max_neg_pairs is not None and max_neg_pairs > 0 and neg_idx.numel() > max_neg_pairs:
            perm = torch.randperm(neg_idx.numel(), device=device)[:max_neg_pairs]
            neg_idx = neg_idx[perm]

        sel_idx = torch.cat([pos_idx, neg_idx], dim=0) if neg_idx.numel() > 0 else pos_idx
        row = logits[i, sel_idx]
        exp_row = torch.exp(row)

        numer = exp_row[:pos_idx.numel()].sum()
        denom = exp_row.sum().clamp_min(eps)

        losses.append(-(torch.log(numer.clamp_min(eps)) - torch.log(denom)))

    if not losses:
        return None

    return torch.stack(losses).mean()


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
