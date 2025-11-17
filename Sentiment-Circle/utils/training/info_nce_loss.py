"""InfoNCE loss computation."""
from typing import Optional, List
import torch
import torch.nn.functional as F


def compute_info_nce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    max_pos_pairs: int,
    max_neg_pairs: int,
) -> Optional[torch.Tensor]:
    """
    Compute InfoNCE (contrastive) loss.
    
    Args:
        embeddings: Embedding vectors [B, D]
        labels: Labels [B]
        tau: Temperature parameter
        max_pos_pairs: Maximum positive pairs per sample
        max_neg_pairs: Maximum negative pairs per sample
        
    Returns:
        Loss tensor or None
    """
    if embeddings.size(0) < 2:
        return None

    device = embeddings.device
    eps = 1e-8

    # Normalize embeddings
    z = F.normalize(embeddings.float(), dim=1)
    B = z.size(0)

    lbls = labels.to(device=device, dtype=torch.long).view(-1)

    # Compute similarity matrix
    sim = z @ z.t()
    logits = sim / max(tau, eps)

    # Create masks
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & (~self_mask)
    neg_mask = (~pos_mask) & (~self_mask)

    if not pos_mask.any():
        return None

    # Full batch loss (if no pair limits)
    if (max_pos_pairs is None or max_pos_pairs <= 0) and (max_neg_pairs is None or max_neg_pairs <= 0):
        exp_logits = torch.exp(logits) * (~self_mask)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
        numer = (exp_logits * pos_mask).sum(dim=1, keepdim=True)
        valid = pos_mask.any(dim=1)
        if not valid.any():
            return None
        log_prob = torch.log(numer.clamp_min(eps)) - torch.log(denom)
        return -(log_prob.squeeze(1)[valid]).mean()

    # Sample-wise loss with pair limits
    losses: List[torch.Tensor] = []
    for i in range(B):
        pos_idx = torch.nonzero(pos_mask[i], as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue
        neg_idx = torch.nonzero(neg_mask[i], as_tuple=True)[0]

        # Sample positive pairs
        if max_pos_pairs is not None and max_pos_pairs > 0 and pos_idx.numel() > max_pos_pairs:
            perm = torch.randperm(pos_idx.numel(), device=device)[:max_pos_pairs]
            pos_idx = pos_idx[perm]
        
        # Sample negative pairs
        if max_neg_pairs is not None and max_neg_pairs > 0 and neg_idx.numel() > max_neg_pairs:
            perm = torch.randperm(neg_idx.numel(), device=device)[:max_neg_pairs]
            neg_idx = neg_idx[perm]

        # Compute loss for this sample
        sel_idx = torch.cat([pos_idx, neg_idx], dim=0) if neg_idx.numel() > 0 else pos_idx
        row = logits[i, sel_idx]
        exp_row = torch.exp(row)

        numer = exp_row[:pos_idx.numel()].sum()
        denom = exp_row.sum().clamp_min(eps)

        losses.append(-(torch.log(numer.clamp_min(eps)) - torch.log(denom)))

    if not losses:
        return None

    return torch.stack(losses).mean()
