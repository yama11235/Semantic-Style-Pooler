"""Helper utilities for loss computation."""
from typing import Optional, Tuple, Dict, List, Any
import torch
from utils.data.batch_utils import flatten_strings


def prepare_head_batch(
    batch_active_heads: Any,
    head: str,
    labels: Optional[torch.Tensor],
    device: torch.device,
    head2idx: Dict[str, int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Prepare batch data for a specific head.
    
    Args:
        batch_active_heads: Active heads information
        head: Target head name
        labels: Label tensor
        device: Compute device
        head2idx: Mapping from head name to index
        
    Returns:
        Tuple of (mask, valid, subset_labels) or None
    """
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
    """
    Create a mask for samples belonging to a specific head.
    
    Args:
        active_heads_field: Active heads data
        head: Target head name
        device: Compute device
        head2idx: Mapping from head name to index
        
    Returns:
        Boolean mask tensor or None
    """
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
    """
    Create mask for finite values.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Boolean mask for finite values
    """
    if tensor.dtype.is_floating_point:
        return torch.isfinite(tensor)
    return torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)


def apply_valid_to_mask(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Apply validity mask to existing mask.
    
    Args:
        mask: Original mask
        valid: Validity mask
        
    Returns:
        Refined mask
    """
    refined_mask = mask.clone()
    refined_mask[mask] = valid
    return refined_mask


def accumulate_loss(
    current: Optional[torch.Tensor],
    new_loss: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Accumulate losses.
    
    Args:
        current: Current accumulated loss
        new_loss: New loss to add
        
    Returns:
        Accumulated loss
    """
    if new_loss is None:
        return current
    if current is None:
        return new_loss
    return current + new_loss
