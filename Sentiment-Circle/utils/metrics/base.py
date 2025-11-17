"""Base utilities and data structures for metrics computation."""
from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional
import numpy as np
import math


@dataclass
class HeadContext:
    """Context data for computing metrics for a specific head."""
    head: str
    objective: str
    scores: Optional[np.ndarray] = None
    pos_scores: Optional[np.ndarray] = None
    neg_scores: Optional[np.ndarray] = None
    anchor_probs: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    active_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    centroids: Optional[Dict[int, np.ndarray]] = None


def flatten_floats(data: Union[List[float], List[List[float]]]) -> List[float]:
    """
    Flatten nested float lists and remove NaN values.
    
    Args:
        data: List of floats or list of lists of floats
        
    Returns:
        Flattened list of valid floats
    """
    if not data:
        return []
    
    if all(isinstance(elem, float) and not math.isnan(elem) for elem in data):
        return data  # type: ignore

    flattened: List[float] = []
    for elem in data:
        if isinstance(elem, list):
            for x in elem:
                if isinstance(x, float) and not math.isnan(x):
                    flattened.append(x)
        else:
            if isinstance(elem, float) and not math.isnan(elem):
                flattened.append(elem)

    return flattened


def to_float_array(value: Optional[Union[np.ndarray, List[float]]]) -> Optional[np.ndarray]:
    """Convert value to float numpy array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr


def build_head_vector(*arrays: Optional[np.ndarray]) -> np.ndarray:
    """
    Build a single vector by concatenating multiple arrays.
    
    Args:
        *arrays: Variable number of optional numpy arrays
        
    Returns:
        Concatenated vector with NaN values removed
    """
    cleaned: List[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=float).flatten()
        flat = flat[~np.isnan(flat)]
        if flat.size > 0:
            cleaned.append(flat)
    if cleaned:
        return np.concatenate(cleaned, axis=0)
    return np.array([], dtype=float)


def select_centroids(
    entry: Optional[Dict],
    mode: str,
) -> Optional[Dict[int, np.ndarray]]:
    """
    Select centroids based on evaluation mode.
    
    Args:
        entry: Dictionary containing centroid data
        mode: Either "classifier" or "original"
        
    Returns:
        Dictionary mapping label IDs to centroids
    """
    if not entry:
        return None
    if isinstance(entry, dict) and any(
        isinstance(k, str) and k in ("classifier", "original") for k in entry.keys()
    ):
        primary = "original" if mode == "original" else "classifier"
        if entry.get(primary):
            return entry[primary]
        fallback = "classifier" if primary == "original" else "original"
        return entry.get(fallback)
    return entry
