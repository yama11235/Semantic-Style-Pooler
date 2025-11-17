"""Main metrics computation module."""
from typing import Dict, Callable, Optional
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr, pearsonr
from transformers import EvalPrediction

from .base import HeadContext, to_float_array, build_head_vector, select_centroids
from .regression import compute_regression_metrics
from .classification import compute_binary_metrics, compute_infonce_metrics
from .contrastive import compute_contrastive_metrics
from ..IsoScore_functions.IsoScore import IsoScore
from ..IsoScore_functions.existing_scores import cosine_score, partition_score, varex_score, id_score


# Track if original embeddings have been processed (for one-time calculation)
_original_embeddings_processed = {}


def compute_isotropy_scores(embeddings: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Compute isotropy scores for embedding space.
    
    Args:
        embeddings: 2D array of embeddings (samples x dimensions)
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of isotropy scores
    """
    if embeddings is None or embeddings.size == 0:
        return {}
    
    if len(embeddings.shape) != 2:
        return {}
    
    if embeddings.shape[0] < 2:
        return {}
    
    scores = {}
    prefix_str = f"{prefix}_" if prefix else ""
    
    try:
        # points format: dimensions x samples (transpose of embeddings)
        points = embeddings.T
        
        # cos: 1 - cosine_score (lower cosine means higher variance)
        cos = cosine_score(points, num_samples=100000, pca_reconfig=False)
        scores[f"{prefix_str}cos"] = 1.0 - cos
        
        # part: partition score
        part = partition_score(points)
        scores[f"{prefix_str}part"] = part
        
        # iso: IsoScore
        iso = IsoScore(points)
        scores[f"{prefix_str}iso"] = iso
        
        # varex: variance explained score
        varex = varex_score(points, p=0.3)
        scores[f"{prefix_str}varex"] = varex
        
        # id_score: intrinsic dimensionality score
        id_sc = id_score(points)
        scores[f"{prefix_str}id_score"] = id_sc
        
    except Exception as e:
        # If any computation fails, skip that metric
        pass
    
    return scores


OBJECTIVE_HANDLERS: Dict[str, Callable[[HeadContext], Dict[str, float]]] = {
    "regression": compute_regression_metrics,
    "binary_classification": compute_binary_metrics,
    "infoNCE": compute_infonce_metrics,
    "contrastive_logit": compute_contrastive_metrics,
}


def compute_metrics(
    eval_pred: EvalPrediction,
    classifier_configs: dict,
    id2_head: dict,
    train_centroid_getter: Optional[Callable[[], Dict[str, Dict[str, Dict[int, np.ndarray]]]]] = None,
    embedding_eval_mode: str = "classifier",
) -> dict:
    """
    Compute evaluation metrics for multi-head classifiers.
    
    Args:
        eval_pred: Predictions and labels from evaluation
        classifier_configs: Configuration for each classifier head
        id2_head: Mapping from head ID to head name
        train_centroid_getter: Function to get training set centroids
        embedding_eval_mode: "classifier" or "original" for embedding evaluation
        
    Returns:
        Dictionary of computed metrics
        
    Notes:
        - eval_pred.predictions contains outputs for each head
        - eval_pred.label_ids contains labels and active_heads information
        - For InfoNCE objectives, behavior depends on embedding_eval_mode:
          * "classifier": Use head-specific embeddings
          * "original": Use original embeddings (avg/cls/max pooling)
    """
    global _original_embeddings_processed
    
    preds = eval_pred.predictions
    label_dict = eval_pred.label_ids

    labels = label_dict["labels"] if isinstance(label_dict, dict) else label_dict
    labels = np.asarray(labels)
    active_head_ids = label_dict.get("active_heads", []) if isinstance(label_dict, dict) else []
    active_heads = [id2_head[i] for i in active_head_ids]

    metrics: Dict[str, float] = {}
    head_vectors: Dict[str, np.ndarray] = {}
    train_centroids = train_centroid_getter() if train_centroid_getter is not None else {}

    # Build vector for original embeddings (if available)
    origin_vector = build_head_vector(
        to_float_array(preds.get("original_avg")),
        to_float_array(preds.get("original_cls")),
        to_float_array(preds.get("original_max")),
    )
    if origin_vector.size > 0:
        head_vectors["original"] = origin_vector

    # Compute isotropy scores for original embeddings (one-time only, using training data)
    for pool_key in ("original_avg", "original_cls", "original_max"):
        if pool_key not in _original_embeddings_processed:
            # Try to get training embeddings from centroids
            embeddings = None
            for head_name, head_data in train_centroids.items():
                embedding_key = f"{pool_key}_all_embeddings"
                if embedding_key in head_data:
                    embeddings = head_data[embedding_key]
                    break  # Use the first available head's embeddings
            
            if embeddings is not None and embeddings.size > 0:
                isotropy_metrics = compute_isotropy_scores(embeddings, prefix=pool_key)
                metrics.update(isotropy_metrics)
                _original_embeddings_processed[pool_key] = True

    # Process each classifier head
    for head, cfg in classifier_configs.items():
        objective = cfg.get("objective", "")
        active_mask = np.asarray([h == head for h in active_heads], dtype=bool)

        scores = to_float_array(preds.get(head))
        pos_scores = to_float_array(preds.get(f"{head}_pos_similarity"))
        neg_scores = to_float_array(preds.get(f"{head}_neg_similarity"))
        anchor_probs = to_float_array(preds.get(f"{head}_anchor_prob"))

        handler = OBJECTIVE_HANDLERS.get(objective)

        # InfoNCE: mode-dependent evaluation
        if objective == "infoNCE" and handler is not None:
            head_entry = (train_centroids or {}).get(head)

            if embedding_eval_mode == "classifier":
                # Use head-specific embeddings
                embedding_source = preds.get(head)
                embeddings = to_float_array(embedding_source)
                centroids = select_centroids(head_entry, mode="classifier") if head_entry else None

                ctx = HeadContext(
                    head=head,
                    objective=objective,
                    scores=scores,
                    pos_scores=pos_scores,
                    neg_scores=neg_scores,
                    anchor_probs=anchor_probs,
                    embeddings=embeddings,
                    labels=labels,
                    active_mask=active_mask,
                    centroids=centroids,
                )
                metrics.update(handler(ctx))
                
                # Compute isotropy scores for classifier embeddings (every evaluation step, using training data)
                train_embeddings = None
                if head_entry and "classifier_all_embeddings" in head_entry:
                    train_embeddings = head_entry["classifier_all_embeddings"]
                
                if train_embeddings is not None and train_embeddings.size > 0:
                    isotropy_metrics = compute_isotropy_scores(train_embeddings, prefix=head)
                    metrics.update(isotropy_metrics)

            elif embedding_eval_mode == "original":
                # Evaluate with original embeddings (3 pooling methods)
                for pool_key in ("original_avg", "original_cls", "original_max"):
                    embedding_source = preds.get(pool_key)
                    embeddings = to_float_array(embedding_source)
                    if embeddings is None:
                        continue

                    pool_centroids: Optional[Dict[int, np.ndarray]] = None
                    if isinstance(head_entry, dict) and pool_key in head_entry:
                        pool_centroids = head_entry[pool_key]
                    elif head_entry is not None:
                        pool_centroids = select_centroids(head_entry, mode="original")

                    ctx = HeadContext(
                        head=head,
                        objective=objective,
                        scores=scores,
                        pos_scores=pos_scores,
                        neg_scores=neg_scores,
                        anchor_probs=anchor_probs,
                        embeddings=embeddings,
                        labels=labels,
                        active_mask=active_mask,
                        centroids=pool_centroids,
                    )
                    local_metrics = handler(ctx)
                    # Prefix metrics with pooling method
                    prefixed = {f"{pool_key}_{k}": v for k, v in local_metrics.items()}
                    metrics.update(prefixed)

        else:
            # Other objectives (regression, binary classification, contrastive)
            if handler is not None:
                ctx = HeadContext(
                    head=head,
                    objective=objective,
                    scores=scores,
                    pos_scores=pos_scores,
                    neg_scores=neg_scores,
                    anchor_probs=anchor_probs,
                    embeddings=None,
                    labels=labels,
                    active_mask=active_mask,
                    centroids=None,
                )
                metrics.update(handler(ctx))

        # Build score vector for correlation analysis
        head_vectors[head] = build_head_vector(scores, pos_scores, neg_scores)

    # Compute pairwise correlations between heads
    for h1, h2 in combinations(head_vectors.keys(), 2):
        v1 = np.asarray(head_vectors[h1]).flatten()
        v2 = np.asarray(head_vectors[h2]).flatten()

        if v1.size <= 1 or v2.size <= 1:
            continue
        if len(v1) != len(v2):
            continue

        pcc = pearsonr(v1, v2)[0]
        scc = spearmanr(v1, v2).correlation

        metrics[f"{h1}_vs_{h2}_relation-pearson"] = float(pcc)
        metrics[f"{h1}_vs_{h2}_relation-spearman"] = float(scc)

    return metrics


# Export original function name for backward compatibility
__all__ = ["compute_metrics", "OBJECTIVE_HANDLERS"]
