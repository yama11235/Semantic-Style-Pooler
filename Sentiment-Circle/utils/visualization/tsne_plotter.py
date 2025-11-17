"""T-SNE visualization utilities."""
from typing import Dict, Optional, List, Any
import numpy as np
import torch
from utils.visualization.plot_2d import plot_tsne_embedding_space


class TSNEVisualizer:
    """Handle T-SNE visualization during training."""
    
    def __init__(
        self,
        tsne_save_dir: Optional[str],
        classifier_configs: Dict,
        tsne_label_mappings: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        """
        Initialize T-SNE visualizer.
        
        Args:
            tsne_save_dir: Directory to save T-SNE plots
            classifier_configs: Configuration for each classifier head
            tsne_label_mappings: Mapping from label IDs to names
        """
        self.tsne_save_dir = tsne_save_dir
        self.classifier_configs = classifier_configs
        self.tsne_label_mappings = tsne_label_mappings or {}
    
    def save_tsne_plot(
        self,
        predictions: Dict,
        label_ids: Any,
        metric_key_prefix: str,
        embedding_mode: str,
        seed: int,
        global_step: int,
        centroid_getter_fn,
        to_numpy_fn,
        extract_labels_fn,
        extract_active_heads_fn,
    ) -> None:
        """
        Save T-SNE plots for current predictions.
        
        Args:
            predictions: Model predictions dictionary
            label_ids: Label information
            metric_key_prefix: Metric prefix (e.g., "eval")
            embedding_mode: "classifier" or "original"
            seed: Random seed for reproducibility
            global_step: Current training step
            centroid_getter_fn: Function to get training centroids
            to_numpy_fn: Function to convert tensors to numpy
            extract_labels_fn: Function to extract labels
            extract_active_heads_fn: Function to extract active heads
        """
        if not self.tsne_save_dir:
            return
        
        plotting_kwargs = {
            "metric_key_prefix": metric_key_prefix,
            "save_dir": self.tsne_save_dir,
            "tsne_label_mappings": self.tsne_label_mappings,
            "seed": seed,
            "global_step": global_step,
        }

        if embedding_mode == "original":
            self._plot_original_embeddings(
                predictions,
                label_ids,
                centroid_getter_fn,
                to_numpy_fn,
                extract_labels_fn,
                extract_active_heads_fn,
                plotting_kwargs,
            )
        else:
            self._plot_classifier_embeddings(
                predictions,
                label_ids,
                to_numpy_fn,
                extract_labels_fn,
                extract_active_heads_fn,
                plotting_kwargs,
            )
    
    def _plot_original_embeddings(
        self,
        predictions: Dict,
        label_ids: Any,
        centroid_getter_fn,
        to_numpy_fn,
        extract_labels_fn,
        extract_active_heads_fn,
        plotting_kwargs: Dict,
    ) -> None:
        """Plot embeddings from original (pretrained) encoder."""
        from utils.training.centroid_calculator import CentroidCalculator
        
        train_centroids = centroid_getter_fn()
        
        for pool_key in ("original_avg", "original_cls", "original_max", "original_last"):
            embeddings = predictions.get(pool_key)
            emb_array = to_numpy_fn(embeddings)
            if emb_array is None or emb_array.ndim != 2 or emb_array.shape[0] < 2:
                continue

            labels = extract_labels_fn(label_ids, emb_array.shape[0])
            head_names = extract_active_heads_fn(label_ids, emb_array.shape[0])
            
            ref_emb, ref_labels, ref_heads = CentroidCalculator.collect_reference_centroids(
                pool_key, train_centroids
            )

            plot_tsne_embedding_space(
                embeddings=emb_array,
                labels=labels,
                head_name=pool_key,
                point_head_names=head_names,
                reference_embeddings=ref_emb,
                reference_labels=ref_labels,
                reference_head_names=ref_heads,
                reference_label_suffix=" (train centroid)",
                **plotting_kwargs,
            )
    
    def _plot_classifier_embeddings(
        self,
        predictions: Dict,
        label_ids: Any,
        to_numpy_fn,
        extract_labels_fn,
        extract_active_heads_fn,
        plotting_kwargs: Dict,
    ) -> None:
        """Plot embeddings from classifier heads."""
        for head_name in self.classifier_configs.keys():
            head_preds = predictions.get(head_name)
            emb_array = to_numpy_fn(head_preds)
            if emb_array is None or emb_array.ndim != 2 or emb_array.shape[0] == 0:
                continue

            labels = extract_labels_fn(label_ids, emb_array.shape[0])
            active_heads = extract_active_heads_fn(label_ids, emb_array.shape[0])
            mask = np.array([h == head_name for h in active_heads], dtype=bool)
            if not mask.any():
                continue

            head_embeddings = emb_array[mask]
            head_labels = labels[mask]
            if head_embeddings.shape[0] < 2:
                continue

            plot_tsne_embedding_space(
                embeddings=head_embeddings,
                labels=head_labels,
                head_name=head_name,
                **plotting_kwargs,
            )
