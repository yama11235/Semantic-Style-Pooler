"""Centroid calculation utilities for training."""
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import torch
from collections import defaultdict


class CentroidCalculator:
    """Calculate centroids for training data by head and label."""
    
    def __init__(self, classifier_configs: Dict, head2idx: Dict[str, int], idx2head: Dict[int, str]):
        """
        Initialize centroid calculator.
        
        Args:
            classifier_configs: Configuration for each classifier head
            head2idx: Mapping from head name to index
            idx2head: Mapping from index to head name
        """
        self.classifier_configs = classifier_configs
        self.head2idx = head2idx
        self.idx2head = idx2head
    
    def build_train_centroids(
        self,
        train_dataset,
        model,
        dataloader,
        compute_loss_fn,
    ) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
        """
        Calculate centroids from training dataset.
        
        Returns:
            Dictionary structure:
            {
                head_name: {
                    "classifier": { label_int: np.ndarray },
                    "original_avg": { label_int: np.ndarray },
                    "original_cls": { label_int: np.ndarray },
                    "original_max": { label_int: np.ndarray },
                    "original_last": { label_int: np.ndarray },
                    "classifier_all_embeddings": np.ndarray,  # All embeddings for this head
                    "original_avg_all_embeddings": np.ndarray,
                    "original_cls_all_embeddings": np.ndarray,
                    "original_max_all_embeddings": np.ndarray,
                    "original_last_all_embeddings": np.ndarray,
                },
                ...
            }
        """
        if train_dataset is None:
            return {}

        was_training = model.training
        model.eval()

        # Accumulator dictionaries
        head_sums_classifier: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        head_counts_classifier: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        head_sums_original: Dict[str, Dict[str, Dict[int, np.ndarray]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        head_counts_original: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        
        # Accumulators for all embeddings (not just centroids)
        head_all_embeddings_classifier: Dict[str, List[np.ndarray]] = defaultdict(list)
        head_all_embeddings_original: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )

        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_inputs(batch)
                _, outputs = compute_loss_fn(model, inputs, return_outputs=True)

                labels_tensor = inputs.get("labels")
                if labels_tensor is None:
                    continue
                # convert bf16 to float32 for numpy conversion
                labels_np = labels_tensor.detach().cpu().to(torch.float32).view(-1).numpy()
                batch_size = labels_np.shape[0]

                # Convert head outputs to numpy
                head_arrays: Dict[str, np.ndarray] = {}
                for head_name, cfg in self.classifier_configs.items():
                    if cfg.get("objective") != "infoNCE":
                        continue
                    head_tensor = outputs.get(head_name)
                    if head_tensor is None:
                        continue
                    arr = head_tensor.detach().cpu().to(torch.float32).numpy()
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    head_arrays[head_name] = arr

                # Convert original embeddings to numpy
                original_arrays: Dict[str, np.ndarray] = {}
                for key in ("original_avg", "original_cls", "original_max", "original_last"):
                    tensor = outputs.get(key)
                    if tensor is None:
                        continue
                    arr = tensor.detach().cpu().to(torch.float32).numpy()
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    original_arrays[key] = arr

                # Get active heads
                active_heads_field = inputs.get("active_heads", [])
                head_list = self._flatten_strings(active_heads_field)
                if len(head_list) < batch_size:
                    head_list = head_list + [""] * (batch_size - len(head_list))
                elif len(head_list) > batch_size:
                    head_list = head_list[:batch_size]

                # Accumulate per sample
                for idx in range(batch_size):
                    if idx >= len(labels_np):
                        continue

                    head_name = head_list[idx] if idx < len(head_list) else None
                    if not head_name:
                        continue

                    cfg = self.classifier_configs.get(head_name)
                    if not cfg or cfg.get("objective") != "infoNCE":
                        continue

                    label_value = labels_np[idx]
                    if not np.isfinite(label_value):
                        continue
                    label_int = int(label_value)

                    # Classifier embeddings
                    classifier_arr = head_arrays.get(head_name)
                    if classifier_arr is not None and idx < classifier_arr.shape[0]:
                        vector_classifier = classifier_arr[idx]
                        if np.isfinite(vector_classifier).all():
                            # Accumulate for centroids
                            sums_for_head = head_sums_classifier[head_name]
                            if label_int in sums_for_head:
                                sums_for_head[label_int] += vector_classifier
                            else:
                                sums_for_head[label_int] = vector_classifier.copy()
                            head_counts_classifier[head_name][label_int] += 1
                            # Store all embeddings
                            head_all_embeddings_classifier[head_name].append(vector_classifier)

                    # Original embeddings (pooling variants)
                    for pool_key, arr in original_arrays.items():
                        if idx >= arr.shape[0]:
                            continue
                        vec = arr[idx]
                        if not np.isfinite(vec).all():
                            continue

                        # Accumulate for centroids
                        sums_for_pool = head_sums_original[head_name][pool_key]
                        if label_int in sums_for_pool:
                            sums_for_pool[label_int] += vec
                        else:
                            sums_for_pool[label_int] = vec.copy()
                        head_counts_original[head_name][pool_key][label_int] += 1
                        # Store all embeddings
                        head_all_embeddings_original[head_name][pool_key].append(vec)

        if was_training:
            model.train()

        # Convert sums to centroids
        centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}

        all_heads = set(head_sums_classifier.keys()) | set(head_sums_original.keys())
        for head_name in all_heads:
            centroids_head: Dict[str, Dict[int, np.ndarray]] = {}

            # Classifier centroids
            label_sums_classifier = head_sums_classifier.get(head_name, {})
            if label_sums_classifier:
                centroids_head["classifier"] = {}
                for label_value, vector_sum in label_sums_classifier.items():
                    count = head_counts_classifier[head_name][label_value]
                    if count <= 0:
                        continue
                    centroids_head["classifier"][label_value] = (
                        vector_sum / count
                    ).astype(np.float32)

            # Original centroids (per pooling method)
            label_sums_original_by_pool = head_sums_original.get(head_name, {})
            for pool_key, label_sums in label_sums_original_by_pool.items():
                if not label_sums:
                    continue
                centroids_head[pool_key] = {}
                for label_value, vector_sum in label_sums.items():
                    count = head_counts_original[head_name][pool_key][label_value]
                    if count <= 0:
                        continue
                    centroids_head[pool_key][label_value] = (
                        vector_sum / count
                    ).astype(np.float32)
            
            # Store all embeddings (for isotropy score calculation)
            if head_name in head_all_embeddings_classifier and head_all_embeddings_classifier[head_name]:
                centroids_head["classifier_all_embeddings"] = np.vstack(
                    head_all_embeddings_classifier[head_name]
                ).astype(np.float32)
            
            for pool_key in ("original_avg", "original_cls", "original_max", "original_last"):
                if head_name in head_all_embeddings_original and pool_key in head_all_embeddings_original[head_name]:
                    embeddings_list = head_all_embeddings_original[head_name][pool_key]
                    if embeddings_list:
                        centroids_head[f"{pool_key}_all_embeddings"] = np.vstack(
                            embeddings_list
                        ).astype(np.float32)

            if centroids_head:
                centroids[head_name] = centroids_head

        return centroids
    
    @staticmethod
    def _prepare_inputs(batch):
        """Prepare batch inputs (to be implemented by trainer)."""
        return batch
    
    @staticmethod
    def _flatten_strings(field):
        """Flatten string field."""
        if isinstance(field, (list, tuple)):
            result = []
            for item in field:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, (list, tuple)):
                    result.extend(CentroidCalculator._flatten_strings(item))
            return result
        return []
    
    @staticmethod
    def collect_reference_centroids(
        pool_key: str,
        train_centroids: Optional[Dict[str, Dict[str, Dict[int, np.ndarray]]]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Optional[str]]]]:
        """
        Collect centroids for reference points in visualization.
        
        Args:
            pool_key: Pooling key (e.g., "original_avg")
            train_centroids: Training centroids dictionary
            
        Returns:
            Tuple of (embeddings, labels, head_names)
        """
        if not train_centroids:
            return None, None, None

        centroid_embeddings: List[np.ndarray] = []
        centroid_labels: List[int] = []
        centroid_heads: List[Optional[str]] = []

        for head_name, entries in train_centroids.items():
            pool_centroids = entries.get(pool_key)
            if not pool_centroids:
                continue
            for label_value, vector in pool_centroids.items():
                centroid_embeddings.append(vector)
                centroid_labels.append(label_value)
                centroid_heads.append(head_name)

        if not centroid_embeddings:
            return None, None, None

        emb_array = np.asarray(centroid_embeddings, dtype=np.float32)
        label_array = np.asarray(centroid_labels, dtype=int)
        return emb_array, label_array, centroid_heads
