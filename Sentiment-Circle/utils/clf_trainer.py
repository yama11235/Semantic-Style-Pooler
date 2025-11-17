"""Custom trainer with multi-head classifier support - refactored version."""
import os
import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer
from typing import List, Dict, Any, Optional
from collections import defaultdict

from utils.data.batch_utils import (
    BatchPartitioner,
    extract_unique_strings,
    flatten_strings,
)
from utils.training.objectives import (
    AngleNCEObjective,
    BinaryClassificationObjective,
    ContrastiveLogitObjective,
    HeadObjective,
    InfoNCEObjective,
    RegressionObjective,
)
from utils.loss_function import (
    compute_pair_correlation_penalty,
    compute_pair_loss,
    compute_single_loss,
    compute_triplet_correlation_penalty,
    compute_triplet_loss,
    fill_missing_output_keys,
)
from utils.training.centroid_calculator import CentroidCalculator
from utils.visualization.tsne_plotter import TSNEVisualizer


def pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe Pearson correlation on GPU."""
    if x.numel() < 2 or y.numel() < 2:
        return x.new_tensor(0.0)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return x.new_tensor(0.0)
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = torch.sqrt((xm * xm).sum() * (ym * ym).sum()).clamp_min(eps)
    return num / den


class CustomTrainer(Trainer):
    """Custom trainer for multi-head classifier training."""
    
    def __init__(
        self,
        *args,
        classifier_configs,
        dtype=torch.float16,
        tsne_save_dir: Optional[str] = None,
        corr_labels: Optional[Dict[str, Dict[str, float]]] = None,
        corr_weights: Optional[Dict[str, Dict[str, float]]] = None,
        tsne_label_mappings: Optional[Dict[str, Dict[int, str]]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.classifier_configs = classifier_configs
        self.dtype = dtype
        self.tsne_save_dir = tsne_save_dir
        self._tsne_current_dataset = None
        self._last_eval_predictions: Optional[Any] = None
        self._last_eval_label_ids: Optional[Any] = None
        self._cached_train_centroids: Optional[Dict[str, Dict[str, Dict[int, np.ndarray]]]] = None
        self._train_centroids_dirty: bool = True
        self.tsne_label_mappings: Dict[str, Dict[int, str]] = tsne_label_mappings or {}

        self.loss_fns = {}
        self.info_nce_params: Dict[str, Dict[str, Any]] = {}
        self.angle_nce_params: Dict[str, Dict[str, Any]] = {}

        for name, cfg in classifier_configs.items():
            obj = cfg["objective"]
            if obj == "infoNCE":
                self.info_nce_params[name] = {
                    "tau": cfg.get("tau", 1.0),
                    "pos_pairs": cfg.get("inbatch_positive_pairs", 10),
                    "neg_pairs": cfg.get("inbatch_negative_pairs", 64),
                }
            elif obj == "angleNCE":
                self.angle_nce_params[name] = {
                    "angle_map": cfg.get("angle_map", {}),
                    "id2label": cfg.get("id2label", {}),
                    "pos_pairs": cfg.get("inbatch_positive_pairs", 10),
                    "neg_pairs": cfg.get("inbatch_negative_pairs", 64),
                }
        
        self.head2idx = {head: i for i, head in enumerate(classifier_configs.keys())}
        self.idx2head = {idx: head for head, idx in self.head2idx.items()}
        
        self.head_objectives: Dict[str, HeadObjective] = {}
        for name, cfg in classifier_configs.items():
            obj = cfg.get("objective")
            if obj == "infoNCE":
                self.head_objectives[name] = InfoNCEObjective(name, cfg)
            elif obj == "angleNCE":
                self.head_objectives[name] = AngleNCEObjective(name, cfg)
            elif obj == "regression":
                self.head_objectives[name] = RegressionObjective(name, cfg)
            elif obj == "binary_classification":
                self.head_objectives[name] = BinaryClassificationObjective(name, cfg)
            elif obj == "contrastive_logit":
                self.head_objectives[name] = ContrastiveLogitObjective(name, cfg)

        self.corr_labels = defaultdict(dict)
        if corr_labels:
            for head_i, mapping in corr_labels.items():
                if mapping:
                    self.corr_labels[head_i].update(mapping)

        self.corr_weights = defaultdict(dict)
        if corr_weights:
            for head_i, mapping in corr_weights.items():
                if mapping:
                    self.corr_weights[head_i].update(mapping)

        if self.tsne_save_dir is not None:
            os.makedirs(self.tsne_save_dir, exist_ok=True)
        
        self._initial_eval_completed = False
        self._current_eval_embedding_mode = "classifier"
        
        self.centroid_calculator = CentroidCalculator(
            classifier_configs, self.head2idx, self.idx2head
        )
        self.tsne_visualizer = TSNEVisualizer(
            tsne_save_dir, classifier_configs, tsne_label_mappings
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        partitioner = BatchPartitioner(
            attention_mask=inputs["attention_mask"],
            attention_mask_2=inputs.get("attention_mask_2"),
            attention_mask_3=inputs.get("attention_mask_3"),
        )
        bsz = inputs["input_ids"].size(0)

        inputs1 = partitioner.slice(inputs, "single", device)
        inputs2 = partitioner.slice(inputs, "pair", device)
        inputs3 = partitioner.slice(inputs, "triplet", device)

        active_heads = extract_unique_strings(inputs["active_heads"])
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        single_loss, outputs1 = compute_single_loss(self, model, inputs1, active_heads, device)
        if single_loss is not None:
            total_loss = total_loss + single_loss

        pair_loss, outputs2 = compute_pair_loss(self, model, inputs2, active_heads, device)
        if pair_loss is not None:
            total_loss = total_loss + pair_loss

        pair_corr_penalty = compute_pair_correlation_penalty(self, outputs2, device)
        if pair_corr_penalty is not None:
            total_loss = total_loss + pair_corr_penalty

        triplet_loss, outputs3 = compute_triplet_loss(self, model, inputs3, active_heads, device)
        if triplet_loss is not None:
            total_loss = total_loss + triplet_loss

        triplet_corr_penalty = compute_triplet_correlation_penalty(self, outputs3, device)
        if triplet_corr_penalty is not None:
            total_loss = total_loss + triplet_corr_penalty

        full_outputs = partitioner.merge(
            {"single": outputs1, "pair": outputs2, "triplet": outputs3},
            device=device,
        )

        fill_missing_output_keys(self, full_outputs, bsz, device)

        if not return_outputs:
            return total_loss
        return total_loss, full_outputs

    def training_step(self, model, inputs, num_items_in_batch: Optional[int] = None):
        output = super().training_step(model, inputs, num_items_in_batch)
        self._train_centroids_dirty = True
        return output

    def evaluation_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        labels_tensor = inputs["labels"]
        ah_list = flatten_strings(inputs["active_heads"])
        ah_ids = [self.head2idx[h] for h in ah_list]
        ah_tensor = torch.tensor(ah_ids, device=loss.device, dtype=torch.long)

        label_dict = {"labels": labels_tensor, "active_heads": ah_tensor}
        return loss, logits, label_dict

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return self.evaluation_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        previous_dataset = self._tsne_current_dataset
        self._tsne_current_dataset = dataset
        global_step = self.state.global_step if self.state is not None else 0
        use_original_now = (not self._initial_eval_completed) and global_step == 0
        self._current_eval_embedding_mode = "original" if use_original_now else "classifier"
        try:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            if self._current_eval_embedding_mode == "original":
                self._initial_eval_completed = True
            self._current_eval_embedding_mode = "classifier"
            self._tsne_current_dataset = previous_dataset

    def evaluation_loop(self, dataloader, description: str, prediction_loss_only: Optional[bool] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval"):
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        self._last_eval_predictions = output.predictions
        self._last_eval_label_ids = output.label_ids
        if (self.tsne_save_dir and self._tsne_current_dataset is not None and self._tsne_current_dataset is self.eval_dataset and metric_key_prefix.startswith("eval")):
            self._save_tsne_plot(metric_key_prefix)
        return output

    def get_train_label_centroids(self) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
        if self.train_dataset is None:
            return {}
        if not self._train_centroids_dirty and self._cached_train_centroids is not None:
            return self._cached_train_centroids
        self.centroid_calculator._prepare_inputs = self._prepare_inputs
        centroids = self.centroid_calculator.build_train_centroids(
            self.train_dataset, self.model, self.get_eval_dataloader(self.train_dataset), self.compute_loss
        )
        self._cached_train_centroids = centroids
        self._train_centroids_dirty = False
        return centroids

    def _to_numpy(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _extract_labels(self, label_ids: Any, target_len: int) -> np.ndarray:
        labels = label_ids.get("labels") if isinstance(label_ids, dict) else label_ids
        arr = self._to_numpy(labels)
        if arr is None or arr.size == 0:
            return np.arange(target_len)
        flat = arr.reshape(-1)
        if flat.shape[0] < target_len:
            pad = np.full(target_len - flat.shape[0], -1)
            flat = np.concatenate([flat, pad])
        elif flat.shape[0] > target_len:
            flat = flat[:target_len]
        return flat

    def _extract_active_head_names(self, label_ids: Any, target_len: int) -> List[Optional[str]]:
        if not isinstance(label_ids, dict):
            return [None] * target_len
        active = label_ids.get("active_heads")
        arr = self._to_numpy(active)
        if arr is None or arr.size == 0:
            return [None] * target_len
        flat = arr.reshape(-1)
        names: List[Optional[str]] = []
        for i in range(target_len):
            if i < flat.shape[0]:
                idx = int(flat[i])
                names.append(self.idx2head.get(idx))
            else:
                names.append(None)
        return names

    @property
    def use_original_eval_embeddings(self) -> bool:
        return self._current_eval_embedding_mode == "original"

    def _save_tsne_plot(self, metric_key_prefix: str) -> None:
        seed = getattr(self.args, "seed", 42) or 42
        global_step = self.state.global_step if self.state is not None else 0
        self.tsne_visualizer.save_tsne_plot(
            predictions=self._last_eval_predictions,
            label_ids=self._last_eval_label_ids,
            metric_key_prefix=metric_key_prefix,
            embedding_mode=self._current_eval_embedding_mode,
            seed=seed,
            global_step=global_step,
            centroid_getter_fn=self.get_train_label_centroids,
            to_numpy_fn=self._to_numpy,
            extract_labels_fn=self._extract_labels,
            extract_active_heads_fn=self._extract_active_head_names,
        )
