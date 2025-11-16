import os

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from utils.sentence_batch_utils import (
    BatchPartitioner,
    extract_unique_strings,
    flatten_strings,
)
from utils.head_objectives import (
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
from utils.plot_2D import plot_tsne_embedding_space

def pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    GPU 上でも安全に動く Pearson 相関係数の実装。
    要素数チェックと分散ゼロ回避を含みます。
    """
    # 要素数チェック
    if x.numel() < 2 or y.numel() < 2:
        return x.new_tensor(0.0)
    # NaN/Inf を除外
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return x.new_tensor(0.0)
    # 標本平均を引く
    xm = x - x.mean()
    ym = y - y.mean()
    # 分子・分母
    num = (xm * ym).sum()
    den = torch.sqrt((xm * xm).sum() * (ym * ym).sum()).clamp_min(eps)
    return num / den

class CustomTrainer(Trainer):
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
        """
        classifier_configs: dict of { head_name: {objective: "infoNCE"/"angleNCE", ...}, ... }
        """
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

        # 損失関数リストを準備
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        フロー：
        A) バッチを 1文 vs 2文 vs 3文 に分割
        B) 1文サブバッチは通常の encode() → embeddingから損失ペアを構成して計算
        C) 2文サブバッチを active_heads ごとに分けて教師損失
        D) 2文サブバッチ全体で相関ペナルティ
        E) 3文サブバッチは contrastive_logit
        F) 3文サブバッチ全体で相関ペナルティ
        """
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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
            {
                "single": outputs1,
                "pair": outputs2,
                "triplet": outputs3,
            },
            device=device,
        )

        fill_missing_output_keys(self, full_outputs, bsz, device)

        if not return_outputs:
            return total_loss

        return total_loss, full_outputs

    def training_step(self, model, inputs, num_items_in_batch: Optional[int] = None):
        # print("Starting training step...")
        output = super().training_step(model, inputs, num_items_in_batch)
        self._train_centroids_dirty = True
        return output


    def evaluation_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        # print(f"eval loss: {loss}")
        labels_tensor = inputs["labels"]  # torch.Tensor
        # 文字列 → 整数に変換
        ah_list = flatten_strings(inputs["active_heads"])  # 例: ["emotion", "emotion", ...]

        ah_ids = [ self.head2idx[h] for h in ah_list ]
        ah_tensor = torch.tensor(ah_ids, device=loss.device, dtype=torch.long)

        label_dict = {
            "labels": labels_tensor,      # torch.Tensor
            "active_heads": ah_tensor,    # torch.Tensor (整数)
        }
        return loss, logits, label_dict

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return self.evaluation_step(model, inputs, prediction_loss_only, ignore_keys)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """オリジナルの evaluate を拡張し、評価前後で T-SNE プロットを生成する。"""
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

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self._last_eval_predictions = output.predictions
        self._last_eval_label_ids = output.label_ids

        if (
            self.tsne_save_dir
            and self._tsne_current_dataset is not None
            and self._tsne_current_dataset is self.eval_dataset
            and metric_key_prefix.startswith("eval")
        ):
            self._save_tsne_plot(metric_key_prefix)

        return output

    def get_train_label_centroids(self) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
        if self.train_dataset is None:
            return {}
        if not self._train_centroids_dirty and self._cached_train_centroids is not None:
            return self._cached_train_centroids
        centroids = self._build_train_label_centroids()
        self._cached_train_centroids = centroids
        self._train_centroids_dirty = False
        return centroids

    def _build_train_label_centroids(self) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
        dataset = self.train_dataset
        if dataset is None:
            return {}

        dataloader = self.get_eval_dataloader(dataset)
        model = self.model
        was_training = model.training
        model.eval()

        head_sums_classifier: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        head_counts_classifier: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        head_sums_original: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        head_counts_original: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_inputs(batch)
                _, outputs = self.compute_loss(model, inputs, return_outputs=True)
                labels_tensor = inputs.get("labels")
                if labels_tensor is None:
                    continue
                labels_np = labels_tensor.detach().cpu().view(-1).numpy()
                batch_size = labels_np.shape[0]

                original_embeddings = outputs.get("embeddings")
                original_np = None
                if original_embeddings is not None:
                    original_np = original_embeddings.detach().cpu().numpy()
                    if original_np.ndim != 2 or original_np.shape[0] == 0:
                        original_np = None

                head_arrays: Dict[str, np.ndarray] = {}
                for head_name, cfg in self.classifier_configs.items():
                    if cfg.get("objective") != "infoNCE":
                        continue
                    head_tensor = outputs.get(head_name)
                    if head_tensor is None:
                        continue
                    arr = head_tensor.detach().cpu().numpy()
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    head_arrays[head_name] = arr

                active_heads_field = inputs.get("active_heads", [])
                head_list = flatten_strings(active_heads_field)
                if len(head_list) < batch_size:
                    head_list = head_list + [""] * (batch_size - len(head_list))
                elif len(head_list) > batch_size:
                    head_list = head_list[:batch_size]

                for idx in range(batch_size):
                    if idx >= len(labels_np):
                        continue
                    head_name = head_list[idx] if idx < len(head_list) else None
                    if not head_name:
                        continue
                    cfg = self.classifier_configs.get(head_name)
                    # infoNCE のみ対応
                    if not cfg or cfg.get("objective") != "infoNCE":
                        continue
                    label_value = labels_np[idx]
                    if not np.isfinite(label_value):
                        continue
                    label_int = int(label_value)

                    classifier_arr = head_arrays.get(head_name)
                    vector_classifier = None
                    if classifier_arr is not None and idx < classifier_arr.shape[0]:
                        vector_classifier = classifier_arr[idx]
                        if not np.isfinite(vector_classifier).all():
                            vector_classifier = None

                    vector_original = None
                    if original_np is not None and idx < original_np.shape[0]:
                        vector_original = original_np[idx]
                        if not np.isfinite(vector_original).all():
                            vector_original = None

                    if vector_classifier is not None:
                        sums_for_head = head_sums_classifier[head_name]
                        if label_int in sums_for_head:
                            sums_for_head[label_int] += vector_classifier
                        else:
                            sums_for_head[label_int] = vector_classifier.copy()
                        head_counts_classifier[head_name][label_int] += 1

                    if vector_original is not None:
                        sums_for_original = head_sums_original[head_name]
                        if label_int in sums_for_original:
                            sums_for_original[label_int] += vector_original
                        else:
                            sums_for_original[label_int] = vector_original.copy()
                        head_counts_original[head_name][label_int] += 1

        if was_training:
            model.train()

        centroids: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
        all_heads = set(head_sums_classifier.keys()) | set(head_sums_original.keys())
        for head_name in all_heads:
            centroids[head_name] = {}
            label_sums_classifier = head_sums_classifier.get(head_name, {})
            if label_sums_classifier:
                centroids[head_name]["classifier"] = {}
                for label_value, vector_sum in label_sums_classifier.items():
                    count = head_counts_classifier[head_name][label_value]
                    if count <= 0:
                        continue
                    centroids[head_name]["classifier"][label_value] = (
                        vector_sum / count
                    ).astype(np.float32)
            label_sums_original = head_sums_original.get(head_name, {})
            if label_sums_original:
                centroids[head_name]["original"] = {}
                for label_value, vector_sum in label_sums_original.items():
                    count = head_counts_original[head_name][label_value]
                    if count <= 0:
                        continue
                    centroids[head_name]["original"][label_value] = (
                        vector_sum / count
                    ).astype(np.float32)
            if not centroids[head_name]:
                centroids.pop(head_name)

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
        labels = None
        if isinstance(label_ids, dict):
            labels = label_ids.get("labels")
        else:
            labels = label_ids

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
        if not self.tsne_save_dir:
            return
        if not isinstance(self._last_eval_predictions, dict):
            return

        seed = getattr(self.args, "seed", 42) or 42
        global_step = self.state.global_step if self.state is not None else 0
        plotting_kwargs = {
            "metric_key_prefix": metric_key_prefix,
            "save_dir": self.tsne_save_dir,
            "tsne_label_mappings": self.tsne_label_mappings,
            "seed": seed,
            "global_step": global_step,
        }

        if self.use_original_eval_embeddings:
            embeddings = self._last_eval_predictions.get("embeddings")
            emb_array = self._to_numpy(embeddings)
            if emb_array is None or emb_array.ndim != 2 or emb_array.shape[0] < 2:
                return
            labels = self._extract_labels(self._last_eval_label_ids, emb_array.shape[0])
            head_names = self._extract_active_head_names(self._last_eval_label_ids, emb_array.shape[0])
            plot_tsne_embedding_space(
                embeddings=emb_array,
                labels=labels,
                head_name=None,
                point_head_names=head_names,
                **plotting_kwargs,
            )
            return

        plotted = False
        for head_name in self.classifier_configs.keys():
            head_preds = self._last_eval_predictions.get(head_name)
            emb_array = self._to_numpy(head_preds)
            if emb_array is None or emb_array.ndim != 2 or emb_array.shape[0] == 0:
                continue

            labels = self._extract_labels(self._last_eval_label_ids, emb_array.shape[0])
            active_heads = self._extract_active_head_names(self._last_eval_label_ids, emb_array.shape[0])
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
            plotted = True

        if plotted:
            return

        embeddings = self._last_eval_predictions.get("embeddings")
        emb_array = self._to_numpy(embeddings)
        if emb_array is None or emb_array.ndim != 2 or emb_array.shape[0] < 2:
            return

        labels = self._extract_labels(self._last_eval_label_ids, emb_array.shape[0])
        head_names = self._extract_active_head_names(self._last_eval_label_ids, emb_array.shape[0])
        plot_tsne_embedding_space(
            embeddings=emb_array,
            labels=labels,
            head_name=None,
            point_head_names=head_names,
            **plotting_kwargs,
        )

