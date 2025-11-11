import torch
import torch.nn.functional as F
from transformers import Trainer
from torch import Tensor
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from .sentence_batch_utils import (
    BatchPartitioner,
    extract_unique_strings,
    flatten_strings,
)
from .head_objectives import (
    AngleNCEObjective,
    BinaryClassificationObjective,
    ContrastiveLogitObjective,
    HeadObjective,
    InfoNCEObjective,
    RegressionObjective,
)

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
        **kwargs
    ):
        """
        classifier_configs: dict of { head_name: {objective: "infoNCE"/"angleNCE", ...}, ... }
        """
        super().__init__(*args, **kwargs)
        self.classifier_configs = classifier_configs
        self.dtype = dtype

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
        self.corr_weights = defaultdict(dict)

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
        labels1 = inputs1.get("labels") if inputs1 else None
        labels2 = inputs2.get("labels") if inputs2 else None
        labels3 = inputs3.get("labels") if inputs3 else None

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        outputs1: Dict[str, torch.Tensor] = {}
        if inputs1:
            outputs1 = model(**inputs1)
            embeddings1 = outputs1.get("embeddings")
            if embeddings1 is None:
                raise ValueError("Model did not return 'embeddings' for single-sentence inputs.")
            for head in active_heads:
                objective = self.head_objectives.get(head)
                if objective is None:
                    continue
                mask = self._head_mask(inputs1.get("active_heads"), head, device)
                if mask is None or mask.sum() == 0:
                    continue
                if labels1 is None:
                    continue
                sub_emb = embeddings1[mask]
                sub_tgt = labels1[mask].flatten()
                valid = self._finite_mask(sub_tgt)
                if valid.sum() == 0:
                    continue
                sub_emb = sub_emb[valid]
                sub_tgt = sub_tgt[valid]
                if sub_emb.size(0) < 2:
                    continue
                loss = objective.compute_single(self, sub_emb, sub_tgt)
                if loss is not None:
                    total_loss = total_loss + loss

        outputs2: Dict[str, torch.Tensor] = {}
        if inputs2:
            outputs2 = model(**inputs2)
            for head in active_heads:
                objective = self.head_objectives.get(head)
                if objective is None:
                    continue
                tensor = outputs2.get(head)
                if tensor is None:
                    continue
                mask = self._head_mask(inputs2.get("active_heads"), head, device)
                if mask is None or mask.sum() == 0:
                    continue
                if labels2 is None:
                    continue
                sub_out = tensor[mask]
                sub_tgt = labels2[mask].flatten()
                valid = self._finite_mask(sub_tgt)
                if valid.sum() == 0:
                    continue
                sub_out = sub_out[valid]
                sub_tgt = sub_tgt[valid]
                loss = objective.compute_pair(sub_out, sub_tgt)
                if loss is not None:
                    total_loss = total_loss + loss

            for hi, row in self.corr_labels.items():
                for hj, target_corr in row.items():
                    if hi not in outputs2 or hj not in outputs2:
                        continue
                    oi = outputs2[hi]
                    oj = outputs2[hj]
                    xi = oi.float()
                    xj = oj.float()
                    if xi.numel() > 1 and xj.numel() > 1:
                        corrmat = torch.corrcoef(torch.stack([xi, xj], dim=0))
                        rho = corrmat[0, 1]
                        weights = self.corr_weights.get(hi, {}).get(hj, 1.0)
                        tgt = torch.tensor(target_corr, device=device, dtype=rho.dtype)
                        total_loss = total_loss + weights * F.mse_loss(rho, tgt)

        outputs3: Dict[str, torch.Tensor] = {}
        if inputs3:
            outputs3 = model(**inputs3)
            for head in active_heads:
                objective = self.head_objectives.get(head)
                if objective is None:
                    continue
                mask = self._head_mask(inputs3.get("active_heads"), head, device)
                if mask is None or mask.sum() == 0:
                    continue
                if labels3 is None:
                    continue
                subset_labels = labels3[mask].flatten()
                valid = self._finite_mask(subset_labels)
                if valid.sum() == 0:
                    continue
                refined_mask = mask.clone()
                refined_mask[mask] = valid
                loss = objective.compute_triplet(
                    self,
                    outputs3,
                    refined_mask,
                    subset_labels[valid],
                )
                if loss is not None:
                    total_loss = total_loss + loss

            for hi, row in self.corr_labels.items():
                for hj, tgt_corr in row.items():
                    pos_i = outputs3.get(f"{hi}_pos_similarity")
                    neg_i = outputs3.get(f"{hi}_neg_similarity")
                    pos_j = outputs3.get(f"{hj}_pos_similarity")
                    neg_j = outputs3.get(f"{hj}_neg_similarity")
                    if pos_i is None or neg_i is None or pos_j is None or neg_j is None:
                        continue
                    vi = torch.cat([pos_i, neg_i], dim=0)
                    vj = torch.cat([pos_j, neg_j], dim=0)
                    xi = vi.float()
                    xj = vj.float()
                    if xi.numel() > 1 and xj.numel() > 1:
                        corrmat = torch.corrcoef(torch.stack([xi, xj], dim=0))
                        rho = corrmat[0, 1]
                        weights = self.corr_weights.get(hi, {}).get(hj, 1.0)
                        tgt = torch.tensor(tgt_corr, device=device, dtype=rho.dtype)
                        total_loss = total_loss + weights * F.mse_loss(rho, tgt)

        full_outputs = partitioner.merge(
            {
                "single": outputs1,
                "pair": outputs2,
                "triplet": outputs3,
            },
            device=device,
        )
        
        # 5) 欠けているキーをすべて埋める
        #    regression/binary head は key = head
        #    contrastive head は pos/neg/anchor_prob の３つ
        for head, cfg in self.classifier_configs.items():
            if head not in full_outputs:
                # スコア・tensor shape=(bsz,)
                full_outputs[head] = torch.full((bsz,), float("nan"), device=device)
            
            # for suffix in ("pos_similarity", "neg_similarity", "anchor_embedding", "positive_embedding", "negative_embedding"):
            if cfg["objective"] == "regression" or cfg["objective"] == "binary_classification":
                for suffix in ("pos_similarity", "neg_similarity"):
                    key = f"{head}_{suffix}"
                    # anchor_prob は (bsz, num_classes) の可能性があるので次元数だけでも合わせたいですが、
                    # ここではシンプルに (bsz, ) or (bsz,1) で nan を入れておく
                    if key not in full_outputs:
                        full_outputs[key] = torch.full((bsz,), float("nan"), device=device)
                    
            if cfg["objective"] == "contrastive_logit":
                for suffix in ("anchor_prob", "positive_prob", "negative_prob"):
                    key = f"{head}_{suffix}"
                    if key not in full_outputs:
                        nclass = cfg.get("output_dim", None)
                        if nclass is None:
                            raise ValueError(f"Missing num_labels in config for {head}")
                        shape = (bsz, nclass)

                        full_outputs[key] = torch.full(
                            shape,
                            float("nan"),
                            dtype=torch.float32,
                            device=device
                        )
            # for suffix in ("_pos_similarity", "_neg_similarity", "_anchor_embedding", "_positive_embedding", "_negative_embedding", ""):
            for suffix in ("_pos_similarity", "_neg_similarity", ""):
                key = f"original{suffix}"
                # anchor_prob は (bsz, num_classes) の可能性があるので次元数だけでも合わせたいですが、
                # ここではシンプルに (bsz, ) or (bsz,1) で nan を入れておく
                if key not in full_outputs:
                    full_outputs[key] = torch.full((bsz,), float("nan"), device=device)
        
        if not return_outputs:
            return total_loss

        return total_loss, full_outputs


    def _compute_info_nce_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        tau: float,
        max_pos_pairs: int,
        max_neg_pairs: int,
    ) -> Optional[torch.Tensor]:
        """
        デフォルト（max_pos_pairs <= 0 かつ max_neg_pairs <= 0）では
        バッチ内の全ペアを用いた InfoNCE を計算する。
        max_pos_pairs / max_neg_pairs が正の場合のみ、その上限でサンプリング近似を行う。
        """
        if embeddings.size(0) < 2:
            return None

        device = embeddings.device
        eps = 1e-8

        # 類似度はコサイン（L2正規化）
        z = F.normalize(embeddings.float(), dim=1)     # (B, D)
        B = z.size(0)

        lbls = labels.to(device=device, dtype=torch.long).view(-1)  # (B,)

        # pairwise cosine similarity
        sim = z @ z.t()                                # (B, B)
        logits = sim / max(tau, eps)

        # マスク作成
        self_mask = torch.eye(B, dtype=torch.bool, device=device)            # 自己
        pos_mask = (lbls.unsqueeze(0) == lbls.unsqueeze(1)) & (~self_mask)   # 同ラベル & 非自己
        neg_mask = (~pos_mask) & (~self_mask)

        if not pos_mask.any():
            return None

        # --- フルペア（デフォルト） ---
        if (max_pos_pairs is None or max_pos_pairs <= 0) and (max_neg_pairs is None or max_neg_pairs <= 0):
            # 分母：自分以外すべて
            exp_logits = torch.exp(logits) * (~self_mask)          # 自己は0に
            denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)  # (B, 1)

            # 分子：ポジティブ全てを合計（multi-positive 版）
            numer = (exp_logits * pos_mask).sum(dim=1, keepdim=True)

            valid = pos_mask.any(dim=1)  # ポジティブを持つアンカーのみ
            if not valid.any():
                return None

            log_prob = torch.log(numer.clamp_min(eps)) - torch.log(denom)
            loss = -(log_prob.squeeze(1)[valid]).mean()
            return loss

        # --- 上限付きサンプリング（互換用） ---
        losses: List[torch.Tensor] = []
        for i in range(B):
            pos_idx = torch.nonzero(pos_mask[i], as_tuple=True)[0]
            if pos_idx.numel() == 0:
                continue
            neg_idx = torch.nonzero(neg_mask[i], as_tuple=True)[0]

            # 上限が与えられていればランダム抽出
            if max_pos_pairs is not None and max_pos_pairs > 0 and pos_idx.numel() > max_pos_pairs:
                perm = torch.randperm(pos_idx.numel(), device=device)[:max_pos_pairs]
                pos_idx = pos_idx[perm]
            if max_neg_pairs is not None and max_neg_pairs > 0 and neg_idx.numel() > max_neg_pairs:
                perm = torch.randperm(neg_idx.numel(), device=device)[:max_neg_pairs]
                neg_idx = neg_idx[perm]

            # 分母：選択した正負の和、分子：選択した正のみの和
            sel_idx = torch.cat([pos_idx, neg_idx], dim=0) if neg_idx.numel() > 0 else pos_idx
            row = logits[i, sel_idx]
            exp_row = torch.exp(row)

            numer = exp_row[:pos_idx.numel()].sum()
            denom = exp_row.sum().clamp_min(eps)

            losses.append(-(torch.log(numer.clamp_min(eps)) - torch.log(denom)))

        if not losses:
            return None

        return torch.stack(losses).mean()

    def evaluation_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        print(f"eval loss: {loss}")
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

    def _head_mask(self, active_heads_field: Any, head: str, device: torch.device) -> Optional[torch.Tensor]:
        if active_heads_field is None:
            return None
        if isinstance(active_heads_field, torch.Tensor):
            if active_heads_field.dtype == torch.bool:
                return active_heads_field.to(device)
            if active_heads_field.dtype in (torch.int32, torch.int64, torch.long):
                idx = self.head2idx.get(head)
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

    @staticmethod
    def _finite_mask(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype.is_floating_point:
            return torch.isfinite(tensor)
        return torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)
