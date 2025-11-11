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
    compute_sentence_partitions,
    extract_unique_strings,
    flatten_strings,
    merge_indexed_outputs,
    slice_input_batch,
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
        bsz     = inputs["input_ids"].size(0)
        # print(f"inputs size: {inputs['input_ids'].size()}")
        # device  = model.device
        # device = inputs["input_ids"].device
        device = next(model.parameters()).device
        torch.cuda.reset_peak_memory_stats(device)
        partitions = compute_sentence_partitions(
            attention_mask=inputs["attention_mask"],
            attention_mask_2=inputs.get("attention_mask_2"),
            attention_mask_3=inputs.get("attention_mask_3"),
        )
        idx1 = partitions["idx_single"]
        idx2 = partitions["idx_pair"]
        idx3 = partitions["idx_triplet"]

        inputs1 = slice_input_batch(inputs, idx1, device)
        inputs2 = slice_input_batch(inputs, idx2, device)
        inputs3 = slice_input_batch(inputs, idx3, device)

        # サブバッチの教師損失（active_heads ごと）
        # print(f"active_heads: {inputs['active_heads']}")
        active_heads = extract_unique_strings(inputs["active_heads"])
        labels1 = inputs1.get("labels", {})
        labels2 = inputs2.get("labels", {})
        labels3 = inputs3.get("labels", {})

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # print(f"idx2: {idx2}, idx3: {idx3}, bsz: {bsz}")

        # print(f"[Before outputs2] alloc={mem_alloc:.2f} GB, reserved={mem_reserved:.2f} GB, peak={mem_peak:.2f} GB")

        # B) 1文サブバッチの encode() → embedding から損失ペアを構成して計算
        outputs1 = {}
        if idx1.numel() > 0:
            outputs1 = model(**inputs1)
            embeddings1 = outputs1["embeddings"]  # (bsz1, dim)
            for head in active_heads:
                head_idx = [True if h == head else False for h in flatten_strings(inputs1["active_heads"])]
                if head_idx.count(True) == 0:
                    continue
                sub_emb = embeddings1[head_idx]  # (sub_bsz, dim)
                sub_tgt = labels1[head_idx].flatten()  # (sub_bsz,)
                
                """
                id2label = {0: "negative", 1: "neutral", 2: "positive"...} - 感情ラベルとidの対応
                angle_map = {"negative": 180.0, "neutral": 90.0, "positive": 0.0} - 感情ラベルと角度の対応
                sub_emb: (sub_bsz, dim) - 埋め込みベクトル
                sub_tgt: (sub_bsz,) - 文章の感情ラベルのid
                """
                
                # print(f"Processing head: {head}, embedding: {sub_emb}, target: {sub_tgt}")
                assert sub_emb.size(0) == sub_tgt.size(0), "Embeddings and targets must have the same batch size."
                mask    = ~torch.isnan(sub_tgt)
                if mask.sum() > 0:
                    sub_emb = sub_emb[mask]
                    sub_tgt = sub_tgt[mask]
                    if sub_emb.size(0) < 2:
                        continue
                    if self.classifier_configs[head]["objective"] == "infoNCE":
                        params = self.info_nce_params.get(head, {})
                        tau = float(params.get("tau", 1.0))
                        pos_pairs = int(params.get("pos_pairs", 0))
                        neg_pairs = int(params.get("neg_pairs", 0))
                        if tau <= 0:
                            tau = 1.0
                        info_nce_loss = self._compute_info_nce_loss(
                            embeddings=sub_emb,
                            labels=sub_tgt.long(),
                            tau=tau,
                            max_pos_pairs=max(pos_pairs, 0),
                            max_neg_pairs=max(neg_pairs, 0),
                        )
                        if info_nce_loss is not None:
                            total_loss = total_loss + info_nce_loss
                    
                    elif self.classifier_configs[head]["objective"] == "angleNCE":
                        pass


        # C) 2文サブバッチの教師損失（active_heads ごと）
        outputs2 = {}
        if idx2.numel() > 0:
            outputs2 = model(**inputs2)
            # print(f"outputs2 keys: {outputs2.keys()}")
        
            # print(f"[After outputs2] alloc={mem_alloc:.2f} GB, reserved={mem_reserved:.2f} GB, peak={mem_peak:.2f} GB")

            for head in active_heads:
                head_idx = [True if h == head else False for h in flatten_strings(inputs2["active_heads"])]
                if head_idx.count(True) == 0:
                    continue
                sub_out = outputs2.get(head)[head_idx]
                sub_tgt = labels2[head_idx].flatten()
                # print(f"input2: {inputs2}")
                # print(f"Processing head: {head}, output: {sub_out}, target: {sub_tgt}")
                assert sub_out.size(0) == sub_tgt.size(0), "Outputs and targets must have the same batch size."
                mask    = ~torch.isnan(sub_tgt)
                if mask.sum() > 0:
                    total_loss = total_loss + self.loss_fns[head](
                        sub_out[mask].to(torch.float32),
                        sub_tgt[mask].to(torch.float32),
                    )

            # D) 2文サブバッチ全体での相関ペナルティ
            for hi, row in self.corr_labels.items():
                for hj, target_corr in row.items():
                    # 両方とも 2文教師損失対象ではなくても、ここは常に計算
                    if hi not in outputs2 or hj not in outputs2:
                        continue
                    oi = outputs2[hi]
                    oj = outputs2[hj]
                    # pearson 相関を使用
                    xi = oi.float()
                    xj = oj.float()
                    if xi.numel() > 1 and xj.numel() > 1:
                        # 2×batch_size の行列を作って相関行列を計算
                        corrmat = torch.corrcoef(torch.stack([xi, xj], dim=0))
                        rho     = corrmat[0, 1]
                        # rho = pearsonr_torch(xi, xj)
                        
                        # spearman 相関の場合
                        # ri = torch.argsort(torch.argsort(oi))
                        # rj = torch.argsort(torch.argsort(oj))
                        # corrmat = torch.corrcoef(torch.stack([ri.float(), rj.float()]))
                        # rho     = corrmat[0,1]    

                        weights = self.corr_weights.get(hi, {}).get(hj, 1.0)
                        tgt     = torch.tensor(target_corr, device=device, dtype=rho.dtype)
                        # print(f"Processing correlation: {hi} vs {hj}, {len(xi)}, {len(xj)}, corr: {rho.item()}, target: {tgt.item()}")
                        total_loss = total_loss + weights * F.mse_loss(rho, tgt)

        # E) 3文サブバッチの contrastive_logit 損失（active_heads ごと）
        outputs3 = {}
        if idx3.numel() > 0:
            outputs3 = model(**inputs3)
            # print(f"outputs3 keys: {outputs3.keys()}")
            
            # print(f"[Before outputs3] alloc={mem_alloc:.2f} GB, reserved={mem_reserved:.2f} GB, peak={mem_peak:.2f} GB")

            # (1) CE & triplet
            for head in active_heads:
                head_idx = [True if h == head else False for h in flatten_strings(inputs3["active_heads"])]
                if head_idx.count(True) == 0:
                    continue
                # anchor_prob の CE
                prob_key = f"{head}_anchor_prob"
                logits = outputs3[prob_key][head_idx].to(torch.float32)
                tgt    = labels3[head_idx].to(device).long()
                # print(f"Processing head: {head}, logits: {logits}, target: {tgt}")
                assert logits.size(0) == tgt.size(0), "Logits and targets must have the same batch size."
                total_loss = total_loss + self.gamma[head] * F.cross_entropy(logits, tgt)
                # margin triplet
                pos = outputs3[f"{head}_pos_similarity"][head_idx]
                neg = outputs3[f"{head}_neg_similarity"][head_idx]
                
                if pos is not None and neg is not None:
                    if self.classifier_configs[head].get("loss_type", "triplet") == "triplet":
                        if self.margin[head] > 0:
                            # close positive, far negative
                            triplet = F.relu(neg - pos + self.margin[head]).mean()
                        else:
                            # close negative, far positive
                            triplet = F.relu(pos - neg - self.margin[head]).mean()
                        total_loss = total_loss + self.alpha[head] * triplet

                    elif self.classifier_configs[head].get("loss_type", "triplet") == "infoNCE":
                        # 1) 温度パラメータ τ の取得（config で指定すると良い）
                        tau = self.classifier_configs[head].get("tau", 1.0)

                        # 2) logits を作成
                        # neg が 1D の場合は (batch,1) に reshape
                        pos = pos.view(-1, 1)
                        neg = neg.view(pos.size(0), -1)  # neg が (batch, N) のときはそのまま

                        logits = torch.cat([pos, neg], dim=1)  # (batch, 1+N)
                        logits = logits / tau

                        # 3) positive が index 0 の教師ラベル
                        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

                        # 4) cross-entropy で -log p_pos を計算
                        infoNCE_loss = F.cross_entropy(logits, labels)

                        # 5) 必要なら重み α を乗じる
                        total_loss = total_loss + self.alpha[head] * infoNCE_loss

            # F) contrastive heads 間の相関ペナルティ
            #    pos/neg を縦連結して順位 → Spearman 近似
            for hi, row in self.corr_labels.items():
                for hj, tgt_corr in row.items():
                    pos_i = outputs3[f"{hi}_pos_similarity"]
                    neg_i = outputs3[f"{hi}_neg_similarity"]
                    pos_j = outputs3[f"{hj}_pos_similarity"]
                    neg_j = outputs3[f"{hj}_neg_similarity"]

                    vi = torch.cat([pos_i, neg_i], dim=0)
                    vj = torch.cat([pos_j, neg_j], dim=0)

                    # ---- Pearson 相関を計算 ----
                    xi = vi.float()
                    xj = vj.float()
                    if xi.numel() > 1 and xj.numel() > 1:
                        # 2行 N 列 の行列にして相関行列を得る
                        corrmat = torch.corrcoef(torch.stack([xi, xj], dim=0))
                        rho     = corrmat[0, 1]
                        # rho = pearsonr_torch(xi, xj)

                        # spearman 相関を使用
                        # ri = torch.argsort(torch.argsort(vi))
                        # rj = torch.argsort(torch.argsort(vj))
                        # corrmat = torch.corrcoef(torch.stack([ri.float(), rj.float()]))
                        # rho     = corrmat[0,1]
                        
                        weights = self.corr_weights.get(hi, {}).get(hj, 1.0)
                        tgt     = torch.tensor(tgt_corr, device=device, dtype=rho.dtype)
                        # print(f"Processing correlation: {hi} vs {hj}, {len(xi)}, {len(xj)}, corr: {rho.item()}, target: {tgt.item()}")
                        total_loss = total_loss + weights * F.mse_loss(rho, tgt)

        # 4) サブバッチ出力を全バッチサイズにマージ
        full_outputs = merge_indexed_outputs(
            bsz,
            device,
            (idx1, outputs1),
            (idx2, outputs2),
            (idx3, outputs3),
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
        
        print(f"total_loss: {total_loss.item()}, full_outputs keys: {full_outputs.keys()}")
        
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
