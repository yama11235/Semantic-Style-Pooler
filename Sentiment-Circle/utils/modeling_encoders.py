import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoModel
import logging
from modeling_config import (
    build_classifiers,
    load_classifiers
)
import os
from sentence_batch_utils import BatchPartitioner
from contextlib import nullcontext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None


def _sentence_kwargs(batch: dict, suffix: str = "") -> dict:
    keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "inputs_embeds",
    ]
    payload = {}
    for key in keys:
        name = key if not suffix else f"{key}_{suffix}"
        payload[key] = batch.get(name)
    return payload


class SentencePath:
    kind = ""

    def __init__(self, model: "BiEncoderForClassification") -> None:
        self.model = model

    def run_full(self, batch: dict, extra_kwargs: dict) -> dict:
        return self._forward(batch, extra_kwargs)

    def run_partition(
        self,
        batch: dict,
        partitioner: BatchPartitioner,
        device: torch.device,
        extra_kwargs: dict,
    ) -> dict:
        subset = partitioner.slice(batch, self.kind, device=device)
        if not subset:
            return {}
        return self._forward(subset, extra_kwargs)

    def _forward(self, batch: dict, extra_kwargs: dict) -> dict:  # pragma: no cover - abstract helper
        raise NotImplementedError


class SingleSentencePath(SentencePath):
    kind = "single"

    def _forward(self, batch: dict, extra_kwargs: dict) -> dict:
        args = _sentence_kwargs(batch)
        return self.model.encode(**args)


class PairSentencePath(SentencePath):
    kind = "pair"

    def _forward(self, batch: dict, extra_kwargs: dict) -> dict:
        first = _sentence_kwargs(batch)
        second = _sentence_kwargs(batch, "2")
        return self.model._forward_binary(
            first["input_ids"],
            first["attention_mask"],
            first["token_type_ids"],
            first["position_ids"],
            first["head_mask"],
            first["inputs_embeds"],
            second["input_ids"],
            second["attention_mask"],
            second["token_type_ids"],
            second["position_ids"],
            second["head_mask"],
            second["inputs_embeds"],
            **extra_kwargs,
        )


class TripletSentencePath(SentencePath):
    kind = "triplet"

    def _forward(self, batch: dict, extra_kwargs: dict) -> dict:
        first = _sentence_kwargs(batch)
        second = _sentence_kwargs(batch, "2")
        third = _sentence_kwargs(batch, "3")
        return self.model.triplet_encode(
            first["input_ids"],
            first["attention_mask"],
            first["token_type_ids"],
            first["position_ids"],
            first["head_mask"],
            first["inputs_embeds"],
            second["input_ids"],
            second["attention_mask"],
            second["token_type_ids"],
            second["position_ids"],
            second["head_mask"],
            second["inputs_embeds"],
            third["input_ids"],
            third["attention_mask"],
            third["token_type_ids"],
            third["position_ids"],
            third["head_mask"],
            third["inputs_embeds"],
            **extra_kwargs,
        )

def calculate_similarity(name, output1, output2, classifier_configs):
    """
    指定された分類器設定に応じてoutput1とoutput2の類似度・距離を計算します。
    
    Args:
        name: 分類器名（str）
        output1, output2: (N, D) あるいは (N, ) のテンソル
        classifier_configs: 分類器ごとの設定辞書
    
    Returns:
        similarity: (N, ) のテンソル（コサイン類似度/ユークリッド距離/内積など）
    """
    config = classifier_configs[name]
    distance_type = config["distance"]
    objective = config.get("objective", None)

    if distance_type == "cosine":
        # コサイン類似度
        similarity = torch.nn.functional.cosine_similarity(output1, output2, dim=1).to(output1.dtype)
        if objective == "binary_classification":
            similarity = torch.sigmoid(similarity).to(output1.dtype)
    elif distance_type == "euclidean":
        # ユークリッド距離
        similarity = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
    elif distance_type == "dot_product":
        # 内積
        similarity = torch.sum(output1 * output2, dim=1)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")

    return similarity

def calculate_pos_neg_similarity(
    name: str,
    output1: torch.Tensor,
    output2: torch.Tensor,
    output3: torch.Tensor,
    classifier_configs: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    anchor-output1 と positive/output2, negative/output3 の類似度（または距離）を計算する。

    Args:
        name: 分類器名
        output1: anchor の出力 (N, D)
        output2: positive ペアの出力 (N, D)
        output3: negative ペアの出力 (N, D)
        classifier_configs: 分類器ごとの設定辞書

    Returns:
        pos_similarity, neg_similarity: それぞれ (N,) のテンソル
    """
    config = classifier_configs[name]
    dist = config["distance"]

    if dist == "cosine":
        # コサイン類似度
        pos_sim = F.cosine_similarity(output1, output2, dim=1).to(output1.dtype)
        neg_sim = F.cosine_similarity(output1, output3, dim=1).to(output1.dtype)

    elif dist == "euclidean":
        # ユークリッド距離
        pos_sim = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        neg_sim = torch.sqrt(torch.sum((output1 - output3) ** 2, dim=1))

    elif dist == "dot_product":
        # 内積
        pos_sim = torch.sum(output1 * output2, dim=1).to(output1.dtype)
        neg_sim = torch.sum(output1 * output3, dim=1).to(output1.dtype)

    else:
        raise ValueError(f"Unknown distance type: {dist}")

    return pos_sim, neg_sim

class QuadrupletLoss:
    def __init__(self, distance_function, margin=1.0):
        'A cosine distance margin quadruplet loss'
        self.margin = margin
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg = self.distance_function(neg1, neg2)
        loss = torch.clamp_min(self.margin + dist_pos - dist_neg, 0)
        return loss.mean()


# Pooler class. Copied and adapted from SimCSE code
class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last', 'last'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs, target_layer=-1):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        dtype = last_hidden.dtype
        # print(f"dtype in Pooler: {dtype}")

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((hidden_states[target_layer] * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)).to(dtype)
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result.to(dtype)
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result.to(dtype)
        elif self.pooler_type == 'last':
            # attention_mask の 1 の数から「最後の有効トークン位置」を計算
            # lengths: (batch_size,)  各サンプルの非パディング長
            lengths = attention_mask.sum(dim=1) - 1

            # 取り出し用のバッチインデックス
            batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
            # hidden_states[target_layer]: (batch_size, seq_len, hidden_size)
            # [batch_idx, lengths] で (batch_size, hidden_size) が得られる
            return hidden_states[target_layer][batch_idx, lengths].to(dtype)
        else:
            raise NotImplementedError
  
class BiEncoderForClassification(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''
    def __init__(self, model_config, classifier_configs):
        """
        Args:
            model_config: Model configuration.
            classifier_configs: Classifier configurations. {classifier_name: classifier_config_dict}
        """
        super().__init__(model_config)
        self.config = model_config
        self.model_path = (
            getattr(model_config, "model_name_or_path", None)
            or getattr(model_config, "name_or_path", None)
        )
        self.classifier_configs = classifier_configs
        
        if getattr(model_config, "device_map", None) is None:
            self.backbone = AutoModel.from_pretrained(
                self.model_path,
                from_tf=bool(self.model_path is not None and '.ckpt' in self.model_path),
                config=model_config,
                cache_dir=getattr(model_config, "cache_dir", None),
                revision=getattr(model_config, "model_revision", None),
                use_auth_token=True if getattr(model_config, "use_auth_token", None) else None,
                attn_implementation=getattr(model_config, "attn_implementation", "eager"),
                add_pooling_layer=False,
            ).base_model.eval()  # 評価モードに設定
            
        else:
            self.backbone = AutoModel.from_pretrained(
                self.model_path,
                # output_loading_info=True,
                device_map=getattr(model_config, "device_map", None),
                from_tf=bool(self.model_path is not None and '.ckpt' in self.model_path),
                config=model_config,
                cache_dir=getattr(model_config, "cache_dir", None),
                revision=getattr(model_config, "model_revision", None),
                use_auth_token=True if getattr(model_config, "use_auth_token", None) else None,
                attn_implementation=getattr(model_config, "attn_implementation", "eager"),
                add_pooling_layer=False,
            ).base_model
            self.backbone.eval()  # 評価モードに設定
            
        self.pooler = Pooler(model_config.pooler_type)
        if model_config.pooler_type in {'avg_first_last', 'avg_top2'} or classifier_configs is not None:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False

        self.embedding_classifiers = nn.ModuleDict()
        self.clf_configs = {}

        if self.classifier_configs:
            # Build classifiers via factory
            self.embedding_classifiers, self.clf_configs = build_classifiers(
                self.classifier_configs, model_config
            )
            self.classifier_save_directory = getattr(
                model_config, 'classifier_save_directory', None
            )
           
        self.post_init()
        self._paths = {
            "single": SingleSentencePath(self),
            "pair": PairSentencePath(self),
            "triplet": TripletSentencePath(self),
        }
        # 分類器をbackboneと同じdevice・dtypeに移動
        # print(f"dtype: {next(self.backbone.parameters()).dtype}, device: {next(self.backbone.parameters()).device}")
        backbone_device = next(self.backbone.parameters()).device
        for name, classifier in self.embedding_classifiers.items():
            classifier.to(device=backbone_device)
            classifier.to(dtype=next(self.backbone.parameters()).dtype)
            # # ダミー入力（バッチサイズ1, 入力次元2560、backboneと同じdevice・分類器のdtypeで作成）
            dummy_input = torch.randn(1, 1024, device=backbone_device, dtype=next(classifier.parameters()).dtype)

            # # 入力のdtype
            # print("入力のdtype:", dummy_input.dtype)

            # # 出力を得る
            output = classifier(dummy_input)

            # # 出力のdtype
            # print("出力のdtype:", output.dtype if isinstance(output, torch.Tensor) else type(output))



    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        # 以下、3文目は (bsz, seq_len) のテンソル or None
        input_ids_3=None,
        attention_mask_3=None,
        token_type_ids_3=None,
        position_ids_3=None,
        head_mask_3=None,
        inputs_embeds_3=None,
        **kwargs,
    ):
        device = next(self.backbone.parameters()).device

        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "token_type_ids_2": token_type_ids_2,
            "position_ids_2": position_ids_2,
            "head_mask_2": head_mask_2,
            "inputs_embeds_2": inputs_embeds_2,
            "input_ids_3": input_ids_3,
            "attention_mask_3": attention_mask_3,
            "token_type_ids_3": token_type_ids_3,
            "position_ids_3": position_ids_3,
            "head_mask_3": head_mask_3,
            "inputs_embeds_3": inputs_embeds_3,
        }
        extra_kwargs = dict(kwargs)
        partitioner = BatchPartitioner(
            attention_mask=attention_mask,
            attention_mask_2=attention_mask_2,
            attention_mask_3=attention_mask_3,
        )

        uniform = partitioner.uniform_kind()
        if uniform:
            outputs = self._paths[uniform].run_full(batch_inputs, extra_kwargs)
            if uniform == "single":
                embeddings = outputs.get("original")
                if embeddings is None:
                    raise ValueError("encode() did not return 'original' embeddings for single-sentence input.")
                outputs.setdefault("embeddings", embeddings)
            # print(f"dtype of outputs: { {k: v.dtype for k, v in outputs.items()} }")
            return outputs

        outputs_by_kind = {}
        for kind, path in self._paths.items():
            outputs = path.run_partition(batch_inputs, partitioner, device, extra_kwargs)
            if kind == "single" and outputs and "embeddings" not in outputs:
                embeddings = outputs.get("original")
                if embeddings is not None:
                    outputs["embeddings"] = embeddings
            outputs_by_kind[kind] = outputs

        return partitioner.merge(outputs_by_kind, device=device)

    # ── それぞれのパスで“本体”を切り出すヘルパー例 ──
    def _forward_binary(
        self,
        input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
        input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, head_mask_2, inputs_embeds_2,
        **kwargs
    ):
        bsz = input_ids.shape[0]
        input_ids = concat_features(input_ids, input_ids_2)
        attention_mask = concat_features(attention_mask, attention_mask_2)
        token_type_ids = concat_features(token_type_ids, token_type_ids_2)
        position_ids = concat_features(position_ids, position_ids_2)
        head_mask = concat_features(head_mask, head_mask_2)
        inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2)
        # print(f"input_ids: {input_ids}, attention_mask: {attention_mask}")
        # print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
        # print(f"device: input_ids {input_ids.device}, attention_mask {attention_mask.device}")
        # print(f"device: backbone {self.backbone.device}")
        outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=self.output_hidden_states,
                )
        # print(f"outputs {outputs}")
        # nanチェック
        if torch.isnan(outputs.last_hidden_state).any():
            raise ValueError("NaN detected in outputs.last_hidden_state")
        outputs_dict = {}
        if self.classifier_configs:
            for name, classifier in self.embedding_classifiers.items():
                target_layer = int(self.classifier_configs[name]["layer"])

                # （通常、hidden_states[0] が入力埋め込み、以降が各層の出力となる）
                # 指定層の出力に対して pooler を適用（例：[CLS] トークンの抽出等）
                pooled_features = self.pooler(attention_mask, outputs, target_layer)  # shape: (2*bsz, hidden_size)
                # print(f"pooled_features: {pooled_features}")
                # print(f"pooled_features: {pooled_features.shape}, dtype: {pooled_features.dtype}, device: {pooled_features.device}")
                # siamese 用に、2*bsz の出力をそれぞれ前半と後半に分割する
                # print(f"bsz: {bsz}, pooled_features shape: {pooled_features.shape}")
                features1, features2 = torch.split(pooled_features, bsz, dim=0)
                # print(f"features1: {features1}, features2: {features2}")

                if self.classifier_configs[name]["type"] != "contrastive_logit":
                    # 各分類器に対して、対象の層のプール済み埋め込みを入力して出力を得る
                    output1 = classifier(features1)  # sentence1用
                    output2 = classifier(features2)  # sentence2用
                    similarity = calculate_similarity(name, output1, output2, self.classifier_configs)
                    # dummy output1, output2, similarity
                    # output1 = torch.randn(bsz, 1, device=features1.device, dtype=features1.dtype)
                    # output2 = torch.randn(bsz, 1, device=features2.device, dtype=features2.dtype)
                    # similarity = torch.zeros(bsz, device=features1.device, dtype=features1.dtype)
                    # print(f"output1: {output1}, output2: {output2}")
                    # print(f"similarity: {similarity}")
                else:
                    output1 = classifier.encode(features1)  # sentence1用
                    output2 = classifier.encode(features2)  # sentence2用
                    # 中間層の埋め込み表現を抽出
                    similarity = calculate_similarity(name, output1, output2, self.classifier_configs)
                        
                outputs_dict[name] = similarity
        
        features = self.pooler(attention_mask, outputs)
        features_1, features_2 = torch.split(features, bsz, dim=0)  # [sentence1], [sentence2]
        logits = cosine_similarity(features_1, features_2, dim=1).to(features_1.dtype)
        # print(f"features1: {features1}")
        # print(f"features2: {features2}")
        # print(f"output1: {output1}")
        # print(f"output2: {output2}")
        # print(f"similarity: {similarity}")
        outputs_dict["original"] = logits
        
        # outputs は dataclass なので、その中の Tensor から dtype を取る
        # print(f"outputs.last_hidden_state dtype: {outputs.last_hidden_state.dtype}")
        # if hasattr(outputs, "hidden_states"):
        #     print(f"outputs.hidden_states[0] dtype: {outputs.hidden_states[0].dtype}")
        # print(f"features dtype: {features.dtype}")
        # print(f"features_1 dtype: {features_1.dtype}")
        # print(f"features_2 dtype: {features_2.dtype}")
        # print(f"similarity dtype: {similarity.dtype}")
        # print(f"logits dtype: {logits.dtype}")
        # print("=====================================")
        return outputs_dict

        
    def encode(            
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            ):
            """        
            Returns the embedding for single sentence.
            """
            outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=self.output_hidden_states,
                )

            outputs_dict = {}
            if self.classifier_configs:
                for name, classifier in self.embedding_classifiers.items():
                    target_layer = int(self.classifier_configs[name]["layer"])
                    pooled_features = self.pooler(attention_mask, outputs, target_layer)
                    # print(f"pooled_features dtype for classifier {name}: {pooled_features.dtype}")
                    # print(classifier)
                    embedding = classifier.encode(pooled_features).to(pooled_features.dtype)
                    # print(f"embedding dtype for classifier {name}: {embedding.dtype}")
                    outputs_dict[name] = embedding
            
            features = self.pooler(attention_mask, outputs)

            outputs_dict["original"] = features  # original embedding
            outputs_dict.setdefault("embeddings", features)
            return outputs_dict
    
    def classify(            
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            ):
            """        
            Returns the embedding for single sentence.
            """ 
            outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=self.output_hidden_states,
                    )
                
            outputs_dict = {}
            if self.classifier_configs:
                for name, classifier in self.embedding_classifiers.items():
                    if self.classifier_configs[name]["type"] != "contrastive_logit":
                        continue  # contrastive_logit のみを対象とする

                    target_layer = int(self.classifier_configs[name]["layer"])
                    pooled_features = self.pooler(attention_mask, outputs, target_layer)

                    embedding, prob = classifier(pooled_features)
                    outputs_dict[f"{name}_prob"] = prob  # probability for classification

            return outputs_dict
    
    def triplet_encode(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_ids_2=None,
            attention_mask_2=None,
            token_type_ids_2=None,
            position_ids_2=None,
            head_mask_2=None,
            inputs_embeds_2=None,
            input_ids_3=None,
            attention_mask_3=None,
            token_type_ids_3=None,
            position_ids_3=None,
            head_mask_3=None,
            inputs_embeds_3=None,
            **kwargs,
            ):
        if input_ids is None or input_ids_2 is None or input_ids_3 is None:
            raise ValueError("input_ids, input_ids_2, input_ids_3 must not be None")
        bsz1 = input_ids.shape[0]
        bsz2 = input_ids_2.shape[0]
        bsz3 = input_ids_3.shape[0]  

        input_ids = concat_features(input_ids, input_ids_2, input_ids_3)
        attention_mask = concat_features(attention_mask, attention_mask_2, attention_mask_3)
        token_type_ids = concat_features(token_type_ids, token_type_ids_2, token_type_ids_3)
        position_ids = concat_features(position_ids, position_ids_2, position_ids_3)
        head_mask = concat_features(head_mask, head_mask_2, head_mask_3)
        inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2, inputs_embeds_3)

        outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
            )
        outputs_dict = {}
        if self.classifier_configs:
            for name, classifier in self.embedding_classifiers.items():                
                target_layer = int(self.classifier_configs[name]["layer"])
                pooled_features = self.pooler(attention_mask, outputs, target_layer)  # shape: (3*bsz, hidden_size)

                # siamese 用に、3*bsz の出力をそれぞれ前半、中間、後半に分割する
                features1, features2, features3 = torch.split(pooled_features, [bsz1, bsz2, bsz3], dim=0)

                if self.classifier_configs[name]["type"] != "contrastive_logit":
                    output1 = classifier(features1)  # sentence1用
                    output2 = classifier(features2)  # sentence2用
                    output3 = classifier(features3)  # sentence3用
                    pos_similarity, neg_similarity = calculate_pos_neg_similarity(
                        name, output1, output2, output3, self.classifier_configs
                    )
                    outputs_dict[f"{name}_pos_similarity"] = pos_similarity
                    outputs_dict[f"{name}_neg_similarity"] = neg_similarity

                else:
                    output1, prob1 = classifier(features1)  # sentence1用
                    output2, prob2 = classifier(features2)  # sentence2用
                    output3, prob3 = classifier(features3)  # sentence3用
                    # 中間層の埋め込み表現を抽出
                    pos_similarity, neg_similarity = calculate_pos_neg_similarity(
                        name, output1, output2, output3, self.classifier_configs
                    )

                    outputs_dict[f"{name}_pos_similarity"] = pos_similarity
                    outputs_dict[f"{name}_neg_similarity"] = neg_similarity
                    outputs_dict[f"{name}_anchor_prob"] = prob1
                    outputs_dict[f"{name}_positive_prob"] = prob2
                    outputs_dict[f"{name}_negative_prob"] = prob3
                    # outputs_dict[f"{name}_anchor_embedding"] = output1
                    # outputs_dict[f"{name}_positive_embedding"] = output2
                    # outputs_dict[f"{name}_negative_embedding"] = output3
                
        
        features = self.pooler(attention_mask, outputs)  # shape: (3*bsz, hidden_size)

        features_1, features_2 , features_3 = torch.split(features, [bsz1, bsz2, bsz3], dim=0)  # [sentence1], [sentence2], [sentence3]
        logits1 = cosine_similarity(features_1, features_2, dim=1).to(features_1.dtype)  # sentence1 と sentence2 の類似度
        logits2 = cosine_similarity(features_1, features_3, dim=1).to(features_1.dtype)  # sentence1 と sentence3 の類似度
        outputs_dict["original_pos_similarity"] = logits1
        outputs_dict["original_neg_similarity"] = logits2
        # outputs_dict["original_anchor_embedding"] = features_1
        # outputs_dict["original_positive_embedding"] = features_2
        # outputs_dict["original_negative_embedding"] = features_3
        
        return outputs_dict    

    def save_pretrained(self, model_save_directory, **kwargs):
        os.makedirs(model_save_directory, exist_ok=True)
        classifier_save_directory = self.classifier_save_directory if self.classifier_save_directory else model_save_directory
        os.makedirs(classifier_save_directory, exist_ok=True)
        # Backboneの保存
        if not self.config.freeze_encoder:
            super().save_pretrained(model_save_directory, **kwargs)
        self.config.save_pretrained(model_save_directory)
        # 各分類器の保存
        for name, module in self.embedding_classifiers.items():
            param_str = f"{self.classifier_configs[name]['type']}_layer:{self.classifier_configs[name]['layer']}_dim:{self.classifier_configs[name]['output_dim']}"
            save_path = os.path.join(classifier_save_directory, param_str,f"{name}_classifier.bin")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(module.state_dict(), save_path)
            self.clf_configs[name].save_pretrained(classifier_save_directory, name)
        
        
    @classmethod
    def from_pretrained(cls,
            pretrained_model_name_or_path,
            model_config,
            classifier_save_directory=None,
            classifier_configs=None,
            classifier_freeze=None,
            **kwargs,
        ):
        # --- 1) __init__ を使って、Hub から backbone を読み込みつつ分類ヘッドだけ作る ---
        model = cls(model_config, classifier_configs)
        
        for name, param in model.backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"{name} に NaN が含まれています")
        
        # --- 2) 分類ヘッドの重みだけ別ディレクトリからロード ---
        if classifier_save_directory and classifier_configs:
            loaded_heads = load_classifiers(
                classifier_configs,
                model_config,
                classifier_save_directory,
                classifier_freeze,
            )
            model.embedding_classifiers = loaded_heads

        return model
