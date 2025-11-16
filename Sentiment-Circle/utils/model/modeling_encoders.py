import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoModel
import logging
from .modeling_config import (
    build_classifiers,
    load_classifiers
)
from .nGPT_model import justnorm, _is_ngpt_block, _normalize_single_ngpt_block
import os
from utils.sentence_batch_utils import BatchPartitioner
from contextlib import nullcontext

from .classifier_strategies import (
    _ContrastiveLogitStrategy,
    _DefaultClassifierStrategy,
)
from .pooler import Pooler
from .sentence_paths import (
    PairSentencePath,
    SingleSentencePath,
    TripletSentencePath,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def concat_features(*features):
    filtered = [feature for feature in features if feature is not None]
    if not filtered:
        return None
    return torch.cat(filtered, dim=0)


class BiEncoderForClassification(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''
    _BACKBONE_ARG_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "inputs_embeds",
    )
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
            
        self.output_hidden_states = True
        self.cls_pooler = Pooler(pooler_type="cls")
        self.avg_pooler = Pooler(pooler_type="avg")
        self.max_pooler = Pooler(pooler_type="max")
        self.embedding_classifiers = nn.ModuleDict()
        self.clf_configs = {}
        self.classifier_strategy = _DefaultClassifierStrategy()

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
        self.backbone_device = next(self.backbone.parameters()).device
        for name, classifier in self.embedding_classifiers.items():
            classifier.to(device=self.backbone_device)
            classifier.to(dtype=next(self.backbone.parameters()).dtype)
            
        # --- nGPT-style Block の有無を検出して初期正規化 ------------------
        self.use_ngpt_blocks = self._detect_ngpt_blocks()
        if self.use_ngpt_blocks:
            logger.info("Detected nGPT-style classifier block(s); applying initial weight normalization.")
            self.normalize_ngpt_matrices()

    @staticmethod
    def _split_by_batch(tensor: torch.Tensor, batch_sizes: list[int]) -> list[torch.Tensor]:
        splits = []
        offset = 0
        for size in batch_sizes:
            splits.append(tensor[offset : offset + size])
            offset += size
        return splits

    def _pool_and_split(
        self,
        attention_mask: torch.Tensor,
        outputs,
        batch_sizes: list[int],
        target_layer: int | None = None,
    ) -> list[torch.Tensor]:
        layer = target_layer if target_layer is not None else -1
        pooled = self.pooler(attention_mask, outputs, layer)
        return self._split_by_batch(pooled, batch_sizes)

    def _get_sequence_features(
        self,
        outputs,
        batch_sizes: list[int],
        target_layer: int | None = None,
    ) -> list[torch.Tensor]:
        if outputs.hidden_states is None:
            raise ValueError("Hidden states are required for GPT/nGPT classifiers but were not returned by the backbone.")
        layer = target_layer if target_layer is not None else -1
        hidden = outputs.hidden_states[layer]
        return self._split_by_batch(hidden, batch_sizes)

    def _infer_batch_size(self, sentence: dict) -> int:
        for name in self._BACKBONE_ARG_NAMES:
            tensor = sentence.get(name)
            if tensor is not None:
                return tensor.shape[0]
        raise ValueError("Cannot infer batch size from empty sentence inputs.")

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
        device = self.backbone_device

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

        # batchデータが全て1種類の入力内容の場合、そのパスだけを実行
        uniform = partitioner.uniform_kind()
        if uniform:
            outputs = self._paths[uniform].run_full(batch_inputs, extra_kwargs)
            return outputs

        # 複数種類の入力内容が混在する場合、各パスを実行して結果をマージ
        outputs_by_kind = {}
        for kind, path in self._paths.items():
            outputs = path.run_partition(batch_inputs, partitioner, device, extra_kwargs)
            outputs_by_kind[kind] = outputs

        return partitioner.merge(outputs_by_kind, device=device)

    # ── それぞれのパスで“本体”を切り出すヘルパー例 ──
    def _forward_binary(
        self,
        input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
        input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, head_mask_2, inputs_embeds_2,
        **kwargs
    ):
        return None  # ダミー --- IGNORE ---

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

        results = {
        "original_avg": self.avg_pooler(attention_mask, outputs.last_hidden_state),
        "original_cls": self.cls_pooler(attention_mask, outputs.last_hidden_state),
        "original_max": self.max_pooler(attention_mask, outputs.last_hidden_state), 
    }
        for name, classifier in self.embedding_classifiers.items():
            target_layer = int(self.classifier_configs[name]["layer"])
            sequence_features = self._get_sequence_features(outputs, [input_ids.size(0)], target_layer)[0]
            features = [(sequence_features, attention_mask)]
            results[name] = self.classifier_strategy.single(name, classifier, features)
            
        return results            
            
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
            return None

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
        return None  # ダミー --- IGNORE ---

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
            dim_value = self.classifier_configs[name].get('output_dim', self.config.hidden_size)
            param_str = f"{self.classifier_configs[name]['type']}_layer:{self.classifier_configs[name]['layer']}_dim:{dim_value}"
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


    def _detect_ngpt_blocks(self) -> bool:
        """
        embedding_classifiers のどこかに use_nGPT=1 の Block があれば True。
        backbone (BERT 本体) には触らない。
        """
        for _, clf in self.embedding_classifiers.items():
            for module in clf.modules():
                if _is_ngpt_block(module):
                    return True
        return False

    def normalize_ngpt_matrices(self) -> None:
        """
        use_nGPT=1 な classifier Block のパラメータを nGPT 方式で L2 正規化する。
        - 初期化時に 1 回
        - train.py から optimizer.step() ごとに呼び出される
        """
        if not getattr(self, "use_ngpt_blocks", False):
            return
        for _, clf in self.embedding_classifiers.items():
            for module in clf.modules():
                if _is_ngpt_block(module):
                    _normalize_single_ngpt_block(module)