from .modeling_encoders import BiEncoderForClassification
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForBiEncoder:
    tokenizer: PreTrainedTokenizerBase
    padding: Optional[bool | str] = "max_length"
    pad_to_multiple_of: Optional[int] = None
    dtype: torch.dtype = torch.float32

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        active_heads = [f["active_heads"] for f in features]
        sentence_groups = self._collect_sentence_groups(features)
        combined_len = self._max_sequence_length(sentence_groups)

        batch = {}
        group1 = self._pad_group(sentence_groups[1], combined_len)
        batch.update(group1)

        group2 = self._pad_group(sentence_groups[2], combined_len)
        for key, value in group2.items():
            batch[f"{key}_2"] = value

        if 3 in sentence_groups:
            group3 = self._pad_group(sentence_groups[3], combined_len)
            for key, value in group3.items():
                batch[f"{key}_3"] = value

        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=self.dtype)
        batch["active_heads"] = active_heads
        return batch

    def _collect_sentence_groups(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        has_sentence3 = any(f.get("input_ids_3") is not None for f in features)
        include_token_type = any(
            ("token_type_ids" in f)
            or ("token_type_ids_2" in f)
            or ("token_type_ids_3" in f)
            for f in features
        )

        groups: Dict[int, List[Dict[str, Any]]] = {1: [], 2: []}
        if has_sentence3:
            groups[3] = []

        for feature in features:
            groups[1].append(self._build_sentence_entry(feature, "", include_token_type))
            groups[2].append(self._build_sentence_entry(feature, "_2", include_token_type))
            if has_sentence3:
                groups[3].append(
                    self._build_sentence_entry(feature, "_3", include_token_type)
                )
        return groups

    def _build_sentence_entry(
        self,
        feature: Dict[str, Any],
        suffix: str,
        include_token_type: bool,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {}
        ids_key = f"input_ids{suffix}"
        attn_key = f"attention_mask{suffix}"
        token_type_key = "token_type_ids" if not suffix else f"token_type_ids{suffix}"

        input_ids = feature.get(ids_key)
        entry["input_ids"] = input_ids if input_ids is not None else []

        attention_mask = feature.get(attn_key)
        entry["attention_mask"] = attention_mask if attention_mask is not None else []

        if token_type_key in feature:
            token_type_ids = feature[token_type_key]
            entry["token_type_ids"] = token_type_ids if token_type_ids is not None else []
        elif include_token_type:
            entry["token_type_ids"] = []

        return entry

    def _max_sequence_length(self, sentence_groups: Dict[int, List[Dict[str, Any]]]) -> int:
        lengths = [
            len(entry["input_ids"])
            for group in sentence_groups.values()
            for entry in group
        ]
        return max(lengths) if lengths else 0

    def _pad_group(
        self,
        entries: List[Dict[str, Any]],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        if not entries:
            return {}
        target_len = max_length if max_length > 0 else None
        return self.tokenizer.pad(
            entries,
            padding=self.padding,
            max_length=target_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
    
def get_model(model_args):
    if model_args.encoding_type == 'bi_encoder':
        return BiEncoderForClassification
    raise ValueError(f'Invalid model type: {model_args.encoding_type}')
