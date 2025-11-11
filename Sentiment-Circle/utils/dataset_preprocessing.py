import ast
import argparse
from typing import Any, Dict, List

try:
    from .sentence_batch_utils import tokenize_optional_sentences
except ImportError:  # pragma: no cover - fallback when executed as a script
    from sentence_batch_utils import tokenize_optional_sentences

def scale_to_range(labels, _min, _max):
    return list(map(lambda x: (x - _min) / (_max - _min), labels))

def parse_dict(dict_str):
    try:
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary literal: {e}")

def get_preprocessing_function(
    tokenizer,
    sentence1_key,
    sentence2_key,
    sentence3_key,
    sentence3_flag,
    aspect_key,            # list of all possible label‐column names
    padding,
    max_seq_length,
    model_args,
    scale=None,
):
    if model_args.encoding_type != 'bi_encoder':
        raise ValueError(f'Invalid model type: {model_args.encoding_type}')

    batch_processor = batch_get_preprocessing_function(
        tokenizer=tokenizer,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        sentence3_key=sentence3_key,
        sentence3_flag=sentence3_flag,
        aspect_key=aspect_key,
        padding=padding,
        max_seq_length=max_seq_length,
        model_args=model_args,
        scale=scale,
    )

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        batched_examples = {k: [v] for k, v in examples.items()}
        processed = batch_processor(batched_examples)
        flattened = {}
        for key, value in processed.items():
            if isinstance(value, list) and key not in {"active_heads", "labels"}:
                flattened[key] = value[0]
            else:
                flattened[key] = value
        return flattened

    return preprocess_function

def batch_get_preprocessing_function(
    tokenizer,
    sentence1_key,
    sentence2_key,
    sentence3_key,
    sentence3_flag,
    aspect_key,            # list of all possible label‐column names
    padding,
    max_seq_length,
    model_args,
    scale=None,
):
    if model_args.encoding_type != 'bi_encoder':
        raise ValueError(f'Invalid model type: {model_args.encoding_type}')

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        is_batch = isinstance(examples[sentence1_key], list)
        if not is_batch:
            examples = {k: [v] for k, v in examples.items()}

        batch_size = len(examples[sentence1_key])
        sentence2_list: List[Any] = examples.get(sentence2_key, [None] * batch_size)
        if not isinstance(sentence2_list, list):
            sentence2_list = [sentence2_list]
        if len(sentence2_list) != batch_size:
            sentence2_list = (sentence2_list + [None] * batch_size)[:batch_size]
        examples[sentence2_key] = sentence2_list

        if sentence3_flag:
            sentence3_list: List[Any] = examples.get(sentence3_key, [None] * batch_size)
            if not isinstance(sentence3_list, list):
                sentence3_list = [sentence3_list]
            if len(sentence3_list) != batch_size:
                sentence3_list = (sentence3_list + [None] * batch_size)[:batch_size]
            examples[sentence3_key] = sentence3_list

        sent1_res = tokenizer(
            examples[sentence1_key],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        sent2_res = tokenize_optional_sentences(
            tokenizer=tokenizer,
            sentences=examples[sentence2_key],
            padding=padding,
            max_length=max_seq_length,
        )
        if sentence3_flag:
            sent3_res = tokenize_optional_sentences(
                tokenizer=tokenizer,
                sentences=examples[sentence3_key],
                padding=padding,
                max_length=max_seq_length,
            )

        out = {}
        out.update(sent1_res)
        out["input_ids_2"] = sent2_res["input_ids"]
        out["attention_mask_2"] = sent2_res["attention_mask"]
        if "token_type_ids" in sent2_res:
            out["token_type_ids_2"] = sent2_res["token_type_ids"]

        if sentence3_flag:
            out["input_ids_3"] = sent3_res["input_ids"]
            out["attention_mask_3"] = sent3_res["attention_mask"]
            if "token_type_ids" in sent3_res:
                out["token_type_ids_3"] = sent3_res["token_type_ids"]

        active_heads, labels = [], []
        for i in range(batch_size):
            found = False
            for head in aspect_key:
                col = examples.get(head)
                value = None if col is None else (col[i] if isinstance(col, list) else col)
                if value is not None:
                    active_heads.append(head)
                    labels.append(value)
                    found = True
                    break
            if not found:
                active_heads.append(None)
                labels.append(None)
        out["active_heads"] = active_heads
        out["labels"] = labels

        if not is_batch:
            return {k: v[0] for k, v in out.items()}

        return out

    return preprocess_function
