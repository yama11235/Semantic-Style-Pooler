"""Label mapping and preparation utilities."""
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from datasets import DatasetDict
from utils.data.preprocessing import parse_dict

logger = logging.getLogger(__name__)


def prepare_label_mappings(
    raw_datasets: DatasetDict,
    model_args,
    data_args,
) -> Tuple[
    DatasetDict,
    List[str],
    Dict[int, str],
    Dict[str, int],
    List[str],
    Optional[Dict],
    Dict[str, Dict],
    Dict,
    Dict,
    Dict[str, Dict[int, str]],
]:
    """
    Prepare label mappings for the datasets.
    
    Args:
        raw_datasets: Raw datasets
        model_args: Model arguments
        data_args: Data arguments
        
    Returns:
        Tuple containing:
        - updated_datasets
        - labels
        - id2label
        - label2id
        - aspect_key
        - classifier_configs
        - classifier_configs_for_trainer
        - corr_labels
        - corr_weights
        - label_name_mappings
    """
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")

    labels = [
        key
        for key in raw_datasets["validation"].features.keys()
        if key not in {"sentence1", "sentence2", "sentence3"}
    ]

    id2label = {i: aspect for i, aspect in enumerate(list(labels))}
    label2id = {aspect: i for i, aspect in enumerate(list(labels))}

    if model_args.classifier_configs is not None:
        if os.path.exists(model_args.classifier_configs):
            classifier_configs = json.load(open(model_args.classifier_configs))
        else:
            classifier_configs = parse_dict(model_args.classifier_configs)
        aspect_key = list(classifier_configs.keys())
    else:
        classifier_configs = None
        aspect_key = getattr(model_args, 'aspect_key', [])

    if aspect_key is None:
        aspect_key = []
    elif isinstance(aspect_key, str):
        aspect_key = [aspect_key]
    else:
        aspect_key = list(aspect_key)

    # Rename label columns
    label_candidates = ["labels", "label", "target"]
    updated_datasets = raw_datasets
    for split, dataset in updated_datasets.items():
        for head in aspect_key:
            if head in dataset.column_names:
                continue
            renamed = False
            for candidate in label_candidates:
                if candidate in dataset.column_names:
                    dataset = dataset.rename_column(candidate, head)
                    renamed = True
                    break
            if not renamed:
                raise ValueError(
                    f"Expected label column for head '{head}' not found in dataset columns: {dataset.column_names}"
                )
        updated_datasets[split] = dataset

    # Build label name mappings
    label_name_mappings: Dict[str, Dict[int, str]] = {}

    for head in aspect_key:
        sample_values: List = []
        for dataset in updated_datasets.values():
            if head in dataset.column_names:
                values = [v for v in dataset[head] if v is not None]
                if values:
                    sample_values.extend(values)
                    break
        if not sample_values:
            continue
        if isinstance(sample_values[0], str):
            unique_values = sorted(
                {
                    v
                    for ds in updated_datasets.values()
                    if head in ds.column_names
                    for v in ds[head]
                }
            )
            label_to_id_map = {label: idx for idx, label in enumerate(unique_values)}
            label_name_mappings[head] = {
                idx: label for label, idx in label_to_id_map.items()
            }

            def _encode_labels(example):
                value = example.get(head)
                if value is not None and value in label_to_id_map:
                    example[head] = label_to_id_map[value]
                return example

            updated_datasets = updated_datasets.map(
                _encode_labels,
                desc=f"Encoding labels for {head}",
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Load correlation labels and weights
    if model_args.corr_labels is not None:
        if os.path.exists(model_args.corr_labels):
            corr_labels = json.load(open(model_args.corr_labels))
        else:
            corr_labels = parse_dict(model_args.corr_labels)
    else:
        corr_labels = {}
        
    if model_args.corr_weights is not None:
        if os.path.exists(model_args.corr_weights):
            corr_weights = json.load(open(model_args.corr_weights))
        else:
            corr_weights = parse_dict(model_args.corr_weights)
    else:
        corr_weights = {}

    classifier_configs_for_trainer = (
        classifier_configs
        if classifier_configs is not None
        else {head: {"objective": getattr(model_args, 'objective', 'infoNCE')} for head in aspect_key}
    )

    return (
        updated_datasets,
        labels,
        id2label,
        label2id,
        aspect_key,
        classifier_configs,
        classifier_configs_for_trainer,
        corr_labels,
        corr_weights,
        label_name_mappings,
    )
