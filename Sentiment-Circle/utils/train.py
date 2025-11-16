"""
Adapted code from HuggingFace run_glue.py

Author: Ameet Deshpande, Carlos E. Jimenez
"""
import json
import logging
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = "11c77110ad8d01487f0db6c65c034f66aaa6841b"
import random
import sys

if __package__ is None or __package__ == "":
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    __package__ = "utils"
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PrinterCallback,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from utils.progress_logger import LogCallback
from utils.dataset_preprocessing import (
    get_preprocessing_function,
    parse_dict,
    batch_get_preprocessing_function,
)
from utils.model.modeling_utils import DataCollatorForBiEncoder, get_model
from utils.model.nGPT_model import NGPTWeightNormCallback
from utils.clf_trainer import CustomTrainer
from utils.metrics import compute_metrics
import wandb
import torch
from datasets import concatenate_datasets, DatasetDict
from typing import List, Union, Optional, Dict, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments(HFTrainingArguments):
    log_time_interval: int = field(
        default=15,
        metadata={
            "help": (
                "Log at each `log_time_interval` seconds. "
                "Default will be to log every 15 seconds."
            )
        },
    )
    wandb_project_name: str = field(
        default="train_classifier",
        metadata={
            "help": "Wandb project name."
        },
    )
    wandb_project: str = field(
        default="train_classifier",
        metadata={
            "help": "Wandb project."
        },
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "A seed for reproducible training. If not set, will use a random seed."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of csv or json files containing the training data."},
    )
    validation_file: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of csv or json files containing the validation data."},
    )
    test_file: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of csv or json files containing the test data."},
    )
    # Dataset specific arguments
    max_similarity: Optional[float] = field(
        default=None, metadata={"help": "Maximum similarity score."}
    )
    min_similarity: Optional[float] = field(
        default=None, metadata={"help": "Minimum similarity score."}
    )

    def __post_init__(self):
        def _get_extension(x: Union[str, List[str]]) -> str:
            """
            文字列ならそのまま、リストなら先頭要素を取り出して拡張子を返す。
            """
            if isinstance(x, list):
                if len(x) == 0:
                    raise ValueError("file list must contain at least one path")
                path = x[0]
            else:
                path = x
            return path.split(".")[-1]

        # 拡張子チェックはすべて _get_extension で行う
        val_ext = _get_extension(self.validation_file)
        # train_file があれば
        if self.train_file:
            train_ext = _get_extension(self.train_file)
            assert train_ext in ["csv", "json"], "`train_file` should be csv or json."
            assert train_ext == val_ext, "`train_file` and `validation_file` must have same extension."
        # test_file があれば
        if self.test_file:
            test_ext = _get_extension(self.test_file)
            assert test_ext in ["csv", "json"], "`test_file` should be csv or json."
            assert test_ext == val_ext, "`test_file` and `validation_file` must have same extension."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_flash_attention: str = field(
        default="eager",
        metadata={
            "help": "Whether to use flash attention."
        },
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device map for the model."
        },
    )   
    # What type of modeling
    encoding_type: Optional[str] = field(
        default="bi_encoder",
        metadata={
            "help": "What kind of model to choose. Options:\
            1) cross_encoder: Full encoder model.\
            2) bi_encoder: Bi-encoder model.\
            3) tri_encoder: Tri-encoder model."
        },
    )
    # Pooler for bi-encoder
    pooler_type: Optional[str] = field(
        default="avg",
        metadata={
            "help": "Pooler type: Options:\
            1) cls: Use [CLS] token.\
            2) avg: Mean pooling."
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={"help": "Freeze encoder weights."}
    )
    transform: Optional[bool] = field(
        default=False,
        metadata={"help": "Use a linear transformation on the encoder output"},
    )
    classifier_configs: Optional[str] = field(
        default=None,
        metadata={
            "help": "classifier configs in json format."
        },
    )
    corr_labels: Optional[str] = field(
        default=None,
    )
    corr_weights: Optional[str] = field(
        default=None,
    )
    aspect_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Aspect key for the dataset. If you don't train a classifier, this is needed to specify the aspect key."
        },
    )
    objective: Optional[str] = field(
        default="regression",
        metadata={
            "help": "Objective for the model. Options:\
            1) regression: Regression task.\
            2) binary classification: Classification task."
        },
    )
    
def load_raw_datasets(
    model_args: "ModelArguments",
    data_args: "DataTrainingArguments",
    training_args: TrainingArguments,
    seed: int,
) -> Tuple[DatasetDict, bool]:
    data_files: Dict[str, List[str]] = {}

    if training_args.do_train and data_args.train_file:
        data_files["train"] = data_args.train_file
        logger.info("Load train files: %s", data_args.train_file)
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
        logger.info("Load validation files: %s", data_args.validation_file)
    if data_args.test_file:
        data_files["test"] = data_args.test_file
        logger.info("Load test files: %s", data_args.test_file)
    elif training_args.do_predict:
        raise ValueError("test_file argument is missing. required for do_predict.")

    example_file = next((files[0] for files in data_files.values() if files), None)
    if example_file is None:
        raise ValueError(
            "At least one of `train_file`, `validation_file`, or `test_file` must be provided."
        )

    file_type = "csv" if example_file.endswith(".csv") else "json"
    logger.debug("Resolved file type for datasets: %s", file_type)

    raw_datasets: Dict[str, datasets.Dataset] = {}

    for split in ["train", "validation", "test"]:
        paths = data_files.get(split, [])
        if not paths:
            continue

        if split == "train":
            sample_limit = data_args.max_train_samples
        elif split == "validation":
            sample_limit = data_args.max_eval_samples
        else:
            sample_limit = data_args.max_predict_samples

        first = load_dataset(
            file_type,
            data_files={split: [paths[0]]},
            split=split,
            cache_dir=model_args.cache_dir,
        )

        if sample_limit is not None and len(first) > sample_limit:
            first = first.shuffle(seed=seed).select(range(sample_limit))

        datasets_to_concat = [first]
        for path in paths[1:]:
            dataset = load_dataset(
                file_type,
                data_files={split: [path]},
                split=split,
                cache_dir=model_args.cache_dir,
            )
            if sample_limit is not None and len(dataset) > sample_limit:
                dataset = dataset.shuffle(seed=seed).select(range(sample_limit))

            datasets_to_concat.append(dataset)

        raw_datasets[split] = concatenate_datasets(datasets_to_concat).shuffle(
            seed=training_args.seed
        )

    dataset_dict = DatasetDict(raw_datasets)

    for split in dataset_dict.keys():
        dataset_dict[split] = dataset_dict[split].shuffle(seed=seed)

    for split, dataset in dataset_dict.items():
        column_names = dataset.column_names
        if "sentence1" not in column_names:
            text_column = next(
                (candidate for candidate in ["text", "sentence", "document", "utterance"] if candidate in column_names),
                None,
            )
            if text_column is None:
                raise ValueError(
                    "Dataset must include a text column named 'sentence1' or one of "
                    "['text', 'sentence', 'document', 'utterance']."
                )
            dataset = dataset.rename_column(text_column, "sentence1")
        dataset_dict[split] = dataset

    sentence3_flag = any(
        "sentence3" in dataset.column_names for dataset in dataset_dict.values()
    )

    return dataset_dict, sentence3_flag


def prepare_label_mappings(
    raw_datasets: DatasetDict,
    model_args: "ModelArguments",
    data_args: "DataTrainingArguments",
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
        aspect_key = model_args.aspect_key

    if aspect_key is None:
        aspect_key = []
    elif isinstance(aspect_key, str):
        aspect_key = [aspect_key]
    else:
        aspect_key = list(aspect_key)

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
        else {head: {"objective": model_args.objective} for head in aspect_key}
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


def initialize_wandb(
    model_args: "ModelArguments",
    training_args: TrainingArguments,
    classifier_configs: Optional[Dict],
    max_train_samples: Optional[int],
):
    wandb_project = training_args.wandb_project.replace("_train.csv", "").replace(
        "data_preprocessed/", ""
    )
    wandb_project_name = training_args.wandb_project_name
    model_id = model_args.model_name_or_path.split("/")[-1]
    sample_suffix = str(max_train_samples) if max_train_samples is not None else "all"

    if model_args.freeze_encoder:
        project_name = f"{model_id}_{sample_suffix}_{wandb_project}"
        return wandb.init(
            project=project_name,
            entity="2959648335-university-of-tokyo",
            name=wandb_project_name,
            config=classifier_configs,
        )

    project_name = f"{model_id}-Hot_{sample_suffix}_{wandb_project}"
    return wandb.init(
        project=project_name,
        entity="2959648335-university-of-tokyo",
        name=wandb_project_name,
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        filename="logs/train.log",
        filemode="a",
    )
    training_args.log_level = "info"
    # Keep custom columns (e.g., active_heads) for the data collator.
    training_args.remove_unused_columns = False
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s" % training_args)
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    seed = training_args.seed
    random.seed(seed)
    
    raw_datasets, sentence3_flag = load_raw_datasets(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        seed=seed,
    )
    # if data_args.min_similarity is None:
    #     data_args.min_similarity = min(labels)
    #     logger.warning(
    #         f"Setting min_similarity: {data_args.min_similarity}. Override by setting --min_similarity."
    #     )
    # if data_args.max_similarity is None:
    #     data_args.max_similarity = max(labels)
    #     logger.warning(
    #         f"Setting max_similarity: {data_args.max_similarity}. Override by setting --max_similarity."
    #     )        

    (
        raw_datasets,
        labels,
        id2label,
        label2id,
        aspect_key,
        classifier_configs,
        classifier_configs_for_trainer,
        corr_labels,
        corr_weights,
        label_name_mappings,
    ) = prepare_label_mappings(
        raw_datasets=raw_datasets,
        model_args=model_args,
        data_args=data_args,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        # finetuning_task=None,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if training_args.fp16:
        config.torch_dtype = torch.float32
    elif training_args.bf16:
        config.torch_dtype = torch.bfloat16
    model_cls = get_model(model_args)
    config.update(
        {
            "freeze_encoder": model_args.freeze_encoder,
            "use_auth_token": model_args.use_auth_token,
            "model_revision": model_args.model_revision,
            "cache_dir": model_args.cache_dir,
            "model_name_or_path": model_args.model_name_or_path,
            "pooler_type": model_args.pooler_type,
            "transform": model_args.transform,
            "attn_implementation": model_args.use_flash_attention,
            "device_map": model_args.device_map,
        }
    )
    labels = list(classifier_configs_for_trainer.keys())
    id2_head = {i: head for i, head in enumerate(labels)}
    model = model_cls(model_config=config, classifier_configs=classifier_configs)
    if model_args.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True

    # --- ここから追加: nGPT 判定 & optimizer 設定補正 -------------------
    use_ngpt_riemann = bool(getattr(model, "use_ngpt_blocks", False))
    if use_ngpt_riemann:
        logger.info(
            "nGPT-style classifier detected (model.use_ngpt_blocks=True). "
            "Enabling pseudo-Riemann weight normalization and nGPT-friendly optimizer settings."
        )
        # nGPT 元コードにならって weight_decay=0, warmup なしに寄せる
        if training_args.weight_decay != 0.0:
            logger.warning(
                f"Overriding weight_decay from {training_args.weight_decay} to 0.0 for nGPT."
            )
            training_args.weight_decay = 0.0

        if getattr(training_args, "warmup_steps", 0) != 0:
            logger.warning(
                f"Overriding warmup_steps from {training_args.warmup_steps} to 0 for nGPT."
            )
            training_args.warmup_steps = 0

        if getattr(training_args, "warmup_ratio", 0.0) != 0.0:
            logger.warning(
                f"Overriding warmup_ratio from {training_args.warmup_ratio} to 0.0 for nGPT."
            )
            training_args.warmup_ratio = 0.0
    else:
        logger.info(
            "No nGPT-style classifier detected (model.use_ngpt_blocks=False). "
            "Training will use standard optimizer settings without Riemann projection."
        )

    logger.debug("Model architecture: %s", model)
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "longest"
    else:
        padding = False
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d."
            % (
                data_args.max_seq_length,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    if sentence3_flag:
        preprocess_function = batch_get_preprocessing_function(
            tokenizer,
            sentence1_key="sentence1",
            sentence2_key="sentence2",
            sentence3_key="sentence3",
            sentence3_flag=sentence3_flag,
            aspect_key=aspect_key,
            padding=padding,
            max_seq_length=max_seq_length,
            model_args=model_args,
            scale=None
        )
        batched = True        
        
    else:
        preprocess_function = get_preprocessing_function(
            tokenizer,
            sentence1_key="sentence1",
            sentence2_key="sentence2",
            sentence3_key="sentence3",
            sentence3_flag=sentence3_flag,
            aspect_key=aspect_key,
            padding=padding,
            max_seq_length=max_seq_length,
            model_args=model_args,
            scale=None
        )
        batched = False
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        logger.debug("Raw datasets before tokenization: %s", raw_datasets)
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=batched,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names,
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        train_dataset_size = len(train_dataset)
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else train_dataset_size
        )
    else:
        train_dataset = None
        train_dataset_size = 0
        max_train_samples = data_args.max_train_samples or 0
    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
    # Log a few random samples from the training set:
    if training_args.do_train and train_dataset is not None:
        sample_count = min(3, train_dataset_size)
        indices = random.sample(range(train_dataset_size), sample_count) if sample_count > 0 else []
        for index in indices:
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    collator_dtype = getattr(config, "torch_dtype", torch.float32)

    logger.debug(
        "Torch dtype: %s, collator dtype: %s", config.torch_dtype, collator_dtype
    )
    data_collator = DataCollatorForBiEncoder(
        tokenizer=tokenizer,
        padding="max_length",
        pad_to_multiple_of=None,
        dtype=collator_dtype,
    )
    
    _ = initialize_wandb(
        model_args=model_args,
        training_args=training_args,
        classifier_configs=classifier_configs,
        max_train_samples=max_train_samples,
    )
        
    trainer_state = {"trainer": None}

    def train_centroid_getter():
        trainer_obj = trainer_state["trainer"]
        if trainer_obj is None:
            return {}
        return trainer_obj.get_train_label_centroids()

    def compute_fn(eval_pred):
        trainer_obj = trainer_state["trainer"]
        embedding_mode = "classifier"
        if trainer_obj is not None and trainer_obj.use_original_eval_embeddings:
            embedding_mode = "original"
        return compute_metrics(
            eval_pred,
            classifier_configs=classifier_configs_for_trainer,
            id2_head=id2_head,
            train_centroid_getter=train_centroid_getter,
            embedding_eval_mode=embedding_mode,
        )
    
    ngpt_callback = NGPTWeightNormCallback(enabled=use_ngpt_riemann)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        classifier_configs=classifier_configs_for_trainer,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_fn,
        tokenizer=tokenizer,
        callbacks=[LogCallback, ngpt_callback],
        dtype=collator_dtype,
        corr_labels=corr_labels,
        corr_weights=corr_weights,
        tsne_save_dir=os.path.join(training_args.output_dir, "tsne_plots"),
        tsne_label_mappings=label_name_mappings,
    )
    trainer_state["trainer"] = trainer
    
    trainer.remove_callback(PrinterCallback)
    
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            
        trainer.evaluate(eval_dataset=eval_dataset)
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Test
    if training_args.do_predict:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)        
    # except Exception as e:
    #     print(f"WANDB failed to initialize: {e}", file=sys.stderr)
    #     wandb.init(mode="disabled")

if __name__ == "__main__":
    main()
