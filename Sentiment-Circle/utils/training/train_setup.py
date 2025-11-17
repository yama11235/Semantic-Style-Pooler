"""Training setup utilities."""
import os
import logging
import random
from typing import Optional, Dict, Tuple
import torch
import wandb
from transformers import AutoConfig, AutoTokenizer, PrinterCallback
from utils.data.preprocessing import get_preprocessing_function, batch_get_preprocessing_function
from utils.model.modeling_utils import DataCollatorForBiEncoder, get_model
from utils.model.nGPT_model import NGPTWeightNormCallback
from utils.clf_trainer import CustomTrainer
from utils.metrics import compute_metrics
from utils.progress_logger import LogCallback

logger = logging.getLogger(__name__)


def initialize_wandb(
    model_args,
    training_args,
    classifier_configs: Optional[Dict],
    max_train_samples: Optional[int],
):
    """Initialize Weights & Biases logging."""
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


def setup_model_and_config(
    model_args,
    training_args,
    labels,
    id2label,
    label2id,
    classifier_configs,
) -> Tuple[any, any, bool]:
    """
    Setup model configuration and initialize model.
    
    Returns:
        Tuple of (config, model, use_ngpt_riemann)
    """
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
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
            "attn_implementation": model_args.use_flash_attention,
            "device_map": model_args.device_map,
        }
    )
    
    model = model_cls(model_config=config, classifier_configs=classifier_configs)
    
    # Freeze or unfreeze encoder
    for param in model.backbone.parameters():
        param.requires_grad = not model_args.freeze_encoder

    # Check for nGPT and adjust optimizer settings
    use_ngpt_riemann = bool(getattr(model, "use_ngpt_blocks", False))
    if use_ngpt_riemann:
        logger.info(
            "nGPT-style classifier detected. Enabling pseudo-Riemann weight normalization "
            "and nGPT-friendly optimizer settings."
        )
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
            "No nGPT-style classifier detected. Training will use standard optimizer settings."
        )

    logger.debug("Model architecture: %s", model)
    return config, model, use_ngpt_riemann


def setup_tokenizer(model_args):
    """Setup tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    return tokenizer


def prepare_datasets(
    raw_datasets,
    tokenizer,
    data_args,
    model_args,
    training_args,
    aspect_key,
    sentence3_flag,
):
    """
    Tokenize and prepare datasets.
    
    Returns:
        Tuple of (train_dataset, eval_dataset, predict_dataset, max_train_samples)
    """
    # Determine padding strategy
    if data_args.pad_to_max_length:
        padding = "longest"
    else:
        padding = False
        
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d.",
            data_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    # Choose preprocessing function
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
    
    # Tokenize datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        logger.debug("Raw datasets before tokenization: %s", raw_datasets)
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=batched,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["train"].column_names,
        )
    
    # Prepare train dataset
    train_dataset = None
    max_train_samples = 0
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
        
        # Log samples
        sample_count = min(3, train_dataset_size)
        indices = random.sample(range(train_dataset_size), sample_count) if sample_count > 0 else []
        for index in indices:
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    # Prepare eval dataset
    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
    
    # Prepare predict dataset
    predict_dataset = None
    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
    
    return train_dataset, eval_dataset, predict_dataset, max_train_samples


def create_trainer(
    model,
    config,
    training_args,
    classifier_configs_for_trainer,
    tokenizer,
    train_dataset,
    eval_dataset,
    corr_labels,
    corr_weights,
    label_name_mappings,
    use_ngpt_riemann,
    id2_head,
):
    """
    Create and configure the trainer.
    
    Returns:
        Tuple of (trainer, trainer_state)
    """
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
    
    trainer_state = {"trainer": None}

    def train_centroid_getter():
        trainer_obj = trainer_state["trainer"]
        if trainer_obj is None:
            return {}
        return trainer_obj.get_train_label_centroids()

    def compute_fn(eval_pred):
        trainer_obj = trainer_state["trainer"]
        embedding_mode = "classifier"
        if trainer_obj is not None and hasattr(trainer_obj, '_current_eval_embedding_mode'):
            embedding_mode = trainer_obj._current_eval_embedding_mode
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
    
    return trainer, trainer_state
