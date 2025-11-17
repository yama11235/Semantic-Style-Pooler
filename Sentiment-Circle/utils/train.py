"""
Adapted code from HuggingFace run_glue.py

Author: Ameet Deshpande, Carlos E. Jimenez

Main training script for multi-classifier embedding models.
"""
import logging
import os
import sys
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = "11c77110ad8d01487f0db6c65c034f66aaa6841b"

if __package__ is None or __package__ == "":
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    __package__ = "utils"

import datasets
import transformers
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from utils.config import TrainingArguments, DataTrainingArguments, ModelArguments
from utils.data import load_raw_datasets, prepare_label_mappings
from utils.training import (
    initialize_wandb,
    setup_model_and_config,
    setup_tokenizer,
    prepare_datasets,
    create_trainer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        filename="logs/train.log",
        filemode="a",
    )
    training_args.log_level = "info"
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
    
    # Check for checkpoints
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
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed
    seed = training_args.seed
    random.seed(seed)
    
    # Load datasets
    raw_datasets, sentence3_flag = load_raw_datasets(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        seed=seed,
    )
    
    # Prepare label mappings
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
    
    # Setup model and config
    config, model, use_ngpt_riemann = setup_model_and_config(
        model_args=model_args,
        training_args=training_args,
        labels=list(classifier_configs_for_trainer.keys()),
        id2label=id2label,
        label2id=label2id,
        classifier_configs=classifier_configs,
    )
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(model_args)
    
    # Prepare datasets
    train_dataset, eval_dataset, predict_dataset, max_train_samples = prepare_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        aspect_key=aspect_key,
        sentence3_flag=sentence3_flag,
    )
    
    # Initialize wandb
    _ = initialize_wandb(
        model_args=model_args,
        training_args=training_args,
        classifier_configs=classifier_configs,
        max_train_samples=max_train_samples,
    )
    
    # Create trainer
    id2_head = {i: head for i, head in enumerate(classifier_configs_for_trainer.keys())}
    trainer, trainer_state = create_trainer(
        model=model,
        config=config,
        training_args=training_args,
        classifier_configs_for_trainer=classifier_configs_for_trainer,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        corr_labels=corr_labels,
        corr_weights=corr_weights,
        label_name_mappings=label_name_mappings,
        use_ngpt_riemann=use_ngpt_riemann,
        id2_head=id2_head,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        trainer.evaluate(eval_dataset=eval_dataset)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        metrics = train_result.metrics
        metrics["train_samples"] = min(
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset),
            len(train_dataset)
        )
        
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Testing
    if training_args.do_predict:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        metrics["predict_samples"] = min(
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset),
            len(predict_dataset)
        )
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
