#!/bin/bash
source ../env/.venv/bin/activate
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR="${PROJECT_ROOT}/dataset"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
mkdir -p "${OUTPUT_ROOT}"

MODEL_NAME=${MODEL:-mixedbread-ai/mxbai-embed-large-v1}
POOLER_TYPE=${POOLER_TYPE:-avg}
MAX_SEQ_LEN=${MAX_SEQ_LENGTH:-256}
LEARNING_RATE=${LR:-1e-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
NUM_EPOCHS=${NUM_EPOCHS:-10}
GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS:-1}
TAU=${TAU:-0.05}
POS_PAIRS=${INBATCH_POS_PAIRS:-0}
NEG_PAIRS=${INBATCH_NEG_PAIRS:-0}
SEED=${SEED:-42}
FP16=${FP16:-true}
BF16=${BF16:-false}
FREEZE_ENCODER=${FREEZE_ENCODER:-true}
OUTPUT_DIR="${OUTPUT_ROOT}/sentiment_info_nce"
mkdir -p "${OUTPUT_DIR}"

CONFIG_PATH="${OUTPUT_DIR}/classifier_config.json"
cat <<JSON >"${CONFIG_PATH}"
{
  "sentiment": {
    "type": "linear",
    "objective": "infoNCE",
    "distance": "cosine",
    "output_dim": 256,
    "dropout": 0.1,
    "tau": ${TAU},
    "inbatch_positive_pairs": ${POS_PAIRS},
    "inbatch_negative_pairs": ${NEG_PAIRS}
  }
}
JSON

WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-sentiment_info_nce}
WANDB_PROJECT=${WANDB_PROJECT:-sentiment_circle}
mkdir -p "${OUTPUT_DIR}/${WANDB_PROJECT}/${WANDB_PROJECT_NAME}"

python "${SCRIPT_DIR}/train.py" \
  --model_name_or_path "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_file "${DATA_DIR}/Train_df.csv" \
  --validation_file "${DATA_DIR}/Valid_df.csv" \
  --test_file "${DATA_DIR}/Test_df.csv" \
  --classifier_configs "${CONFIG_PATH}" \
  --encoding_type bi_encoder \
  --pooler_type "${POOLER_TYPE}" \
  --max_seq_length ${MAX_SEQ_LEN} \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs ${NUM_EPOCHS} \
  --warmup_ratio 0.1 \
  --lr_scheduler_type linear \
  --logging_steps 50 \
  --eval_steps 50 \
  --eval_strategy steps \
  --save_strategy no \
  --do_train \
  --do_eval \
  --do_predict \
  --seed ${SEED} \
  --fp16 ${FP16} \
  --bf16 ${BF16} \
  --freeze_encoder ${FREEZE_ENCODER} \
  --overwrite_output_dir True \
  --wandb_project_name ${WANDB_PROJECT_NAME} \
  --wandb_project ${WANDB_PROJECT} \
  --report_to wandb > "${OUTPUT_DIR}/${WANDB_PROJECT}/${WANDB_PROJECT_NAME}/train.log" 2>&1
