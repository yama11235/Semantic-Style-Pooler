#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT="${SCRIPT_DIR}/train.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "train.sh not found at ${TRAIN_SCRIPT}" >&2
  exit 1
fi

POOLER_OPTIONS=("avg" "max" "cls")
# FREEZE_ENCODER_OPTIONS=("true" "false")
FREEZE_ENCODER_OPTIONS=("true")
LEARNING_RATES=("5e-4" "1e-4" "5e-5")
TAU_VALUES=("0.25" "0.5" "0.75")
# CLASSIFIER_TYPES=("GPT" "nGPT")
CLASSIFIER_TYPES=("GPT")
SEED_VALUES=("42")

BASE_WANDB_PROJECT=${WANDB_PROJECT:-sentiment_circle_hypra}

sanitize_token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value// /}"
  echo "${value}"
}

for seed in "${SEED_VALUES[@]}"; do
  for pooler in "${POOLER_OPTIONS[@]}"; do
    pooler_token=$(sanitize_token "${pooler}")
    for freeze in "${FREEZE_ENCODER_OPTIONS[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for tau in "${TAU_VALUES[@]}"; do
          for classifier in "${CLASSIFIER_TYPES[@]}"; do
            lr_token=$(sanitize_token "${lr}")
            tau_token=$(sanitize_token "${tau}")
            project_name="pool_${pooler_token}_freeze_${freeze}_lr_${lr_token}_tau_${tau_token}_clf_${classifier}_seed_${seed}"

            echo "=== Running: POOLER=${pooler}, FREEZE_ENCODER=${freeze}, LR=${lr}, TAU=${tau}, CLASSIFIER=${classifier}, SEED=${seed} ==="

            POOLER_TYPE="${pooler}" \
            FREEZE_ENCODER="${freeze}" \
            LR="${lr}" \
            TAU="${tau}" \
            CLASSIFIER_TYPE="${classifier}" \
            WANDB_PROJECT="${BASE_WANDB_PROJECT}" \
            WANDB_PROJECT_NAME="${project_name}" \
            SEED="${seed}" \
            bash "${TRAIN_SCRIPT}"
          done
        done
      done
    done
  done
done
