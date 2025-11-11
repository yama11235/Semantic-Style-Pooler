#!/bin/bash
# Todo
# comand line arguments, save path, wandb

# source /works/data3/users/yama11235/yama11235/SLBERT/my-project/.venv/bin/activate

set -eux

# model=${MODEL:-mixedbread-ai/mxbai-embed-large-v1}  # pre-trained model
# pooler_type=${POOLER_TYPE:-avg}  # avg, cls, mean, last
# max_seq_length=${MAX_SEQ_LENGTH:-512}  # maximum sequence length for the model
# start_layer=${START_LAYER:-23}  # start layer number for the encoder
# layer_num=${LAYER_NUM:-23}  # layer number to use for the encoder
# use_flash_attention=${USE_FLASH_ATTENTION:-eager}  # whether to use flash attention
# fp16=${FP16:-true}  # whether to use fp16
# bf16=${BF16:-false}  # whether to use bf16
# device_map=${DEVICE_MAP:-cuda}  # device map for the model
# lr=${LR:-1e-4}  # learning rate
# train_batch_size=${TRAIN_BATCH_SIZE:-300}

model=${MODEL:-Qwen/Qwen3-Embedding-8B}  # pre-trained model
pooler_type=${POOLER_TYPE:-last}  # avg, cls, mean, last
max_seq_length=${MAX_SEQ_LENGTH:-512}  # maximum sequence length for the model
start_layer=${START_LAYER:-35}  # start layer number for the encoder
layer_num=${LAYER_NUM:-35}  # layer number to use for the encoder
use_flash_attention=${USE_FLASH_ATTENTION:-eager}
fp16=${FP16:-false}
bf16=${BF16:-true} # whether to use flash attention
device_map=${DEVICE_MAP:-cuda}
lr=${LR:-1e-4}  # learning rate
train_batch_size=${TRAIN_BATCH_SIZE:-150}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-True}  # whether to use gradient checkpointing

encoding=${ENCODER_TYPE:-bi_encoder}  # cross_encoder, bi_encoder, tri_encoder  # learning rate
# wd=${WD:-0.1}  # weight decay
transform=${TRANSFORM:-False}  # whether to use an additional linear layer after the encoder
seed=${SEED:-42}
output_dir=${OUTPUT_DIR:-output_style_cls}  # output directory for the model

train_data_size=${TRAIN_DATA_SIZE:-5000}  # number of training samples
# config=clf_data:${train_file_name}_lr:${lr}_wd:${wd}_seed:${seed}


# データセットと対応するラベル名のペアを定義
sets=(
  # "stsb stsb_score"
  # "sick sick_score"
  # "Opusparcus opusparcus_score"
  # "CxC cxc_score"
  # "STS3k sts3k_score"
  # "ArgPairs argpairs_score"
  # "BWS bws_score"
  # "FinSTS finsts_score"
  # "SemRel2024 SemRel2024_score"
  # "APT apt_label"
  # "PARADE parade_label"
  # "Webis-CPC-11 webis_cpc_label"
  # "AskUbuntu askubuntu_label"
  # "PAWS-Wiki paws_wiki_label"
  # "QQP qqp_label"
  "ELSA_joy-sadness-anger-surprise-love-fear emotion"
  "ELSA_conversational-poetic-formal-narrative style"
  "Paradetox_neutral-toxic toxic"
  "APPDIA_neutral-offensive offensive"
  "WNC_subjective-objective objectivity"
  "MTST_positive-negative negative"
)

# ファイル名の構築
# ループで変数を自動生成
for idx in "${!sets[@]}"; do
  # bash の配列は 0 始まりなので +1
  i=$((idx + 1))
  # 空白区切りで dataset_id と label を取得
  read -r dataset_id dataset_label <<< "${sets[idx]}"

  # 変数名を eval で生成
  eval "train_file${i}_id=\"$dataset_id\""
  eval "train_file${i}_label=\"$dataset_label\""

  eval "train_file${i}=\"data_preprocessed/${dataset_id}_train.csv\""
  eval "eval_file${i}=\"data_preprocessed/${dataset_id}_test.csv\""
  eval "test_file${i}=\"data_preprocessed/${dataset_id}_test.csv\""
done

# ① train, eval, test のファイル名変数をそれぞれ配列にまとめる
train_files=()
validation_files=()
test_files=()

train_file_ids=()
train_file_labels=()
layer_lists=()

for i in {1..6}; do
  # 間接展開用に変数名を作る
  tf_var="train_file${i}"
  vf_var="eval_file${i}"
  tfst_var="test_file${i}"

  # 配列に追加
  train_files+=( "${!tf_var}" )
  validation_files+=( "${!vf_var}" )
  test_files+=( "${!tfst_var}" )

  # ファイル名から id と label を取得
  train_file_id_var="train_file${i}_id"
  train_file_label_var="train_file${i}_label" 

  train_file_ids+=( "${!train_file_id_var}" )
  train_file_labels+=( "${!train_file_label_var}" )

  # layer_lists+=( 23 )  # すべてのファイルで同じ層を使用するため、固定値を追加
  layer_lists+=( 35 )  # すべてのファイルで同じ層を使用するため、固定値を追加
done

# margin_set=(0.2 0.4 0.6)
# alpha_set=(1.0 2.0)
# margin_set=(-0.2 -0.6 0.2 0.6)  # マージンのセット
margin_set=(0.2)  # マージンのセット
gamma_set=(5.0)  # ガンマのセット
alpha_set=(0.0)  # アルファのセット
layer=35  # 使用する層

for gamma in "${gamma_set[@]}"; do
  for margin in "${margin_set[@]}"; do
    for alpha in "${alpha_set[@]}"; do
      # classifier_configs をヒアドキュメントで組み立て
      classifier_configs=$(cat <<EOF
{
  "style": {"type":"contrastive_logit","objective":"contrastive_logit","distance":"cosine","intermediate_dim":256,"output_dim":4,"dropout":0.1,"layer": ${layer},"margin": ${margin}, "alpha": ${alpha}, "gamma": ${gamma}},
  "emotion": {"type":"contrastive_logit","objective":"contrastive_logit","distance":"cosine","intermediate_dim":256,"output_dim":6,"dropout":0.1,"layer": ${layer},"margin": ${margin}, "alpha": ${alpha}, "gamma": ${gamma}},
  "toxic": {"type":"contrastive_logit","objective":"contrastive_logit","distance":"cosine","intermediate_dim":256,"output_dim":2,"dropout":0.1,"layer": ${layer},"margin": ${margin}, "alpha": ${alpha}, "gamma": ${gamma}},
  "offensive": {"type":"contrastive_logit","objective":"contrastive_logit","distance":"cosine","intermediate_dim":256,"output_dim":2,"dropout":0.1,"layer": ${layer},"margin": ${margin}, "alpha": ${alpha}, "gamma": ${gamma}},
  "objectivity": {"type":"contrastive_logit","objective":"contrastive_logit","distance":"cosine","intermediate_dim":256,"output_dim":2,"dropout":0.1,"layer": ${layer},"margin": ${margin}, "alpha": ${alpha}, "gamma": ${gamma}},
  "negative": {"type":"contrastive_logit","objective":"contrastive_logit","distance":"cosine","intermediate_dim":256,"output_dim":2,"dropout":0.1,"layer": ${layer},"margin": ${margin}, "alpha": ${alpha}, "gamma": ${gamma}}
}
EOF
      )
    
      # config=alpha:${alpha}_margin:${margin}_gamma:${gamma}
      config=alpha:${alpha}_gamma:${gamma}
      wandb_project_name=${config}
      wandb_project=style_cls

      config_path="${output_dir}/${model}/${wandb_project}/${config}/clf_config.json"
      mkdir -p "$(dirname "$config_path")"
      echo "$classifier_configs" > "$config_path"

      corr_labels=$(cat <<EOF
{}
EOF
      )
      corr_weights=$(cat <<EOF
{}
EOF
      )
      corr_labels_path="${output_dir}/${model}/${wandb_project}/${config}/corr_labels.json"
      corr_weights_path="${output_dir}/${model}/${wandb_project}/${config}/corr_weights.json"
      mkdir -p "$(dirname "$corr_weights_path")"
      mkdir -p "$(dirname "$corr_labels_path")"
      echo "$corr_labels" > "$corr_labels_path"
      echo "$corr_weights" > "$corr_weights_path"

      CUDA_VISIBLE_DEVICES=0 python train.py \
        --output_dir "${output_dir}/${model}/${wandb_project}/${config}/" \
        --classifier_configs "${config_path}" \
        --corr_labels "${corr_labels_path}" \
        --corr_weights "${corr_weights_path}" \
        --model_name_or_path ${model} \
        --encoding_type ${encoding} \
        --pooler_type ${pooler_type} \
        --freeze_encoder True \
        --max_seq_length ${max_seq_length} \
        --train_file "${train_files[@]}" \
        --validation_file "${validation_files[@]}" \
        --test_file "${test_files[@]}" \
        --do_train \
        --do_eval \
        --eval_strategy "steps" \
        --eval_steps 50 \
        --logging_steps 50 \
        --per_device_train_batch_size ${train_batch_size} \
        --per_device_eval_batch_size 192 \
        --gradient_accumulation_steps 2 \
        --learning_rate ${lr} \
        --num_train_epochs 10 \
        --lr_scheduler_type constant \
        --warmup_ratio 0.1 \
        --log_level info \
        --disable_tqdm False \
        --save_strategy steps \
        --save_steps 5000 \
        --seed ${seed} \
        --data_seed ${seed} \
        --fp16 ${fp16} \
        --bf16 ${bf16} \
        --log_time_interval 15 \
        --remove_unused_columns False \
        --wandb_project_name ${wandb_project_name} \
        --wandb_project ${wandb_project} \
        --max_train_samples ${train_data_size} \
        --max_eval_samples 5000 \
        --max_predict_samples 5000 \
        --use_flash_attention ${use_flash_attention} \
        --overwrite_output_dir True \
        --device_map ${device_map} \
        --report_to wandb > "${output_dir}/${model}/${wandb_project}/${config}/style_classifier.log" 2>&1
      done
    done
  done