#!/bin/bash
set -e

# --- 我們預處理好的 Arrow 數據集資料夾路徑 ---
PROCESSED_DATASET_PATH="../data/arrow_dataset"

python main.py \
    --output_dir=output_dir \
    --overwrite_output_dir \
    --model_name_or_path google/electra-base-discriminator \
    --tokenizer_name ./midi_tokenizer \
    --trust_remote_code True \
    --dataset_name ${PROCESSED_DATASET_PATH} \
    --max_seq_length 20 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 2048 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --save_steps 1000 \
    --eval_steps 1000

echo "腳本執行完畢。"