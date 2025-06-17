#!/bin/bash
set -e

# --- 我們預處理好的 Arrow 數據集資料夾路徑 ---
PROCESSED_DATASET_PATH="/home/afaifai/Mamba-GAN/data/arrow_dataset"

python /home/afaifai/Mamba-GAN/test/ELECTRA/main.py \
    --output_dir /content/drive/MyDrive/temp \
    --overwrite_output_dir \
    --model_name_or_path google/electra-base-discriminator \
    --tokenizer_name /home/afaifai/Mamba-GAN/test/ELECTRA/midi_tokenizer \
    --trust_remote_code True \
    \
    # --- 數據設定 (核心修改) ---
    # 直接告訴它從哪個資料夾載入 train split 和 validation split
    \\
    --train_file ${PROCESSED_DATASET_PATH}/train \
    --validation_file ${PROCESSED_DATASET_PATH}/validation \
    \
    # << 不再需要 --line_by_line >> 因為數據已不是文字行
    \
    # ... 其他參數如 max_seq_length, do_train 等不變 ...
    \\
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