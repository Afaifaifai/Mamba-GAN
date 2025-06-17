#!/bin/bash

# 確保腳本在遇到錯誤時立即停止
set -e

# --- 執行新版的 run_mlm.py ---
# 目標：在新腳本上，嚴格復現舊指令的實驗設定，並使用 ELECTRA 模型

python /home/afaifai/Mamba-GAN/test/ELECTRA/main.py \
    --output_dir /content/drive/MyDrive/temp \
    --overwrite_output_dir \
    --model_name_or_path google/electra-base-discriminator \
    --tokenizer_name ./midi_tokenizer \
    --trust_remote_code True \
    --train_file ../data/maestro_magenta_s5_t3/train_all_data.txt \
    --validation_file ../data/maestro_magenta_s5_t3/valid_all_data.txt \
    --line_by_line \
    --max_seq_length 20 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 2048 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --save_steps 1000 \
    --eval_steps 1000
    # --config_overrides "num_hidden_layers=5" \

echo "腳本執行完畢。"