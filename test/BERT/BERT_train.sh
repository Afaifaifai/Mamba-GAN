#!/usr/bin/env bash
set -e
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py37

# 再執行你的程式
python3 main.py \
    --overwrite_output_dir \
    --output_dir=/content/drive/MyDrive/temp \
    --train_dir=../data/maestro_magenta_s5_t3/train \
    --eval_dir=../data/maestro_magenta_s5_t3/valid \
    --test_dir=../data/maestro_magenta_s5_t3/test \
    --vocab_file=../data/maestro_magenta_s5_t3/vocab.txt \
    --num_hidden_layers=5 \
    --event_type=magenta \
    --model_type=bert \
    --block_size=20 \
    --tokenizer_name=midi_tokenizer \
    --do_train \
    --evaluate_during_training \
    --do_eval \
    --mlm