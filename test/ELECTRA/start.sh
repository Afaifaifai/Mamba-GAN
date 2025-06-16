# 執行新版的 run_mlm.py
python main.py \
    # --- 專案與路徑設定 ---
    --output_dir ./output_dir \
    --overwrite_output_dir \
    \
    # --- 模型與 Tokenizer 設定 (核心修改) ---
    --model_type electra \
    --config_name google/electra-base-discriminator \
    --tokenizer_name ./my-midi-tokenizer \
    --config_overrides "num_hidden_layers=5" \
    \
    # --- 數據設定 (核心修改) ---
    --train_file ../data/maestro_magenta_s5_t3/train_data.txt \
    --validation_file ../data/maestro_magenta_s5_t3/valid_data.txt \
    --line_by_line \
    --max_seq_length 512 \
    \
    # --- 訓練與評估流程控制 ---
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --mlm \
    --mlm_probability 0.15