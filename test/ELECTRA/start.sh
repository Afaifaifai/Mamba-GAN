# 執行新版 run_mlm.py，目標：訓練一個用於 MambaGAN 判別器的 ELECTRA 模型
python main.py \
    --output_dir ./output_dir \
    --overwrite_output_dir \
    # --- 模型與 Tokenizer 設定 (核心修改) ---
    # 選擇與 BERT-base 同級的 ELECTRA-base 作為模型
    --model_name_or_path google/electra-base-discriminator \
    # << 關鍵 >>: 指定上一步打包好的 Tokenizer 目錄
    --tokenizer_name ./midi_tokenizer/ \
    # << 關鍵 >>: 覆寫模型配置，使其符合您舊的 5 層隱藏層設定
    --config_overrides "num_hidden_layers=5" \
    \
    # --- 數據設定 (完全對應舊指令) ---
    --train_file ../data/maestro_magenta_s5_t3/train.txt \
    --validation_file ../data/maestro_magenta_s5_t3/valid.txt \
    --line_by_line \
    # 忠實還原舊指令的序列長度設定
    --max_seq_length 20 \
    \
    # --- 訓練與評估流程控制 (完全對應舊指令) ---
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    # 忠實還原舊指令的批次大小設定 (來自 README)
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 2048 \
    # 啟用 Masked Language Modeling
    --mlm \
    \
    # --- 其他超參數 (建議保持預設或根據需要調整) ---
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --save_steps 1000 \
    --eval_steps 1000