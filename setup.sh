#!/usr/bin/env bash
set -e

# 1. 安裝 Miniconda（若已安裝可跳過）
if [ ! -d /usr/local/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /usr/local/miniconda3
fi

# 2. 把 conda 加到 PATH
export PATH="/usr/local/miniconda3/bin:$PATH"

# 3. 初始化 shell（只要第一次跑一次）
source /usr/local/miniconda3/etc/profile.d/conda.sh

# 4. 建立／更新 py37 環境
if ! conda env list | grep -q py37; then
  conda create -y -n py37 python=3.7
fi

# 5. 安裝你指定的套件
bash requirements.sh

echo "=== py37 環境準備完成 ==="
