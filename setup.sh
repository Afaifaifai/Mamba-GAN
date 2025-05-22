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
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
pip install nltk
pip install texttable
pip install yacs

pip install transformers==2.5.1
pip install tqdm==4.41.1
pip install tensorboardX
pip install note_seq

echo "=== py37 環境準備完成 ==="
