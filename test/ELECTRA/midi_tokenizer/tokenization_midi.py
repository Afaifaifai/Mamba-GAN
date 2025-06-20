# The following codes are gently modified based on HuggingFace Transformer language-modeling
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import logging
import os
from transformers import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
import numpy as np

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": os.path.join(os.path.realpath(os.path.dirname(os.path.realpath(__file__))), "magenta_vocab_file.txt")}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "midi_tokenizer": VOCAB_FILES_NAMES["vocab_file"],
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "midi_tokenizer": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "midi_tokenizer": {"do_lower_case": True},
}


def load_vocab(vocab_vocab_file):
    """Loads a vocabulary file into a dictionary."""
    with open(vocab_vocab_file, 'r') as f:
        contents = f.read().strip().split()
    vocab = collections.OrderedDict()
    for index, token in enumerate(contents):
        if index == 1:
            vocab['[PAD]'] = 1
        else:
            vocab[token] = index
    vocab['[MASK]'] = len(vocab)
    return vocab


# class MIDITokenizer(BertTokenizer):
#     vocab_files_names = VOCAB_FILES_NAMES
#     pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
#     pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
#     max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

#     def __init__(
#             self,
#             vocab_file,
#             do_lower_case=True,
#             do_basic_tokenize=True,
#             never_split=None,
#             unk_token="[UNK]",
#             sep_token="[SEP]",
#             pad_token="[PAD]",
#             cls_token="[CLS]",
#             mask_token="[MASK]",
#             tokenize_chinese_chars=True,
#             **kwargs
#     ):
#         super().__init__(
#             vocab_file,
#             do_lower_case=do_lower_case,
#             do_basic_tokenize=do_basic_tokenize,
#             never_split=never_split,
#             unk_token=unk_token,
#             sep_token=sep_token,
#             pad_token=pad_token,
#             cls_token=cls_token,
#             mask_token=mask_token,
#             **kwargs,
#         )

#         self.build_vocab_file(vocab_file)
#         self.do_basic_tokenize = do_basic_tokenize
#         if do_basic_tokenize:
#             self.basic_tokenizer = BasicTokenizer(
#                 do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
#             )

#     def build_vocab_file(self, vocab_file, event_type='magenta'):
#         self.vocab = load_vocab(vocab_file)
#         self.event_type = event_type
#         self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

# 檔案名稱: tokenization_midi.py (最終修正版)

import collections
import logging
import os
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# 我們不再需要那些全域變數了，因為 from_pretrained 會自動處理

class MIDITokenizer(BertTokenizer):
    """
    一個為 MIDI 事件客製化的 BertTokenizer。
    它覆寫了 _load_vocab 方法以實現特殊的詞彙表載入規則。
    """
    def __init__(self, vocab_file, **kwargs):
        # __init__ 現在變得非常乾淨，直接呼叫父類別即可
        # 所有複雜的載入邏輯都交給 from_pretrained 和我們覆寫的方法
        super().__init__(vocab_file, **kwargs)

    def _load_vocab(self, vocab_file: str) -> collections.OrderedDict:
        """
        這是我們覆寫的核心方法。
        from_pretrained 在找到 vocab.txt 後，會呼叫這個方法來載入詞彙。
        我們在這裡實現您所有客製化的載入邏輯。
        """
        print(f"--- 正在使用客製化的 _load_vocab 方法載入: {vocab_file} ---")
        vocab = collections.OrderedDict()
        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                # 這裡的讀取方式可能需要根據您的 vocab 檔案格式微調
                # 假設您的 vocab.txt 每個 token 佔一行
                tokens = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            raise IOError(f"無法讀取詞彙表檔案 {vocab_file}。錯誤: {e}")

        for index, token in enumerate(tokens):
            # 這是您最關鍵的、被完整保留下來的客製化規則
            if index == 1 and token == "<PAD>":
                print("偵測到特殊規則：強制設定 [PAD] 的 ID 為 1")
                vocab['[PAD]'] = 1
            else:
                vocab[token] = index
        
        # 舊版程式碼在 load_vocab 最後才加入 [MASK]
        # 但標準的 BertTokenizer 會在 __init__ 中自動處理
        # 我們這裡確保它一定存在即可
        if '[MASK]' not in vocab:
             vocab['[MASK]'] = len(vocab)
             print("已自動加入 [MASK] token。")

        return vocab
    

if __name__ == '__main__':
    # 獲取此腳本檔案 (tokenization_midi.py) 所在的絕對目錄路徑
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # 基於腳本所在目錄，來組合出詞彙表和儲存目錄的絕對路徑
    # VOCAB_PATH = os.path.join(SCRIPT_DIR, "midi_tokenizer", "magenta_vocab_file.txt")
    # SAVE_DIRECTORY = os.path.join(SCRIPT_DIR, "midi_tokenizer")
    VOCAB_PATH = os.path.join(SCRIPT_DIR, "magenta_vocab_file.txt")
    SAVE_DIRECTORY = SCRIPT_DIR
    # ==============================================================================

    print(f"正在從詞彙表檔案: '{VOCAB_PATH}' 建立 Tokenizer...")

    try:
        # 實例化您的客製化 Tokenizer
        # 舊指令中沒有傳遞其他參數給 Tokenizer，所以這裡也只傳入 vocab_file
        tokenizer = MIDITokenizer(vocab_file=VOCAB_PATH)

        # 將這個物件儲存為 Hugging Face 的標準格式
        tokenizer.save_pretrained(SAVE_DIRECTORY)

        print("-" * 50)
        print(f"🎉 成功！您的客製化 Tokenizer 已儲存至 '{SAVE_DIRECTORY}' 資料夾。")
        print("下一步，您可以在 run_mlm.py 的指令中使用這個路徑。")
        print("-" * 50)

    except FileNotFoundError:
        print(f"錯誤：找不到詞彙表檔案！請確認 '{VOCAB_PATH}' 路徑是否正確。")
    except Exception as e:
        print(f"發生預期外的錯誤: {e}")

