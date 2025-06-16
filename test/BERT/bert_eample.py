import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ----------------------------------------------------------------------------
# 1. 定義 BERT 判別器模型
# ----------------------------------------------------------------------------
class BERTDiscriminator(nn.Module):
    """
    使用 BERT 作為骨幹的判別器模型。
    它接收一段文字，並輸出一個單一的數值（logit），代表該文字的「真實度」。
    """
    def __init__(self, model_name='bert-base-chinese'):
        """
        初始化函數。
        :param model_name: 要使用的 Hugging Face 預訓練模型名稱。
                         'bert-base-chinese' 是一個常用且強大的中文模型。
        """
        super(BERTDiscriminator, self).__init__()
        
        # 載入預訓練的 BERT 模型作為基礎
        self.bert = BertModel.from_pretrained(model_name)
        
        # BERT 的隱藏層大小 (對於 'bert-base-*' 系列，通常是 768)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 定義判別器的「頭部」(head)
        # 這是一個簡單的線性層，將 BERT 輸出的 [CLS] 向量 (768維)
        # 映射到一個單一的數值 (1維)
        self.discriminator_head = nn.Linear(self.bert_hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        前向傳播函數。
        :param input_ids: 經過 tokenizer 處理後的 token ID。
        :param attention_mask: 對應的 attention mask。
        :return: 一個代表「真實度」的 logit 分數。
        """
        # 將輸入傳遞給 BERT 模型
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 我們使用 BERT 輸出的 [CLS] token 的表示 (pooled_output)
        # 這個向量被設計來代表整個輸入序列的語義
        pooled_output = outputs.pooler_output
        
        # 將 [CLS] 向量傳遞給我們的線性層，得到最終的 logit 分數
        logit = self.discriminator_head(pooled_output)
        
        return logit

# ----------------------------------------------------------------------------
# 2. 實際使用範例
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- 步驟 1: 載入 Tokenizer 並實例化模型 ---
    MODEL_NAME = 'bert-base-chinese'
    
    # 載入與模型對應的 tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # 建立我們的判別器模型
    discriminator = BERTDiscriminator(model_name=MODEL_NAME)
    
    # 將模型設為評估模式 (因為我們只是做推論，不是在訓練)
    discriminator.eval()

    # --- 步驟 2: 準備輸入資料 ---
    # 假設我們有兩句話，一句是真實的，一句是從 GAN 生成器來的（假的）
    real_sentence = "今天天氣真好，我們一起去公園散步吧。"
    fake_sentence = "天空 香蕉 跑步 昨天 書本 因為。" # 一句語法不通的假話

    sentences = [real_sentence, fake_sentence]
    print(f"準備判斷的句子: {sentences}\n")

    # --- 步驟 3: 將文字轉換為模型輸入格式 ---
    # 使用 tokenizer 進行編碼，並轉換成 PyTorch Tensors
    inputs = tokenizer(
        sentences, 
        padding=True,       # 將句子填充到相同的長度
        truncation=True,    # 如果句子太長，則截斷
        max_length=128,     # 最大長度
        return_tensors="pt" # 返回 PyTorch Tensors
    )
    
    print(f"Tokenizer 輸出 (input_ids): \n{inputs['input_ids']}\n")
    print(f"Tokenizer 輸出 (attention_mask): \n{inputs['attention_mask']}\n")

    # --- 步驟 4: 進行推論並獲得分數 ---
    # 使用 torch.no_grad() 以節省計算資源，因為我們不需要計算梯度
    with torch.no_grad():
        scores = discriminator(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )

    # --- 步驟 5: 解讀輸出 ---
    print("--- 判斷結果 ---")
    # 模型的輸出是一個 (batch_size, 1) 的張量，我們把它攤平成一維
    scores = scores.squeeze(-1) 
    
    for i, sentence in enumerate(sentences):
        # 分數越高，代表模型認為該句子越「真實」
        # 注意：在未經訓練前，這個分數是隨機的！
        print(f"句子: '{sentence}'")
        print(f"判別器分數 (logit): {scores[i].item():.4f}\n")
        
    print("注意：以上分數是來自未經訓練的判別器，因此是隨機的。")
    print("在實際的 GAN 訓練中，模型會學習如何區分真實與生成的文本。")