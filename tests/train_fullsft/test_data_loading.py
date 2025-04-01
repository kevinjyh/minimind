import os
import json
import pytest
import torch
import matplotlib
matplotlib.use('Agg')  # 設置為非互動式後端
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.dataset import SFTDataset
from transformers import AutoTokenizer
from model.LMConfig import LMConfig

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

@pytest.fixture
def lm_config():
    return LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=False)

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

@pytest.fixture
def sample_data(tmp_path):
    """創建測試數據"""
    data = [
        {
            "conversations": [
                {"role": "user", "content": "你好，這是一個測試。"},
                {"role": "assistant", "content": "是的，這是一個測試。"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "今天天氣很好。"},
                {"role": "assistant", "content": "是的，天氣確實很好。"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "機器學習很有趣。"},
                {"role": "assistant", "content": "是的，機器學習非常有趣。"}
            ]
        }
    ]
    
    data_file = tmp_path / "test_data.jsonl"
    with open(data_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return str(data_file)

def test_dataset_initialization(lm_config, tokenizer, sample_data):
    """測試數據集初始化"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    
    # 驗證數據集大小
    assert len(dataset) == 3
    
    # 驗證數據格式
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 3  # X, Y, loss_mask
    
    # 驗證張量形狀
    X, Y, loss_mask = sample
    assert X.shape[0] <= lm_config.max_seq_len
    assert Y.shape[0] <= lm_config.max_seq_len
    assert loss_mask.shape[0] <= lm_config.max_seq_len

def test_data_loader(lm_config, tokenizer, sample_data):
    """測試數據載入器"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 驗證批次大小
    batch = next(iter(dataloader))
    X, Y, loss_mask = batch
    assert X.shape[0] == 2  # batch_size
    
    # 驗證數據類型
    assert X.dtype == torch.long
    assert Y.dtype == torch.long
    assert loss_mask.dtype == torch.int64

def test_data_preprocessing(lm_config, tokenizer, sample_data, tmp_path):
    """測試數據預處理"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    
    # 收集所有序列長度
    seq_lengths = []
    for i in range(len(dataset)):
        X, _, _ = dataset[i]
        seq_lengths.append(X.shape[0])
    
    # 繪製序列長度分佈圖
    plt.figure(figsize=(10, 6))
    plt.hist(seq_lengths, bins=20, alpha=0.75)
    plt.title('序列長度分佈')
    plt.xlabel('序列長度')
    plt.ylabel('頻率')
    plt.savefig(os.path.join(tmp_path, 'sequence_length_distribution.png'))
    plt.close()

def test_data_tokenization(lm_config, tokenizer, sample_data):
    """測試數據標記化"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    
    # 檢查標記化結果
    for i in range(len(dataset)):
        X, Y, _ = dataset[i]
        
        # 驗證輸入輸出對應關係
        decoded_input = tokenizer.decode(X)
        decoded_output = tokenizer.decode(Y)
        
        # 確保解碼後的文本是有效的
        assert len(decoded_input) > 0
        assert len(decoded_output) > 0

def test_loss_mask(lm_config, tokenizer, sample_data):
    """測試損失遮罩"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    
    # 檢查損失遮罩
    for i in range(len(dataset)):
        _, _, loss_mask = dataset[i]
        
        # 驗證遮罩值
        assert torch.all(loss_mask >= 0)  # 所有值應該非負
        assert torch.all(loss_mask <= 1)  # 所有值應該小於等於1
        
        # 驗證遮罩形狀
        assert loss_mask.shape[0] <= lm_config.max_seq_len

def test_data_augmentation(lm_config, tokenizer, sample_data):
    """測試數據增強（如果有的話）"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    
    # 獲取同一個樣本的兩次載入結果
    sample1 = dataset[0]
    sample2 = dataset[0]
    
    # 驗證一致性（如果沒有隨機增強）
    X1, Y1, mask1 = sample1
    X2, Y2, mask2 = sample2
    
    assert torch.all(X1 == X2)
    assert torch.all(Y1 == Y2)
    assert torch.all(mask1 == mask2)

def test_data_statistics(lm_config, tokenizer, sample_data, tmp_path):
    """測試數據統計"""
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    
    # 收集統計信息
    vocab_usage = {}
    for i in range(len(dataset)):
        X, Y, _ = dataset[i]
        for token in X:
            token_id = token.item()
            vocab_usage[token_id] = vocab_usage.get(token_id, 0) + 1
    
    # 繪製詞彙使用分佈圖
    plt.figure(figsize=(12, 6))
    plt.bar(vocab_usage.keys(), vocab_usage.values())
    plt.title('詞彙使用分佈')
    plt.xlabel('詞彙ID')
    plt.ylabel('使用次數')
    plt.savefig(os.path.join(tmp_path, 'vocabulary_usage.png'))
    plt.close() 