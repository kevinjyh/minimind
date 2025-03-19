import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# 修正：添加專案根目錄到系統路徑
root_dir = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, root_dir)  # 使用insert(0)確保優先搜索

from model.LMConfig import LMConfig
from model.model import FeedForward


# 設置測試環境
def setup_function():
    """在每個測試函數之前設置測試環境"""
    # 建立預設配置
    global default_config, small_config, auto_hidden_config
    default_config = LMConfig(
        dim=512,
        hidden_dim=1024,
        multiple_of=64,
        dropout=0.1
    )
    
    # 建立一個較小的配置，用於測試計算
    small_config = LMConfig(
        dim=64,
        hidden_dim=128,
        multiple_of=16,
        dropout=0.0
    )
    
    # 建立一個配置，其中 hidden_dim 為 None，測試自動計算
    auto_hidden_config = LMConfig(
        dim=512,
        hidden_dim=None,
        multiple_of=64,
        dropout=0.1
    )


def test_initialization():
    """測試 FeedForward 類的初始化"""
    ffn = FeedForward(default_config)
    
    # 檢查線性層的維度
    assert ffn.w1.in_features == default_config.dim
    assert ffn.w1.out_features == default_config.hidden_dim
    assert ffn.w3.in_features == default_config.dim
    assert ffn.w3.out_features == default_config.hidden_dim
    assert ffn.w2.in_features == default_config.hidden_dim
    assert ffn.w2.out_features == default_config.dim
    
    # 檢查 dropout 比率
    assert ffn.dropout.p == default_config.dropout


def test_auto_hidden_dim():
    """測試當 hidden_dim 為 None 時的自動計算"""
    ffn = FeedForward(auto_hidden_config)
    
    # 檢查 hidden_dim 是否已正確計算
    expected_hidden_dim = 4 * auto_hidden_config.dim
    expected_hidden_dim = int(2 * expected_hidden_dim / 3)
    expected_hidden_dim = auto_hidden_config.multiple_of * ((expected_hidden_dim + auto_hidden_config.multiple_of - 1) // auto_hidden_config.multiple_of)
    
    assert auto_hidden_config.hidden_dim == expected_hidden_dim
    assert ffn.w1.out_features == expected_hidden_dim


def test_forward_shape():
    """測試前向傳播的輸出形狀"""
    ffn = FeedForward(default_config)
    
    # 創建隨機輸入
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, default_config.dim)
    
    # 執行前向傳播
    output = ffn(x)
    
    # 檢查輸出形狀
    assert output.shape == (batch_size, seq_len, default_config.dim)


def test_forward_calculation():
    """測試前向傳播的計算正確性"""
    ffn = FeedForward(small_config)
    
    # 使用確定性種子
    torch.manual_seed(42)
    
    # 創建簡單輸入
    batch_size = 1
    seq_len = 1
    x = torch.ones(batch_size, seq_len, small_config.dim)
    
    # 執行前向傳播
    output = ffn(x)
    
    # 手動計算預期輸出
    with torch.no_grad():
        hidden1 = ffn.w1(x)
        hidden3 = ffn.w3(x)
        hidden1_activated = F.silu(hidden1)
        hidden = hidden1_activated * hidden3
        expected_output = ffn.w2(hidden)
    
    # 檢查輸出與預期是否一致
    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)


def test_dropout_effect():
    """測試 dropout 對輸出的影響"""
    dropout_config = LMConfig(
        dim=64,
        hidden_dim=128,
        multiple_of=16,
        dropout=0.5
    )
    
    ffn = FeedForward(dropout_config)
    
    # 固定隨機種子
    torch.manual_seed(42)
    
    # 創建輸入
    x = torch.ones(1, 1, dropout_config.dim)
    
    # 訓練模式下執行
    ffn.train()
    output_train = ffn(x)
    
    # 評估模式下執行
    ffn.eval()
    output_eval = ffn(x)
    
    # 檢查訓練和評估模式下輸出是否不同 (受 dropout 影響)
    assert not torch.allclose(output_train, output_eval, rtol=1e-5, atol=1e-5)


def test_different_batch_size():
    """測試不同批次大小的情況"""
    ffn = FeedForward(default_config)
    
    # 檢查不同批次大小
    batch_sizes = [1, 8, 16]
    seq_len = 10
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, seq_len, default_config.dim)
        output = ffn(x)
        assert output.shape == (batch_size, seq_len, default_config.dim)


def test_different_sequence_length():
    """測試不同序列長度的情況"""
    ffn = FeedForward(default_config)
    
    # 檢查不同序列長度
    batch_size = 2
    seq_lengths = [1, 16, 64, 128]
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, default_config.dim)
        output = ffn(x)
        assert output.shape == (batch_size, seq_len, default_config.dim) 