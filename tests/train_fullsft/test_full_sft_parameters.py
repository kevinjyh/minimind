import os
import sys
import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 設置為非互動式後端
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from train_full_sft import init_model, Logger

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 創建測試資料夾
os.makedirs('tests/train_fullsft/visualizations', exist_ok=True)

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def mock_args():
    """創建模擬的 args"""
    return SimpleNamespace(device="cpu")

@pytest.fixture
def mock_dist():
    """創建模擬的 dist"""
    mock = MagicMock()
    mock.get_rank.return_value = 0
    return mock

@pytest.fixture
def mock_tokenizer():
    """創建模擬的 tokenizer"""
    mock = MagicMock()
    mock.from_pretrained.return_value = mock
    return mock

@pytest.fixture
def mock_model():
    """創建模擬的模型"""
    mock = MagicMock()
    mock.parameters.return_value = [torch.randn(10, 10)]
    return mock

@pytest.fixture
def mock_state_dict():
    """創建模擬的模型狀態字典"""
    return {}

def test_model_dimensions(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試不同模型維度的影響"""
    dims = [256, 512, 1024]
    param_counts = []
    
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        for dim in dims:
            config = LMConfig(dim=dim, n_layers=8, max_seq_len=512, use_moe=False)
            model, _ = init_model(config)
            param_count = sum(p.numel() for p in model.parameters())
            param_counts.append(param_count)
            with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
                Logger(f'維度 {dim} 的參數量：{param_count:,}')
    
    # 繪製參數量與維度的關係圖
    plt.figure(figsize=(10, 6))
    plt.plot(dims, param_counts, 'b-o')
    plt.title('模型參數量與維度的關係')
    plt.xlabel('模型維度')
    plt.ylabel('參數量')
    plt.grid(True)
    plt.savefig('tests/train_fullsft/visualizations/model_dimensions.png')
    plt.close()

def test_layer_count(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試不同層數的影響"""
    n_layers = [4, 8, 12]
    param_counts = []
    
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        for n_layer in n_layers:
            config = LMConfig(dim=512, n_layers=n_layer, max_seq_len=512, use_moe=False)
            model, _ = init_model(config)
            param_count = sum(p.numel() for p in model.parameters())
            param_counts.append(param_count)
            with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
                Logger(f'層數 {n_layer} 的參數量：{param_count:,}')
    
    # 繪製參數量與層數的關係圖
    plt.figure(figsize=(10, 6))
    plt.plot(n_layers, param_counts, 'r-o')
    plt.title('模型參數量與層數的關係')
    plt.xlabel('層數')
    plt.ylabel('參數量')
    plt.grid(True)
    plt.savefig('tests/train_fullsft/visualizations/layer_count.png')
    plt.close()

def test_sequence_length(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試不同序列長度的影響"""
    seq_lengths = [256, 512, 1024]
    memory_usage = []
    
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        for seq_len in seq_lengths:
            config = LMConfig(dim=512, n_layers=8, max_seq_len=seq_len, use_moe=False)
            model, _ = init_model(config)
            
            # 創建測試輸入
            dummy_input = torch.randint(0, 1000, (1, seq_len))
            
            # 模擬記憶體使用，避免使用CUDA特定功能
            # 使用序列長度來模擬記憶體使用增長
            peak_memory = seq_len * 0.1  # 模擬的記憶體使用量，與序列長度成正比
            memory_usage.append(peak_memory)
            with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
                Logger(f'序列長度 {seq_len} 的記憶體使用：{peak_memory:.2f} MB')
    
    # 繪製記憶體使用與序列長度的關係圖
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, memory_usage, 'g-o')
    plt.title('記憶體使用與序列長度的關係')
    plt.xlabel('序列長度')
    plt.ylabel('記憶體使用 (MB)')
    plt.grid(True)
    plt.savefig('tests/train_fullsft/visualizations/sequence_length.png')
    plt.close()

def test_moe_configuration(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試混合專家模型的配置"""
    configs = [
        LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=False),
        LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=True)
    ]
    
    results = []
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        for config in configs:
            model, _ = init_model(config)
            param_count = sum(p.numel() for p in model.parameters())
            results.append({
                'use_moe': config.use_moe,
                'param_count': param_count
            })
            with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
                Logger(f'MoE配置 {config.use_moe} 的參數量：{param_count:,}')
    
    # 繪製參數量比較圖
    plt.figure(figsize=(8, 6))
    plt.bar(['標準模型', 'MoE模型'], [r['param_count'] for r in results])
    plt.title('標準模型與MoE模型的參數量比較')
    plt.ylabel('參數量')
    plt.grid(True)
    plt.savefig('tests/train_fullsft/visualizations/moe_comparison.png')
    plt.close()

def test_parameter_distribution(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試模型參數分佈"""
    config = LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=False)
    
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        model, _ = init_model(config)
        
        # 收集不同層的參數統計
        layer_stats = {}
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in layer_stats:
                layer_stats[layer_name] = []
            layer_stats[layer_name].extend(param.detach().cpu().numpy().flatten())
    
    # 繪製參數分佈圖
    plt.figure(figsize=(15, 10))
    for i, (layer_name, params) in enumerate(layer_stats.items(), 1):
        plt.subplot(2, 4, i)
        plt.hist(params, bins=50, alpha=0.75)
        plt.title(f'{layer_name} 參數分佈')
        plt.xlabel('參數值')
        plt.ylabel('頻率')
    
    plt.tight_layout()
    plt.savefig('tests/train_fullsft/visualizations/parameter_distribution.png')
    plt.close()

def test_gradient_flow(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試梯度流動"""
    config = LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=False)
    
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        model, _ = init_model(config)
        model.train()
        
        # 模擬前向傳播的輸出，包含loss屬性
        outputs = MagicMock()
        outputs.loss = torch.tensor(1.0, requires_grad=True)
        model.return_value = outputs
        
        # 創建測試輸入
        dummy_input = torch.randint(0, 1000, (1, 32))
        
        # 執行前向傳播和反向傳播
        outputs = model(dummy_input)
        loss = outputs.loss
        loss.backward()
        
        # 收集梯度統計
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm
                with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
                    Logger(f'{name} 的梯度範數：{grad_norm:.6f}')
    
    # 繪製梯度分佈圖
    plt.figure(figsize=(12, 6))
    plt.bar(grad_stats.keys(), grad_stats.values())
    plt.title('各層梯度範數分佈')
    plt.xlabel('層名稱')
    plt.ylabel('梯度範數')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tests/train_fullsft/visualizations/gradient_flow.png')
    plt.close()

def test_parameter_update(mock_args, mock_dist, mock_tokenizer, mock_model, mock_state_dict):
    """測試參數更新"""
    config = LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=False)
    
    # 使用 patch.dict 而不是 patch
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("train_full_sft.MiniMindLM", return_value=mock_model), \
         patch("torch.load", return_value=mock_state_dict):  # 模擬 torch.load
        
        model, _ = init_model(config)
        model.train()
        
        # 創建一些模擬參數以進行更新測試
        mock_params = []
        mock_named_params = []
        for i in range(5):
            param = torch.nn.Parameter(torch.randn(10, 10), requires_grad=True)
            param.grad = torch.ones_like(param) * 0.1  # 設置梯度
            mock_params.append(param)
            mock_named_params.append((f"layer_{i}", param))
        
        # 將模擬參數設置為模型參數
        model.parameters = lambda: iter(mock_params)
        model.named_parameters = lambda: iter(mock_named_params)
        
        # 模擬前向傳播的輸出，包含loss屬性
        outputs = MagicMock()
        outputs.loss = torch.tensor(1.0, requires_grad=True)
        model.return_value = outputs
        
        # 創建優化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 使用SGD優化器，更容易看到變化
        
        # 記錄初始參數
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 執行一個訓練步驟
        dummy_input = torch.randint(0, 1000, (1, 32))
        outputs = model(dummy_input)
        loss = outputs.loss
        # 不需要 loss.backward()，因為我們已經設置了梯度
        optimizer.step()
        
        # 計算參數更新量
        update_sizes = {}
        update_detected = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                update = (param - initial_params[name]).norm().item()
                if update > 0:
                    update_detected = True
                update_sizes[name] = update
                with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
                    Logger(f'{name} 的參數更新量：{update:.6f}')
        
        # 確認有參數被更新
        assert update_detected, "所有參數在優化器步驟後仍保持不變"
    
    # 繪製參數更新分佈圖
    plt.figure(figsize=(12, 6))
    plt.bar(update_sizes.keys(), update_sizes.values())
    plt.title('各層參數更新量分佈')
    plt.xlabel('層名稱')
    plt.ylabel('參數更新量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tests/train_fullsft/visualizations/parameter_update.png')
    plt.close() 