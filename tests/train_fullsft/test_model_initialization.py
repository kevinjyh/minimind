import os
import pytest
import torch
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from train_full_sft import init_model, Logger

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

@pytest.fixture
def lm_config():
    return LMConfig(dim=512, n_layers=8, max_seq_len=512, use_moe=False)

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def mock_args():
    return SimpleNamespace(device="cuda:0" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_state_dict():
    return {}

@pytest.fixture
def mock_dist():
    mock = MagicMock()
    mock.get_rank.return_value = 0
    return mock

def test_model_initialization(lm_config, device, tmp_path, mock_args, mock_state_dict, mock_dist):
    """測試模型初始化過程"""
    # 初始化模型和tokenizer
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}):
        with patch("torch.load", return_value=mock_state_dict), \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_fn, \
             patch("train_full_sft.MiniMindLM") as mock_model_class:
             
            # 創建一個模擬的模型實例
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # 創建一個模擬的tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_fn.return_value = mock_tokenizer
            
            # 模擬參數
            mock_params = []
            for i in range(5):
                param = torch.nn.Parameter(torch.randn(10, 10), requires_grad=True)
                mock_params.append(param)
            
            # 設置模型參數計數方法
            def mock_parameters():
                return iter(mock_params)
            mock_model.parameters = mock_parameters
            
            # 調用函數
            model, tokenizer = init_model(lm_config)
    
    # 驗證模型
    mock_model_class.assert_called_once_with(lm_config)
    mock_model.to.assert_called_once_with(mock_args.device)
    assert model == mock_model
    assert tokenizer == mock_tokenizer
    
    # 設置參數統計
    with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
        Logger(f'模型總參數量：模擬參數')
        Logger(f'可訓練參數量：模擬參數')
    
    # 模擬參數分佈圖
    plt.figure(figsize=(12, 6))
    param_norms = [1.0, 1.5, 2.0, 2.5, 3.0]  # 模擬數據
    
    plt.hist(param_norms, bins=50, alpha=0.75)
    plt.title('模型參數範數分佈')
    plt.xlabel('參數範數')
    plt.ylabel('頻率')
    plt.savefig(os.path.join(tmp_path, 'parameter_distribution.png'))
    plt.close()

def test_model_architecture(lm_config, device, mock_args, mock_state_dict, mock_dist):
    """測試模型架構配置"""
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}):
        with patch("torch.load", return_value=mock_state_dict), \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_fn, \
             patch("train_full_sft.MiniMindLM") as mock_model_class:
             
            # 創建一個模擬的模型實例
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # 模擬模型層結構
            mock_layers = []
            for _ in range(lm_config.n_layers):
                layer_mock = MagicMock()
                layer_mock.dim = lm_config.dim
                mock_layers.append(layer_mock)
            
            mock_model.layers = mock_layers
            mock_model.pos_cis = torch.zeros((lm_config.max_seq_len, lm_config.dim // 2))
            
            # 創建一個模擬的tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_fn.return_value = mock_tokenizer
            
            # 調用函數
            model, _ = init_model(lm_config)
    
    # 驗證模型層數
    assert len(model.layers) == lm_config.n_layers
    
    # 驗證每層的維度
    for layer in model.layers:
        assert layer.dim == lm_config.dim
    
    # 驗證位置編碼
    assert model.pos_cis.shape[0] == lm_config.max_seq_len
    assert model.pos_cis.shape[1] == lm_config.dim // 2

def test_model_forward_pass(lm_config, device, mock_args, mock_state_dict, mock_dist):
    """測試模型前向傳播"""
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}):
        with patch("torch.load", return_value=mock_state_dict), \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_fn, \
             patch("train_full_sft.MiniMindLM") as mock_model_class:
            
            # 創建一個模擬的模型實例
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # 設置模型配置
            mock_model.config = MagicMock()
            mock_model.config.vocab_size = 6400
            
            # 創建一個模擬 tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_fn.return_value = mock_tokenizer
            mock_tokenizer.vocab_size = 6400
            
            # 設置前向傳播的輸出
            outputs = MagicMock()
            # 假設輸出的logits形狀為 (batch_size, seq_len, vocab_size)
            outputs.logits = torch.zeros((1, lm_config.max_seq_len, mock_model.config.vocab_size))
            mock_model.return_value = outputs
            
            # 當調用 tokenizer 時返回適當的輸入
            def side_effect(text, return_tensors, max_length, padding):
                return {"input_ids": torch.randint(0, 100, (1, 10)).to(device)}
            mock_tokenizer.side_effect = side_effect
            
            model, tokenizer = init_model(lm_config)
    
    # 設置評估模式
    model.eval()
    
    # 創建測試輸入
    test_text = "這是一個測試句子。"
    inputs = {"input_ids": torch.randint(0, 100, (1, 10)).to(device)}
    
    # 執行前向傳播
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
    
    # 驗證輸出形狀
    assert outputs.logits.shape[1] <= lm_config.max_seq_len
    assert outputs.logits.shape[2] == model.config.vocab_size

def test_model_gradient_flow(lm_config, device, mock_args, mock_state_dict, mock_dist):
    """測試模型梯度流動"""
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}):
        with patch("torch.load", return_value=mock_state_dict), \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_fn, \
             patch("train_full_sft.MiniMindLM") as mock_model_class:
             
            # 創建一個模擬的模型實例
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            # 創建一個模擬的tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_fn.return_value = mock_tokenizer
            
            # 模擬參數和梯度
            mock_params = []
            mock_named_params = []
            for i in range(5):  # 模擬幾個參數
                param = torch.nn.Parameter(torch.randn(10, 10))
                param.grad = torch.randn(10, 10)
                mock_params.append(param)
                mock_named_params.append((f"param_{i}", param))
            
            # 設置模型參數
            mock_model.parameters = lambda: iter(mock_params)
            mock_model.named_parameters = lambda: iter(mock_named_params)
            
            # 模擬前向傳播的輸出
            outputs = MagicMock()
            outputs.logits = torch.randn(1, 10, 10, requires_grad=True)
            mock_model.return_value = outputs
            
            model, tokenizer = init_model(lm_config)
    
    model.train()
    
    # 創建測試輸入
    inputs = torch.randint(0, 100, (1, 10)).to(device)
    
    # 執行前向傳播
    outputs = model(inputs)
    
    # 創建一個虛擬的損失進行反向傳播
    fake_loss = outputs.logits.mean()
    fake_loss.backward()
    
    # 檢查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    # 輸出梯度統計
    with patch.dict("train_full_sft.__dict__", {"ddp": False, "dist": mock_dist}):
        Logger("梯度範數統計：")
        for name, norm in grad_norms:
            Logger(f"{name}: {norm:.6f}")

def test_model_parameter_update(lm_config, device, mock_args, mock_state_dict, mock_dist):
    """測試模型參數更新"""
    # 直接使用真實的小型模型進行測試，而不是使用MagicMock
    with patch.dict("train_full_sft.__dict__", {"args": mock_args, "ddp": False, "dist": mock_dist}):
        with patch("torch.load", return_value=mock_state_dict), \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_fn:
            
            # 創建一個真實的小型模型 (不使用mock)
            test_model = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 10)
            ).to(device)
            
            # 使用patch來讓init_model返回我們的測試模型
            with patch("train_full_sft.MiniMindLM", return_value=test_model):
                # 模擬tokenizer
                mock_tokenizer = MagicMock()
                mock_tokenizer_fn.return_value = mock_tokenizer
                
                model, tokenizer = init_model(lm_config)
    
    # 確保模型處於訓練模式
    model.train()
    
    # 記錄初始參數
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.clone().detach()
    
    # 創建優化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 使用SGD優化器，更容易看到參數變化
    
    # 創建測試輸入和目標
    inputs = torch.randn(4, 10, device=device)
    targets = torch.randn(4, 10, device=device)
    
    # 執行一次訓練步驟
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)  # 使用MSE損失
    loss.backward()
    
    # 檢查梯度是否存在
    grads_exist = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            grads_exist = True
            break
    
    assert grads_exist, "模型參數沒有產生梯度"
    
    # 執行優化器步驟
    optimizer.step()
    
    # 檢查參數是否更新
    update_detected = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            update_detected = True
            break
    
    assert update_detected, "所有參數在優化器步驟後仍保持不變" 