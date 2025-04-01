import json
import os
import pytest
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset
from transformers import AutoTokenizer
from train_full_sft import train_epoch, Logger

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
def tokenizer():
    return AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

@pytest.fixture
def sample_data(tmp_path):
    """創建測試數據"""
    # 調整數據格式以符合SFTDataset的要求，使用'conversations'字段
    data = [
        {
            "conversations": [
                {"role": "human", "content": "你好，這是一個測試。"},
                {"role": "assistant", "content": "是的，這是一個測試。"}
            ]
        },
        {
            "conversations": [
                {"role": "human", "content": "今天天氣很好。"},
                {"role": "assistant", "content": "是的，天氣確實很好。"}
            ]
        },
        {
            "conversations": [
                {"role": "human", "content": "機器學習很有趣。"},
                {"role": "assistant", "content": "是的，機器學習非常有趣。"}
            ]
        }
    ]
    
    data_file = tmp_path / "test_data.jsonl"
    with open(data_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return str(data_file)

@pytest.fixture
def model(lm_config, device):
    model = MiniMindLM(lm_config)
    model = model.to(device)
    return model

@pytest.fixture
def train_loader(lm_config, tokenizer, sample_data):
    dataset = SFTDataset(sample_data, tokenizer, max_length=lm_config.max_seq_len)
    return DataLoader(dataset, batch_size=2, shuffle=True)

def test_training_step(model, train_loader, device):
    """測試單個訓練步驟"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # 獲取一個批次
    batch = next(iter(train_loader))
    X, Y, loss_mask = [b.to(device) for b in batch]
    
    # 執行前向傳播
    outputs = model(X)
    
    # 手動計算損失 - MiniMindLM返回的是帶有logits屬性的對象
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        Y.view(-1)
    ).view(Y.size())
    
    # 應用損失掩碼
    loss = (loss * loss_mask).sum() / loss_mask.sum()
    
    # 如果有輔助損失，加上它
    if hasattr(outputs, 'aux_loss'):
        loss += outputs.aux_loss
    
    # 驗證損失值
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    
    # 執行反向傳播
    scaler.scale(loss).backward()
    
    # 驗證梯度
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
    
    # 執行優化步驟
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

def test_training_epoch(model, train_loader, device, tmp_path):
    """測試完整訓練epoch"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # 記錄訓練過程
    losses = []
    learning_rates = []
    
    # 執行一個epoch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X, Y, loss_mask = [b.to(device) for b in (X, Y, loss_mask)]
        
        # 前向傳播
        outputs = model(X)
        
        # 手動計算損失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        
        # 應用損失掩碼
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        # 如果有輔助損失，加上它
        if hasattr(outputs, 'aux_loss'):
            loss += outputs.aux_loss
        
        # 記錄損失和學習率
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 反向傳播和優化
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # 繪製訓練曲線
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('訓練損失變化')
    plt.xlabel('訓練步數')
    plt.ylabel('損失值')
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates)
    plt.title('學習率變化')
    plt.xlabel('訓練步數')
    plt.ylabel('學習率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(tmp_path, 'training_curves.png'))
    plt.close()

def test_gradient_clipping(model, train_loader, device):
    """測試梯度裁剪"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # 獲取一個批次
    batch = next(iter(train_loader))
    X, Y, loss_mask = [b.to(device) for b in batch]
    
    # 執行前向傳播
    outputs = model(X)
    
    # 手動計算損失
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        Y.view(-1)
    ).view(Y.size())
    
    # 應用損失掩碼
    loss = (loss * loss_mask).sum() / loss_mask.sum()
    
    # 如果有輔助損失，加上它
    if hasattr(outputs, 'aux_loss'):
        loss += outputs.aux_loss
    
    # 執行反向傳播
    scaler.scale(loss).backward()
    
    # 記錄裁剪前的梯度範數
    pre_clip_norms = []
    for param in model.parameters():
        if param.grad is not None:
            pre_clip_norms.append(param.grad.norm().item())
    
    # 執行梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 記錄裁剪後的梯度範數
    post_clip_norms = []
    for param in model.parameters():
        if param.grad is not None:
            post_clip_norms.append(param.grad.norm().item())
    
    # 驗證梯度裁剪效果
    assert all(norm <= 1.0 for norm in post_clip_norms)

def test_loss_calculation(model, train_loader, device):
    """測試損失計算"""
    model.train()
    
    # 獲取一個批次
    batch = next(iter(train_loader))
    X, Y, loss_mask = [b.to(device) for b in batch]
    
    # 執行前向傳播
    outputs = model(X)
    
    # 驗證輸出包含logits
    assert hasattr(outputs, 'logits')
    assert isinstance(outputs.logits, torch.Tensor)
    assert outputs.logits.requires_grad
    
    # 手動計算損失
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        Y.view(-1)
    ).view(Y.size())
    
    # 應用損失掩碼
    loss = (loss * loss_mask).sum() / loss_mask.sum()
    
    # 驗證損失計算
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    
    # 驗證輔助損失（如果有的話）
    if hasattr(outputs, 'aux_loss'):
        # aux_loss可能是張量也可能是數值
        aux_loss = outputs.aux_loss
        if isinstance(aux_loss, (int, float)):
            # 如果是數值型，將其轉換為張量
            aux_loss = torch.tensor(aux_loss, dtype=torch.float, device=device)
            # 驗證輔助損失可以被加到主損失上
            combined_loss = loss + aux_loss
            assert isinstance(combined_loss, torch.Tensor)
        else:
            # 如果已經是張量，直接驗證
            assert isinstance(aux_loss, torch.Tensor)

def test_model_update(model, train_loader, device):
    """測試模型參數更新"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # 記錄初始參數
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # 執行一個訓練步驟
    batch = next(iter(train_loader))
    X, Y, loss_mask = [b.to(device) for b in batch]
    
    # 前向傳播
    outputs = model(X)
    
    # 手動計算損失
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        Y.view(-1)
    ).view(Y.size())
    
    # 應用損失掩碼
    loss = (loss * loss_mask).sum() / loss_mask.sum()
    
    # 如果有輔助損失，加上它
    if hasattr(outputs, 'aux_loss'):
        loss += outputs.aux_loss
    
    # 反向傳播和優化
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    # 驗證參數更新
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param, initial_params[name])

def test_training_metrics(model, train_loader, device, tmp_path):
    """測試訓練指標"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # 記錄訓練指標
    metrics = {
        'loss': [],
        'grad_norm': [],
        'param_norm': []
    }
    
    # 執行一個epoch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X, Y, loss_mask = [b.to(device) for b in (X, Y, loss_mask)]
        
        # 前向傳播
        outputs = model(X)
        
        # 手動計算損失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        
        # 應用損失掩碼
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        # 如果有輔助損失，加上它
        if hasattr(outputs, 'aux_loss'):
            loss += outputs.aux_loss
        
        # 記錄損失
        metrics['loss'].append(loss.item())
        
        # 反向傳播
        scaler.scale(loss).backward()
        
        # 記錄梯度範數
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        metrics['grad_norm'].append(grad_norm)
        
        # 記錄參數範數
        param_norm = sum(p.norm().item() for p in model.parameters())
        metrics['param_norm'].append(param_norm)
        
        # 優化步驟
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # 繪製指標圖
    plt.figure(figsize=(15, 5))
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(1, 3, i)
        plt.plot(values)
        plt.title(f'{metric_name} 變化')
        plt.xlabel('訓練步數')
        plt.ylabel(metric_name)
    
    plt.tight_layout()
    plt.savefig(os.path.join(tmp_path, 'training_metrics.png'))
    plt.close() 