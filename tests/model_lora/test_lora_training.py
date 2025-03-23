import pytest
import torch
import sys
import os
from pathlib import Path
import tempfile

# 確保優先搜索項目根目錄
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from model.model_lora import LoRA, apply_lora, load_lora, save_lora


class TestLoRATraining:
    """測試 LoRA 在訓練情境中的表現"""
    
    @pytest.fixture
    def simple_training_model(self):
        """創建一個簡單的可訓練模型"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.act = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(20, 20)
                self.output = torch.nn.Linear(20, 1)
                self.device = "cpu"
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.act(x)
                x = self.linear2(x)
                x = self.act(x)
                return self.output(x)
                
        return SimpleModel()
    
    def test_lora_parameter_count(self, simple_training_model):
        """測試 LoRA 參數量相比原始模型有顯著減少"""
        model = simple_training_model
        
        # 計算原始模型參數量
        original_params = sum(p.numel() for p in model.parameters())
        
        # 應用 LoRA
        apply_lora(model, rank=4)
        
        # 獲取 LoRA 參數
        lora_params = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                lora_params += sum(p.numel() for p in module.lora.parameters())
        
        # 檢查 LoRA 參數是否少於原始參數
        assert lora_params < original_params
        print(f"原始參數: {original_params}, LoRA 參數: {lora_params}")
    
    def test_lora_training(self, simple_training_model):
        """測試僅訓練 LoRA 參數而凍結原始網絡"""
        model = simple_training_model
        apply_lora(model, rank=4)
        
        # 凍結原始模型參數
        for param in model.parameters():
            param.requires_grad = False
        
        # 啟用 LoRA 參數的梯度
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                for param in module.lora.parameters():
                    param.requires_grad = True
        
        # 創建簡單的訓練數據
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        # 定義優化器和損失函數
        optimizer = torch.optim.SGD([p for name, p in model.named_parameters() if p.requires_grad], lr=0.1)
        criterion = torch.nn.MSELoss()
        
        # 記錄初始參數
        original_params = {}
        lora_init_params = {}
        
        for name, param in model.named_parameters():
            if 'lora' not in name:
                original_params[name] = param.clone()
            else:
                lora_init_params[name] = param.clone()
        
        # 訓練模型
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # 檢查原始參數是否未變化
        for name, param in model.named_parameters():
            if 'lora' not in name:
                assert torch.allclose(param, original_params[name])
        
        # 檢查 LoRA 參數是否變化
        lora_params_changed = False
        for name, param in model.named_parameters():
            if 'lora' in name:
                if not torch.allclose(param, lora_init_params[name]):
                    lora_params_changed = True
                    break
        
        assert lora_params_changed, "LoRA 參數應該在訓練後發生變化"
    
    def test_fine_tuning_effect(self):
        """測試 LoRA 微調對模型效果的影響"""
        # 創建一個簡單的回歸任務
        X_train = torch.linspace(-5, 5, 100).reshape(-1, 1)
        y_train = torch.sin(X_train) + 0.1 * torch.randn_like(X_train)
        
        # 定義一個小的神經網絡
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1)
                )
                self.device = "cpu"
                
            def forward(self, x):
                return self.layers(x)
        
        # 創建預訓練模型並進行簡單訓練
        pretrained_model = SimpleNet()
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        for _ in range(100):
            optimizer.zero_grad()
            y_pred = pretrained_model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        
        # 複製預訓練模型用於全參數微調
        full_ft_model = SimpleNet()
        full_ft_model.load_state_dict(pretrained_model.state_dict())
        
        # 複製預訓練模型用於 LoRA 微調
        lora_model = SimpleNet()
        lora_model.load_state_dict(pretrained_model.state_dict())
        apply_lora(lora_model, rank=4)
        
        # 凍結 lora_model 原始參數
        for param in lora_model.parameters():
            param.requires_grad = False
        
        # 啟用 LoRA 參數的梯度
        for name, module in lora_model.named_modules():
            if hasattr(module, 'lora'):
                for param in module.lora.parameters():
                    param.requires_grad = True
        
        # 創建一個新數據集（代表領域轉換）
        X_new = torch.linspace(-5, 5, 100).reshape(-1, 1)
        y_new = torch.cos(X_new) + 0.1 * torch.randn_like(X_new)  # 使用餘弦函數代表新領域
        
        # 全參數微調
        full_optimizer = torch.optim.Adam(full_ft_model.parameters(), lr=0.01)
        
        # LoRA 微調
        lora_optimizer = torch.optim.Adam([p for p in lora_model.parameters() if p.requires_grad], lr=0.01)
        
        # 訓練兩個模型
        for _ in range(50):
            # 全參數微調
            full_optimizer.zero_grad()
            y_full_pred = full_ft_model(X_new)
            full_loss = criterion(y_full_pred, y_new)
            full_loss.backward()
            full_optimizer.step()
            
            # LoRA 微調
            lora_optimizer.zero_grad()
            y_lora_pred = lora_model(X_new)
            lora_loss = criterion(y_lora_pred, y_new)
            lora_loss.backward()
            lora_optimizer.step()
        
        # 評估三個模型在新數據上的表現
        with torch.no_grad():
            pretrained_loss = criterion(pretrained_model(X_new), y_new).item()
            full_ft_loss = criterion(full_ft_model(X_new), y_new).item()
            lora_loss = criterion(lora_model(X_new), y_new).item()
            
        # LoRA 微調應該比預訓練模型更好
        assert lora_loss < pretrained_loss
        
        # 計算參數數量
        full_param_count = sum(p.numel() for p in full_ft_model.parameters() if p.requires_grad)
        lora_param_count = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        # 輸出統計信息供分析
        print(f"預訓練模型損失: {pretrained_loss:.6f}")
        print(f"全參數微調損失: {full_ft_loss:.6f} (參數數量: {full_param_count})")
        print(f"LoRA 微調損失: {lora_loss:.6f} (參數數量: {lora_param_count})")
        
        # 比較參數效率
        assert lora_param_count < full_param_count
        print(f"LoRA 參數效率: {full_param_count / lora_param_count:.2f}x")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 