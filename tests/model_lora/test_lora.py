import pytest
import torch
import sys
import os
from pathlib import Path
import tempfile

# 修正：使用 insert(0) 確保優先搜索，並使用 absolute() 確保絕對路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from model.model_lora import LoRA, apply_lora, load_lora, save_lora

# LoRA 的全名是 "Low-Rank Adaptation"
class TestLoRA:
    """測試 LoRA 實現"""
    
    @pytest.fixture
    def sample_lora_layer(self):
        """創建一個測試用的 LoRA 層"""
        # 這裡的參數根據實際 LoRA 實現進行調整
        return LoRA(
            in_features=768,
            out_features=768,
            rank=8
        )
    
    def test_lora_initialization(self, sample_lora_layer):
        """測試 LoRA 層的初始化是否正確"""
        lora = sample_lora_layer
        
        # 測試結構
        assert lora.rank == 8
        assert lora.A.weight.shape == (8, 768)
        assert lora.B.weight.shape == (768, 8)
        
        # 測試初始化
        assert torch.allclose(torch.mean(lora.A.weight), torch.tensor(0.0), atol=1e-1)
        assert torch.std(lora.A.weight).item() < 0.03
        assert torch.all(lora.B.weight == 0)
    
    def test_lora_forward(self, sample_lora_layer):
        """測試 LoRA 的前向傳播"""
        lora = sample_lora_layer
        x = torch.randn(1, 768)
        
        # 執行前向傳播
        output = lora(x)
        
        # 驗證輸出形狀
        assert output.shape == (1, 768)
        
        # 驗證輸出是通過 A->B 的矩陣乘法計算的
        expected = lora.B(lora.A(x))
        assert torch.allclose(output, expected)
    
    def test_different_ranks(self):
        """測試不同秩值對 LoRA 的影響"""
        ranks = [4, 16, 32]
        in_features = 512
        out_features = 512
        
        for rank in ranks:
            lora = LoRA(in_features, out_features, rank)
            assert lora.A.weight.shape == (rank, in_features)
            assert lora.B.weight.shape == (out_features, rank)


class TestApplyLoRA:
    """測試 apply_lora 函數的功能"""
    
    @pytest.fixture
    def simple_model(self):
        """創建一個簡單的模型用於測試"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(256, 256)
                self.linear2 = torch.nn.Linear(256, 512)  # 非方形矩陣，不應被 LoRA 修改
                self.linear3 = torch.nn.Linear(512, 512)
                self.device = "cpu"
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return self.linear3(x)
        
        return SimpleModel()
    
    def test_apply_lora(self, simple_model):
        """測試 LoRA 應用到模型上"""
        model = simple_model
        original_forward1 = model.linear1.forward
        original_forward3 = model.linear3.forward
        
        # 應用 LoRA
        apply_lora(model, rank=12)
        
        # 檢查是否只有方形矩陣的層被應用了 LoRA
        assert hasattr(model.linear1, 'lora')
        assert not hasattr(model.linear2, 'lora')
        assert hasattr(model.linear3, 'lora')
        
        # 檢查 LoRA 是否具有正確的尺寸
        assert model.linear1.lora.rank == 12
        assert model.linear3.lora.rank == 12
        
        # 確認 forward 方法已被修改
        assert model.linear1.forward != original_forward1
        assert model.linear3.forward != original_forward3
    
    def test_lora_forward_with_model(self, simple_model):
        """測試應用 LoRA 後模型的前向傳播"""
        model = simple_model
        
        # 創建輸入
        x = torch.randn(1, 256)
        
        # 未應用 LoRA 前的輸出
        original_output = model(x.clone())
        
        # 應用 LoRA
        apply_lora(model, rank=8)
        
        # 應用 LoRA 後的輸出
        lora_output = model(x.clone())
        
        # 由於 LoRA 的 B 矩陣初始化為零，初始輸出應該相同
        # 但由於浮點運算，我們使用 allclose 而不是完全相等
        assert torch.allclose(original_output, lora_output)


class TestSaveLoadLoRA:
    """測試 LoRA 權重的保存和加載"""
    
    @pytest.fixture
    def model_with_lora(self):
        """創建一個應用了 LoRA 的模型"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(100, 100)
                self.device = "cpu"
        
        model = SimpleModel()
        apply_lora(model, rank=4)
        return model
    
    def test_save_load_lora(self, model_with_lora):
        """測試保存和加載 LoRA 權重"""
        model = model_with_lora
        
        # 修改 LoRA 權重以便我們可以檢測變化
        model.linear.lora.A.weight.data.fill_(0.1)
        model.linear.lora.B.weight.data.fill_(0.2)
        
        # 使用 delete=False 參數並手動刪除文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # 保存 LoRA 權重
            save_lora(model, temp_path)
            
            # 重置權重
            model.linear.lora.A.weight.data.fill_(0.0)
            model.linear.lora.B.weight.data.fill_(0.0)
            
            # 加載權重
            load_lora(model, temp_path)
            
            # 檢查權重是否正確加載
            assert torch.all(model.linear.lora.A.weight == 0.1)
            assert torch.all(model.linear.lora.B.weight == 0.2)
        finally:
            # 確保刪除臨時文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_partial_lora_loading(self):
        """測試部分 LoRA 層的加載"""
        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(64, 64)
                self.linear2 = torch.nn.Linear(64, 64)
                self.device = "cpu"
        
        # 創建兩個模型
        model1 = ComplexModel()
        model2 = ComplexModel()
        
        # 為兩個模型應用 LoRA
        apply_lora(model1, rank=4)
        apply_lora(model2, rank=4)
        
        # 修改第一個模型的權重
        model1.linear1.lora.A.weight.data.fill_(0.5)
        model1.linear2.lora.A.weight.data.fill_(0.3)
        
        # 使用 delete=False 並手動管理文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # 保存第一個模型的 LoRA 權重
            save_lora(model1, temp_path)
            
            # 加載到第二個模型
            load_lora(model2, temp_path)
            
            # 檢查第二個模型的權重是否正確加載
            assert torch.all(model2.linear1.lora.A.weight == 0.5)
            assert torch.all(model2.linear2.lora.A.weight == 0.3)
        finally:
            # 確保刪除臨時文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
