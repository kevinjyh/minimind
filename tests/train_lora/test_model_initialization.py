import os
import sys
import pytest
import torch
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import LoRA, apply_lora, save_lora, load_lora
from train_lora import init_model


class TestLoRAModelInitialization:
    """測試 LoRA 模型初始化相關功能"""
    
    def setup_method(self):
        """測試前設置"""
        self.dim = 512
        self.n_layers = 8
        self.max_seq_len = 512
        self.use_moe = False
        self.lm_config = LMConfig(
            dim=self.dim, 
            n_layers=self.n_layers, 
            max_seq_len=self.max_seq_len,
            use_moe=self.use_moe
        )
        
        # 確保測試在 CPU 上運行，避免 GPU 相依性
        self.device = torch.device("cpu")
    
    @pytest.mark.parametrize("dim,n_layers,max_seq_len,use_moe", [
        (512, 8, 512, False),
        (768, 12, 1024, False),
        (512, 8, 512, True),
    ])
    def test_lm_config_initialization(self, dim, n_layers, max_seq_len, use_moe):
        """測試不同參數下的 LMConfig 初始化"""
        config = LMConfig(dim=dim, n_layers=n_layers, max_seq_len=max_seq_len, use_moe=use_moe)
        
        assert config.dim == dim, f"期望 dim 為 {dim}，但得到 {config.dim}"
        assert config.n_layers == n_layers, f"期望 n_layers 為 {n_layers}，但得到 {config.n_layers}"
        assert config.max_seq_len == max_seq_len, f"期望 max_seq_len 為 {max_seq_len}，但得到 {config.max_seq_len}"
        assert config.use_moe == use_moe, f"期望 use_moe 為 {use_moe}，但得到 {config.use_moe}"
    
    def test_lora_module_creation(self):
        """測試 LoRA 模組創建"""
        in_features = 512
        out_features = 512
        rank = 16
        
        lora = LoRA(in_features, out_features, rank)
        
        # 檢查 LoRA 模組是否正確創建
        assert isinstance(lora, LoRA), "應該創建 LoRA 實例"
        assert lora.rank == rank, f"期望 rank 為 {rank}，但得到 {lora.rank}"
        
        # 檢查 LoRA 參數形狀
        assert lora.A.weight.shape == (rank, in_features), f"A 權重形狀應為 ({rank}, {in_features})"
        assert lora.B.weight.shape == (out_features, rank), f"B 權重形狀應為 ({out_features}, {rank})"
        
        # 檢查初始化權重
        assert torch.mean(lora.A.weight).item() != 0, "A 權重應使用高斯初始化"
        assert torch.all(lora.B.weight == 0), "B 權重應使用全零初始化"
        
        # 記錄檢查結果到檔案
        os.makedirs(os.path.join(os.path.dirname(__file__), "visualizations"), exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), "visualizations/lora_module_check.txt"), "w", encoding="utf-8") as f:
            f.write(f"LoRA 模組初始化成功\n")
            f.write(f"輸入特徵數: {in_features}\n")
            f.write(f"輸出特徵數: {out_features}\n")
            f.write(f"秩(Rank): {rank}\n")
            f.write(f"A 權重形狀: {lora.A.weight.shape}\n")
            f.write(f"B 權重形狀: {lora.B.weight.shape}\n")
            f.write(f"A 權重均值: {torch.mean(lora.A.weight).item()}\n")
            f.write(f"A 權重標準差: {torch.std(lora.A.weight).item()}\n")
            f.write(f"B 權重均值: {torch.mean(lora.B.weight).item()}\n")
    
    def test_lora_forward_pass(self):
        """測試 LoRA 模組前向傳播"""
        in_features = 512
        out_features = 512
        rank = 16
        batch_size = 2
        
        lora = LoRA(in_features, out_features, rank)
        
        # 創建隨機輸入
        x = torch.randn(batch_size, in_features)
        
        # 前向傳播
        output = lora(x)
        
        # 檢查輸出形狀
        assert output.shape == (batch_size, out_features), f"輸出形狀應為 ({batch_size}, {out_features})"
        
        # 手動計算檢查前向傳播
        expected_output = lora.B(lora.A(x))
        assert torch.allclose(output, expected_output), "前向傳播結果與預期不符"
        
        # 記錄檢查結果到檔案
        with open(os.path.join(os.path.dirname(__file__), "visualizations/lora_forward_check.txt"), "w", encoding="utf-8") as f:
            f.write(f"LoRA 前向傳播測試成功\n")
            f.write(f"輸入形狀: {x.shape}\n")
            f.write(f"輸出形狀: {output.shape}\n")
            f.write(f"A 輸出形狀: {lora.A(x).shape}\n")
            f.write(f"輸出前 10 個值: {output[0, :10].detach().numpy()}\n")
    
    def test_apply_lora_to_model(self):
        """測試將 LoRA 應用到模型"""
        model = MiniMindLM(self.lm_config).to(self.device)
        
        # 記錄原始參數數量
        original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 應用 LoRA
        apply_lora(model, rank=16)
        
        # 計算 LoRA 參數數量
        lora_params_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                lora_params_count += sum(p.numel() for p in module.lora.parameters())
        
        # 檢查是否存在 LoRA 模組
        assert lora_params_count > 0, "應該存在 LoRA 參數"
        
        # 記錄結果到檔案
        with open(os.path.join(os.path.dirname(__file__), "visualizations/apply_lora_check.txt"), "w", encoding="utf-8") as f:
            f.write(f"LoRA 應用測試成功\n")
            f.write(f"原始模型參數數量: {original_params}\n")
            f.write(f"LoRA 參數數量: {lora_params_count}\n")
            f.write(f"LoRA 參數佔比: {lora_params_count / original_params * 100:.6f}%\n")
            
            # 記錄添加了 LoRA 的層
            f.write("\n添加了 LoRA 的層:\n")
            for name, module in model.named_modules():
                if hasattr(module, 'lora'):
                    f.write(f"- {name}: 輸入大小={module.weight.shape[1]}, 輸出大小={module.weight.shape[0]}\n")
    
    def test_save_load_lora(self):
        """測試保存和加載 LoRA 權重"""
        model = MiniMindLM(self.lm_config).to(self.device)
        apply_lora(model, rank=16)
        
        # 創建臨時檔案路徑
        temp_path = os.path.join(os.path.dirname(__file__), "visualizations/temp_lora.pth")
        
        # 保存 LoRA 權重
        save_lora(model, temp_path)
        
        # 檢查檔案是否存在
        assert os.path.exists(temp_path), "LoRA 權重檔案應該被創建"
        
        # 修改原始 LoRA 權重以檢查加載
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                module.lora.A.weight.data.fill_(1.0)
                module.lora.B.weight.data.fill_(0.5)
        
        # 加載 LoRA 權重
        load_lora(model, temp_path)
        
        # 檢查權重是否被正確恢復
        is_restored = True
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                if torch.mean(module.lora.A.weight).item() == 1.0 or torch.mean(module.lora.B.weight).item() == 0.5:
                    is_restored = False
                    break
        
        assert is_restored, "LoRA 權重應該被正確加載"
        
        # 記錄結果到檔案
        with open(os.path.join(os.path.dirname(__file__), "visualizations/lora_save_load_check.txt"), "w", encoding="utf-8") as f:
            f.write(f"LoRA 保存和加載測試成功\n")
            f.write(f"權重檔案路徑: {temp_path}\n")
            f.write(f"檔案大小: {os.path.getsize(temp_path) / 1024:.2f} KB\n")
            
            # 記錄加載後的權重統計
            f.write("\n加載後的 LoRA 權重統計:\n")
            for name, module in model.named_modules():
                if hasattr(module, 'lora'):
                    f.write(f"- {name}: \n")
                    f.write(f"  A 均值: {torch.mean(module.lora.A.weight).item()}\n")
                    f.write(f"  B 均值: {torch.mean(module.lora.B.weight).item()}\n")
    
    @patch("train_lora.AutoTokenizer")
    def test_init_model_function(self, mock_tokenizer):
        """測試 init_model 函數"""
        # 模擬 tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # 使用 SimpleNamespace 模擬 args 物件
        mock_args = SimpleNamespace(device="cpu")
        
        # 模擬 state_dict 和 load_state_dict
        mock_state_dict = MagicMock()
        
        # 使用 patch.dict 來模擬全局變量
        with patch.dict("train_lora.__dict__", {
            "args": mock_args,
            "ddp": False,
            "dist": MagicMock()
        }):
            # 使用 hook 監控模型創建與載入
            with patch("train_lora.MiniMindLM") as mock_model:
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance
                mock_model_instance.to.return_value = mock_model_instance
                
                # 模擬 torch.load
                with patch("torch.load", return_value=mock_state_dict):
                    # 調用函數
                    model, tokenizer = init_model(self.lm_config)
                    
                    # 檢查結果
                    mock_tokenizer.from_pretrained.assert_called_once_with('./model/minimind_tokenizer')
                    mock_model.assert_called_once_with(self.lm_config)
                    mock_model_instance.load_state_dict.assert_called_once_with(mock_state_dict, strict=False)
                    mock_model_instance.to.assert_called_once_with(mock_args.device)
                    assert model == mock_model_instance
                    assert tokenizer == mock_tokenizer_instance 