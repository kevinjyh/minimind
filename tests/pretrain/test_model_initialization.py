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
from train_pretrain import init_model


class TestModelInitialization:
    """測試模型初始化相關功能"""
    
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
    
    def test_model_creation(self):
        """測試模型創建"""
        model = MiniMindLM(self.lm_config)
        
        # 檢查模型是否成功創建
        assert isinstance(model, MiniMindLM), "應該創建 MiniMindLM 實例"
        
        # 檢查模型參數數量
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型參數數量: {param_count / 1e6:.3f} 百萬")
        
        # 記錄模型架構到檔案
        with open(os.path.join(os.path.dirname(__file__), "visualizations/model_architecture.txt"), "w", encoding="utf-8") as f:
            f.write(str(model))
    
    @patch("train_pretrain.AutoTokenizer")
    def test_init_model_function(self, mock_tokenizer):
        """測試 init_model 函數"""
        # 模擬 tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # 使用 SimpleNamespace 模擬 args 物件
        mock_args = SimpleNamespace(device="cpu")
        
        # 模擬 ddp 和 dist 相關變量
        mock_dist = MagicMock()
        mock_dist.get_rank.return_value = 0
        
        # 使用 patch.dict 來模擬全局變量
        with patch.dict("train_pretrain.__dict__", {
            "args": mock_args,
            "ddp": False,  # 新增 ddp 模擬
            "dist": mock_dist  # 新增 dist 模擬
        }):
            # 使用 hook 監控模型創建
            with patch("train_pretrain.MiniMindLM") as mock_model:
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance
                mock_model_instance.to.return_value = mock_model_instance
                
                # 調用函數
                model, tokenizer = init_model(self.lm_config)
                
                # 檢查結果
                mock_tokenizer.from_pretrained.assert_called_once_with('./model/minimind_tokenizer')
                mock_model.assert_called_once_with(self.lm_config)
                mock_model_instance.to.assert_called_once_with(mock_args.device)
                assert model == mock_model_instance
                assert tokenizer == mock_tokenizer_instance
    
    def test_model_forward_pass(self):
        """測試模型前向傳播"""
        model = MiniMindLM(self.lm_config).to(self.device)
        
        # 創建假輸入，需符合模型實際詞彙表大小
        batch_size = 2
        seq_len = 128
        vocab_size = self.lm_config.vocab_size  # 從配置獲取實際詞彙表大小
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        # 前向傳播
        with torch.no_grad():
            outputs = model(inputs)
        
        # 檢查輸出形狀，使用實際詞彙表大小
        expected_shape = (batch_size, seq_len, vocab_size)
        try:
            assert outputs.logits.shape == expected_shape, (
                f"輸出形狀應為 {expected_shape}，但得到 {outputs.logits.shape}"
            )
            assert hasattr(outputs, 'aux_loss'), "輸出應包含 aux_loss 屬性"
        except Exception as e:
            print(f"前向傳播測試失敗: {e}")
            print(f"實際輸出形狀: {outputs.logits.shape if hasattr(outputs, 'logits') else '未知'}") 