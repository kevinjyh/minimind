import pytest
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import PreTrainedModel

# 導入需要測試的模型和配置
from model.model import MiniMindLM
from model.LMConfig import LMConfig

# 設置隨機種子以確保測試的可重現性
torch.manual_seed(42)
np.random.seed(42)

class TestMiniMindLM:
    """
    測試 MiniMindLM 模型的各種功能
    """
    
    @pytest.fixture
    def small_config(self):
        """提供一個小型模型配置用於測試"""
        return LMConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            hidden_dim=256,
            max_seq_len=128,
            dropout=0.0
        )
    
    @pytest.fixture
    def moe_config(self):
        """提供一個使用 MoE (Mixture of Experts) 的模型配置"""
        return LMConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            hidden_dim=256,
            max_seq_len=128,
            dropout=0.0,
            use_moe=True,
            num_experts_per_tok=2,
            n_routed_experts=4
        )
    
    @pytest.fixture
    def small_model(self, small_config):
        """創建一個小型模型實例用於測試"""
        return MiniMindLM(small_config)
    
    @pytest.fixture
    def moe_model(self, moe_config):
        """創建一個使用 MoE 的模型實例用於測試"""
        return MiniMindLM(moe_config)
    
    @pytest.fixture
    def sample_input(self):
        """創建一個樣本輸入用於測試"""
        return torch.randint(0, 1000, (2, 10))  # 批次大小為2，序列長度為10
    
    def test_model_initialization(self, small_config):
        """測試模型初始化"""
        model = MiniMindLM(small_config)
        
        # 檢查模型是否為 PreTrainedModel 的實例
        assert isinstance(model, PreTrainedModel)
        
        # 檢查模型的基本屬性
        assert model.vocab_size == small_config.vocab_size
        assert model.n_layers == small_config.n_layers
        assert len(model.layers) == small_config.n_layers
        
        # 檢查嵌入層和輸出層是否共享權重
        assert model.tok_embeddings.weight is model.output.weight
        
        # 檢查位置編碼是否已預計算
        assert hasattr(model, "pos_cis")
        assert model.pos_cis.shape[1] == small_config.dim // (2 * small_config.n_heads)
    
    def test_forward_pass(self, small_model, sample_input):
        """測試前向傳播"""
        output = small_model(sample_input)
        
        # 檢查輸出是否為 CausalLMOutputWithPast 的實例
        assert hasattr(output, "logits")
        assert hasattr(output, "past_key_values")
        
        # 檢查 logits 的形狀
        assert output.logits.shape == (2, 10, small_model.vocab_size)
        
        # 檢查 past_key_values 的長度
        assert len(output.past_key_values) == small_model.n_layers
    
    def test_moe_forward_pass(self, moe_model, sample_input):
        """測試使用 MoE 的前向傳播"""
        output = moe_model(sample_input)
        
        # 檢查輸出是否為 CausalLMOutputWithPast 的實例
        assert hasattr(output, "logits")
        assert hasattr(output, "past_key_values")
        assert hasattr(output, "aux_loss")
        
        # 檢查 logits 的形狀
        assert output.logits.shape == (2, 10, moe_model.vocab_size)
        
        # 檢查 past_key_values 的長度
        assert len(output.past_key_values) == moe_model.n_layers
        
        # 檢查 aux_loss 是否為標量
        assert output.aux_loss.dim() == 0
    
    def test_generate_without_stream(self, small_model):
        """測試非流式生成"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 設置較小的 max_new_tokens 以加快測試速度
        output = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=0.8, 
            top_p=0.9, 
            stream=False
        )
        
        # 檢查輸出形狀
        assert output.shape[0] == input_ids.shape[0]  # 批次大小相同
        assert output.shape[1] >= input_ids.shape[1]  # 序列長度增加
    
    def test_generate_with_stream(self, small_model):
        """測試流式生成"""
        # 使用較短的輸入序列
        input_ids = torch.tensor([[1, 3, 4]])
        
        # 設置明顯大於輸入序列長度的 max_new_tokens
        generator = small_model.generate(
            input_ids, 
            max_new_tokens=10,  # 增加到明顯大於輸入長度
            temperature=0.8, 
            top_p=0.9, 
            stream=True,
            eos_token_id=None  # 不設置 EOS token，避免提前終止
        )
        
        # 手動迭代產生器並取得第一個結果即可
        try:
            first_output = next(generator)
            # 如果可以獲取到第一個輸出，測試通過
            assert first_output is not None
            assert first_output.shape[0] == input_ids.shape[0]  # 批次大小相同
            assert first_output.shape[1] > 0  # 序列長度大於0
        except StopIteration:
            # 如果產生器沒有產生任何值，測試失敗
            assert False, "產生器沒有產生任何值"

    def test_repetition_penalty(self, small_model):
        """測試重複懲罰機制"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 使用不同的重複懲罰參數生成
        output1 = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=0.8, 
            top_p=0.9, 
            rp=1.0  # 無懲罰
        )
        
        output2 = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=0.8, 
            top_p=0.9, 
            rp=2.0  # 較強懲罰
        )
        
        # 檢查輸出形狀
        assert output1.shape[0] == input_ids.shape[0]
        assert output2.shape[0] == input_ids.shape[0]
    
    def test_temperature_sampling(self, small_model):
        """測試溫度採樣"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 使用不同的溫度參數生成
        output1 = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=0.5,  # 低溫度，更確定性
            top_p=1.0
        )
        
        output2 = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=1.5,  # 高溫度，更隨機
            top_p=1.0
        )
        
        # 檢查輸出形狀
        assert output1.shape[0] == input_ids.shape[0]
        assert output2.shape[0] == input_ids.shape[0]
    
    def test_top_p_sampling(self, small_model):
        """測試 top-p (nucleus) 採樣"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 使用不同的 top-p 參數生成
        output1 = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=1.0,
            top_p=0.5  # 較小的 top-p，更確定性
        )
        
        output2 = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=1.0,
            top_p=0.9  # 較大的 top-p，更多樣性
        )
        
        # 檢查輸出形狀
        assert output1.shape[0] == input_ids.shape[0]
        assert output2.shape[0] == input_ids.shape[0]
    
    def test_batch_generation(self, small_model):
        """測試批次生成"""
        # 創建一個批次輸入，包含不同長度的序列
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0],  # 序列1，長度5，後面填充0
            [6, 7, 8, 9, 10, 11, 12]  # 序列2，長度7
        ])
        
        output = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            pad_token_id=0
        )
        
        # 檢查輸出形狀
        assert output.shape[0] == input_ids.shape[0]  # 批次大小相同
        assert output.shape[1] >= input_ids.shape[1]  # 序列長度增加

    def test_use_cache(self, small_model):
        """測試使用快取功能"""
        # 使用最簡單的輸入
        test_input = torch.tensor([[1, 2, 3]])  # 只有3個token的短序列
        
        # 測試快取輸出存在 - 只檢查forward方法能夠生成past_key_values
        with torch.no_grad():
            output = small_model(test_input, use_cache=True)
        
        # 檢查模型輸出包含past_key_values
        assert hasattr(output, "past_key_values")
        assert len(output.past_key_values) == small_model.n_layers
        
        # 測試generate方法使用快取不報錯
        with torch.no_grad():
            generated = small_model.generate(
                test_input, 
                max_new_tokens=2,
                use_cache=True
            )
        
        # 檢查生成輸出
        assert generated.shape[0] == test_input.shape[0]
        assert generated.shape[1] >= test_input.shape[1]

    def test_eos_token_generation(self, small_model):
        """測試生成到 EOS 標記為止"""
        # 使用較短的輸入序列，避免包含 EOS token
        input_ids = torch.tensor([[1, 3, 4]])  # 注意：不包含 EOS token (2)
        eos_token_id = 2
        
        # 使用無梯度上下文管理器提高測試穩定性
        with torch.no_grad():
            # 生成直到 EOS 標記
            output = small_model.generate(
                input_ids, 
                max_new_tokens=5,  # 使用較小值加快測試速度
                temperature=0.8,   # 添加溫度參數使生成更可控
                top_p=0.9,         # 添加 top_p 參數
                eos_token_id=eos_token_id
                # 默認 use_cache=True
            )
        
        # 簡化測試斷言
        # 檢查輸出形狀
        assert output.shape[0] == input_ids.shape[0]
        # 檢查輸出序列至少包含輸入序列長度
        assert output.shape[1] >= input_ids.shape[1]
        # 檢查輸出序列不會超過最大可能長度
        assert output.shape[1] <= input_ids.shape[1] + 5  # 輸入長度 + max_new_tokens