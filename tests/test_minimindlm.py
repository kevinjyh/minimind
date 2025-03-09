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
        assert model.pos_cis.shape[1] == small_config.dim // small_config.n_heads
    
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
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 設置較小的 max_new_tokens 以加快測試速度
        generator = small_model.generate(
            input_ids, 
            max_new_tokens=5, 
            temperature=0.8, 
            top_p=0.9, 
            stream=True
        )
        
        # 收集流式生成的所有輸出
        outputs = list(generator)
        
        # 檢查是否有輸出
        assert len(outputs) > 0
        
        # 檢查最後一個輸出的形狀
        assert outputs[-1].shape[0] == input_ids.shape[0]  # 批次大小相同
        assert outputs[-1].shape[1] > 0  # 序列長度大於0
    
    def test_use_cache(self, small_model, sample_input):
        """測試使用快取功能"""
        # 第一次前向傳播，啟用快取
        output1 = small_model(sample_input, use_cache=True)
        past_key_values = output1.past_key_values
        
        # 第二次前向傳播，使用快取
        output2 = small_model(
            sample_input[:, -1:],  # 只使用最後一個 token
            past_key_values=past_key_values,
            use_cache=True,
            start_pos=sample_input.shape[1] - 1
        )
        
        # 檢查輸出
        assert output2.logits.shape == (2, 1, small_model.vocab_size)
        assert len(output2.past_key_values) == small_model.n_layers
    
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
    
    def test_eos_token_generation(self, small_model):
        """測試生成到 EOS 標記為止"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        eos_token_id = 2  # 使用2作為 EOS 標記
        
        # 生成直到 EOS 標記
        output = small_model.generate(
            input_ids, 
            max_new_tokens=20,  # 設置較大的最大新標記數
            eos_token_id=eos_token_id
        )
        
        # 檢查輸出形狀
        assert output.shape[0] == input_ids.shape[0]