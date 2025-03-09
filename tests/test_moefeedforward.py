import pytest
import torch
import torch.nn as nn
import sys
import os
from typing import Tuple, List, Optional

# 添加模型目錄到路徑中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.LMConfig import LMConfig
from model.model import MOEFeedForward, FeedForward, MoEGate


class TestMOEFeedForward:
    """MOEFeedForward類測試用例"""
    
    @pytest.fixture
    def basic_config(self):
        """基本測試配置"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            hidden_dim=1024,
            use_moe=True,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts=True,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    
    @pytest.fixture
    def no_shared_expert_config(self):
        """無共享專家的配置"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            hidden_dim=1024,
            use_moe=True,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts=None,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    
    def test_init(self, basic_config):
        """測試MOEFeedForward的初始化"""
        moe_ff = MOEFeedForward(basic_config)
        
        # 檢查專家數量
        assert len(moe_ff.experts) == basic_config.n_routed_experts
        
        # 檢查是否包含門控機制
        assert isinstance(moe_ff.gate, MoEGate)
        
        # 檢查是否初始化了共享專家
        assert hasattr(moe_ff, "shared_experts")
        assert isinstance(moe_ff.shared_experts, FeedForward)
    
    def test_init_no_shared(self, no_shared_expert_config):
        """測試沒有共享專家的MOEFeedForward初始化"""
        moe_ff = MOEFeedForward(no_shared_expert_config)
        
        # 檢查是否不包含共享專家
        assert not hasattr(moe_ff, "shared_experts") or moe_ff.shared_experts is None
    
    def test_forward_training_mode(self, basic_config):
        """測試訓練模式下的前向傳播"""
        moe_ff = MOEFeedForward(basic_config)
        moe_ff.train()  # 設置為訓練模式
        
        # 創建輸入張量 [batch_size, seq_len, dim]
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, basic_config.dim)
        
        # 前向傳播
        output = moe_ff(x)
        
        # 檢查輸出形狀
        assert output.shape == x.shape
        
        # 檢查是否有輔助損失
        assert hasattr(moe_ff, "aux_loss")
    
    def test_forward_eval_mode(self, basic_config):
        """測試評估模式下的前向傳播"""
        moe_ff = MOEFeedForward(basic_config)
        moe_ff.eval()  # 設置為評估模式
        
        # 創建輸入張量 [batch_size, seq_len, dim]
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, basic_config.dim)
        
        # 前向傳播
        with torch.no_grad():
            output = moe_ff(x)
        
        # 檢查輸出形狀
        assert output.shape == x.shape
    
    def test_forward_no_shared_experts(self, no_shared_expert_config):
        """測試沒有共享專家時的前向傳播"""
        moe_ff = MOEFeedForward(no_shared_expert_config)
        
        # 創建輸入張量 [batch_size, seq_len, dim]
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, no_shared_expert_config.dim)
        
        # 前向傳播
        output = moe_ff(x)
        
        # 檢查輸出形狀
        assert output.shape == x.shape
    
    def test_expert_selection(self, basic_config):
        """測試專家選擇機制"""
        moe_ff = MOEFeedForward(basic_config)
        moe_ff.train()  # 設置為訓練模式
        
        # 創建輸入張量 [batch_size, seq_len, dim]
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, basic_config.dim)
        
        # 獲取門控機制的輸出
        with torch.no_grad():
            # 模擬門控機制
            topk_idx, topk_weight, aux_loss = moe_ff.gate(x)
        
        # 檢查topk_idx和topk_weight的形狀
        # 根據實際實現，topk_idx 的形狀應該是 [batch_size * seq_len, num_experts_per_tok]
        expected_shape = (batch_size * seq_len, basic_config.num_experts_per_tok)
        assert topk_idx.shape == expected_shape
        assert topk_weight.shape == expected_shape
        
        # 檢查權重總和是否為1
        weight_sums = topk_weight.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums))
    
    def test_moe_infer_method(self, basic_config):
        """測試moe_infer方法"""
        moe_ff = MOEFeedForward(basic_config)
        moe_ff.eval()  # 設置為評估模式
        
        # 創建輸入張量 [batch_size*seq_len, dim]
        batch_size, seq_len = 2, 4
        x_flat = torch.randn(batch_size * seq_len, basic_config.dim)
        
        # 創建專家索引和權重
        flat_expert_indices = torch.randint(
            0, basic_config.n_routed_experts, 
            (batch_size * seq_len * basic_config.num_experts_per_tok,)
        )
        flat_expert_weights = torch.rand(batch_size * seq_len * basic_config.num_experts_per_tok, 1)
        flat_expert_weights = flat_expert_weights / flat_expert_weights.sum()
        
        # 調用moe_infer方法
        with torch.no_grad():
            output = moe_ff.moe_infer(x_flat, flat_expert_indices, flat_expert_weights)
        
        # 檢查輸出形狀
        assert output.shape == x_flat.shape
    
    def test_parameter_impact(self, basic_config):
        """測試不同參數配置對MOEFeedForward的影響"""
        # 測試不同專家數量
        configs = []
        for n_experts in [2, 4, 8]:
            for experts_per_tok in [1, 2]:
                config = LMConfig(
                    dim=512,
                    n_layers=8,
                    n_heads=8,
                    hidden_dim=1024,
                    use_moe=True,
                    num_experts_per_tok=experts_per_tok,
                    n_routed_experts=n_experts,
                    n_shared_experts=True,
                    aux_loss_alpha=0.1,
                    seq_aux=True,
                    norm_topk_prob=True
                )
                configs.append(config)
        
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, basic_config.dim)
        
        for config in configs:
            moe_ff = MOEFeedForward(config)
            moe_ff.eval()
            
            with torch.no_grad():
                output = moe_ff(x)
            
            assert output.shape == x.shape
    
    def test_aux_loss(self, basic_config):
        """測試輔助損失計算"""
        # 測試seq_aux=True的情況
        config_seq_aux = basic_config
        
        # 測試seq_aux=False的情況
        config_no_seq_aux = LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            hidden_dim=1024,
            use_moe=True,
            num_experts_per_tok=2,
            n_routed_experts=4,
            n_shared_experts=True,
            aux_loss_alpha=0.1,
            seq_aux=False,
            norm_topk_prob=True
        )
        
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, basic_config.dim)
        
        # 測試seq_aux=True的輔助損失
        moe_ff_seq_aux = MOEFeedForward(config_seq_aux)
        moe_ff_seq_aux.train()
        _ = moe_ff_seq_aux(x)
        aux_loss_seq = moe_ff_seq_aux.aux_loss
        
        # 測試seq_aux=False的輔助損失
        moe_ff_no_seq_aux = MOEFeedForward(config_no_seq_aux)
        moe_ff_no_seq_aux.train()
        _ = moe_ff_no_seq_aux(x)
        aux_loss_no_seq = moe_ff_no_seq_aux.aux_loss
        
        # 驗證輔助損失是浮點數
        assert isinstance(aux_loss_seq, torch.Tensor) or isinstance(aux_loss_seq, float)
        assert isinstance(aux_loss_no_seq, torch.Tensor) or isinstance(aux_loss_no_seq, float) 