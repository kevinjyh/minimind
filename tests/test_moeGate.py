import pytest
import torch
import torch.nn as nn
import sys
import os
import math
from torch.nn import functional as F

# 添加專案根目錄到系統路徑，以便可以引入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.LMConfig import LMConfig
from model.model import MoEGate

# 固定隨機種子以確保測試的可重複性
@pytest.fixture(scope="module")
def set_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    return 42

class TestMoEGate:
    """測試 MoEGate 類的功能"""
    
    @pytest.fixture
    def default_config(self):
        """創建默認配置"""
        return LMConfig(
            dim=512,
            num_experts_per_tok=2,
            n_routed_experts=4,
            scoring_func='softmax',
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True,
            use_moe=True
        )
    
    @pytest.fixture
    def single_expert_config(self):
        """創建只選擇一個專家的配置"""
        return LMConfig(
            dim=512,
            num_experts_per_tok=1,
            n_routed_experts=4,
            scoring_func='softmax',
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True,
            use_moe=True
        )
    
    @pytest.fixture
    def no_aux_loss_config(self):
        """創建沒有輔助損失的配置"""
        return LMConfig(
            dim=512,
            num_experts_per_tok=2,
            n_routed_experts=4,
            scoring_func='softmax',
            aux_loss_alpha=0.0,  # 設置為0表示無輔助損失
            seq_aux=True,
            norm_topk_prob=True,
            use_moe=True
        )
    
    @pytest.fixture
    def no_norm_topk_config(self):
        """創建不進行topk概率歸一化的配置"""
        return LMConfig(
            dim=512,
            num_experts_per_tok=2,
            n_routed_experts=4,
            scoring_func='softmax',
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=False,  # 不進行歸一化
            use_moe=True
        )
    
    @pytest.fixture
    def token_aux_config(self):
        """創建使用token級別而非序列級別輔助損失的配置"""
        return LMConfig(
            dim=512,
            num_experts_per_tok=2,
            n_routed_experts=4,
            scoring_func='softmax',
            aux_loss_alpha=0.1,
            seq_aux=False,  # 使用token級別輔助損失
            norm_topk_prob=True,
            use_moe=True
        )
    
    def test_initialization(self, default_config, set_seed):
        """測試 MoEGate 初始化是否正確"""
        moe_gate = MoEGate(default_config)
        
        # 檢查實例化後的屬性是否符合預期
        assert moe_gate.top_k == default_config.num_experts_per_tok
        assert moe_gate.n_routed_experts == default_config.n_routed_experts
        assert moe_gate.scoring_func == default_config.scoring_func
        assert moe_gate.alpha == default_config.aux_loss_alpha
        assert moe_gate.seq_aux == default_config.seq_aux
        assert moe_gate.norm_topk_prob == default_config.norm_topk_prob
        assert moe_gate.gating_dim == default_config.dim
        
        # 檢查權重形狀是否正確
        assert moe_gate.weight.shape == (default_config.n_routed_experts, default_config.dim)
    
    def test_forward_shape(self, default_config, set_seed):
        """測試 forward 方法的輸出形狀是否正確"""
        moe_gate = MoEGate(default_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = default_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 檢查輸出形狀
        expected_topk_idx_shape = (batch_size * seq_len, default_config.num_experts_per_tok)
        expected_topk_weight_shape = (batch_size * seq_len, default_config.num_experts_per_tok)
        
        assert topk_idx.shape == expected_topk_idx_shape
        assert topk_weight.shape == expected_topk_weight_shape
        assert isinstance(aux_loss, torch.Tensor) and aux_loss.numel() == 1
    
    def test_forward_values_range(self, default_config, set_seed):
        """測試 forward 方法的輸出值範圍是否正確"""
        moe_gate = MoEGate(default_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = default_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 檢查索引範圍
        assert topk_idx.min() >= 0
        assert topk_idx.max() < default_config.n_routed_experts
        
        # 檢查權重範圍（應該在0到1之間，且總和接近1）
        assert topk_weight.min() >= 0
        assert topk_weight.max() <= 1
        
        if default_config.norm_topk_prob:
            # 如果設置了規範化，則每個token的專家權重總和應接近1
            row_sums = topk_weight.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
        
        # 檢查輔助損失是否為非負數
        assert aux_loss >= 0
    
    def test_single_expert(self, single_expert_config, set_seed):
        """測試當每個token只選一個專家時的行為"""
        moe_gate = MoEGate(single_expert_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = single_expert_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 檢查形狀
        assert topk_idx.shape == (batch_size * seq_len, 1)
        assert topk_weight.shape == (batch_size * seq_len, 1)
        
        # 當只選一個專家時，權重值應該在 0 到 1 之間
        # 注意：當 top_k=1 時，權重不會被歸一化為 1，而是保持 softmax 的輸出值
        assert topk_weight.min() >= 0
        assert topk_weight.max() <= 1
    
    def test_no_aux_loss(self, no_aux_loss_config, set_seed):
        """測試無輔助損失時的行為"""
        moe_gate = MoEGate(no_aux_loss_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = no_aux_loss_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 檢查輔助損失是否為0
        assert aux_loss == 0
    
    def test_train_vs_eval_mode(self, default_config, set_seed):
        """測試訓練模式和評估模式的差異"""
        moe_gate = MoEGate(default_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = default_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 訓練模式
        moe_gate.train()
        _, _, train_aux_loss = moe_gate(hidden_states)
        
        # 評估模式
        moe_gate.eval()
        _, _, eval_aux_loss = moe_gate(hidden_states)
        
        # 檢查在評估模式下輔助損失是否為0
        assert eval_aux_loss == 0
    
    def test_token_aux_loss(self, token_aux_config, set_seed):
        """測試token級別輔助損失的行為"""
        moe_gate = MoEGate(token_aux_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = token_aux_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 確保輔助損失存在且為標量
        assert isinstance(aux_loss, torch.Tensor)
        assert aux_loss.numel() == 1
        assert aux_loss >= 0
    
    def test_no_norm_topk(self, no_norm_topk_config, set_seed):
        """測試不進行topk規範化時的行為"""
        moe_gate = MoEGate(no_norm_topk_config)
        
        batch_size = 2
        seq_len = 3
        hidden_dim = no_norm_topk_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 檢查權重值的範圍（應該在0到1之間，但總和可能不等於1）
        assert topk_weight.min() >= 0
        assert topk_weight.max() <= 1
        
        # 檢查加總是否不等於1（由於softmax僅在專家維度上操作，選擇前k個後不進行歸一化，所以總和不會精確為1）
        row_sums = topk_weight.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
    
    def test_expert_load_balancing(self, default_config, set_seed):
        """測試專家負載均衡（通過輔助損失實現）"""
        moe_gate = MoEGate(default_config)
        moe_gate.train()  # 確保在訓練模式
        
        batch_size = 10
        seq_len = 20
        hidden_dim = default_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 計算每個專家被選擇的次數
        expert_counts = torch.zeros(default_config.n_routed_experts, device=hidden_states.device)
        for i in range(batch_size * seq_len):
            for j in range(default_config.num_experts_per_tok):
                expert_counts[topk_idx[i, j]] += topk_weight[i, j]
        
        # 檢查輔助損失與專家使用的不平衡程度相關
        # 在理想情況下，每個專家的使用比例應接近 1/n_routed_experts
        expert_usage = expert_counts / expert_counts.sum()
        ideal_usage = torch.ones_like(expert_usage) / default_config.n_routed_experts
        
        # 簡單檢查：驗證輔助損失存在，說明它在鼓勵負載均衡
        assert aux_loss > 0


# 用於直接運行測試的主程序
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 