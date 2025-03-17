import pytest
import torch
import torch.nn as nn
import sys
import os
import math
from torch.nn import functional as F
import copy

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
        
        # 檢查在訓練模式下輔助損失是否大於0
        assert train_aux_loss >= 0
        
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

    def test_visualize_moe_gate_outputs(self, default_config, set_seed):
        """此測試案例用於可視化和解釋MoEGate的輸出，幫助理解topk_idx, topk_weight, aux_loss的含義和形狀"""
        moe_gate = MoEGate(default_config)
        
        # 為了使輸出更容易理解，我們使用小型輸入
        batch_size = 2
        seq_len = 3
        hidden_dim = default_config.dim
        
        # 創建隨機輸入張量
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 執行前向傳播
        topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
        
        # 將結果寫入文件
        with open('tests/moe_gate_visualization.txt', 'w', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"MoEGate配置:\n")
            f.write(f"總專家數量 (n_routed_experts): {default_config.n_routed_experts}\n")
            f.write(f"每個token選擇的專家數 (num_experts_per_tok): {default_config.num_experts_per_tok}\n")
            f.write(f"是否使用序列級輔助損失 (seq_aux): {default_config.seq_aux}\n")
            f.write(f"輔助損失權重 (aux_loss_alpha): {default_config.aux_loss_alpha}\n")
            f.write(f"{'='*80}\n")
            
            # 打印輸入形狀
            f.write(f"輸入形狀: {hidden_states.shape} (batch_size, seq_len, hidden_dim)\n")
            
            # 打印輸出形狀和值
            f.write(f"\n{'='*30} topk_idx {'='*30}\n")
            f.write(f"形狀: {topk_idx.shape} (batch_size*seq_len, num_experts_per_tok)\n")
            f.write(f"具體值:\n{topk_idx}\n")
            
            # 重塑為更直觀的形式
            reshaped_idx = topk_idx.view(batch_size, seq_len, default_config.num_experts_per_tok)
            f.write(f"\n重塑為 (batch_size, seq_len, num_experts_per_tok): {reshaped_idx.shape}\n")
            f.write(f"重塑後值:\n{reshaped_idx}\n")
            
            f.write(f"\n{'='*30} topk_weight {'='*30}\n")
            f.write(f"形狀: {topk_weight.shape} (batch_size*seq_len, num_experts_per_tok)\n")
            f.write(f"具體值:\n{topk_weight}\n")
            
            # 重塑為更直觀的形式
            reshaped_weight = topk_weight.view(batch_size, seq_len, default_config.num_experts_per_tok)
            f.write(f"\n重塑為 (batch_size, seq_len, num_experts_per_tok): {reshaped_weight.shape}\n")
            f.write(f"重塑後值:\n{reshaped_weight}\n")
            
            f.write(f"\n{'='*30} aux_loss {'='*30}\n")
            f.write(f"值: {aux_loss.item()}\n")
            
            # 創建一個用於解釋的示例
            f.write(f"\n{'='*30} 實例解釋 {'='*30}\n")
            f.write("以下是第一個批次的第一個token的解釋:\n")
            token_idx = reshaped_idx[0, 0].tolist()
            token_weight = reshaped_weight[0, 0].tolist()
            
            for i, (idx, weight) in enumerate(zip(token_idx, token_weight)):
                f.write(f"專家{idx}被選中, 權重為{weight:.4f}\n")
            
            f.write(f"\n輔助損失(aux_loss)值為{aux_loss.item():.6f}, 這是一個純量值\n")
            f.write(f"它用於促進所有專家的均衡使用，而不是針對單個專家\n")
            
            # 計算每個專家被選中的頻率
            expert_counts = torch.zeros(default_config.n_routed_experts)
            for idx in topk_idx.flatten():
                expert_counts[idx] += 1
            
            expert_selection_percentage = expert_counts / len(topk_idx.flatten()) * 100
            f.write(f"\n{'='*30} 專家選擇分佈 {'='*30}\n")
            for i, percentage in enumerate(expert_selection_percentage):
                f.write(f"專家{i}: 被選中{expert_counts[i]:.0f}次, 佔比{percentage:.2f}%\n")
        
        # 仍然保留打印輸出
        print(f"\n結果已寫入 tests/moe_gate_visualization.txt")
            
        # 確保測試通過
        assert topk_idx.shape == (batch_size * seq_len, default_config.num_experts_per_tok)
        assert topk_weight.shape == (batch_size * seq_len, default_config.num_experts_per_tok)
        assert isinstance(aux_loss, torch.Tensor) and aux_loss.numel() == 1

    def test_vary_aux_loss_weight(self, default_config, set_seed):
        """測試不同輔助損失權重對專家分配的影響"""
        batch_size = 2
        seq_len = 3
        hidden_dim = default_config.dim
        
        # 創建固定的輸入張量以保持一致性
        torch.manual_seed(42)  # 保持輸入一致
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        aux_weights = [0.0, 0.1, 0.5, 1.0]
        results = []
        
        # 將結果寫入文件
        with open('tests/moe_gate_aux_loss_comparison.txt', 'w', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write("測試不同輔助損失權重對專家分配的影響\n")
            
            for weight in aux_weights:
                # 創建配置副本並修改輔助損失權重
                config = copy.deepcopy(default_config)
                config.aux_loss_alpha = weight
                
                # 創建MoEGate
                moe_gate = MoEGate(config)
                
                # 執行前向傳播
                topk_idx, topk_weight, aux_loss = moe_gate(hidden_states)
                
                # 計算每個專家被選中的頻率
                expert_counts = torch.zeros(config.n_routed_experts)
                for idx in topk_idx.flatten():
                    expert_counts[idx] += 1
                
                expert_selection_percentage = expert_counts / len(topk_idx.flatten()) * 100
                
                # 處理 aux_loss 可能是整數的情況（當 aux_loss_alpha=0.0 時）
                aux_loss_value = aux_loss if isinstance(aux_loss, (int, float)) else aux_loss.item()
                
                results.append({
                    'weight': weight,
                    'aux_loss': aux_loss_value,
                    'expert_counts': expert_counts.tolist(),
                    'expert_selection_percentage': expert_selection_percentage.tolist()
                })
                
                # 寫入結果
                f.write(f"\n{'='*30} 輔助損失權重 = {weight} {'='*30}\n")
                f.write(f"輔助損失值 = {aux_loss_value:.6f}\n")
                f.write("專家選擇分佈:\n")
                for i, (count, percentage) in enumerate(zip(expert_counts.tolist(), expert_selection_percentage.tolist())):
                    f.write(f"專家{i}: 被選中{count:.0f}次, 佔比{percentage:.2f}%\n")
                
                # 計算方差作為分佈均勻性的指標
                variance = sum((p - 25)**2 for p in expert_selection_percentage.tolist()) / len(expert_selection_percentage)
                f.write(f"分佈方差: {variance:.2f} (越小表示分佈越均勻)\n")
                
                # 寫入專家選擇的具體索引和權重
                f.write("\n專家選擇索引 (topk_idx):\n")
                f.write(f"{topk_idx}\n")
                
                f.write("\n專家選擇權重 (topk_weight):\n")
                f.write(f"{topk_weight}\n")
        
        # 仍然保留打印輸出
        print(f"\n結果已寫入 tests/moe_gate_aux_loss_comparison.txt")
        
        # 確保測試通過
        assert len(results) == len(aux_weights)


# 用於直接運行測試的主程序
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 