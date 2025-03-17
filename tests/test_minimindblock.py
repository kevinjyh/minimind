import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# 添加根目錄到系統路徑
root_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(root_dir)

from model.model import MiniMindBlock, RMSNorm, Attention, FeedForward, MOEFeedForward
from model.LMConfig import LMConfig


class TestMiniMindBlock:
    """
    測試MiniMindBlock類的功能和工作原理
    """
    
    @pytest.fixture
    def basic_config(self):
        """基本配置，使用標準前饋網絡(非MoE)"""
        return LMConfig(
            dim=512,  # 隱藏層維度
            n_layers=8,  # 層數
            n_heads=8,  # 注意力頭數
            n_kv_heads=2,  # KV頭數 (用於分組查詢注意力)
            vocab_size=6400,  # 詞彙表大小
            max_seq_len=128,  # 最大序列長度
            use_moe=False,  # 不使用MoE (混合專家)
            flash_attn=False,  # 不使用Flash注意力
        )
    
    @pytest.fixture
    def moe_config(self):
        """MoE配置，使用混合專家前饋網絡"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=6400,
            max_seq_len=128,
            use_moe=True,  # 使用MoE
            n_routed_experts=4,  # 路由專家數量
            num_experts_per_tok=2,  # 每個token選擇的專家數量
            flash_attn=False,
        )
    
    @pytest.fixture
    def basic_block(self, basic_config):
        """創建基本MiniMindBlock實例"""
        return MiniMindBlock(layer_id=0, config=basic_config)
    
    @pytest.fixture
    def moe_block(self, moe_config):
        """創建MoE版本的MiniMindBlock實例"""
        return MiniMindBlock(layer_id=0, config=moe_config)
    
    @pytest.fixture
    def sample_input(self, basic_config):
        """生成樣本輸入數據"""
        batch_size = 2
        seq_len = 10
        dim = basic_config.dim
        
        # 生成隨機輸入張量
        x = torch.randn(batch_size, seq_len, dim)
        
        # 生成位置編碼 (pos_cis)
        head_dim = dim // basic_config.n_heads
        theta = basic_config.rope_theta
        
        # 計算頻率基底
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        
        # 生成位置序列
        t = torch.arange(seq_len)
        
        # 計算外積得到位置-維度頻率矩陣
        freqs = torch.outer(t, freqs).float()
        
        # 轉換為複數形式
        pos_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return x, pos_cis
    
    def test_init(self, basic_block, moe_block, basic_config, moe_config):
        """測試初始化是否正確"""
        # 測試基本塊屬性
        assert basic_block.n_heads == basic_config.n_heads
        assert basic_block.dim == basic_config.dim
        assert basic_block.head_dim == basic_config.dim // basic_config.n_heads
        assert basic_block.layer_id == 0
        
        # 測試基本塊組件類型
        assert isinstance(basic_block.attention, Attention)
        assert isinstance(basic_block.attention_norm, RMSNorm)
        assert isinstance(basic_block.ffn_norm, RMSNorm)
        assert isinstance(basic_block.feed_forward, FeedForward)
        
        # 測試MoE塊組件類型
        assert isinstance(moe_block.feed_forward, MOEFeedForward)
    
    def test_forward_shape(self, basic_block, sample_input):
        """測試前向傳播的輸出形狀是否正確"""
        x, pos_cis = sample_input
        batch_size, seq_len, dim = x.shape
        
        # 執行前向傳播
        output, past_kv = basic_block(x, pos_cis, past_key_value=None, use_cache=False)
        
        # 檢查輸出形狀
        assert output.shape == (batch_size, seq_len, dim)
        assert past_kv is None  # 當use_cache=False時，past_kv應為None
    
    def test_forward_with_cache(self, basic_block, sample_input):
        """測試啟用緩存的前向傳播，支持GQA模式"""
        x, pos_cis = sample_input
        batch_size, seq_len, dim = x.shape
        
        # 執行啟用緩存的前向傳播
        output, past_kv = basic_block(x, pos_cis, past_key_value=None, use_cache=True)
        
        # 檢查輸出形狀
        assert output.shape == (batch_size, seq_len, dim)
        
        # 檢查past_kv緩存
        assert past_kv is not None
        assert len(past_kv) == 2  # 包含key和value兩個緩存
        
        # 檢查緩存形狀是否符合GQA模式
        k_cache, v_cache = past_kv
        n_kv_heads = basic_block.attention.n_local_kv_heads
        head_dim = basic_block.head_dim
        
        # 緩存形狀應為 [batch_size, seq_len, n_kv_heads, head_dim]
        expected_cache_shape = (batch_size, seq_len, n_kv_heads, head_dim)
        assert k_cache.shape == expected_cache_shape, f"Key緩存形狀不符, 期望: {expected_cache_shape}, 實際: {k_cache.shape}"
        assert v_cache.shape == expected_cache_shape, f"Value緩存形狀不符, 期望: {expected_cache_shape}, 實際: {v_cache.shape}"
        
        # 測試緩存增量更新
        # 創建一個新的單個token輸入，模擬自回歸生成
        new_token = torch.randn(batch_size, 1, dim)
        new_pos_cis = pos_cis[:1]  # 只用第一個位置的編碼
        
        # 使用先前的緩存進行前向傳播
        new_output, new_past_kv = basic_block(new_token, new_pos_cis, past_key_value=past_kv, use_cache=True)
        
        # 檢查新輸出形狀
        assert new_output.shape == (batch_size, 1, dim)
        
        # 檢查更新後的緩存
        assert new_past_kv is not None
        assert len(new_past_kv) == 2
        
        # 新緩存應該包含原始序列加上新token
        new_k_cache, new_v_cache = new_past_kv
        expected_new_cache_shape = (batch_size, seq_len + 1, n_kv_heads, head_dim)
        assert new_k_cache.shape == expected_new_cache_shape
        assert new_v_cache.shape == expected_new_cache_shape
        
        # 驗證緩存的連續性：新緩存的前seq_len部分應該與舊緩存相同
        assert torch.allclose(new_k_cache[:, :seq_len, :, :], k_cache)
        assert torch.allclose(new_v_cache[:, :seq_len, :, :], v_cache)
    
    def test_residual_connection(self, basic_block, sample_input):
        """測試殘差連接的作用"""
        x, pos_cis = sample_input
        
        # 保存原始輸入
        original_x = x.clone()
        
        # 禁用注意力和前饋網絡，僅測試殘差連接
        with torch.no_grad():
            # 暫時修改forward函數，使注意力和前饋網絡返回零張量
            original_attention_forward = basic_block.attention.forward
            original_ff_forward = basic_block.feed_forward.forward
            
            def zero_attention(*args, **kwargs):
                zeros = torch.zeros_like(args[0])
                return zeros, None
            
            def zero_ff(*args, **kwargs):
                return torch.zeros_like(args[0])
            
            basic_block.attention.forward = zero_attention
            basic_block.feed_forward.forward = zero_ff
            
            # 執行前向傳播
            output, _ = basic_block(x, pos_cis)
            
            # 恢復原始函數
            basic_block.attention.forward = original_attention_forward
            basic_block.feed_forward.forward = original_ff_forward
        
        # 由於注意力和前饋網絡返回零張量，如果殘差連接工作正常，輸出應等於輸入
        assert torch.allclose(output, original_x)
    
    def test_attention_norm(self, basic_block, sample_input):
        """測試注意力標準化層"""
        x, pos_cis = sample_input
        
        # 獲取標準化層
        norm_layer = basic_block.attention_norm
        
        # 應用標準化
        normalized_x = norm_layer(x)
        
        # 檢查標準化後的形狀
        assert normalized_x.shape == x.shape
        
        # 標準化後，每個特徵的方差應接近1
        # 沿最後一個維度計算方差（特徵維度）
        var = normalized_x.var(dim=-1)
        
        # 平均方差應接近1
        assert 0.9 <= var.mean().item() <= 1.1
    
    def test_ffn_norm(self, basic_block, sample_input):
        """測試前饋網絡標準化層"""
        x, pos_cis = sample_input
        
        # 獲取標準化層
        norm_layer = basic_block.ffn_norm
        
        # 應用標準化
        normalized_x = norm_layer(x)
        
        # 檢查標準化後的形狀
        assert normalized_x.shape == x.shape
        
        # 標準化後，每個特徵的方差應接近1
        # 沿最後一個維度計算方差（特徵維度）
        var = normalized_x.var(dim=-1)
        
        # 平均方差應接近1
        assert 0.9 <= var.mean().item() <= 1.1
    
    def test_attention_and_ffn_contribute(self, basic_block, sample_input):
        """測試注意力和前饋網絡是否都對輸出有貢獻"""
        x, pos_cis = sample_input
        
        # 正常前向傳播
        normal_output, _ = basic_block(x, pos_cis)
        
        # 禁用注意力，測試前饋網絡的貢獻
        with torch.no_grad():
            original_attention_forward = basic_block.attention.forward
            
            def zero_attention(*args, **kwargs):
                zeros = torch.zeros_like(args[0])
                return zeros, None
            
            basic_block.attention.forward = zero_attention
            
            no_attn_output, _ = basic_block(x, pos_cis)
            
            basic_block.attention.forward = original_attention_forward
        
        # 禁用前饋網絡，測試注意力的貢獻
        with torch.no_grad():
            original_ff_forward = basic_block.feed_forward.forward
            
            def zero_ff(*args, **kwargs):
                return torch.zeros_like(args[0])
            
            basic_block.feed_forward.forward = zero_ff
            
            no_ff_output, _ = basic_block(x, pos_cis)
            
            basic_block.feed_forward.forward = original_ff_forward
        
        # 正常輸出應與禁用任一組件的輸出不同
        assert not torch.allclose(normal_output, no_attn_output)
        assert not torch.allclose(normal_output, no_ff_output)
        assert not torch.allclose(no_attn_output, no_ff_output)
    
    def test_moe_vs_standard_ffn(self, basic_block, moe_block, sample_input):
        """比較標準前饋網絡和MoE前饋網絡的輸出差異"""
        x, pos_cis = sample_input
        
        # 使用標準前饋網絡的前向傳播
        standard_output, _ = basic_block(x, pos_cis)
        
        # 使用MoE前饋網絡的前向傳播
        moe_output, _ = moe_block(x, pos_cis)
        
        # 兩種前饋網絡的輸出形狀應相同
        assert standard_output.shape == moe_output.shape
        
        # 但輸出值應有差異
        assert not torch.allclose(standard_output, moe_output)
        
        # 檢查MOE層是否有輔助損失
        assert hasattr(moe_block.feed_forward, 'aux_loss')
    
    def test_past_key_value_reuse(self, basic_block, sample_input):
        """測試past_key_value的重用機制"""
        x, pos_cis = sample_input
        batch_size, seq_len, dim = x.shape
        
        # 第一次前向傳播，生成緩存
        _, past_kv = basic_block(x, pos_cis, use_cache=True)
        
        # 模擬新的輸入token（第二個序列）
        new_token = torch.randn(batch_size, 1, dim)
        # 確保 new_pos_cis 的切片正確
        new_pos_cis = pos_cis[seq_len-1:seq_len]  # 應該是上一個序列的最後一個位置編碼
        
        # 使用先前的緩存進行第二次前向傳播
        output_with_cache, new_past_kv = basic_block(new_token, new_pos_cis, past_key_value=past_kv, use_cache=True)
        
        # 檢查輸出形狀
        assert output_with_cache.shape == (batch_size, 1, dim)
        
        # 新緩存的第一個維度（序列長度）應比舊緩存增加1
        assert new_past_kv[0].shape[1] == past_kv[0].shape[1] + 1 