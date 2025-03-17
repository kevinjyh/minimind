import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path
import math

# 修正：添加專案根目錄到系統路徑
root_dir = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, root_dir)  # 使用insert(0)確保優先搜索

from model.model import Attention, precompute_pos_cis, repeat_kv, apply_rotary_emb
from model.LMConfig import LMConfig

class TestGQAMechanism:
    """
    專門測試 Grouped Query Attention (GQA) 機制
    
    GQA 是一種注意力機制，其中多個 query 頭共享同一個 key-value 頭，
    即 n_heads > n_kv_heads，且 n_heads 是 n_kv_heads 的倍數。
    """
    
    @pytest.fixture
    def gqa_configs(self):
        """提供不同 n_heads 和 n_kv_heads 比例的 GQA 配置"""
        return [
            # 1:1 (標準自注意力)
            LMConfig(dim=64, n_heads=4, n_kv_heads=4, max_seq_len=32, flash_attn=False),
            # 2:1 (最常見的 GQA 配置)
            LMConfig(dim=64, n_heads=4, n_kv_heads=2, max_seq_len=32, flash_attn=False),
            # 4:1 (高複用比例)
            LMConfig(dim=64, n_heads=8, n_kv_heads=2, max_seq_len=32, flash_attn=False),
            # 8:1 (極高複用比例)
            LMConfig(dim=64, n_heads=8, n_kv_heads=1, max_seq_len=32, flash_attn=False)
        ]
    
    @pytest.fixture
    def batch_size(self):
        return 2
    
    @pytest.fixture
    def seq_len(self):
        return 8
    
    def test_head_ratio_computation(self, gqa_configs):
        """測試不同 n_heads 和 n_kv_heads 比例下的 n_rep 計算正確性"""
        for config in gqa_configs:
            attn = Attention(config)
            
            expected_n_rep = config.n_heads // config.n_kv_heads
            assert attn.n_rep == expected_n_rep, f"n_rep計算錯誤，期望{expected_n_rep}，得到{attn.n_rep}"
            
            # 檢查 n_local_heads 和 n_local_kv_heads 設置正確
            assert attn.n_local_heads == config.n_heads
            assert attn.n_local_kv_heads == config.n_kv_heads
    
    def test_repeat_kv_shapes(self, gqa_configs, batch_size, seq_len):
        """測試不同 GQA 比例下 repeat_kv 函數的輸出形狀正確性"""
        for config in gqa_configs:
            attn = Attention(config)
            head_dim = config.dim // config.n_heads
            
            # 創建 key-value 表示
            kv = torch.randn(batch_size, seq_len, config.n_kv_heads, head_dim)
            
            # 應用 repeat_kv 函數
            kv_repeated = repeat_kv(kv, attn.n_rep)
            
            # 檢查輸出形狀是否符合預期
            expected_shape = (batch_size, seq_len, config.n_heads, head_dim)
            assert kv_repeated.shape == expected_shape, \
                f"repeat_kv 輸出形狀錯誤，期望 {expected_shape}，得到 {kv_repeated.shape}"
    
    def test_repeat_kv_values(self, gqa_configs, batch_size, seq_len):
        """測試 repeat_kv 函數複製值的正確性"""
        for config in gqa_configs:
            attn = Attention(config)
            head_dim = config.dim // config.n_heads
            
            # 創建帶有明確模式的 key-value 表示
            kv = torch.zeros(batch_size, seq_len, config.n_kv_heads, head_dim)
            for i in range(config.n_kv_heads):
                kv[:, :, i, i] = i + 1  # 每個頭有唯一的標識值
            
            # 應用 repeat_kv 函數
            kv_repeated = repeat_kv(kv, attn.n_rep)
            
            # 檢查每個 query 頭是否正確地從對應的 kv 頭獲取值
            for kv_head_idx in range(config.n_kv_heads):
                for rep_idx in range(attn.n_rep):
                    q_head_idx = kv_head_idx * attn.n_rep + rep_idx
                    # 檢查特徵值是否正確複製
                    assert torch.all(kv_repeated[:, :, q_head_idx, kv_head_idx] == kv_head_idx + 1), \
                        f"KV頭 {kv_head_idx} 的值未正確複製到 Query頭 {q_head_idx}"
    
    def test_attention_forward_with_gqa(self, gqa_configs, batch_size, seq_len):
        """測試帶有不同 GQA 比例的注意力前向傳播"""
        for config in gqa_configs:
            attn = Attention(config)
            
            # 創建輸入
            x = torch.randn(batch_size, seq_len, config.dim)
            pos_cis = precompute_pos_cis(
                dim=config.dim // config.n_heads, 
                end=config.max_seq_len,
                theta=config.rope_theta
            )[:seq_len]
            
            # 執行前向傳播
            output, _ = attn(x, pos_cis)
            
            # 檢查輸出形狀
            assert output.shape == x.shape, \
                f"注意力輸出形狀錯誤，期望 {x.shape}，得到 {output.shape}"
    
    def test_attention_cache_with_gqa(self, gqa_configs, batch_size, seq_len):
        """測試不同 GQA 比例下注意力快取的正確性"""
        for config in gqa_configs:
            attn = Attention(config)
            head_dim = config.dim // config.n_heads
            
            # 創建輸入
            x = torch.randn(batch_size, seq_len, config.dim)
            pos_cis = precompute_pos_cis(
                dim=config.dim // config.n_heads, 
                end=config.max_seq_len,
                theta=config.rope_theta
            )[:seq_len]
            
            # 執行快取模式的前向傳播
            _, cache = attn(x, pos_cis, use_cache=True)
            
            # 檢查快取形狀
            k_cache, v_cache = cache
            expected_cache_shape = (batch_size, seq_len, config.n_kv_heads, head_dim)
            assert k_cache.shape == expected_cache_shape, \
                f"Key快取形狀錯誤，期望 {expected_cache_shape}，得到 {k_cache.shape}"
            assert v_cache.shape == expected_cache_shape, \
                f"Value快取形狀錯誤，期望 {expected_cache_shape}，得到 {v_cache.shape}"
            
            # 創建新的單個token輸入
            new_x = torch.randn(batch_size, 1, config.dim)
            new_pos_cis = precompute_pos_cis(
                dim=config.dim // config.n_heads, 
                end=config.max_seq_len,
                theta=config.rope_theta
            )[seq_len:seq_len+1]
            
            # 使用快取進行前向傳播
            _, new_cache = attn(new_x, new_pos_cis, past_key_value=cache, use_cache=True)
            
            # 檢查新快取形狀
            new_k_cache, new_v_cache = new_cache
            expected_new_cache_shape = (batch_size, seq_len + 1, config.n_kv_heads, head_dim)
            assert new_k_cache.shape == expected_new_cache_shape, \
                f"新Key快取形狀錯誤，期望 {expected_new_cache_shape}，得到 {new_k_cache.shape}"
            assert new_v_cache.shape == expected_new_cache_shape, \
                f"新Value快取形狀錯誤，期望 {expected_new_cache_shape}，得到 {new_v_cache.shape}"
            
            # 檢查快取連續性
            assert torch.allclose(new_k_cache[:, :seq_len, :, :], k_cache), \
                "新舊Key快取不連續"
            assert torch.allclose(new_v_cache[:, :seq_len, :, :], v_cache), \
                "新舊Value快取不連續"
    
    def test_apply_rotary_emb_with_gqa(self, gqa_configs, batch_size, seq_len):
        """測試旋轉位置編碼在 GQA 機制下的行為"""
        for config in gqa_configs:
            # 計算頭維度
            head_dim = config.dim // config.n_heads
            
            # 創建查詢和鍵表示
            xq = torch.randn(batch_size, seq_len, config.n_heads, head_dim)
            xk = torch.randn(batch_size, seq_len, config.n_kv_heads, head_dim)
            
            # 創建位置編碼
            pos_cis = precompute_pos_cis(
                dim=head_dim, 
                end=config.max_seq_len,
                theta=config.rope_theta
            )[:seq_len]
            
            # 應用旋轉位置編碼
            xq_out, xk_out = apply_rotary_emb(xq, xk, pos_cis)
            
            # 檢查輸出形狀
            assert xq_out.shape == xq.shape, \
                f"旋轉後的查詢形狀錯誤，期望 {xq.shape}，得到 {xq_out.shape}"
            assert xk_out.shape == xk.shape, \
                f"旋轉後的鍵形狀錯誤，期望 {xk.shape}，得到 {xk_out.shape}"
    
    def test_different_gqa_behavior(self, batch_size, seq_len):
        """測試不同 GQA 配置對模型行為的影響"""
        # 創建固定的輸入數據
        dim = 64
        input_data = torch.zeros(batch_size, seq_len, dim)
        
        # 在不同位置設置不同的特徵值，用於跟踪注意力模式
        for i in range(seq_len):
            input_data[:, i, i % dim] = 1.0
        
        # 位置編碼
        pos_cis = precompute_pos_cis(
            dim=dim // 4,  # 使用 n_heads=4 的頭維度
            end=32,
            theta=1e6
        )[:seq_len]
        
        # 創建不同 GQA 配置的模型
        std_attn = Attention(LMConfig(dim=dim, n_heads=4, n_kv_heads=4, max_seq_len=32, flash_attn=False))
        gqa_attn_2_1 = Attention(LMConfig(dim=dim, n_heads=4, n_kv_heads=2, max_seq_len=32, flash_attn=False))
        gqa_attn_4_1 = Attention(LMConfig(dim=dim, n_heads=4, n_kv_heads=1, max_seq_len=32, flash_attn=False))
        
        # 運行所有模型
        with torch.no_grad():
            std_output, _ = std_attn(input_data, pos_cis)
            gqa_2_1_output, _ = gqa_attn_2_1(input_data, pos_cis)
            gqa_4_1_output, _ = gqa_attn_4_1(input_data, pos_cis)
        
        # 檢查输出差異
        std_gqa_2_1_diff = (std_output - gqa_2_1_output).abs().max().item()
        std_gqa_4_1_diff = (std_output - gqa_4_1_output).abs().max().item()
        gqa_2_1_4_1_diff = (gqa_2_1_output - gqa_4_1_output).abs().max().item()
        
        # 確認不同 GQA 配置產生不同的輸出
        assert std_gqa_2_1_diff > 1e-5, "標準注意力和2:1 GQA應該產生不同輸出"
        assert std_gqa_4_1_diff > 1e-5, "標準注意力和4:1 GQA應該產生不同輸出"
        assert gqa_2_1_4_1_diff > 1e-5, "2:1 GQA和4:1 GQA應該產生不同輸出"
        
        # 檢查 GQA 是否保留了輸入的基本信息
        assert torch.allclose(std_output.norm(), gqa_2_1_output.norm(), rtol=0.5), \
            "GQA應該保留輸入的基本信息量"
        assert torch.allclose(std_output.norm(), gqa_4_1_output.norm(), rtol=0.5), \
            "GQA應該保留輸入的基本信息量" 