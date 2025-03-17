import torch
import pytest
import sys
import os
from pathlib import Path

# 修正：確保可以導入模型模塊
# 添加專案根目錄到系統路徑（向上層到專案根目錄）
root_dir = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, root_dir)  # 使用insert(0)確保優先搜索

from model.LMConfig import LMConfig
from model.model import Attention, precompute_pos_cis, repeat_kv

class TestAttention:
    """測試MiniMind中的Attention機制"""
    
    @pytest.fixture
    def tiny_config(self):
        """創建一個極小的配置用於測試"""
        return LMConfig(
            dim=64,           # 極小的隱藏維度
            n_heads=4,        # 注意力頭數
            n_kv_heads=2,     # KV頭數，測試grouped-query attention
            max_seq_len=32,   # 短序列長度，適合CPU測試
            flash_attn=False, # 禁用flash attention以確保CPU兼容性
            dropout=0.0       # 禁用dropout以確保結果確定性
        )
    
    @pytest.fixture
    def batch_size(self):
        return 2
    
    @pytest.fixture
    def seq_len(self):
        return 16
    
    @pytest.fixture
    def sample_input(self, tiny_config, batch_size, seq_len):
        """創建樣本輸入數據"""
        return torch.randn(batch_size, seq_len, tiny_config.dim)
    
    @pytest.fixture
    def sample_pos_cis(self, tiny_config, seq_len):
        """創建樣本位置編碼"""
        return precompute_pos_cis(
            dim=tiny_config.dim // tiny_config.n_heads, 
            end=tiny_config.max_seq_len,
            theta=tiny_config.rope_theta
        )[:seq_len]
    
    def test_attention_init(self, tiny_config):
        """測試Attention層初始化"""
        attn = Attention(tiny_config)
        
        # 檢查頭數和維度
        assert attn.n_local_heads == tiny_config.n_heads
        assert attn.n_local_kv_heads == tiny_config.n_kv_heads
        assert attn.n_rep == tiny_config.n_heads // tiny_config.n_kv_heads
        assert attn.head_dim == tiny_config.dim // tiny_config.n_heads
        
        # 檢查線性層維度
        assert attn.wq.weight.shape == (tiny_config.n_heads * attn.head_dim, tiny_config.dim)
        assert attn.wk.weight.shape == (tiny_config.n_kv_heads * attn.head_dim, tiny_config.dim)
        assert attn.wv.weight.shape == (tiny_config.n_kv_heads * attn.head_dim, tiny_config.dim)
        assert attn.wo.weight.shape == (tiny_config.dim, tiny_config.n_heads * attn.head_dim)
    
    def test_forward_shape(self, tiny_config, sample_input, sample_pos_cis):
        """測試前向傳播的輸出形狀"""
        attn = Attention(tiny_config)
        output, _ = attn(sample_input, sample_pos_cis)
        
        # 檢查輸出維度是否與輸入維度一致
        assert output.shape == sample_input.shape
    
    def test_kv_cache(self, tiny_config, sample_input, sample_pos_cis):
        """測試KV緩存功能，支持 n_heads != n_kv_heads 的GQA情況"""
        batch_size, seq_len = sample_input.shape[:2]
        
        # 使用原始配置，確保GQA情況（n_heads != n_kv_heads）能正確測試
        attn = Attention(tiny_config)
        
        # 第一次前向傳播，使用KV緩存
        output1, past_kv = attn(sample_input, sample_pos_cis, use_cache=True)
        
        # 檢查KV緩存的形狀
        assert len(past_kv) == 2  # (K, V)
        k_cache, v_cache = past_kv
        
        # 檢查 k_cache 和 v_cache 的形狀是否正確
        # 注意：在實際模型中，緩存的形狀是 [batch_size, seq_len, n_kv_heads, head_dim]
        assert k_cache.shape == (batch_size, seq_len, tiny_config.n_kv_heads, attn.head_dim)
        assert v_cache.shape == (batch_size, seq_len, tiny_config.n_kv_heads, attn.head_dim)
        
        # 使用緩存進行第二次前向傳播
        next_token = torch.randn(batch_size, 1, tiny_config.dim)  # 只有一個token
        next_pos_cis = precompute_pos_cis(
            dim=tiny_config.dim // tiny_config.n_heads, 
            end=tiny_config.max_seq_len,
            theta=tiny_config.rope_theta
        )[seq_len:seq_len+1]  # 使用下一個位置的編碼
        
        output2, past_kv2 = attn(next_token, next_pos_cis, past_key_value=past_kv, use_cache=True)
        
        # 檢查輸出形狀
        assert output2.shape == (batch_size, 1, tiny_config.dim)
        
        # 檢查新的KV緩存是否正確附加了新數據
        assert past_kv2[0].shape == (batch_size, seq_len + 1, tiny_config.n_kv_heads, attn.head_dim)
        assert past_kv2[1].shape == (batch_size, seq_len + 1, tiny_config.n_kv_heads, attn.head_dim)
        
        # 驗證緩存的連續性：新緩存的前seq_len部分應該與舊緩存相同
        assert torch.allclose(past_kv2[0][:, :seq_len, :, :], k_cache)
        assert torch.allclose(past_kv2[1][:, :seq_len, :, :], v_cache)
    
    def test_gqa_mechanism(self, tiny_config, sample_input, sample_pos_cis):
        """測試GQA機制 - 專門測試 n_heads > n_kv_heads 的情況"""
        # 確保 tiny_config 的 n_heads 和 n_kv_heads 符合 GQA 要求
        assert tiny_config.n_heads > tiny_config.n_kv_heads
        assert tiny_config.n_heads % tiny_config.n_kv_heads == 0, "n_heads必須是n_kv_heads的倍數"
        
        attn = Attention(tiny_config)
        n_rep = attn.n_rep
        
        # 驗證 n_rep 計算正確
        assert n_rep == tiny_config.n_heads // tiny_config.n_kv_heads
        
        # 實現一個簡化版本的前向傳播過程來驗證GQA機制
        batch_size, seq_len, dim = sample_input.shape
        
        # 手動計算q, k, v投影
        q = attn.wq(sample_input).view(batch_size, seq_len, tiny_config.n_heads, attn.head_dim)
        k = attn.wk(sample_input).view(batch_size, seq_len, tiny_config.n_kv_heads, attn.head_dim)
        v = attn.wv(sample_input).view(batch_size, seq_len, tiny_config.n_kv_heads, attn.head_dim)
        
        # 手動重複k和v
        k_repeated = repeat_kv(k, n_rep)
        v_repeated = repeat_kv(v, n_rep)
        
        # 驗證重複後的形狀是否正確
        assert k_repeated.shape == (batch_size, seq_len, tiny_config.n_heads, attn.head_dim)
        assert v_repeated.shape == (batch_size, seq_len, tiny_config.n_heads, attn.head_dim)
        
        # 檢查重複模式：同一KV頭的值被複製到多個Q頭
        for kv_head_idx in range(tiny_config.n_kv_heads):
            for rep_idx in range(n_rep):
                q_head_idx = kv_head_idx * n_rep + rep_idx
                # 驗證k和v確實被正確重複
                assert torch.allclose(k_repeated[:, :, q_head_idx, :], k[:, :, kv_head_idx, :])
                assert torch.allclose(v_repeated[:, :, q_head_idx, :], v[:, :, kv_head_idx, :])
        
        # 執行前向傳播確認整體功能正常
        output, _ = attn(sample_input, sample_pos_cis)
        assert output.shape == sample_input.shape
    
    def test_repeat_kv_function(self, tiny_config):
        """測試repeat_kv函數"""
        bs, seq_len = 2, 10
        n_kv_heads, head_dim = tiny_config.n_kv_heads, tiny_config.dim // tiny_config.n_heads
        n_rep = tiny_config.n_heads // tiny_config.n_kv_heads
        
        # 創建樣本KV數據
        kv = torch.randn(bs, seq_len, n_kv_heads, head_dim)
        
        # 應用repeat_kv
        kv_repeated = repeat_kv(kv, n_rep)
        
        # 檢查形狀
        assert kv_repeated.shape == (bs, seq_len, n_kv_heads * n_rep, head_dim)
        
        # 檢查內容 - 每組n_rep個頭應該有相同的數據
        for i in range(n_kv_heads):
            for j in range(n_rep):
                # 驗證重複的頭與原始頭一致
                assert torch.allclose(
                    kv_repeated[:, :, i * n_rep + j, :], 
                    kv[:, :, i, :]
                )
    
    def test_attention_pattern(self, tiny_config):
        """測試注意力模式的基本特性 - 主要是因果性掩碼"""
        # 創建一個簡單的輸入，其中每個位置都是唯一標識的
        bs, seq_len = 1, 8
        dim = tiny_config.dim
        
        # 使用one-hot向量來識別每個位置
        x = torch.zeros(bs, seq_len, dim)
        for i in range(seq_len):
            x[:, i, i % dim] = 1.0
        
        # 創建位置編碼
        pos_cis = precompute_pos_cis(
            dim=dim // tiny_config.n_heads, 
            end=tiny_config.max_seq_len,
            theta=tiny_config.rope_theta
        )[:seq_len]
        
        # 初始化Attention層，禁用dropout以確保結果確定性
        attn = Attention(tiny_config)
        
        # 記錄中間值的hook
        query_list = []
        key_list = []
        value_list = []
        
        # 註冊hooks來捕獲中間計算結果
        def get_q_hook(module, input, output):
            query_list.append(output.detach())
        
        def get_k_hook(module, input, output):
            key_list.append(output.detach())
        
        def get_v_hook(module, input, output):
            value_list.append(output.detach())
        
        # 註冊hooks
        q_hook = attn.wq.register_forward_hook(get_q_hook)
        k_hook = attn.wk.register_forward_hook(get_k_hook)
        v_hook = attn.wv.register_forward_hook(get_v_hook)
        
        try:
            # 執行前向傳播
            _ = attn(x, pos_cis)
            
            # 驗證Query, Key, Value的形狀
            assert len(query_list) == 1
            assert len(key_list) == 1
            assert len(value_list) == 1
            
            q = query_list[0]
            k = key_list[0]
            v = value_list[0]
            
            # 檢查Q, K, V的形狀
            assert q.shape == (bs, seq_len, tiny_config.n_heads * attn.head_dim)
            assert k.shape == (bs, seq_len, tiny_config.n_kv_heads * attn.head_dim)
            assert v.shape == (bs, seq_len, tiny_config.n_kv_heads * attn.head_dim)
            
            # 手動計算注意力分數，檢查因果掩碼效果
            q_reshaped = q.view(bs, seq_len, tiny_config.n_heads, attn.head_dim).transpose(1, 2)
            k_reshaped = k.view(bs, seq_len, tiny_config.n_kv_heads, attn.head_dim).transpose(1, 2)
            k_reshaped = repeat_kv(k.view(bs, seq_len, tiny_config.n_kv_heads, attn.head_dim), attn.n_rep).transpose(1, 2)
            
            # 計算注意力分數
            scores = (q_reshaped @ k_reshaped.transpose(-2, -1)) / (attn.head_dim ** 0.5)
            
            # 檢查因果掩碼：確保未來信息不會泄露
            # 對於每個位置i，它只能關注位置j<=i的token
            for i in range(seq_len):
                for j in range(i+1, seq_len):
                    # 檢查未來位置的分數是否為負無窮大（掩碼後）
                    mask_value = attn.mask[0, 0, i, j].item()
                    assert mask_value == float('-inf'), f"位置({i},{j})的掩碼值應為負無窮大，實際為{mask_value}"
        
        finally:
            # 移除hooks
            q_hook.remove()
            k_hook.remove()
            v_hook.remove()
            
    def test_edge_cases(self, tiny_config):
        """測試邊緣情況，如極短序列和n_heads=n_kv_heads的特例"""
        # 測試極短序列（單個token）
        bs = 2
        single_token = torch.randn(bs, 1, tiny_config.dim)
        single_pos = precompute_pos_cis(
            dim=tiny_config.dim // tiny_config.n_heads, 
            end=tiny_config.max_seq_len,
            theta=tiny_config.rope_theta
        )[:1]
        
        attn = Attention(tiny_config)
        output, cache = attn(single_token, single_pos, use_cache=True)
        
        # 檢查單個token的輸出形狀
        assert output.shape == single_token.shape
        assert cache[0].shape == (bs, 1, tiny_config.n_kv_heads, attn.head_dim)
        
        # 測試n_heads=n_kv_heads的特例（無需重複KV）
        equal_heads_config = LMConfig(
            dim=64, 
            n_heads=2, 
            n_kv_heads=2,  # 相等的頭數
            max_seq_len=32,
            flash_attn=False,
            dropout=0.0
        )
        
        equal_attn = Attention(equal_heads_config)
        assert equal_attn.n_rep == 1  # 無需重複
        
        # 確認前向傳播正常工作
        sample = torch.randn(bs, 4, equal_heads_config.dim)
        sample_pos = precompute_pos_cis(
            dim=equal_heads_config.dim // equal_heads_config.n_heads, 
            end=equal_heads_config.max_seq_len,
            theta=equal_heads_config.rope_theta
        )[:4]
        
        output, _ = equal_attn(sample, sample_pos)
        assert output.shape == sample.shape 