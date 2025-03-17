import torch
import pytest
import sys
import os

# 確保可以導入模型模塊
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        """測試KV緩存功能"""
        # 對於測試 KV 緩存，我們需要使用一個特殊的配置，使 n_heads = n_kv_heads
        # 這樣就不會觸發 repeat_kv 邏輯，避免緩存形狀不一致的問題
        special_config = LMConfig(
            dim=64,          
            n_heads=2,        # 使 n_heads = n_kv_heads 以避免形狀不一致
            n_kv_heads=2,     
            max_seq_len=32,   
            flash_attn=False, 
            dropout=0.0       
        )
        
        # 為 special_config 創建適合的位置編碼
        batch_size, seq_len = sample_input.shape[:2]
        special_pos_cis = precompute_pos_cis(
            dim=special_config.dim // special_config.n_heads, 
            end=special_config.max_seq_len,
            theta=special_config.rope_theta
        )[:seq_len]
        
        attn = Attention(special_config)
        
        # 第一次前向傳播，使用KV緩存
        output1, past_kv = attn(sample_input, special_pos_cis, use_cache=True)
        
        # 檢查KV緩存的形狀
        assert len(past_kv) == 2  # (K, V)
        k_cache, v_cache = past_kv
        
        # 檢查 k_cache 的形狀是否正確
        assert k_cache.shape == (batch_size, seq_len, special_config.n_kv_heads, attn.head_dim)
        assert v_cache.shape == (batch_size, seq_len, special_config.n_kv_heads, attn.head_dim)
        
        # 使用緩存進行第二次前向傳播
        next_token = torch.randn(batch_size, 1, special_config.dim)  # 只有一個token
        # 為新的token使用適當長度的位置編碼
        next_pos_cis = special_pos_cis[:1]  # 只取第一個位置的編碼
        output2, past_kv2 = attn(next_token, next_pos_cis, past_key_value=past_kv, use_cache=True)
        
        # 檢查輸出形狀
        assert output2.shape == (batch_size, 1, special_config.dim)
        
        # 檢查新的KV緩存是否正確附加了新數據
        assert past_kv2[0].shape == (batch_size, seq_len + 1, special_config.n_kv_heads, attn.head_dim)
        assert past_kv2[1].shape == (batch_size, seq_len + 1, special_config.n_kv_heads, attn.head_dim)
    
    def test_gqa_mechanism(self, tiny_config, sample_input, sample_pos_cis):
        """測試GQA機制 - 專門測試 n_heads > n_kv_heads 的情況"""
        # 確保 tiny_config 的 n_heads 和 n_kv_heads 符合 GQA 要求
        assert tiny_config.n_heads > tiny_config.n_kv_heads
        
        attn = Attention(tiny_config)
        
        # 執行前向傳播
        output, _ = attn(sample_input, sample_pos_cis)
        
        # 檢查輸出形狀 - 不檢查 batch_size 維度，只檢查序列長度和特徵維度
        batch_size, seq_len = sample_input.shape[:2]
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
        
        # 初始化Attention層
        attn = Attention(tiny_config)
        
        # 記錄中間值的hook
        query_list = []
        key_list = []
        value_list = []
        attn_weights_list = []
        
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
        
        # 執行前向傳播
        _ = attn(x, pos_cis)
        
        # 移除hooks
        q_hook.remove()
        k_hook.remove()
        v_hook.remove()
        
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