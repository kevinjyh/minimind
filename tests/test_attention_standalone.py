import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytest
import numpy as np

class SimpleAttention(nn.Module):
    """簡化版的自注意力機制，用於理解核心原理"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 確保隱藏維度可以被頭數整除
        assert self.head_dim * num_heads == hidden_size, "hidden_size必須能被num_heads整除"
        
        # 創建查詢、鍵、值和輸出投影
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x, causal_mask=True):
        """
        前向傳播計算
        
        參數:
            x: 形狀為 [batch_size, seq_len, hidden_size] 的輸入張量
            causal_mask: 是否應用因果掩碼（適用於生成模型）
            
        返回:
            形狀為 [batch_size, seq_len, hidden_size] 的輸出張量
        """
        batch_size, seq_len, _ = x.shape
        
        # 線性投影並分割頭
        # 形狀變化: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 轉置準備多頭注意力計算
        # 形狀變化: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 計算注意力分數: (Q·K^T)/sqrt(d_k)
        # 形狀變化: [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, head_dim, seq_len]
        #         -> [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 如果需要因果掩碼（只關注序列中的前面位置）
        if causal_mask:
            # 創建上三角掩碼（主對角線以上為1，表示被掩蓋）
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores.masked_fill_(mask, float("-inf"))
        
        # 使用softmax獲取注意力權重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 應用注意力權重到值矩陣
        # 形狀: [batch_size, num_heads, seq_len, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        #     -> [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # 轉置回原始形狀並合併頭
        # 形狀: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        #     -> [batch_size, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 最後的線性投影
        output = self.o_proj(context)
        
        return output, attn_weights

class TestSimpleAttention:
    """測試簡化版的自注意力機制"""
    
    @pytest.fixture
    def batch_size(self):
        return 2
    
    @pytest.fixture
    def seq_len(self):
        return 10
    
    @pytest.fixture
    def hidden_size(self):
        return 64
    
    @pytest.fixture
    def num_heads(self):
        return 4
    
    @pytest.fixture
    def sample_input(self, batch_size, seq_len, hidden_size):
        """創建樣本輸入"""
        return torch.randn(batch_size, seq_len, hidden_size)
    
    def test_attention_shape(self, sample_input, hidden_size, num_heads):
        """測試注意力機制的輸出形狀"""
        attn = SimpleAttention(hidden_size, num_heads)
        output, attn_weights = attn(sample_input)
        
        # 驗證輸出形狀與輸入形狀相同
        assert output.shape == sample_input.shape
        
        # 驗證注意力權重形狀正確
        batch_size, seq_len, _ = sample_input.shape
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_causal_mask(self, sample_input, hidden_size, num_heads):
        """測試因果掩碼的有效性"""
        attn = SimpleAttention(hidden_size, num_heads)
        _, attn_weights = attn(sample_input, causal_mask=True)
        
        # 獲取第一個批次、第一個頭的注意力權重
        weights = attn_weights[0, 0].detach().numpy()
        
        # 確保上三角（不包括對角線）全為0，表示每個token只能看到自己和之前的token
        seq_len = sample_input.shape[1]
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                # 因為softmax後極小的數值是接近0但不等於0，所以我們檢查是否非常接近0
                assert abs(weights[i, j]) < 1e-8, f"位置({i},{j})的值應該接近0，但得到{weights[i,j]}"
    
    def test_no_causal_mask(self, sample_input, hidden_size, num_heads):
        """測試沒有因果掩碼時的行為"""
        attn = SimpleAttention(hidden_size, num_heads)
        _, attn_weights = attn(sample_input, causal_mask=False)
        
        # 沒有因果掩碼時，權重分佈應該更均勻
        weights = attn_weights[0, 0].detach().numpy()
        
        # 檢查上三角部分不是全為0
        seq_len = sample_input.shape[1]
        upper_triangle_sum = np.sum(weights[np.triu_indices(seq_len, k=1)])
        assert upper_triangle_sum > 0, "沒有因果掩碼時，上三角部分不應該全為0"
    
    def test_attention_pattern_with_one_hot(self, hidden_size, num_heads):
        """測試使用one-hot向量時的注意力模式"""
        # 創建簡單的one-hot向量序列
        batch_size, seq_len = 1, 5
        x = torch.zeros(batch_size, seq_len, hidden_size)
        
        # 在第一個位置設置固定的pattern，使其成為獨特的"查詢目標"
        x[:, 0, :5] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        
        # 在其他位置設置不同的pattern
        x[:, 1, :5] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
        x[:, 2, :5] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        x[:, 3, :5] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        x[:, 4, :5] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        
        # 創建特殊的簡化注意力層，使第一個token尋找與第二個token相似的模式
        attn = SimpleAttention(hidden_size, num_heads)
        
        # 將q_proj的權重設為單位矩陣，這樣q就是原始輸入
        attn.q_proj.weight.data = torch.eye(hidden_size)
        
        # 將k_proj的權重也設為單位矩陣，這樣k也是原始輸入
        attn.k_proj.weight.data = torch.eye(hidden_size)
        
        # 禁用因果掩碼，讓所有token可以互相看到
        _, attn_weights = attn(x, causal_mask=False)
        
        # 分析第一個token對其他token的注意力分佈
        first_token_attn = attn_weights[0, 0, 0].detach().numpy()
        
        # 第一個token應該主要關注自己（索引0），因為每個token的表示都是唯一的one-hot
        assert first_token_attn[0] > 0.5, f"第一個token對自己的注意力應該大於0.5，但得到{first_token_attn[0]}"
        
    def test_multi_head_independence(self, sample_input, hidden_size, num_heads):
        """測試多頭注意力機制的獨立性"""
        attn = SimpleAttention(hidden_size, num_heads)
        
        # 將每個頭的權重設置為不同的模式
        head_dim = hidden_size // num_heads
        for h in range(num_heads):
            # 設置查詢投影，使每個頭關注不同的特徵
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            
            # 清除這部分的權重
            attn.q_proj.weight.data[:, start_idx:end_idx] = 0
            
            # 在對角線位置設置非零值，使頭h只關注輸入的該部分
            for i in range(start_idx, end_idx):
                attn.q_proj.weight.data[i, i] = 1.0
        
        # 使用相同的方法設置鍵投影
        for h in range(num_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            attn.k_proj.weight.data[:, start_idx:end_idx] = 0
            for i in range(start_idx, end_idx):
                attn.k_proj.weight.data[i, i] = 1.0
        
        # 前向傳播並檢查注意力權重
        _, attn_weights = attn(sample_input, causal_mask=False)
        
        # 每個頭的注意力模式應該不同
        # 我們可以通過計算頭之間的相關性來檢驗這一點
        batch_size, seq_len, _ = sample_input.shape
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                head_i = attn_weights[0, i].flatten().detach().numpy()
                head_j = attn_weights[0, j].flatten().detach().numpy()
                
                # 計算相關性（這裡我們只是簡單檢查它們是否完全相同）
                assert not np.array_equal(head_i, head_j), f"頭{i}和頭{j}不應該有完全相同的注意力模式"
                
    def test_self_attention_demonstration(self):
        """一個更直觀的自注意力示例"""
        # 創建一個包含三種不同'主題'的簡單序列
        hidden_size = 6
        seq_len = 6
        
        # 創建三個one-hot向量代表三個主題
        topic1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        topic2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        topic3 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # 創建一個序列，其中每兩個token屬於同一主題
        x = torch.zeros(1, seq_len, hidden_size)
        x[0, 0] = topic1
        x[0, 1] = topic1
        x[0, 2] = topic2
        x[0, 3] = topic2
        x[0, 4] = topic3
        x[0, 5] = topic3
        
        # 創建一個簡單的注意力層
        attn = SimpleAttention(hidden_size, num_heads=1)
        
        # 將投影矩陣設為單位矩陣，這樣我們可以直接看到主題之間的關係
        attn.q_proj.weight.data = torch.eye(hidden_size)
        attn.k_proj.weight.data = torch.eye(hidden_size)
        attn.v_proj.weight.data = torch.eye(hidden_size)
        attn.o_proj.weight.data = torch.eye(hidden_size)
        
        # 前向傳播並獲取注意力權重
        _, attn_weights = attn(x, causal_mask=False)
        
        # 分析注意力模式
        weights = attn_weights[0, 0].detach().numpy()
        
        # 同一主題的token應該彼此高度關注
        # 主題1 (token 0-1)
        assert weights[0, 0] > 0.4, f"Token 0對自己的注意力應該>0.4，但得到{weights[0,0]}"
        assert weights[0, 1] > 0.4, f"Token 0對Token 1的注意力應該>0.4，但得到{weights[0,1]}"
        assert weights[1, 0] > 0.4, f"Token 1對Token 0的注意力應該>0.4，但得到{weights[1,0]}"
        
        # 主題2 (token 2-3)
        assert weights[2, 2] > 0.4, f"Token 2對自己的注意力應該>0.4，但得到{weights[2,2]}"
        assert weights[2, 3] > 0.4, f"Token 2對Token 3的注意力應該>0.4，但得到{weights[2,3]}"
        
        # 主題3 (token 4-5)
        assert weights[4, 4] > 0.4, f"Token 4對自己的注意力應該>0.4，但得到{weights[4,4]}"
        assert weights[4, 5] > 0.4, f"Token 4對Token 5的注意力應該>0.4，但得到{weights[4,5]}"
        
        # 不同主題之間的注意力應該很低
        assert weights[0, 2] < 0.1, f"Token 0對Token 2的注意力應該<0.1，但得到{weights[0,2]}"
        assert weights[0, 4] < 0.1, f"Token 0對Token 4的注意力應該<0.1，但得到{weights[0,4]}"
        assert weights[2, 4] < 0.1, f"Token 2對Token 4的注意力應該<0.1，但得到{weights[2,4]}" 