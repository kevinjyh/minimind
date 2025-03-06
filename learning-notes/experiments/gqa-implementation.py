import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisualizedGQA(nn.Module):
    """
    帶有數據流可視化功能的 Grouped-Query Attention (GQA) 實作
    """
    def __init__(self, 
                 d_model=512,        # 模型維度
                 n_query_heads=8,    # Query 的頭數
                 n_kv_heads=2,       # Key/Value 的頭數 (Query 的 1/4)
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_query_heads
        
        assert d_model % n_query_heads == 0, "d_model 必須能被 n_query_heads 整除"
        assert n_query_heads % n_kv_heads == 0, "Query 頭數必須是 KV 頭數的整數倍"
        
        self.kv_groups = n_query_heads // n_kv_heads  # 每個 KV 頭被多少個 Query 頭共享
        
        # 分別為 Q、K、V 創建投影矩陣
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 為了可視化而特意初始化權重
        with torch.no_grad():
            # 初始化 Q 投影為近似單位矩陣
            q_weights = torch.zeros(d_model, d_model)
            for i in range(d_model):
                q_weights[i, i] = 1.0
            self.q_proj.weight.copy_(q_weights)
            
            # 初始化 K 投影 (維度較小)
            k_weights = torch.zeros(self.n_kv_heads * self.head_dim, d_model)
            for i in range(min(k_weights.shape)):
                k_weights[i, i] = 1.0
            self.k_proj.weight.copy_(k_weights)
            
            # 初始化 V 投影 (維度較小)
            v_weights = torch.zeros(self.n_kv_heads * self.head_dim, d_model)
            for i in range(min(v_weights.shape)):
                v_weights[i, i] = 1.0
            self.v_proj.weight.copy_(v_weights)
            
            # 初始化輸出投影
            out_weights = torch.zeros(d_model, d_model)
            for i in range(d_model):
                out_weights[i, i] = 1.0
            self.output_proj.weight.copy_(out_weights)
    
    def reshape_for_scores(self, x, n_heads):
        """重塑張量以便計算注意力分數"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, n_heads, self.head_dim)
        # 轉置為 [batch_size, n_heads, seq_len, head_dim]
        return x.transpose(1, 2)
    
    def forward(self, x, print_details=True):
        """
        前向傳播並打印詳細信息
        
        參數:
            x: 形狀為 [batch_size, seq_len, d_model] 的輸入張量
            
        回傳:
            形狀為 [batch_size, seq_len, d_model] 的輸出張量
        """
        batch_size, seq_len, _ = x.shape
        
        if print_details:
            print("\n" + "=" * 50)
            print("Grouped-Query Attention (GQA) 數據流可視化")
            print("=" * 50)
            print(f"\n模型參數:")
            print(f"- 模型維度 (d_model): {self.d_model}")
            print(f"- Query 頭數: {self.n_query_heads}")
            print(f"- Key/Value 頭數: {self.n_kv_heads}")
            print(f"- 每個頭的維度: {self.head_dim}")
            print(f"- KV 組共享比例: 每個 KV 頭被 {self.kv_groups} 個 Query 頭共享")
            
            print(f"\n輸入形狀: {x.shape}")
            print(f"輸入範例 (第一個序列的前兩個位置):")
            print(f"{x[0, 0:2].detach().numpy().round(2)}")
        
        # 1. 計算 Q、K、V 投影
        q = self.q_proj(x)  # [batch_size, seq_len, d_model]
        k = self.k_proj(x)  # [batch_size, seq_len, n_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, n_kv_heads * head_dim]
        
        if print_details:
            print(f"\n投影後的形狀:")
            print(f"Q: {q.shape}")
            print(f"K: {k.shape}")
            print(f"V: {v.shape}")
        
        # 2. 重塑為多頭形式
        q = self.reshape_for_scores(q, self.n_query_heads)  # [batch_size, n_query_heads, seq_len, head_dim]
        k = self.reshape_for_scores(k, self.n_kv_heads)     # [batch_size, n_kv_heads, seq_len, head_dim]
        v = self.reshape_for_scores(v, self.n_kv_heads)     # [batch_size, n_kv_heads, seq_len, head_dim]
        
        if print_details:
            print(f"\n重塑後的多頭形狀:")
            print(f"Q: {q.shape} - {self.n_query_heads} 個頭")
            print(f"K: {k.shape} - {self.n_kv_heads} 個頭")
            print(f"V: {v.shape} - {self.n_kv_heads} 個頭")
        
        # 3. 展示頭的分組方式 (QKV維度分解)
        if print_details:
            print(f"\n頭的分組方式 (以第一個批次第一個位置為例):")
            print(f"- 查詢維度 (Q): {q.shape[-1]} x {q.shape[1]} = {q.shape[-1] * q.shape[1]}")
            for i in range(min(3, self.n_query_heads)):
                print(f"  Q頭 #{i} 維度值: {q[0, i, 0, :4].detach().numpy().round(2)}...")
            
            print(f"- 鍵值維度 (K/V): {k.shape[-1]} x {k.shape[1]} = {k.shape[-1] * k.shape[1]}")
            for i in range(min(2, self.n_kv_heads)):
                print(f"  K頭 #{i} 維度值: {k[0, i, 0, :4].detach().numpy().round(2)}...")
                print(f"  V頭 #{i} 維度值: {v[0, i, 0, :4].detach().numpy().round(2)}...")
        
        # 4. GQA 的核心：複製 K 和 V 頭以匹配 Q 頭的數量
        # 創建一個映射，顯示每個 Query 頭使用哪個 KV 頭
        q_to_kv_map = torch.div(torch.arange(self.n_query_heads), self.kv_groups, rounding_mode='floor')
        
        if print_details:
            print(f"\nGQA 核心映射 - 每個 Query 頭使用哪個 KV 頭:")
            for q_idx in range(self.n_query_heads):
                kv_idx = q_to_kv_map[q_idx].item()
                print(f"  Query 頭 #{q_idx} → 使用 KV 頭 #{kv_idx}")
        
        # 5. 執行注意力計算
        # 為了計算方便，我們在這裡將 k、v 擴展以匹配 q 的頭數
        k_expanded = torch.zeros_like(q)  # [batch_size, n_query_heads, seq_len, head_dim]
        v_expanded = torch.zeros_like(q)  # [batch_size, n_query_heads, seq_len, head_dim]
        
        for q_idx in range(self.n_query_heads):
            kv_idx = q_to_kv_map[q_idx].item()
            k_expanded[:, q_idx] = k[:, kv_idx]
            v_expanded[:, q_idx] = v[:, kv_idx]
        
        if print_details:
            print(f"\n擴展後的 KV 形狀:")
            print(f"K_expanded: {k_expanded.shape}")
            print(f"V_expanded: {v_expanded.shape}")
        
        # 6. 計算注意力分數
        # 縮放點積注意力
        attention_scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # [batch_size, n_query_heads, seq_len, seq_len]
        
        # 應用 softmax 獲取注意力權重
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if print_details:
            print(f"\n注意力分數形狀: {attention_scores.shape}")
            print(f"注意力矩陣示例 (第一個批次第一個頭的部分值):")
            print(f"{attention_scores[0, 0, :2, :2].detach().numpy().round(2)}")
            
            print(f"\n注意力概率形狀: {attention_probs.shape}")
            print(f"注意力概率示例 (第一個批次第一個頭的部分值):")
            print(f"{attention_probs[0, 0, :2, :2].detach().numpy().round(4)}")
        
        # 7. 計算注意力輸出
        context = torch.matmul(attention_probs, v_expanded)
        # [batch_size, n_query_heads, seq_len, head_dim]
        
        # 8. 重塑回原始維度
        context = context.transpose(1, 2).contiguous()
        # [batch_size, seq_len, n_query_heads, head_dim]
        
        context = context.reshape(batch_size, seq_len, self.d_model)
        # [batch_size, seq_len, d_model]
        
        if print_details:
            print(f"\n注意力輸出 (重塑前): {context.shape}")
        
        # 9. 應用輸出投影
        output = self.output_proj(context)
        # [batch_size, seq_len, d_model]
        
        if print_details:
            print(f"\n最終輸出形狀: {output.shape}")
            print(f"輸出示例 (第一個批次的前兩個位置):")
            print(f"{output[0, 0:2].detach().numpy().round(2)}")
            
            # 顯示輸入和輸出的變化
            print(f"\n輸入與輸出對比 (第一個序列的第一個位置):")
            print(f"輸入: {x[0, 0, :5].detach().numpy().round(2)}...")
            print(f"輸出: {output[0, 0, :5].detach().numpy().round(2)}...")
        
        return output


# 運行 GQA 演示
def demo_gqa():
    print("\n" + "=" * 50)
    print("Grouped-Query Attention (GQA) 演示")
    print("=" * 50)
    
    # 設置參數 - 使用較小的維度以便觀察
    d_model = 64      # 模型維度
    n_query_heads = 8 # Query 頭數
    n_kv_heads = 2    # KV 頭數 (Query 頭數的 1/4)
    
    # 創建模型
    gqa = VisualizedGQA(d_model=d_model, n_query_heads=n_query_heads, n_kv_heads=n_kv_heads)
    
    # 創建輸入數據
    batch_size = 1
    seq_len = 4
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 對於較好的觀察，可以使用較為特殊的輸入
    with torch.no_grad():
        # 設置第一個位置的前幾個維度
        x[0, 0, :8] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 設置第二個位置的前幾個維度
        x[0, 1, :8] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 運行模型
    output = gqa(x)
    
    # 比較標準 MHA 和 GQA 的效率
    print("\n" + "=" * 50)
    print("MHA vs GQA 效率比較")
    print("=" * 50)
    
    # 計算 MHA 和 GQA 的參數量
    mha_params = 3 * d_model * d_model  # Q, K, V 投影 + 輸出投影
    gqa_params = d_model * d_model + 2 * (n_kv_heads * d_model * d_model // n_query_heads) + d_model * d_model
    
    # 計算 MHA 和 GQA 的計算量
    # 假設序列長度為 L
    L = 1024  # 假設序列長度
    mha_flops = 2 * L * L * d_model  # 注意力計算的複雜度
    gqa_flops = 2 * L * L * d_model * (n_kv_heads / n_query_heads)  # GQA 的複雜度
    
    print(f"MHA 參數量: {mha_params}")
    print(f"GQA 參數量: {gqa_params}")
    print(f"參數減少比例: {(1 - gqa_params / mha_params) * 100:.2f}%")
    
    print(f"\nMHA 計算量 (FLOPS): {mha_flops}")
    print(f"GQA 計算量 (FLOPS): {gqa_flops}")
    print(f"計算減少比例: {(1 - gqa_flops / mha_flops) * 100:.2f}%")


# 顯示 GQA 與 MHA 的優化比較
def compare_gqa_mha():
    print("\n" + "=" * 50)
    print("GQA 與標準 MHA 的比較")
    print("=" * 50)
    
    print("\n【GQA 的主要優點】")
    print("1. 參數效率：KV 頭數減少，但維持 Query 頭的表達能力")
    print("2. 計算效率：減少了注意力計算量，特別是在長序列上")
    print("3. 記憶體優化：減少了 KV 緩存的大小，適合推理階段")
    
    print("\n【標準 MHA vs GQA 結構差異】")
    print("- MHA: 每個 Query 頭對應獨立的 Key/Value 頭")
    print("- GQA: 多個 Query 頭共享同一個 Key/Value 頭")
    
    print("\n【GQA 在 LLM 中的應用】")
    print("- 被用於 PaLM、LLaMA 2、Mixtral 等模型")
    print("- 在推理優化中特別有用，可減少 KV 緩存佔用")
    print("- 實驗表明，與標準 MHA 相比，GQA 可以在保持性能的同時減少計算")


if __name__ == "__main__":
    demo_gqa()
    compare_gqa_mha()
