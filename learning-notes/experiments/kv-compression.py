import torch
import torch.nn as nn
import numpy as np
import sys
import datetime
import traceback
import os

# 設置日誌輸出
class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.log_file.write("="*50 + "\n")
        self.log_file.write(f"日誌開始時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("="*50 + "\n\n")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

class ErrorLogger:
    def __init__(self, file_path):
        self.terminal = sys.stderr
        self.log_file = open(file_path, 'a')
        
    def write(self, message):
        self.terminal.write(message)
        if message.strip():
            if "error" in message.lower() or "exception" in message.lower():
                self.log_file.write(f"[ERROR] {message}")
            elif "warning" in message.lower():
                self.log_file.write(f"[WARNING] {message}")
            else:
                self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# 初始化日誌系統
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "kv-compression.log")
sys.stdout = Logger(log_file)
sys.stderr = ErrorLogger(log_file)

# 捕獲未處理的異常
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    sys.stderr.write(f"[ERROR] 未捕獲的異常:\n{error_message}")

sys.excepthook = handle_exception

class GQAProjection(nn.Module):
    """
    Grouped-Query Attention的投影層，展示Key/Value維度壓縮
    """
    def __init__(self, d_model=768, n_query_heads=8, n_kv_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_query_heads
        
        # 標準Query投影：d_model -> d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 壓縮的Key投影：d_model -> (head_dim * n_kv_heads)
        self.k_proj = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        
        # 壓縮的Value投影：d_model -> (head_dim * n_kv_heads)
        self.v_proj = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        
        # 初始化投影矩陣，使其易於觀察
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化投影矩陣以便觀察壓縮過程"""
        with torch.no_grad():
            # 初始化Q投影為簡單的識別矩陣
            q_weight = torch.eye(self.d_model)
            self.q_proj.weight.copy_(q_weight)
            
            # 初始化K投影為壓縮矩陣
            k_weight = torch.zeros(self.head_dim * self.n_kv_heads, self.d_model)
            for i in range(self.head_dim * self.n_kv_heads):
                # 對應的輸入維度索引，重複映射多個輸入維度到同一個輸出維度
                in_idx = i * (self.d_model // (self.head_dim * self.n_kv_heads))
                k_weight[i, in_idx] = 1.0
            self.k_proj.weight.copy_(k_weight)
            
            # 初始化V投影為壓縮矩陣
            v_weight = k_weight.clone()  # 使用與K相同的壓縮方式
            self.v_proj.weight.copy_(v_weight)
    
    def forward(self, x, print_details=True):
        """前向傳播，展示壓縮過程"""
        batch_size, seq_len, _ = x.shape
        
        if print_details:
            sys.stdout.write("\n===== GQA投影與維度壓縮 =====\n")
            sys.stdout.write(f"模型維度 (d_model): {self.d_model}\n")
            sys.stdout.write(f"Query頭數: {self.n_query_heads}\n")
            sys.stdout.write(f"Key/Value頭數: {self.n_kv_heads}\n")
            sys.stdout.write(f"每個頭的維度: {self.head_dim}\n")
            sys.stdout.write(f"壓縮率: {self.n_query_heads / self.n_kv_heads}x\n\n")
            
            sys.stdout.write(f"輸入形狀: {x.shape}\n")
            sys.stdout.write(f"示例輸入向量 (截取): {x[0, 0, :10].detach().numpy().round(2)}...\n")
        
        # 應用投影
        q = self.q_proj(x)  # [batch_size, seq_len, d_model]
        k = self.k_proj(x)  # [batch_size, seq_len, head_dim * n_kv_heads]
        v = self.v_proj(x)  # [batch_size, seq_len, head_dim * n_kv_heads]
        
        if print_details:
            sys.stdout.write("\n===== 線性投影後 =====\n")
            sys.stdout.write(f"Q投影形狀: {q.shape}\n")
            sys.stdout.write(f"K投影形狀: {k.shape}\n")
            sys.stdout.write(f"V投影形狀: {v.shape}\n")
            
            sys.stdout.write(f"\nQ投影後 (截取): {q[0, 0, :10].detach().numpy().round(2)}...\n")
            sys.stdout.write(f"K投影後 (全部): {k[0, 0].detach().numpy().round(2)}\n")
            
            # 展示投影矩陣的壓縮效果
            compression_ratio = self.d_model / (self.head_dim * self.n_kv_heads)
            
            sys.stdout.write(f"\n===== 維度壓縮過程 =====\n")
            sys.stdout.write(f"原始維度: {self.d_model}\n")
            sys.stdout.write(f"壓縮後維度: {self.head_dim * self.n_kv_heads}\n")
            sys.stdout.write(f"壓縮比例: {compression_ratio}:1\n")
            
            # 視覺化展示壓縮過程中的維度映射
            sys.stdout.write("\n維度映射示例 (輸入→輸出):\n")
            map_indexes = 5  # 只顯示前幾個維度的映射
            for i in range(min(map_indexes, self.head_dim * self.n_kv_heads)):
                # 找出矩陣中該行最大值的位置
                in_dims = torch.where(self.k_proj.weight[i] > 0.5)[0]
                in_range = f"{in_dims[0].item()}"
                if len(in_dims) > 1:
                    in_range += f"-{in_dims[-1].item()}"
                sys.stdout.write(f"輸入維度 {in_range} → K輸出維度 {i}\n")
        
        # 重塑為多頭形式
        q = q.view(batch_size, seq_len, self.n_query_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        if print_details:
            sys.stdout.write("\n===== 重塑為多頭形式 =====\n")
            sys.stdout.write(f"Q多頭形狀: {q.shape} (batch, seq_len, n_query_heads, head_dim)\n")
            sys.stdout.write(f"K多頭形狀: {k.shape} (batch, seq_len, n_kv_heads, head_dim)\n")
            sys.stdout.write(f"V多頭形狀: {v.shape} (batch, seq_len, n_kv_heads, head_dim)\n")
        
        return q, k, v


def compare_attention_mechanisms():
    """比較不同注意力機制的參數量和內存使用"""
    d_model = 768
    seq_len = 1024
    batch_size = 1
    
    # 不同機制的頭數設置
    mechanisms = {
        "標準多頭注意力 (MHA)": {"n_q": 8, "n_kv": 8},
        "分組查詢注意力 (GQA)": {"n_q": 8, "n_kv": 2},
        "多查詢單鍵值注意力 (MQA)": {"n_q": 8, "n_kv": 1},
    }
    
    sys.stdout.write("\n===== 不同注意力機制的比較 =====\n")
    sys.stdout.write(f"模型維度: {d_model}, 序列長度: {seq_len}\n")
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write(f"{'機制':<25} {'參數量':<15} {'KV內存':<15} {'性能影響':<10}\n")
    sys.stdout.write("-" * 60 + "\n")
    
    # 基準參數 (MHA)
    base_params = 3 * d_model * d_model
    base_kv_memory = 2 * d_model * seq_len
    
    for name, config in mechanisms.items():
        n_q = config["n_q"]
        n_kv = config["n_kv"]
        head_dim = d_model // n_q
        
        # 計算參數
        q_params = d_model * d_model
        kv_params = 2 * d_model * (head_dim * n_kv)
        total_params = q_params + kv_params
        
        # 計算KV內存
        kv_memory = 2 * (head_dim * n_kv) * seq_len
        
        # 估計的性能影響 (基於研究數據的近似值)
        if n_kv == n_q:  # MHA
            perf_impact = "基準"
        elif n_kv == 1:  # MQA
            perf_impact = "-2~4%"
        else:  # GQA
            perf_impact = "-0.5~1%"
        
        # 打印比較結果
        params_reduction = (1 - total_params / base_params) * 100
        memory_reduction = (1 - kv_memory / base_kv_memory) * 100
        
        sys.stdout.write(f"{name:<25} {total_params:,} ({params_reduction:.1f}%) {kv_memory:,} ({memory_reduction:.1f}%) {perf_impact:<10}\n")
    
    sys.stdout.write("-" * 60 + "\n")


# 運行演示代碼
def run_demo():
    # 創建輸入
    d_model = 768
    batch_size = 1
    seq_len = 2
    
    # 創建測試輸入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 測試GQA投影
    gqa_proj = GQAProjection(d_model=768, n_query_heads=8, n_kv_heads=2)
    q, k, v = gqa_proj(x)
    
    # 比較不同注意力機制
    compare_attention_mechanisms()


if __name__ == "__main__":
    run_demo()
