# https://claude.ai/chat/9340a237-bfbd-41e2-821d-d785888ee8f2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import datetime
import traceback
from moe_implementation import SimplifiedMoE
import os

# 設置日誌輸出
class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        # 寫入開頭的分割線和時間戳記
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

# 設置日誌
log_dir = os.path.dirname(os.path.abspath(__file__))
sys.stdout = Logger(os.path.join(log_dir, "visualized_fnn.log"))
sys.stderr = ErrorLogger(os.path.join(log_dir, "visualized_fnn.log"))

# 捕獲未處理的異常並記錄
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # 正常處理 KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"[ERROR] 未捕獲的異常:\n{error_message}")

sys.excepthook = handle_exception

class VisualizedFeedForward(nn.Module):
    """
    帶有數據流可視化功能的前饋網路
    """
    def __init__(self, d_model=4, d_ff=8, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 創建兩個線性層，不使用偏置以便觀察轉換
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        
        # 設置激活函數
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"不支援的激活函數: {activation}")
        
        # 初始化權重以便觀察
        with torch.no_grad():
            # 設置第一層權重為單位矩陣擴展
            w1 = torch.zeros(d_ff, d_model)
            for i in range(min(d_model, d_ff)):
                w1[i, i] = 2.0  # 使用 2.0 作為乘數以便觀察
            self.linear1.weight.copy_(w1)
            
            # 設置第二層權重
            w2 = torch.zeros(d_model, d_ff)
            for i in range(min(d_model, d_ff)):
                w2[i, i] = 0.5  # 使用 0.5 以便與第一層 2.0 相乘後得到 1.0
            self.linear2.weight.copy_(w2)
    
    def forward(self, x, print_details=True):
        """
        前向傳播並打印詳細信息
        
        參數:
            x: 形狀為 [batch_size, seq_len, d_model] 的輸入張量
            print_details: 是否打印中間計算過程
            
        回傳:
            形狀為 [batch_size, seq_len, d_model] 的輸出張量
        """
        batch_size, seq_len, d_model = x.shape
        
        if print_details:
            print("\n" + "=" * 50)
            print("前饋網路數據流可視化")
            print("=" * 50)
            print(f"\n輸入形狀: {x.shape}")
            print(f"輸入數值:\n{x.detach().numpy().round(2)}")
            
            # 顯示第一層權重
            print(f"\n第一層權重 (從 {d_model} 到 {self.d_ff}):")
            print(f"{self.linear1.weight.detach().numpy().round(2)}")
        
        # 第一個線性層轉換
        linear1_out = self.linear1(x)
        
        if print_details:
            print(f"\n第一層線性轉換輸出:")
            print(f"形狀: {linear1_out.shape}")
            print(f"數值:\n{linear1_out.detach().numpy().round(2)}")
        
        # 應用激活函數
        activation_out = self.activation(linear1_out)
        
        if print_details:
            print(f"\n激活函數輸出:")
            print(f"形狀: {activation_out.shape}")
            print(f"數值:\n{activation_out.detach().numpy().round(2)}")
            
            # 激活前後的差異比較
            diff = activation_out - linear1_out
            # 計算差異的絕對值總和，以查看激活函數的影響
            impact = diff.abs().sum().item()
            print(f"\n激活函數影響度: {impact:.4f}")
            
            # 顯示第二層權重
            print(f"\n第二層權重 (從 {self.d_ff} 到 {d_model}):")
            print(f"{self.linear2.weight.detach().numpy().round(2)}")
        
        # 第二個線性層轉換
        output = self.linear2(activation_out)
        
        if print_details:
            print(f"\n第二層線性轉換輸出 (最終輸出):")
            print(f"形狀: {output.shape}")
            print(f"數值:\n{output.detach().numpy().round(2)}")
            
            # 對比輸入和輸出的差異
            print(f"\n輸入與輸出對比:")
            comp_table = []
            for i in range(min(3, batch_size * seq_len)):  # 只顯示前三個令牌
                bidx, sidx = i // seq_len, i % seq_len
                comp_table.append([
                    f"令牌[{bidx},{sidx}]",
                    f"{x[bidx, sidx].detach().numpy().round(2)}",
                    f"{output[bidx, sidx].detach().numpy().round(2)}",
                    f"{(output[bidx, sidx] - x[bidx, sidx]).detach().numpy().round(2)}"
                ])
            
            headers = ["位置", "輸入", "輸出", "差異"]
            print(tabulate(comp_table, headers=headers))
        
        return output


# 用於比較 FNN 和 MoE 的函數
def compare_fnn_moe():
    print("\n" + "=" * 50)
    print("FNN vs MoE 對比")
    print("=" * 50)
    
    # 測試數據
    x = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ]
    ])
    
    # 創建模型
    fnn = VisualizedFeedForward(d_model=4, d_ff=8)
    moe = SimplifiedMoE(d_model=4, num_experts=3, k=2)
    
    # 運行模型
    print("\n【標準前饋網路處理】")
    fnn_out = fnn(x)
    
    print("\n【Mixture of Experts 處理】")
    moe_out = moe(x)
    
    print("\n【FNN vs MoE 關鍵區別】")
    print("1. FNN: 所有令牌使用相同的參數")
    print("2. MoE: 每個令牌動態選擇不同的專家")
    print("3. FNN: 計算路徑固定")
    print("4. MoE: 計算路徑由令牌內容決定")
    
    print("\n【輸出比較】")
    print(f"FNN 輸出:\n{fnn_out.detach().numpy().round(4)}")
    print(f"MoE 輸出:\n{moe_out.detach().numpy().round(4)}")


# 用於表格顯示的簡易函數
def tabulate(data, headers):
    """簡易的表格顯示函數"""
    # 計算每列的最大寬度
    widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
    
    # 建立表頭
    header = " | ".join(f"{h:{w}s}" for h, w in zip(headers, widths))
    separator = "-+-".join("-" * w for w in widths)
    
    # 建立表格內容
    rows = [header, separator]
    for row in data:
        rows.append(" | ".join(f"{str(cell):{w}s}" for cell, w in zip(row, widths)))
    
    return "\n".join(rows)


# 運行演示
def run_demo():
    # 創建測試數據
    x = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        ]
    ])
    
    # 創建並運行可視化的前饋網路
    fnn = VisualizedFeedForward(d_model=4, d_ff=8)
    fnn_output = fnn(x)
    
    # 比較 FNN 和 MoE
    compare_fnn_moe()


if __name__ == "__main__":
    run_demo()
