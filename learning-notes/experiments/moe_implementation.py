# https://claude.ai/chat/9340a237-bfbd-41e2-821d-d785888ee8f2
import torch
import torch.nn as nn
import torch.nn.functional as F
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
sys.stdout = Logger(os.path.join(log_dir, "moe_implementation.log"))
sys.stderr = ErrorLogger(os.path.join(log_dir, "moe_implementation.log"))

# 捕獲未處理的異常並記錄
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # 正常處理 KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"[ERROR] 未捕獲的異常:\n{error_message}")

sys.excepthook = handle_exception

class Expert(nn.Module):
    """
    專家模型，基本上就是一個標準的前饋網路
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"不支援的激活函數: {activation}")
    
    def forward(self, x):
        # 與標準前饋網路相同的前向傳播
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MoE(nn.Module):
    """
    Mixture of Experts 實作
    """
    def __init__(self, d_model, d_ff, num_experts, k=2, dropout=0.1, activation='gelu'):
        """
        參數:
            d_model (int): 輸入和輸出的維度
            d_ff (int): 專家隱藏層的維度
            num_experts (int): 專家的數量
            k (int): 每個令牌選擇的專家數量 (top-k 專家)
            dropout (float): dropout 率
            activation (str): 激活函數類型
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        
        # 建立多個專家
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout, activation) 
                                      for _ in range(num_experts)])
        
        # 路由器網路：決定每個令牌使用哪些專家
        self.router = nn.Linear(d_model, num_experts)
        
        # 為了可視化，額外添加一個變數來儲存路由分數
        self.routing_weights = None
    
    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: 形狀為 [batch_size, seq_len, d_model] 的輸入張量
            
        回傳:
            形狀為 [batch_size, seq_len, d_model] 的輸出張量
        """
        batch_size, seq_len, d_model = x.shape
        
        # 將輸入重塑為二維以簡化處理
        x_flat = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # 計算路由分數
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # 應用 softmax 得到分配給每個專家的權重
        routing_weights = F.softmax(router_logits, dim=1)
        self.routing_weights = routing_weights.detach()  # 儲存用於可視化
        
        # 選擇 top-k 專家
        routing_weights, indices = torch.topk(routing_weights, self.k, dim=1)
        
        # 重新正規化 top-k 權重，使其總和為 1
        routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
        
        # 初始化輸出張量
        final_output = torch.zeros_like(x_flat)
        
        # 對每個專家應用計算
        for expert_idx in range(self.num_experts):
            # 找出使用當前專家的所有位置
            # 創建一個布爾遮罩，指示哪些位置使用了這個專家
            expert_mask = (indices == expert_idx).any(dim=1)
            
            if expert_mask.any():
                # 僅提取需要此專家的令牌
                expert_inputs = x_flat[expert_mask]
                
                # 應用專家計算
                expert_output = self.experts[expert_idx](expert_inputs)
                
                # 找出在 indices 的哪些位置使用了當前專家
                expert_positions = (indices == expert_idx).int()
                
                # 取出對應的權重
                # 實際中，需要更複雜的索引方式，這裡簡化為遍歷
                for token_idx, token_experts in enumerate(indices):
                    if expert_idx in token_experts:
                        # 找出 expert_idx 在 token_experts 中的位置
                        weight_idx = (token_experts == expert_idx).nonzero().item()
                        weight = routing_weights[token_idx, weight_idx]
                        
                        # 如果這個令牌使用了這個專家
                        if expert_mask[token_idx]:
                            # 找出該令牌在經過過濾後的專家輸入中的索引
                            filtered_idx = expert_mask[:token_idx].sum().item()
                            # 將加權的專家輸出添加到最終輸出
                            final_output[token_idx] += weight * expert_output[filtered_idx]
        
        # 將輸出重塑回原始形狀
        output = final_output.reshape(batch_size, seq_len, d_model)
        
        return output


# 用於可視化的簡化版 MoE
class SimplifiedMoE(nn.Module):
    """
    簡化版 MoE，用於更容易觀察路由決策和數據流
    """
    def __init__(self, d_model=4, num_experts=3, k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        
        # 創建簡化的專家，每個只是一個帶權重的線性轉換
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_experts)
        ])
        
        # 在初始化時使用不同的權重以便觀察
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                # 設置每個專家的權重為數值 i+1 (使它們有明顯區別)
                expert.weight.fill_(float(i + 1))
        
        # 簡化的路由器也是線性層
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 初始化路由器權重，使其易於觀察
        with torch.no_grad():
            self.router.weight.copy_(torch.eye(num_experts, d_model))
        
        self.routing_weights = None
    
    def forward(self, x, print_details=True):
        batch_size, seq_len, d_model = x.shape
        
        # 打印輸入
        if print_details:
            print(f"\n===== 輸入數據 =====")
            print(f"形狀: {x.shape}")
            print(f"數值:\n{x.detach().numpy().round(2)}")
        
        # 將輸入重塑為二維以簡化處理
        x_flat = x.reshape(-1, d_model)
        
        # 計算路由分數
        router_logits = self.router(x_flat)
        
        if print_details:
            print(f"\n===== 路由器邏輯 =====")
            print(f"形狀: {router_logits.shape}")
            print(f"數值:\n{router_logits.detach().numpy().round(2)}")
        
        # 應用 softmax 得到路由權重
        routing_weights = F.softmax(router_logits, dim=1)
        self.routing_weights = routing_weights.detach()
        
        if print_details:
            print(f"\n===== 路由權重 (Softmax後) =====")
            print(f"形狀: {routing_weights.shape}")
            print(f"數值:\n{routing_weights.detach().numpy().round(4)}")
        
        # 選擇 top-k 專家
        top_k_weights, indices = torch.topk(routing_weights, self.k, dim=1)
        
        # 重新正規化 top-k 權重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        
        if print_details:
            print(f"\n===== Top-{self.k} 路由權重 =====")
            print(f"形狀: {top_k_weights.shape}")
            print(f"數值:\n{top_k_weights.detach().numpy().round(4)}")
            print(f"\n===== Top-{self.k} 專家索引 =====")
            print(f"形狀: {indices.shape}")
            print(f"數值:\n{indices.detach().numpy()}")
        
        # 初始化輸出張量
        final_output = torch.zeros_like(x_flat)
        
        # 透過範例追蹤一個令牌的完整過程
        if print_details:
            token_idx = 0  # 追蹤第一個令牌
            print(f"\n===== 追蹤令牌 #{token_idx} 的處理過程 =====")
            print(f"輸入: {x_flat[token_idx].detach().numpy().round(2)}")
            print(f"選擇的專家: {indices[token_idx].detach().numpy()}")
            print(f"專家權重: {top_k_weights[token_idx].detach().numpy().round(4)}")
        
        # 對每個專家計算其貢獻
        for expert_idx in range(self.num_experts):
            # 找出使用當前專家的所有位置
            expert_mask = (indices == expert_idx).any(dim=1)
            
            if expert_mask.any():
                # 提取需要此專家的令牌
                expert_inputs = x_flat[expert_mask]
                
                # 應用專家計算
                expert_output = self.experts[expert_idx](expert_inputs)
                
                if print_details and expert_idx in indices[token_idx]:
                    idx_in_top_k = (indices[token_idx] == expert_idx).nonzero().item()
                    weight = top_k_weights[token_idx][idx_in_top_k]
                    print(f"\n專家 {expert_idx} 處理:")
                    print(f"輸入: {x_flat[token_idx].detach().numpy().round(2)}")
                    print(f"輸出: {self.experts[expert_idx](x_flat[token_idx:token_idx+1]).detach().numpy().round(2).flatten()}")
                    print(f"權重: {weight.item():.4f}")
                    print(f"加權輸出: {(weight * self.experts[expert_idx](x_flat[token_idx:token_idx+1])).detach().numpy().round(4).flatten()}")
                
                # 以更複雜但更正確的方式處理權重和貢獻
                for i, (mask, idx_list) in enumerate(zip(expert_mask, indices)):
                    if not mask:
                        continue
                    
                    # 找出此專家在當前令牌的 top-k 列表中的位置
                    expert_positions = (idx_list == expert_idx).nonzero()
                    if len(expert_positions) == 0:
                        continue
                        
                    expert_pos = expert_positions.item()
                    weight = top_k_weights[i, expert_pos]
                    
                    # 找出該令牌在過濾後的專家輸入中的索引
                    filtered_idx = expert_mask[:i].sum().item()
                    
                    # 將加權的專家輸出添加到最終輸出
                    final_output[i] += weight * expert_output[filtered_idx]
        
        # 將輸出重塑回原始形狀
        output = final_output.reshape(batch_size, seq_len, d_model)
        
        if print_details:
            print(f"\n===== 最終輸出 =====")
            print(f"形狀: {output.shape}")
            print(f"數值:\n{output.detach().numpy().round(4)}")
            print(f"\n===== 追蹤令牌 #{token_idx} 的最終輸出 =====")
            print(f"最終輸出: {output[0, 0].detach().numpy().round(4)}")
        
        return output


# 運行示範
def demo_moe():
    print("=" * 50)
    print("簡化版 MoE 演示")
    print("=" * 50)
    
    # 創建簡化版 MoE 模型
    moe = SimplifiedMoE(d_model=4, num_experts=3, k=2)
    
    # 創建一些測試數據
    # 使用特意設計的輸入以便觀察路由決策
    x = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0],  # 應該主要路由到專家 0
            [0.0, 1.0, 0.0, 0.0],  # 應該主要路由到專家 1
            [0.0, 0.0, 1.0, 0.0]   # 應該主要路由到專家 2
        ]
    ])
    
    # 運行模型並打印詳細過程
    output = moe(x, print_details=True)
    
    print("\n" + "=" * 50)
    print("觀察路由決策分佈")
    print("=" * 50)
    
    # 顯示整體路由決策分佈
    routes = moe.routing_weights.numpy()
    for i in range(3):
        print(f"令牌 {i} 路由分佈: {routes[i].round(4)}")
    
    print("\n路由決策熱度圖:")
    for i in range(3):
        route_str = ""
        for j in range(3):
            intensity = int(routes[i, j] * 10)
            route_str += "█" * intensity + "░" * (10 - intensity) + f" {routes[i, j]:.2f} "
        print(f"令牌 {i}: {route_str}")


if __name__ == "__main__":
    demo_moe()