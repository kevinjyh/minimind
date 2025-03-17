import pytest
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from pathlib import Path
import time
import platform

# 修正：添加專案根目錄到系統路徑
root_dir = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, root_dir)  # 使用insert(0)確保優先搜索

from model.model import MOEFeedForward, FeedForward
from model.LMConfig import LMConfig
from model.model import MoEGate

# 設置隨機種子以確保測試的可重現性
torch.manual_seed(42)
np.random.seed(42)

# 創建輸出目錄
OUTPUT_DIR = Path("tests/visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 設置中文字體以確保圖表正確顯示中文
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系統用黑體
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系統
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux系統

plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

class TestMOEFeedForwardVisualization:
    """MOEFeedForward類的可視化測試，幫助理解專家選擇和貢獻"""
    
    @pytest.fixture
    def basic_config(self):
        """基本測試配置"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            hidden_dim=1024,
            use_moe=True,
            num_experts_per_tok=2,
            n_routed_experts=8,  # 使用8個專家以便於觀察模式
            n_shared_experts=True,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    
    def visualize_expert_distribution(self, expert_indices, n_experts, save_path=None):
        """可視化專家選擇分佈"""
        # 在繪圖前添加字體確認
        if not plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        
        counts = torch.bincount(expert_indices.flatten(), minlength=n_experts)
        expert_usage = counts.float() / counts.sum()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_experts), expert_usage.cpu().numpy())
        plt.xlabel('專家索引', fontsize=12)
        plt.ylabel('選擇頻率', fontsize=12)
        plt.title('專家選擇分佈', fontsize=14, pad=20)  # 添加 pad 避免標題被切斷
        plt.xticks(range(n_experts))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(OUTPUT_DIR / save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_expert_weights(self, expert_indices, expert_weights, n_experts, save_path=None):
        """可視化專家權重分佈"""
        # 在繪圖前添加字體確認
        if not plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        
        # 將權重按專家分組
        expert_total_weights = torch.zeros(n_experts, device=expert_weights.device)
        for i in range(n_experts):
            mask = (expert_indices == i)
            if mask.any():
                expert_total_weights[i] = expert_weights[mask].sum()
        
        # 正規化權重
        expert_total_weights = expert_total_weights / expert_total_weights.sum()
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_experts), expert_total_weights.cpu().numpy())
        plt.xlabel('專家索引', fontsize=12)
        plt.ylabel('權重比例', fontsize=12)
        plt.title('專家權重分佈', fontsize=14, pad=20)
        plt.xticks(range(n_experts))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(OUTPUT_DIR / save_path)
            plt.close()
        else:
            plt.show()
    
    def test_visualize_expert_selection(self, basic_config):
        """測試並可視化專家選擇過程"""
        moe_ff = MOEFeedForward(basic_config)
        moe_ff.train()  # 設置為訓練模式
        
        # 創建輸入張量 [batch_size, seq_len, dim]
        batch_size, seq_len = 10, 20  # 使用較大的批次以獲得更好的統計結果
        x = torch.randn(batch_size, seq_len, basic_config.dim)
        
        # 獲取門控機制的輸出
        with torch.no_grad():
            topk_idx, topk_weight, _ = moe_ff.gate(x)
        
        # 可視化專家選擇分佈
        flat_topk_idx = topk_idx.view(-1)
        self.visualize_expert_distribution(
            flat_topk_idx, 
            basic_config.n_routed_experts,
            save_path="moe_expert_selection_distribution.png"
        )
        
        # 可視化專家權重分佈
        flat_topk_weight = topk_weight.view(-1)
        self.visualize_expert_weights(
            flat_topk_idx,
            flat_topk_weight,
            basic_config.n_routed_experts,
            save_path="moe_expert_weight_distribution.png"
        )
    
    def test_visualize_expert_capacity(self, basic_config):
        """測試並可視化專家容量和負載"""
        # 在繪圖前添加字體確認
        if not plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        
        # 創建多個不同配置的MoE模型
        configs = []
        expert_counts = [4, 8, 16]
        experts_per_tok = [1, 2, 4]
        
        results = {}
        
        for n_experts in expert_counts:
            for n_per_tok in experts_per_tok:
                if n_per_tok > n_experts:
                    continue  # 跳過無效配置
                    
                config = LMConfig(
                    dim=512,
                    n_layers=8,
                    n_heads=8,
                    hidden_dim=1024,
                    use_moe=True,
                    num_experts_per_tok=n_per_tok,
                    n_routed_experts=n_experts,
                    n_shared_experts=True,
                    aux_loss_alpha=0.1,
                    seq_aux=True,
                    norm_topk_prob=True
                )
                
                moe_ff = MOEFeedForward(config)
                moe_ff.train()
                
                # 創建輸入張量 [batch_size, seq_len, dim]
                batch_size, seq_len = 10, 20
                x = torch.randn(batch_size, seq_len, config.dim)
                
                # 獲取門控機制的輸出
                with torch.no_grad():
                    topk_idx, _, aux_loss = moe_ff.gate(x)
                
                # 計算每個專家處理的標記數量
                token_counts = torch.bincount(
                    topk_idx.view(-1), 
                    minlength=config.n_routed_experts
                )
                
                # 計算專家負載標準差（負載均衡度量）
                std_load = token_counts.float().std().item()
                max_load = token_counts.max().item()
                min_load = token_counts.min().item()
                
                # 存儲結果
                key = f"E{n_experts}_T{n_per_tok}"
                results[key] = {
                    "std": std_load,
                    "max": max_load,
                    "min": min_load,
                    "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                    "token_counts": token_counts.cpu().numpy()
                }
        
        # 可視化專家負載均衡
        fig, axes = plt.subplots(len(expert_counts), len(experts_per_tok), figsize=(15, 10))
        fig.suptitle('專家負載分佈 (不同專家數量和每個標記選擇的專家數)', fontsize=16, y=1.02)
        
        for i, n_experts in enumerate(expert_counts):
            for j, n_per_tok in enumerate(experts_per_tok):
                if n_per_tok > n_experts:
                    continue  # 跳過無效配置
                    
                key = f"E{n_experts}_T{n_per_tok}"
                if key in results:
                    if len(expert_counts) > 1 and len(experts_per_tok) > 1:
                        ax = axes[i, j]
                    elif len(expert_counts) > 1:
                        ax = axes[i]
                    elif len(experts_per_tok) > 1:
                        ax = axes[j]
                    else:
                        ax = axes
                    
                    token_counts = results[key]["token_counts"]
                    ax.bar(range(len(token_counts)), token_counts)
                    ax.set_title(f'E={n_experts}, T={n_per_tok}\nSTD={results[key]["std"]:.2f}')
                    ax.set_xlabel('專家索引', fontsize=10)
                    ax.set_ylabel('處理標記數', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "moe_expert_load_distribution.png")
        plt.close()
        
        # 輸出結果摘要
        print("專家負載均衡測試結果：")
        for key, data in results.items():
            print(f"{key}: STD={data['std']:.2f}, MAX={data['max']}, MIN={data['min']}, AUX_LOSS={data['aux_loss']:.4f}")
    
    def test_visualize_inference_optimization(self, basic_config):
        """測試並可視化推理優化對性能的影響 (CPU版本)"""
        # 在繪圖前添加字體確認
        if not plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        
        moe_ff = MOEFeedForward(basic_config)
        moe_ff.eval()  # 設置為評估模式
        
        # 創建不同大小的輸入張量測試性能
        batch_sizes = [1, 2, 4, 8]  # 減少批次大小以適應CPU
        seq_lens = [10, 20, 40]     # 減少序列長度以加快測試
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                key = f"B{batch_size}_S{seq_len}"
                results[key] = {"vanilla_time": 0, "optimized_time": 0}
                
                # 創建輸入
                x = torch.randn(batch_size, seq_len, basic_config.dim)
                
                # 使用Python的time模組進行計時
                import time
                
                # 測試使用標準前向傳播的時間
                with torch.no_grad():
                    # 預熱
                    topk_idx, topk_weight, _ = moe_ff.gate(x)
                    output = moe_ff(x)
                    
                    # 計時
                    start_time = time.time()
                    for _ in range(5):  # 減少迭代次數以加快測試
                        topk_idx, topk_weight, _ = moe_ff.gate(x)
                        output = moe_ff(x)
                    end_time = time.time()
                    
                    results[key]["vanilla_time"] = (end_time - start_time) * 1000 / 5  # 轉換為毫秒
                
                # 測試優化的推理方法
                x_flat = x.view(-1, x.shape[-1])
                with torch.no_grad():
                    # 預熱
                    topk_idx, topk_weight, _ = moe_ff.gate(x)
                    flat_topk_idx = topk_idx.view(-1)
                    flat_topk_weight = topk_weight.view(-1, 1)
                    output = moe_ff.moe_infer(x_flat, flat_topk_idx, flat_topk_weight)
                    
                    # 計時
                    start_time = time.time()
                    for _ in range(5):  # 減少迭代次數以加快測試
                        topk_idx, topk_weight, _ = moe_ff.gate(x)
                        flat_topk_idx = topk_idx.view(-1)
                        flat_topk_weight = topk_weight.view(-1, 1)
                        output = moe_ff.moe_infer(x_flat, flat_topk_idx, flat_topk_weight)
                    end_time = time.time()
                    
                    results[key]["optimized_time"] = (end_time - start_time) * 1000 / 5  # 轉換為毫秒
        
        # 可視化結果
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(results))
        width = 0.35
        
        vanilla_times = [results[key]["vanilla_time"] for key in results]
        optimized_times = [results[key]["optimized_time"] for key in results]
        
        rects1 = ax.bar(x - width/2, vanilla_times, width, label='標準推理')
        rects2 = ax.bar(x + width/2, optimized_times, width, label='優化推理')
        
        ax.set_ylabel('執行時間 (ms)', fontsize=12)
        ax.set_title('不同批次大小和序列長度的推理性能比較 (CPU)', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(list(results.keys()))
        ax.legend()
        
        # 添加數值標籤
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 點垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        add_labels(rects1)
        add_labels(rects2)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "moe_inference_optimization_comparison.png")
        plt.close()
        
        # 計算加速比並創建單獨的圖表
        speedups = []
        labels = []
        for key, data in results.items():
            speedup = data["vanilla_time"] / data["optimized_time"] if data["optimized_time"] > 0 else 1.0
            speedups.append(speedup)
            labels.append(key)
            print(f"{key}: 標準={data['vanilla_time']:.2f}ms, 優化={data['optimized_time']:.2f}ms, 加速比={speedup:.2f}x")
        
        # 繪製加速比圖表
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(labels, speedups, color='green')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)  # 添加基準線
        ax.set_ylabel('加速比 (標準時間/優化時間)', fontsize=12)
        ax.set_title('MoE推理優化加速比', fontsize=14, pad=20)
        ax.set_ylim(bottom=0)  # 確保y軸從0開始
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 點垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "moe_inference_speedup.png")
        plt.close() 