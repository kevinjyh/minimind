import matplotlib
matplotlib.use('Agg')  # 使用無窗口後端

import pytest
import torch
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# 確保優先搜索項目根目錄
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from model.model_lora import LoRA, apply_lora, load_lora, save_lora


class TestLoRAVisualization:
    """測試 LoRA 的可視化效果"""
    
    def setup_method(self):
        """設置測試環境，確保輸出目錄存在"""
        self.output_dir = Path(__file__).parent / "visualization_output"
        self.output_dir.mkdir(exist_ok=True)
        
        # 設置 matplotlib 支援中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑體字
        plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
    
    def test_visualize_weight_transformation(self):
        """視覺化 LoRA 對權重的轉換效果"""
        # 創建一個簡單的線性層
        original_layer = torch.nn.Linear(20, 20)
        
        # 記錄原始權重
        original_weight = original_layer.weight.data.clone()
        
        # 創建 LoRA 層
        lora = LoRA(20, 20, rank=4)
        
        # 設置一些非零值到 B 矩陣以查看效果
        lora.B.weight.data = torch.randn_like(lora.B.weight.data) * 0.1
        
        # 模擬 LoRA 添加的結果
        input_data = torch.eye(20)  # 單位矩陣作為輸入
        lora_output = lora(input_data)
        combined_weight = original_weight + lora_output.detach()
        
        # 將權重轉換為 numpy 以便可視化
        original_weight_np = original_weight.numpy()
        lora_contribution_np = lora_output.detach().numpy()
        combined_weight_np = combined_weight.numpy()
        
        # 創建可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(original_weight_np, cmap='coolwarm')
        axes[0].set_title('原始權重')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(lora_contribution_np, cmap='coolwarm')
        axes[1].set_title('LoRA 貢獻 (BA)')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(combined_weight_np, cmap='coolwarm')
        axes[2].set_title('組合權重 (W + BA)')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "weight_transformation.png"))
        plt.close()
    
    def test_visualize_rank_effect(self):
        """視覺化不同秩值對 LoRA 效果的影響"""
        # 創建原始層
        original_layer = torch.nn.Linear(20, 20)
        original_weight = original_layer.weight.data.clone()
        
        # 不同的秩值
        ranks = [1, 2, 4, 8, 16]
        
        # 創建圖形
        fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(5 * (len(ranks) + 1), 5))
        
        # 顯示原始權重
        im0 = axes[0].imshow(original_weight.numpy(), cmap='coolwarm')
        axes[0].set_title('原始權重')
        plt.colorbar(im0, ax=axes[0])
        
        # 對每個秩值，創建 LoRA 並顯示組合權重
        for i, rank in enumerate(ranks):
            # 創建 LoRA 層
            lora = LoRA(20, 20, rank=rank)
            
            # 設置一些非零值以查看效果
            lora.A.weight.data = torch.randn_like(lora.A.weight.data) * 0.1
            lora.B.weight.data = torch.randn_like(lora.B.weight.data) * 0.1
            
            # 計算 LoRA 貢獻
            input_data = torch.eye(20)
            lora_output = lora(input_data)
            
            # 組合權重
            combined_weight = original_weight + lora_output.detach()
            
            # 顯示
            im = axes[i+1].imshow(combined_weight.numpy(), cmap='coolwarm')
            axes[i+1].set_title(f'秩={rank}')
            plt.colorbar(im, ax=axes[i+1])
        
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "rank_effect.png"))
        plt.close()
    
    def test_visualize_training_dynamics(self):
        """視覺化 LoRA 訓練動態"""
        # 創建一個簡單的回歸問題
        X = torch.linspace(-3, 3, 100).reshape(-1, 1)
        y_true = torch.sin(X) + 0.1 * torch.randn_like(X)
        
        # 創建一個簡單的模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(1, 20)
                self.act = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(20, 20)
                self.output = torch.nn.Linear(20, 1)
                self.device = "cpu"
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.act(x)
                x = self.linear2(x)
                x = self.act(x)
                return self.output(x)
        
        # 創建和訓練預訓練模型
        model = SimpleModel()
        
        # 簡單預訓練
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        for _ in range(500):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
        
        # 獲取預訓練模型預測
        with torch.no_grad():
            pretrained_preds = model(X).numpy()
        
        # 應用 LoRA 並進行微調
        model_lora = SimpleModel()
        model_lora.load_state_dict(model.state_dict())
        apply_lora(model_lora, rank=4)
        
        # 凍結原始參數
        for param in model_lora.parameters():
            param.requires_grad = False
            
        # 啟用 LoRA 參數的梯度
        for name, module in model_lora.named_modules():
            if hasattr(module, 'lora'):
                for param in module.lora.parameters():
                    param.requires_grad = True
        
        # 新數據集（領域轉換）
        y_new = torch.cos(X) + 0.1 * torch.randn_like(X)
        
        # LoRA 微調
        lora_optimizer = torch.optim.Adam([p for p in model_lora.parameters() if p.requires_grad], lr=0.01)
        lora_preds_history = []
        
        # 記錄不同訓練階段的預測
        epochs = [0, 10, 25, 50, 100, 200]
        current_epoch = 0
        
        for epoch in range(max(epochs) + 1):
            lora_optimizer.zero_grad()
            y_lora_pred = model_lora(X)
            loss = criterion(y_lora_pred, y_new)
            loss.backward()
            lora_optimizer.step()
            
            if epoch in epochs:
                with torch.no_grad():
                    lora_preds_history.append(model_lora(X).numpy())
                    current_epoch = epoch
        
        # 可視化訓練動態
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 使用色弱友好的顏色方案和不同的標記
        # 原始數據
        ax.scatter(X.numpy(), y_true.numpy(), s=20, color='#0072B2', marker='o', label='原始數據 (sin)')
        ax.scatter(X.numpy(), y_new.numpy(), s=20, color='#D55E00', marker='s', label='新數據 (cos)')
        
        # 預訓練模型預測 - 使用黑色實線而非綠色
        ax.plot(X.numpy(), pretrained_preds, linewidth=3, color='#000000', linestyle='-', label='預訓練模型')
        
        # LoRA 不同階段預測 - 使用不同線型和標記的組合
        # 色弱友好的顏色方案
        colors = ['#0072B2', '#56B4E9', '#CC79A7', '#E69F00', '#F0E442', '#009E73']
        linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1))]
        markers = ['o', 's', '^', 'v', 'D', 'x']
        
        for i, (epoch, preds) in enumerate(zip(epochs, lora_preds_history)):
            ax.plot(X.numpy(), preds, 
                    linewidth=2, 
                    color=colors[i], 
                    linestyle=linestyles[i], 
                    marker=markers[i],
                    markevery=20, # 每20個點標記一次，避免太擁擠
                    markersize=8,
                    label=f'LoRA 訓練 {epoch} 輪')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('LoRA 訓練動態 - 領域適應', fontsize=14)
        
        # 增加圖例的可讀性
        ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "training_dynamics.png"), dpi=300)
        plt.close()
    
    def test_visualize_parameter_efficiency(self):
        """視覺化 LoRA 的參數效率"""
        # 不同的模型尺寸和對應的 LoRA 秩值
        model_sizes = [64, 128, 256, 512, 1024]
        lora_ranks = [4, 8, 16, 32]
        
        # 計算參數數量
        full_params = []
        lora_params_by_rank = {rank: [] for rank in lora_ranks}
        
        for size in model_sizes:
            # 全參數數量（方形矩陣）
            full_params.append(size * size)
            
            # 不同 LoRA 秩的參數數量
            for rank in lora_ranks:
                # LoRA 參數數量 = (輸入特徵 x 秩) + (秩 x 輸出特徵)
                lora_params_by_rank[rank].append(size * rank + rank * size)
        
        # 計算參數減少比例
        reduction_ratios = {rank: [] for rank in lora_ranks}
        for i, full in enumerate(full_params):
            for rank in lora_ranks:
                reduction_ratios[rank].append(full / lora_params_by_rank[rank][i])
        
        # 繪製圖形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 參數數量圖
        ax1.plot(model_sizes, full_params, 'o-', linewidth=2, markersize=8, label='全參數')
        for rank in lora_ranks:
            ax1.plot(model_sizes, lora_params_by_rank[rank], 'o-', linewidth=2, markersize=8, label=f'LoRA 秩={rank}')
        
        ax1.set_xlabel('模型尺寸')
        ax1.set_ylabel('參數數量')
        ax1.set_title('參數數量隨模型尺寸變化')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        
        # 參數減少比例圖
        for rank in lora_ranks:
            ax2.plot(model_sizes, reduction_ratios[rank], 'o-', linewidth=2, markersize=8, label=f'LoRA 秩={rank}')
        
        ax2.set_xlabel('模型尺寸')
        ax2.set_ylabel('參數減少比例 (全參數/LoRA)')
        ax2.set_title('參數效率隨模型尺寸變化')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "parameter_efficiency.png"))
        plt.close()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 