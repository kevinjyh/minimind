import os
import sys
import pytest
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from unittest.mock import patch, MagicMock

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from train_lora import get_lr


class TestLearningRate:
    """測試學習率調整相關功能"""
    
    def setup_method(self):
        """測試前設置"""
        # 學習率相關參數
        self.base_lr = 5e-5  # LoRA 訓練的學習率通常較小
        self.total_steps = 1000
        
        # 確保輸出目錄存在
        self.output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 設置中文字體
        self.set_chinese_font()
    
    def set_chinese_font(self):
        """設置支援繁體中文的字體"""
        # 根據不同作業系統設置字體
        system = sys.platform
        if system == 'win32':
            # Windows 系統常見中文字體
            fonts = ['Microsoft JhengHei', 'DFKai-SB', 'MingLiU', 'SimHei', 'KaiTi']
        elif system == 'darwin':
            # macOS 系統常見中文字體
            fonts = ['PingFang TC', 'STHeiti', 'Heiti TC', 'Apple LiGothic']
        else:
            # Linux 系統常見中文字體
            fonts = ['WenQuanYi Zen Hei', 'Noto Sans CJK TC', 'Noto Sans TC']
        
        # 嘗試設置第一個可用的字體
        font_found = False
        for font in fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
                # 驗證字體是否可用
                test_font = FontProperties(family=font)
                if test_font.get_name() != 'sans-serif':
                    font_found = True
                    self.font = font
                    break
            except:
                continue
        
        # 若未找到合適字體，使用默認設置並發出警告
        if not font_found:
            plt.rcParams['font.sans-serif'] = ['sans-serif']
            self.font = 'sans-serif'
            print("警告: 未找到支援繁體中文的字體，將使用默認字體")
    
    def test_get_lr_function(self):
        """測試 get_lr 函數"""
        # 測試學習率計算是否符合預期
        lr_start = get_lr(0, self.total_steps, self.base_lr)
        lr_middle = get_lr(self.total_steps // 2, self.total_steps, self.base_lr)
        lr_end = get_lr(self.total_steps - 1, self.total_steps, self.base_lr)
        
        # 檢查起始學習率
        assert lr_start > lr_end, "起始學習率應大於結束學習率"
        # 檢查學習率範圍，擴大容忍範圍
        assert lr_end == pytest.approx(self.base_lr / 10, rel=1e-4), "結束學習率應等於基礎學習率的十分之一"
        assert lr_start > lr_end, "起始學習率應大於結束學習率"
        
        # 將結果寫入檔案
        with open(os.path.join(self.output_dir, "lr_test_results.txt"), "w", encoding="utf-8") as f:
            f.write(f"基礎學習率: {self.base_lr}\n")
            f.write(f"總步數: {self.total_steps}\n")
            f.write(f"起始學習率 (step=0): {lr_start}\n")
            f.write(f"中間學習率 (step={self.total_steps//2}): {lr_middle}\n")
            f.write(f"結束學習率 (step={self.total_steps-1}): {lr_end}\n")
    
    def test_lr_schedule_visualization(self):
        """視覺化學習率調度"""
        # 生成學習率曲線數據
        steps = np.arange(0, self.total_steps)
        lr_values = [get_lr(step, self.total_steps, self.base_lr) for step in steps]
        
        # 繪製學習率曲線圖
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lr_values)
        plt.title('LoRA 學習率調度曲線', fontproperties=FontProperties(family=self.font))
        plt.xlabel('訓練步數', fontproperties=FontProperties(family=self.font))
        plt.ylabel('學習率', fontproperties=FontProperties(family=self.font))
        plt.grid(True)
        plt.tight_layout()
        
        # 保存圖表
        plt.savefig(os.path.join(self.output_dir, "lr_schedule_curve.png"), dpi=300)
        plt.close()
    
    @pytest.mark.parametrize("base_lr", [1e-4, 5e-5, 1e-5])
    def test_lr_with_different_base_rates(self, base_lr):
        """測試不同基礎學習率下的調度效果 (較小的學習率適合 LoRA)"""
        # 生成學習率曲線數據
        steps = np.arange(0, self.total_steps, 10)  # 減少點數以加快測試
        lr_values = [get_lr(step, self.total_steps, base_lr) for step in steps]
        
        # 記錄最大和最小學習率
        max_lr = max(lr_values)
        min_lr = min(lr_values)
        
        # 將結果寫入檔案
        with open(os.path.join(self.output_dir, f"lr_test_base_{base_lr}.txt"), "w", encoding="utf-8") as f:
            f.write(f"基礎學習率: {base_lr}\n")
            f.write(f"最大學習率: {max_lr}\n")
            f.write(f"最小學習率: {min_lr}\n")
            f.write(f"學習率比例 (max/min): {max_lr/min_lr if min_lr > 0 else 'N/A'}\n")
    
    def test_compare_different_schedules(self):
        """比較不同學習率調度策略對 LoRA 訓練的影響"""
        # 定義不同的學習率調度函數
        def cosine_schedule(step, total_steps, lr):
            """余弦退火學習率調度 (與 get_lr 相同)"""
            return lr / 10 + 0.5 * lr * (1 + np.cos(np.pi * step / total_steps))
        
        def linear_schedule(step, total_steps, lr):
            """線性學習率調度"""
            return lr * (1 - step / total_steps) + lr / 10 * (step / total_steps)
        
        def step_schedule(step, total_steps, lr):
            """階梯式學習率調度"""
            if step < total_steps * 0.3:
                return lr
            elif step < total_steps * 0.6:
                return lr * 0.1
            else:
                return lr * 0.01
        
        # 生成學習率曲線數據
        steps = np.arange(0, self.total_steps)
        cosine_values = [cosine_schedule(step, self.total_steps, self.base_lr) for step in steps]
        linear_values = [linear_schedule(step, self.total_steps, self.base_lr) for step in steps]
        step_values = [step_schedule(step, self.total_steps, self.base_lr) for step in steps]
        
        # 繪製比較圖
        plt.figure(figsize=(12, 8))
        plt.plot(steps, cosine_values, label='余弦退火', linewidth=2)
        plt.plot(steps, linear_values, label='線性衰減', linewidth=2)
        plt.plot(steps, step_values, label='階梯式衰減', linewidth=2)
        
        plt.title('LoRA 訓練不同學習率調度策略比較', fontproperties=FontProperties(family=self.font))
        plt.xlabel('訓練步數', fontproperties=FontProperties(family=self.font))
        plt.ylabel('學習率', fontproperties=FontProperties(family=self.font))
        plt.legend(prop=FontProperties(family=self.font))
        plt.grid(True)
        plt.tight_layout()
        
        # 保存圖表
        plt.savefig(os.path.join(self.output_dir, "lr_schedule_comparison.png"), dpi=300)
        plt.close()
    
    def test_lora_specific_learning_rates(self):
        """測試 LoRA 特定的學習率值範圍效果"""
        # LoRA 常用的學習率範圍
        lora_learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
        
        # 為每個學習率生成調度曲線
        plt.figure(figsize=(12, 8))
        
        for lr in lora_learning_rates:
            steps = np.arange(0, self.total_steps)
            lr_values = [get_lr(step, self.total_steps, lr) for step in steps]
            plt.plot(steps, lr_values, label=f'基礎 LR = {lr}', linewidth=2)
        
        plt.title('LoRA 不同基礎學習率的調度曲線', fontproperties=FontProperties(family=self.font))
        plt.xlabel('訓練步數', fontproperties=FontProperties(family=self.font))
        plt.ylabel('學習率', fontproperties=FontProperties(family=self.font))
        plt.legend(prop=FontProperties(family=self.font))
        plt.grid(True)
        plt.yscale('log')  # 使用對數尺度以便觀察較小的學習率
        plt.tight_layout()
        
        # 保存圖表
        plt.savefig(os.path.join(self.output_dir, "lora_lr_comparison.png"), dpi=300)
        plt.close()
        
        # 記錄各學習率的最大與最小值
        with open(os.path.join(self.output_dir, "lora_learning_rates.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 特定學習率範圍分析:\n\n")
            for lr in lora_learning_rates:
                max_lr = get_lr(0, self.total_steps, lr)
                min_lr = get_lr(self.total_steps-1, self.total_steps, lr)
                f.write(f"基礎學習率: {lr}\n")
                f.write(f"最大學習率: {max_lr}\n")
                f.write(f"最小學習率: {min_lr}\n")
                f.write(f"學習率比例 (max/min): {max_lr/min_lr}\n")
                f.write("\n") 