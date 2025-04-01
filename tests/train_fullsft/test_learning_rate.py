import os
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_full_sft import get_lr, Logger

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def test_learning_rate_calculation():
    """測試學習率計算函數"""
    # 測試參數
    total_steps = 1000
    base_lr = 1e-4
    
    # 計算不同階段的學習率
    lr_values = []
    for step in range(total_steps):
        lr = get_lr(step, total_steps, base_lr)
        lr_values.append(lr)
    
    # 驗證學習率範圍
    assert min(lr_values) >= base_lr / 10
    assert max(lr_values) <= base_lr + (base_lr / 10)
    
    # 驗證學習率變化趨勢
    assert lr_values[0] > lr_values[total_steps//2]  # 開始時較高
    assert lr_values[total_steps//2] > lr_values[-1]  # 結束時較低

def test_learning_rate_visualization(tmp_path):
    """測試學習率變化視覺化"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 計算學習率變化
    steps = np.arange(total_steps)
    lr_values = [get_lr(step, total_steps, base_lr) for step in steps]
    
    # 繪製學習率變化圖
    plt.figure(figsize=(12, 6))
    plt.plot(steps, lr_values, 'b-', label='學習率')
    plt.title('學習率變化趨勢')
    plt.xlabel('訓練步數')
    plt.ylabel('學習率')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(tmp_path, 'learning_rate_trend.png'))
    plt.close()

def test_learning_rate_schedule():
    """測試學習率調度"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 計算關鍵點的學習率
    lr_start = get_lr(0, total_steps, base_lr)
    lr_mid = get_lr(total_steps//2, total_steps, base_lr)
    lr_end = get_lr(total_steps-1, total_steps, base_lr)
    
    # 驗證學習率調度特性
    assert lr_start > lr_mid > lr_end  # 單調遞減
    assert abs(lr_start - base_lr) < 1e-5  # 起始值接近基礎學習率
    assert abs(lr_end - base_lr/10) < 1e-6  # 結束值接近基礎學習率的1/10

def test_learning_rate_consistency():
    """測試學習率計算的一致性"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 多次計算相同步數的學習率
    step = 500
    lr_values = [get_lr(step, total_steps, base_lr) for _ in range(10)]
    
    # 驗證一致性
    assert all(abs(lr - lr_values[0]) < 1e-10 for lr in lr_values)

def test_learning_rate_boundaries():
    """測試學習率邊界條件"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 測試邊界情況
    lr_start = get_lr(0, total_steps, base_lr)
    lr_end = get_lr(total_steps-1, total_steps, base_lr)
    
    # 驗證邊界值
    assert abs(lr_start - base_lr) < 1e-5
    assert abs(lr_end - base_lr/10) < 1e-6

def test_learning_rate_smoothness(tmp_path):
    """測試學習率變化的平滑性"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 計算學習率變化
    lr_values = [get_lr(step, total_steps, base_lr) for step in range(total_steps)]
    
    # 計算相鄰點之間的變化
    lr_changes = np.diff(lr_values)
    
    # 繪製學習率變化率圖
    plt.figure(figsize=(12, 6))
    plt.plot(lr_changes, 'r-', label='學習率變化率')
    plt.title('學習率變化率')
    plt.xlabel('訓練步數')
    plt.ylabel('學習率變化')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(tmp_path, 'learning_rate_smoothness.png'))
    plt.close()
    
    # 驗證變化的平滑性
    max_change = np.max(np.abs(lr_changes))
    assert max_change < base_lr / 100  # 確保變化不會太劇烈

def test_learning_rate_optimization():
    """測試學習率優化效果"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 模擬訓練過程中的學習率變化
    lr_values = []
    for step in range(total_steps):
        lr = get_lr(step, total_steps, base_lr)
        lr_values.append(lr)
    
    # 計算學習率的統計特性
    lr_mean = np.mean(lr_values)
    lr_std = np.std(lr_values)
    
    # 驗證學習率分佈
    assert lr_mean < base_lr  # 平均學習率應該小於基礎學習率
    assert lr_std < base_lr/2  # 標準差不應該太大

def test_learning_rate_epoch_transition():
    """測試學習率在epoch之間的過渡"""
    total_steps = 1000
    base_lr = 1e-4
    
    # 模擬多個epoch的學習率變化
    epochs = 3
    all_lr_values = []
    
    for epoch in range(epochs):
        epoch_lr_values = []
        for step in range(total_steps):
            lr = get_lr(step, total_steps, base_lr)
            epoch_lr_values.append(lr)
        all_lr_values.extend(epoch_lr_values)
    
    # 驗證epoch之間的過渡
    for i in range(1, epochs):
        epoch_start_idx = i * total_steps
        assert abs(all_lr_values[epoch_start_idx] - base_lr) < 1e-5  # 每個epoch開始時應該重置 