# LoRA（Low-Rank Adaptation）測試指南

本文檔旨在幫助您通過測試案例理解 LoRA 的工作原理和實現細節。LoRA 是一種參數高效的模型微調技術，通過添加低秩矩陣來調整預訓練模型的權重，而無需更新所有原始參數。

## [LoRA的工作原理](../../learning-notes/topics/LoRA的工作原理.md)

## 測試案例概述

測試被分為兩個主要文件：

- `test_lora.py`：測試基本結構和功能
- `test_lora_training.py`：測試 LoRA 在訓練場景中的應用

### test_lora.py 中的測試案例

#### TestLoRA 類

1. **test_lora_initialization**：
   - 測試 LoRA 層的初始化是否正確
   - 驗證矩陣 A 和 B 的尺寸是否符合預期
   - 檢查初始化方式（A 使用正態分佈，B 為零）

2. **test_lora_forward**：
   - 測試 LoRA 的前向傳播
   - 確認輸出形狀是否正確
   - 驗證計算路徑是否正確遵循 A->B 的順序

3. **test_different_ranks**：
   - 測試不同秩值對 LoRA 的影響
   - 驗證不同秩值下矩陣尺寸的變化

#### TestApplyLoRA 類

1. **test_apply_lora**：
   - 測試 apply_lora 函數是否正確應用到模型上
   - 檢查是否只有方形矩陣的線性層被應用了 LoRA
   - 驗證前向傳播函數是否被正確修改

2. **test_lora_forward_with_model**：
   - 測試將 LoRA 應用到模型後的前向傳播
   - 由於 B 矩陣初始化為零，確認初始輸出應與原始模型相同

3. test_apply_lora_to_small_model 測試
   - 對小型 MiniMindLM 模型應用 LoRA，秩為 8
   - 首先識別並計數模型中所有符合條件的方陣線性層
   - 檢查 LoRA 應用後的情況：
      - 確認所有方陣線性層都應用了 LoRA
      - 驗證 LoRA 的秩設置正確
      - 確保前向傳播方法已被修改
   - 最後執行模型的前向傳播，確保模型仍然能正常工作

4. test_apply_lora_to_moe_model 測試
   - 對帶有 MoE (Mixture of Experts) 的 MiniMindLM 模型應用 LoRA，秩為 16
   - 與第一個測試類似，但專門針對 MoE 結構的模型
   - 驗證在複雜的 MoE 架構中，LoRA 仍能正確應用於所有符合條件的線性層
   - 確認應用 LoRA 後，MoE 模型的前向傳播功能仍然正常

這兩個測試將幫助您確認 apply_lora 函數能夠正確地疊加低秩層於不同配置的 MiniMindLM 模型中的 nn.Linear 層，並且不會破壞模型的基本功能。
這些測試案例會：

   1. 識別和統計模型中所有符合條件的方陣線性層
   2. 確認這些層在應用 LoRA 後被正確修改
   3. 驗證模型在應用 LoRA 後仍能正常運行

#### TestSaveLoadLoRA 類

1. **test_save_load_lora**：
   - 測試 LoRA 權重的保存和加載功能
   - 驗證權重是否能夠正確保存和恢復

2. **test_partial_lora_loading**：
   - 測試在有多個 LoRA 層的情況下，權重的保存和加載
   - 驗證不同層之間的權重是否能夠正確對應

### test_lora_training.py 中的測試案例

#### TestLoRATraining 類

1. **test_lora_parameter_count**：
   - 比較 LoRA 參數量與原始模型參數量
   - 驗證 LoRA 是否確實降低了需要訓練的參數數量

2. **test_lora_training**：
   - 測試僅訓練 LoRA 參數而凍結原始網絡
   - 確認原始參數在訓練後保持不變
   - 驗證 LoRA 參數是否有適當更新

3. **test_fine_tuning_effect**：
   - 比較 LoRA 微調與全參數微調的效果
   - 評估參數效率和性能之間的權衡
   - 分析 LoRA 在領域適應任務中的效果

## 如何調整測試參數深入理解 LoRA

### 1. 探索秩值的影響

秩值是 LoRA 的核心參數，決定了低秩矩陣的大小：

```python
# 在 test_different_ranks 中嘗試不同的秩值
ranks = [1, 2, 4, 8, 16, 32, 64]
```

較低的秩值意味著更少的參數，但可能限制表達能力；較高的秩值則相反。通過調整秩值並觀察性能變化，可以理解 LoRA 中秩值選擇的重要性。

### 2. 調整初始化參數

嘗試不同的初始化方法和參數：

```python
# 在 LoRA 類的 __init__ 方法中修改
self.A.weight.data.normal_(mean=0.0, std=0.1)  # 嘗試不同標準差
self.B.weight.data.normal_(mean=0.0, std=0.01)  # 嘗試非零初始化 B
```

這將幫助您理解初始化對 LoRA 收斂速度和穩定性的影響。

### 3. 嘗試不同的層選擇策略

默認情況下，`apply_lora` 僅應用於方形矩陣的線性層，但您可以嘗試修改選擇策略：

```python
# 修改 apply_lora 函數中的條件
if isinstance(module, nn.Linear):  # 應用於所有線性層
```

這可以幫助理解為什麼 LoRA 通常只應用於特定層，以及不同選擇策略的效果。

### 4. 探索不同的學習率和優化器

在 `test_lora_training` 中嘗試不同的優化設置：

```python
# 嘗試不同的優化器
optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if p.requires_grad], lr=0.01)
```

```python
# 為 LoRA 參數使用較高的學習率
lora_params = [p for name, p in model.named_parameters() if 'lora' in name]
optimizer = torch.optim.SGD([{'params': lora_params, 'lr': 0.1}], lr=0.01)
```
