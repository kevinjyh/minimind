# MOEFeedForward 訓練模式解說

## 基本概念

`MOEFeedForward` 是混合專家系統（Mixture of Experts, MoE）中的一個重要組件，它負責將輸入數據分發給多個專家進行處理，並將專家的輸出進行加權組合。

## 訓練模式下的代碼操作

### 1. 輸入數據重複

```python
x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
```

這行代碼的作用是將輸入數據重複，使得每個 token 可以被多個專家處理。具體來說：

- `self.config.num_experts_per_tok` 表示每個 token 選擇的專家數量
- `repeat_interleave` 在批次維度（dim=0）上重複數據
- 例如，如果 `num_experts_per_tok = 2`，則每個 token 會被重複兩次

### 2. 專家輸出收集

```python
y = torch.empty_like(x, dtype=torch.float16)
for i, expert in enumerate(self.experts):
    y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
```

這部分代碼使用布爾索引來收集各個專家的輸出：

- 創建一個與輸入相同形狀的空張量 `y`，使用 float16 數據類型
- 遍歷每個專家，使用布爾索引 `flat_topk_idx == i` 選擇對應的輸入數據
- 將專家的輸出存儲在 `y` 的對應位置

### 3. 加權組合專家輸出

```python
y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
```

這行代碼實現了專家輸出的加權組合，包含三個主要步驟：

1. **重塑張量 `y`**：
   ```python
   y.view(*topk_weight.shape, -1)
   ```
   - 將 `y` 張量重塑為與 `topk_weight` 相同的形狀，但保留最後一個維度
   - 如果 `topk_weight` 形狀為 `(batch_size, seq_len, num_experts_per_tok)`，則 `y` 會被重塑為 `(batch_size, seq_len, num_experts_per_tok, hidden_dim)`

2. **擴展權重維度**：
   ```python
   topk_weight.unsqueeze(-1)
   ```
   - 在 `topk_weight` 的最後添加一個維度
   - 從 `(batch_size, seq_len, num_experts_per_tok)` 變為 `(batch_size, seq_len, num_experts_per_tok, 1)`

3. **乘法和求和**：
   - 首先，`y_reshaped` 和 `weights_expanded` 進行元素級乘法。由於 `y_reshaped` 的形狀是 `(2, 2, 3)`，而 `weights_expanded` 的形狀是 `(2, 2, 1)`，PyTorch 會自動廣播 `weights_expanded` 以匹配 `y_reshaped` 的形狀。
   - 具體來說，對於每個位置 `(i, j, k)`，計算如下：
     \[
     \text{result}[i, j, k] = \text{y_reshaped}[i, j, k] \times \text{weights_expanded}[i, j, 0]
     \]
   - 因此，乘法結果會是：
     ```python
     tensor([[[1.2, 2.4, 3.6],
              [1.2, 2.4, 3.6]],

             [[2.4, 3.0, 3.6],
              [8.4, 10.5, 12.6]]])
     ```
   - 接下來，對乘法結果沿著維度 1 進行求和。維度 1 對應於 `y_reshaped` 和 `weights_expanded` 的第二個維度。
   - 具體來說，對於每個位置 `(i, k)`，計算如下：
     \[
     \text{result}[i, k] = \sum_{j=0}^{1} \text{乘法結果}[i, j, k]
     \]
   - 因此，求和結果會是：
     ```python
     [[2.4, 4.8, 7.2],
      [10.8, 13.5, 16.2]]
     ```

### 數值示例

假設有以下數據：
```python
y = torch.tensor([[2.0, 4.0, 6.0],
                  [3.0, 6.0, 9.0],
                  [8.0, 10.0, 12.0],
                  [12.0, 15.0, 18.0]])
# y.shape => (4, 3)

topk_weight = torch.tensor([[0.6, 0.4],
                           [0.3, 0.7]])
# topk_weight.shape => (2, 2)
```

加權組合的過程：
1. 重塑 `y`：`y_reshaped = y.view(*topk_weight.shape, -1)`
   - `topk_weight`形狀為`(2, 2)`
   - `y`形狀從`(4, 3)`變為：`(2, 2, 3)`
      ```python
      y_reshaped =
      tensor([[[2.0, 4.0, 6.0],
               [3.0, 6.0, 9.0]],

               [[8.0, 10.0, 12.0],
               [12.0, 15.0, 18.0]]])
      ```

2. 擴展權重：`weights_expanded = topk_weight.unsqueeze(-1)`
   - 形狀變為：`(2, 2, 1)`
   ```python
   tensor([[[0.6],
            [0.4]],

            [[0.3],
            [0.7]]])
   ```

3. 乘法和求和：`result = (y_reshaped * weights_expanded).sum(dim=1)`
   - 首先，`y_reshaped` 和 `weights_expanded` 進行元素級乘法。由於 `y_reshaped` 的形狀是 `(2, 2, 3)`，而 `weights_expanded` 的形狀是 `(2, 2, 1)`，PyTorch 會自動廣播 `weights_expanded` 以匹配 `y_reshaped` 的形狀。
   - 具體來說，對於每個位置 `(i, j, k)`，計算如下：
     \[
     \text{result}[i, j, k] = \text{y_reshaped}[i, j, k] \times \text{weights_expanded}[i, j, 0]
     \]
   - 因此，乘法結果會是：
     ```python
     tensor([[[1.2, 2.4, 3.6],
              [1.2, 2.4, 3.6]],

             [[2.4, 3.0, 3.6],
              [8.4, 10.5, 12.6]]])
     ```
   - 接下來，對乘法結果沿著維度 1 進行求和。維度 1 對應於 `y_reshaped` 和 `weights_expanded` 的第二個維度。
   - 具體來說，對於每個位置 `(i, k)`，計算如下：
     \[
     \text{result}[i, k] = \sum_{j=0}^{1} \text{乘法結果}[i, j, k]
     \]
   - 因此，求和結果會是：
     ```python
     [[2.4, 4.8, 7.2],
      [10.8, 13.5, 16.2]]
     ```

## 總結

`MOEFeedForward` 在訓練模式下的操作主要包含三個步驟：
1. 重複輸入數據以支持多專家處理
2. 使用布爾索引收集各專家的輸出
3. 通過加權組合將多個專家的輸出整合為最終結果

這種設計使得模型能夠：
- 動態選擇多個專家處理每個 token
- 根據專家的權重進行靈活的輸出組合
- 在訓練過程中學習最佳的路由策略 