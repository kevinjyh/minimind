# MiniMindLM._stream函式分析

在附上的代碼段落中，這段代碼的主要目的是實現基於概率的篩選，通常用於生成模型的解碼過程中，以控制生成的多樣性。這裡使用了 top-p (nucleus) 取樣方法。讓我們逐行分析這段代碼的具體含義和作用。

## 代碼分析:model/model.py (lines 353-362)

```python
# 初始化生成參數
start, first_seq, past_kvs = input_ids.shape[1], True, None  # 獲取初始序列長度，標記首次生成

while input_ids.shape[1] < max_new_tokens - 1:  # 持續生成直到達到最大長度
    # ▼▼ 條件判斷處理首次生成與後續生成 ▼▼
    if first_seq or not use_cache:
        # 首次生成需處理完整序列
        out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
    else:
        # 後續生成只需處理最後一個token (自回歸特性)
        out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                   start_pos=input_ids.shape[1] - 1, **args)
    
    # ▼▼ 提取關鍵數據 ▼▼
    logits, past_kvs = out.logits[:, -1, :], out.past_key_values  # 取最後一個token的預測分數
    
    # ▼▼ 反覆懲罰機制 ▼▼
    logits[:, list(set(input_ids.tolist()[0]))] /= rp  # 降低已出現token的生成概率
    logits /= (temperature + 1e-9)  # 溫度調節控制隨機性
```

主要功能模組解析：

1. **自回歸生成控制**
   - `input_ids[:, -1:]` 切片操作：每次只取序列最後一個token作為輸入
   - `past_kvs` 參數：儲存歷史注意力鍵值對加速計算
   - 典型輸入形演變：`[A,B,C] → [C] → [D] → [E]...`

2. **反覆懲罰機制**
   - 數學公式：$\text{adjusted\_logits}[t] = \frac{\text{logits}[t]}{rp}, \forall t \in \text{seen\_tokens}$
   - 範例：若已生成token為[1,3]，rp=2：

     ```text
     原始logits: [0.1, 0.4, 0.3, 0.2]
     調整後logits: [0.1/2, 0.4, 0.3/2, 0.2] = [0.05, 0.4, 0.15, 0.2]
     ```

3. **溫度調節**
   - 低溫 (temperature→0)：強化高概率token，輸出更確定

     ```text
     logits = [1, 3, 2] → softmax = [0.09, 0.67, 0.24]
     ```

   - 高溫 (temperature=1)：保持原始分佈

     ```text
     logits = [1, 3, 2] → softmax = [0.09, 0.67, 0.24] (不變)
     ```

   - 超高溫 (temperature=10)：平滑分佈

     ```text
     logits = [1, 3, 2] → softmax ≈ [0.24, 0.47, 0.29]
     ```

流程圖解：

```text
初始輸入
   ↓
[完整序列處理] → 保存past_kvs → [取最後token] → 循環處理
   ↓                        ↑
生成新token ← 調整logits ← 溫度控制
```

## 代碼分析:model/model.py (lines 363-376)

```python
if top_p is not None and top_p < 1.0:
      sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
      sorted_probs = F.softmax(sorted_logits, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      sorted_indices_to_remove = cumulative_probs > top_p
      sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
      sorted_indices_to_remove[:, 0] = False
      indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
      logits[indices_to_remove] = -float('Inf')
input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
input_ids = torch.cat((input_ids, input_ids_next), dim=1)
yield input_ids[:, start:]
if input_ids_next.item() == eos_token_id:
      break
```

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
```

1. **排序 logits**：
   - `torch.sort(logits, descending=True, dim=-1)` 將 `logits` 按照最後一個維度（即每個 token 的 logits）進行降序排序。
   - `sorted_logits` 是排序後的 logits，`sorted_indices` 是原始 logits 中對應的索引。

    ```python
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    ```

2. **計算概率**：
   - `F.softmax(sorted_logits, dim=-1)` 將排序後的 logits 轉換為概率分佈。這樣可以得到每個 token 被選中的概率。

    ```python
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    ```

3. **計算累積概率**：
   - `torch.cumsum(sorted_probs, dim=-1)` 計算累積概率，這樣可以方便地確定哪些 token 的累積概率超過了指定的 `top_p` 閾值。

    ```python
    sorted_indices_to_remove = cumulative_probs > top_p
    ```

4. **確定要移除的索引**：
   - `cumulative_probs > top_p` 生成一個布爾張量，標記出累積概率超過 `top_p` 的位置。這些位置的 token 將被移除。

    ```python
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    ```

5. **調整布爾張量**：
   - `sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()` 將布爾張量的第一個元素設置為 `False`，這樣可以確保至少保留一個 token（即概率最高的 token），並將其餘的布爾值向左移動。
   - 這樣的操作確保了在進行篩選時，第一個 token 不會被移除。

    ```python
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    ```

6. **生成最終要移除的索引**：
   - `scatter` 函數將布爾張量 `sorted_indices_to_remove` 的值根據 `sorted_indices` 的位置映射回原始的 logits 索引，生成最終的 `indices_to_remove` 張量。

    ```python
    logits[indices_to_remove] = -float('Inf')
    ```

7. **設置 logits 為負無窮**：
   - 將要移除的 token 的 logits 設置為 `-float('Inf')`，這樣在後續的 softmax 計算中，這些 token 的概率將變為 0，從而不會被選中。

### 總結

這段代碼的主要作用是實現 top-p (nucleus) 取樣，通過計算 logits 的概率分佈，然後根據累積概率篩選出要移除的 token。這樣的設計可以控制生成的多樣性，確保生成的內容不會過於集中在少數幾個 token 上，從而提高生成文本的質量和多樣性。
