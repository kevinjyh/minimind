`repeat_kv` 在大型語言模型 (LLM) 中主要用於 **Grouped-Query Attention (GQA)** 或 **Multi-Query Attention (MQA)** 機制，其作用是將 Key (K) 和 Value (V) 張量進行擴展，以適應多頭注意力 (Multi-Head Attention) 的並行計算需求。

來源: model\model.py(L57-L66)
```python:model/model.py
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
```

參數說明：
- `x`：Key 或 Value 張量，形狀為 `(bs, slen, n_kv_heads, head_dim)`
    - `bs`：批次大小 (batch size)
    - `slen`：序列長度 (sequence length)
    - `n_kv_heads`：Key/Value 的頭數 (number of key/value heads)
    - `head_dim`：每個頭的維度 (dimension of each head)
- `n_rep`：重複次數，表示每個 Key/Value 頭需要重複多少次

運作方式：
1. **檢查重複次數**：如果 `n_rep` 為 1，表示不需要重複，直接返回原始張量。
2. **擴展維度**：在 `n_kv_heads` 後插入一個維度，用於重複。
3. **重複擴展**：使用 `expand` 沿新增的維度重複 `n_rep` 次。
4. **重塑形狀**：將重複的維度合併到 `n_kv_heads`，得到最終的形狀 `(bs, slen, n_kv_heads * n_rep, head_dim)`。

作用：
- **GQA/MQA 實現**：在 GQA 和 MQA 中，Key 和 Value 的頭數通常少於 Query 的頭數。`repeat_kv` 負責將 Key 和 Value 的頭數擴展到與 Query 相同，以便進行注意力計算。
- **並行計算**：通過重複 Key 和 Value，可以讓不同的 Query 頭與不同的 Key/Value 頭進行並行計算，提高計算效率。
- **降低顯存佔用**：相比於傳統的多頭注意力，GQA 和 MQA 可以顯著降低 Key 和 Value 張量的顯存佔用，尤其是在長序列場景下。

舉例：
假設 Query 有 8 個頭，而 Key 和 Value 只有 2 個頭，則 `n_rep` 為 4。`repeat_kv` 會將 Key 和 Value 的每個頭重複 4 次，最終得到 8 個頭，與 Query 的頭數一致。

總結：
`repeat_kv` 函式是 GQA 和 MQA 的關鍵組成部分，它通過重複 Key 和 Value 張量，實現了高效的多頭注意力計算，並降低了顯存佔用，對於訓練和部署大型語言模型至關重要。
