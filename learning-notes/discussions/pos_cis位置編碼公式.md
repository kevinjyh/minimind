# pos_cis 位置編碼公式

`precompute_pos_cis` 函式的演算法可以轉換成以下數學公式：

1. **計算頻率**：
    $$
   \text{freqs} = \frac{1}{\theta^{\frac{j}{\text{dim}}}} \quad \text{for } j = 0, 2, 4, \ldots, \text{dim} - 1
   $$

2. **生成時間序列**：
    $$
   t = [0, 1, 2, \ldots, \text{end} - 1]
   $$

3. **計算外積**：
    $$
   \text{freqs\_matrix} = t \otimes \text{freqs}
   $$

4. **計算位置編碼**：
    $$
   \text{pos\_cis} = \text{polar}(1, \text{freqs\_matrix})
   $$

這裡，$\text{polar}(r, \theta)$ 表示將極坐標轉換為複數形式，其中 $r$ 是模（在這裡為 1），$\theta$ 是相位角（即 $\text{freqs\_matrix}$ 的值）。

## 總結

整體的數學公式可以表示為：

$$
\text{pos\_cis} = \text{polar}(1, t \otimes \frac{1}{\theta^{\frac{j}{\text{dim}}}}) \quad \text{for } j = 0, 2, 4, \ldots, \text{dim} - 1
$$

這些公式描述了如何計算位置編碼的過程，從頻率的計算到最終的複數形式的生成。
