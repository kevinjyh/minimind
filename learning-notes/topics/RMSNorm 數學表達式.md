# RMSNorm 數學表達式

RMSNorm 的 forward 函數以數學式表達為：

$$\text{RMSNorm}(x) = w \cdot \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}}$$

其中：

- $x$ 是輸入向量
- $w$ 是可學習的權重參數 (self.weight)
- $n$ 是向量 $x$ 的維度
- $\epsilon$ 是一個很小的常數 (self.eps)，用來確保分母不為零
- $\sum_{i=1}^{n}x_i^2$ 表示 $x$ 的所有元素的平方和
- $\frac{1}{n}\sum_{i=1}^{n}x_i^2$ 是 $x$ 的均方值 (mean square value)

代碼中的實現：

```python
def forward(self, x):
    return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
```

其中：

- `x.pow(2)` 計算 $x^2$
- `.mean(-1, keepdim=True)` 計算最後一個維度的平均值，相當於 $\frac{1}{n}\sum_{i=1}^{n}x_i^2$
- `torch.rsqrt(...)` 計算倒數平方根，相當於 $\frac{1}{\sqrt{...}}$
- `x.float() * ...` 進行元素級乘法，相當於 $\frac{x}{\sqrt{...}}$
- `self.weight * ...` 應用可學習的縮放參數，相當於 $w \cdot \frac{x}{\sqrt{...}}$
- `.type_as(x)` 確保輸出與輸入具有相同的數據類型
