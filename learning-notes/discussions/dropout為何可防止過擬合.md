# Dropout 技術防止過擬合機制解析

## 基本概念

`dropout` 是一種神經網絡正則化技術，在訓練階段隨機丟棄部分神經元輸出，其防止過擬合的核心原理如下：

![Dropout 運作示意圖](https://miro.medium.com/v2/resize:fit:1400/1*iWQzxhVlvadk6VAJjsgXgg.png)

## 三層作用機制

### 1. 神經元協同抑制

| 機制特性          | 無 Dropout          | 有 Dropout          |
|-------------------|---------------------|---------------------|
| 神經元依賴性      | 高度依賴固定組合    | 獨立特徵提取能力    |
| 特徵組合模式      | 靜態組合            | 動態隨機組合        |
| 信息冗餘度        | 低                  | 高                  |

```python
# PyTorch 實現代碼示例
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

### 2. 隱含模型平均效應

- **子網絡抽樣**：每次訓練隨機產生 $2^n$ 種子網絡結構（n=神經元數）
- **投票機制**：測試階段整合所有子網絡預測結果
- **誤差補償**：不同子網絡的錯誤模式互補

### 3. 噪聲注入機制

- **輸入擾動**：隨機屏蔽產生數據變體
- **梯度平滑**：損失函數地形平坦化
- **魯棒特徵**：迫使學習噪聲不敏感特徵

## 訓練/測試階段差異

| 階段      | Dropout 狀態 | 輸出調整               | 數學表達式                     |
|-----------|--------------|------------------------|--------------------------------|
| 訓練      | 啟用         | 無                     | $y = \text{mask} \cdot x$     |
| 推理      | 關閉         | 乘補數係數             | $y = (1-p) \cdot x$           |

## 參數設置建議

1. **基礎網絡架構**
   - 原始神經元數量增加 10-20%
   - 使用 He/Kaiming 初始化

2. **丟棄率選擇**

   ```mermaid
   graph LR
   A[輸入層] -->|p=0.2| B[隱藏層]
   B -->|p=0.5| C[輸出層]
   ```

3. **組合技巧**
   - 與 L2 正則化並用時需降低強度
   - 配合 BatchNorm 時注意層序
   - 在 RNN 中建議僅用於全連接層

## 實證效果

在 ImageNet 數據集上的比較實驗：

![Dropout 效果比較](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-24_at_2.46.28_PM.png)

## 進階變體

1. **Spatial Dropout**：CNN 特徵圖整通道丟棄
2. **Alpha Dropout**：保持自歸一化特性
3. **DropConnect**：隨機斷開權重連接
4. **Zoneout**：RNN 隱藏狀態隨機保持

此技術通過引入受控隨機性，有效提升模型泛化能力，已成為現代深度學習模型的標準組件。
