# MiniMind Attention機制測試說明

本目錄包含用於理解和測試MiniMind中Attention機制的測試案例。這些測試案例專為低計算資源環境設計，可在普通CPU（如筆電的NVIDIA GeForce MX450）上運行。

## 測試文件說明

本目錄包含以下測試文件：

1. **test_attention.py**：針對MiniMind原始模型中的Attention類的測試案例
   - 測試基本功能和維度一致性
   - 測試KV緩存機制
   - 測試repeat_kv函數
   - 分析注意力模式和掩碼效果

2. **test_attention_standalone.py**：包含一個獨立實現的簡化版自注意力機制
   - 提供清晰易讀的注意力計算實現
   - 包含更多直觀的測試案例，展示注意力的工作原理
   - 特別設計了可視化注意力權重的方法

3. **attention_visualization.py**：專門用於可視化注意力機制的互動式工具
   - 提供三種不同的注意力機制視覺化演示
   - 使用熱力圖直觀展示注意力權重分佈
   - 幫助理解注意力機制的核心概念

## 如何運行測試

要運行測試，您需要安裝pytest和PyTorch：

```bash
pip install pytest torch
```

然後，在專案根目錄運行：

```bash
# 運行所有測試
pytest tests/

# 運行特定測試文件
pytest tests/test_attention.py
pytest tests/test_attention_standalone.py

# 運行特定測試函數
pytest tests/test_attention_standalone.py::TestSimpleAttention::test_self_attention_demonstration
```

## 使用 attention_visualization.py 可視化注意力機制

`attention_visualization.py` 是一個互動式工具，專門用於可視化和理解注意力機制的核心概念。

### 使用方法

運行以下命令啟動可視化工具：

```bash
python tests/attention_visualization.py
```

程式會提示您選擇以下三種演示之一：

1. **因果掩碼效果演示**：展示因果掩碼（causal mask）如何影響注意力分佈
2. **主題關注模式演示**：展示自注意力如何自動識別和關注相似的主題
3. **多頭注意力機制演示**：展示不同注意力頭如何關注輸入的不同方面

### 演示原理與重點

#### 1. 因果掩碼效果演示 (demo_causal_mask)

- **原理**：展示有無因果掩碼時注意力權重的差異
- **重點**：
  - 有因果掩碼時，每個位置只能關注自己和之前的位置（上三角部分被遮蔽）
  - 無因果掩碼時，每個位置可以關注序列中的任何位置
  - 這是自迴歸生成模型（如GPT）的核心機制，確保模型只基於已知信息生成

#### 2. 主題關注模式演示 (demo_topic_attention)

- **原理**：使用one-hot向量創建具有明確主題結構的序列，展示自注意力如何捕捉相似性
- **重點**：
  - 同一主題的token之間會形成高注意力權重
  - 不同主題的token之間會形成低注意力權重
  - 熱力圖清晰展示了主題聚類效果
  - 這展示了自注意力作為相似度計算的本質

#### 3. 多頭注意力機制演示 (demo_multi_head_attention)

- **原理**：創建包含多種特徵（主題和情感）的序列，展示不同注意力頭如何專注於不同特徵
- **重點**：
  - 頭1專注於捕捉主題相似性
  - 頭2專注於捕捉情感相似性
  - 這展示了多頭注意力如何允許模型同時關注不同類型的關係
  - 解釋了為什麼多頭注意力比單頭注意力更強大

### 技術細節

- 使用 Matplotlib 和 Seaborn 繪製熱力圖
- 支持中文顯示，自動檢測並使用系統中可用的中文字體
- 使用 SimpleAttention 類作為基礎，通過控制權重矩陣展示不同概念
- 提供詳細的註釋和解釋，幫助理解每個演示的目的

### 擴展建議

- 嘗試修改代碼中的參數（如序列長度、隱藏維度、頭數等）觀察效果變化
- 創建自己的序列模式，觀察注意力機制如何捕捉這些模式
- 結合 RoPE（旋轉位置編碼）觀察位置信息如何影響注意力計算

## 使用測試了解Attention機制

這些測試案例設計用於幫助您深入理解Attention機制的工作原理：

### 基本概念測試

1. **多頭注意力（Multi-head Attention）**：
   - `test_multi_head_independence`展示了不同注意力頭如何關注不同的特徵

2. **注意力掩碼（Attention Mask）**：
   - `test_causal_mask`和`test_no_causal_mask`展示了因果掩碼如何確保生成模型只關注已有內容

3. **注意力權重（Attention Weights）**：
   - `test_attention_pattern_with_one_hot`展示了不同token之間的注意力分配

### 直觀理解測試

1. **主題關注測試**：
   - `test_self_attention_demonstration`展示了相似內容之間如何自動形成高注意力權重
   - 通過控制query/key的投影矩陣，清晰展示注意力的本質是相似度計算

2. **KV緩存測試**：
   - `test_kv_cache`展示了模型如何重用之前計算過的key和value以加速生成

## 學習建議

1. **閱讀SimpleAttention實現**：`test_attention_standalone.py`中的SimpleAttention類提供了注意力機制的完整、簡潔實現，附有詳細註釋

2. **修改並觀察效果**：
   - 嘗試修改`causal_mask`參數，觀察注意力模式變化
   - 更改sequence長度、隱藏層維度等參數，觀察對結果的影響
   - 嘗試添加position embedding，理解位置信息的重要性

3. **可視化注意力權重**：
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # 獲取注意力權重
   _, attn_weights = attn(x, causal_mask=False)
   weights = attn_weights[0, 0].detach().numpy()
   
   # 可視化
   plt.figure(figsize=(10, 8))
   sns.heatmap(weights, annot=True, cmap="YlGnBu")
   plt.title("Attention Weights")
   plt.xlabel("Key position")
   plt.ylabel("Query position")
   plt.show()
   ```

4. **連接到MiniMind原始模型**：
   - 理解SimpleAttention與原始Attention類之間的差異
   - 特別關注MiniMind如何實現rotary position embedding和grouped-query attention

## 測試案例擴展建議

想要更深入理解，可以嘗試實現以下額外測試：

1. 實現一個測試，展示position embedding如何影響注意力計算
2. 創建一個測試，展示不同溫度參數如何影響注意力分佈
3. 對比原始模型的Attention與簡化版SimpleAttention的行為差異

通過這些測試，您可以在不需要完整訓練模型的情況下，深入理解大型語言模型中最核心的Attention機制。 