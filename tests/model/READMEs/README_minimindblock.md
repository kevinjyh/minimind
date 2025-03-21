# MiniMindBlock 測試指南

本文檔提供了關於 `MiniMindBlock` 類別的測試案例說明，並指導如何通過這些測試更深入理解該模組的工作原理。

## 我的Cursor提示詞

我想透過 pytest 測試案例的方式來完全理解 `class MiniMindBlock` 的代碼原理及功能，請依以下需求完成我的目的：

- 以 pytest 測試的模式，並將測試源碼檔寫入 `本專案根目錄\tests\` 下。
- 將以下主題創建並寫入 `本專案根目錄\tests\` 下適當的 READMD 檔案：
  - 各測試案例功能及作用
  - 以研究代碼各功能的角度，寫下如何調整測試案例的各參數，以深入理解該類別代碼
  - 以研究代碼各功能的角度，寫下如何由淺入深的順序來研讀測試案例或方式。

## 簡介

`MiniMindBlock` 是一個神經網絡模組，它實現了類似於 Transformer 模型中的一個區塊，主要包含以下部分：

1. 自注意力機制（Self-Attention）
2. 前饋神經網絡（Feed-Forward Network）
3. 層標準化（Layer Normalization，這裡使用 RMSNorm）
4. 殘差連接（Residual Connection）

測試案例設計旨在幫助使用者了解這些組件如何交互工作，以及如何影響模型的整體行為。

## 測試案例功能及作用

### 1. 初始化測試 (`test_init`)

**功能**：測試 `MiniMindBlock` 的初始化過程和屬性設置是否正確。

**作用**：

- 驗證配置參數是否正確應用到模型中
- 檢查各子模組（注意力、標準化層、前饋網絡）是否正確初始化
- 確認 MoE（Mixture of Experts）版本與標準版本的差異

### 2. 前向傳播形狀測試 (`test_forward_shape`)

**功能**：測試前向傳播的輸出形狀是否符合預期。

**作用**：

- 確保模型處理後的張量維度正確
- 驗證模型能夠處理批次數據
- 確認不使用緩存時的輸出形狀

### 3. 緩存機制測試 (`test_forward_with_cache`)

**功能**：測試模型使用緩存（KV Cache）時的前向傳播。

**作用**：

- 了解 KV 緩存的結構和形狀
- 確認啟用緩存時模型的行為
- 檢驗緩存的生成是否符合預期

### 4. 殘差連接測試 (`test_residual_connection`)

**功能**：測試殘差連接的功能。

**作用**：

- 驗證當注意力和前饋網絡輸出為零時，輸出應等於輸入
- 理解殘差連接在 Transformer 架構中的重要性
- 檢查殘差連接的實現是否正確

### 5. 標準化層測試 (`test_attention_norm` 和 `test_ffn_norm`)

**功能**：測試 RMSNorm 層的作用。

**作用**：

- 觀察標準化對輸入數據的影響
- 確認標準化後的統計特性（方差接近 1）
- 了解為什麼在注意力和前饋網絡前需要標準化

### 6. 組件貢獻測試 (`test_attention_and_ffn_contribute`)

**功能**：測試注意力和前饋網絡各自對模型輸出的貢獻。

**作用**：

- 驗證每個組件都對最終輸出有實質性影響
- 理解注意力和前饋網絡的相對重要性
- 確認兩個組件在功能上是互補的

### 7. MoE 與標準前饋網絡比較 (`test_moe_vs_standard_ffn`)

**功能**：比較使用標準前饋網絡和 MoE 前饋網絡的區別。

**作用**：

- 了解 MoE 的行為特性
- 觀察兩種網絡架構的輸出差異
- 檢查 MoE 模型的輔助損失生成

### 8. 過去緩存重用測試 (`test_past_key_value_reuse`)

**功能**：測試利用過去的 KV 緩存進行增量生成的機制。

**作用**：

- 了解語言模型如何使用緩存提高生成效率
- 觀察增量式生成時緩存的變化
- 確認緩存重用的正確性

## 如何調整測試參數深入理解代碼

### 1. 修改配置參數

通過調整 `basic_config` 和 `moe_config` 中的參數，可以觀察不同參數對模型行為的影響：

```python
@pytest.fixture
def basic_config(self):
    return LMConfig(
        dim=512,           # 嘗試改變隱藏維度大小，如 256、1024
        n_layers=8,        # 單元測試中這個參數影響不大
        n_heads=8,         # 嘗試改變頭數，如 4、16
        n_kv_heads=2,      # 嘗試改變 KV 頭數，如 1、4、8（與 n_heads 相同）
        vocab_size=6400,   # 單元測試中影響不大
        max_seq_len=128,   # 可調整以測試不同長度的序列
        use_moe=False,
        flash_attn=False,
    )
```

特別關注：

- `n_heads` 和 `n_kv_heads` 的關係：這影響分組查詢注意力的行為
- `dim` 的大小：影響模型的容量和計算複雜度
- `use_moe`：切換使用標準前饋網絡或 MoE 架構

### 2. 調整測試輸入

修改 `sample_input` 中的參數可以測試模型對不同輸入特性的響應：

```python
@pytest.fixture
def sample_input(self, basic_config):
    batch_size = 2         # 嘗試更大的批次大小，如 4、8
    seq_len = 10           # 嘗試不同的序列長度，如 1、32、128
    # ...
```

注意事項：

- 較長的序列可以測試注意力機制的長依賴性能力
- 單一序列長度（seq_len=1）可以測試增量生成情境
- 大批次可以測試並行處理能力

### 3. 研究殘差連接的影響

在 `test_residual_connection` 中，可以修改測試方式，例如：

- 只禁用前饋網絡，保留注意力機制
- 只禁用注意力機制，保留前饋網絡
- 使用不同強度的殘差信號（例如，將原始輸入乘以係數）

### 4. 探索 MoE 的作用

在 `test_moe_vs_standard_ffn` 中，可以更深入地比較兩種架構：

- 記錄並比較計算時間
- 檢查 MoE 選擇了哪些專家（通過訪問 `moe_block.feed_forward.gate` 內部狀態）
- 調整 `n_routed_experts` 和 `num_experts_per_tok` 參數以觀察其影響

### 5. 研究緩存機制

在 `test_past_key_value_reuse` 中：

- 嘗試模擬更長的生成序列（多次重用緩存）
- 檢查緩存大小隨序列長度的增長
- 比較啟用和禁用緩存的性能差異

## 由淺入深研讀測試案例的建議順序

### 第一階段：基本概念

1. **`test_init`**：了解 `MiniMindBlock` 的基本結構和組件
2. **`test_forward_shape`**：熟悉輸入和輸出的形狀和維度
3. **`test_attention_norm` 和 `test_ffn_norm`**：理解標準化層的作用

### 第二階段：組件交互

1. **`test_residual_connection`**：了解殘差連接的重要性
2. **`test_attention_and_ffn_contribute`**：理解各組件的貢獻
3. **`test_forward_with_cache`**：初步了解緩存機制

### 第三階段：進階特性

1. **`test_past_key_value_reuse`**：深入理解增量式生成和緩存重用
2. **`test_moe_vs_standard_ffn`**：了解 MoE 架構的特點和優勢

### 第四階段：實驗和調整

- 修改配置參數，觀察對模型行為的影響
- 調整測試輸入的特性（批次大小、序列長度）
- 利用測試工具查看中間結果和內部狀態

## 總結

測試 `MiniMindBlock` 不僅是為了確認其功能正確性，更是深入理解其工作原理的絕佳方式。通過調整參數、觀察輸出和分析中間狀態，可以獲得對 Transformer 架構核心組件如何交互的深刻認識。

特別是，這些測試幫助我們理解：

1. 自注意力機制如何處理序列數據
2. 殘差連接如何幫助梯度流動和信息傳遞
3. 層標準化如何穩定訓練過程
4. MoE 架構如何通過專家路由提高模型容量和效率
5. KV 緩存如何優化自迴歸生成效率
