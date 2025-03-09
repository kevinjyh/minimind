# FeedForward 測試說明

本文件說明如何執行 `FeedForward` 類的單元測試。

## 測試內容概述

`test_feedforward.py` 測試檔包含以下測試：

1. **初始化測試**：確保 `FeedForward` 類能夠正確初始化，線性層維度和 dropout 參數設置符合預期。
2. **自動計算 hidden_dim**：測試當 `hidden_dim=None` 時，是否能正確根據公式自動計算 hidden_dim。
3. **前向傳播形狀測試**：確保輸出張量的形狀與輸入匹配。
4. **前向傳播計算測試**：確認前向傳播的計算結果與預期一致。
5. **Dropout 測試**：驗證 dropout 在訓練和評估模式下的不同表現。
6. **批次大小測試**：確保模型能處理不同批次大小的輸入。
7. **序列長度測試**：確保模型能處理不同序列長度的輸入。

## 測試環境要求

- Python 3.8+
- PyTorch 1.7+
- Transformers 庫

## 執行測試的方法

### 方法一：使用 unittest 直接執行

```bash
python -m unittest tests/test_feedforward.py
```

### 方法二：使用 pytest 執行

```bash
pytest tests/test_feedforward.py -v
```

使用 `-v` 參數可以顯示詳細的測試結果。

### 方法三：在 Python 中直接執行測試文件

```bash
python tests/test_feedforward.py
```

## 測試結果解釋

測試成功後，你應該會看到類似以下的輸出：

```
......
----------------------------------------------------------------------
Ran 7 tests in X.XXs

OK
```

如果有測試失敗，會顯示具體的失敗原因和堆疊追蹤，幫助你診斷問題。

## 常見問題及解決方案

1. **導入錯誤**：如果遇到模塊導入錯誤，請確保你的 Python 路徑中包含專案根目錄。
   ```bash
   # 在專案根目錄下執行
   export PYTHONPATH=$PYTHONPATH:$(pwd)  # Linux/Mac
   set PYTHONPATH=%PYTHONPATH%;%cd%      # Windows
   ```

2. **CUDA 相關錯誤**：如果測試在 GPU 上運行時發生錯誤，可以嘗試將測試限制在 CPU 上：
   ```python
   # 在測試開始前添加
   torch.cuda.is_available = lambda: False
   ```

3. **失去確定性**：如果對比測試結果不一致，請確保在相關測試中設置了隨機種子：
   ```python
   torch.manual_seed(42)
   ``` 