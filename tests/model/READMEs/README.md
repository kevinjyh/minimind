# MiniMind 測試系統

本目錄包含 MiniMind 專案的測試案例，按照源碼文件類型進行分類。

## 測試目錄結構

```
tests/
  ├── model/                 # model.py 相關測試
  │   ├── test_attention.py      # 注意力機制測試
  │   ├── test_gqa.py            # GQA 機制測試
  │   ├── test_minimindblock.py  # MiniMindBlock 測試
  │   ├── test_moefeedforward.py # MOE 前饋層測試
  │   ├── test_minimindlm.py     # MiniMindLM 模型測試
  │   └── test_minimindlm_generation.py  # 生成功能測試
  │
  ├── model_lora/            # model_lora.py 相關測試
  │   └── test_lora.py           # LoRA 機制測試
  │
  ├── dataset/               # dataset.py 相關測試
  │   └── test_dataset.py        # 數據集和批處理測試
  │
  ├── config/                # LMConfig.py 相關測試
  │   └── test_config.py         # 模型配置測試
  │
  ├── README.md              # 測試文檔
  └── __init__.py            # 包初始化文件
```

## 使用方法

在 Cursor IDE 中，您可以通過右側的 TEST EXPLORER 面板方便地瀏覽和執行測試：

1. 選擇特定的源碼模塊（model、model_lora、dataset、config）
2. 展開對應的測試文件
3. 點擊特定的測試方法執行單一測試
4. 或點擊測試文件執行文件中的所有測試
5. 或點擊模塊執行該模塊的所有測試

## 添加新測試

要添加新的測試案例，請遵循以下步驟：

1. 確定要測試的源碼文件
2. 在對應的子目錄中創建或更新測試文件
3. 按照現有測試的模式添加新的測試方法

### 測試文件模板

```python
import pytest
import torch
import sys
from pathlib import Path

# 添加專案根目錄到系統路徑
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.目標模塊 import 目標類

class Test目標類:
    """測試目標類的實現"""
    
    @pytest.fixture
    def sample_instance(self):
        """創建測試實例"""
        return 目標類(參數)
    
    def test_功能1(self, sample_instance):
        """測試特定功能"""
        # 測試代碼
        assert 預期結果
```

## 常見問題

### Q: 測試無法導入模塊?
A: 確保測試文件中包含正確的導入路徑：
```python
sys.path.append(str(Path(__file__).parent.parent.parent))
```

### Q: 如何跳過特定測試?
A: 使用 `@pytest.mark.skip(reason="原因")` 裝飾器

### Q: 如何運行一組特定的測試?
A: 在 TEST EXPLORER 中選擇特定模塊或在命令行中使用：
```bash
python -m pytest tests/model/test_attention.py -v
``` 