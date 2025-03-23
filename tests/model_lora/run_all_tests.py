import pytest
import os
import sys
from pathlib import Path

# 確保可以導入主項目
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

if __name__ == "__main__":
    # 運行當前目錄中的所有測試文件
    current_dir = Path(__file__).parent
    
    print("===============================================")
    print("開始運行 LoRA 測試套件")
    print("===============================================")
    
    # 依序運行測試文件
    test_files = [
        "test_lora.py",
        "test_lora_training.py",
        "test_lora_visualization.py"
    ]
    
    for test_file in test_files:
        file_path = current_dir / test_file
        if file_path.exists():
            print(f"\n運行測試文件: {test_file}")
            print("-" * 50)
            pytest.main(["-xvs", str(file_path)])
        else:
            print(f"警告: 找不到測試文件 {test_file}")
    
    print("\n===============================================")
    print("所有測試完成")
    print("===============================================")
    print("\n可視化結果保存在: " + str(current_dir / "visualization_output"))
    print("===============================================") 