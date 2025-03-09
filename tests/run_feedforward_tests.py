#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FeedForward 模組測試運行器
這個腳本提供了一個簡單的方式來運行 FeedForward 類的所有測試。
"""

import unittest
import sys
import os

# 確保可以從專案根目錄導入模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 導入測試模組
from tests.test_feedforward import TestFeedForward


def run_all_tests():
    """運行所有 FeedForward 相關測試"""
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFeedForward)
    
    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回測試結果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("開始運行 FeedForward 測試...")
    success = run_all_tests()
    
    # 根據測試結果設置退出碼
    sys.exit(0 if success else 1) 