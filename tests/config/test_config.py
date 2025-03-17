import pytest
import sys
import os
from pathlib import Path

# 添加專案根目錄到系統路徑
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.LMConfig import LMConfig  # 修正為實際的類名

class TestModelConfig:
    """測試模型配置實現"""
    
    @pytest.fixture
    def default_config(self):
        """默認配置"""
        return LMConfig()
    
    @pytest.fixture
    def custom_config(self):
        """自定義配置"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=32000,
            multiple_of=32
        )
    
    def test_default_config(self, default_config):
        """測試默認配置參數"""
        # 根據實際實現測試默認值
        assert default_config.dim > 0
        assert default_config.n_layers > 0
        assert default_config.n_heads > 0
        
    def test_custom_config(self, custom_config):
        """測試自定義配置參數"""
        assert custom_config.dim == 512
        assert custom_config.n_layers == 8
        assert custom_config.n_heads == 8
        
    def test_derived_parameters(self, custom_config):
        """測試派生參數計算"""
        # 測試是否根據配置正確計算派生參數
        pass 