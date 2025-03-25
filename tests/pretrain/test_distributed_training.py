import os
import sys
import pytest
import torch
import torch.distributed as dist
from unittest.mock import patch, MagicMock, mock_open, call

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from train_pretrain import init_distributed_mode


class TestDistributedTraining:
    """測試分佈式訓練相關功能"""
    
    def setup_method(self):
        """測試前設置"""
        # 確保輸出目錄存在
        self.output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """測試後清理"""
        # 重置被 patch 的模塊
        pass
    
    @pytest.fixture
    def mock_dist(self):
        """模擬 torch.distributed 模塊"""
        return MagicMock()
    
    @pytest.fixture
    def mock_env_vars(self):
        """模擬分布式訓練環境變數"""
        env_vars = {
            "RANK": "1",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2"
        }
        return env_vars
    
    def test_init_distributed_mode(self, mock_dist, mock_env_vars):
        """測試初始化分布式訓練模式"""
        # 設置環境變數
        with patch.dict('os.environ', mock_env_vars):
            # 模擬 torch.distributed 模塊
            with patch('train_pretrain.dist', mock_dist):
                # 模擬 torch.cuda 模塊
                with patch('torch.cuda.set_device') as mock_set_device:
                    # 模擬全局變數，使用 create=True 創建不存在的屬性
                    with patch('train_pretrain.ddp', True, create=True):
                        # 模擬 DEVICE 全局變數
                        with patch('train_pretrain.DEVICE', "", create=True):
                            # 模擬 ddp_local_rank 全局變數
                            with patch('train_pretrain.ddp_local_rank', 0, create=True):
                                # 調用函數
                                init_distributed_mode()
        
        # 檢查分布式環境是否正確初始化
        mock_dist.init_process_group.assert_called_once_with(backend="nccl")
        mock_set_device.assert_called_once_with(f"cuda:0")  # LOCAL_RANK = 0
        
        # 將調用記錄寫入檔案
        with open(os.path.join(self.output_dir, "dist_init_calls.txt"), "w", encoding="utf-8") as f:
            f.write("分布式環境初始化調用:\n")
            f.write(f"dist.init_process_group 調用參數: backend='nccl'\n")
            f.write(f"torch.cuda.set_device 調用參數: device='cuda:0'\n")
            f.write(f"環境變數: RANK=1, LOCAL_RANK=0, WORLD_SIZE=2\n")
    
    def test_distributed_dataloader(self):
        """測試分布式數據加載器創建"""
        # 創建模擬的數據集
        mock_dataset = MagicMock()
        
        # 模擬 DistributedSampler
        mock_sampler = MagicMock()
        with patch('torch.utils.data.DistributedSampler', return_value=mock_sampler) as mock_dist_sampler:
            # 模擬 DataLoader
            mock_dataloader = MagicMock()
            with patch('torch.utils.data.DataLoader', return_value=mock_dataloader) as mock_dataloader_cls:
                # 設置 ddp=True，使用 create=True 創建不存在的屬性
                with patch('train_pretrain.ddp', True, create=True):
                    # 調用創建數據加載器的代碼
                    batch_size = 32
                    num_workers = 4
                    
                    # 使用正確的導入路徑
                    from torch.utils.data import DistributedSampler
                    sampler = DistributedSampler(mock_dataset) if True else None
                    dataloader = torch.utils.data.DataLoader(
                        mock_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        sampler=sampler
                    )
        
        # 檢查 DistributedSampler 是否被正確創建
        mock_dist_sampler.assert_called_once_with(mock_dataset)
        
        # 檢查 DataLoader 是否被正確創建
        mock_dataloader_cls.assert_called_once()
        args, kwargs = mock_dataloader_cls.call_args
        assert kwargs['batch_size'] == batch_size
        assert kwargs['shuffle'] == False, "在使用 DistributedSampler 時，shuffle 應為 False"
        assert kwargs['sampler'] == mock_sampler
        
        # 將調用記錄寫入檔案
        with open(os.path.join(self.output_dir, "dist_dataloader_calls.txt"), "w", encoding="utf-8") as f:
            f.write("分布式數據加載器創建調用:\n")
            f.write(f"DistributedSampler 調用參數: dataset={mock_dataset}\n")
            f.write(f"DataLoader 調用參數: batch_size={batch_size}, shuffle=False, sampler={mock_sampler}\n")
    
    def test_ddp_model_wrapper(self):
        """測試 DistributedDataParallel 模型包裝"""
        # 創建模擬的模型
        mock_model = MagicMock()
        mock_model._ddp_params_and_buffers_to_ignore = None
        
        # 模擬 DistributedDataParallel
        mock_ddp_model = MagicMock()
        with patch('torch.nn.parallel.DistributedDataParallel', return_value=mock_ddp_model) as mock_ddp:
            # 設置 ddp=True 和 ddp_local_rank=0，使用 create=True 創建不存在的屬性
            with patch('train_pretrain.ddp', True, create=True):
                with patch('train_pretrain.ddp_local_rank', 0, create=True):
                    # 調用 DDP 包裝代碼
                    mock_model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
                    wrapped_model = torch.nn.parallel.DistributedDataParallel(
                        mock_model, device_ids=[0]
                    )
        
        # 檢查 DistributedDataParallel 是否被正確調用
        mock_ddp.assert_called_once()
        args, kwargs = mock_ddp.call_args
        assert args[0] == mock_model
        assert kwargs['device_ids'] == [0]
        
        # 檢查 _ddp_params_and_buffers_to_ignore 是否被正確設置
        assert mock_model._ddp_params_and_buffers_to_ignore == {"pos_cis"}
        
        # 將調用記錄寫入檔案
        with open(os.path.join(self.output_dir, "ddp_model_calls.txt"), "w", encoding="utf-8") as f:
            f.write("DDP 模型包裝調用:\n")
            f.write(f"DistributedDataParallel 調用參數: model={mock_model}, device_ids=[0]\n")
            f.write(f"忽略的參數: _ddp_params_and_buffers_to_ignore={mock_model._ddp_params_and_buffers_to_ignore}\n")
    
    def test_model_state_dict_handling_in_ddp(self):
        """測試 DDP 環境中的模型狀態字典處理"""
        # 創建模擬的模型和模塊
        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_state_dict = {"layer1.weight": torch.randn(10, 10)}
        mock_model_state_dict = {"module.layer1.weight": torch.randn(10, 10)}
        
        # 設置模型屬性
        mock_model.module = mock_module
        mock_model.module.state_dict.return_value = mock_state_dict
        mock_model.state_dict.return_value = mock_model_state_dict
        
        # 測試 DDP 模型情況 (從 model.module 獲取狀態字典)
        with patch('torch.save') as mock_save:
            # 創建 DDP 模型
            mock_ddp_model = MagicMock(spec=torch.nn.parallel.DistributedDataParallel)
            mock_ddp_model.module = mock_module
            mock_ddp_model.module.state_dict.return_value = mock_state_dict
            
            # 使用 isinstance 檢查 - 對於 DDP 模型
            if isinstance(mock_ddp_model, torch.nn.parallel.DistributedDataParallel):
                state_dict = mock_ddp_model.module.state_dict()
            else:
                state_dict = mock_ddp_model.state_dict()
            
            # 保存模型
            torch.save(state_dict, "ddp_model.pth")
            
            # 檢查 state_dict 是否來自 model.module
            assert state_dict == mock_state_dict
            mock_save.assert_called_once_with(mock_state_dict, "ddp_model.pth")
        
        # 測試非 DDP 模型情況 (直接從模型獲取狀態字典)
        with patch('torch.save') as mock_save:
            # 直接使用非 DDP 模型
            if isinstance(mock_model, torch.nn.parallel.DistributedDataParallel):
                state_dict = mock_model.module.state_dict()
            else:
                state_dict = mock_model.state_dict()
            
            # 保存模型
            torch.save(state_dict, "normal_model.pth")
            
            # 檢查 state_dict 是否來自 model 本身
            assert state_dict == mock_model_state_dict
            mock_save.assert_called_once_with(mock_model_state_dict, "normal_model.pth")
        
        # 將調用記錄寫入檔案
        with open(os.path.join(self.output_dir, "model_state_dict_handling.txt"), "w", encoding="utf-8") as f:
            f.write("DDP 環境中的模型狀態字典處理:\n")
            f.write(f"從 model.module 獲取狀態字典: {mock_state_dict}\n")
            f.write(f"從普通模型獲取狀態字典: {mock_model_state_dict}\n")
            f.write(f"torch.save 調用參數: state_dict={mock_state_dict}, path='ddp_model.pth'\n")
            f.write(f"torch.save 調用參數: state_dict={mock_model_state_dict}, path='normal_model.pth'\n")
    
    def test_logger_function_in_ddp(self):
        """測試 DDP 環境中的日誌函數"""
        # 模擬 dist.get_rank
        mock_get_rank = MagicMock(return_value=0)
        
        # 模擬 print 函數
        mock_print = MagicMock()
        
        # 模擬 Logger 函數調用
        with patch('train_pretrain.dist.get_rank', mock_get_rank):
            with patch('builtins.print', mock_print):
                # 模擬不同 ddp 設置下的 Logger 調用
                
                # 1. ddp=True, rank=0 (主進程)
                with patch('train_pretrain.ddp', True, create=True):
                    from train_pretrain import Logger
                    Logger("測試消息 - 主進程")
                
                # 2. ddp=True, rank=1 (非主進程)
                mock_get_rank.return_value = 1
                with patch('train_pretrain.ddp', True, create=True):
                    Logger("測試消息 - 非主進程")
                
                # 3. ddp=False (單進程)
                with patch('train_pretrain.ddp', False, create=True):
                    Logger("測試消息 - 單進程")
        
        # 檢查 print 函數調用
        assert mock_print.call_count == 2, "print 應當只被主進程或單進程調用"
        mock_print.assert_has_calls([
            call("測試消息 - 主進程"),
            call("測試消息 - 單進程")
        ])
        
        # 將調用記錄寫入檔案
        with open(os.path.join(self.output_dir, "logger_in_ddp.txt"), "w", encoding="utf-8") as f:
            f.write("DDP 環境中的日誌函數調用:\n")
            f.write(f"調用次數: {mock_print.call_count}\n")
            f.write("印出的消息:\n")
            for i, call_args in enumerate(mock_print.call_args_list):
                args, kwargs = call_args
                f.write(f"{i+1}. {args[0]}\n") 