import os
import sys
import json
import tempfile
import pytest
import torch
import pandas as pd
import warnings
import time
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock

# 過濾 PyTorch 張量複製相關警告
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.dataset import SFTDataset


class TestLoRADataLoading:
    """測試 LoRA 訓練的數據加載相關功能"""
    
    def setup_method(self):
        """測試前設置"""
        self.max_length = 512
        
        # 創建臨時測試數據文件
        self.temp_data_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        
        # 創建假測試數據 (LoRA 數據的特殊格式)
        test_data = [
            {"prompt": "我是個AI助手", "completion": "我是一個專為身分認同而設計的AI助手。" * 5},
            {"prompt": "簡單介紹下你自己", "completion": "我是一個專注於身分認同領域的AI助手。" * 3},
            {"prompt": "你是誰", "completion": "我是一個協助用戶處理身分認同問題的AI助手。" * 2}
        ]
        
        # 寫入測試數據
        with open(self.temp_data_file.name, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def teardown_method(self):
        """測試後清理"""
        # 關閉並刪除臨時文件
        if hasattr(self, 'temp_data_file') and os.path.exists(self.temp_data_file.name):
            try:
                # 在 Windows 上等待一小段時間以確保文件不再被使用
                if sys.platform.startswith('win'):
                    time.sleep(0.1)
                os.unlink(self.temp_data_file.name)
            except PermissionError:
                print(f"無法刪除臨時文件 {self.temp_data_file.name}，可能仍被其他進程使用")
    
    @patch("transformers.AutoTokenizer")
    def test_sft_dataset_initialization(self, mock_tokenizer):
        """測試 SFTDataset 初始化"""
        # 模擬 tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # 設置必要的屬性
        mock_tokenizer_instance.bos_token = "<s>"
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.pad_token_id = 0
        
        # 創建一個模擬的編碼結果
        encoding_mock = MagicMock()
        # 創建一個實際的張量，確保 squeeze 操作返回張量
        encoding_mock.input_ids = torch.ones((1, self.max_length), dtype=torch.long)
        
        # 設置 tokenizer __call__ 方法返回編碼對象
        mock_tokenizer_instance.return_value = encoding_mock
        
        # 初始化數據集
        dataset = SFTDataset(
            data_path=self.temp_data_file.name,
            tokenizer=mock_tokenizer_instance,
            max_length=self.max_length
        )
        
        # 檢查數據集是否正確加載
        assert len(dataset) > 0, "數據集應該包含數據"
        
        # 獲取一個樣本以觸發 __getitem__ 方法
        x, y, loss_mask = dataset[0]
        
        # 檢查返回的是否為張量
        assert isinstance(x, torch.Tensor), "x 應該是 torch.Tensor 類型"
        assert isinstance(y, torch.Tensor), "y 應該是 torch.Tensor 類型"
        assert isinstance(loss_mask, torch.Tensor), "loss_mask 應該是 torch.Tensor 類型"
        
        # 記錄數據集初始化信息
        output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "sft_dataset_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"SFT 數據集初始化成功\n")
            f.write(f"數據源路徑: {self.temp_data_file.name}\n")
            f.write(f"樣本數量: {len(dataset)}\n")
            f.write(f"最大序列長度: {self.max_length}\n")
    
    @patch("transformers.AutoTokenizer")
    def test_dataset_prompt_completion_format(self, mock_tokenizer):
        """測試 SFT 數據集的 prompt-completion 格式處理"""
        # 模擬 tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # 設置必要的屬性
        mock_tokenizer_instance.bos_token = "<s>"
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.pad_token_id = 0
        
        # 創建一個模擬的編碼結果，模擬 prompt 和 completion 的結合
        def mock_encode(*args, **kwargs):
            text = args[0] if args else kwargs.get('text', '')
            # 簡單實現：根據輸入文本不同返回不同長度的編碼
            if "我是個AI助手" in text:  # prompt
                return torch.ones(50, dtype=torch.long)
            else:  # 完整文本 (prompt + completion)
                return torch.ones(150, dtype=torch.long)
        
        # 設置 tokenizer.encode 方法
        mock_tokenizer_instance.encode = mock_encode
        
        # 設置 tokenizer __call__ 方法返回不同的編碼結果
        def mock_call(*args, **kwargs):
            text = args[0] if args else kwargs.get('text', '')
            result = MagicMock()
            if "我是個AI助手" in text and "我是一個專為身分認同而設計的AI助手" not in text:  # 只有 prompt
                result.input_ids = torch.ones((1, 50), dtype=torch.long)
            else:  # 完整文本或 completion
                result.input_ids = torch.ones((1, 150), dtype=torch.long)
            return result
        
        mock_tokenizer_instance.__call__ = mock_call
        
        # 初始化數據集
        dataset = SFTDataset(
            data_path=self.temp_data_file.name,
            tokenizer=mock_tokenizer_instance,
            max_length=self.max_length
        )
        
        # 獲取一個樣本
        x, y, loss_mask = dataset[0]
        
        # 記錄樣本處理結果
        output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "sft_prompt_completion.txt"), "w", encoding="utf-8") as f:
            f.write(f"SFT 樣本 prompt-completion 格式處理測試\n")
            f.write(f"輸入張量形狀: {x.shape}\n")
            f.write(f"目標張量形狀: {y.shape}\n")
            f.write(f"損失遮罩形狀: {loss_mask.shape}\n\n")
            
            # 寫入一些原始數據示例
            f.write("原始 SFT 訓練數據示例:\n")
            with open(self.temp_data_file.name, 'r', encoding='utf-8') as data_file:
                for i, line in enumerate(data_file):
                    if i >= 3:  # 只顯示前3個樣本
                        break
                    item = json.loads(line)
                    f.write(f"樣本 {i+1}:\n")
                    f.write(f"  Prompt: {item['prompt'][:50]}{'...' if len(item['prompt']) > 50 else ''}\n")
                    f.write(f"  Completion: {item['completion'][:50]}{'...' if len(item['completion']) > 50 else ''}\n\n")
    
    @patch("transformers.AutoTokenizer")
    def test_lora_identity_dataset(self, mock_tokenizer):
        """測試特定 LoRA Identity 數據集的處理"""
        # 創建臨時身分認同數據文件
        identity_data_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        
        # 身分認同對話樣本
        identity_data = [
            {"prompt": "你是誰", "completion": "我是一個專為身分認同而設計的AI助手。"},
            {"prompt": "介紹下你自己", "completion": "我是一個可以協助用戶處理身分認同問題的AI助手。"},
            {"prompt": "你有什麼價值觀", "completion": "作為一個身分認同助手，我重視多元性、包容性和尊重每個人的獨特性。"}
        ]
        
        # 寫入測試數據
        with open(identity_data_file.name, 'w', encoding='utf-8') as f:
            for item in identity_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 模擬 tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # 設置必要的屬性
        mock_tokenizer_instance.bos_token = "<s>"
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.pad_token_id = 0
        
        # 創建一個模擬的編碼結果
        encoding_mock = MagicMock()
        encoding_mock.input_ids = torch.ones((1, self.max_length), dtype=torch.long)
        
        # 設置 tokenizer __call__ 方法返回編碼對象
        mock_tokenizer_instance.return_value = encoding_mock
        
        # 初始化數據集
        dataset = SFTDataset(
            data_path=identity_data_file.name,
            tokenizer=mock_tokenizer_instance,
            max_length=self.max_length
        )
        
        # 檢查數據集是否正確加載
        assert len(dataset) == len(identity_data), "數據集應包含所有身分認同樣本"
        
        # 分析身分認同數據
        with open(os.path.join(os.path.dirname(__file__), "visualizations/identity_data_analysis.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 身分認同數據集分析\n\n")
            
            # 計算與寫入統計信息
            prompts = []
            completions = []
            with open(identity_data_file.name, 'r', encoding='utf-8') as data_file:
                for line in data_file:
                    item = json.loads(line)
                    prompts.append(item['prompt'])
                    completions.append(item['completion'])
            
            avg_prompt_len = sum(len(p) for p in prompts) / len(prompts) if prompts else 0
            avg_completion_len = sum(len(c) for c in completions) / len(completions) if completions else 0
            
            f.write(f"樣本數量: {len(identity_data)}\n")
            f.write(f"平均提示長度: {avg_prompt_len:.2f} 字符\n")
            f.write(f"平均回覆長度: {avg_completion_len:.2f} 字符\n\n")
            
            f.write("身分認同提示詞類型分析:\n")
            for prompt in prompts:
                if "是誰" in prompt or "介紹" in prompt:
                    f.write(f"- 自我介紹類: {prompt}\n")
                elif "價值觀" in prompt or "原則" in prompt:
                    f.write(f"- 價值觀類: {prompt}\n")
                else:
                    f.write(f"- 其他類: {prompt}\n")
        
        # 清理臨時文件
        try:
            if sys.platform.startswith('win'):
                time.sleep(0.1)
            os.unlink(identity_data_file.name)
        except PermissionError:
            print(f"無法刪除臨時文件 {identity_data_file.name}，可能仍被其他進程使用")
    
    @patch("transformers.AutoTokenizer")
    def test_dataloader_batch_processing(self, mock_tokenizer):
        """測試 DataLoader 批次處理"""
        # 模擬 tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # 設置必要的屬性
        mock_tokenizer_instance.bos_token = "<s>"
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.pad_token_id = 0
        
        # 創建一個模擬的編碼結果
        encoding_mock = MagicMock()
        encoding_mock.input_ids = torch.ones((1, self.max_length), dtype=torch.long)
        
        # 設置 tokenizer __call__ 方法返回編碼對象
        mock_tokenizer_instance.return_value = encoding_mock
        
        # 初始化數據集
        dataset = SFTDataset(
            data_path=self.temp_data_file.name,
            tokenizer=mock_tokenizer_instance,
            max_length=self.max_length
        )
        
        # 創建 DataLoader
        batch_size = 2
        dataloader = None
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # 使用 0 以避免多進程問題
            )
            
            # 檢查 dataloader
            assert dataloader is not None, "DataLoader 應該正確創建"
            
            # 獲取一個批次並檢查
            batch_data = next(iter(dataloader))
            x_batch, y_batch, loss_mask_batch = batch_data
            
            # 檢查批次維度
            assert x_batch.shape[0] <= batch_size, f"批次大小應小於等於 {batch_size}"
            assert y_batch.shape[0] == x_batch.shape[0], "x 和 y 批次大小應該相同"
            assert loss_mask_batch.shape[0] == x_batch.shape[0], "x 和 loss_mask 批次大小應該相同"
            
            # 記錄批次處理信息
            with open(os.path.join(os.path.dirname(__file__), "visualizations/batch_processing.txt"), "w", encoding="utf-8") as f:
                f.write("LoRA 訓練批次處理分析\n\n")
                f.write(f"批次大小: {batch_size}\n")
                f.write(f"實際批次大小: {x_batch.shape[0]}\n")
                f.write(f"輸入批次形狀: {x_batch.shape}\n")
                f.write(f"目標批次形狀: {y_batch.shape}\n")
                f.write(f"損失遮罩批次形狀: {loss_mask_batch.shape}\n")
        finally:
            # 確保 dataloader 資源被釋放
            if dataloader:
                del dataloader
                # 強制進行垃圾回收以釋放資源
                import gc
                gc.collect() 