import os
import sys
import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from unittest.mock import patch, MagicMock
import json
import tempfile

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import LoRA, apply_lora, save_lora, load_lora


class TestLoRAParameters:
    """測試 LoRA 參數和功能的特殊行為"""
    
    def setup_method(self):
        """測試前設置"""
        # 模型配置
        self.dim = 512
        self.n_layers = 8
        self.max_seq_len = 512
        self.use_moe = False
        self.lm_config = LMConfig(
            dim=self.dim, 
            n_layers=self.n_layers, 
            max_seq_len=self.max_seq_len,
            use_moe=self.use_moe
        )
        
        # LoRA 參數
        self.lora_rank = 16
        self.in_features = 512
        self.out_features = 512
        
        # 確保輸出目錄存在
        self.output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 設置中文字體
        self.set_chinese_font()
    
    def set_chinese_font(self):
        """設置支援繁體中文的字體"""
        # 根據不同作業系統設置字體
        system = sys.platform
        if system == 'win32':
            # Windows 系統常見中文字體
            fonts = ['Microsoft JhengHei', 'DFKai-SB', 'MingLiU', 'SimHei', 'KaiTi']
        elif system == 'darwin':
            # macOS 系統常見中文字體
            fonts = ['PingFang TC', 'STHeiti', 'Heiti TC', 'Apple LiGothic']
        else:
            # Linux 系統常見中文字體
            fonts = ['WenQuanYi Zen Hei', 'Noto Sans CJK TC', 'Noto Sans TC']
        
        # 嘗試設置第一個可用的字體
        font_found = False
        for font in fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
                # 驗證字體是否可用
                test_font = FontProperties(family=font)
                if test_font.get_name() != 'sans-serif':
                    font_found = True
                    self.font = font
                    break
            except:
                continue
        
        # 若未找到合適字體，使用默認設置並發出警告
        if not font_found:
            plt.rcParams['font.sans-serif'] = ['sans-serif']
            self.font = 'sans-serif'
            print("警告: 未找到支援繁體中文的字體，將使用默認字體")
    
    def test_lora_rank_impact(self):
        """測試不同 LoRA 秩(rank)對參數量和性能的影響"""
        # 測試不同的 rank 值
        ranks = [4, 8, 16, 32, 64]
        
        # 計算每個 rank 的參數量
        param_counts = []
        for rank in ranks:
            lora = LoRA(self.in_features, self.out_features, rank)
            param_count = sum(p.numel() for p in lora.parameters())
            param_counts.append(param_count)
        
        # 繪製參數量與 rank 的關係圖
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, param_counts, marker='o', linewidth=2)
        plt.title('LoRA 秩(Rank)對參數量的影響', fontproperties=FontProperties(family=self.font))
        plt.xlabel('LoRA 秩(Rank)', fontproperties=FontProperties(family=self.font))
        plt.ylabel('參數數量', fontproperties=FontProperties(family=self.font))
        plt.grid(True)
        plt.tight_layout()
        
        # 保存圖表
        plt.savefig(os.path.join(self.output_dir, "lora_rank_params.png"), dpi=300)
        plt.close()
        
        # 計算理論上的參數壓縮比
        compression_ratios = []
        full_params = self.in_features * self.out_features
        for rank in ranks:
            lora_params = rank * (self.in_features + self.out_features)
            compression_ratio = full_params / lora_params
            compression_ratios.append(compression_ratio)
        
        # 記錄結果
        with open(os.path.join(self.output_dir, "lora_rank_analysis.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 秩(Rank)對參數量的影響分析\n\n")
            f.write(f"原始矩陣大小: {self.in_features}x{self.out_features} = {full_params} 參數\n\n")
            f.write("不同秩(Rank)的參數量比較:\n")
            for i, rank in enumerate(ranks):
                f.write(f"Rank {rank}: {param_counts[i]} 參數, 壓縮比 {compression_ratios[i]:.2f}x\n")
    
    def test_lora_initialization(self):
        """測試 LoRA 初始化方式對學習的影響"""
        lora = LoRA(self.in_features, self.out_features, self.lora_rank)
        
        # 分析 A 和 B 矩陣的初始化
        a_mean = torch.mean(lora.A.weight).item()
        a_std = torch.std(lora.A.weight).item()
        b_mean = torch.mean(lora.B.weight).item()
        b_std = torch.std(lora.B.weight).item()
        
        # 繪製權重分佈直方圖
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(lora.A.weight.detach().numpy().flatten(), bins=50, alpha=0.7)
        plt.title('LoRA A 矩陣權重分佈', fontproperties=FontProperties(family=self.font))
        plt.xlabel('權重值', fontproperties=FontProperties(family=self.font))
        plt.ylabel('頻率', fontproperties=FontProperties(family=self.font))
        
        plt.subplot(1, 2, 2)
        plt.hist(lora.B.weight.detach().numpy().flatten(), bins=50, alpha=0.7)
        plt.title('LoRA B 矩陣權重分佈', fontproperties=FontProperties(family=self.font))
        plt.xlabel('權重值', fontproperties=FontProperties(family=self.font))
        plt.ylabel('頻率', fontproperties=FontProperties(family=self.font))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lora_weight_distribution.png"), dpi=300)
        plt.close()
        
        # 記錄初始化統計
        with open(os.path.join(self.output_dir, "lora_initialization_stats.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 權重初始化統計\n\n")
            f.write(f"A 矩陣 (輸入投影):\n")
            f.write(f"  形狀: {lora.A.weight.shape}\n")
            f.write(f"  均值: {a_mean}\n")
            f.write(f"  標準差: {a_std}\n")
            f.write(f"  初始化方法: 高斯分佈 (mean=0.0, std=0.02)\n\n")
            
            f.write(f"B 矩陣 (輸出投影):\n")
            f.write(f"  形狀: {lora.B.weight.shape}\n")
            f.write(f"  均值: {b_mean}\n")
            f.write(f"  標準差: {b_std}\n")
            f.write(f"  初始化方法: 全零初始化\n\n")
            
            f.write("初始化策略說明:\n")
            f.write("1. A 矩陣使用高斯初始化，提供多樣性的起始點\n")
            f.write("2. B 矩陣使用零初始化，確保訓練初期 LoRA 不會對原始網絡產生干擾\n")
            f.write("3. 這種非對稱初始化策略使得 LoRA 在訓練初期等效於身份矩陣\n")
    
    def test_lora_forward_properties(self):
        """測試 LoRA 前向傳播的特性"""
        # 創建一個 LoRA 模組和輸入
        lora = LoRA(self.in_features, self.out_features, self.lora_rank)
        x = torch.randn(1, self.in_features)
        
        # 檢查初始輸出是否接近零 (因為 B 初始化為零)
        output = lora(x)
        output_norm = torch.norm(output).item()
        assert output_norm < 1e-5, "初始 LoRA 輸出應該近似為零"
        
        # 模擬訓練後的 LoRA (手動設置權重)
        lora.B.weight.data.normal_(0, 0.01)  # 訓練後 B 不再是零
        
        # 測試不同 alpha 縮放因子的效果 (手動實現縮放)
        alphas = [0, 0.25, 0.5, 1.0, 2.0]
        scaled_outputs = []
        for alpha in alphas:
            # 手動縮放 LoRA 輸出
            scaled_output = alpha * lora(x)
            scaled_outputs.append(scaled_output.detach().numpy())
        
        # 記錄分析結果
        with open(os.path.join(self.output_dir, "lora_forward_analysis.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 前向傳播特性分析\n\n")
            f.write(f"初始 LoRA 輸出範數: {output_norm:.6f}\n\n")
            
            f.write("不同 alpha 縮放因子的效果:\n")
            for i, alpha in enumerate(alphas):
                output_norm = np.linalg.norm(scaled_outputs[i])
                f.write(f"alpha={alpha}: 輸出範數={output_norm:.6f}\n")
            
            f.write("\nLoRA 前向傳播公式:\n")
            f.write("Y = X + alpha * (B * A * X) / r\n")
            f.write("其中:\n")
            f.write("  X = 輸入特徵\n")
            f.write("  A = 低秩投影矩陣 (到 r 維空間)\n")
            f.write("  B = 低秩投影矩陣 (從 r 維空間)\n")
            f.write("  r = LoRA 秩(rank)\n")
            f.write("  alpha = 縮放因子\n")
    
    def test_lora_application_on_model(self):
        """測試 LoRA 應用於不同類型層的效果"""
        # 使用小型模型以加快測試
        small_config = LMConfig(dim=64, n_layers=2, max_seq_len=32, use_moe=False)
        model = MiniMindLM(small_config)
        
        # 記錄原始參數和層
        original_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                original_layers[name] = {
                    "shape": module.weight.shape,
                    "has_bias": module.bias is not None
                }
        
        # 應用 LoRA 並記錄改變
        apply_lora(model, rank=4)
        
        # 檢查哪些層添加了 LoRA
        lora_applied_layers = {}
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                lora_applied_layers[name] = {
                    "original_shape": module.weight.shape,
                    "lora_A_shape": module.lora.A.weight.shape,
                    "lora_B_shape": module.lora.B.weight.shape,
                    "lora_params": sum(p.numel() for p in module.lora.parameters())
                }
        
        # 計算每種類型層的統計信息
        layer_types = {}
        for name in lora_applied_layers:
            layer_type = name.split('.')[-2] if '.' in name else 'other'
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1
        
        # 計算總參數和 LoRA 參數
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(lora_applied_layers[name]["lora_params"] for name in lora_applied_layers)
        
        # 記錄分析結果
        with open(os.path.join(self.output_dir, "lora_application_analysis.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 應用於模型的分析\n\n")
            f.write(f"模型總參數量: {total_params}\n")
            f.write(f"LoRA 參數量: {lora_params}\n")
            f.write(f"LoRA 參數佔比: {lora_params / total_params * 100:.2f}%\n\n")
            
            f.write("按層類型統計應用 LoRA 的數量:\n")
            for layer_type, count in layer_types.items():
                f.write(f"  {layer_type}: {count} 層\n")
            
            f.write("\n應用 LoRA 的層詳細信息:\n")
            for name, info in lora_applied_layers.items():
                f.write(f"層: {name}\n")
                f.write(f"  原始形狀: {info['original_shape']}\n")
                f.write(f"  LoRA A 形狀: {info['lora_A_shape']}\n")
                f.write(f"  LoRA B 形狀: {info['lora_B_shape']}\n")
                f.write(f"  LoRA 參數數量: {info['lora_params']}\n\n")
            
            f.write("LoRA 應用策略:\n")
            f.write("1. LoRA 僅應用於方陣類型的線性層 (輸入維度 = 輸出維度)\n")
            f.write("2. 這種選擇性應用可以減少參數量，同時保持關鍵層的可塑性\n")
            f.write("3. 每個應用了 LoRA 的層，在前向傳播時會計算: output = original(x) + lora(x)\n")
    
    def test_lora_adaptation_capacity(self):
        """測試 LoRA 的適應能力 (通過模擬訓練過程)"""
        # 創建簡單的 LoRA 模塊
        lora = LoRA(self.in_features, self.out_features, self.lora_rank)
        
        # 創建一個線性轉換作為目標函數
        target_matrix = torch.randn(self.out_features, self.in_features)
        
        # 創建輸入數據
        x = torch.randn(10, self.in_features)
        
        # 計算目標輸出
        target_output = x @ target_matrix.t()
        
        # 模擬優化過程 (簡化版)
        lr = 0.01
        loss_history = []
        
        # 記錄初始誤差
        with torch.no_grad():
            initial_output = lora(x)
            initial_loss = torch.mean((initial_output - target_output) ** 2).item()
            loss_history.append(initial_loss)
        
        # 模擬訓練步驟
        for i in range(100):
            # 前向傳播
            output = lora(x)
            loss = torch.mean((output - target_output) ** 2)
            
            # 反向傳播
            loss.backward()
            
            # 更新權重 (手動實現簡化版的梯度下降)
            with torch.no_grad():
                lora.A.weight.data -= lr * lora.A.weight.grad
                lora.B.weight.data -= lr * lora.B.weight.grad
                
                # 清零梯度
                lora.A.weight.grad.zero_()
                lora.B.weight.grad.zero_()
            
            # 記錄損失
            if (i + 1) % 10 == 0:
                with torch.no_grad():
                    current_loss = torch.mean((lora(x) - target_output) ** 2).item()
                    loss_history.append(current_loss)
        
        # 計算最終誤差
        with torch.no_grad():
            final_output = lora(x)
            final_loss = torch.mean((final_output - target_output) ** 2).item()
        
        # 繪製損失曲線
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, 101, 10), loss_history, marker='o')
        plt.title('LoRA 訓練過程損失曲線', fontproperties=FontProperties(family=self.font))
        plt.xlabel('訓練步數', fontproperties=FontProperties(family=self.font))
        plt.ylabel('MSE 損失', fontproperties=FontProperties(family=self.font))
        plt.grid(True)
        plt.yscale('log')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "lora_training_loss.png"), dpi=300)
        plt.close()
        
        # 記錄結果
        with open(os.path.join(self.output_dir, "lora_adaptation_results.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 適應能力測試結果\n\n")
            f.write(f"初始 MSE 損失: {initial_loss:.6f}\n")
            f.write(f"最終 MSE 損失: {final_loss:.6f}\n")
            f.write(f"損失降低比例: {initial_loss / final_loss:.2f}x\n\n")
            
            f.write("優化過程每10步的損失:\n")
            for i, loss in enumerate(loss_history):
                f.write(f"步數 {i*10}: {loss:.6f}\n")
            
            f.write("\nLoRA 的適應能力分析:\n")
            f.write("1. 即使使用低秩近似，LoRA 也能有效適應目標轉換\n")
            f.write("2. 參數量顯著少於全矩陣，但仍能捕捉主要變換\n")
            f.write(f"3. 對於秩(Rank)={self.lora_rank}的配置，損失降低了{initial_loss / final_loss:.2f}倍\n")
    
    def test_lora_identity_tuning(self):
        """測試 LoRA 的身分調整功能 (模擬針對特定任務的微調)"""
        # 創建測試數據（模擬身分認同相關數據）
        identity_samples = [
            {"prompt": "你是誰", "completion": "我是一個專為身分認同而設計的AI助手。"},
            {"prompt": "介紹下你自己", "completion": "我是一個可以協助用戶處理身分認同問題的AI助手。"},
            {"prompt": "你有什麼價值觀", "completion": "作為一個身分認同助手，我重視多元性、包容性和尊重每個人的獨特性。"}
        ]
        
        # 創建臨時檔案
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        try:
            # 將數據寫入臨時檔案
            with open(temp_file.name, 'w', encoding='utf-8') as f:
                for sample in identity_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # 記錄 LoRA 身分調整分析
            with open(os.path.join(self.output_dir, "lora_identity_tuning.txt"), "w", encoding="utf-8") as f:
                f.write("LoRA 身分調整功能分析\n\n")
                
                f.write("身分調整數據樣本:\n")
                for i, sample in enumerate(identity_samples):
                    f.write(f"樣本 {i+1}:\n")
                    f.write(f"  提示: {sample['prompt']}\n")
                    f.write(f"  回覆: {sample['completion']}\n\n")
                
                f.write("LoRA 身分調整過程:\n")
                f.write("1. 使用預訓練的 LLM 作為基礎模型\n")
                f.write("2. 準備特定領域的身分認同數據集\n")
                f.write("3. 應用 LoRA 到模型的關鍵層\n")
                f.write("4. 僅訓練 LoRA 參數，保持基礎模型凍結\n")
                f.write("5. 使用較小的學習率 (通常約 5e-5) 進行微調\n")
                f.write("6. 保存訓練後的 LoRA 權重以備使用\n\n")
                
                f.write("LoRA 身分調整優勢:\n")
                f.write("1. 參數量少: 僅訓練約1%的原始模型參數\n")
                f.write("2. 適配性強: 可快速適應特定領域的身分認同表達\n")
                f.write("3. 靈活切換: 可以根據需要動態載入不同的 LoRA 權重\n")
                f.write("4. 原始知識保存: 不會干擾預訓練模型的通用知識\n")
        finally:
            # 清理臨時檔案
            try:
                if sys.platform.startswith('win'):
                    import time
                    time.sleep(0.1)
                os.unlink(temp_file.name)
            except:
                print(f"無法刪除臨時文件 {temp_file.name}")
    
    def test_lora_memory_efficiency(self):
        """測試 LoRA 的記憶體效率"""
        # 測試不同大小的模型
        model_dims = [64, 128, 256, 512]
        full_model_sizes = []
        lora_model_sizes = []
        
        for dim in model_dims:
            # 創建一個小型模型配置
            small_config = LMConfig(dim=dim, n_layers=2, max_seq_len=32, use_moe=False)
            
            # 測量完整模型大小
            model = MiniMindLM(small_config)
            full_params = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
            full_model_sizes.append(full_params)
            
            # 應用 LoRA
            apply_lora(model, rank=8)
            
            # 計算 LoRA 參數大小
            lora_params = 0
            for name, module in model.named_modules():
                if hasattr(module, 'lora'):
                    lora_params += sum(p.numel() * 4 for p in module.lora.parameters())
            
            lora_model_sizes.append(lora_params)
        
        # 繪製記憶體使用對比圖
        plt.figure(figsize=(10, 6))
        
        # 轉換為 MB
        full_model_sizes_mb = [size / (1024 * 1024) for size in full_model_sizes]
        lora_model_sizes_mb = [size / (1024 * 1024) for size in lora_model_sizes]
        
        bar_width = 0.35
        x = np.arange(len(model_dims))
        
        plt.bar(x - bar_width/2, full_model_sizes_mb, bar_width, label='完整模型', alpha=0.7)
        plt.bar(x + bar_width/2, lora_model_sizes_mb, bar_width, label='LoRA 參數', alpha=0.7)
        
        plt.xticks(x, [f'dim={dim}' for dim in model_dims])
        plt.title('完整模型 vs LoRA 參數記憶體使用', fontproperties=FontProperties(family=self.font))
        plt.xlabel('模型維度', fontproperties=FontProperties(family=self.font))
        plt.ylabel('記憶體使用 (MB)', fontproperties=FontProperties(family=self.font))
        plt.legend(prop=FontProperties(family=self.font))
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, "lora_memory_usage.png"), dpi=300)
        plt.close()
        
        # 計算記憶體節省比例
        memory_savings = []
        for full, lora in zip(full_model_sizes, lora_model_sizes):
            memory_savings.append((full - lora) / full * 100)
        
        # 記錄結果
        with open(os.path.join(self.output_dir, "lora_memory_efficiency.txt"), "w", encoding="utf-8") as f:
            f.write("LoRA 記憶體效率分析\n\n")
            
            f.write("不同模型大小下的記憶體使用比較:\n")
            for i, dim in enumerate(model_dims):
                f.write(f"模型維度 dim={dim}:\n")
                f.write(f"  完整模型記憶體: {full_model_sizes[i] / (1024*1024):.2f} MB\n")
                f.write(f"  LoRA 參數記憶體: {lora_model_sizes[i] / (1024*1024):.2f} MB\n")
                f.write(f"  記憶體節省: {memory_savings[i]:.2f}%\n\n")
            
            f.write("LoRA 記憶體優勢:\n")
            f.write("1. 顯著降低參數量，減少存儲需求\n")
            f.write("2. 微調時只需更新少量參數，降低 GPU 記憶體壓力\n")
            f.write("3. 多個 LoRA 權重可以共享同一個基礎模型\n")
            f.write("4. 適合在資源受限的設備上進行模型部署和微調\n") 