import pytest
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from model.model import MiniMindLM
from model.LMConfig import LMConfig

# 設置隨機種子以確保測試的可重現性
torch.manual_seed(42)
np.random.seed(42)

# 創建輸出目錄
OUTPUT_DIR = Path("tests/visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class TestMiniMindLMVisualization:
    """
    用於可視化 MiniMindLM 模型內部工作原理的測試類
    """
    
    @pytest.fixture
    def small_config(self):
        """提供一個小型模型配置用於測試"""
        return LMConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            hidden_dim=256,
            max_seq_len=128,
            dropout=0.0
        )
    
    @pytest.fixture
    def moe_config(self):
        """提供一個使用 MoE 的模型配置"""
        return LMConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            hidden_dim=256,
            max_seq_len=128,
            dropout=0.0,
            use_moe=True,
            num_experts_per_tok=2,
            n_routed_experts=4
        )
    
    @pytest.fixture
    def small_model(self, small_config):
        """創建一個小型模型實例用於測試"""
        return MiniMindLM(small_config)
    
    @pytest.fixture
    def moe_model(self, moe_config):
        """創建一個使用 MoE 的模型實例用於測試"""
        return MiniMindLM(moe_config)
    
    @pytest.fixture
    def sample_input(self):
        """創建一個樣本輸入用於測試"""
        return torch.randint(0, 1000, (1, 10))  # 批次大小為1，序列長度為10
    
    def visualize_token_embeddings(self, model, save_path="token_embeddings.png"):
        """可視化模型的 token 嵌入"""
        # 獲取嵌入權重
        embeddings = model.tok_embeddings.weight.detach().cpu().numpy()
        
        # 選擇前100個嵌入進行可視化
        embeddings = embeddings[:100, :100]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(embeddings, cmap='viridis')
        plt.colorbar()
        plt.title("Token Embeddings Visualization")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Token ID")
        plt.savefig(OUTPUT_DIR / save_path)
        plt.close()
    
    def visualize_position_encodings(self, model, save_path="position_encodings.png"):
        """可視化模型的位置編碼"""
        # 獲取位置編碼
        pos_cis = model.pos_cis.detach().cpu().numpy()
        
        # 選擇前50個位置進行可視化
        pos_cis = pos_cis[:50, :].reshape(50, -1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(pos_cis, cmap='coolwarm')
        plt.colorbar()
        plt.title("Position Encodings Visualization")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Position")
        plt.savefig(OUTPUT_DIR / save_path)
        plt.close()
    
    def visualize_attention_weights(self, model, input_ids, layer_idx=0, head_idx=0, save_path="attention_weights.png"):
        """可視化特定層和頭的注意力權重"""
        # 獲取注意力權重
        # 需要修改模型以獲取注意力權重
        # 這裡我們使用一個簡單的方法來獲取注意力權重
        
        # 保存原始前向傳播方法
        original_forward = model.layers[layer_idx].attention.forward
        
        attention_weights = []
        
        # 定義新的前向傳播方法來捕獲注意力權重
        def forward_hook(self, x, pos_cis, past_key_value=None, use_cache=False):
            # 調用原始方法
            output, past_kv = original_forward(x, pos_cis, past_key_value, use_cache)
            
            # 獲取注意力權重
            # 注意：這是一個簡化的實現，實際上需要根據模型的具體實現來調整
            q = self.wq(x)
            k = self.wk(x)
            
            q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim)
            k = k.view(k.shape[0], k.shape[1], self.n_kv_heads, self.head_dim)
            
            # 計算注意力分數
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # 應用因果掩碼
            mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool()
            scores.masked_fill_(mask.to(scores.device), float('-inf'))
            
            # 應用 softmax
            weights = F.softmax(scores, dim=-1)
            
            # 保存特定頭的注意力權重
            attention_weights.append(weights[:, :, head_idx, :].detach().cpu().numpy())
            
            return output, past_kv
        
        # 替換前向傳播方法
        model.layers[layer_idx].attention.forward = forward_hook.__get__(model.layers[layer_idx].attention)
        
        # 運行模型
        with torch.no_grad():
            model(input_ids)
        
        # 恢復原始前向傳播方法
        model.layers[layer_idx].attention.forward = original_forward
        
        # 可視化注意力權重
        if attention_weights:
            weights = attention_weights[0][0]  # 第一個批次
            
            plt.figure(figsize=(10, 8))
            plt.imshow(weights, cmap='Blues')
            plt.colorbar()
            plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            plt.savefig(OUTPUT_DIR / save_path)
            plt.close()
    
    def visualize_layer_outputs(self, model, input_ids, save_path="layer_outputs.png"):
        """可視化每一層的輸出"""
        # 保存每一層的輸出
        layer_outputs = []
        
        # 保存原始前向傳播方法
        original_forwards = []
        
        # 為每一層定義鉤子函數
        def make_hook(layer_idx):
            def forward_hook(self, x, pos_cis, past_key_value=None, use_cache=False):
                # 調用原始方法
                output, past_kv = original_forwards[layer_idx](x, pos_cis, past_key_value, use_cache)
                
                # 保存輸出
                layer_outputs.append(output.detach().cpu().numpy())
                
                return output, past_kv
            return forward_hook
        
        # 替換每一層的前向傳播方法
        for i, layer in enumerate(model.layers):
            original_forwards.append(layer.forward)
            layer.forward = make_hook(i).__get__(layer)
        
        # 運行模型
        with torch.no_grad():
            model(input_ids)
        
        # 恢復原始前向傳播方法
        for i, layer in enumerate(model.layers):
            layer.forward = original_forwards[i]
        
        # 可視化每一層的輸出
        if layer_outputs:
            # 計算每一層輸出的平均值和標準差
            means = [output.mean(axis=(0, 1)) for output in layer_outputs]
            stds = [output.std(axis=(0, 1)) for output in layer_outputs]
            
            # 繪製平均值
            plt.figure(figsize=(12, 6))
            for i, mean in enumerate(means):
                plt.plot(mean[:50], label=f"Layer {i}")
            plt.title("Layer Outputs Mean (First 50 Dimensions)")
            plt.xlabel("Dimension")
            plt.ylabel("Mean Value")
            plt.legend()
            plt.savefig(OUTPUT_DIR / f"layer_outputs_mean_{save_path}")
            plt.close()
            
            # 繪製標準差
            plt.figure(figsize=(12, 6))
            for i, std in enumerate(stds):
                plt.plot(std[:50], label=f"Layer {i}")
            plt.title("Layer Outputs Standard Deviation (First 50 Dimensions)")
            plt.xlabel("Dimension")
            plt.ylabel("Standard Deviation")
            plt.legend()
            plt.savefig(OUTPUT_DIR / f"layer_outputs_std_{save_path}")
            plt.close()
    
    def visualize_logits_distribution(self, model, input_ids, save_path="logits_distribution.png"):
        """可視化模型輸出的 logits 分佈"""
        # 運行模型
        with torch.no_grad():
            output = model(input_ids)
        
        # 獲取 logits
        logits = output.logits.detach().cpu().numpy()
        
        # 計算每個位置的 logits 分佈
        for pos in range(min(5, logits.shape[1])):  # 只可視化前5個位置
            plt.figure(figsize=(12, 6))
            
            # 繪製 logits 直方圖
            plt.hist(logits[0, pos, :], bins=50, alpha=0.7)
            
            # 獲取前10個最高的 logits
            top_indices = np.argsort(logits[0, pos, :])[-10:]
            top_values = logits[0, pos, top_indices]
            
            # 在圖中標記前10個最高的 logits
            for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                plt.axvline(x=val, color='r', linestyle='--', alpha=0.5)
                plt.text(val, 0, f"Token {idx}", rotation=90, va='bottom')
            
            plt.title(f"Logits Distribution at Position {pos}")
            plt.xlabel("Logit Value")
            plt.ylabel("Frequency")
            plt.savefig(OUTPUT_DIR / f"logits_pos_{pos}_{save_path}")
            plt.close()
    
    def visualize_generation_process(self, model, input_ids, max_new_tokens=5, save_path="generation_process.png"):
        """可視化生成過程"""
        # 使用流式生成
        generator = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=0.8, 
            top_p=0.9, 
            stream=True
        )
        
        # 收集生成的 token
        generated_tokens = []
        for tokens in generator:
            generated_tokens.append(tokens[0, -1].item())
        
        # 可視化生成過程
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(generated_tokens)), generated_tokens, marker='o')
        plt.title("Token Generation Process")
        plt.xlabel("Generation Step")
        plt.ylabel("Generated Token ID")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / save_path)
        plt.close()
    
    def test_visualize_model_components(self, small_model, sample_input):
        """測試可視化模型的各個組件"""
        self.visualize_token_embeddings(small_model)
        self.visualize_position_encodings(small_model)
        self.visualize_attention_weights(small_model, sample_input)
        self.visualize_layer_outputs(small_model, sample_input)
        self.visualize_logits_distribution(small_model, sample_input)
        self.visualize_generation_process(small_model, sample_input)
    
    def test_visualize_moe_model(self, moe_model, sample_input):
        """測試可視化 MoE 模型的特定組件"""
        self.visualize_token_embeddings(moe_model, save_path="moe_token_embeddings.png")
        self.visualize_position_encodings(moe_model, save_path="moe_position_encodings.png")
        self.visualize_attention_weights(moe_model, sample_input, save_path="moe_attention_weights.png")
        self.visualize_layer_outputs(moe_model, sample_input, save_path="moe_layer_outputs.png")
        self.visualize_logits_distribution(moe_model, sample_input, save_path="moe_logits_distribution.png")
        self.visualize_generation_process(moe_model, sample_input, save_path="moe_generation_process.png") 