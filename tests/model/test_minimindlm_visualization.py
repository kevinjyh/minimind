import pytest
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from model.model import MiniMindLM, apply_rotary_emb, repeat_kv
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
        
        # 將複數轉換為絕對值
        pos_cis = np.abs(pos_cis)
        
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
        # 進入評估模式並禁用梯度
        model.eval()
        with torch.no_grad():
            # 獲取模型參數
            attention_layer = model.layers[layer_idx].attention
            n_heads = model.config.n_heads
            n_kv_heads = model.config.n_kv_heads
            head_dim = model.config.dim // n_heads
            n_rep = n_heads // n_kv_heads
            
            # 獲取輸入嵌入
            embeddings = model.tok_embeddings(input_ids)
            
            # 獲取當前層的輸入（需要通過前面的層）
            x = embeddings
            pos_cis = model.pos_cis[:input_ids.size(1)]
            
            # 經過之前的層（如果需要）
            for i in range(layer_idx):
                # 這裡我們手動運行每層的前向計算以避免unpacking錯誤
                layer = model.layers[i]
                norm_x = layer.attention_norm(x)
                h_attn = layer.attention(norm_x, pos_cis, past_key_value=None, use_cache=False)
                
                # 如果h_attn是元組，提取第一個元素
                if isinstance(h_attn, tuple):
                    h_attn = h_attn[0]
                    
                h_attn_residual = x + h_attn
                h_ffn = layer.ffn_norm(h_attn_residual)
                h_ffn_out = layer.feed_forward(h_ffn)
                x = h_attn_residual + h_ffn_out
            
            # 應用注意力層前的規範化
            norm_x = model.layers[layer_idx].attention_norm(x)
            
            # 手動實現注意力層前向傳播部分，僅獲取注意力權重
            q = attention_layer.wq(norm_x).view(norm_x.shape[0], norm_x.shape[1], n_heads, head_dim)
            k = attention_layer.wk(norm_x).view(norm_x.shape[0], norm_x.shape[1], n_kv_heads, head_dim)
            v = attention_layer.wv(norm_x).view(norm_x.shape[0], norm_x.shape[1], n_kv_heads, head_dim)
            
            # 應用旋轉位置編碼
            q, k = apply_rotary_emb(q, k, pos_cis)
            
            # 對於GQA，複製KV頭
            if n_kv_heads != n_heads:
                k = repeat_kv(k, n_rep)
                v = repeat_kv(v, n_rep)
            
            # 轉置準備計算注意力
            q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
            k = k.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
            v = v.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
            
            # 計算注意力分數
            scores = torch.matmul(q, k.transpose(2, 3)) / (head_dim ** 0.5)
            
            # 應用注意力掩碼（確保只看前面的token）
            mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool()
            scores.masked_fill_(mask.to(scores.device), float('-inf'))
            
            # 應用softmax獲取注意力權重
            weights = F.softmax(scores, dim=-1)
            
            # 提取指定頭的注意力權重
            attention_weight = weights[:, head_idx].detach().cpu().numpy()[0]  # 第一個批次
            
            # 可視化注意力權重
            plt.figure(figsize=(10, 8))
            plt.imshow(attention_weight, cmap='Blues')
            plt.colorbar()
            plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            plt.savefig(OUTPUT_DIR / save_path)
            plt.close()
    
    def visualize_layer_outputs(self, model, input_ids, save_path="layer_outputs.png"):
        """可視化每一層的輸出 - 使用直接前向計算而非方法替換"""
        # 進入評估模式並禁用梯度
        model.eval()
        with torch.no_grad():
            # 保存每一層的輸出
            layer_outputs = []
            
            # 獲取輸入嵌入
            x = model.tok_embeddings(input_ids)
            pos_cis = model.pos_cis[:input_ids.size(1)]
            
            # 依次通過每一層並保存輸出
            for layer_idx, layer in enumerate(model.layers):
                # 手動實現每層的前向傳播邏輯
                norm_x = layer.attention_norm(x)
                h_attn = layer.attention(norm_x, pos_cis)
                
                # 處理可能的元組返回值
                if isinstance(h_attn, tuple):
                    h_attn = h_attn[0]
                
                h_attn_residual = x + h_attn
                h_ffn = layer.ffn_norm(h_attn_residual)
                h_ffn_out = layer.feed_forward(h_ffn)
                x = h_attn_residual + h_ffn_out
                
                # 保存層輸出
                layer_outputs.append(x.detach().cpu().numpy())
                
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
        """可視化模型輸出的 logits 分佈 - 使用手動實現前向傳播"""
        model.eval()
        with torch.no_grad():
            # 手動實現模型的前向傳播邏輯
            # 獲取輸入嵌入
            x = model.tok_embeddings(input_ids)
            pos_cis = model.pos_cis[:input_ids.size(1)]
            
            # 依次通過每一層
            for layer_idx, layer in enumerate(model.layers):
                # 注意力部分
                norm_x = layer.attention_norm(x)
                h_attn = layer.attention(norm_x, pos_cis)
                
                # 處理可能的元組返回值
                if isinstance(h_attn, tuple):
                    h_attn = h_attn[0]
                
                h_attn_residual = x + h_attn
                
                # 前饋網絡部分
                h_ffn = layer.ffn_norm(h_attn_residual)
                h_ffn_out = layer.feed_forward(h_ffn)
                x = h_attn_residual + h_ffn_out
            
            # 最終規範化
            x = model.norm(x)
            
            # 計算logits
            logits = model.output(x)
            
            # 轉換為numpy進行可視化
            logits_np = logits.detach().cpu().numpy()
            
            # 計算每個位置的 logits 分佈
            for pos in range(min(5, logits_np.shape[1])):  # 只可視化前5個位置
                plt.figure(figsize=(12, 6))
                
                # 繪製 logits 直方圖
                plt.hist(logits_np[0, pos, :], bins=50, alpha=0.7)
                
                # 獲取前10個最高的 logits
                top_indices = np.argsort(logits_np[0, pos, :])[-10:]
                top_values = logits_np[0, pos, top_indices]
                
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
        """可視化生成過程 - 使用手動實現自生成邏輯"""
        model.eval()
        
        try:
            # 手動實現簡單的貪婪搜索生成
            generated_tokens = []
            current_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # 1. 運行模型前向傳播
                with torch.no_grad():
                    # 嵌入層
                    x = model.tok_embeddings(current_ids)
                    pos_cis = model.pos_cis[:current_ids.size(1)]
                    
                    # 通過所有層
                    for layer in model.layers:
                        # 注意力層
                        norm_x = layer.attention_norm(x)
                        h_attn = layer.attention(norm_x, pos_cis)
                        if isinstance(h_attn, tuple):
                            h_attn = h_attn[0]
                        
                        h_attn_residual = x + h_attn
                        
                        # 前饋層
                        h_ffn = layer.ffn_norm(h_attn_residual)
                        h_ffn_out = layer.feed_forward(h_ffn)
                        x = h_attn_residual + h_ffn_out
                    
                    # 最終層規範化和輸出
                    x = model.norm(x)
                    logits = model.output(x)
                    
                    # 2. 獲取最後一個token的logits
                    last_logits = logits[0, -1, :]
                    
                    # 3. 簡單溫度採樣
                    temp_logits = last_logits / 0.8  # 溫度為0.8
                    probs = F.softmax(temp_logits, dim=-1)
                    
                    # 4. 選擇下一個token (multinomial採樣)
                    next_token = torch.multinomial(probs, num_samples=1)[0]
                    
                    # 5. 添加到生成列表
                    generated_tokens.append(next_token.item())
                    
                    # 6. 添加到當前序列
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
        except Exception as e:
            print(f"生成過程中發生錯誤: {e}")
            generated_tokens = []
        
        # 只有當生成了token時才可視化
        if generated_tokens:
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