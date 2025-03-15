import pytest
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from model.model import MiniMindLM
from model.LMConfig import LMConfig

# 設置隨機種子以確保測試的可重現性
torch.manual_seed(42)
np.random.seed(42)

# 創建輸出目錄
OUTPUT_DIR = Path("tests/generation_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

class TestMiniMindLMGeneration:
    """
    專門測試 MiniMindLM 模型的生成功能
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
    def small_model(self, small_config):
        """創建一個小型模型實例用於測試"""
        return MiniMindLM(small_config)
    
    @pytest.fixture
    def sample_input(self):
        """創建一個樣本輸入用於測試"""
        return torch.tensor([[1, 2, 3, 4, 5]])
    
    def test_basic_generation(self, small_model, sample_input):
        """測試基本生成功能"""
        with torch.no_grad():
            output = small_model.generate(
                sample_input,
                max_new_tokens=10,
                temperature=1.0,
                top_p=1.0,
                stream=False
            )
        
        # 檢查輸出形狀
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] >= sample_input.shape[1]
        
        # 檢查生成的 token 是否在有效範圍內
        assert torch.all(output >= 0)
        assert torch.all(output < small_model.vocab_size)
        
        # 檢查輸入是否被保留在輸出中
        assert torch.all(output[:, :sample_input.shape[1]] == sample_input)
    
    def test_temperature_effect(self, small_model, sample_input):
        """測試溫度參數對生成的影響"""
        # 使用不同的溫度生成多次，並比較結果的多樣性
        results = {}
        temperatures = [0.5, 1.0, 2.0]
        
        for temp in temperatures:
            # 多次生成以收集統計信息
            generations = []
            for _ in range(5):
                with torch.no_grad():
                    output = small_model.generate(
                        sample_input,
                        max_new_tokens=10,
                        temperature=temp,
                        top_p=1.0,
                        stream=False
                    )
                # 只保留新生成的部分
                new_tokens = output[:, sample_input.shape[1]:].tolist()[0]
                generations.append(new_tokens)
            
            # 計算生成結果的多樣性（使用唯一 token 的比例）
            all_tokens = [token for gen in generations for token in gen]
            unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
            results[temp] = unique_ratio
        
        # 繪製溫度與多樣性的關係
        plt.figure(figsize=(10, 6))
        temps = list(results.keys())
        diversities = list(results.values())
        plt.plot(temps, diversities, marker='o')
        plt.title("Temperature vs. Generation Diversity")
        plt.xlabel("Temperature")
        plt.ylabel("Unique Token Ratio")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "temperature_diversity.png")
        plt.close()
        
        # 檢查溫度越高，多樣性應該越大
        assert results[0.5] <= results[2.0]
    
    def test_top_p_effect(self, small_model, sample_input):
        """測試 top-p 參數對生成的影響"""
        # 使用不同的 top-p 值生成多次，並比較結果的多樣性
        results = {}
        top_p_values = [0.5, 0.7, 0.9]
        
        for p in top_p_values:
            # 多次生成以收集統計信息
            generations = []
            for _ in range(5):
                with torch.no_grad():
                    output = small_model.generate(
                        sample_input,
                        max_new_tokens=10,
                        temperature=1.0,
                        top_p=p,
                        stream=False
                    )
                # 只保留新生成的部分
                new_tokens = output[:, sample_input.shape[1]:].tolist()[0]
                generations.append(new_tokens)
            
            # 計算生成結果的多樣性（使用唯一 token 的比例）
            all_tokens = [token for gen in generations for token in gen]
            unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
            results[p] = unique_ratio
        
        # 繪製 top-p 與多樣性的關係
        plt.figure(figsize=(10, 6))
        ps = list(results.keys())
        diversities = list(results.values())
        plt.plot(ps, diversities, marker='o')
        plt.title("Top-p vs. Generation Diversity")
        plt.xlabel("Top-p Value")
        plt.ylabel("Unique Token Ratio")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "top_p_diversity.png")
        plt.close()
        
        # 檢查 top-p 越大，多樣性應該越大
        assert results[0.5] <= results[0.9]
    
    def test_repetition_penalty_effect(self, small_model, sample_input):
        """測試重複懲罰參數對生成的影響"""
        # 使用不同的重複懲罰值生成，並比較結果中的重複情況
        results = {}
        rp_values = [1.0, 1.5, 2.0]
        
        for rp in rp_values:
            # 生成較長的序列以觀察重複情況
            with torch.no_grad():
                output = small_model.generate(
                    sample_input,
                    max_new_tokens=30,
                    temperature=1.0,
                    top_p=1.0,
                    rp=rp,
                    stream=False
                )
            
            # 只保留新生成的部分
            new_tokens = output[:, sample_input.shape[1]:].tolist()[0]
            
            # 計算重複 n-gram 的比例
            def count_repeated_ngrams(tokens, n=2):
                if len(tokens) < n:
                    return 0
                
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
                unique_ngrams = set(ngrams)
                
                if not ngrams:
                    return 0
                
                return 1 - len(unique_ngrams) / len(ngrams)
            
            # 計算 2-gram 和 3-gram 的重複率
            repeat_2gram = count_repeated_ngrams(new_tokens, 2)
            repeat_3gram = count_repeated_ngrams(new_tokens, 3)
            
            results[rp] = (repeat_2gram, repeat_3gram)
        
        # 繪製重複懲罰與重複率的關係
        plt.figure(figsize=(12, 6))
        rps = list(results.keys())
        repeat_2grams = [r[0] for r in results.values()]
        repeat_3grams = [r[1] for r in results.values()]
        
        plt.plot(rps, repeat_2grams, marker='o', label='2-gram Repetition')
        plt.plot(rps, repeat_3grams, marker='s', label='3-gram Repetition')
        plt.title("Repetition Penalty vs. N-gram Repetition Rate")
        plt.xlabel("Repetition Penalty")
        plt.ylabel("Repetition Rate")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "repetition_penalty_effect.png")
        plt.close()
        
        # 檢查重複懲罰越大，重複率應該越低
        assert results[1.0][0] >= results[2.0][0]
    
    def test_stream_vs_direct_generation(self, small_model, sample_input):
        """比較流式生成和直接生成的結果"""
        # 直接生成
        start_time = time.time()
        with torch.no_grad():
            direct_output = small_model.generate(
                sample_input,
                max_new_tokens=20,
                temperature=1.0,
                top_p=0.9,
                stream=False
            )
        direct_time = time.time() - start_time
        
        # 流式生成
        start_time = time.time()
        with torch.no_grad():
            generator = small_model.generate(
                sample_input,
                max_new_tokens=20,
                temperature=1.0,
                top_p=0.9,
                stream=True
            )
            stream_outputs = list(generator)
        stream_time = time.time() - start_time
        
        # 構建完整的流式輸出
        stream_final = torch.cat([sample_input, stream_outputs[-1]], dim=1) if stream_outputs else sample_input
        
        # 比較兩種方法的輸出
        if direct_output.shape == stream_final.shape:
            match_ratio = (direct_output == stream_final).float().mean().item()
        else:
            match_ratio = 0
        
        # 記錄結果
        with open(OUTPUT_DIR / "stream_vs_direct.txt", "w") as f:
            f.write(f"Direct generation time: {direct_time:.4f} seconds\n")
            f.write(f"Stream generation time: {stream_time:.4f} seconds\n")
            f.write(f"Output match ratio: {match_ratio:.4f}\n")
            f.write(f"Direct output shape: {direct_output.shape}\n")
            f.write(f"Stream final output shape: {stream_final.shape}\n")
            
            f.write("\nDirect output tokens:\n")
            f.write(str(direct_output.tolist()))
            
            f.write("\nStream final output tokens:\n")
            f.write(str(stream_final.tolist()))
    
    def test_eos_token_effect(self, small_model, sample_input):
        """測試 EOS 標記對生成長度的影響"""
        # 不使用 EOS 標記
        with torch.no_grad():
            output_no_eos = small_model.generate(
                sample_input,
                max_new_tokens=30,
                temperature=1.0,
                top_p=0.9,
                eos_token_id=None
            )
        
        # 使用一個可能生成的 token 作為 EOS 標記
        # 選擇一個在前5個位置生成概率較高的 token
        with torch.no_grad():
            logits = small_model(sample_input).logits
            likely_tokens = torch.argsort(logits[0, -1], descending=True)[:10].tolist()
        
        eos_token_id = likely_tokens[0]  # 使用最可能的 token 作為 EOS
        
        with torch.no_grad():
            output_with_eos = small_model.generate(
                sample_input,
                max_new_tokens=30,
                temperature=1.0,
                top_p=0.9,
                eos_token_id=eos_token_id
            )
        
        # 記錄結果
        with open(OUTPUT_DIR / "eos_token_effect.txt", "w") as f:
            f.write(f"EOS token ID: {eos_token_id}\n")
            f.write(f"Output without EOS shape: {output_no_eos.shape}\n")
            f.write(f"Output with EOS shape: {output_with_eos.shape}\n")
            
            f.write("\nOutput without EOS tokens:\n")
            f.write(str(output_no_eos.tolist()))
            
            f.write("\nOutput with EOS tokens:\n")
            f.write(str(output_with_eos.tolist()))
    
    def test_cache_performance(self, small_model, sample_input):
        """測試快取對生成性能的影響"""
        # 不使用快取
        start_time = time.time()
        with torch.no_grad():
            small_model.generate(
                sample_input,
                max_new_tokens=20,
                temperature=1.0,
                top_p=0.9,
                use_cache=False
            )
        no_cache_time = time.time() - start_time
        
        # 使用快取
        start_time = time.time()
        with torch.no_grad():
            small_model.generate(
                sample_input,
                max_new_tokens=20,
                temperature=1.0,
                top_p=0.9,
                use_cache=True
            )
        with_cache_time = time.time() - start_time
        
        # 記錄結果
        with open(OUTPUT_DIR / "cache_performance.txt", "w") as f:
            f.write(f"Generation time without cache: {no_cache_time:.4f} seconds\n")
            f.write(f"Generation time with cache: {with_cache_time:.4f} seconds\n")
            f.write(f"Speedup ratio: {no_cache_time / with_cache_time:.2f}x\n")
        
        # 繪製性能比較圖
        plt.figure(figsize=(8, 6))
        plt.bar(['Without Cache', 'With Cache'], [no_cache_time, with_cache_time])
        plt.title("Cache Performance Comparison")
        plt.ylabel("Generation Time (seconds)")
        plt.grid(axis='y')
        plt.savefig(OUTPUT_DIR / "cache_performance.png")
        plt.close()
        
        # 檢查使用快取應該更快
        assert with_cache_time < no_cache_time
    
    def test_batch_generation_consistency(self, small_model):
        """測試批次生成的一致性"""
        # 創建兩個相同的輸入序列
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ])
        
        # 批次生成
        with torch.no_grad():
            batch_output = small_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,  # 使用確定性設置
                top_p=1.0
            )
        
        # 檢查兩個序列的生成結果是否相同
        are_identical = torch.all(batch_output[0] == batch_output[1]).item()
        
        # 記錄結果
        with open(OUTPUT_DIR / "batch_consistency.txt", "w") as f:
            f.write(f"Batch outputs are identical: {are_identical}\n")
            f.write(f"Batch output shape: {batch_output.shape}\n")
            
            f.write("\nFirst sequence output:\n")
            f.write(str(batch_output[0].tolist()))
            
            f.write("\nSecond sequence output:\n")
            f.write(str(batch_output[1].tolist()))
        
        # 檢查相同輸入應該產生相同輸出（在確定性設置下）
        assert are_identical
    
    def test_different_length_inputs(self, small_model):
        """測試不同長度輸入的生成"""
        # 創建不同長度的輸入序列
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5],
            [1, 2, 3],
            [1, 2, 3, 4, 5, 6, 7]
        ])
        
        # 生成
        with torch.no_grad():
            output = small_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                top_p=0.9
            )
        
        # 檢查每個輸入序列的前綴是否被保留
        for i in range(input_ids.shape[0]):
            input_len = input_ids[i].shape[0]
            assert torch.all(output[i, :input_len] == input_ids[i])
        
        # 記錄結果
        with open(OUTPUT_DIR / "different_length_inputs.txt", "w") as f:
            f.write(f"Output shape: {output.shape}\n")
            
            for i in range(input_ids.shape[0]):
                f.write(f"\nInput {i} (length {input_ids[i].shape[0]}):\n")
                f.write(str(input_ids[i].tolist()))
                
                f.write(f"\nOutput {i}:\n")
                f.write(str(output[i].tolist()))
                
                f.write(f"\nNew tokens {i}:\n")
                f.write(str(output[i, input_ids[i].shape[0]:].tolist()))
                f.write("\n" + "-"*50)
    
    def test_generation_with_padding(self, small_model):
        """測試帶填充的生成"""
        # 創建帶填充的輸入序列
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0],  # 序列1，長度5，後面填充0
            [6, 7, 8, 9, 10, 11, 12]  # 序列2，長度7
        ])
        
        # 生成
        with torch.no_grad():
            output = small_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=0
            )
        
        # 檢查輸出形狀
        assert output.shape[0] == input_ids.shape[0]
        
        # 記錄結果
        with open(OUTPUT_DIR / "generation_with_padding.txt", "w") as f:
            f.write(f"Output shape: {output.shape}\n")
            
            for i in range(input_ids.shape[0]):
                f.write(f"\nInput {i}:\n")
                f.write(str(input_ids[i].tolist()))
                
                f.write(f"\nOutput {i}:\n")
                f.write(str(output[i].tolist()))
                
                # 計算非填充 token 的數量
                non_pad_count = (output[i] != 0).sum().item()
                f.write(f"\nNon-padding tokens: {non_pad_count}\n")
                f.write("-"*50) 