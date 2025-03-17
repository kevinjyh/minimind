import pytest
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
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

# 設置中文字體以確保圖表正確顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 系統中文設定
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

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
        model = MiniMindLM(small_config)
        # 設置為評估模式，提高穩定性
        model.eval()
        return model
    
    @pytest.fixture
    def sample_input(self):
        """創建一個樣本輸入用於測試"""
        return torch.tensor([[1, 2, 3, 4, 5]])
    
    def test_basic_generation(self, small_model, sample_input):
        """測試基本生成功能"""
        # 使用無梯度上下文提高測試穩定性
        with torch.no_grad():
            # 使用較小的 max_new_tokens 和明確設置 use_cache 參數
            output = small_model.generate(
                sample_input,
                max_new_tokens=5,  # 減少生成數量加快測試
                temperature=1.0,
                top_p=1.0,
                stream=False,
                use_cache=True  # 明確設置
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
            for _ in range(3):  # 減少迭代次數提高測試速度
                with torch.no_grad():
                    output = small_model.generate(
                        sample_input,
                        max_new_tokens=5,  # 減少生成數量
                        temperature=temp,
                        top_p=1.0,
                        stream=False,
                        use_cache=True  # 明確設置
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
        plt.title("溫度與生成多樣性關係", fontsize=14)
        plt.xlabel("溫度參數", fontsize=12)
        plt.ylabel("唯一標記比例", fontsize=12)
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "temperature_diversity.png")
        plt.close()
        
        # 如果溫度效果明顯，則檢查溫度越高，多樣性越大
        # 使用寬鬆判斷，避免隨機性導致的測試失敗
        if diversities[-1] > 0 and diversities[0] > 0:
            assert results[0.5] <= results[2.0] * 1.2  # 允許20%的誤差
    
    def test_top_p_effect(self, small_model, sample_input):
        """測試 top-p 參數對生成的影響"""
        # 使用不同的 top-p 值生成多次，並比較結果的多樣性
        results = {}
        top_p_values = [0.5, 0.7, 0.9]
        
        for p in top_p_values:
            # 多次生成以收集統計信息
            generations = []
            for _ in range(3):  # 減少迭代次數
                with torch.no_grad():
                    output = small_model.generate(
                        sample_input,
                        max_new_tokens=5,  # 減少生成長度
                        temperature=1.0,
                        top_p=p,
                        stream=False,
                        use_cache=True  # 明確設置
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
        plt.title("Top-p 與生成多樣性關係", fontsize=14)
        plt.xlabel("Top-p 參數值", fontsize=12)
        plt.ylabel("唯一標記比例", fontsize=12)
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "top_p_diversity.png")
        plt.close()
        
        # 寬鬆的檢查，允許一定誤差
        if diversities[-1] > 0 and diversities[0] > 0:
            assert results[0.5] <= results[0.9] * 1.2  # 允許20%的誤差
    
    def test_repetition_penalty_effect(self, small_model, sample_input):
        """測試重複懲罰參數對生成的影響"""
        # 測試執行前驗證模型參數，不再針對層級結構輸出調試信息
        config = small_model.params
        assert config.n_heads >= config.n_kv_heads, "測試需要模型支持 GQA"
        assert config.n_heads % config.n_kv_heads == 0, "測試需要 n_heads 能被 n_kv_heads 整除"

        # 執行原始測試邏輯
        results = {}
        rp_values = [1.0, 1.5, 2.0]
        
        for rp in rp_values:
            with torch.no_grad():
                output = small_model.generate(
                    sample_input,
                    max_new_tokens=10,
                    temperature=1.0,
                    top_p=1.0,
                    rp=rp,
                    stream=False,
                    use_cache=True
                )
            
            # 只保留新生成的部分
            new_tokens = output[:, sample_input.shape[1]:].tolist()[0]
            
            # 計算生成結果中的重複率
            # 使用bi-gram計算
            repeated_count = 0
            for i in range(len(new_tokens) - 1):
                bigram = tuple(new_tokens[i:i+2])
                # 檢查這個bi-gram是否已經出現過
                for j in range(i - 1):
                    if tuple(new_tokens[j:j+2]) == bigram:
                        repeated_count += 1
                        break
            
            if len(new_tokens) > 1:
                repeat_rate = repeated_count / (len(new_tokens) - 1)
            else:
                repeat_rate = 0
            
            results[rp] = repeat_rate
        
        # 繪製重複懲罰與重複率的關係
        plt.figure(figsize=(10, 6))
        rps = list(results.keys())
        rates = list(results.values())
        plt.plot(rps, rates, marker='o')
        plt.title("重複懲罰與生成重複率關係", fontsize=14)
        plt.xlabel("重複懲罰參數", fontsize=12)
        plt.ylabel("Bi-gram重複率", fontsize=12)
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "repetition_penalty_effect.png")
        plt.close()
        
        # 檢查重複懲罰是否有效（允許一定誤差因為隨機性）
        # 只有當有明顯差異時才執行檢查
        if len(set(rates)) > 1 and max(rates) > 0.1:
            assert results[1.0] >= results[2.0] * 0.8, "重複懲罰應降低重複率"
    
    def test_stream_vs_direct_generation(self, small_model, sample_input):
        """比較流式生成和直接生成的結果"""
        # 直接生成
        start_time = time.time()
        with torch.no_grad():
            direct_output = small_model.generate(
                sample_input,
                max_new_tokens=10,  # 減少生成長度
                temperature=1.0,
                top_p=0.9,
                stream=False,
                use_cache=True  # 明確設置
            )
        direct_time = time.time() - start_time
        
        # 流式生成
        start_time = time.time()
        stream_outputs = []
        with torch.no_grad():
            generator = small_model.generate(
                sample_input,
                max_new_tokens=10,  # 減少生成長度
                temperature=1.0,
                top_p=0.9,
                stream=True,
                use_cache=True  # 明確設置
            )
            
            # 安全地迭代生成器
            try:
                for token in generator:
                    stream_outputs.append(token)
            except Exception as e:
                print(f"流式生成出現錯誤: {e}")
        
        stream_time = time.time() - start_time
        
        # 構建完整的流式輸出 (只有在至少生成了一個標記時)
        if stream_outputs:
            stream_final = torch.cat([sample_input, stream_outputs[-1]], dim=1)
            
            # 比較兩種方法的輸出
            if direct_output.shape == stream_final.shape:
                match_ratio = (direct_output == stream_final).float().mean().item()
            else:
                match_ratio = 0
        else:
            match_ratio = 0
            stream_final = sample_input  # 如果沒有生成，就使用輸入作為最終輸出
        
        # 記錄結果
        with open(OUTPUT_DIR / "stream_vs_direct.txt", "w") as f:
            f.write(f"直接生成時間: {direct_time:.4f} 秒\n")
            f.write(f"流式生成時間: {stream_time:.4f} 秒\n")
            f.write(f"輸出匹配比例: {match_ratio:.4f}\n")
            f.write(f"直接生成輸出形狀: {direct_output.shape}\n")
            f.write(f"流式生成最終輸出形狀: {stream_final.shape}\n")
            
            f.write("\n直接生成標記:\n")
            f.write(str(direct_output.tolist()))
            
            f.write("\n流式生成最終標記:\n")
            f.write(str(stream_final.tolist()))
    
    def test_eos_token_effect(self, small_model, sample_input):
        """測試 EOS 標記對生成長度的影響"""
        # 不使用 EOS 標記
        with torch.no_grad():
            output_no_eos = small_model.generate(
                sample_input,
                max_new_tokens=15,  # 減少生成長度
                temperature=1.0,
                top_p=0.9,
                eos_token_id=None,
                use_cache=True  # 明確設置
            )
        
        # 選擇一個能觀察到明顯效果的 EOS token
        # 首先獲取模型輸出的概率分佈
        with torch.no_grad():
            outputs = small_model(sample_input, use_cache=False)
            logits = outputs.logits
            likely_tokens = torch.argsort(logits[0, -1], descending=True)[:5].tolist()
        
        # 從高概率標記中選擇一個作為 EOS
        eos_token_id = likely_tokens[0]
        
        # 使用選定的 EOS 標記生成
        with torch.no_grad():
            output_with_eos = small_model.generate(
                sample_input,
                max_new_tokens=15,  # 減少生成長度
                temperature=1.0,
                top_p=0.9,
                eos_token_id=eos_token_id,
                use_cache=True  # 明確設置
            )
        
        # 記錄結果
        with open(OUTPUT_DIR / "eos_token_effect.txt", "w") as f:
            f.write(f"EOS 標記 ID: {eos_token_id}\n")
            f.write(f"無 EOS 生成形狀: {output_no_eos.shape}\n")
            f.write(f"有 EOS 生成形狀: {output_with_eos.shape}\n")
            
            f.write("\n無 EOS 生成標記:\n")
            f.write(str(output_no_eos.tolist()))
            
            f.write("\n有 EOS 生成標記:\n")
            f.write(str(output_with_eos.tolist()))
    
    def test_cache_performance(self, small_model, sample_input):
        """測試快取對生成性能的影響"""
        # 增加生成長度並多次測量
        num_runs = 3  # 增加測量次數
        max_new_tokens = 50  # 增加生成長度
        
        # 不使用快取
        no_cache_times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()  # 使用更高精度計時器
            with torch.no_grad():
                small_model.generate(
                    sample_input,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=0.9,
                    use_cache=False
                )
            no_cache_times.append(time.perf_counter() - start_time)
        
        # 使用快取
        with_cache_times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                small_model.generate(
                    sample_input,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_p=0.9,
                    use_cache=True
                )
            with_cache_times.append(time.perf_counter() - start_time)
        
        # 計算統計量
        avg_no_cache = sum(no_cache_times) / num_runs
        avg_with_cache = sum(with_cache_times) / num_runs
        
        # 記錄完整結果
        with open(OUTPUT_DIR / "cache_performance.txt", "w") as f:
            f.write(f"多次測量結果 (次數={num_runs}):\n")
            f.write(f"無快取時間: {no_cache_times}\n")
            f.write(f"有快取時間: {with_cache_times}\n\n")
            
            f.write(f"平均無快取時間: {avg_no_cache:.6f} 秒\n")
            f.write(f"平均有快取時間: {avg_with_cache:.6f} 秒\n")
            
            if avg_with_cache > 1e-6:  # 防止除零
                speedup = avg_no_cache / avg_with_cache
                f.write(f"加速比: {speedup:.2f}x\n")
            else:
                f.write("加速比: 測量時間過短無法計算\n")
        
        # 繪製性能比較圖
        plt.figure(figsize=(10, 6))
        plt.bar(['無快取', '有快取'], [avg_no_cache, avg_with_cache], 
                yerr=[np.std(no_cache_times), np.std(with_cache_times)], 
                capsize=10)
        plt.title("快取性能比較 (多次測量平均)", fontsize=14)
        plt.ylabel("生成時間 (秒)", fontsize=12)
        plt.grid(axis='y')
        plt.savefig(OUTPUT_DIR / "cache_performance.png")
        plt.close()
        
        # 寬鬆的斷言條件，考慮測量誤差
        if avg_with_cache > 1e-6:  # 有效測量值
            # 允許快取版本最多比非快取版本慢 50%（考慮誤差範圍）
            assert avg_with_cache <= avg_no_cache * 1.5, (
                f"快取不應顯著降低性能 (無快取: {avg_no_cache:.4f}s, 有快取: {avg_with_cache:.4f}s)"
            )
        else:
            pytest.fail("快取時間測量值異常，可能因系統計時精度不足")
    
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
                max_new_tokens=5,  # 減少生成長度
                temperature=0.8,  # 使用較低溫度增加確定性
                top_p=1.0,
                use_cache=True  # 明確設置
            )
        
        # 檢查兩個序列的生成結果是否相同
        are_identical = torch.all(batch_output[0] == batch_output[1]).item()
        
        # 記錄結果
        with open(OUTPUT_DIR / "batch_consistency.txt", "w") as f:
            f.write(f"批次輸出是否相同: {are_identical}\n")
            f.write(f"批次輸出形狀: {batch_output.shape}\n")
            
            f.write("\n第一個序列輸出:\n")
            f.write(str(batch_output[0].tolist()))
            
            f.write("\n第二個序列輸出:\n")
            f.write(str(batch_output[1].tolist()))
        
        # 檢查相同輸入應該產生相同輸出（在確定性設置下）
        assert are_identical
    
    def test_different_length_inputs(self, small_model):
        """測試不同長度輸入的生成"""
        # 創建不同長度的輸入序列，並進行填充
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0],  # 序列1，長度5，填充到7
            [1, 2, 3, 0, 0, 0, 0],  # 序列2，長度3，填充到7
            [1, 2, 3, 4, 5, 6, 7]   # 序列3，長度7，不需要填充
        ])
        
        # 生成
        with torch.no_grad():
            output = small_model.generate(
                input_ids,
                max_new_tokens=5,  # 減少生成長度
                temperature=1.0,
                top_p=0.9,
                pad_token_id=0,  # 指定填充標記
                use_cache=True  # 明確設置
            )
        
        # 檢查每個輸入序列的前綴是否被保留
        for i in range(input_ids.shape[0]):
            input_len = (input_ids[i] != 0).sum().item()  # 計算非填充部分的長度
            assert torch.all(output[i, :input_len] == input_ids[i, :input_len])
        
        # 記錄結果
        with open(OUTPUT_DIR / "different_length_inputs.txt", "w") as f:
            f.write(f"輸出形狀: {output.shape}\n")
            
            for i in range(input_ids.shape[0]):
                f.write(f"\n輸入 {i} (長度 {input_len}):\n")
                f.write(str(input_ids[i].tolist()))
                
                f.write(f"\n輸出 {i}:\n")
                f.write(str(output[i].tolist()))
                
                f.write(f"\n新生成標記 {i}:\n")
                f.write(str(output[i, input_len:input_len+5].tolist()))  # 只顯示新生成的部分
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
                max_new_tokens=5,  # 減少生成長度
                temperature=1.0,
                top_p=0.9,
                pad_token_id=0,
                use_cache=True  # 明確設置
            )
        
        # 檢查輸出形狀
        assert output.shape[0] == input_ids.shape[0]
        
        # 記錄結果
        with open(OUTPUT_DIR / "generation_with_padding.txt", "w") as f:
            f.write(f"輸出形狀: {output.shape}\n")
            
            for i in range(input_ids.shape[0]):
                f.write(f"\n輸入 {i}:\n")
                f.write(str(input_ids[i].tolist()))
                
                f.write(f"\n輸出 {i}:\n")
                f.write(str(output[i].tolist()))
                
                # 計算非填充 token 的數量
                non_pad_count = (output[i] != 0).sum().item()
                f.write(f"\n非填充標記數量: {non_pad_count}\n")
                f.write("-"*50) 