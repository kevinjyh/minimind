import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import time
from collections import defaultdict
import platform
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 設定中文字體
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系統用黑體
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系統
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # Linux系統

plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 添加根目錄到系統路徑（向上三層到專案根目錄）
root_dir = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, root_dir)  # 使用insert(0)確保優先搜索

from model.model import MiniMindBlock, Attention, FeedForward, MOEFeedForward, apply_rotary_emb
from model.LMConfig import LMConfig

# 創建輸出目錄
output_dir = Path(__file__).parent / "minimindblock_visualization"
output_dir.mkdir(exist_ok=True, parents=True)


class TestMiniMindBlockVisualization:
    """
    可視化MiniMindBlock的內部工作原理，幫助深入理解其功能
    """
    
    @pytest.fixture
    def basic_config(self):
        """基本配置，使用標準前饋網絡(非MoE)"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,  # 使用4個KV頭，以便更好地可視化分組查詢注意力
            vocab_size=6400,
            max_seq_len=128,
            use_moe=False,
            flash_attn=False,
        )
    
    @pytest.fixture
    def moe_config(self):
        """MoE配置，使用混合專家前饋網絡"""
        return LMConfig(
            dim=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=6400,
            max_seq_len=128,
            use_moe=True,
            n_routed_experts=4,
            num_experts_per_tok=2,
            flash_attn=False,
        )
    
    @pytest.fixture
    def instrumented_basic_block(self, basic_config):
        """創建一個帶有鉤子(hooks)的MiniMindBlock，以捕獲內部狀態"""
        block = MiniMindBlock(layer_id=0, config=basic_config)
        self._add_hooks(block)
        return block
    
    @pytest.fixture
    def instrumented_moe_block(self, moe_config):
        """創建一個帶有鉤子的MoE版本MiniMindBlock"""
        block = MiniMindBlock(layer_id=0, config=moe_config)
        self._add_hooks(block)
        return block
    
    def _add_hooks(self, block):
        """向模型添加鉤子以捕獲中間狀態"""
        # 存儲各層的中間狀態
        block.activations = {}
        
        # 捕獲注意力標準化後的輸入
        def attn_norm_hook(module, input, output):
            block.activations["attn_norm_output"] = output.detach().clone()
        block.attention_norm.register_forward_hook(attn_norm_hook)
        
        # 捕獲注意力輸出（在殘差連接前）
        def attn_hook(module, input, output):
            # output[0]是注意力的輸出，output[1]是past_kv
            block.activations["attn_output"] = output[0].detach().clone()
        block.attention.register_forward_hook(attn_hook)
        
        # 捕獲注意力後的殘差輸出
        def post_attn_residual_hook(module, input, output):
            # 這個鉤子需要手動放置在forward方法中
            pass
        
        # 捕獲FFN標準化後的輸入
        def ffn_norm_hook(module, input, output):
            block.activations["ffn_norm_output"] = output.detach().clone()
        block.ffn_norm.register_forward_hook(ffn_norm_hook)
        
        # 捕獲FFN輸出
        def ffn_hook(module, input, output):
            block.activations["ffn_output"] = output.detach().clone()
        block.feed_forward.register_forward_hook(ffn_hook)
        
        # 如果是MoE版本，添加鉤子捕獲MoE的門控決策
        if isinstance(block.feed_forward, MOEFeedForward):
            def moe_gate_hook(module, input, output):
                # 捕獲專家選擇和權重
                block.activations["moe_topk_idx"] = output[0].detach().clone()
                block.activations["moe_topk_weight"] = output[1].detach().clone()
            block.feed_forward.gate.register_forward_hook(moe_gate_hook)
        
        # 修改前向傳播以捕獲更多中間狀態
        original_forward = block.forward
        
        def instrumented_forward(x, pos_cis, past_key_value=None, use_cache=False):
            # 保存原始輸入
            block.activations["input"] = x.detach().clone()
            
            # 調用注意力
            h_attn, past_kv = block.attention(
                block.attention_norm(x), 
                pos_cis,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            # 添加殘差連接
            h = x + h_attn
            block.activations["post_attn_residual"] = h.detach().clone()
            
            # 前饋網絡處理
            ffn_output = block.feed_forward(block.ffn_norm(h))
            
            # 最終殘差連接
            out = h + ffn_output
            block.activations["final_output"] = out.detach().clone()
            
            return out, past_kv
        
        # 替換前向傳播方法
        block.forward = instrumented_forward
    
    @pytest.fixture
    def sample_input(self, basic_config):
        """生成有意義的樣本輸入數據"""
        batch_size = 1  # 使用單一批次便於可視化
        seq_len = 20    # 較長的序列以顯示注意力模式
        dim = basic_config.dim
        
        # 創建一個有明顯模式的輸入（例如，重複的信號或特殊結構）
        # 這將使可視化更有意義
        x = torch.zeros(batch_size, seq_len, dim)
        
        # 在不同位置添加不同特徵的峰值，創建一些明顯的模式
        for i in range(seq_len):
            # 在不同位置添加不同的峰值模式
            if i % 4 == 0:
                x[:, i, :dim//4] = 1.0  # 第一組特徵
            elif i % 4 == 1:
                x[:, i, dim//4:dim//2] = 1.0  # 第二組特徵
            elif i % 4 == 2:
                x[:, i, dim//2:3*dim//4] = 1.0  # 第三組特徵
            else:
                x[:, i, 3*dim//4:] = 1.0  # 第四組特徵
        
        # 添加一些噪聲以使其更自然
        x += torch.randn_like(x) * 0.1
        
        # 生成位置編碼 (pos_cis)
        head_dim = dim // basic_config.n_heads
        theta = basic_config.rope_theta
        
        # 計算頻率基底
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        
        # 生成位置序列
        t = torch.arange(seq_len)
        
        # 計算外積得到位置-維度頻率矩陣
        freqs = torch.outer(t, freqs).float()
        
        # 轉換為複數形式
        pos_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return x, pos_cis
    
    def test_visualize_attention_patterns(self, instrumented_basic_block, sample_input):
        """可視化注意力權重模式"""
        x, pos_cis = sample_input
        
        # 運行前向傳播以捕獲中間狀態
        output, _ = instrumented_basic_block(x, pos_cis)
        
        # 重新計算注意力權重以進行可視化
        # 因為我們無法直接從模型中獲取注意力權重
        with torch.no_grad():
            # 獲取注意力輸入
            attn_input = instrumented_basic_block.activations["attn_norm_output"]
            
            # 獲取配置參數
            n_heads = instrumented_basic_block.n_heads
            n_kv_heads = instrumented_basic_block.attention.n_local_kv_heads
            head_dim = instrumented_basic_block.head_dim
            
            # 使用注意力模塊的權重矩陣計算Q、K、V
            wq_weight = instrumented_basic_block.attention.wq.weight
            wk_weight = instrumented_basic_block.attention.wk.weight
            wv_weight = instrumented_basic_block.attention.wv.weight
            
            # 計算Q、K、V
            xq = torch.matmul(attn_input, wq_weight.t())
            xk = torch.matmul(attn_input, wk_weight.t())
            xv = torch.matmul(attn_input, wv_weight.t())
            
            # 調整形狀
            batch_size, seq_len, _ = attn_input.shape
            xq = xq.view(batch_size, seq_len, n_heads, head_dim)
            xk = xk.view(batch_size, seq_len, n_kv_heads, head_dim)
            xv = xv.view(batch_size, seq_len, n_kv_heads, head_dim)
            
            # 應用旋轉位置編碼
            xq, xk = apply_rotary_emb(xq, xk, pos_cis)
            
            # 處理注意力頭的差異（GQA）
            n_rep = n_heads // n_kv_heads
            if n_rep > 1:
                xk = xk.repeat_interleave(n_rep, dim=2)
                xv = xv.repeat_interleave(n_rep, dim=2)
            
            # 轉置以符合注意力計算
            q = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
            k = xk.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
            
            # 計算注意力分數
            scores = torch.matmul(q, k.transpose(2, 3)) / (head_dim ** 0.5)
            
            # 應用掩碼（確保自迴歸性質）
            mask = torch.triu(torch.ones_like(scores), diagonal=1) * -1e9
            scores = scores + mask
            
            # 應用softmax得到注意力權重
            attn_weights = torch.softmax(scores, dim=-1)
        
        # 可視化每個頭的注意力權重
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(n_heads, 8)):  # 最多顯示8個頭
            ax = axes[i]
            # 獲取當前頭的注意力權重
            head_weights = attn_weights[0, i].cpu().numpy()
            
            # 繪製熱力圖
            sns.heatmap(head_weights, ax=ax, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"Attention Head {i}", fontproperties='SimHei')
            ax.set_xlabel("Key Position", fontproperties='SimHei')
            ax.set_ylabel("Query Position", fontproperties='SimHei')
        
        plt.tight_layout()
        plt.savefig(output_dir / "attention_patterns.png")
        plt.close()
    
    def test_visualize_residual_contributions(self, instrumented_basic_block, sample_input):
        """可視化殘差連接如何貢獻於最終輸出"""
        x, pos_cis = sample_input
        
        # 運行前向傳播以捕獲中間狀態
        output, _ = instrumented_basic_block(x, pos_cis)
        
        # 獲取各階段的輸出
        original_input = instrumented_basic_block.activations["input"]
        attn_output = instrumented_basic_block.activations["attn_output"]
        post_attn_residual = instrumented_basic_block.activations["post_attn_residual"]
        ffn_output = instrumented_basic_block.activations["ffn_output"]
        final_output = instrumented_basic_block.activations["final_output"]
        
        # 計算每個階段的相對貢獻（使用L2範數）
        def compute_relative_norm(tensor):
            return torch.norm(tensor.view(-1), p=2).item()
        
        input_norm = compute_relative_norm(original_input)
        attn_norm = compute_relative_norm(attn_output)
        ffn_norm = compute_relative_norm(ffn_output)
        
        # 計算相對於最終輸出的貢獻比例
        final_norm = compute_relative_norm(final_output)
        input_contribution = input_norm / final_norm
        attn_contribution = attn_norm / final_norm
        ffn_contribution = ffn_norm / final_norm
        
        # 可視化各組件的貢獻
        fig, ax = plt.subplots(figsize=(10, 6))
        contributions = [input_contribution, attn_contribution, ffn_contribution]
        labels = ["原始輸入", "注意力輸出", "前饋網絡輸出"]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        
        ax.bar(labels, contributions, color=colors)
        ax.set_title("各組件對最終輸出的貢獻比例（L2範數）")
        ax.set_ylabel("相對貢獻（佔最終輸出的比例）")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 添加數值標籤
        for i, v in enumerate(contributions):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig(output_dir / "residual_contributions.png")
        plt.close()
        
        # 可視化每個特徵維度的變化
        # 選擇一個序列位置進行可視化
        seq_pos = 0
        
        # 抽取該位置的數據
        original_features = original_input[0, seq_pos].cpu().numpy()
        attn_features = attn_output[0, seq_pos].cpu().numpy()
        post_attn_features = post_attn_residual[0, seq_pos].cpu().numpy()
        ffn_contribution_features = ffn_output[0, seq_pos].cpu().numpy()
        final_features = final_output[0, seq_pos].cpu().numpy()
        
        # 可視化前16個特徵維度
        feature_count = 16
        feature_indices = range(feature_count)
        
        plt.figure(figsize=(14, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(feature_indices, original_features[:feature_count], marker='o', label="原始輸入")
        plt.plot(feature_indices, post_attn_features[:feature_count], marker='s', label="注意力後")
        plt.title(f"序列位置 {seq_pos} 的特徵變化：原始輸入 → 注意力後")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(feature_indices, post_attn_features[:feature_count], marker='o', label="注意力後")
        plt.plot(feature_indices, final_features[:feature_count], marker='s', label="最終輸出")
        plt.title(f"序列位置 {seq_pos} 的特徵變化：注意力後 → 最終輸出")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(feature_indices, attn_features[:feature_count], marker='o', label="注意力貢獻")
        plt.plot(feature_indices, ffn_contribution_features[:feature_count], marker='s', label="FFN貢獻")
        plt.title(f"序列位置 {seq_pos} 的各組件純貢獻")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_changes.png")
        plt.close()
    
    def test_visualize_moe_routing(self, instrumented_moe_block, sample_input):
        """可視化MoE模型的路由決策"""
        x, pos_cis = sample_input
        
        # 確保我們使用的是MoE模型
        assert isinstance(instrumented_moe_block.feed_forward, MOEFeedForward), "此測試需要MoE配置"
        
        # 運行前向傳播以捕獲中間狀態
        output, _ = instrumented_moe_block(x, pos_cis)
        
        # 獲取MoE路由決策
        moe_topk_idx = instrumented_moe_block.activations["moe_topk_idx"]
        moe_topk_weight = instrumented_moe_block.activations["moe_topk_weight"]
        
        # 分析專家分配
        n_experts = instrumented_moe_block.feed_forward.config.n_routed_experts
        k = instrumented_moe_block.feed_forward.config.num_experts_per_tok
        
        # 重塑以便於處理
        batch_size, seq_len, _ = x.shape
        expert_indices = moe_topk_idx.view(batch_size, seq_len, k)
        expert_weights = moe_topk_weight.view(batch_size, seq_len, k)
        
        # 計算每個專家的使用頻率
        expert_counts = torch.zeros(n_experts)
        for i in range(n_experts):
            expert_counts[i] = (expert_indices == i).sum().item()
        
        expert_usage = expert_counts / (batch_size * seq_len * k)
        
        # 可視化專家使用分佈
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_experts), expert_usage.cpu().numpy())
        plt.title("MoE專家使用分佈")
        plt.xlabel("專家索引")
        plt.ylabel("使用頻率")
        plt.xticks(range(n_experts))
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 添加數值標籤
        for i, v in enumerate(expert_usage):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig(output_dir / "moe_expert_usage.png")
        plt.close()
        
        # 可視化每個位置的專家分配
        plt.figure(figsize=(12, 8))
        
        # 為每個序列位置顯示分配的專家
        expert_allocation = np.zeros((seq_len, n_experts))
        
        for pos in range(seq_len):
            for expert_idx in range(k):
                expert = expert_indices[0, pos, expert_idx].item()
                weight = expert_weights[0, pos, expert_idx].item()
                expert_allocation[pos, expert] = weight
        
        # 繪製熱力圖
        sns.heatmap(expert_allocation, cmap="viridis", 
                   xticklabels=[f"專家 {i}" for i in range(n_experts)],
                   yticklabels=[f"位置 {i}" for i in range(seq_len)])
        plt.title("MoE專家分配與權重")
        plt.ylabel("序列位置")
        plt.xlabel("專家")
        
        plt.tight_layout()
        plt.savefig(output_dir / "moe_expert_allocation.png")
        plt.close()
    
    def test_compare_performance(self, basic_config, moe_config, sample_input):
        """比較標準模型和MoE模型的性能"""
        x, pos_cis = sample_input
        batch_size, seq_len, dim = x.shape
        
        # 創建更大的輸入來展示性能差異
        large_batch_size = 16
        large_seq_len = 128
        large_input = torch.randn(large_batch_size, large_seq_len, dim)
        
        # 重新計算位置編碼
        head_dim = dim // basic_config.n_heads
        theta = basic_config.rope_theta
        
        # 計算頻率基底
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        
        # 生成位置序列
        t = torch.arange(large_seq_len)
        
        # 計算外積得到位置-維度頻率矩陣
        freqs = torch.outer(t, freqs).float()
        
        # 轉換為複數形式
        large_pos_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        # 創建模型
        standard_block = MiniMindBlock(layer_id=0, config=basic_config)
        moe_block = MiniMindBlock(layer_id=0, config=moe_config)
        
        # 比較前向傳播時間
        iterations = 10
        
        # 標準模型計時
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(iterations):
            with torch.no_grad():
                standard_output, _ = standard_block(large_input, large_pos_cis)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        standard_time = (time.time() - start_time) / iterations
        
        # MoE模型計時
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(iterations):
            with torch.no_grad():
                moe_output, _ = moe_block(large_input, large_pos_cis)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        moe_time = (time.time() - start_time) / iterations
        
        # 可視化性能比較
        plt.figure(figsize=(10, 6))
        models = ["標準模型", "MoE模型"]
        times = [standard_time, moe_time]
        
        plt.bar(models, times, color=["#3498db", "#e74c3c"])
        plt.title(f"模型前向傳播時間比較 (batch={large_batch_size}, seq_len={large_seq_len})")
        plt.ylabel("每次迭代平均時間 (秒)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 添加數值標籤
        for i, v in enumerate(times):
            plt.text(i, v + 0.01, f"{v:.4f}s", ha="center")
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png")
        plt.close()
        
        # 返回速度比較結果
        result = {
            "standard_time": standard_time,
            "moe_time": moe_time,
            "speedup_ratio": standard_time / moe_time if moe_time > 0 else float('inf')
        }
        
        # 保存結果到文本文件
        with open(output_dir / "performance_results.txt", "w", encoding='utf-8') as f:
            f.write(f"標準模型時間: {standard_time:.6f} 秒\n")
            f.write(f"MoE模型時間: {moe_time:.6f} 秒\n")
            f.write(f"速度比率 (標準/MoE): {result['speedup_ratio']:.6f}\n")
        
        # 改為添加斷言(assert)來驗證性能
        assert moe_time > standard_time * 0.8, "MoE模型不應比標準模型慢超過20%"
        assert result['speedup_ratio'] < 1.0, "標準模型應比MoE模型快"

        # 可選：打印結果但不返回
        print(f"\n性能測試結果: {result}") 