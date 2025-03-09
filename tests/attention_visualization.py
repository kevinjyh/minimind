import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib as mpl

# 設置 Matplotlib 支持中文顯示
# 方法1：使用系統中已有的中文字體
try:
    # 嘗試使用不同的中文字體，根據操作系統可用性
    if sys.platform.startswith('win'):  # Windows
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif sys.platform.startswith('darwin'):  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux 或其他
        font_list = ['Noto Sans CJK TC', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
    
    # 嘗試設置字體，直到找到可用的
    font_found = False
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
            # 測試字體是否可用
            mpl.font_manager.findfont(font)
            font_found = True
            print(f"使用字體: {font}")
            break
        except:
            continue
    
    if not font_found:
        print("警告: 未找到支持中文的字體，圖表中的中文可能無法正確顯示")
except Exception as e:
    print(f"設置字體時出錯: {e}")
    print("圖表中的中文可能無法正確顯示")

# 確保可以導入模型模塊
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入自定義的SimpleAttention
from test_attention_standalone import SimpleAttention

def visualize_attention_weights(attn_weights, title="Attention Weights"):
    """將注意力權重可視化為熱力圖"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Key位置")
    plt.ylabel("Query位置")
    plt.tight_layout()
    plt.show()

def demo_causal_mask():
    """展示因果掩碼的效果"""
    # 創建一個簡單的注意力層
    hidden_size = 64
    num_heads = 1
    attn = SimpleAttention(hidden_size, num_heads)
    
    # 創建隨機輸入
    batch_size, seq_len = 1, 6
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 1. 使用因果掩碼
    _, attn_weights_causal = attn(x, causal_mask=True)
    
    # 2. 不使用因果掩碼
    _, attn_weights_no_causal = attn(x, causal_mask=False)
    
    # 展示掩碼前後的對比
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(attn_weights_causal[0, 0].detach().numpy(), 
                annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("有因果掩碼的注意力權重")
    plt.xlabel("Key位置")
    plt.ylabel("Query位置")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(attn_weights_no_causal[0, 0].detach().numpy(), 
                annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("無因果掩碼的注意力權重")
    plt.xlabel("Key位置")
    plt.ylabel("Query位置")
    
    plt.tight_layout()
    plt.show()

def demo_topic_attention():
    """展示主題之間的注意力模式"""
    # 創建一個帶有明確主題模式的序列
    hidden_size = 6
    seq_len = 6
    
    # 創建三個表示不同主題的one-hot向量
    topic1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    topic2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    topic3 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    # 創建一個序列，其中每兩個token屬於同一主題
    x = torch.zeros(1, seq_len, hidden_size)
    x[0, 0] = topic1  # 主題1
    x[0, 1] = topic1  # 主題1
    x[0, 2] = topic2  # 主題2
    x[0, 3] = topic2  # 主題2
    x[0, 4] = topic3  # 主題3
    x[0, 5] = topic3  # 主題3
    
    # 創建一個簡單的注意力層
    attn = SimpleAttention(hidden_size, num_heads=1)
    
    # 使權重矩陣為單位矩陣，以便直接反映輸入特徵的相似性
    attn.q_proj.weight.data = torch.eye(hidden_size)
    attn.k_proj.weight.data = torch.eye(hidden_size)
    attn.v_proj.weight.data = torch.eye(hidden_size)
    attn.o_proj.weight.data = torch.eye(hidden_size)
    
    # 計算注意力
    _, attn_weights = attn(x, causal_mask=False)
    
    # 可視化結果 - 添加自定義標籤
    weights = attn_weights[0, 0].detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("不同主題間的注意力權重")
    
    # 添加自定義軸標籤
    token_labels = ["主題1-1", "主題1-2", "主題2-1", "主題2-2", "主題3-1", "主題3-2"]
    plt.xticks(np.arange(len(token_labels))+0.5, token_labels, rotation=45)
    plt.yticks(np.arange(len(token_labels))+0.5, token_labels, rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # 輸出解釋
    print("注意觀察：")
    print("1. 同一主題內的token之間有高注意力權重")
    print("2. 不同主題間的token之間有低注意力權重")
    print("3. 這展示了自注意力機制如何自動捕捉序列中的相似模式")

def demo_multi_head_attention():
    """展示多頭注意力機制，不同頭關注不同特徵"""
    # 創建一個稍複雜的輸入，包含多種模式
    hidden_size = 8
    num_heads = 2
    seq_len = 4
    
    # 創建輸入序列，每個token包含兩種特徵
    x = torch.zeros(1, seq_len, hidden_size)
    
    # 前半部分表示"主題"特徵，後半部分表示"情感"特徵
    # Token 0: 主題A + 正面情感
    x[0, 0, :4] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 主題A
    x[0, 0, 4:] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 正面情感
    
    # Token 1: 主題A + 負面情感
    x[0, 1, :4] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 主題A
    x[0, 1, 4:] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 負面情感
    
    # Token 2: 主題B + 正面情感
    x[0, 2, :4] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 主題B
    x[0, 2, 4:] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 正面情感
    
    # Token 3: 主題B + 負面情感
    x[0, 3, :4] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 主題B
    x[0, 3, 4:] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 負面情感
    
    # 創建多頭注意力層
    attn = SimpleAttention(hidden_size, num_heads=num_heads)
    
    # 設置第一個頭只關注前半部分特徵（主題）
    # 設置第二個頭只關注後半部分特徵（情感）
    
    # 重設權重
    attn.q_proj.weight.data.zero_()
    attn.k_proj.weight.data.zero_()
    
    # 頭1關注前半部分（主題）
    head_dim = hidden_size // num_heads
    for i in range(4):
        # 對應主題特徵的部分
        attn.q_proj.weight.data[i, i] = 1.0
        attn.k_proj.weight.data[i, i] = 1.0
    
    # 頭2關注後半部分（情感）
    for i in range(4, 8):
        # 對應情感特徵的部分
        attn.q_proj.weight.data[i, i] = 1.0
        attn.k_proj.weight.data[i, i] = 1.0
    
    # 計算注意力
    _, attn_weights = attn(x, causal_mask=False)
    
    # 獲取每個頭的注意力權重
    head1_weights = attn_weights[0, 0].detach().numpy()
    head2_weights = attn_weights[0, 1].detach().numpy()
    
    # 可視化兩個頭的注意力模式
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(head1_weights, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("頭1：關注主題特徵")
    token_labels = ["主題A正", "主題A負", "主題B正", "主題B負"]
    plt.xticks(np.arange(len(token_labels))+0.5, token_labels, rotation=45)
    plt.yticks(np.arange(len(token_labels))+0.5, token_labels, rotation=0)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(head2_weights, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("頭2：關注情感特徵")
    plt.xticks(np.arange(len(token_labels))+0.5, token_labels, rotation=45)
    plt.yticks(np.arange(len(token_labels))+0.5, token_labels, rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    print("注意觀察：")
    print("1. 頭1主要關注主題相似性：主題A的token之間有高注意力，主題B的token之間也有高注意力")
    print("2. 頭2主要關注情感相似性：正面情感的token之間有高注意力，負面情感的token之間也有高注意力")
    print("3. 這展示了多頭注意力如何允許模型同時關注不同類型的關係")

if __name__ == "__main__":
    print("Attention可視化工具")
    
    while True:
        print("選擇要運行的演示：")
        print("1. 因果掩碼效果演示")
        print("2. 主題關注模式演示")
        print("3. 多頭注意力機制演示")
        print("q. 退出")
        
        choice = input("請輸入選項（1-3 或 q）：")
        
        if choice == "1":
            demo_causal_mask()
        elif choice == "2":
            demo_topic_attention()
        elif choice == "3":
            demo_multi_head_attention()
        elif choice.lower() == "q":
            print("退出程序。")
            break
        else:
            print("無效選項！請輸入1-3之間的數字或 q 退出。") 