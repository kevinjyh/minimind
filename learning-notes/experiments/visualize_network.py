import torchinfo
import torch
import torch.nn as nn
from torchviz import make_dot
import os

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(4, 8, bias=True)

    def forward(self, x):
        return self.linear(x)

# 初始化模型
model = SimpleNet()

# 输入张量
input_tensor = torch.randn(1, 4)

# 輸出網絡結構摘要到控制台
summary_result = torchinfo.summary(model, input_data=input_tensor, device='cpu')
print(summary_result)

# 生成網絡圖
output = model(input_tensor)
dot = make_dot(output, params=dict(model.named_parameters()))

# 設定保存路徑
save_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取當前腳本的目錄（即experiments目錄）
save_path = os.path.join(save_dir, "network")

# 保存圖像
dot.render(save_path, format="png", cleanup=True)
print(f"網絡圖已保存至: {save_path}.png")
