import torch
from torch import nn


class Midori(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


midori = Midori()  # 创建的一个神经网络midori
x = torch.tensor(1.0)  # 创建输入，将1.0转换为tensor数据类型
output = midori(x)  # 将x输入到网络midori中，得到输出output
print(output)  # tensor(2.)

