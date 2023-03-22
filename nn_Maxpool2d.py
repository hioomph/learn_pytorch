"""
    池化函数使用某一位置的相邻输出的总体特征来替代在网络在该位置的输出
    本质是降采样（特征降维），可以大幅减少网络的参数量
"""

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

# 1
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
# [[]] ==> 二维矩阵
# dtype=torch.float32 ==> 解决：RuntimeError: "max_pool2d" not implemented for 'Long'

input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)  # torch.Size([1, 1, 5, 5])

# 2
dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Midori(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

midori = Midori()
# 1
# output = midori(input)
# print(output)
# # tensor([[[[2., 3.],
# #           [5., 1.]]]])

# 2
writer = SummaryWriter(log_dir="logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("max_pool_input", imgs, global_step=step)
    output = midori(imgs)
    writer.add_images("max_pool_output", output, global_step=step)
    step += 1

writer.close()

