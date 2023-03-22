import torch
from torch import nn
import torch.nn.functional as F

# 输入图像
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])  # [[]] ==> 二维矩阵

# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 修改input和kernel的尺寸
print(input.shape)  # torch.Size([5, 5])
print(kernel.shape)  # torch.Size([3, 3])
# 初始化的input和kernel的尺寸不满足conv2d函数所传参数需要的尺寸，因此需要进行重设
# https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d 参数维度要求
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)  # torch.Size([1, 1, 5, 5])
print(kernel.shape)  # torch.Size([1, 1, 3, 3])

# 调用conv2d
output = F.conv2d(input, kernel, stride=1)
print(output)
print(output.shape)  # torch.Size([1, 1, 3, 3])

output2 = F.conv2d(input, kernel, stride=2)
print(output2)
print(output2.shape)  # torch.Size([1, 1, 2, 2])

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
print(output3.shape)  # torch.Size([1, 1, 5, 5])
