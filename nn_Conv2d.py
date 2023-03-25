import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../DATASET/CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloard = DataLoader(dataset, batch_size=64)


class Midori(nn.Module):
    def __init__(self):
        super(Midori, self).__init__()
        # 定义卷积层
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    # 定义forward函数，并定义其输出
    def forward(self, x):
        x = self.conv1(x)  # 使x进行一层卷积，得到输出
        return x


modori = Midori()
print(modori)
# Midori(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )

step = 0
writer = SummaryWriter("logs")
for data in dataloard:
    imgs, targets = data
    output = modori(imgs)
    # print(imgs.shape)    # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, global_step=step)
    # writer.add_images("output", output, global_step=step)
    # 此处报错是因为tensorboard不知道6个通道的图像该如何显示
    # 对output进行reshape处理： torch.Size([64, 6, 30, 30]) -> torch.Size([xxx, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()  # 此处一定要关闭，很重要
