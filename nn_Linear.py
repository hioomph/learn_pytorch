import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../DATASET/CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Midori(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


midori = Midori()
# writer = SummaryWriter(log_dir="logs")
# step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # output = torch.reshape(imgs, (1, 1, 1, -1))  # 线性
    # print(output.shape)  # torch.Size([1, 1, 1, 196608])
    # output = midori(output)
    # print(output.shape)  # torch.Size([1, 1, 1, 10])

    output = torch.flatten(imgs)
    print(output.shape)  # torch.Size([196608])
    output = midori(output)
    print(output.shape)  # torch.Size([10])

# writer.close()
