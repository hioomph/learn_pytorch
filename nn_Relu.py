import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import ReLU, Sigmoid
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
#
# input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Midori(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


midori = Midori()
# output = midori(input)
# print(output)

writer = SummaryWriter(log_dir="logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = midori(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()



