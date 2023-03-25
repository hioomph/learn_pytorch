import torchvision
from torch.utils.tensorboard import SummaryWriter

# 使用torchvision内置的数据集
# 说明文档： https://pytorch.org/vision/0.9/datasets.html#cifar
# root="./dataset" => 在运行该程序后，会在dataset文件夹下保存数据集
# 1、将原始数据集由PIL格式转换为Tensor格式
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="../DATASET/CIFAR10", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="../DATASET/CIFAR10", train=False, transform=dataset_transform, download=True)
# 上述原始数据集的格式为PIL，如果要用transform进行处理，需要转换为Tensor格式
# 注意：如果要在dataset文件夹下放已经下载好的数据集，要传入.tar.gz的压缩包格式，否则无法识别，会重新下载


# print(test_set[0])
writer = SummaryWriter("logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()  # 此处一定要关闭，很重要

# 2、查看数据集中的第一个数据
# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)

# print(train_set[0])  # (<PIL.Image.Image image mode=RGB size=32x32 at 0x1BE5ABCEF60>, 3)
# print(train_set.classes)
#
# img, target = train_set[0]
# print(img)  # <PIL.Image.Image image mode=RGB size=32x32 at 0x1BE5ABCEF60>
# print(target)  # 3，对应第一张图片所属classes的类别
# img.show()


