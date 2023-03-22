import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# batch_size=4      => 每次从dtest_data中取四张图片进行打包
# shuffle=True      => 在每个epoch对数据进行打乱
# num_workers=0     => 开的进程数，使用cpu时最好设置为0
# drop_last=False   => 当数据集中的图片数量不满足batch_size的倍数时，最后剩下的图片是否去掉

# 测试数据集中的第一张图片
img, target = test_data[0]
print(img.shape)  # torch.Size([3, 32, 32])
print(target)  # 3

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        # 第一个batch_size(=4时)：shape==torch.Size([4, 3, 32, 32]), targets==tensor([0, 3, 5, 4])
        writer.add_images("Epoch:{}".format(epoch), imgs, step)  # 注意这里是add_images而不是add_image！
        step += 1
writer.close()  # 此处一定要关闭，很重要


