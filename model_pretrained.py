# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)  # 只加载网络模型
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 加载网络模型及具体参数
print(vgg16_true)  #

train_data = torchvision.datasets.CIFAR10(root="../DATASET/CIFAR10", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
# vgg16最终对1000个类别进行分类，而CIFAR10只包含10个类别，如果想用vgg16来对CIFAR10进行分类，则需要对vgg16进行修改
# 方式1：在整个vgg16的最后一层后添加新的一层add_linear
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 方式2：在vgg16最后的classifier模块中添加一个线性层
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


