from model_save import *
import torchvision

# 方式1 ==> 保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2 ==> 保存方式1，加载参数
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
# print(vgg16)

# 陷阱1
model = torch.load('midori_method1.pth')
print(model)
