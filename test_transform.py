import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
    python的用法 =》 tensor数据类型
    通过 tansform.Totensor 去解决两个问题：
      1、transform该如何使用（python）
      2、tensor数据类型相较于普通的数据类型有何区别？
    
    tansform.Totensor:
      Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
"""
# 绝对路径 D:\PostGraduate\pythonex\6 learn_pytorch\data\train\ants_image\0013035.jpg
# 相对路径 data/train/ants_image/0013035.jpg

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x2166CC830B8>

# 1、transform该如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

# 2、tensor数据类型相较于普通的数据类型有何区别？
# tensor数据类型包含了神经网络所需的参数
cv_img = cv2.imread(img_path)
# print(cv_img)

writer = SummaryWriter("logs")
writer.add_image("tensor_img", tensor_img)
writer.close()

