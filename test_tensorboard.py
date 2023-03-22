from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writter = SummaryWriter("logs")
img_path = r"data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))  # <class 'numpy.ndarray'>
print(img_array.shape)  # (512, 768, 3)

writter.add_image(tag="test", img_tensor=img_array, global_step=1, dataformats='HWC')

# y = 2x
# for i in range(100):
#     # scalar_value ==> x轴
#     # global_step  ==> y轴
#     writter.add_scalar(tag="y=2x", scalar_value=2 * i, global_step=i)

writter.close()
