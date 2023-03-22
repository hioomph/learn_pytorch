from torch.utils.data import Dataset
from PIL import Image
import os  # 关于系统的库


class MyData(Dataset):

    # 为整个class提供一个全局变量，用self指定
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 获取所有图片的地址列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取图片名称
        img_item_path = os.path.join(self.path, img_name)  # 获取图片相对路径
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)  # 返回数据集的长度


root_path = r'dataset/train'
ants_label_path = r'ants'
bees_label_path = r'bees'
ants_dataset = MyData(root_path, ants_label_path)
bees_dataset = MyData(root_path, bees_label_path)

train_dataset = ants_dataset + bees_dataset
