from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/osuma.png")
print(img)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1199x674 at 0x20A80AB6D30>

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])  # tensor(1.)
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])  # tensor(0.6000)
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)  # (1199, 674)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)  # <PIL.Image.Image image mode=RGB size=512x512 at 0x18EB6349518>
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize)
print(img_resize.size)

# Compose - Resize - 2
trans_resize_2 = transforms.Resize(512)  # 将长和宽中较长的边设置为512，另一边等比例缩放
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize_2", img_resize_2)
print(img_resize_2.size)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
