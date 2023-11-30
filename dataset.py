import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image

# 定义数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载CIFAR-10数据集
cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# 使用数据加载器加载数据
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# 从数据加载器中获取一张图像
original_image, original_label = next(iter(data_loader))

