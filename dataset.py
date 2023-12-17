import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download the CIFAR-10 dataset
cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Use a data loader to load the data
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# Get one image from the data loader
original_image, original_label = next(iter(data_loader))


