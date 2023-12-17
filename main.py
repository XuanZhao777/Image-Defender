import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from adversarial_attack import adversarial_attack
from separate_attack import separate_attack
from correct_attack import correct_attack
from combine_parts import combine_parts
from torchvision.models import resnet18
from test import test
import tkinter as tk
from PIL import Image, ImageTk

# Load a pre-trained ResNet18 model
model = resnet18(weights='IMAGENET1K_V1')
model.eval()

# Define the data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download the CIFAR-10 dataset
cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Use a data loader to load the data
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

# Get an image and its label from the data loader
original_image, original_label = next(iter(data_loader))

# Define attack parameters and correction factor
epsilon = 0.03
alpha = 0.0015
iterations = 80
correction_factor = 0.5

# Perform adversarial attack on the image
ground_truth_labels = torch.tensor([2])
perturbed_image = adversarial_attack(model, original_image, label=ground_truth_labels, epsilon=epsilon, alpha=alpha, iterations=iterations)

# Separate attacked and unattacked parts
attacked_part, unattacked_part = separate_attack(model, original_image, perturbed_image)

# Correct the attacked part
corrected_part = correct_attack(model, original_image, attacked_part, max_iterations=100, lr=0.01)

# Combine the unattacked and corrected parts
combined_image = combine_parts(original_image, unattacked_part, corrected_part)

# Test the combined image
test(combined_image, original_image)

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 6, 1)
plt.imshow(original_image.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Original Image")

plt.subplot(1, 6, 2)
plt.imshow(perturbed_image.squeeze().detach().permute(1, 2, 0).detach().numpy())
plt.title("Adversarial Image")

plt.subplot(1, 6, 3)
plt.imshow(torch.clamp(attacked_part.squeeze().detach().permute(1, 2, 0), 0, 1).detach().numpy(), cmap='gray')
plt.title("Attacked Part")

plt.subplot(1, 6, 4)
plt.imshow(torch.clamp(unattacked_part.squeeze().detach().permute(1, 2, 0), 0, 1).detach().numpy(), cmap='gray')
plt.title("Unattacked Part")

plt.subplot(1, 6, 5)
plt.imshow(torch.clamp(corrected_part.squeeze().detach().permute(1, 2, 0), 0, 1).detach().numpy(), cmap='gray')
plt.title("Corrected Part")

plt.subplot(1, 6, 6)
plt.imshow(combined_image.squeeze().detach().permute(1, 2, 0).detach().numpy())
plt.title("Combined Image")

plt.show()

