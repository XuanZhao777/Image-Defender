import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from PIL import Image, ImageTk
import tkinter as tk
import matplotlib.pyplot as plt
from adversarial_attack import adversarial_attack
from separate_attack import separate_attack
from correct_attack import correct_attack
from combine_parts import combine_parts
from test import test

class ImageDisplayApp:
    def __init__(self, root, model, epsilon, threshold, correction_factor):
        self.root = root
        self.model = model
        self.epsilon = epsilon
        self.threshold = threshold
        self.correction_factor = correction_factor

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.data_loader = DataLoader(self.cifar10_dataset, batch_size=1, shuffle=True)
        self.data_iterator = iter(self.data_loader)
        self.original_image, self.original_label = next(self.data_iterator)

        self.setup_gui()

    def setup_gui(self):
        self.root.title("Adversarial Attack Demo")

        self.show_original_image()
        self.show_adversarial_image()
        self.show_combined_image()

        self.next_button = tk.Button(self.root, text="Next", command=self.load_next_image)
        self.next_button.grid(row=1, column=1, padx=10, pady=10)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.root.destroy)
        self.stop_button.grid(row=1, column=2, padx=10, pady=10)

    def show_image(self, image, title, size=(300, 300), row=None, col=None):
        img = Image.fromarray((image.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))
        img = ImageTk.PhotoImage(img.resize(size))
        panel = tk.Label(self.root, image=img)
        panel.image = img
        panel.grid(row=row, column=col, padx=10, pady=10)
        plt.title(title)

    def show_original_image(self):
        self.show_image(self.original_image, "Original Image", row=0, col=0)

    def show_adversarial_image(self):
        ground_truth_labels = torch.tensor([2])
        perturbed_image = adversarial_attack(self.model, self.original_image, label=ground_truth_labels, epsilon=self.epsilon)
        self.show_image(perturbed_image, "Adversarial Image", row=0, col=1)

    def show_combined_image(self):
        ground_truth_labels = torch.tensor([2])
        perturbed_image = adversarial_attack(self.model, self.original_image, label=ground_truth_labels, epsilon=self.epsilon)
        attacked_part, unattacked_part = separate_attack(self.model, self.original_image, perturbed_image, threshold=self.threshold)
        corrected_part = correct_attack(self.original_image, attacked_part)
        not_attacked_part = 1 - attacked_part
        combined_image = combine_parts(self.original_image, not_attacked_part, corrected_part)
        test_result = test(combined_image, self.original_image)
        self.show_image(combined_image, f"Combined Image\nTest Result: {test_result}", row=0, col=2)

    def load_next_image(self):
        try:
            self.original_image, self.original_label = next(self.data_iterator)
            self.show_original_image()
            self.show_adversarial_image()
            self.show_combined_image()
        except StopIteration:
            print("No more images in the dataset.")

if __name__ == "__main__":
    model = resnet18(weights='IMAGENET1K_V1')
    model.eval()

    epsilon = 0.1
    threshold = 0.1
    correction_factor = 0.5

    root = tk.Tk()
    app = ImageDisplayApp(root, model, epsilon, threshold, correction_factor)
    root.mainloop()
