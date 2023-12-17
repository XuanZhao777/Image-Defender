import torch
import torch.nn as nn
from torch.autograd import Variable

def separate_attack(model, original_image, perturbed_image, k=1):
    # Create a new leaf tensor, copying the data from perturbed_image
    perturbed_image_clone = perturbed_image.clone().detach()
    perturbed_image_clone.requires_grad = True

    # Get the model outputs
    original_output = model(original_image)
    perturbed_output = model(perturbed_image_clone)

    # Calculate the loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(perturbed_output, torch.argmax(original_output, dim=1))

    # Calculate gradients
    loss.backward()

    # Check for gradient computation
    if perturbed_image_clone.grad is None:
        raise ValueError("Gradient is not computed.")

    # Calculate the difference between the original and perturbed images
    difference = torch.abs(perturbed_image_clone.grad.data)

    # Calculate the dynamic threshold
    mean_diff = torch.mean(difference)
    std_diff = torch.std(difference)
    dynamic_threshold = mean_diff + k * std_diff

    # Create a binary mask
    mask = difference > dynamic_threshold

    # Separate the attacked and unattacked parts
    attacked_part = perturbed_image.clone()
    unattacked_part = original_image.clone()

    # Extract the attacked part from the original image
    attacked_part[mask] = original_image[mask]

    # Extract the unattacked part from the perturbed image
    unattacked_part[~mask] = perturbed_image[~mask]

    return attacked_part, unattacked_part

