import torch
import torch.nn as nn
import torch.optim as optim

def correct_attack(model, original_image, perturbed_image, max_iterations=100, lr=0.01):
    # Ensure the model is in evaluation mode
    model.eval()

    # Clone the perturbed image and enable gradient computation
    perturbed_image = perturbed_image.clone().detach()
    perturbed_image.requires_grad = True

    # Choose an optimizer
    optimizer = optim.Adam([perturbed_image], lr=lr)

    for iteration in range(max_iterations):
        # Reset gradients
        optimizer.zero_grad()

        # Get the model outputs
        perturbed_output = model(perturbed_image)
        original_output = model(original_image)

        # Define the loss function
        loss = nn.MSELoss()(perturbed_output, original_output)

        # Backpropagation
        loss.backward()
        optimizer.step()


    return perturbed_image
