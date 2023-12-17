import torch
import torch.nn as nn

def i_fgsm_attack(image, epsilon, data_grad):
    # Generate perturbation and directly add it to the image, while constraining pixel values within the [0, 1] range
    perturbed_image = torch.clamp(image + epsilon * data_grad.sign(), 0, 1)
    return perturbed_image

def adversarial_attack(model, image, label, epsilon, alpha, iterations):
    """
    Iterative Fast Gradient Sign Method (I-FGSM) Attack

    Parameters:
    model: The model being attacked.
    image: Input image.
    label: True label corresponding to the image.
    epsilon: Total perturbation budget.
    alpha: Perturbation step size for each iteration.
    iterations: Total number of iterations.
    """
    # Create a copy of the original image to create a leaf node
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True

    for i in range(iterations):
        # Forward pass
        output = model(perturbed_image)

        # Calculate loss
        loss = torch.nn.CrossEntropyLoss()(output, label)

        # Backward pass, compute gradients
        model.zero_grad()
        loss.backward()

        # Get the gradients of the image
        data_grad = perturbed_image.grad.data

        # Clear the current gradients
        perturbed_image.grad = None

        # Call I-FGSM attack to compute the new perturbed image
        perturbed_image = i_fgsm_attack(perturbed_image, alpha, data_grad)

        # Clip the perturbation within the epsilon range
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)

        # Reset requires_grad for the next iteration
        perturbed_image = perturbed_image.detach().clone()
        perturbed_image.requires_grad = True

    # Clip the pixel values of the final perturbed image within the [0, 1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

