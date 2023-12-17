import torch
import numpy as np

def combine_parts(original_image, not_attacked_part, corrected_part):
    # Convert PyTorch tensors to NumPy arrays
    original_array = original_image.detach().cpu().numpy()
    not_attacked_array = not_attacked_part.detach().cpu().numpy()
    corrected_array = corrected_part.detach().cpu().numpy()

    # Ensure all arrays are 3-dimensional (C, H, W)
    if original_array.ndim == 4:
        original_array = original_array.squeeze(0)
    if not_attacked_array.ndim == 4:
        not_attacked_array = not_attacked_array.squeeze(0)
    if corrected_array.ndim == 4:
        corrected_array = corrected_array.squeeze(0)

    # Print shapes for debugging purposes
    print("Original array shape:", original_array.shape)
    print("Not attacked array shape:", not_attacked_array.shape)
    print("Corrected array shape:", corrected_array.shape)

    # Ensure all arrays have the same shape
    if original_array.shape != not_attacked_array.shape or original_array.shape != corrected_array.shape:
        raise ValueError("All input arrays must have the same shape.")

    # Directly add the not attacked and corrected parts and subtract the overlapping parts
    combined_array = not_attacked_array + corrected_array - original_array

    # Clip the result to ensure values are within a legal range
    combined_array = np.clip(combined_array, 0, 1)

    # Convert the combined array back to a PyTorch tensor
    combined_tensor = torch.from_numpy(combined_array)

    return combined_tensor



