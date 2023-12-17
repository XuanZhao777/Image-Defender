import numpy as np
import cv2

def test(combined_image, original_image):
    # Convert PyTorch Tensor to NumPy arrays
    combined_array = combined_image.detach().cpu().numpy()
    original_array = original_image.detach().cpu().numpy()

    # Calculate Mean Squared Error
    mse = np.mean((original_array - combined_array) ** 2)

    # Calculate PSNR
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    # Print evaluation results
    print("<70dB: Poor quality,\n"
          ">70dB: Good quality")
    print(f'Our PSNR: {psnr:.2f} dB')

    # Return the PSNR value
    return psnr
