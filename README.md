# Image-Defender
Workflow:
![IMG_2044(20231202-192545)](https://github.com/XuanZhao777/Image-Defender/assets/149707203/5a20a134-1f24-4bac-99b7-c3bab1fab41d)

# Database:
CIFAR10

# Attack Part:
Iterative Fast Gradient Sign Method (I-FGSM) attack, which is a variant of the Fast Gradient Sign Method (FGSM). The I-FGSM attack is an iterative approach where, in each iteration, a small perturbation is added to the input image in the direction that maximizes the loss. This process is repeated for a specified number of iterations.

# Separation Part:
This method utilizes gradient information and a dynamic threshold to identify attacked pixels. Subsequently, it replaces the attacked portion in the original image with the corresponding portion from the attacked image, creating two separated parts.

# Correction Part:
This is a form of correction method based on the Mean Squared Error (MSE) loss. The idea is to adjust the attacked image by minimizing the mean squared error between the original image and the attacked image. The method utilizes gradient descent over a series of iterations to update the attacked image, aiming to reduce the difference between the original and attacked images. The ultimate goal is to make the attacked image closely resemble the original image through iterative optimization.

# Combination Part:
Simple combination, currently unimproved.
Moreover, the final combined image is just the corrected image. It is not a combination of the unattacked and corrected parts.
This is a fake code.

# Validation Part:
Validation using the PSNR method to assess the accuracy between the combined image and the original image

In reality, the attacked image is not visually distinguishable. However, I visualized the attacked parts, which may result in colored spots in the final combined image. Although it affects the visual output, it helps identify areas that were attacked but not corrected.

# Make an App:
Executing the command "pyinstaller --onefile App.py" in the terminal will package the program into a standalone executable.



