# Image-Defender
Workflow:
![IMG_2044(20231202-192545)](https://github.com/XuanZhao777/Image-Defender/assets/149707203/5a20a134-1f24-4bac-99b7-c3bab1fab41d)

# Database:
CIFAR10

# Attack Part:
FGSM Attack

# Separation Part:
ResNet18 Model

# Correction Part:
Simple separation, currently unimproved

# Combination Part:
Simple combination, currently unimproved.
Moreover, the final combined image is just the corrected image. It is not a combination of the unattacked and corrected parts.
This is a fake code.

# Validation Part:
Validation using the PSNR method to assess the accuracy between the combined image and the original image

In reality, the attacked image is not visually distinguishable. However, I visualized the attacked parts, which may result in colored spots in the final combined image. Although it affects the visual output, it helps identify areas that were attacked but not corrected.

# Make an App:
Executing the command "pyinstaller --onefile App.py" in the terminal will package the program into a standalone executable.



