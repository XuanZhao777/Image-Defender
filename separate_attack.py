import torch
import matplotlib.pyplot as plt


def separate_attack(original_image, perturbed_image, threshold=0.1):
    # 计算对抗性扰动
    image_difference = perturbed_image - original_image

    # 应用阈值，将受到攻击的部分标记为1，未受到攻击的部分标记为0
    mask = torch.abs(image_difference) > threshold
    attacked_part = mask.float()



    plt.show()

    return attacked_part
