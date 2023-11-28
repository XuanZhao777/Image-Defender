import torch


def correct_attack(original_image, attacked_part, correction_factor=0.5):
    # 复制原始图像
    corrected_image = original_image.clone()

    # 对受到攻击的部分进行校正恢复
    corrected_image[attacked_part == 1] = original_image[attacked_part == 1] * correction_factor

    return corrected_image
