import torch


def combine_parts(original_image, not_attacked_part, corrected_part):
    # 复制原始图像
    combined_image = original_image.clone()

    # 将未受攻击的部分和校正后的部分结合在一起
    combined_image[not_attacked_part == 1] = corrected_part[not_attacked_part == 1]

    return combined_image
