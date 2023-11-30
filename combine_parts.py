import torch
import numpy as np

def combine_parts(original_image, not_attacked_part, corrected_part):
    # 将 PyTorch 张量转换为 NumPy 数组以便更容易进行操作
    original_array = original_image.detach().cpu().numpy()
    not_attacked_array = not_attacked_part.detach().cpu().numpy()
    corrected_array = corrected_part.detach().cpu().numpy()

    # 找到校正部分和原始图像之间的差异
    differences = np.where(not_attacked_array  != original_array)

    # 用 not_attacked_array 中的相应部分替换 corrected_array 中的不同部分
    combined_array = np.copy(not_attacked_array)
    combined_array[differences] = corrected_array[differences]

    # 将合并后的数组转换回 PyTorch 张量
    combined_tensor = torch.from_numpy(combined_array)

    return combined_tensor

