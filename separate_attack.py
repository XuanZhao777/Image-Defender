import torch
import torch.nn as nn
from torch.autograd import Variable

def separate_attack(model, original_image, perturbed_image, threshold=0.1):
    # 将图像转换为Variable
    original_image_var = Variable(original_image, requires_grad=False)
    perturbed_image_var = Variable(perturbed_image, requires_grad=True)

    # 获取模型的输出
    original_output = model(original_image_var)
    perturbed_output = model(perturbed_image_var)

    # 计算损失（可以根据具体任务选择损失函数）
    criterion = nn.CrossEntropyLoss()
    loss = criterion(perturbed_output, torch.argmax(original_output, dim=1))

    # 计算梯度
    loss.backward()

    # 确保梯度计算完毕
    if perturbed_image_var.grad is None:
        raise ValueError("Gradient is not computed. Ensure perturbed_image has requires_grad=True.")

    # 计算原始图像和受到攻击的图像之间的差异
    difference = torch.abs(perturbed_image_var.grad.data)

    # 将差异与阈值比较，得到二值掩码
    mask = difference > threshold

    # 提取受到攻击的部分和未受到攻击的部分
    attacked_part = perturbed_image.clone()
    attacked_part[mask] = 1  # 用白色表示受到攻击的部分

    unattacked_part = perturbed_image.clone()
    unattacked_part[~mask] = 0  # 用黑色表示未受到攻击的部分

    return attacked_part, unattacked_part
