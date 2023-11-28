import torch
import torch.nn as nn

def fgsm_attack(image, epsilon, data_grad):
    # 获取输入图像的梯度方向
    sign_data_grad = data_grad.sign()
    # 生成扰动
    perturbed_image = image + epsilon * sign_data_grad
    # 将图像像素值限制在合法范围内
    scaled_perturbation = epsilon * sign_data_grad
    perturbed_image = image + scaled_perturbation

    # 将图像像素值限制在 [0, 1] 范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def adversarial_attack(model, image, label, epsilon):
    # 使输入图像的梯度计算可用
    image.requires_grad = True
    # 正向传播
    output = model(image)
    # 计算损失
    loss = nn.CrossEntropyLoss()(output, label)
    # 反向传播，计算梯度
    model.zero_grad()
    loss.backward()
    # 获取图像的梯度
    data_grad = image.grad.data
    # 执行对抗攻击
    perturbed_image = fgsm_attack(image, epsilon, data_grad)
    return perturbed_image
