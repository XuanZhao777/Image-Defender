import numpy as np
import cv2

def test(combined_image, original_image):
    # 将PyTorch Tensor转换为NumPy数组
    combined_array = combined_image.detach().cpu().numpy()
    original_array = original_image.detach().cpu().numpy()

    # 计算均方误差
    mse = np.mean((original_array - combined_array) ** 2)

    # 计算PSNR
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    # 打印评估结果
    print("<70dB: Poor quality,\n"
          ">70dB: Good quality")
    print(f'Our PSNR: {psnr:.2f} dB')

    # 返回PSNR值
    return psnr
