import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def ndarray2tensor(ndarray):
    tensor = torch.from_numpy(ndarray.transpose((2, 0, 1))).float()
    # Normalize pixel values to [0,1]
    tensor = tensor / 255.0
    return tensor

def calc_psnr(sr, hr, max_val=255.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if MSE is zero
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calc_ssim(sr, hr, data_range=255.0, size_average=True):
    sr_np = sr.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
    hr_np = hr.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
    sr_np = (sr_np * data_range).astype(np.uint8)
    hr_np = (hr_np * data_range).astype(np.uint8)
    ssim_value = ssim(sr_np, hr_np, multichannel=True, data_range=data_range)
    return ssim_value if size_average else np.mean(ssim_value)