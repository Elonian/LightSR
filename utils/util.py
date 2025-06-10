import math
import torch
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import skimage.color as sc  # For RGB?YCbCr conversion

def ndarray2tensor(ndarray):
    # Optionally convert RGB to Y here if needed
    tensor = torch.from_numpy(ndarray.transpose((2, 0, 1))).float() / 255.0
    return tensor


def calc_psnr(sr, hr, max_val=255.0):
    # Convert tensors [C, H, W] ? [H, W, C] ndarrays
    sr_np = sr.cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.cpu().numpy().transpose(1, 2, 0)

    # Compute Y channel only
    if sr_np.shape[2] == 3:
        sr_y = sc.rgb2ycbcr((sr_np * max_val).astype(np.uint8))[..., 0]
        hr_y = sc.rgb2ycbcr((hr_np * max_val).astype(np.uint8))[..., 0]
    else:
        sr_y = (sr_np[..., 0] * max_val).astype(np.uint8)
        hr_y = (hr_np[..., 0] * max_val).astype(np.uint8)

    mse = np.mean((sr_y.astype(np.float64) - hr_y.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

def calc_ssim(sr, hr, data_range=255.0, size_average=True):
    sr_np = sr.cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.cpu().numpy().transpose(1, 2, 0)

    if sr_np.shape[2] == 3:
        sr_y = sc.rgb2ycbcr((sr_np * data_range).astype(np.uint8))[..., 0]
        hr_y = sc.rgb2ycbcr((hr_np * data_range).astype(np.uint8))[..., 0]
    else:
        sr_y = (sr_np[..., 0] * data_range).astype(np.uint8)
        hr_y = (hr_np[..., 0] * data_range).astype(np.uint8)

    val = ssim(sr_y, hr_y, data_range=data_range)
    return val if size_average else np.mean(val)