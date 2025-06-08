import torch
import numpy as np

def ndarray2tensor(ndarray):
    # Convert from HWC (height, width, channels) NumPy array to CHW tensor
    tensor = torch.from_numpy(ndarray.transpose((2, 0, 1))).float()
    # Normalize pixel values to [0,1]
    tensor = tensor / 255.0
    return tensor