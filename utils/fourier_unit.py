import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierUnit(nn.Module):
    def __init__(self, dim, fft_norm='ortho'):
        super().__init__()
        self.fft_norm = fft_norm
        self.conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        # real FFT -> complex tensor (B, C, H, W//2+1)
        ffted = torch.fft.rfft2(x, norm=self.fft_norm)  # (B, C, H, W//2+1)
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        ffted = torch.cat([ffted_real, ffted_imag], dim=1)  # (B, 2C, H, W//2+1)
        ffted = self.act(self.conv(ffted))
        ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(ffted_real, ffted_imag)
        out = torch.fft.irfft2(ffted, s=(H, W), norm=self.fft_norm)
        return out
