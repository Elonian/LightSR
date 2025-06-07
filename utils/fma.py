import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fourier_unit import FourierUnit

class FourierModulatedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=8):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.spectral = FourierUnit(dim)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ln = x.permute(0,2,3,1)
        x_ln = self.norm(x_ln).permute(0,3,1,2)
        A = self.spectral(x_ln)
        V = self.value(x_ln)
        head_dim = C // self.num_heads
        A = A.view(B, self.num_heads, head_dim, H*W)
        V = V.view(B, self.num_heads, head_dim, H*W)
        p = self.patch_size
        num_patches = (H*W) // (p*p)
        A_patches = A.chunk(num_patches, dim=-1)
        V_patches = V.chunk(num_patches, dim=-1)
        attn_chunks = [ap * vp for ap, vp in zip(A_patches, V_patches)]
        attn = torch.cat(attn_chunks, dim=-1)
        attn = F.softmax(attn, dim=-1)
        out = attn.view(B, C, H, W)
        out = out + self.cpe(x)
        out = self.proj(out) + x
        return out