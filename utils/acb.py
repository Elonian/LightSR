import torch
import torch.nn as nn
from utils.fma import FourierModulatedAttention
from utils.dml import DynamicMixingLayer

class AttentiveConvBlock(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=8, kernel_sizes=[5, 7]):
        super().__init__()
        self.fma = FourierModulatedAttention(dim, num_heads, patch_size)
        self.dml = DynamicMixingLayer(dim, kernel_sizes)

    def forward(self, x):
        x = self.fma(x)
        x = self.dml(x)
        return x

if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)
    acb = AttentiveConvBlock(dim=C)
    y = acb(x)
    assert y.shape == x.shape
    print("ACB test passed. Output shape:", y.shape)