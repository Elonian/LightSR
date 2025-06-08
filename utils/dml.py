import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.layernorm as LayerNorm
from utils.aggregate import Aggregator

class DynamicMixingLayer(nn.Module):
    def __init__(self, in_channels, num_kernels = 16, kernel_sizes=[5, 7]):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be divisible by 2 for channel split"

        self.norm = nn.LayerNorm(in_channels)
        self.expand = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

        self.branch1 = Aggregator(in_channels, kernel_size=kernel_sizes[0], groups =in_channels, num_kernels=num_kernels, bias=True)
        self.branch2 = Aggregator(in_channels, kernel_size=kernel_sizes[1], groups =in_channels, num_kernels=num_kernels, bias=True)

        self.merge = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x).permute(0, 3, 1, 2)
        x = self.expand(x)

        x1, x2 = torch.chunk(x, 2, dim=1)
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)

        out = torch.cat([out1, out2], dim=1)
        out = self.merge(out)
        return out + residual

if __name__ == "__main__":
    def test_dml_layer():
        B, C, H, W = 2, 64, 32, 32
        kernel_sizes = [5, 7]

        assert C % 2 == 0, "C must be divisible by 2 for channel splitting"

        x = torch.randn(B, C, H, W)
        dml = DynamicMixingLayer(in_channels=C, kernel_sizes=kernel_sizes)
        y = dml(x)

        assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"
        print("✅ DML unit test passed. Output shape:", y.shape)


    # Run test
    test_dml_layer()