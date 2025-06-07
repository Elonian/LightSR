import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.kernel_gen = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.GELU(),
            nn.Linear(in_channels // 4, in_channels * kernel_size * kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size

        pooled = self.gap(x).view(B, C)
        weights = self.kernel_gen(pooled).view(B * C, 1, K, K)

        x = x.reshape(1, B * C, H, W)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        out = F.conv2d(x, weights, groups=B * C)
        out = out.view(B, C, H, W)
        return out


class DynamicMixingLayer(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[5, 7]):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be divisible by 2 for channel split"

        self.norm = nn.LayerNorm(in_channels)
        self.expand = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

        self.branch1 = DynamicDepthwiseConv(in_channels, kernel_sizes[0])
        self.branch2 = DynamicDepthwiseConv(in_channels, kernel_sizes[1])

        self.merge = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x).permute(0, 3, 1, 2)
        x = self.expand(x)

        x1, x2 = torch.chunk(x, 2, dim=1)
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)

        out = torch.cat([out1, out2], dim=1)
        out = self.merge(out)
        return out

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