import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, dim, kernel_size, groups=1, num_kernels=1, bias=True, reduction=8, init_weight=True):
        super().__init__()
        assert dim % groups == 0
        self.dim = dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.num_kernels = num_kernels
        self.use_dynamic = num_kernels > 1

        if self.use_dynamic:
            # Kernel Attention
            if dim != 3:
                mid_channels = dim // reduction
            else:
                mid_channels = num_kernels

            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, mid_channels, 1),
                nn.GELU(),
                nn.Conv2d(mid_channels, num_kernels, 1),
                nn.Sigmoid()
            )

            # Kernel weights and biases
            self.weight = nn.Parameter(
                torch.randn(num_kernels, dim, dim // groups, kernel_size, kernel_size),
                requires_grad=True
            )
            if bias:
                self.bias = nn.Parameter(torch.zeros(num_kernels, dim))
            else:
                self.bias = None

            if init_weight:
                self._initialize_weights()
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=groups,
                                  padding=kernel_size // 2, bias=bias)

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        if not self.use_dynamic:
            return self.conv(x)

        B, C, H, W = x.shape

        # Generate attention weights
        attn = self.attention(x)
        attn = attn.view(B, self.num_kernels)
        # Reshape input for grouped conv
        x_reshaped = x.contiguous().view(1, B * C, H, W)
        # Aggregate kernel weights
        weight_reshaped = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attn, weight_reshaped).view(B * C, C // self.groups, self.kernel_size, self.kernel_size)
        # Aggregate bias
        if self.bias is not None:
            bias = torch.mm(attn, self.bias).view(-1)
            out = F.conv2d(x_reshaped, weight=weight, bias=bias, stride=1,
                           padding=self.kernel_size // 2, groups=B * self.groups)
        else:
            out = F.conv2d(x_reshaped, weight=weight, bias=None, stride=1,
                           padding=self.kernel_size // 2, groups=B * self.groups)

        out = out.view(B, C, H, W)
        return out

# if __name__ == "__main__":
#     B, C, H, W = 2, 64, 32, 32
#     x = torch.randn(B, C, H, W)

#     model = Aggregator(dim=C, kernel_size=3, groups=C, num_kernels=4)
#     y = model(x)
#     print("Dynamic:", y.shape)

#     model_static = Aggregator(dim=C, kernel_size=3, groups=C, num_kernels=1)
#     y_static = model_static(x)
#     print("Static:", y_static.shape)