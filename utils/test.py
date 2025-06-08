
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import math


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


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
        ffted_real, ffted_imag = ffted.real, ffted.imag
        ffted = torch.cat([ffted_real, ffted_imag], dim=1)  # (B, 2C, H, W//2+1)

        ffted = self.act(self.conv(ffted))
        ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(ffted_real, ffted_imag)
        out = torch.fft.irfft2(ffted, s=(H, W), norm=self.fft_norm)
        return out


class FConvMod(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []
        for a, v in zip(a_all, v_all):
            attn = a * v
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut
        
        return out


def test_layernorm_channels_first():
    x = torch.randn(2, 8, 16, 16)
    ln = LayerNorm(8, data_format='channels_first')
    out = ln(x)
    
    # Mean should be approximately zero
    mean = out.mean(dim=1)
    assert torch.allclose(mean.mean(), torch.tensor(0.0), atol=1e-5)

    # Std should be approximately 1
    std = out.std(dim=1, unbiased=False)
    assert torch.allclose(std.mean(), torch.tensor(1.0), atol=1e-5)
    print("LayerNorm channels_first passed.")

def test_layernorm_channels_last():
    x = torch.randn(2, 16, 16, 8)
    ln = LayerNorm(8, data_format='channels_last')
    out = ln(x)

    # Use the same weight and bias parameters for the reference
    ref = F.layer_norm(x, (8,), ln.weight, ln.bias, ln.eps)
    assert torch.allclose(out, ref, atol=1e-5), "LayerNorm channels_last output mismatch"
    print("LayerNorm channels_last passed.")


def test_fconvmod_forward():
    B, C, H, W = 2, 8, 32, 32
    x = torch.randn(B, C, H, W)
    model = FConvMod(dim=C, num_heads=2)
    out = model(x)
    
    assert out.shape == (B, C, H, W), f"Expected shape {(B, C, H, W)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"
    print("FConvMod forward pass passed.")

if __name__ == "__main__":
    test_layernorm_channels_first()
    test_layernorm_channels_last()
    test_fconvmod_forward()
