import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fourier_unit import FourierUnit
from utils.layernorm import LayerNorm

class FourierModulatedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_chunks = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_chunks = num_chunks
        
        self.norm = LayerNorm(dim)
        self.spectral_unit = FourierUnit(dim)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.positional_encoding = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
        num_chunks: int, number of chunks to split spatial dimension into for local attention
        """

        B, C, H, W = x.shape
        spatial_size = H * W

        residual = x
        x_norm = self.norm(x)
        spectral_features = self.spectral_unit(x_norm)
        value_features = self.value_proj(x_norm)

        spectral_reshaped = spectral_features.view(B, self.num_heads, self.head_dim, spatial_size)
        value_reshaped = value_features.view(B, self.num_heads, self.head_dim, spatial_size)

        # Safety: Clamp num_chunks between 1 and spatial_size to avoid invalid splits
        num_chunks = max(1, min(self.num_chunks, spatial_size))

        chunk_len = math.ceil(spatial_size / num_chunks)
        spectral_chunks = torch.split(spectral_reshaped, chunk_len, dim=-1)
        value_chunks = torch.split(value_reshaped, chunk_len, dim=-1)

        attention_chunks = []
        for s_chunk, v_chunk in zip(spectral_chunks, value_chunks):
            attention_chunk = s_chunk * v_chunk
            attention_chunks.append(attention_chunk)

        attention = torch.cat(attention_chunks, dim=-1)
        attention = F.softmax(attention, dim=-1)
        attention_reshaped = attention.view(B, C, H, W)

        attention_with_pos = attention_reshaped + self.positional_encoding(x)
        projected = self.output_proj(attention_with_pos)
        out = projected + residual

        return out
