import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierUnit(nn.Module):
    def __init__(self, channels, fft_norm_mode='ortho'):
        super(FourierUnit, self).__init__()
        self.fft_norm_mode = fft_norm_mode
        self.complex_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W), real-valued input feature maps
        Returns:
            Tensor of shape (B, C, H, W), real-valued output after Fourier modulation
        """
        batch_size, channels, height, width = x.shape

        fft_transformed = torch.fft.rfft2(x, norm=self.fft_norm_mode)
        real_part = fft_transformed.real
        imag_part = fft_transformed.imag
        complex_features = torch.cat([real_part, imag_part], dim=1)

        processed = self.activation(self.complex_conv(complex_features))

        real_processed, imag_processed = torch.chunk(processed, 2, dim=1)
        fft_processed = torch.complex(real_processed, imag_processed)

        output = torch.fft.irfft2(fft_processed, s=(height, width), norm=self.fft_norm_mode)

        return output
