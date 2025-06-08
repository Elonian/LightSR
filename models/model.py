import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils.acb import AttentiveConvBlock

class SRConvnet(nn.Module):
    def __init__(self, scale, num_heads = 8, num_kernels = 8, dimension = 64, num_acb = 8):

        super().__init__()
        self.scale = scale

        self.conv1 = nn.Conv2d(3, dimension, kernel_size=3, padding=1)

        self.acb_layers = nn.ModuleList([
            AttentiveConvBlock(dim=dimension, num_heads=num_heads, num_chunks=4, kernel_sizes=[5, 7], num_kernels=num_kernels)
            for _ in range(num_acb)
        ])

        self.conv2 = nn.Conv2d(dimension, dimension * self.scale * self.scale, kernel_size=1, padding=0)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        self.conv2_5 = nn.Conv2d(dimension, dimension * 4, kernel_size=1, padding=0)
        self.pixel_shuffle_2_5 = nn.PixelShuffle(2)

        self.activation = nn.GELU()
        self.conv3 = nn.Conv2d(dimension, 3, kernel_size=3, padding=1)
        self.interpolate = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)


    def forward(self, x):
        residual = x

        x = self.conv1(x)
        
        mid_residual = x
        for layer in self.acb_layers:
            x = layer(x)
        x = x + mid_residual

        if self.scale == 4: ## Special case for scale 4 for stability
            x = self.activation(self.pixel_shuffle_2_5(self.conv2_5(x)))
            x = self.activation(self.pixel_shuffle_2_5(self.conv2_5(x)))
        else:
            x = self.activation(self.pixel_shuffle(self.conv2(x)))

        x = self.conv3(x)

        residual = self.interpolate(residual)

        return x + residual

def test_srconvnet():
    scale = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRConvnet(scale=scale, num_kernels = 8, num_acb = 8).to(device)
    model.eval()

    print("Model Architecture:")
    summary(model, input_size=(3, 64, 64), device="cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 3, 64, 64).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_srconvnet()


