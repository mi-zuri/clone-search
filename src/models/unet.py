"""U-Net model for face inpainting."""

import torch
import torch.nn as nn
from typing import List


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for face inpainting.

    Takes 4-channel input (RGB + mask) and outputs 3-channel RGB.

    Architecture:
        Encoder: 4 blocks (64→128→256→512)
        Bottleneck: 512→1024→512
        Decoder: 4 blocks with skip connections
        Output: Conv(64, 3, 1)
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, bilinear: bool = True):
        """Initialize U-Net.

        Args:
            in_channels: Number of input channels (3 RGB + 1 mask = 4)
            out_channels: Number of output channels (3 RGB)
            bilinear: Use bilinear upsampling (True) or transposed conv (False)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 4, H, W) - masked RGB + mask channel

        Returns:
            Output tensor of shape (B, 3, H, W)
        """
        # Encoder
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024 (or 512 if bilinear)

        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512
        x = self.up2(x, x3)   # 256
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 64

        # Output
        logits = self.outc(x)

        # Sigmoid to ensure output is in [0, 1]
        return torch.sigmoid(logits)


class InpaintingLoss(nn.Module):
    """Loss function for inpainting task."""

    def __init__(self, l1_weight: float = 1.0, mask_weight: float = 6.0):
        """Initialize inpainting loss.

        Args:
            l1_weight: Weight for L1 reconstruction loss
            mask_weight: Extra weight for masked region loss
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.mask_weight = mask_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Compute inpainting loss.

        Args:
            pred: Predicted image (B, 3, H, W)
            target: Ground truth image (B, 3, H, W)
            mask: Binary mask where 1 = masked region (B, 1, H, W)

        Returns:
            Total loss
        """
        # L1 loss for entire image
        l1_loss = torch.abs(pred - target)

        # Weight masked region more heavily
        weighted_loss = l1_loss * (1 + (self.mask_weight - 1) * mask)

        return self.l1_weight * weighted_loss.mean()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image
        target: Ground truth image
        max_val: Maximum pixel value

    Returns:
        PSNR in dB
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, size_average: bool = True) -> float:
    """Compute Structural Similarity Index.

    Args:
        pred: Predicted image (B, C, H, W)
        target: Ground truth image (B, C, H, W)
        window_size: Size of the Gaussian window
        size_average: Average over all pixels

    Returns:
        SSIM value
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_target = nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = nn.functional.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = nn.functional.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    return ssim_map.mean(dim=[1, 2, 3]).item()
