"""Face encoder with dual heads for embedding and attribute classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FaceEncoder(nn.Module):
    """Face encoder with dual-head architecture.

    Architecture:
        Input: (B, 3, 256, 256)
        Conv2d(3, 64, 7, stride=2, pad=3) + BN + ReLU + MaxPool  → (B, 64, 64, 64)
        Conv2d(64, 128, 3, pad=1) + BN + ReLU                    → (B, 128, 64, 64)
        Conv2d(128, 128, 3, stride=2, pad=1) + BN + ReLU         → (B, 128, 32, 32)
        Conv2d(128, 256, 3, pad=1) + BN + ReLU                   → (B, 256, 32, 32)
        Conv2d(256, 256, 3, stride=2, pad=1) + BN + ReLU         → (B, 256, 16, 16)
        Conv2d(256, 512, 3, pad=1) + BN + ReLU                   → (B, 512, 16, 16)
        GlobalAvgPool                                             → (B, 512)
        FC(512, 128) → embedding
        FC(512, 40)  → attributes
    """

    def __init__(self, embedding_dim: int = 128, num_attributes: int = 40):
        """Initialize Face Encoder.

        Args:
            embedding_dim: Dimension of the embedding vector
            num_attributes: Number of face attributes to predict
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attributes = num_attributes

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 256 -> 64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 64

            # Block 2: 64 -> 32
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1),

            # Block 3: 32 -> 16
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1),

            # Block 4: 16 -> 16
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
        )

        # Attribute head
        self.attribute_head = nn.Sequential(
            nn.Linear(512, num_attributes),
        )

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 256, 256)

        Returns:
            Dictionary with 'embedding' and 'attributes'
        """
        # Extract features
        features = self.features(x)

        # Global average pooling
        pooled = self.gap(features)
        pooled = pooled.view(pooled.size(0), -1)

        # Get embedding (L2 normalized for cosine similarity)
        embedding = self.embedding_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Get attribute predictions
        attributes = self.attribute_head(pooled)

        return {
            'embedding': embedding,
            'attributes': attributes,
            'features': features  # For Grad-CAM
        }

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the embedding vector.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Embedding tensor of shape (B, embedding_dim)
        """
        features = self.features(x)
        pooled = self.gap(features)
        pooled = pooled.view(pooled.size(0), -1)
        embedding = self.embedding_head(pooled)
        return F.normalize(embedding, p=2, dim=1)

    def get_last_conv_layer(self) -> nn.Module:
        """Get the last convolutional layer for Grad-CAM."""
        # Return the last ConvBlock's conv layer
        return self.features[-1].conv


class EncoderLoss(nn.Module):
    """Combined loss for encoder training."""

    def __init__(self, attribute_weight: float = 1.0):
        """Initialize encoder loss.

        Args:
            attribute_weight: Weight for attribute loss
        """
        super().__init__()
        self.attribute_weight = attribute_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss.

        Args:
            outputs: Model outputs with 'attributes' key
            targets: Ground truth attributes of shape (B, 40)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Attribute loss
        attr_loss = self.bce_loss(outputs['attributes'], targets)

        # Total loss
        total_loss = self.attribute_weight * attr_loss

        loss_dict = {
            'attr_loss': attr_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


def compute_attribute_accuracy(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               threshold: float = 0.5) -> Dict[str, float]:
    """Compute attribute prediction accuracy.

    Args:
        predictions: Predicted logits of shape (B, 40)
        targets: Ground truth attributes of shape (B, 40)
        threshold: Threshold for binary classification

    Returns:
        Dictionary with accuracy metrics
    """
    # Apply sigmoid and threshold
    pred_binary = (torch.sigmoid(predictions) > threshold).float()

    # Per-attribute accuracy
    correct = (pred_binary == targets).float()
    per_attr_acc = correct.mean(dim=0)  # (40,)

    # Mean accuracy across all attributes
    mean_acc = correct.mean().item()

    return {
        'mean_accuracy': mean_acc,
        'per_attribute_accuracy': per_attr_acc.cpu().numpy()
    }
