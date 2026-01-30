"""FaceEncoder with MobileNetV2 backbone for face embeddings and attribute prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class FaceEncoder(nn.Module):
    """Face encoder with MobileNetV2 backbone and three output heads.

    Architecture:
        Input: (B, 3, 256, 256)
            ↓
        MobileNetV2.features → (B, 1280, 8, 8)
            ↓
        AdaptiveAvgPool2d(1) → (B, 1280)
            ↓
        ┌────────────┬──────────────┬─────────────┐
        │ Embed Head │ Proj Head    │ Attr Head   │
        │ L2-norm    │ L2-norm      │ (logits)    │
        └────────────┴──────────────┴─────────────┘
            ↓               ↓              ↓
          (B,64)         (B,64)         (B,40)

    The embedding head produces L2-normalized embeddings for similarity search.
    The projection head is used during SimCLR contrastive training.
    The attribute head predicts 40 binary facial attributes (CelebA).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        projection_dim: int = 64,
        num_attributes: int = 40,
        freeze_layers: int = 14,
    ):
        """Initialize FaceEncoder.

        Args:
            embedding_dim: Output dimension for face embeddings (default: 64).
            projection_dim: Output dimension for SimCLR projection head (default: 64).
            num_attributes: Number of binary attributes to predict (default: 40).
            freeze_layers: Number of MobileNetV2 feature blocks to freeze (default: 14).
                           Blocks 0-14 are frozen, blocks 15-18 are trainable.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_attributes = num_attributes
        self.freeze_layers = freeze_layers

        # Load pretrained MobileNetV2 backbone
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # 19 blocks, outputs (B, 1280, H/32, W/32)

        # Freeze early layers (0 to freeze_layers inclusive)
        self._freeze_backbone_layers()

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Embedding head: 1280 → 256 → 64 with L2 normalization
        self.embedding_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

        # Projection head for SimCLR: 64 → 128 → 64 with L2 normalization
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, projection_dim),
        )

        # Attribute prediction head: 1280 → 40
        self.attribute_head = nn.Linear(1280, num_attributes)

    def _freeze_backbone_layers(self):
        """Freeze MobileNetV2 layers 0 to freeze_layers (inclusive)."""
        for i, layer in enumerate(self.features):
            if i <= self.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(
        self, x: torch.Tensor, return_projection: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass through the encoder.

        Args:
            x: Input tensor of shape (B, 3, 256, 256).
            return_projection: If True, also return projection embeddings for SimCLR.

        Returns:
            If return_projection=False:
                (embedding, attr_logits) where:
                    - embedding: L2-normalized face embeddings (B, embedding_dim)
                    - attr_logits: Attribute prediction logits (B, num_attributes)

            If return_projection=True:
                (embedding, projection, attr_logits) where:
                    - embedding: L2-normalized face embeddings (B, embedding_dim)
                    - projection: L2-normalized SimCLR projections (B, projection_dim)
                    - attr_logits: Attribute prediction logits (B, num_attributes)
        """
        # Extract features from backbone
        features = self.features(x)  # (B, 1280, 8, 8) for 256x256 input

        # Global average pooling
        pooled = self.pool(features)  # (B, 1280, 1, 1)
        pooled = pooled.flatten(1)  # (B, 1280)

        # Embedding head with L2 normalization
        embedding = self.embedding_head(pooled)  # (B, embedding_dim)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Attribute prediction (from pooled features, not embedding)
        attr_logits = self.attribute_head(pooled)  # (B, num_attributes)

        if return_projection:
            # Projection head with L2 normalization (for SimCLR)
            projection = self.projection_head(embedding)  # (B, projection_dim)
            projection = F.normalize(projection, p=2, dim=1)
            return embedding, projection, attr_logits

        return embedding, attr_logits

    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_params(self) -> int:
        """Return the number of frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
