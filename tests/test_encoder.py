"""Tests for FaceEncoder architecture."""

import pytest
import torch

from src.models.encoder import FaceEncoder


class TestFaceEncoderOutputShapes:
    """Test output tensor shapes."""

    def test_embedding_shape(self):
        """Embedding output should be (B, 64)."""
        model = FaceEncoder()
        x = torch.randn(2, 3, 256, 256)
        embedding, attr_logits = model(x)
        assert embedding.shape == (2, 64)

    def test_attribute_logits_shape(self):
        """Attribute logits should be (B, 40)."""
        model = FaceEncoder()
        x = torch.randn(2, 3, 256, 256)
        embedding, attr_logits = model(x)
        assert attr_logits.shape == (2, 40)

    def test_projection_shape(self):
        """Projection output should be (B, 64) when requested."""
        model = FaceEncoder()
        x = torch.randn(2, 3, 256, 256)
        embedding, projection, attr_logits = model(x, return_projection=True)
        assert projection.shape == (2, 64)

    def test_custom_dimensions(self):
        """Test with custom embedding/projection/attribute dimensions."""
        model = FaceEncoder(
            embedding_dim=128,
            projection_dim=256,
            num_attributes=20,
        )
        x = torch.randn(4, 3, 256, 256)
        embedding, projection, attr_logits = model(x, return_projection=True)

        assert embedding.shape == (4, 128)
        assert projection.shape == (4, 256)
        assert attr_logits.shape == (4, 20)


class TestL2Normalization:
    """Test that embeddings and projections are L2 normalized."""

    def test_embedding_is_normalized(self):
        """Embedding vectors should have unit L2 norm."""
        model = FaceEncoder()
        x = torch.randn(4, 3, 256, 256)
        embedding, _ = model(x)

        norms = torch.norm(embedding, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_projection_is_normalized(self):
        """Projection vectors should have unit L2 norm."""
        model = FaceEncoder()
        x = torch.randn(4, 3, 256, 256)
        _, projection, _ = model(x, return_projection=True)

        norms = torch.norm(projection, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestLayerFreezing:
    """Test MobileNetV2 layer freezing."""

    def test_default_freezing(self):
        """Default: layers 0-14 frozen, layers 15-18 trainable."""
        model = FaceEncoder(freeze_layers=14)

        # Check frozen layers
        for i in range(15):
            for param in model.features[i].parameters():
                assert not param.requires_grad, f"Layer {i} should be frozen"

        # Check trainable layers
        for i in range(15, 19):
            for param in model.features[i].parameters():
                assert param.requires_grad, f"Layer {i} should be trainable"

    def test_all_heads_trainable(self):
        """All head layers should be trainable."""
        model = FaceEncoder()

        for name, param in model.embedding_head.named_parameters():
            assert param.requires_grad, f"Embedding head {name} should be trainable"

        for name, param in model.projection_head.named_parameters():
            assert param.requires_grad, f"Projection head {name} should be trainable"

        for name, param in model.attribute_head.named_parameters():
            assert param.requires_grad, f"Attribute head {name} should be trainable"

    def test_trainable_param_count(self):
        """Verify approximate trainable parameter count (~1.9M).

        Breakdown:
        - MobileNetV2 layers 15-18: ~1.5M
        - Embedding head (1280->256->64): ~0.35M
        - Projection head (64->128->64): ~0.02M
        - Attribute head (1280->40): ~0.05M
        """
        model = FaceEncoder()
        trainable = model.get_trainable_params()
        frozen = model.get_frozen_params()

        # Trainable should be ~1.9M (layers 15-18 + heads)
        assert 1_800_000 < trainable < 2_100_000, f"Trainable params: {trainable}"

        # Frozen should be ~700K (layers 0-14)
        assert 600_000 < frozen < 800_000, f"Frozen params: {frozen}"


class TestForwardBackward:
    """Test forward and backward passes."""

    def test_forward_pass(self):
        """Forward pass should run without error."""
        model = FaceEncoder()
        x = torch.randn(2, 3, 256, 256)

        # Without projection
        embedding, attr_logits = model(x)
        assert embedding is not None
        assert attr_logits is not None

        # With projection
        embedding, projection, attr_logits = model(x, return_projection=True)
        assert embedding is not None
        assert projection is not None
        assert attr_logits is not None

    def test_backward_pass(self):
        """Backward pass should compute gradients without error."""
        model = FaceEncoder()
        x = torch.randn(2, 3, 256, 256)

        embedding, projection, attr_logits = model(x, return_projection=True)

        # Compute a simple loss
        loss = embedding.sum() + projection.sum() + attr_logits.sum()
        loss.backward()

        # Check that gradients exist for trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_1(self):
        """Should work with batch size 1."""
        model = FaceEncoder()
        x = torch.randn(1, 3, 256, 256)
        embedding, attr_logits = model(x)

        assert embedding.shape == (1, 64)
        assert attr_logits.shape == (1, 40)

    def test_different_input_sizes(self):
        """Should work with different input sizes (backbone is fully convolutional)."""
        model = FaceEncoder()

        # 224x224 (standard ImageNet size)
        x = torch.randn(2, 3, 224, 224)
        embedding, attr_logits = model(x)
        assert embedding.shape == (2, 64)

        # 128x128
        x = torch.randn(2, 3, 128, 128)
        embedding, attr_logits = model(x)
        assert embedding.shape == (2, 64)


class TestModelModes:
    """Test model evaluation and training modes."""

    def test_eval_mode(self):
        """Model should work in eval mode."""
        model = FaceEncoder()
        model.eval()

        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            embedding, attr_logits = model(x)

        assert embedding.shape == (2, 64)
        assert attr_logits.shape == (2, 40)

    def test_train_mode(self):
        """Model should work in train mode."""
        model = FaceEncoder()
        model.train()

        x = torch.randn(2, 3, 256, 256)
        embedding, attr_logits = model(x)

        assert embedding.shape == (2, 64)
        assert attr_logits.shape == (2, 40)
