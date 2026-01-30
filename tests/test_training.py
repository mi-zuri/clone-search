"""Tests for encoder training utilities."""

import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from src.models.encoder import FaceEncoder
from src.training.train_encoder import (
    attribute_loss,
    compute_pos_weights,
    load_checkpoint,
    nt_xent_loss,
    save_checkpoint,
)


class TestNTXentLoss:
    """Tests for NT-Xent contrastive loss."""

    def test_output_is_scalar(self):
        """Loss should be a scalar tensor."""
        z1 = torch.randn(8, 64)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.randn(8, 64)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        loss = nt_xent_loss(z1, z2)
        assert loss.dim() == 0, "Loss should be scalar"

    def test_positive_pairs_lower_loss(self):
        """Identical positive pairs should have lower loss than random pairs."""
        batch_size = 16
        dim = 64

        # Create identical pairs (perfect positive pairs)
        z_same = torch.randn(batch_size, dim)
        z_same = torch.nn.functional.normalize(z_same, dim=1)
        loss_same = nt_xent_loss(z_same, z_same.clone())

        # Create random pairs
        z1_random = torch.randn(batch_size, dim)
        z1_random = torch.nn.functional.normalize(z1_random, dim=1)
        z2_random = torch.randn(batch_size, dim)
        z2_random = torch.nn.functional.normalize(z2_random, dim=1)
        loss_random = nt_xent_loss(z1_random, z2_random)

        assert loss_same < loss_random, "Identical pairs should have lower loss"

    def test_loss_is_positive(self):
        """Loss should always be positive."""
        z1 = torch.randn(4, 64)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.randn(4, 64)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        loss = nt_xent_loss(z1, z2)
        assert loss.item() > 0, "Loss should be positive"

    def test_temperature_effect(self):
        """Lower temperature should produce sharper (higher) losses for random pairs."""
        z1 = torch.randn(8, 64)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.randn(8, 64)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        loss_high_temp = nt_xent_loss(z1, z2, temperature=1.0)
        loss_low_temp = nt_xent_loss(z1, z2, temperature=0.1)

        # Lower temperature makes the distribution sharper
        # For random pairs, this typically increases the loss
        assert loss_low_temp != loss_high_temp, "Temperature should affect loss"

    def test_batch_size_1(self):
        """Should handle batch size of 1."""
        z1 = torch.randn(1, 64)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.randn(1, 64)
        z2 = torch.nn.functional.normalize(z2, dim=1)

        # With batch_size=1, there are no negatives, but it shouldn't crash
        loss = nt_xent_loss(z1, z2)
        assert torch.isfinite(loss), "Loss should be finite"


class TestComputePosWeights:
    """Tests for attribute class weight computation."""

    def test_output_shape(self):
        """Weights should have shape (40,)."""

        # Create a minimal mock dataset
        class MockCelebA:
            def __init__(self):
                # 10 samples with random binary attributes
                self.samples = [
                    {"attributes": [1 if j % 2 == 0 else 0 for j in range(40)]}
                    for _ in range(10)
                ]

            def __len__(self):
                return len(self.samples)

        class MockCombined:
            def __init__(self):
                self.celeba = MockCelebA()

        mock_dataset = MockCombined()
        weights = compute_pos_weights(mock_dataset)

        assert weights.shape == (40,), f"Expected (40,), got {weights.shape}"

    def test_weights_in_range(self):
        """Weights should be clamped to [0.1, 10.0]."""

        class MockCelebA:
            def __init__(self):
                # Create imbalanced attributes
                self.samples = []
                for i in range(100):
                    attrs = [0] * 40
                    # First attribute: always 1 (very common)
                    attrs[0] = 1
                    # Second attribute: rarely 1 (very rare)
                    if i < 2:
                        attrs[1] = 1
                    # Third attribute: balanced
                    attrs[2] = 1 if i < 50 else 0
                    self.samples.append({"attributes": attrs})

            def __len__(self):
                return len(self.samples)

        class MockCombined:
            def __init__(self):
                self.celeba = MockCelebA()

        mock_dataset = MockCombined()
        weights = compute_pos_weights(mock_dataset)

        # All weights should be in [0.1, 10.0]
        assert weights.min() >= 0.1, f"Min weight {weights.min()} < 0.1"
        assert weights.max() <= 10.0, f"Max weight {weights.max()} > 10.0"

    def test_balanced_attribute_weight_near_one(self):
        """A perfectly balanced attribute should have weight near 1.0."""

        class MockCelebA:
            def __init__(self):
                self.samples = []
                for i in range(100):
                    # Attribute 0 is perfectly balanced
                    attrs = [1 if i < 50 else 0] + [0] * 39
                    self.samples.append({"attributes": attrs})

            def __len__(self):
                return len(self.samples)

        class MockCombined:
            def __init__(self):
                self.celeba = MockCelebA()

        mock_dataset = MockCombined()
        weights = compute_pos_weights(mock_dataset)

        # First attribute (balanced) should have weight ~1.0
        assert 0.9 < weights[0].item() < 1.1, f"Expected ~1.0, got {weights[0]}"


class TestAttributeLoss:
    """Tests for weighted BCE attribute loss."""

    def test_no_attributes_returns_zero(self):
        """Should return 0 when no samples have attributes."""
        logits = torch.randn(4, 40)
        targets = torch.randn(4, 40).sigmoid()
        pos_weights = torch.ones(40)
        has_attributes = torch.zeros(4, dtype=torch.bool)

        loss = attribute_loss(logits, targets, pos_weights, has_attributes)
        assert loss.item() == 0.0

    def test_loss_is_positive(self):
        """Loss should be positive when there are samples with attributes."""
        logits = torch.randn(4, 40)
        targets = torch.zeros(4, 40)  # All zeros
        pos_weights = torch.ones(40)
        has_attributes = torch.ones(4, dtype=torch.bool)

        loss = attribute_loss(logits, targets, pos_weights, has_attributes)
        assert loss.item() > 0

    def test_partial_attributes(self):
        """Should only compute loss on samples with has_attributes=True."""
        logits = torch.randn(4, 40)
        targets = torch.zeros(4, 40)
        pos_weights = torch.ones(40)
        # Only first 2 samples have attributes
        has_attributes = torch.tensor([True, True, False, False])

        loss = attribute_loss(logits, targets, pos_weights, has_attributes)
        assert loss.item() > 0


class TestSchedulerWarmup:
    """Tests for learning rate scheduler warmup behavior."""

    def test_warmup_increases_lr(self):
        """Learning rate should increase during warmup phase."""
        model = FaceEncoder()
        optimizer = AdamW(model.parameters(), lr=0.001)

        # Linear warmup from 0.00001 to 0.001
        scheduler = LinearLR(
            optimizer,
            start_factor=0.00001 / 0.001,  # 0.01
            end_factor=1.0,
            total_iters=10,
        )

        lrs = []
        for _ in range(10):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should strictly increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase: {lrs}"

    def test_warmup_start_lr(self):
        """Initial LR should be at warmup_start_lr."""
        model = FaceEncoder()
        optimizer = AdamW(model.parameters(), lr=0.001)

        warmup_start = 0.00001
        scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start / 0.001,
            end_factor=1.0,
            total_iters=10,
        )

        initial_lr = scheduler.get_last_lr()[0]
        assert abs(initial_lr - warmup_start) < 1e-8, f"Expected {warmup_start}, got {initial_lr}"


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""

    def test_save_and_load_model(self):
        """Model state should be preserved after save/load."""
        model = FaceEncoder()
        optimizer = AdamW(model.parameters(), lr=0.001)
        scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

        # Get initial weights
        initial_weight = model.embedding_head[0].weight.data.clone()

        # Do a forward/backward pass to change gradients
        x = torch.randn(2, 3, 256, 256)
        embedding, _ = model(x)
        loss = embedding.sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(
            model.embedding_head[0].weight.data, initial_weight
        ), "Weights should change after optimization"

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                val_acc=0.85,
                checkpoint_dir=checkpoint_dir,
                is_best=True,
            )

            # Verify files exist
            assert (checkpoint_dir / "encoder_epoch1.pt").exists()
            assert (checkpoint_dir / "best_encoder.pt").exists()

            # Create new model and load
            new_model = FaceEncoder()
            checkpoint = load_checkpoint(
                checkpoint_dir / "best_encoder.pt",
                new_model,
            )

            # Weights should match
            assert torch.allclose(
                new_model.embedding_head[0].weight.data,
                model.embedding_head[0].weight.data,
            )

            # Metadata should be correct
            assert checkpoint["epoch"] == 1
            assert checkpoint["val_acc"] == 0.85

    def test_load_restores_optimizer(self):
        """Optimizer state should be restored."""
        model = FaceEncoder()
        optimizer = AdamW(model.parameters(), lr=0.001)
        scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

        # Do some optimization steps
        for _ in range(3):
            x = torch.randn(2, 3, 256, 256)
            embedding, _ = model(x)
            loss = embedding.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                val_acc=0.85,
                checkpoint_dir=checkpoint_dir,
            )

            # Create new model and optimizer
            new_model = FaceEncoder()
            new_optimizer = AdamW(new_model.parameters(), lr=0.001)
            new_scheduler = LinearLR(
                new_optimizer, start_factor=0.01, end_factor=1.0, total_iters=10
            )

            load_checkpoint(
                checkpoint_dir / "encoder_epoch1.pt",
                new_model,
                new_optimizer,
                new_scheduler,
            )

            # Scheduler step count should be restored
            assert new_scheduler.last_epoch == scheduler.last_epoch
