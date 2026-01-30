"""Tests for data loading and augmentation pipelines."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentations import (
    get_simclr_augmentations,
    get_standard_augmentations,
    get_val_augmentations,
)
from src.data.dataset import (
    CelebADataset,
    CombinedDataset,
    FFHQDataset,
    SimCLRWrapper,
)


# Test configuration
DATA_ROOT = Path(__file__).parent.parent / "data"
CELEBA_ROOT = DATA_ROOT / "celeba"
CELEBA_ATTR = DATA_ROOT / "CelebAMask-HQ-attribute-anno.txt"
CELEBA_MASKS = DATA_ROOT / "celeba_masks"
FFHQ_ROOT = DATA_ROOT / "ffhq"
IMAGE_SIZE = 256


@pytest.fixture
def celeba_dataset():
    """Create CelebA dataset with first 10 samples."""
    return CelebADataset(
        root=CELEBA_ROOT,
        attr_path=CELEBA_ATTR,
        indices=list(range(10)),
        image_size=IMAGE_SIZE,
    )


@pytest.fixture
def celeba_dataset_with_masks():
    """Create CelebA dataset with masks."""
    return CelebADataset(
        root=CELEBA_ROOT,
        attr_path=CELEBA_ATTR,
        masks_root=CELEBA_MASKS,
        indices=list(range(10)),
        image_size=IMAGE_SIZE,
    )


@pytest.fixture
def ffhq_dataset():
    """Create FFHQ dataset with first 10 samples."""
    return FFHQDataset(
        root=FFHQ_ROOT,
        indices=list(range(10)),
        image_size=IMAGE_SIZE,
    )


class TestCelebADataset:
    """Tests for CelebADataset."""

    def test_loads_images(self, celeba_dataset):
        """Test that images load correctly."""
        sample = celeba_dataset[0]
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_loads_attributes(self, celeba_dataset):
        """Test that attributes load and convert correctly."""
        sample = celeba_dataset[0]
        assert "attributes" in sample
        assert isinstance(sample["attributes"], torch.Tensor)
        assert sample["attributes"].shape == (40,)
        # Check values are 0 or 1 (converted from -1/1)
        assert torch.all((sample["attributes"] == 0) | (sample["attributes"] == 1))

    def test_returns_path(self, celeba_dataset):
        """Test that image path is returned."""
        sample = celeba_dataset[0]
        assert "path" in sample
        assert isinstance(sample["path"], str)
        assert sample["path"].endswith(".jpg")

    def test_returns_image_id(self, celeba_dataset):
        """Test that image ID is returned."""
        sample = celeba_dataset[0]
        assert "image_id" in sample
        assert isinstance(sample["image_id"], int)

    def test_loads_masks(self, celeba_dataset_with_masks):
        """Test that semantic masks load when provided."""
        sample = celeba_dataset_with_masks[0]
        assert "mask" in sample
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)
        # Mask should be binary (0 or 1)
        assert torch.all((sample["mask"] == 0) | (sample["mask"] == 1))

    def test_length(self, celeba_dataset):
        """Test dataset length."""
        assert len(celeba_dataset) == 10

    def test_attribute_names(self):
        """Test that attribute names are correctly defined."""
        assert len(CelebADataset.ATTRIBUTE_NAMES) == 40
        assert "Male" in CelebADataset.ATTRIBUTE_NAMES
        assert "Smiling" in CelebADataset.ATTRIBUTE_NAMES


class TestFFHQDataset:
    """Tests for FFHQDataset."""

    def test_loads_images(self, ffhq_dataset):
        """Test that images load correctly."""
        sample = ffhq_dataset[0]
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_returns_path(self, ffhq_dataset):
        """Test that image path is returned."""
        sample = ffhq_dataset[0]
        assert "path" in sample
        assert isinstance(sample["path"], str)
        assert sample["path"].endswith(".png")

    def test_returns_image_id(self, ffhq_dataset):
        """Test that image ID is returned."""
        sample = ffhq_dataset[0]
        assert "image_id" in sample
        assert isinstance(sample["image_id"], int)

    def test_no_attributes(self, ffhq_dataset):
        """Test that FFHQ samples don't have attributes."""
        sample = ffhq_dataset[0]
        assert "attributes" not in sample

    def test_length(self, ffhq_dataset):
        """Test dataset length."""
        assert len(ffhq_dataset) == 10


class TestCombinedDataset:
    """Tests for CombinedDataset."""

    def test_merges_datasets(self, celeba_dataset, ffhq_dataset):
        """Test that datasets are merged correctly."""
        combined = CombinedDataset(celeba_dataset, ffhq_dataset)
        assert len(combined) == len(celeba_dataset) + len(ffhq_dataset)

    def test_celeba_has_attributes_flag(self, celeba_dataset, ffhq_dataset):
        """Test that CelebA samples have has_attributes=True."""
        combined = CombinedDataset(celeba_dataset, ffhq_dataset)
        sample = combined[0]  # First sample is from CelebA
        assert sample["has_attributes"] is True
        assert sample["source"] == "celeba"

    def test_ffhq_has_attributes_flag(self, celeba_dataset, ffhq_dataset):
        """Test that FFHQ samples have has_attributes=False."""
        combined = CombinedDataset(celeba_dataset, ffhq_dataset)
        # FFHQ samples start after CelebA
        sample = combined[len(celeba_dataset)]
        assert sample["has_attributes"] is False
        assert sample["source"] == "ffhq"
        # FFHQ samples should have dummy attributes for collation
        assert "attributes" in sample
        assert sample["attributes"].shape == (40,)

    def test_sampler_weights(self, celeba_dataset, ffhq_dataset):
        """Test weighted sampler generation."""
        combined = CombinedDataset(celeba_dataset, ffhq_dataset)
        weights = combined.get_sampler_weights(celeba_ratio=0.75)
        assert weights.shape == (len(combined),)
        # All weights should be positive
        assert torch.all(weights > 0)


class TestSimCLRWrapper:
    """Tests for SimCLRWrapper."""

    def test_returns_two_views(self, celeba_dataset):
        """Test that wrapper returns two augmented views."""
        simclr_transform = get_simclr_augmentations(IMAGE_SIZE)
        wrapped = SimCLRWrapper(celeba_dataset, simclr_transform)
        sample = wrapped[0]

        assert "view1" in sample
        assert "view2" in sample
        assert isinstance(sample["view1"], torch.Tensor)
        assert isinstance(sample["view2"], torch.Tensor)
        assert sample["view1"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert sample["view2"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_views_are_different(self, celeba_dataset):
        """Test that the two views are different (due to random augmentation)."""
        simclr_transform = get_simclr_augmentations(IMAGE_SIZE)
        wrapped = SimCLRWrapper(celeba_dataset, simclr_transform)
        sample = wrapped[0]

        # Views should be different due to random augmentation
        # (very unlikely to be identical)
        assert not torch.allclose(sample["view1"], sample["view2"])

    def test_preserves_attributes(self, celeba_dataset):
        """Test that attributes are preserved through wrapper."""
        simclr_transform = get_simclr_augmentations(IMAGE_SIZE)
        wrapped = SimCLRWrapper(celeba_dataset, simclr_transform)
        sample = wrapped[0]

        assert "attributes" in sample
        assert sample["attributes"].shape == (40,)

    def test_preserves_path(self, celeba_dataset):
        """Test that path is preserved through wrapper."""
        simclr_transform = get_simclr_augmentations(IMAGE_SIZE)
        wrapped = SimCLRWrapper(celeba_dataset, simclr_transform)
        sample = wrapped[0]

        assert "path" in sample
        assert isinstance(sample["path"], str)

    def test_length(self, celeba_dataset):
        """Test wrapper length matches underlying dataset."""
        simclr_transform = get_simclr_augmentations(IMAGE_SIZE)
        wrapped = SimCLRWrapper(celeba_dataset, simclr_transform)
        assert len(wrapped) == len(celeba_dataset)


class TestAugmentations:
    """Tests for augmentation pipelines."""

    def test_simclr_augmentations_output_shape(self, celeba_dataset):
        """Test SimCLR augmentations output correct shape."""
        # Get raw PIL image
        celeba_dataset.transform = None
        sample = celeba_dataset[0]
        image = sample["image"]

        transform = get_simclr_augmentations(IMAGE_SIZE)
        output = transform(image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_standard_augmentations_output_shape(self, celeba_dataset):
        """Test standard augmentations output correct shape."""
        celeba_dataset.transform = None
        sample = celeba_dataset[0]
        image = sample["image"]

        transform = get_standard_augmentations(IMAGE_SIZE)
        output = transform(image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_augmentations_output_shape(self, celeba_dataset):
        """Test validation augmentations output correct shape."""
        celeba_dataset.transform = None
        sample = celeba_dataset[0]
        image = sample["image"]

        transform = get_val_augmentations(IMAGE_SIZE)
        output = transform(image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_val_augmentations_deterministic(self, celeba_dataset):
        """Test validation augmentations are deterministic."""
        celeba_dataset.transform = None
        sample = celeba_dataset[0]
        image = sample["image"]

        transform = get_val_augmentations(IMAGE_SIZE)
        output1 = transform(image)
        output2 = transform(image)

        # Validation augmentations should be deterministic
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
