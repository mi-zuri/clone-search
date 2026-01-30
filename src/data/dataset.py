"""Dataset classes for CelebA-HQ, FFHQ, and combined training."""

from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset


class TransformableDataset(Protocol):
    """Protocol for datasets with a transform attribute."""

    transform: Optional[Callable]

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

from .augmentations import get_val_augmentations


class CelebADataset(Dataset):
    """CelebA-HQ dataset with attributes and optional masks.

    Attributes:
        root: Path to celeba images (e.g., data/celeba)
        attr_path: Path to CelebAMask-HQ-attribute-anno.txt
        masks_root: Optional path to semantic masks (e.g., data/celeba_masks)
        transform: Image transformation pipeline
        indices: Optional list of indices to use (for train/val/test splits)
    """

    # All 40 CelebA-HQ attributes
    ATTRIBUTE_NAMES = [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]

    # Semantic mask regions available in CelebA-HQ
    MASK_REGIONS = [
        "skin",
        "nose",
        "l_eye",
        "r_eye",
        "l_brow",
        "r_brow",
        "l_ear",
        "r_ear",
        "ear_r",  # alternative naming
        "mouth",
        "u_lip",
        "l_lip",
        "hair",
        "hat",
        "neck",
        "cloth",
    ]

    def __init__(
        self,
        root: str | Path,
        attr_path: str | Path,
        masks_root: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        indices: Optional[list[int]] = None,
        image_size: int = 256,
    ):
        self.root = Path(root)
        self.masks_root = Path(masks_root) if masks_root else None
        self.transform = transform or get_val_augmentations(image_size)
        self.image_size = image_size

        # Parse attribute file
        self.samples = []
        self._parse_attributes(attr_path)

        # Filter by indices if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def _parse_attributes(self, attr_path: str | Path) -> None:
        """Parse CelebAMask-HQ-attribute-anno.txt.

        Format:
        - Line 1: total count
        - Line 2: attribute names (space-separated)
        - Lines 3+: filename  attr1 attr2 ... (double-space between name and attrs)
        """
        with open(attr_path, "r") as f:
            lines = f.readlines()

        # Skip header lines (count and attribute names)
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue

            # Double-space separates filename from attributes
            parts = line.split("  ")
            filename = parts[0].strip()
            attr_values = list(map(int, parts[1].split()))

            # Convert -1/1 to 0/1 for BCE loss
            attr_values = [(v + 1) // 2 for v in attr_values]

            # Get image ID from filename (e.g., "123.jpg" -> 123)
            image_id = int(filename.split(".")[0])

            self.samples.append(
                {
                    "filename": filename,
                    "image_id": image_id,
                    "attributes": attr_values,
                }
            )

    def _get_subfolder(self, image_id: int) -> str:
        """Get subfolder for image ID (images are grouped in folders of 1000)."""
        return str(image_id // 1000)

    def _load_masks(self, image_id: int, subfolder: str) -> Optional[torch.Tensor]:
        """Load and combine semantic masks for the given image.

        Returns a single binary mask where 1 indicates any semantic region.
        """
        if self.masks_root is None:
            return None

        combined_mask = None
        mask_folder = self.masks_root / subfolder

        for region in self.MASK_REGIONS:
            mask_path = mask_folder / f"{image_id:05d}_{region}.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((self.image_size, self.image_size), Resampling.NEAREST)
                mask_array = np.array(mask, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array)
                mask_tensor = (mask_tensor > 0).float()

                if combined_mask is None:
                    combined_mask = mask_tensor
                else:
                    combined_mask = torch.maximum(combined_mask, mask_tensor)

        return combined_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_id = sample["image_id"]
        subfolder = self._get_subfolder(image_id)

        # Load image
        image_path = self.root / subfolder / sample["filename"]
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        result = {
            "image": image,
            "attributes": torch.tensor(sample["attributes"], dtype=torch.float32),
            "path": str(image_path),
            "image_id": image_id,
        }

        # Optionally load masks
        mask = self._load_masks(image_id, subfolder)
        if mask is not None:
            result["mask"] = mask

        return result


class FFHQDataset(Dataset):
    """FFHQ dataset for unlabeled face images.

    Scans directory for actual .png files to handle non-sequential numbering.

    Attributes:
        root: Path to FFHQ images (e.g., data/ffhq)
        transform: Image transformation pipeline
        indices: Optional list of indices to use (for subset selection)
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        indices: Optional[list[int]] = None,
        image_size: int = 256,
        total_images: int = 50000,
    ):
        self.root = Path(root)
        self.transform = transform or get_val_augmentations(image_size)

        # Scan directory for actual files instead of assuming sequential numbering
        self.image_paths = self._scan_files(total_images)

        # Apply subset selection if indices provided
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]

    def _scan_files(self, max_files: int) -> list[Path]:
        """Scan root directory for .png files, sorted by name.

        Args:
            max_files: Maximum number of files to include

        Returns:
            Sorted list of image paths
        """
        image_paths = []

        # Scan all subdirectories
        for subfolder in sorted(self.root.iterdir()):
            if not subfolder.is_dir():
                continue

            # Find all .png files in subfolder
            for img_path in sorted(subfolder.glob("*.png")):
                image_paths.append(img_path)
                if len(image_paths) >= max_files:
                    return image_paths

        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]

        # Extract image_id from filename (e.g., "00123.png" -> 123)
        image_id = int(image_path.stem)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "path": str(image_path),
            "image_id": image_id,
        }


class CombinedDataset(Dataset):
    """Combined CelebA + FFHQ dataset with has_attributes flag.

    Used for training with mixed batches where CelebA provides
    attribute labels and FFHQ provides additional face diversity.
    """

    def __init__(
        self,
        celeba_dataset: CelebADataset,
        ffhq_dataset: FFHQDataset,
    ):
        self.celeba = celeba_dataset
        self.ffhq = ffhq_dataset
        self.celeba_len = len(celeba_dataset)
        self.ffhq_len = len(ffhq_dataset)
        self._transform: Optional[Callable] = None

    @property
    def transform(self) -> Optional[Callable]:
        """Get current transform (from underlying datasets)."""
        return self._transform

    @transform.setter
    def transform(self, value: Optional[Callable]) -> None:
        """Set transform on both underlying datasets."""
        self._transform = value
        self.celeba.transform = value
        self.ffhq.transform = value

    def __len__(self) -> int:
        return self.celeba_len + self.ffhq_len

    def __getitem__(self, idx: int) -> dict:
        if idx < self.celeba_len:
            # CelebA sample - has attributes
            sample = self.celeba[idx]
            sample["has_attributes"] = True
            sample["source"] = "celeba"
        else:
            # FFHQ sample - no attributes
            ffhq_idx = idx - self.celeba_len
            sample = self.ffhq[ffhq_idx]
            sample["has_attributes"] = False
            sample["source"] = "ffhq"
            # Add dummy attributes for batch collation
            sample["attributes"] = torch.zeros(40, dtype=torch.float32)

        return sample

    def get_sampler_weights(self, celeba_ratio: float = 0.75) -> torch.Tensor:
        """Get sampling weights for WeightedRandomSampler.

        Args:
            celeba_ratio: Target ratio of CelebA samples per batch (default 0.75)

        Returns:
            Tensor of weights for each sample
        """
        # Calculate weights to achieve desired ratio
        # If we want 75% CelebA and have N_c CelebA and N_f FFHQ samples:
        # w_c * N_c / (w_c * N_c + w_f * N_f) = 0.75
        # Setting w_f = 1: w_c * N_c = 3 * N_f, so w_c = 3 * N_f / N_c
        ffhq_weight = 1.0
        celeba_weight = (celeba_ratio / (1 - celeba_ratio)) * (
            self.ffhq_len / self.celeba_len
        )

        weights = torch.zeros(len(self))
        weights[: self.celeba_len] = celeba_weight
        weights[self.celeba_len :] = ffhq_weight

        return weights


class SimCLRWrapper(Dataset):
    """Wrapper that returns two augmented views for SimCLR training.

    Wraps any dataset and applies the SimCLR augmentation pipeline twice
    to generate positive pairs for contrastive learning.
    """

    def __init__(
        self,
        dataset: Union[CelebADataset, FFHQDataset, CombinedDataset],
        simclr_transform: Callable,
    ):
        self.dataset = dataset
        self.simclr_transform = simclr_transform
        # Store original transform for reference
        self._original_transform = dataset.transform
        # Disable transform to get raw PIL images
        dataset.transform = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        # Get raw sample without transforms
        sample = self.dataset[idx]
        image = sample["image"]

        # If image is already a tensor (from previous transform), convert back
        if isinstance(image, torch.Tensor):
            # This shouldn't happen with our setup, but handle it
            raise ValueError("SimCLRWrapper expects raw PIL images from dataset")

        # Apply SimCLR augmentation twice for two views
        view1 = self.simclr_transform(image)
        view2 = self.simclr_transform(image)

        # Build result with both views
        result = {
            "view1": view1,
            "view2": view2,
            "path": sample.get("path", ""),
            "image_id": sample.get("image_id", idx),
        }

        # Preserve other fields
        if "attributes" in sample:
            result["attributes"] = sample["attributes"]
        if "has_attributes" in sample:
            result["has_attributes"] = sample["has_attributes"]
        if "source" in sample:
            result["source"] = sample["source"]
        if "mask" in sample:
            result["mask"] = sample["mask"]

        return result
