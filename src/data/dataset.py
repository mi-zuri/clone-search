"""Dataset classes for CelebAMask-HQ and FFHQ."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from .augmentations import get_train_transforms, get_val_transforms, get_inpaint_transforms


# CelebA attribute names in order
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]

# Semantic mask parts for inpainting
MASK_PARTS = {
    'skin': 1,
    'l_brow': 2,
    'r_brow': 3,
    'l_eye': 4,
    'r_eye': 5,
    'eye_g': 6,  # eyeglasses
    'l_ear': 7,
    'r_ear': 8,
    'ear_r': 9,  # earring
    'nose': 10,
    'mouth': 11,
    'u_lip': 12,
    'l_lip': 13,
    'neck': 14,
    'neck_l': 15,  # necklace
    'cloth': 16,
    'hair': 17,
    'hat': 18
}

# Parts to use for inpainting masks
INPAINT_PARTS = ['l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'mouth', 'u_lip', 'l_lip']


class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ dataset for face attribute classification and inpainting."""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 256,
        for_inpainting: bool = False,
        transform=None
    ):
        """Initialize CelebAMask-HQ dataset.

        Args:
            root: Path to CelebAMask-HQ directory
            split: One of 'train', 'val', 'test'
            image_size: Target image size
            for_inpainting: If True, returns mask for inpainting
            transform: Optional custom transform
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.for_inpainting = for_inpainting

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif for_inpainting:
            self.transform = get_inpaint_transforms(image_size, is_train=(split == 'train'))
        elif split == 'train':
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Find image directory
        self.img_dir = self._find_img_dir()
        self.mask_dir = self._find_mask_dir()

        # Load attributes
        self.attributes = self._load_attributes()

        # Get image list and split
        self.image_ids = self._get_split_ids()

    def _find_img_dir(self) -> Path:
        """Find the image directory in the dataset."""
        # Common locations
        candidates = [
            self.root / 'CelebA-HQ-img',
            self.root / 'CelebAMask-HQ' / 'CelebA-HQ-img',
            self.root / 'img',
            self.root / 'images',
        ]
        for path in candidates:
            if path.exists():
                return path
        # Fallback: search for jpg files
        for path in self.root.rglob('*.jpg'):
            return path.parent
        raise FileNotFoundError(f"Could not find image directory in {self.root}")

    def _find_mask_dir(self) -> Optional[Path]:
        """Find the mask directory in the dataset."""
        candidates = [
            self.root / 'CelebAMask-HQ-mask-anno',
            self.root / 'CelebAMask-HQ' / 'CelebAMask-HQ-mask-anno',
            self.root / 'mask',
            self.root / 'masks',
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_attributes(self) -> Optional[pd.DataFrame]:
        """Load attribute annotations."""
        candidates = [
            self.root / 'CelebA-HQ-attribute-anno.txt',
            self.root / 'CelebAMask-HQ' / 'CelebA-HQ-attribute-anno.txt',
            self.root / 'list_attr_celeba.txt',
        ]

        for attr_file in candidates:
            if attr_file.exists():
                # Check if first line is count or header
                with open(attr_file, 'r') as f:
                    first_line = f.readline().strip()

                try:
                    int(first_line)
                    # Standard format: count on line 1, header on line 2
                    skiprows = 2
                except ValueError:
                    # No count line, header is on line 1
                    skiprows = 1

                # Read with explicit column names (header has 40 attrs, data has 41 cols)
                df = pd.read_csv(
                    attr_file,
                    sep=r'\s+',
                    skiprows=skiprows,
                    header=None,
                    names=['filename'] + CELEBA_ATTRIBUTES,
                    index_col='filename'
                )

                # Convert -1/1 to 0/1
                df = (df + 1) // 2

                return df

        return None

    def _get_split_ids(self) -> List[int]:
        """Get image IDs for the current split."""
        # Get all image files
        all_images = sorted(self.img_dir.glob('*.jpg')) + sorted(self.img_dir.glob('*.png'))
        total = len(all_images)

        # Extract IDs (assuming filenames like '00000.jpg' or similar)
        ids = []
        for img_path in all_images:
            try:
                img_id = int(img_path.stem)
                ids.append(img_id)
            except ValueError:
                ids.append(img_path.stem)

        ids = sorted(ids) if all(isinstance(i, int) for i in ids) else ids

        # Split: 80% train, 10% val, 10% test
        n_train = int(total * 0.8)
        n_val = int(total * 0.1)

        if self.split == 'train':
            return ids[:n_train]
        elif self.split == 'val':
            return ids[n_train:n_train + n_val]
        else:  # test
            return ids[n_train + n_val:]

    def _get_semantic_mask(self, idx: int) -> Optional[np.ndarray]:
        """Load and combine semantic masks for inpainting."""
        if self.mask_dir is None:
            return None

        # Masks are organized in folders 0, 1, 2, ... (500 images each)
        folder_idx = idx // 2000
        mask_folder = self.mask_dir / str(folder_idx)

        if not mask_folder.exists():
            # Try flat structure
            mask_folder = self.mask_dir

        # Combine masks for inpainting parts
        combined_mask = np.zeros((512, 512), dtype=np.uint8)

        # Randomly select 1-3 parts to mask
        num_parts = np.random.randint(1, 4)
        parts_to_mask = np.random.choice(INPAINT_PARTS, size=min(num_parts, len(INPAINT_PARTS)), replace=False)

        for part in parts_to_mask:
            # Mask filename format: {idx:05d}_{part}.png
            mask_file = mask_folder / f"{idx:05d}_{part}.png"
            if mask_file.exists():
                part_mask = np.array(Image.open(mask_file).convert('L'))
                combined_mask = np.maximum(combined_mask, part_mask)

        return combined_mask

    def _generate_random_mask(self, height: int, width: int) -> np.ndarray:
        """Generate random rectangle mask for inpainting."""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Random rectangle size
        h = np.random.randint(40, min(100, height // 2))
        w = np.random.randint(40, min(100, width // 2))

        # Random position (centered in face region)
        center_y, center_x = height // 2, width // 2
        y = np.random.randint(max(0, center_y - h), min(height - h, center_y + h // 2))
        x = np.random.randint(max(0, center_x - w), min(width - w, center_x + w // 2))

        mask[y:y+h, x:x+w] = 255

        return mask

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.image_ids[idx]

        # Load image
        if isinstance(img_id, int):
            img_path = self.img_dir / f"{img_id}.jpg"
            if not img_path.exists():
                img_path = self.img_dir / f"{img_id:05d}.jpg"
            if not img_path.exists():
                img_path = self.img_dir / f"{img_id}.png"
        else:
            img_path = self.img_dir / f"{img_id}.jpg"

        image = np.array(Image.open(img_path).convert('RGB'))

        result = {}

        if self.for_inpainting:
            # Get mask for inpainting
            mask = self._get_semantic_mask(img_id if isinstance(img_id, int) else idx)
            if mask is None or mask.max() == 0:
                mask = self._generate_random_mask(image.shape[0], image.shape[1])

            # Resize mask to match image
            mask_pil = Image.fromarray(mask).resize((image.shape[1], image.shape[0]), Image.NEAREST)
            mask = np.array(mask_pil)

            # Apply transforms
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed['image'].float() / 255.0
            mask_tensor = transformed['mask'].float().unsqueeze(0) / 255.0

            # Create masked image
            masked_image = image_tensor * (1 - mask_tensor)

            result['image'] = image_tensor
            result['mask'] = mask_tensor
            result['masked_image'] = masked_image
        else:
            # Apply transforms for classification
            transformed = self.transform(image=image)
            result['image'] = transformed['image']

            # Get attributes
            if self.attributes is not None and isinstance(img_id, int):
                try:
                    # Try multiple filename formats
                    candidates = [
                        f"{img_id}.jpg",      # e.g., "1.jpg"
                        f"{img_id - 1}.jpg",  # 0-indexed: image 00001.jpg -> attr 0.jpg
                    ]
                    attrs = None
                    for filename in candidates:
                        if filename in self.attributes.index:
                            attrs = self.attributes.loc[filename].values
                            break
                    if attrs is None:
                        # Fallback to row index
                        attrs = self.attributes.iloc[img_id].values
                    result['attributes'] = torch.tensor(attrs, dtype=torch.float32)
                except (KeyError, IndexError):
                    result['attributes'] = torch.zeros(40, dtype=torch.float32)
            else:
                result['attributes'] = torch.zeros(40, dtype=torch.float32)

        result['idx'] = img_id if isinstance(img_id, int) else idx

        return result


class FFHQDataset(Dataset):
    """FFHQ dataset for search gallery."""

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        transform=None,
        max_images: Optional[int] = None
    ):
        """Initialize FFHQ dataset.

        Args:
            root: Path to FFHQ directory
            image_size: Target image size
            transform: Optional custom transform
            max_images: Maximum number of images to use
        """
        self.root = Path(root)
        self.image_size = image_size

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_val_transforms(image_size)

        # Find all images
        self.image_paths = self._find_images(max_images)

    def _find_images(self, max_images: Optional[int] = None) -> List[Path]:
        """Find all images in the dataset."""
        extensions = ['*.jpg', '*.png', '*.jpeg']
        paths = []

        for ext in extensions:
            paths.extend(self.root.rglob(ext))

        paths = sorted(paths)

        if max_images is not None:
            paths = paths[:max_images]

        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        transformed = self.transform(image=image)

        return {
            'image': transformed['image'],
            'idx': idx,
            'path': str(img_path)
        }


def get_dataloaders(
    celeba_root: str,
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 4,
    for_inpainting: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get train, val, test dataloaders for CelebAMask-HQ.

    Args:
        celeba_root: Path to CelebAMask-HQ
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        for_inpainting: If True, load masks for inpainting

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import platform
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"Windows detected: setting num_workers=0 (was {num_workers})")
        num_workers = 0

    train_dataset = CelebAMaskHQDataset(
        celeba_root, split='train', image_size=image_size, for_inpainting=for_inpainting
    )
    val_dataset = CelebAMaskHQDataset(
        celeba_root, split='val', image_size=image_size, for_inpainting=for_inpainting
    )
    test_dataset = CelebAMaskHQDataset(
        celeba_root, split='test', image_size=image_size, for_inpainting=for_inpainting
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
