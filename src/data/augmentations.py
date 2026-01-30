"""Augmentation pipelines for training."""

from typing import Callable

import torch
import torchvision.transforms.v2 as T


# ImageNet normalization (standard for pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_simclr_augmentations(image_size: int = 256) -> Callable:
    """Strong augmentations for SimCLR contrastive learning.

    Includes ColorJitter, GaussianBlur, and HorizontalFlip to create
    diverse positive pairs while preserving facial identity features.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToImage(),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_standard_augmentations(image_size: int = 256) -> Callable:
    """Lighter augmentations for UNet training.

    Preserves spatial structure needed for reconstruction while
    adding mild variations for robustness.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    )
                ],
                p=0.3,
            ),
            T.ToImage(),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_augmentations(image_size: int = 256) -> Callable:
    """Validation/inference augmentations - resize and normalize only."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToImage(),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
