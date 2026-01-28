"""Data augmentation pipelines using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 256):
    """Get training augmentation pipeline.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 256):
    """Get validation/test augmentation pipeline.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_inpaint_transforms(image_size: int = 256, is_train: bool = True):
    """Get transforms for inpainting task (without normalization for mask).

    Args:
        image_size: Target image size
        is_train: Whether this is for training

    Returns:
        Albumentations Compose object
    """
    transforms_list = [A.Resize(image_size, image_size)]

    if is_train:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.3
            ),
        ])

    transforms_list.append(ToTensorV2())

    return A.Compose(
        transforms_list,
        additional_targets={'mask': 'mask'}
    )
