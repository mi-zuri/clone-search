"""Utility to identify train/val/test splits based on config conventions.

Training used fixed index-based splits:
- CelebA: 0-14999 train, 15000-19999 val, 20000-29999 test
- FFHQ: first 5000 files (sorted) for train, rest for test
"""

import yaml

from src.data.dataset import CelebADataset, FFHQDataset
from src.data.augmentations import get_val_augmentations


def get_test_datasets(
    config_path: str = "configs/config.yaml",
) -> tuple[CelebADataset, FFHQDataset]:
    """Get test-only datasets (images not used in training or validation).

    Returns:
        Tuple of (celeba_test, ffhq_test) datasets
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    transform = get_val_augmentations(image_size)

    # CelebA test: indices 20000-29999
    train_end = data_cfg["celeba_train"]  # 15000
    val_end = train_end + data_cfg["celeba_val"]  # 20000
    total_celeba = 30000
    test_indices = list(range(val_end, total_celeba))

    celeba_test = CelebADataset(
        root=data_cfg["celeba_path"],
        attr_path=data_cfg["celeba_attr_path"],
        transform=transform,
        indices=test_indices,
        image_size=image_size,
    )

    # FFHQ test: skip first 5000 files (used in training)
    ffhq_train_count = data_cfg["ffhq_train_subset"]  # 5000
    ffhq_total = data_cfg["ffhq_total"]  # 50000

    # Create full FFHQ to get all file paths, then select test portion
    ffhq_full = FFHQDataset(
        root=data_cfg["ffhq_path"],
        transform=transform,
        image_size=image_size,
        total_images=ffhq_total,
    )

    # Select only test indices (5000+)
    ffhq_test_indices = list(range(ffhq_train_count, len(ffhq_full)))
    ffhq_test = FFHQDataset(
        root=data_cfg["ffhq_path"],
        transform=transform,
        indices=ffhq_test_indices,
        image_size=image_size,
        total_images=ffhq_total,
    )

    return celeba_test, ffhq_test


def get_split_info(config_path: str = "configs/config.yaml") -> dict:
    """Get summary of data splits.

    Returns:
        Dict with counts for each split
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_end = data_cfg["celeba_train"]
    val_end = train_end + data_cfg["celeba_val"]

    return {
        "celeba_train": {"start": 0, "end": train_end, "count": train_end},
        "celeba_val": {"start": train_end, "end": val_end, "count": val_end - train_end},
        "celeba_test": {"start": val_end, "end": 30000, "count": 30000 - val_end},
        "ffhq_train": {"start": 0, "end": data_cfg["ffhq_train_subset"], "count": data_cfg["ffhq_train_subset"]},
        "ffhq_test": {"start": data_cfg["ffhq_train_subset"], "end": data_cfg["ffhq_total"], "count": data_cfg["ffhq_total"] - data_cfg["ffhq_train_subset"]},
    }


if __name__ == "__main__":
    info = get_split_info()
    print("Data splits:")
    for name, split in info.items():
        print(f"  {name}: {split['count']} images (indices {split['start']}-{split['end']-1})")

    print("\nLoading test datasets...")
    celeba_test, ffhq_test = get_test_datasets()
    print(f"  CelebA test: {len(celeba_test)} images")
    print(f"  FFHQ test: {len(ffhq_test)} images")
