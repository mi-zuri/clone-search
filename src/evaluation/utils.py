"""Shared utilities for evaluation scripts."""

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.encoder import FaceEncoder


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(
    checkpoint_path: str,
    config_path: str = "configs/config.yaml",
    device: torch.device | None = None,
) -> FaceEncoder:
    """Load FaceEncoder from checkpoint in eval mode.

    Args:
        checkpoint_path: Path to encoder checkpoint (best_encoder.pt)
        config_path: Path to config.yaml for model hyperparameters
        device: Device to load model on (auto-detected if None)

    Returns:
        FaceEncoder model in eval mode
    """
    if device is None:
        device = get_device()

    # Load config for model architecture
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model with correct architecture
    model = FaceEncoder(
        embedding_dim=config["encoder"]["embedding_dim"],
        projection_dim=config["encoder"]["projection_dim"],
        num_attributes=config["encoder"]["num_attributes"],
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model


def load_gallery(gallery_path: str):
    """Load FaceSearchEngine from gallery NPZ file.

    Args:
        gallery_path: Path to gallery_index.npz

    Returns:
        FaceSearchEngine instance
    """
    from src.search.engine import FaceSearchEngine

    return FaceSearchEngine(gallery_path)


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Save data as pretty-printed JSON.

    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation (default: 2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

    print(f"Saved results to {path}")
