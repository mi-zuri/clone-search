"""Visualize embedding space with t-SNE colored by attributes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import CelebADataset
from src.evaluation.utils import get_device, load_encoder
from src.search.splits import get_test_datasets


# Attributes to visualize (semantically interesting, well-balanced)
VISUALIZATION_ATTRIBUTES = ["Male", "Smiling", "Young", "Eyeglasses", "Attractive"]


def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and attributes from dataset.

    Args:
        model: FaceEncoder in eval mode
        dataset: CelebA dataset with attributes
        n_samples: Number of samples to extract
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        Tuple of (embeddings, attributes) arrays
    """
    # Use subset if needed
    if n_samples < len(dataset):
        indices = list(range(n_samples))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_embeddings = []
    all_attributes = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            images = batch["image"].to(device)
            attributes = batch["attributes"]

            embedding, _ = model(images)

            all_embeddings.append(embedding.cpu().numpy())
            all_attributes.append(attributes.numpy())

    embeddings = np.vstack(all_embeddings)
    attributes = np.vstack(all_attributes)

    return embeddings, attributes


def run_tsne(embeddings: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    """Run t-SNE dimensionality reduction.

    Args:
        embeddings: Input embeddings of shape (N, D)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations

    Returns:
        2D embeddings of shape (N, 2)
    """
    print(f"Running t-SNE on {len(embeddings)} samples (64D -> 2D)...")
    print(f"  perplexity={perplexity}, n_iter={n_iter}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )

    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"  t-SNE complete. Final KL divergence: {tsne.kl_divergence_:.4f}")

    return embeddings_2d


def create_scatter_plot(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    attr_name: str,
    output_path: Path,
) -> None:
    """Create a scatter plot colored by binary attribute.

    Args:
        embeddings_2d: 2D embeddings from t-SNE (N, 2)
        labels: Binary labels for the attribute (N,)
        attr_name: Attribute name for title/legend
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot negative class first (background)
    neg_mask = labels == 0
    ax.scatter(
        embeddings_2d[neg_mask, 0],
        embeddings_2d[neg_mask, 1],
        c="#4A90A4",  # Muted blue
        alpha=0.5,
        s=15,
        label=f"Not {attr_name}",
        edgecolors="none",
    )

    # Plot positive class on top
    pos_mask = labels == 1
    ax.scatter(
        embeddings_2d[pos_mask, 0],
        embeddings_2d[pos_mask, 1],
        c="#E07A5F",  # Muted coral
        alpha=0.5,
        s=15,
        label=attr_name,
        edgecolors="none",
    )

    ax.set_title(f"t-SNE Embedding Space: {attr_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_aspect("equal")

    # Remove axis ticks (t-SNE coordinates are not meaningful)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add sample counts
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    ax.text(
        0.02, 0.98,
        f"N={len(labels)} ({n_pos} pos, {n_neg} neg)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def create_combined_plot(
    embeddings_2d: np.ndarray,
    attributes: np.ndarray,
    attr_names: list[str],
    output_path: Path,
) -> None:
    """Create a combined subplot figure with all attributes.

    Args:
        embeddings_2d: 2D embeddings from t-SNE (N, 2)
        attributes: Full attribute matrix (N, 40)
        attr_names: List of attribute names to visualize
        output_path: Path to save the combined plot
    """
    n_attrs = len(attr_names)
    n_cols = 3
    n_rows = (n_attrs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    attr_indices = {name: i for i, name in enumerate(CelebADataset.ATTRIBUTE_NAMES)}

    for i, attr_name in enumerate(attr_names):
        ax = axes[i]
        attr_idx = attr_indices[attr_name]
        labels = attributes[:, attr_idx]

        neg_mask = labels == 0
        pos_mask = labels == 1

        ax.scatter(
            embeddings_2d[neg_mask, 0],
            embeddings_2d[neg_mask, 1],
            c="#4A90A4",
            alpha=0.4,
            s=10,
            label=f"Not {attr_name}",
            edgecolors="none",
        )
        ax.scatter(
            embeddings_2d[pos_mask, 0],
            embeddings_2d[pos_mask, 1],
            c="#E07A5F",
            alpha=0.4,
            s=10,
            label=attr_name,
            edgecolors="none",
        )

        ax.set_title(attr_name, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal")

    # Hide unused subplots
    for i in range(n_attrs, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("t-SNE Visualization by Attributes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved combined plot: {output_path}")


def visualize_embeddings(
    encoder_path: str,
    output_dir: str,
    n_samples: int = 2000,
    config_path: str = "configs/config.yaml",
    batch_size: int = 64,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """Create t-SNE visualizations of embedding space.

    Args:
        encoder_path: Path to trained encoder checkpoint
        output_dir: Directory to save visualization PNGs
        n_samples: Number of samples to visualize
        config_path: Path to config.yaml
        batch_size: Batch size for inference
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
    """
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading encoder from {encoder_path}")
    model = load_encoder(encoder_path, config_path, device)

    # Get test dataset
    print("Loading CelebA test dataset...")
    celeba_test, _ = get_test_datasets(config_path)
    print(f"Available samples: {len(celeba_test)}, using: {min(n_samples, len(celeba_test))}")

    # Extract embeddings
    embeddings, attributes = extract_embeddings(
        model, celeba_test, n_samples, batch_size, device
    )
    print(f"Extracted embeddings: {embeddings.shape}")

    # Run t-SNE
    embeddings_2d = run_tsne(embeddings, perplexity, n_iter)

    # Save t-SNE coordinates for later use
    np.savez(
        output_path / "tsne_embeddings.npz",
        embeddings_2d=embeddings_2d,
        attributes=attributes,
    )
    print(f"Saved t-SNE coordinates to {output_path / 'tsne_embeddings.npz'}")

    # Create individual plots for each attribute
    print("\nCreating individual attribute plots...")
    attr_indices = {name: i for i, name in enumerate(CelebADataset.ATTRIBUTE_NAMES)}

    for attr_name in VISUALIZATION_ATTRIBUTES:
        attr_idx = attr_indices[attr_name]
        labels = attributes[:, attr_idx]
        create_scatter_plot(
            embeddings_2d,
            labels,
            attr_name,
            output_path / f"tsne_{attr_name}.png",
        )

    # Create combined plot
    print("\nCreating combined subplot...")
    create_combined_plot(
        embeddings_2d,
        attributes,
        VISUALIZATION_ATTRIBUTES,
        output_path / "tsne_combined.png",
    )

    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding space with t-SNE")
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/best_encoder.pt",
        help="Path to encoder checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Output directory for visualization PNGs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="t-SNE iterations",
    )

    args = parser.parse_args()

    visualize_embeddings(
        encoder_path=args.encoder,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        config_path=args.config,
        batch_size=args.batch_size,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )


if __name__ == "__main__":
    main()
