"""FAISS-based face search engine for 80k gallery images."""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Fix OpenMP conflict between PyTorch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.data import CelebADataset, FFHQDataset, get_val_augmentations
from src.models.encoder import FaceEncoder


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_gallery(
    encoder_path: str,
    output_path: str,
    config_path: str = "configs/config.yaml",
    batch_size: int = 64,
) -> None:
    """Build the gallery index from CelebA and FFHQ datasets.

    Processes all 30k CelebA + 50k FFHQ images through the encoder,
    extracts embeddings and attribute predictions, and saves to NPZ.

    Args:
        encoder_path: Path to trained encoder checkpoint (best_encoder.pt)
        output_path: Output path for gallery NPZ file
        config_path: Path to config.yaml for dataset paths
        batch_size: Batch size for inference (default: 64)
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading encoder from {encoder_path}")
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    model = FaceEncoder(
        embedding_dim=config["encoder"]["embedding_dim"],
        projection_dim=config["encoder"]["projection_dim"],
        num_attributes=config["encoder"]["num_attributes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create datasets with validation augmentations (no random transforms)
    image_size = config["data"]["image_size"]
    transform = get_val_augmentations(image_size)

    print("Loading CelebA dataset...")
    celeba = CelebADataset(
        root=config["data"]["celeba_path"],
        attr_path=config["data"]["celeba_attr_path"],
        transform=transform,
        image_size=image_size,
    )

    print("Loading FFHQ dataset...")
    ffhq = FFHQDataset(
        root=config["data"]["ffhq_path"],
        transform=transform,
        image_size=image_size,
        total_images=config["data"]["ffhq_total"],
    )

    print(f"CelebA: {len(celeba)} images, FFHQ: {len(ffhq)} images")
    print(f"Total gallery size: {len(celeba) + len(ffhq)}")

    # Collect all embeddings and attributes
    all_embeddings = []
    all_attributes = []
    all_paths = []
    all_sources = []

    # Get project root for relative paths
    project_root = Path.cwd()

    # Process CelebA
    print("\nProcessing CelebA...")
    celeba_loader = DataLoader(
        celeba, batch_size=batch_size, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        for batch in tqdm(celeba_loader, desc="CelebA"):
            images = batch["image"].to(device)
            paths = batch["path"]

            embedding, attr_logits = model(images)
            attrs = torch.sigmoid(attr_logits)

            all_embeddings.append(embedding.cpu().numpy())
            all_attributes.append(attrs.cpu().numpy())

            # Store relative paths
            for p in paths:
                path_obj = Path(p)
                if path_obj.is_absolute():
                    rel_path = str(path_obj.relative_to(project_root))
                else:
                    rel_path = str(path_obj)
                all_paths.append(rel_path)
                all_sources.append("celeba")

    # Process FFHQ
    print("\nProcessing FFHQ...")
    ffhq_loader = DataLoader(ffhq, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch in tqdm(ffhq_loader, desc="FFHQ"):
            images = batch["image"].to(device)
            paths = batch["path"]

            embedding, attr_logits = model(images)
            attrs = torch.sigmoid(attr_logits)

            all_embeddings.append(embedding.cpu().numpy())
            all_attributes.append(attrs.cpu().numpy())

            # Store relative paths
            for p in paths:
                path_obj = Path(p)
                if path_obj.is_absolute():
                    rel_path = str(path_obj.relative_to(project_root))
                else:
                    rel_path = str(path_obj)
                all_paths.append(rel_path)
                all_sources.append("ffhq")

    # Concatenate all results
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    attributes = np.vstack(all_attributes).astype(np.float32)
    paths = np.array(all_paths)
    sources = np.array(all_sources)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Attributes shape: {attributes.shape}")

    # Verify embeddings are L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms - mean: {norms.mean():.4f}, std: {norms.std():.6f}")

    # Save to NPZ
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_file,
        embeddings=embeddings,
        attributes=attributes,
        paths=paths,
        sources=sources,
    )

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"\nSaved gallery index to {out_file} ({file_size_mb:.1f} MB)")


class FaceSearchEngine:
    """FAISS-based face search engine with attribute filtering.

    Uses IndexFlatIP (inner product) for cosine similarity search
    on L2-normalized embeddings.
    """

    # Reference to CelebA attribute names for convenience
    ATTRIBUTE_NAMES = CelebADataset.ATTRIBUTE_NAMES

    def __init__(self, gallery_path: str):
        """Initialize the search engine from a gallery NPZ file.

        Args:
            gallery_path: Path to gallery_index.npz created by build_gallery()
        """
        import faiss  # Lazy import to avoid OpenMP conflicts with PyTorch

        self.gallery_path = Path(gallery_path)

        # Load gallery data
        data = np.load(gallery_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.attributes = data["attributes"].astype(np.float32)
        self.paths = data["paths"]
        self.sources = data["sources"]

        # Ensure embeddings are C-contiguous for FAISS
        self.embeddings = np.ascontiguousarray(self.embeddings)

        # Build FAISS index (inner product = cosine sim for L2-normalized vectors)
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings)

        print(f"Loaded gallery with {len(self.embeddings)} faces")
        print(f"Embedding dim: {embedding_dim}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        attribute_filters: Optional[dict[str, bool]] = None,
    ) -> list[dict]:
        """Search for similar faces in the gallery.

        Args:
            query_embedding: L2-normalized query embedding of shape (64,) or (1, 64)
            k: Number of results to return
            attribute_filters: Optional dict mapping attribute names to required values.
                              e.g., {"Smiling": True, "Eyeglasses": False}

        Returns:
            List of k result dicts, each containing:
                - path: Relative path to image
                - similarity: Cosine similarity score (inner product)
                - final_score: Score after attribute filtering (if applied)
                - attributes: Dict mapping attribute names to probabilities
                - source: "celeba" or "ffhq"
        """
        # Ensure query is 2D, float32, and C-contiguous
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = np.ascontiguousarray(query)

        # Determine how many to fetch (fetch more if filtering)
        fetch_k = k * 10 if attribute_filters else k

        # Search
        similarities, indices = self.index.search(query, fetch_k)
        similarities = similarities[0]  # Shape: (fetch_k,)
        indices = indices[0]

        # Build results
        results = []
        for sim, idx in zip(similarities, indices):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue

            # Build attribute dict
            attr_probs = self.attributes[idx]
            attr_dict = {
                name: float(prob)
                for name, prob in zip(self.ATTRIBUTE_NAMES, attr_probs)
            }

            result = {
                "path": str(self.paths[idx]),
                "similarity": float(sim),
                "final_score": float(sim),
                "attributes": attr_dict,
                "source": str(self.sources[idx]),
            }
            results.append(result)

        # Apply attribute filtering if specified
        if attribute_filters:
            results = self._apply_attribute_filters(results, attribute_filters)

        # Return top k results
        return results[:k]

    def _apply_attribute_filters(
        self, results: list[dict], filters: dict[str, bool]
    ) -> list[dict]:
        """Re-rank results based on attribute filter confidence.

        For each result, the final_score is:
            similarity × product(attr_confidence for each filter)

        where attr_confidence = prob if filter is True, (1-prob) if False.

        Args:
            results: List of search results
            filters: Dict mapping attribute names to required values

        Returns:
            Re-ranked results list
        """
        for result in results:
            confidence_product = 1.0
            for attr_name, required in filters.items():
                if attr_name not in self.ATTRIBUTE_NAMES:
                    raise ValueError(f"Unknown attribute: {attr_name}")

                prob = result["attributes"][attr_name]
                # Confidence that the attribute matches the requirement
                confidence = prob if required else (1.0 - prob)
                confidence_product *= confidence

            # Update final score
            result["final_score"] = result["similarity"] * confidence_product

        # Sort by final_score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def get_embedding_by_index(self, idx: int) -> np.ndarray:
        """Get embedding for a specific gallery index."""
        return self.embeddings[idx].copy()

    def __len__(self) -> int:
        """Return number of faces in gallery."""
        return len(self.embeddings)


def main():
    """CLI entrypoint for building gallery index."""
    parser = argparse.ArgumentParser(description="Build face search gallery index")
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/best_encoder.pt",
        help="Path to encoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/gallery_index.npz",
        help="Output path for gallery NPZ",
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

    args = parser.parse_args()

    build_gallery(
        encoder_path=args.encoder,
        output_path=args.output,
        config_path=args.config,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
