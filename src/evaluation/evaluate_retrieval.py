"""Evaluate retrieval quality with Recall@K and MRR metrics.

Uses attribute agreement as a relevance proxy since CelebA lacks person IDs.
Computes metrics at two thresholds: 80% (32/40 attrs) and 90% (36/40 attrs).
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.utils import get_device, load_encoder, load_gallery, save_json
from src.search.splits import get_test_datasets


# Relevance thresholds (number of matching attributes out of 40)
THRESHOLD_80 = 32  # 80% = 32/40 attrs match
THRESHOLD_90 = 36  # 90% = 36/40 attrs match


def compute_attribute_agreement(query_attrs: np.ndarray, gallery_attrs: np.ndarray) -> int:
    """Compute number of matching attributes between query and gallery item.

    Args:
        query_attrs: Binary attributes for query (40,)
        gallery_attrs: Binary attributes for gallery item (40,)

    Returns:
        Number of matching attributes (0-40)
    """
    query_binary = (query_attrs > 0.5).astype(int)
    gallery_binary = (gallery_attrs > 0.5).astype(int)
    return int((query_binary == gallery_binary).sum())


def is_relevant(query_attrs: np.ndarray, gallery_attrs: np.ndarray, threshold: int) -> bool:
    """Check if gallery item is relevant to query based on attribute agreement.

    Args:
        query_attrs: Binary attributes for query (40,)
        gallery_attrs: Binary attributes for gallery item (40,)
        threshold: Minimum number of matching attributes

    Returns:
        True if attribute agreement >= threshold
    """
    return compute_attribute_agreement(query_attrs, gallery_attrs) >= threshold


def compute_retrieval_metrics(
    query_paths: list[str],
    query_attrs: np.ndarray,
    search_results: list[list[dict]],
    gallery_attrs: np.ndarray,
    gallery_paths: np.ndarray,
    threshold: int,
    k_values: list[int] = [1, 5, 10],
) -> dict:
    """Compute Recall@K and MRR metrics.

    Args:
        query_paths: List of query image paths
        query_attrs: Query attribute matrix (N_query, 40)
        search_results: List of search result lists for each query
        gallery_attrs: Gallery attribute matrix (N_gallery, 40)
        gallery_paths: Gallery image paths
        threshold: Attribute agreement threshold for relevance
        k_values: K values for Recall@K

    Returns:
        Dict with recall@k and mrr metrics
    """
    recalls = {k: [] for k in k_values}
    reciprocal_ranks = []

    # Build path to index mapping for gallery
    path_to_idx = {str(p): i for i, p in enumerate(gallery_paths)}

    for q_path, q_attrs, results in zip(query_paths, query_attrs, search_results):
        # Find first relevant result (for MRR) and check relevance at each k
        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            result_path = result["path"]

            # Skip self-matches
            if result_path == q_path:
                continue

            # Get gallery attributes for this result
            if result_path in path_to_idx:
                gallery_idx = path_to_idx[result_path]
                result_attrs = gallery_attrs[gallery_idx]
            else:
                # Result not in gallery (shouldn't happen)
                continue

            # Check relevance
            if is_relevant(q_attrs, result_attrs, threshold):
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        # Compute metrics
        for k in k_values:
            # Recall@K: 1 if at least one relevant in top-k, else 0
            if first_relevant_rank is not None and first_relevant_rank <= k:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)

        # MRR: 1/rank of first relevant, or 0 if none found
        if first_relevant_rank is not None:
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k]))
    metrics["mrr"] = float(np.mean(reciprocal_ranks))

    return metrics


def evaluate_retrieval(
    encoder_path: str,
    gallery_path: str,
    output_path: str,
    config_path: str = "configs/config.yaml",
    batch_size: int = 64,
    search_k: int = 100,
) -> dict:
    """Evaluate retrieval quality using CelebA test set as queries.

    Args:
        encoder_path: Path to trained encoder checkpoint
        gallery_path: Path to gallery NPZ file
        output_path: Path to save JSON results
        config_path: Path to config.yaml
        batch_size: Batch size for inference
        search_k: Number of results to retrieve per query

    Returns:
        Dict with retrieval metrics
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading encoder from {encoder_path}")
    model = load_encoder(encoder_path, config_path, device)

    # Load gallery
    print(f"Loading gallery from {gallery_path}")
    engine = load_gallery(gallery_path)
    print(f"Gallery size: {len(engine)}")

    # Get test dataset
    print("Loading CelebA test dataset...")
    celeba_test, _ = get_test_datasets(config_path)
    print(f"Query samples: {len(celeba_test)}")

    loader = DataLoader(
        celeba_test, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Encode all queries
    print("Encoding query images...")
    query_embeddings = []
    query_attrs = []
    query_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding queries"):
            images = batch["image"].to(device)
            attrs = batch["attributes"]
            paths = batch["path"]

            embedding, _ = model(images)

            query_embeddings.append(embedding.cpu().numpy())
            query_attrs.append(attrs.numpy())
            query_paths.extend(paths)

    query_embeddings = np.vstack(query_embeddings)
    query_attrs = np.vstack(query_attrs)

    print(f"Query embeddings shape: {query_embeddings.shape}")

    # Search gallery for each query
    print(f"Searching gallery (k={search_k})...")
    all_results = []

    for i in tqdm(range(len(query_embeddings)), desc="Searching"):
        results = engine.search(query_embeddings[i], k=search_k)
        all_results.append(results)

    # Compute metrics at both thresholds
    print("\nComputing retrieval metrics...")

    metrics_80 = compute_retrieval_metrics(
        query_paths,
        query_attrs,
        all_results,
        engine.attributes,
        engine.paths,
        threshold=THRESHOLD_80,
    )

    metrics_90 = compute_retrieval_metrics(
        query_paths,
        query_attrs,
        all_results,
        engine.attributes,
        engine.paths,
        threshold=THRESHOLD_90,
    )

    results = {
        "threshold_80": {
            "description": "80% attribute match (32/40 attrs)",
            **metrics_80,
        },
        "threshold_90": {
            "description": "90% attribute match (36/40 attrs)",
            **metrics_90,
        },
        "n_queries": len(query_embeddings),
        "n_gallery": len(engine),
        "search_k": search_k,
    }

    # Save results
    save_json(results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Queries: {results['n_queries']}, Gallery: {results['n_gallery']}")
    print()
    print("80% Threshold (32/40 attrs match):")
    print(f"  Recall@1:  {metrics_80['recall@1']:.4f}")
    print(f"  Recall@5:  {metrics_80['recall@5']:.4f}")
    print(f"  Recall@10: {metrics_80['recall@10']:.4f}")
    print(f"  MRR:       {metrics_80['mrr']:.4f}")
    print()
    print("90% Threshold (36/40 attrs match):")
    print(f"  Recall@1:  {metrics_90['recall@1']:.4f}")
    print(f"  Recall@5:  {metrics_90['recall@5']:.4f}")
    print(f"  Recall@10: {metrics_90['recall@10']:.4f}")
    print(f"  MRR:       {metrics_90['mrr']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality with Recall@K and MRR")
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/best_encoder.pt",
        help="Path to encoder checkpoint",
    )
    parser.add_argument(
        "--gallery",
        type=str,
        default="checkpoints/gallery_index.npz",
        help="Path to gallery NPZ file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/retrieval_metrics.json",
        help="Output path for JSON results",
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
        help="Batch size for encoding queries",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=100,
        help="Number of results to retrieve per query",
    )

    args = parser.parse_args()

    evaluate_retrieval(
        encoder_path=args.encoder,
        gallery_path=args.gallery,
        output_path=args.output,
        config_path=args.config,
        batch_size=args.batch_size,
        search_k=args.search_k,
    )


if __name__ == "__main__":
    main()
