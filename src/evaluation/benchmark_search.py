"""Benchmark search performance: latency, throughput, and memory usage."""

import argparse
import gc
import os
import time

import numpy as np

from src.evaluation.utils import load_gallery, save_json


def measure_latency(
    engine,
    n_queries: int = 100,
    k: int = 5,
    attribute_filters: dict | None = None,
) -> dict:
    """Measure search latency statistics.

    Args:
        engine: FaceSearchEngine instance
        n_queries: Number of queries to run
        k: Number of results per query
        attribute_filters: Optional attribute filters

    Returns:
        Dict with latency statistics (in milliseconds)
    """
    # Generate random query embeddings (normalized)
    np.random.seed(42)
    query_embeddings = np.random.randn(n_queries, engine.embeddings.shape[1]).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Warm-up (first few queries may be slower due to caching)
    for i in range(min(5, n_queries)):
        _ = engine.search(query_embeddings[i], k=k, attribute_filters=attribute_filters)

    # Measure latencies
    latencies = []
    for i in range(n_queries):
        start = time.perf_counter()
        _ = engine.search(query_embeddings[i], k=k, attribute_filters=attribute_filters)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def measure_throughput(
    engine,
    duration_seconds: float = 5.0,
    k: int = 5,
    attribute_filters: dict | None = None,
) -> float:
    """Measure search throughput (queries per second).

    Args:
        engine: FaceSearchEngine instance
        duration_seconds: How long to run the benchmark
        k: Number of results per query
        attribute_filters: Optional attribute filters

    Returns:
        Queries per second
    """
    # Generate random queries
    np.random.seed(42)
    query_embeddings = np.random.randn(1000, engine.embeddings.shape[1]).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Warm-up
    for i in range(5):
        _ = engine.search(query_embeddings[i], k=k, attribute_filters=attribute_filters)

    # Run for fixed duration
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        _ = engine.search(query_embeddings[count % len(query_embeddings)], k=k, attribute_filters=attribute_filters)
        count += 1

    elapsed = time.perf_counter() - start
    return count / elapsed


def measure_memory(gallery_path: str) -> dict:
    """Measure gallery memory footprint.

    Args:
        gallery_path: Path to gallery NPZ file

    Returns:
        Dict with memory statistics
    """
    # File size on disk
    file_size_mb = os.path.getsize(gallery_path) / (1024 * 1024)

    # Load gallery data to measure array sizes
    data = np.load(gallery_path, allow_pickle=True)
    embeddings = data["embeddings"]
    attributes = data["attributes"]
    paths = data["paths"]

    embeddings_mb = embeddings.nbytes / (1024 * 1024)
    attributes_mb = attributes.nbytes / (1024 * 1024)

    # Paths are stored as object array, estimate size
    paths_chars = sum(len(str(p)) for p in paths)
    paths_mb = paths_chars / (1024 * 1024)

    return {
        "file_size_mb": file_size_mb,
        "embeddings_mb": embeddings_mb,
        "attributes_mb": attributes_mb,
        "paths_mb": paths_mb,
        "total_loaded_mb": embeddings_mb + attributes_mb + paths_mb,
        "n_items": len(embeddings),
        "embedding_dim": embeddings.shape[1],
    }


def benchmark_search(
    gallery_path: str,
    output_path: str,
    n_queries: int = 100,
    throughput_duration: float = 5.0,
) -> dict:
    """Run comprehensive search benchmarks.

    Args:
        gallery_path: Path to gallery NPZ file
        output_path: Path to save JSON results
        n_queries: Number of queries for latency measurement
        throughput_duration: Duration for throughput measurement

    Returns:
        Dict with all benchmark results
    """
    print(f"Loading gallery from {gallery_path}")
    engine = load_gallery(gallery_path)
    print(f"Gallery size: {len(engine)}")

    # Force garbage collection before benchmarks
    gc.collect()

    results = {
        "gallery_size": len(engine),
        "embedding_dim": engine.embeddings.shape[1],
        "scenarios": {},
    }

    # Define benchmark scenarios
    scenarios = [
        {
            "name": "basic_k5",
            "description": "Basic search (k=5, no filters)",
            "k": 5,
            "filters": None,
        },
        {
            "name": "filtered_smiling",
            "description": "Filtered search (k=5, Smiling=True)",
            "k": 5,
            "filters": {"Smiling": True},
        },
        {
            "name": "filtered_multi",
            "description": "Multi-filter search (k=5, Smiling+Male)",
            "k": 5,
            "filters": {"Smiling": True, "Male": True},
        },
        {
            "name": "high_k",
            "description": "High-k search (k=100, no filters)",
            "k": 100,
            "filters": None,
        },
    ]

    for scenario in scenarios:
        print(f"\nBenchmarking: {scenario['description']}")

        # Latency
        print(f"  Measuring latency ({n_queries} queries)...")
        latency = measure_latency(
            engine,
            n_queries=n_queries,
            k=scenario["k"],
            attribute_filters=scenario["filters"],
        )

        # Throughput
        print(f"  Measuring throughput ({throughput_duration}s)...")
        qps = measure_throughput(
            engine,
            duration_seconds=throughput_duration,
            k=scenario["k"],
            attribute_filters=scenario["filters"],
        )

        results["scenarios"][scenario["name"]] = {
            "description": scenario["description"],
            "k": scenario["k"],
            "filters": scenario["filters"],
            "latency": latency,
            "throughput_qps": qps,
        }

        print(f"    Mean latency: {latency['mean_ms']:.3f}ms")
        print(f"    P95 latency:  {latency['p95_ms']:.3f}ms")
        print(f"    Throughput:   {qps:.1f} queries/sec")

    # Memory metrics
    print("\nMeasuring memory footprint...")
    memory = measure_memory(gallery_path)
    results["memory"] = memory

    # Save results
    save_json(results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SEARCH BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Gallery: {results['gallery_size']} faces, {results['embedding_dim']}D embeddings")
    print()

    print("Latency (ms):")
    print(f"{'Scenario':<25} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("-" * 60)
    for name, data in results["scenarios"].items():
        lat = data["latency"]
        print(f"{name:<25} {lat['mean_ms']:>8.3f} {lat['p50_ms']:>8.3f} {lat['p95_ms']:>8.3f} {lat['p99_ms']:>8.3f}")

    print()
    print("Throughput:")
    for name, data in results["scenarios"].items():
        print(f"  {name}: {data['throughput_qps']:.1f} queries/sec")

    print()
    print("Memory:")
    print(f"  File on disk:     {memory['file_size_mb']:.1f} MB")
    print(f"  Embeddings:       {memory['embeddings_mb']:.1f} MB")
    print(f"  Attributes:       {memory['attributes_mb']:.1f} MB")
    print(f"  Total in memory:  {memory['total_loaded_mb']:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark search performance")
    parser.add_argument(
        "--gallery",
        type=str,
        default="checkpoints/gallery_index.npz",
        help="Path to gallery NPZ file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/search_benchmarks.json",
        help="Output path for JSON results",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of queries for latency measurement",
    )
    parser.add_argument(
        "--throughput-duration",
        type=float,
        default=5.0,
        help="Duration in seconds for throughput measurement",
    )

    args = parser.parse_args()

    benchmark_search(
        gallery_path=args.gallery,
        output_path=args.output,
        n_queries=args.n_queries,
        throughput_duration=args.throughput_duration,
    )


if __name__ == "__main__":
    main()
