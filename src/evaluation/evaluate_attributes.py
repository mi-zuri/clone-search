"""Evaluate attribute prediction accuracy on CelebA test set."""

import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CelebADataset
from src.evaluation.utils import get_device, load_encoder, save_json
from src.search.splits import get_test_datasets


def evaluate_attributes(
    encoder_path: str,
    output_path: str,
    config_path: str = "configs/config.yaml",
    batch_size: int = 64,
) -> dict:
    """Evaluate encoder attribute prediction on CelebA test set.

    Args:
        encoder_path: Path to trained encoder checkpoint
        output_path: Path to save JSON results
        config_path: Path to config.yaml
        batch_size: Batch size for inference

    Returns:
        Dict with evaluation metrics
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading encoder from {encoder_path}")
    model = load_encoder(encoder_path, config_path, device)

    # Get test dataset (CelebA only - has ground truth attributes)
    print("Loading CelebA test dataset...")
    celeba_test, _ = get_test_datasets(config_path)
    print(f"Test samples: {len(celeba_test)}")

    loader = DataLoader(
        celeba_test, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Collect all predictions and ground truth
    all_preds = []
    all_targets = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            targets = batch["attributes"]  # Shape: (B, 40)

            # Get attribute predictions
            _, attr_logits = model(images)

            # Threshold at logit=0 (equivalent to sigmoid > 0.5)
            preds = (attr_logits > 0).cpu().float()

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate all batches
    preds = torch.cat(all_preds, dim=0).numpy()  # (N, 40)
    targets = torch.cat(all_targets, dim=0).numpy()  # (N, 40)

    # Compute per-attribute metrics
    attr_names = CelebADataset.ATTRIBUTE_NAMES
    per_attr_metrics = {}

    for i, name in enumerate(attr_names):
        y_true = targets[:, i]
        y_pred = preds[:, i]

        per_attr_metrics[name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division="warn")),
            "positive_rate": float(y_true.mean()),  # Class imbalance info
        }

    # Compute summary statistics
    accuracies = [m["accuracy"] for m in per_attr_metrics.values()]
    balanced_accs = [m["balanced_accuracy"] for m in per_attr_metrics.values()]
    f1_scores = [m["f1"] for m in per_attr_metrics.values()]

    # Sort by accuracy to find best/worst
    sorted_attrs = sorted(
        per_attr_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    best_5 = [{"name": name, **metrics} for name, metrics in sorted_attrs[:5]]
    worst_5 = [{"name": name, **metrics} for name, metrics in sorted_attrs[-5:]]

    results = {
        "summary": {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "mean_balanced_accuracy": sum(balanced_accs) / len(balanced_accs),
            "mean_f1": sum(f1_scores) / len(f1_scores),
            "n_samples": len(targets),
            "n_attributes": len(attr_names),
        },
        "best_5_attributes": best_5,
        "worst_5_attributes": worst_5,
        "per_attribute": per_attr_metrics,
    }

    # Save JSON
    save_json(results, output_path)

    # Also save CSV for easy spreadsheet viewing
    csv_path = Path(output_path).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute", "accuracy", "balanced_accuracy", "f1", "positive_rate"])
        for name, metrics in sorted_attrs:
            writer.writerow([
                name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['balanced_accuracy']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['positive_rate']:.4f}",
            ])
    print(f"Saved CSV to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ATTRIBUTE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test samples:       {results['summary']['n_samples']}")
    print(f"Mean accuracy:      {results['summary']['mean_accuracy']:.4f}")
    print(f"Mean balanced acc:  {results['summary']['mean_balanced_accuracy']:.4f}")
    print(f"Mean F1:            {results['summary']['mean_f1']:.4f}")
    print()
    print("Best 5 attributes:")
    for attr in best_5:
        print(f"  {attr['name']:25s}  acc={attr['accuracy']:.4f}  bal_acc={attr['balanced_accuracy']:.4f}")
    print()
    print("Worst 5 attributes:")
    for attr in worst_5:
        print(f"  {attr['name']:25s}  acc={attr['accuracy']:.4f}  bal_acc={attr['balanced_accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate attribute prediction on CelebA test set")
    parser.add_argument(
        "--encoder",
        type=str,
        default="checkpoints/best_encoder.pt",
        help="Path to encoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/attribute_eval.json",
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
        help="Batch size for inference",
    )

    args = parser.parse_args()

    evaluate_attributes(
        encoder_path=args.encoder,
        output_path=args.output,
        config_path=args.config,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
