#!/usr/bin/env python3
"""
Generate all figures for the REPORT.md from TensorBoard logs and evaluation results.

This script creates 6 publication-quality plots:
1. architecture.png - Model architecture diagram
2. training_loss.png - Training loss curves (total, SimCLR, attribute)
3. learning_rate.png - Learning rate schedule
4. validation_accuracy.png - Validation accuracy over epochs
5. attribute_accuracy.png - Per-attribute accuracy distribution
6. recall_curves.png - Retrieval recall@K metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from tensorboard.backend.event_processing import event_accumulator


# Consistent styling parameters
FIGURE_SIZE_WIDE = (10, 6)
FIGURE_SIZE_SQUARE = (8, 8)
DPI = 150
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK = 12
GRID_ALPHA = 0.3
LINE_WIDTH = 1.5


def read_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse TensorBoard event files and extract scalar metrics.

    Args:
        log_dir: Directory containing TensorBoard event files

    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    # Find all event files
    event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    # Use the most recent event file
    event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    print(f"Reading TensorBoard logs from: {event_file.name}")

    # Load events
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()

    # Extract scalar metrics
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
        print(f"  - {tag}: {len(events)} data points")

    return data


def plot_training_losses(data: Dict[str, List[Tuple[int, float]]], save_path: str):
    """Generate training loss curves plot."""
    print("\nGenerating training_loss.png...")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)

    # Plot each loss component
    loss_metrics = {
        'train/loss_total': ('Total Loss', '#1f77b4'),
        'train/loss_simclr': ('SimCLR Loss', '#ff7f0e'),
        'train/loss_attr': ('Attribute Loss', '#2ca02c')
    }

    for metric, (label, color) in loss_metrics.items():
        if metric in data:
            steps, values = zip(*data[metric])
            ax.plot(steps, values, label=label, color=color, linewidth=LINE_WIDTH)

    ax.set_xlabel('Training Step', fontsize=FONT_LABEL)
    ax.set_ylabel('Loss', fontsize=FONT_LABEL)
    ax.set_title('Training Loss Curves', fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def plot_learning_rate(data: Dict[str, List[Tuple[int, float]]], save_path: str):
    """Generate learning rate schedule plot."""
    print("\nGenerating learning_rate.png...")

    if 'train/learning_rate' not in data:
        print("  Warning: No learning_rate data found, skipping...")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)

    steps, values = zip(*data['train/learning_rate'])
    ax.plot(steps, values, color='#ff7f0e', linewidth=LINE_WIDTH)

    # Highlight warmup region (assume first ~93 steps = 1 epoch)
    warmup_steps = 93  # 561 total / 6 epochs ≈ 93 per epoch
    ax.axvspan(0, warmup_steps, alpha=0.15, color='orange', label='Warmup Phase')

    ax.set_xlabel('Training Step', fontsize=FONT_LABEL)
    ax.set_ylabel('Learning Rate', fontsize=FONT_LABEL)
    ax.set_title('Learning Rate Schedule', fontsize=FONT_TITLE)
    ax.set_yscale('log')
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def plot_validation_accuracy(checkpoints_dir: str, save_path: str):
    """Generate validation accuracy plot from checkpoint files."""
    print("\nGenerating validation_accuracy.png...")

    checkpoint_files = sorted(Path(checkpoints_dir).glob("encoder_epoch*.pt"))
    if not checkpoint_files:
        print("  Warning: No checkpoint files found, skipping...")
        return

    # Extract validation accuracy from each checkpoint
    epochs = []
    accuracies = []

    for ckpt_path in checkpoint_files:
        # Extract epoch number from filename
        epoch = int(ckpt_path.stem.replace('encoder_epoch', ''))

        # Load checkpoint and extract validation accuracy
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # Try both 'val_acc' and 'val_accuracy' keys
        val_acc = checkpoint.get('val_acc') or checkpoint.get('val_accuracy')
        if val_acc is not None:
            epochs.append(epoch)
            accuracies.append(val_acc * 100)  # Convert to percentage
            print(f"  - Epoch {epoch}: {val_acc*100:.2f}%")

    if not epochs:
        print("  Warning: No validation accuracy found in checkpoints")
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)

    ax.plot(epochs, accuracies, marker='o', color='#1f77b4',
            linewidth=LINE_WIDTH, markersize=8)

    # Highlight best epoch
    best_idx = np.argmax(accuracies)
    ax.plot(epochs[best_idx], accuracies[best_idx], marker='*',
            color='gold', markersize=15, label=f'Best: {accuracies[best_idx]:.2f}%')

    ax.set_xlabel('Epoch', fontsize=FONT_LABEL)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=FONT_LABEL)
    ax.set_title('Validation Accuracy Over Training', fontsize=FONT_TITLE)
    ax.set_xticks(epochs)
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def plot_attribute_accuracy(json_path: str, save_path: str):
    """Generate per-attribute accuracy distribution plot."""
    print("\nGenerating attribute_accuracy.png...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract all attributes and sort by accuracy
    attr_data = []
    for attr_name, metrics in data['per_attribute'].items():
        attr_data.append((attr_name, metrics['accuracy'] * 100))

    attr_data.sort(key=lambda x: x[1], reverse=True)
    names, accuracies = zip(*attr_data)

    # Color-code bars based on performance
    colors = []
    for acc in accuracies:
        if acc >= 90:
            colors.append('#2ca02c')  # Green
        elif acc >= 70:
            colors.append('#ff7f0e')  # Orange
        else:
            colors.append('#d62728')  # Red

    fig, ax = plt.subplots(figsize=(10, 12), dpi=DPI)

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accuracies, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=FONT_LABEL)
    ax.set_title('Per-Attribute Classification Accuracy (40 attributes)', fontsize=FONT_TITLE)
    ax.grid(True, axis='x', alpha=GRID_ALPHA)
    ax.tick_params(labelsize=FONT_TICK)

    # Add accuracy labels for top-5 and bottom-5
    for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
        ax.text(accuracies[i] + 1, y_pos[i], f'{accuracies[i]:.1f}%',
                va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.8, label='≥90% (Excellent)'),
        Patch(facecolor='#ff7f0e', alpha=0.8, label='70-90% (Good)'),
        Patch(facecolor='#d62728', alpha=0.8, label='<70% (Challenging)')
    ]
    ax.legend(handles=legend_elements, fontsize=FONT_TICK, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def plot_recall_curves(json_path: str, save_path: str):
    """Generate retrieval recall@K comparison plot."""
    print("\nGenerating recall_curves.png...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract recall metrics
    recall_80 = [
        data['threshold_80']['recall@1'] * 100,
        data['threshold_80']['recall@5'] * 100,
        data['threshold_80']['recall@10'] * 100
    ]
    recall_90 = [
        data['threshold_90']['recall@1'] * 100,
        data['threshold_90']['recall@5'] * 100,
        data['threshold_90']['recall@10'] * 100
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE, dpi=DPI)

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, recall_80, width, label='80% Match (32/40 attrs)',
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, recall_90, width, label='90% Match (36/40 attrs)',
                   color='#d62728', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Metric', fontsize=FONT_LABEL)
    ax.set_ylabel('Recall (%)', fontsize=FONT_LABEL)
    ax.set_title('Retrieval Performance: Recall@K by Similarity Threshold', fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(['Recall@1', 'Recall@5', 'Recall@10'])
    ax.legend(fontsize=FONT_TICK, framealpha=0.9)
    ax.grid(True, axis='y', alpha=GRID_ALPHA)
    ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def create_architecture_diagram(save_path: str):
    """Generate model architecture diagram."""
    print("\nGenerating architecture.png...")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Helper function to create boxes
    def add_box(x, y, width, height, text, color='lightblue'):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=11, weight='bold')

    # Helper function to create arrows
    def add_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black')
        ax.add_patch(arrow)

    # Input
    add_box(4, 8.5, 2, 0.8, 'Input Image\n224×224×3', '#e1f5ff')

    # Backbone
    add_box(3.5, 6.8, 3, 1.2, 'MobileNetV2\nBackbone\n(pretrained)', '#b3e5fc')
    add_arrow(5, 8.5, 5, 8.0)

    # Feature vector
    add_box(4, 5.2, 2, 0.8, 'Features\n1280-d', '#81d4fa')
    add_arrow(5, 6.8, 5, 6.0)

    # Three heads
    head_y = 3.5
    head_width = 2.2
    head_height = 1.0

    # Projection head
    add_box(0.5, head_y, head_width, head_height,
           'Projection Head\n1280 → 128\nL2 normalized', '#ffccbc')
    add_arrow(4, 5.2, 1.6, head_y + head_height)

    # Age head
    add_box(3.9, head_y, head_width, head_height,
           'Age Head\n1280 → 256 → 1\nMSE Loss', '#c5e1a5')
    add_arrow(5, 5.2, 5, head_y + head_height)

    # Attribute head
    add_box(7.3, head_y, head_width, head_height,
           'Attribute Head\n1280 → 512 → 40\nBCE Loss', '#fff9c4')
    add_arrow(6, 5.2, 8.4, head_y + head_height)

    # Outputs
    add_box(0.5, 1.8, head_width, 0.8, 'SimCLR\nEmbedding', '#ff8a65')
    add_arrow(1.6, head_y, 1.6, 2.6)

    add_box(3.9, 1.8, head_width, 0.8, 'Age\nPrediction', '#9ccc65')
    add_arrow(5, head_y, 5, 2.6)

    add_box(7.3, 1.8, head_width, 0.8, 'Attributes\n(40 binary)', '#fff176')
    add_arrow(8.4, head_y, 8.4, 2.6)

    # Title
    ax.text(5, 9.5, 'FaceEncoder Architecture',
           ha='center', fontsize=FONT_TITLE, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")


def main():
    """Generate all report figures."""
    print("=" * 60)
    print("Generating Report Figures for CV Clone Search Project")
    print("=" * 60)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / 'runs' / 'encoder'
    checkpoints_dir = project_root / 'checkpoints'
    results_dir = project_root / 'results'
    output_dir = project_root / 'docs' / 'figures'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Read TensorBoard logs
    try:
        tb_data = read_tensorboard_logs(str(log_dir))
    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        tb_data = {}

    # Generate plots
    try:
        # 1. Architecture diagram
        create_architecture_diagram(str(output_dir / 'architecture.png'))

        # 2. Training losses
        if tb_data:
            plot_training_losses(tb_data, str(output_dir / 'training_loss.png'))

        # 3. Learning rate
        if tb_data:
            plot_learning_rate(tb_data, str(output_dir / 'learning_rate.png'))

        # 4. Validation accuracy
        plot_validation_accuracy(str(checkpoints_dir),
                                str(output_dir / 'validation_accuracy.png'))

        # 5. Attribute accuracy
        plot_attribute_accuracy(str(results_dir / 'attribute_eval.json'),
                               str(output_dir / 'attribute_accuracy.png'))

        # 6. Recall curves
        plot_recall_curves(str(results_dir / 'retrieval_metrics.json'),
                          str(output_dir / 'recall_curves.png'))

        print("\n" + "=" * 60)
        print("✓ All 6 figures generated successfully!")
        print("=" * 60)
        print(f"\nFigures saved to: {output_dir}")
        print("\nGenerated files:")
        for png_file in sorted(output_dir.glob('*.png')):
            size_kb = png_file.stat().st_size / 1024
            print(f"  - {png_file.name} ({size_kb:.1f} KB)")

    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
