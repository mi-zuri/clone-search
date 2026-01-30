"""Training script for FaceEncoder with SimCLR + attribute prediction."""

import argparse
import re  # Added for parsing filenames
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import yaml

from src.data.augmentations import get_simclr_augmentations, get_val_augmentations
from src.data.dataset import CelebADataset, CombinedDataset, FFHQDataset, SimCLRWrapper
from src.models.encoder import FaceEncoder


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.25) -> torch.Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR."""
    batch_size = z1.shape[0]
    device = z1.device

    # Concatenate both views: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarities (diagonal)
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # For each sample, positive pair is at offset batch_size
    pos_indices = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)],
        dim=0,
    ).to(device)  # (2B,)

    loss = F.cross_entropy(sim, pos_indices)
    return loss


def compute_pos_weights(train_dataset: CombinedDataset) -> torch.Tensor:
    """Compute positive class weights for BCEWithLogitsLoss."""
    celeba = train_dataset.celeba
    num_samples = len(celeba)

    pos_counts = torch.zeros(40)
    for i in range(num_samples):
        sample = celeba.samples[i]
        attrs = torch.tensor(sample["attributes"], dtype=torch.float32)
        pos_counts += attrs

    neg_counts = num_samples - pos_counts
    weights = neg_counts / (pos_counts + 1e-8)
    weights = weights.clamp(min=0.1, max=10.0)

    return weights


def attribute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weights: torch.Tensor,
    has_attributes: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted BCE loss for attribute prediction."""
    mask = has_attributes.bool()
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)

    logits_masked = logits[mask]
    targets_masked = targets[mask]

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(logits.device))
    return criterion(logits_masked, targets_masked)


def validate(
    model: FaceEncoder,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run validation and compute per-attribute accuracy."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["attributes"].to(device)

            _, attr_logits = model(images)
            preds = (attr_logits > 0).float()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    correct = (all_preds == all_targets).float()
    per_attr_acc = correct.mean(dim=0).cpu().tolist()
    mean_acc = correct.mean().item()

    model.train()
    return {"mean_acc": mean_acc, "per_attr_acc": per_attr_acc}


def create_dataloaders(
    config: dict,
    dry_run: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    data_cfg = config["data"]
    encoder_cfg = config["encoder"]
    image_size = data_cfg["image_size"]

    train_end = data_cfg["celeba_train"]
    val_end = train_end + data_cfg["celeba_val"]

    if dry_run:
        train_indices = list(range(64))
        val_indices = list(range(train_end, train_end + 32))
        ffhq_indices = list(range(16))
    else:
        train_indices = list(range(train_end))
        val_indices = list(range(train_end, val_end))
        ffhq_indices = list(range(data_cfg["ffhq_train_subset"]))

    celeba_train = CelebADataset(
        root=data_cfg["celeba_path"],
        attr_path=data_cfg["celeba_attr_path"],
        indices=train_indices,
        image_size=image_size,
    )

    ffhq_train = FFHQDataset(
        root=data_cfg["ffhq_path"],
        indices=ffhq_indices,
        image_size=image_size,
    )

    celeba_val = CelebADataset(
        root=data_cfg["celeba_path"],
        attr_path=data_cfg["celeba_attr_path"],
        transform=get_val_augmentations(image_size),
        indices=val_indices,
        image_size=image_size,
    )

    combined_train = CombinedDataset(celeba_train, ffhq_train)
    simclr_transform = get_simclr_augmentations(image_size)
    train_dataset = SimCLRWrapper(combined_train, simclr_transform)

    sampler_weights = combined_train.get_sampler_weights(data_cfg["celeba_batch_ratio"])
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    loader_kwargs = {
        "num_workers": data_cfg["num_workers"],
        "persistent_workers": True,
        "multiprocessing_context": data_cfg["dataloader_spawn_method"],
        "pin_memory": False,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=encoder_cfg["batch_size"],
        sampler=sampler,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        celeba_val,
        batch_size=encoder_cfg["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: FaceEncoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pos_weights: torch.Tensor,
    config: dict,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
) -> int:
    """Train for one epoch with gradient accumulation."""
    model.train()
    encoder_cfg = config["encoder"]
    training_cfg = config["training"]

    accumulation_steps = encoder_cfg["gradient_accumulation_steps"]
    simclr_weight = encoder_cfg["simclr_loss_weight"]
    attr_weight = encoder_cfg["attribute_loss_weight"]
    temperature = encoder_cfg["simclr_temperature"]
    clip_norm = training_cfg["gradient_clip_norm"]

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        attributes = batch["attributes"].to(device)
        has_attributes = batch["has_attributes"].to(device)

        _, proj1, attr_logits1 = model(view1, return_projection=True)
        _, proj2, _ = model(view2, return_projection=True)

        loss_simclr = nt_xent_loss(proj1, proj2, temperature)
        loss_attr = attribute_loss(attr_logits1, attributes, pos_weights, has_attributes)
        loss_total = simclr_weight * loss_simclr + attr_weight * loss_attr

        loss_scaled = loss_total / accumulation_steps
        loss_scaled.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar("train/loss_total", loss_total.item(), global_step)
            writer.add_scalar("train/loss_simclr", loss_simclr.item(), global_step)
            writer.add_scalar("train/loss_attr", loss_attr.item(), global_step)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)

            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"  Step {global_step}: loss={loss_total.item():.4f} "
                    f"(simclr={loss_simclr.item():.4f}, attr={loss_attr.item():.4f})"
                )

    return global_step


def save_checkpoint(
    model: FaceEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_acc: float,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
    }

    path = checkpoint_dir / f"encoder_epoch{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    if is_best:
        best_path = checkpoint_dir / "best_encoder.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint: {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: FaceEncoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def main(config_path: str, epochs: Optional[int] = None, dry_run: bool = False) -> None:
    """Main training function."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    encoder_cfg = config["encoder"]
    training_cfg = config["training"]

    if epochs is not None:
        encoder_cfg["epochs"] = epochs

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = FaceEncoder(
        embedding_dim=encoder_cfg["embedding_dim"],
        projection_dim=encoder_cfg["projection_dim"],
        num_attributes=encoder_cfg["num_attributes"],
    )
    model = model.to(device)
    print(f"Model: {model.get_trainable_params():,} trainable parameters")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, dry_run=dry_run)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("Computing attribute class weights...")
    simclr_wrapper: SimCLRWrapper = train_loader.dataset  # type: ignore[assignment]
    combined_dataset: CombinedDataset = simclr_wrapper.dataset  # type: ignore[assignment]
    pos_weights = compute_pos_weights(combined_dataset)

    optimizer = AdamW(
        model.parameters(),
        lr=encoder_cfg["lr"],
        weight_decay=encoder_cfg["weight_decay"],
    )

    total_steps = len(train_loader) // encoder_cfg["gradient_accumulation_steps"]
    warmup_steps = total_steps * training_cfg["warmup_epochs"]
    main_steps = total_steps * (encoder_cfg["epochs"] - training_cfg["warmup_epochs"])

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=training_cfg["warmup_start_lr"] / encoder_cfg["lr"],
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=main_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    # --- CHECKPOINT RESUME LOGIC ---
    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    start_epoch = 0
    best_val_acc = 0.0
    global_step = 0

    if checkpoint_dir.exists():
        # Look for files matching 'encoder_epoch{N}.pt'
        checkpoints = list(checkpoint_dir.glob("encoder_epoch*.pt"))
        if checkpoints:
            # Sort to find the highest epoch number
            # Uses regex to extract integer from 'encoder_epoch12.pt'
            latest_ckpt = max(
                checkpoints,
                key=lambda p: int(re.search(r'epoch(\d+)', p.name).group(1))
            )

            print(f"Found checkpoint! Resuming training from: {latest_ckpt}")
            ckpt_data = load_checkpoint(latest_ckpt, model, optimizer, scheduler)

            # Epoch in checkpoint is the one that just finished, so we start from next
            start_epoch = ckpt_data["epoch"]
            if "val_acc" in ckpt_data:
                best_val_acc = ckpt_data["val_acc"]

            # Recalculate global_step so TensorBoard continues smoothly
            # (current_epoch * steps_per_epoch)
            global_step = start_epoch * total_steps
            print(f"Resuming at Epoch {start_epoch + 1}, Global Step {global_step}")

    # TensorBoard
    log_dir = Path(training_cfg["log_dir"]) / "encoder"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")

    patience_counter = 0

    print(f"\nStarting training loop (Epochs {start_epoch + 1} to {encoder_cfg['epochs']})...")

    # Start loop from start_epoch
    for epoch in range(start_epoch, encoder_cfg["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{encoder_cfg['epochs']} ===")

        global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            pos_weights=pos_weights,
            config=config,
            device=device,
            writer=writer,
            global_step=global_step,
        )

        print("Running validation...")
        val_metrics = validate(model, val_loader, device)
        val_acc = float(val_metrics["mean_acc"])
        print(f"Validation accuracy: {val_acc:.4f}")

        writer.add_scalar("val/mean_accuracy", val_acc, epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            val_acc=val_acc,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best,
        )

        if patience_counter >= encoder_cfg["early_stopping_patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    writer.close()
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FaceEncoder with SimCLR")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with small data subset for testing",
    )
    args = parser.parse_args()

    main(args.config, args.epochs, args.dry_run)