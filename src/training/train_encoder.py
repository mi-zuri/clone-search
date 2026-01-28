"""Training script for face encoder."""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import CelebAMaskHQDataset, get_dataloaders, CELEBA_ATTRIBUTES
from src.models.encoder import FaceEncoder, EncoderLoss, compute_attribute_accuracy


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device, epoch: int, writer: SummaryWriter):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = batch['attributes'].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss, loss_dict = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(outputs['attributes'].detach())
        all_targets.append(targets.detach())

        pbar.set_postfix({'loss': loss.item()})

        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/batch_loss', loss.item(), global_step)

    # Compute epoch metrics
    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    acc_metrics = compute_attribute_accuracy(all_preds, all_targets)

    return avg_loss, acc_metrics


def validate(model: nn.Module, dataloader, criterion, device, epoch: int, writer: SummaryWriter):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['attributes'].to(device)

            outputs = model(images)
            loss, _ = criterion(outputs, targets)

            running_loss += loss.item()
            all_preds.append(outputs['attributes'])
            all_targets.append(targets)

            pbar.set_postfix({'loss': loss.item()})

    # Compute epoch metrics
    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    acc_metrics = compute_attribute_accuracy(all_preds, all_targets)

    return avg_loss, acc_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Face Encoder')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/CelebAMask-HQ', help='Path to CelebAMask-HQ')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)

    log_dir = Path(config['training']['log_dir']) / f"encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        celeba_root=args.data_dir,
        batch_size=config['encoder']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers'],
        for_inpainting=False
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    model = FaceEncoder(
        embedding_dim=config['encoder']['embedding_dim'],
        num_attributes=config['encoder']['num_attributes']
    ).to(device)

    # Loss and optimizer
    criterion = EncoderLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['encoder']['lr'],
        weight_decay=config['encoder']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['encoder']['epochs'],
        eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['encoder']['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # Update scheduler
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar('Train/epoch_loss', train_loss, epoch)
        writer.add_scalar('Train/mean_accuracy', train_acc['mean_accuracy'], epoch)
        writer.add_scalar('Val/epoch_loss', val_loss, epoch)
        writer.add_scalar('Val/mean_accuracy', val_acc['mean_accuracy'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Log per-attribute accuracy
        for i, attr_name in enumerate(CELEBA_ATTRIBUTES):
            writer.add_scalar(f'Val/attr_{attr_name}', val_acc['per_attribute_accuracy'][i], epoch)

        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc['mean_accuracy']:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc['mean_accuracy']:.4f}")

        # Save checkpoint
        is_best = val_acc['mean_accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_acc['mean_accuracy']

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc['mean_accuracy'],
            'best_val_acc': best_val_acc,
            'config': config
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'encoder_latest.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'encoder_best.pth')
            print(f"  -> New best model saved!")

        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save(checkpoint, checkpoint_dir / f'encoder_epoch_{epoch}.pth')

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device, config['encoder']['epochs'], writer)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['mean_accuracy']:.4f}")

    writer.add_scalar('Test/loss', test_loss, 0)
    writer.add_scalar('Test/mean_accuracy', test_acc['mean_accuracy'], 0)

    writer.close()
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
