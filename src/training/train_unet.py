"""Training script for U-Net inpainting model."""

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
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import CelebAMaskHQDataset, get_dataloaders
from src.models.unet import UNet, InpaintingLoss, compute_psnr, compute_ssim


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device, epoch: int, writer: SummaryWriter):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        masked_images = batch['masked_image'].to(device)

        # Concatenate masked image with mask
        inputs = torch.cat([masked_images, masks], dim=1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, images, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute metrics
        with torch.no_grad():
            psnr = compute_psnr(outputs, images)
            ssim = compute_ssim(outputs, images)
            running_psnr += psnr
            running_ssim += ssim

        pbar.set_postfix({'loss': loss.item(), 'psnr': psnr, 'ssim': ssim})

        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/batch_loss', loss.item(), global_step)

    # Compute epoch metrics
    n = len(dataloader)
    avg_loss = running_loss / n
    avg_psnr = running_psnr / n
    avg_ssim = running_ssim / n

    return avg_loss, avg_psnr, avg_ssim


def validate(model: nn.Module, dataloader, criterion, device, epoch: int, writer: SummaryWriter):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            masked_images = batch['masked_image'].to(device)

            inputs = torch.cat([masked_images, masks], dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, images, masks)

            running_loss += loss.item()

            psnr = compute_psnr(outputs, images)
            ssim = compute_ssim(outputs, images)
            running_psnr += psnr
            running_ssim += ssim

            pbar.set_postfix({'loss': loss.item(), 'psnr': psnr, 'ssim': ssim})

            # Log sample images periodically
            if batch_idx == 0 and epoch % 2 == 0:
                # Log first few images
                n_samples = min(4, images.size(0))
                log_images(
                    writer, epoch,
                    images[:n_samples],
                    masked_images[:n_samples],
                    outputs[:n_samples],
                    masks[:n_samples]
                )

    # Compute epoch metrics
    n = len(dataloader)
    avg_loss = running_loss / n
    avg_psnr = running_psnr / n
    avg_ssim = running_ssim / n

    return avg_loss, avg_psnr, avg_ssim


def log_images(writer: SummaryWriter, epoch: int, originals, masked, reconstructed, masks):
    """Log sample images to TensorBoard."""
    import torchvision.utils as vutils

    # Create grid: [original, masked, reconstructed]
    batch_size = originals.size(0)
    all_images = []

    for i in range(batch_size):
        all_images.extend([originals[i], masked[i], reconstructed[i]])

    grid = vutils.make_grid(all_images, nrow=3, normalize=True, padding=2)
    writer.add_image('Val/samples', grid, epoch)


def main():
    parser = argparse.ArgumentParser(description='Train U-Net Inpainter')
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

    log_dir = Path(config['training']['log_dir']) / f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        celeba_root=args.data_dir,
        batch_size=config['unet']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers'],
        for_inpainting=True
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    model = UNet(in_channels=4, out_channels=3).to(device)

    # Loss and optimizer
    criterion = InpaintingLoss(l1_weight=1.0, mask_weight=6.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['unet']['lr']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['unet']['epochs'],
        eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_psnr = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['unet']['epochs']):
        # Train
        train_loss, train_psnr, train_ssim = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # Update scheduler
        scheduler.step()

        # Log to tensorboard
        writer.add_scalar('Train/epoch_loss', train_loss, epoch)
        writer.add_scalar('Train/PSNR', train_psnr, epoch)
        writer.add_scalar('Train/SSIM', train_ssim, epoch)
        writer.add_scalar('Val/epoch_loss', val_loss, epoch)
        writer.add_scalar('Val/PSNR', val_psnr, epoch)
        writer.add_scalar('Val/SSIM', val_ssim, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Train PSNR={train_psnr:.2f}, Train SSIM={train_ssim:.4f}")
        print(f"           Val Loss={val_loss:.4f}, Val PSNR={val_psnr:.2f}, Val SSIM={val_ssim:.4f}")

        # Save checkpoint
        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'best_val_psnr': best_val_psnr,
            'config': config
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'unet_latest.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'unet_best.pth')
            print(f"  -> New best model saved!")

        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save(checkpoint, checkpoint_dir / f'unet_epoch_{epoch}.pth')

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_psnr, test_ssim = validate(model, test_loader, criterion, device, config['unet']['epochs'], writer)
    print(f"Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}")

    writer.add_scalar('Test/loss', test_loss, 0)
    writer.add_scalar('Test/PSNR', test_psnr, 0)
    writer.add_scalar('Test/SSIM', test_ssim, 0)

    writer.close()
    print(f"\nTraining complete! Best validation PSNR: {best_val_psnr:.2f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
