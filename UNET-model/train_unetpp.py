import os
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from datetime import datetime
import json

# Disable albumentations update check warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# Import enhanced dataloader instead of original
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path
from enhanced_dataloader import get_1_to_3_augmentation_loaders

from utils.metrics import (
    CombinedLoss, DiceLoss, FocalLoss, TverskyLoss,
    dice_coef, iou_score, pixel_accuracy, sensitivity_specificity,
    MetricsTracker, visualize_predictions, create_confusion_matrix_plot
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_detailed_metrics(predictions, targets, threshold=0.5):
    """Calculate comprehensive metrics for a batch"""
    with torch.no_grad():
        dice = dice_coef(predictions, targets, threshold)
        iou = iou_score(predictions, targets, threshold)
        pixel_acc = pixel_accuracy(predictions, targets, threshold)
        sensitivity, specificity, precision = sensitivity_specificity(predictions, targets, threshold)
        
        return {
            'dice': dice,
            'iou': iou,
            'pixel_acc': pixel_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision
        }

def train_epoch(model, loader, criterion, optimizer, device, metrics_tracker=None):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0
    epoch_metrics = {
        'dice': 0, 'iou': 0, 'pixel_acc': 0,
        'sensitivity': 0, 'specificity': 0, 'precision': 0
    }
    
    pbar = tqdm(loader, total=len(loader), desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        images = batch['image'].to(device)
        masks = batch['mask'].float().to(device)
        
        # Ensure masks have correct dimensions [B, 1, H, W]
        if len(masks.shape) == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        batch_metrics = calculate_detailed_metrics(outputs, masks)
        
        for key in epoch_metrics:
            epoch_metrics[key] += batch_metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Dice': f"{batch_metrics['dice']:.4f}",
            'IoU': f"{batch_metrics['iou']:.4f}"
        })
    
    # Calculate average metrics
    num_batches = len(loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    epoch_loss = 0
    epoch_metrics = {
        'dice': 0, 'iou': 0, 'pixel_acc': 0,
        'sensitivity': 0, 'specificity': 0, 'precision': 0
    }
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    pbar = tqdm(loader, total=len(loader), desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].float().to(device)
            
            # Ensure masks have correct dimensions [B, 1, H, W]
            if len(masks.shape) == 3:  # [B, H, W]
                masks = masks.unsqueeze(1)  # [B, 1, H, W]
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            epoch_loss += loss.item()
            batch_metrics = calculate_detailed_metrics(outputs, masks)
            
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            
            # Store for visualization (first batch only)
            if batch_idx == 0:
                all_images = images[:4]  # First 4 images
                all_predictions = outputs[:4]
                all_targets = masks[:4]
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{batch_metrics['dice']:.4f}",
                'IoU': f"{batch_metrics['iou']:.4f}"
            })
    
    # Calculate average metrics
    num_batches = len(loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics, all_images, all_predictions, all_targets

def save_model_checkpoint(model, optimizer, epoch, metrics, model_dir, filename, encoder_name='efficientnet-b0'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_config': {
            'encoder_name': encoder_name,
            'classes': model.segmentation_head[0].out_channels,
            'in_channels': 3
        }
    }
    
    torch.save(checkpoint, os.path.join(model_dir, filename))

def train_model(args):
    """Main training function"""
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Initialize enhanced loaders with 1:3 augmentation
    print("Loading datasets with 1:3 augmentation...")
    loaders = get_1_to_3_augmentation_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_weighted_sampling=True,
        cache_images=False,
        validate_data=True,
        include_negative_samples=True,
        handle_missing_annotations='warn',
        deterministic_augmentation=True
    )
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print(f"Initializing UNet++ with {args.encoder} encoder...")
    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize loss function and optimizer
    if args.loss_function == 'combined':
        criterion = CombinedLoss(
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
            tversky_weight=args.tversky_weight
        )
    elif args.loss_function == 'dice':
        criterion = DiceLoss()
    elif args.loss_function == 'focal':
        criterion = FocalLoss()
    elif args.loss_function == 'tversky':
        criterion = TverskyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=args.scheduler_patience
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    else:
        scheduler = None
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    best_valid_dice = 0
    best_epoch = 0
    early_stopping_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_metrics = train_epoch(
            model, loaders['train'], criterion, optimizer, device, metrics_tracker
        )
        
        # Validation phase
        val_loss, val_metrics, val_images, val_predictions, val_targets = validate_epoch(
            model, loaders['valid'], criterion, device
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if args.scheduler == 'plateau':
                old_lr = current_lr
                scheduler.step(val_metrics['dice'])
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics tracker
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_metrics['dice'],
            'val_dice': val_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'train_pixel_acc': train_metrics['pixel_acc'],
            'val_pixel_acc': val_metrics['pixel_acc'],
            'train_sensitivity': train_metrics['sensitivity'],
            'val_sensitivity': val_metrics['sensitivity'],
            'train_specificity': train_metrics['specificity'],
            'val_specificity': val_metrics['specificity'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'learning_rates': current_lr
        }
        
        metrics_tracker.update(epoch + 1, epoch_metrics)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"Valid - Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['dice'] > best_valid_dice:
            best_valid_dice = val_metrics['dice']
            best_epoch = epoch + 1
            
            # Save best model
            save_model_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                args.model_dir, 'best_model.pth', args.encoder
            )
            
            print(f"✓ Saved best model with Dice: {best_valid_dice:.4f}")
            early_stopping_counter = 0
            
            # Save best predictions visualization
            if val_images is not None:
                visualize_predictions(
                    val_images, val_targets, val_predictions,
                    save_path=os.path.join(args.results_dir, 'best_predictions.png')
                )
        else:
            early_stopping_counter += 1
        
        # Visualize predictions every 10 epochs
        if (epoch + 1) % 10 == 0 and val_images is not None:
            visualize_predictions(
                val_images, val_targets, val_predictions,
                save_path=os.path.join(args.results_dir, f'predictions_epoch_{epoch+1}.png')
            )
        
        # Early stopping
        if early_stopping_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} - no improvement for {args.patience} epochs")
            break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_model_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                args.model_dir, f'checkpoint_epoch_{epoch+1}.pth', args.encoder
            )
    
    # Training completed
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time/3600:.2f} hours")
    print(f"Best Validation Dice: {best_valid_dice:.4f} at epoch {best_epoch}")
    
    # Save final model
    save_model_checkpoint(
        model, optimizer, epoch + 1, val_metrics,
        args.model_dir, 'final_model.pth', args.encoder
    )
    
    # Plot and save final metrics
    metrics_tracker.plot_metrics(
        save_path=os.path.join(args.results_dir, 'training_metrics.png')
    )
    
    # Save metrics summary table
    summary_table = metrics_tracker.get_summary_table()
    if summary_table is not None:
        summary_table.to_csv(os.path.join(args.results_dir, 'metrics_summary.csv'), index=False)
        print("\nMetrics Summary:")
        print(summary_table.to_string(index=False))
    
    # Save training configuration
    config = {
        'model': {
            'architecture': 'UNet++',
            'encoder': args.encoder,
            'image_size': args.image_size,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'loss_function': args.loss_function,
            'scheduler': args.scheduler,
            'best_epoch': best_epoch,
            'best_dice': float(best_valid_dice),
            'training_time_hours': total_training_time / 3600
        },
        'data': {
            'dataset_path': args.data_dir,
            'train_samples': len(loaders['train'].dataset),
            'valid_samples': len(loaders['valid'].dataset),
            'test_samples': len(loaders['test'].dataset)
        }
    }
    
    with open(os.path.join(args.results_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Models saved to: {args.model_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a UNet++ model for psoriasis segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../Dataset_no_preprocessing', help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory to save logs')

    # Model parameters
    parser.add_argument('--encoder', type=str, default='efficientnet-b3',
                       choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3',
                               'efficientnet-b5', 'densenet121', 'densenet201'],
                       help='Encoder architecture')
    parser.add_argument('--image_size', type=int, default=640, help='Input image size')
    
    # Loss function parameters
    parser.add_argument('--loss_function', type=str, default='combined',
                       choices=['combined', 'dice', 'focal', 'tversky', 'bce'],
                       help='Loss function to use')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='Weight for dice loss in combined loss')
    parser.add_argument('--focal_weight', type=float, default=0.4, help='Weight for focal loss in combined loss')
    parser.add_argument('--tversky_weight', type=float, default=0.3, help='Weight for tversky loss in combined loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (reduced for 4x larger dataset)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (reduced due to 4x more data per epoch)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=7, help='Patience for plateau scheduler')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_model(args) 