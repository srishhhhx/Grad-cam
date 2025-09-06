#!/usr/bin/env python3
"""
Simple script to visualize test predictions on 10 images using the best trained model
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from enhanced_dataloader import get_1_to_3_augmentation_loaders
import random
from pathlib import Path

def load_model(model_path, device, encoder_name='efficientnet-b0'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with same architecture
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # No pretrained weights when loading from checkpoint
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"📊 Model validation metrics: {checkpoint['metrics']}")
    
    return model, checkpoint

def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice.item()

def iou_score(pred, target, threshold=0.5):
    """Calculate IoU score"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()

def visualize_predictions(model, test_loader, device, num_images=10, threshold=0.5, save_path=None):
    """Visualize predictions on test images"""
    model.eval()
    
    # Collect samples
    samples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(samples) >= num_images:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            
            # Ensure masks have correct dimensions
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Get predictions
            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            
            # Store samples
            for i in range(images.shape[0]):
                if len(samples) >= num_images:
                    break
                    
                samples.append({
                    'image': images[i].cpu(),
                    'mask': masks[i].cpu(),
                    'prediction': predictions[i].cpu(),
                    'filename': filenames[i]
                })
    
    # Create visualization
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('🔬 Test Set Predictions - U-Net++ Model', fontsize=16, fontweight='bold')
    
    for i, sample in enumerate(samples):
        # Denormalize image for display
        img = sample['image'].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = sample['mask'][0].numpy()
        pred = sample['prediction'][0].numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # Calculate metrics
        dice = dice_coefficient(sample['prediction'], sample['mask'], threshold)
        iou = iou_score(sample['prediction'], sample['mask'], threshold)
        
        # Plot original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image\n{sample["filename"][:20]}...', fontsize=10)
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        # Plot prediction (probability)
        im = axes[i, 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction\n(Probability)', fontsize=10)
        axes[i, 2].axis('off')
        
        # Plot binary prediction with metrics
        axes[i, 3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f'Binary Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}', fontsize=10)
        axes[i, 3].axis('off')
        
        # Add colorbar for probability map (only for first row)
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Probability', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    
    plt.show()
    
    return samples

def main():
    # Configuration
    MODEL_PATH = "models_20250609_105424/best_model.pth"  # Latest best model
    DATA_DIR = "Dataset_no_preprocessing"
    OUTPUT_DIR = "test_predictions_visualization"
    NUM_IMAGES = 10
    THRESHOLD = 0.5
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Available models:")
        for model_dir in sorted(Path(".").glob("models_*")):
            if model_dir.is_dir():
                print(f"  📁 {model_dir}")
                for model_file in model_dir.glob("*.pth"):
                    print(f"    📄 {model_file}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(MODEL_PATH, device, encoder_name='efficientnet-b5')
    
    # Load test data
    print("📂 Loading test dataset...")
    try:
        loaders = get_1_to_3_augmentation_loaders(
            root_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=0,
            image_size=IMAGE_SIZE,
            use_weighted_sampling=False,
            cache_images=False,
            validate_data=False,
            include_negative_samples=True,
            deterministic_augmentation=True
        )
        test_loader = loaders['test']
        print(f"✅ Test dataset loaded: {len(test_loader.dataset)} images")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Visualize predictions
    print(f"🔍 Generating predictions for {NUM_IMAGES} test images...")
    save_path = os.path.join(OUTPUT_DIR, f"test_predictions_{NUM_IMAGES}_images.png")
    
    samples = visualize_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        num_images=NUM_IMAGES,
        threshold=THRESHOLD,
        save_path=save_path
    )
    
    # Print summary statistics
    print("\n📊 Prediction Summary:")
    print("=" * 50)
    
    dice_scores = []
    iou_scores = []
    
    for sample in samples:
        dice = dice_coefficient(sample['prediction'], sample['mask'], THRESHOLD)
        iou = iou_score(sample['prediction'], sample['mask'], THRESHOLD)
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Average IoU Score:  {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Best Dice Score:    {np.max(dice_scores):.4f}")
    print(f"Worst Dice Score:   {np.min(dice_scores):.4f}")
    
    print(f"\n✅ Visualization complete! Check {OUTPUT_DIR}/ for results.")

if __name__ == "__main__":
    main()
