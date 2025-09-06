#!/usr/bin/env python3
"""
Interactive slideshow viewer for test predictions with navigation controls
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
from enhanced_dataloader import get_1_to_3_augmentation_loaders
import random
from pathlib import Path

class PredictionSlideshow:
    def __init__(self, model, test_loader, device, threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        self.current_index = 0
        self.samples = []
        self._colorbar = None  # Initialize colorbar reference
        
        # Load all samples
        self._load_samples()
        
        # Setup matplotlib figure
        self.fig, self.axes = plt.subplots(1, 5, figsize=(25, 5))
        self.fig.suptitle('🔬 Interactive Test Predictions Viewer', fontsize=16, fontweight='bold')
        
        # Add navigation buttons
        self._setup_buttons()
        
        # Display first sample
        self._update_display()
        
    def _load_samples(self):
        """Load all test samples with predictions"""
        print("🔄 Loading test samples and generating predictions...")
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                filenames = batch['filename']
                
                # Ensure masks have correct dimensions
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                
                # Get predictions
                predictions = self.model(images)
                predictions = torch.sigmoid(predictions)
                
                # Store samples
                for i in range(images.shape[0]):
                    self.samples.append({
                        'image': images[i].cpu(),
                        'mask': masks[i].cpu(),
                        'prediction': predictions[i].cpu(),
                        'filename': filenames[i]
                    })
        
        print(f"✅ Loaded {len(self.samples)} test samples")
    
    def _setup_buttons(self):
        """Setup navigation buttons"""
        # Create button axes (adjusted for wider layout)
        ax_prev = plt.axes([0.08, 0.02, 0.08, 0.05])
        ax_next = plt.axes([0.17, 0.02, 0.08, 0.05])
        ax_info = plt.axes([0.28, 0.02, 0.12, 0.05])
        ax_save = plt.axes([0.42, 0.02, 0.08, 0.05])
        ax_jump = plt.axes([0.52, 0.02, 0.08, 0.05])
        
        # Create buttons
        self.btn_prev = Button(ax_prev, '◀ Previous')
        self.btn_next = Button(ax_next, 'Next ▶')
        self.btn_info = Button(ax_info, '📊 Show Stats')
        self.btn_save = Button(ax_save, '💾 Save')
        self.btn_jump = Button(ax_jump, '🎯 Jump to...')
        
        # Connect button events
        self.btn_prev.on_clicked(self._prev_sample)
        self.btn_next.on_clicked(self._next_sample)
        self.btn_info.on_clicked(self._show_stats)
        self.btn_save.on_clicked(self._save_current)
        self.btn_jump.on_clicked(self._jump_to_sample)
    
    def _dice_coefficient(self, pred, target):
        """Calculate Dice coefficient"""
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return dice.item()
    
    def _iou_score(self, pred, target):
        """Calculate IoU score"""
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return iou.item()
    
    def _update_display(self):
        """Update the display with current sample"""
        if not self.samples:
            return

        sample = self.samples[self.current_index]

        # Clear axes and any existing colorbars
        for ax in self.axes:
            ax.clear()

        # Clear any existing colorbars and their axes safely
        if hasattr(self, '_colorbar') and self._colorbar is not None:
            try:
                # Remove the colorbar and its dedicated axes
                if hasattr(self._colorbar, 'ax') and self._colorbar.ax is not None:
                    self._colorbar.ax.remove()
                self._colorbar.remove()
            except (KeyError, ValueError, AttributeError):
                # Colorbar already removed or doesn't exist
                pass
            self._colorbar = None
        
        # Denormalize image for display
        img = sample['image'].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = sample['mask'][0].numpy()
        pred = sample['prediction'][0].numpy()
        pred_binary = (pred > self.threshold).astype(np.float32)
        
        # Calculate metrics
        dice = self._dice_coefficient(sample['prediction'], sample['mask'])
        iou = self._iou_score(sample['prediction'], sample['mask'])
        
        # Plot original image
        self.axes[0].imshow(img)
        self.axes[0].set_title(f'Original Image\n{sample["filename"][:25]}...', fontsize=12)
        self.axes[0].axis('off')
        
        # Plot ground truth mask
        self.axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        self.axes[1].set_title('Ground Truth Mask', fontsize=12)
        self.axes[1].axis('off')
        
        # Plot prediction (probability)
        im = self.axes[2].imshow(pred, cmap='hot', vmin=0, vmax=1)
        self.axes[2].set_title('Prediction\n(Probability)', fontsize=12)
        self.axes[2].axis('off')
        
        # Add colorbar for probability map with fixed positioning
        # Create a dedicated axes for the colorbar to prevent resizing issues
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self._colorbar = plt.colorbar(im, cax=cax)
        self._colorbar.set_label('Probability', rotation=270, labelpad=15)
        
        # Plot binary prediction with metrics
        self.axes[3].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        self.axes[3].set_title(f'Binary Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}', fontsize=12)
        self.axes[3].axis('off')

        # Plot prediction overlay on original image
        overlay_img = img.copy()

        # Create colored mask overlay (red for prediction, green for ground truth)
        pred_colored = np.zeros_like(overlay_img)
        mask_colored = np.zeros_like(overlay_img)

        # Red channel for prediction (semi-transparent)
        pred_colored[:, :, 0] = pred_binary
        # Green channel for ground truth (semi-transparent)
        mask_colored[:, :, 1] = mask

        # Combine overlays: Yellow where both overlap, Red for prediction only, Green for GT only
        combined_overlay = np.maximum(pred_colored, mask_colored)

        # Blend with original image
        overlay_result = cv2.addWeighted(overlay_img, 0.7, combined_overlay, 0.5, 0)

        self.axes[4].imshow(overlay_result)
        self.axes[4].set_title('Prediction Overlay\n🔴 Prediction | 🟢 Ground Truth | 🟡 Overlap', fontsize=12)
        self.axes[4].axis('off')

        # Update main title with navigation info
        self.fig.suptitle(
            f'🔬 Test Predictions Viewer - Sample {self.current_index + 1}/{len(self.samples)} | '
            f'Dice: {dice:.3f} | IoU: {iou:.3f}',
            fontsize=16, fontweight='bold'
        )
        
        # Refresh display
        self.fig.canvas.draw()
    
    def _prev_sample(self, event):
        """Navigate to previous sample"""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()
        else:
            print("📍 Already at the first sample")
    
    def _next_sample(self, event):
        """Navigate to next sample"""
        if self.current_index < len(self.samples) - 1:
            self.current_index += 1
            self._update_display()
        else:
            print("📍 Already at the last sample")
    
    def _show_stats(self, event):
        """Show overall statistics"""
        dice_scores = []
        iou_scores = []
        
        for sample in self.samples:
            dice = self._dice_coefficient(sample['prediction'], sample['mask'])
            iou = self._iou_score(sample['prediction'], sample['mask'])
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        print("\n" + "="*60)
        print("📊 OVERALL TEST SET STATISTICS")
        print("="*60)
        print(f"Total samples: {len(self.samples)}")
        print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"Average IoU Score:  {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
        print(f"Best Dice Score:    {np.max(dice_scores):.4f}")
        print(f"Worst Dice Score:   {np.min(dice_scores):.4f}")
        print(f"Median Dice Score:  {np.median(dice_scores):.4f}")
        print(f"Samples with Dice > 0.8: {sum(1 for d in dice_scores if d > 0.8)}/{len(dice_scores)}")
        print(f"Samples with Dice > 0.5: {sum(1 for d in dice_scores if d > 0.5)}/{len(dice_scores)}")
        print("="*60)
    
    def _save_current(self, event):
        """Save current sample"""
        sample = self.samples[self.current_index]
        filename = f"sample_{self.current_index + 1}_{sample['filename'].replace('.jpg', '')}.png"
        
        # Create output directory
        os.makedirs("saved_predictions", exist_ok=True)
        save_path = os.path.join("saved_predictions", filename)
        
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved current sample to: {save_path}")
    
    def _jump_to_sample(self, event):
        """Jump to specific sample number"""
        try:
            sample_num = int(input(f"Enter sample number (1-{len(self.samples)}): "))
            if 1 <= sample_num <= len(self.samples):
                self.current_index = sample_num - 1
                self._update_display()
                print(f"🎯 Jumped to sample {sample_num}")
            else:
                print(f"❌ Invalid sample number. Please enter 1-{len(self.samples)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def show(self):
        """Show the interactive viewer"""
        print("\n🎮 INTERACTIVE CONTROLS:")
        print("  ◀ Previous - Go to previous sample")
        print("  Next ▶ - Go to next sample") 
        print("  📊 Show Stats - Display overall statistics")
        print("  💾 Save - Save current sample as image")
        print("  🎯 Jump to... - Jump to specific sample number")
        print("  Close window to exit")
        print("\n🔍 Use the buttons below the image to navigate!")
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        plt.show()

def load_model(model_path, device, encoder_name='efficientnet-b5'):
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

def main():
    # Configuration
    MODEL_PATH = "models_20250609_105424/best_model.pth"  # Latest best model
    DATA_DIR = "Dataset_no_preprocessing"
    THRESHOLD = 0.5
    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
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
    
    # Create and show interactive viewer
    print("🎬 Creating interactive slideshow...")
    viewer = PredictionSlideshow(model, test_loader, device, threshold=THRESHOLD)
    viewer.show()

if __name__ == "__main__":
    main()
