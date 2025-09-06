#!/usr/bin/env python3
"""
FIXED prediction script with correct preprocessing
The issue was max_pixel_value=1.0 but feeding 0-255 images
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add UNET-model to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'UNET-model'))

import segmentation_models_pytorch as smp

def load_model(model_path, device, encoder_name='efficientnet-b5'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def get_correct_transforms(image_size=640):
    """Get CORRECTED transforms - the key fix!"""
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0  # FIXED: was 1.0, should be 255.0!
        ),
        ToTensorV2()
    ])
    return transform

def predict_with_correct_preprocessing(model, image_path, device, image_size=640, threshold=0.5):
    """Predict with CORRECTED preprocessing"""
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    print(f"📸 Original image shape: {image.shape}")
    print(f"📊 Original pixel range: [{image.min()}, {image.max()}]")
    
    # Apply CORRECTED transforms
    transform = get_correct_transforms(image_size)
    transformed = transform(image=image, mask=np.zeros((image_size, image_size)))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    print(f"🔧 Transformed tensor shape: {input_tensor.shape}")
    print(f"📊 Transformed range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    print(f"📈 Transformed mean: {input_tensor.mean():.3f}")
    
    # Predict
    with torch.no_grad():
        raw_output = model(input_tensor)
        prediction = torch.sigmoid(raw_output)
        
        print(f"📤 Raw output range: [{raw_output.min():.3f}, {raw_output.max():.3f}]")
        print(f"🎯 Sigmoid range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        print(f"📈 Sigmoid mean: {prediction.mean():.3f}")
        
        # Check different thresholds
        for thresh in [0.1, 0.3, 0.5, 0.7]:
            count = (prediction > thresh).sum().item()
            percentage = (count / prediction.numel()) * 100
            print(f"🎯 Pixels > {thresh}: {count} ({percentage:.2f}%)")
    
    # Convert to numpy and resize back to original size
    prediction_np = prediction.cpu().numpy()[0, 0]
    prediction_resized = cv2.resize(prediction_np, (original_shape[1], original_shape[0]))
    
    # Create binary mask
    binary_mask = (prediction_resized > threshold).astype(np.uint8)
    
    return prediction_resized, binary_mask, image

def create_fixed_visualization(original_image, prediction, binary_mask, image_path, threshold):
    """Create visualization with fixed predictions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'FIXED Predictions - {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
    
    # Row 1: Original, prediction, binary
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(prediction, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Prediction Confidence\nMax: {prediction.max():.3f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(binary_mask, cmap='gray')
    axes[0, 2].set_title(f'Binary Mask (>{threshold})\n{binary_mask.sum()} pixels')
    axes[0, 2].axis('off')
    
    # Row 2: Different thresholds
    thresholds = [0.1, 0.3, 0.7]
    for i, thresh in enumerate(thresholds):
        mask = (prediction > thresh).astype(np.uint8)
        pixel_count = mask.sum()
        percentage = (pixel_count / mask.size) * 100
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Threshold {thresh}\n{pixel_count} pixels ({percentage:.2f}%)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    fixed_path = "FIXED_PREDICTIONS.png"
    plt.savefig(fixed_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return fixed_path

def create_overlay_visualization(original_image, binary_mask, alpha=0.3):
    """Create overlay visualization"""
    
    # Create colored overlay
    overlay = original_image.copy()
    overlay[binary_mask > 0] = [255, 0, 0]  # Red for detected regions
    
    # Blend with original
    blended = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Overlay Visualization', fontsize=16, fontweight='bold')
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title('Detected Regions (Red)')
    axes[1].axis('off')
    
    axes[2].imshow(blended)
    axes[2].set_title(f'Blended Overlay (α={alpha})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    overlay_path = "FIXED_OVERLAY.png"
    plt.savefig(overlay_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return overlay_path

def main():
    """Main function with fixed preprocessing"""
    print("🔧 FIXED PREDICTION WITH CORRECT PREPROCESSING")
    print("=" * 60)
    print("Fixed the max_pixel_value issue: 255.0 instead of 1.0")
    print()
    
    # Parameters
    model_path = "models_20250609_105424/best_model.pth"
    encoder = "efficientnet-b5"
    image_size = 640
    threshold = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test image
    test_dir = "Dataset_no_preprocessing/test"
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_image_path = os.path.join(test_dir, test_images[0])
    
    print(f"🖼️  Test image: {test_images[0]}")
    print(f"🖥️  Device: {device}")
    print()
    
    try:
        # Load model
        print("🔄 Loading model...")
        model = load_model(model_path, device, encoder)
        print("✅ Model loaded successfully")
        
        # Predict with FIXED preprocessing
        print("\n🔮 Running prediction with FIXED preprocessing...")
        prediction, binary_mask, original_image = predict_with_correct_preprocessing(
            model, test_image_path, device, image_size, threshold
        )
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        fixed_path = create_fixed_visualization(original_image, prediction, binary_mask, test_image_path, threshold)
        overlay_path = create_overlay_visualization(original_image, binary_mask)
        
        print(f"✅ Fixed predictions saved: {fixed_path}")
        print(f"✅ Overlay visualization saved: {overlay_path}")
        
        # Summary statistics
        total_pixels = binary_mask.size
        detected_pixels = binary_mask.sum()
        coverage_percentage = (detected_pixels / total_pixels) * 100
        max_confidence = prediction.max()
        mean_confidence = prediction.mean()
        
        print(f"\n📊 RESULTS SUMMARY:")
        print("=" * 25)
        print(f"✅ Max confidence: {max_confidence:.3f}")
        print(f"📈 Mean confidence: {mean_confidence:.3f}")
        print(f"🎯 Detected pixels: {detected_pixels:,}")
        print(f"📊 Coverage: {coverage_percentage:.2f}%")
        print(f"🖼️  Image size: {original_image.shape[:2]}")
        
        if max_confidence > 0.8:
            print(f"🎉 EXCELLENT: High confidence predictions!")
        elif max_confidence > 0.5:
            print(f"✅ GOOD: Reasonable confidence predictions")
        else:
            print(f"⚠️  LOW: Consider checking model or data")
            
        if detected_pixels > 0:
            print(f"✅ SUCCESS: Model detected {detected_pixels} lesion pixels")
        else:
            print(f"ℹ️  INFO: No lesions detected above threshold {threshold}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
