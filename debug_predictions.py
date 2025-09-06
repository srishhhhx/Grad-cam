#!/usr/bin/env python3
"""
Debug script to understand why predictions are poor
Check raw prediction values, preprocessing, and model behavior
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Add UNET-model to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'UNET-model'))

import segmentation_models_pytorch as smp
from utils.dataset import get_transforms

def load_model_debug(model_path, device, encoder_name='efficientnet-b5'):
    """Load model and print detailed info"""
    print(f"🔍 DEBUGGING MODEL LOADING")
    print("=" * 40)
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"✅ Checkpoint loaded from: {model_path}")
    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
    print(f"📈 Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"📊 Metrics: {checkpoint.get('metrics', 'None')}")
    
    # Initialize model
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    print(f"🏗️  Model architecture: UNet++ with {encoder_name}")
    print(f"📐 Input channels: 3, Output classes: 1")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    
    return model

def debug_preprocessing(image_path, image_size=640):
    """Debug the preprocessing pipeline"""
    print(f"\n🔄 DEBUGGING PREPROCESSING")
    print("=" * 35)
    
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📸 Original image shape: {image.shape}")
    print(f"📊 Original pixel range: [{image.min()}, {image.max()}]")
    print(f"📈 Original mean: {image.mean():.2f}")
    
    # Resize
    image_resized = cv2.resize(image, (image_size, image_size))
    print(f"📐 Resized shape: {image_resized.shape}")
    
    # Apply transforms
    transform = get_transforms(image_size=image_size, is_training=False)
    transformed = transform(image=image_resized, mask=np.zeros((image_size, image_size)))
    tensor = transformed['image']
    
    print(f"🔧 Transformed tensor shape: {tensor.shape}")
    print(f"📊 Transformed range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"📈 Transformed mean: {tensor.mean():.3f}")
    print(f"📊 Transformed std: {tensor.std():.3f}")
    
    # Check if normalization looks correct (ImageNet stats)
    expected_mean = torch.tensor([0.485, 0.456, 0.406])
    expected_std = torch.tensor([0.229, 0.224, 0.225])
    
    actual_mean = tensor.mean(dim=[1, 2])
    actual_std = tensor.std(dim=[1, 2])
    
    print(f"🎯 Expected mean: {expected_mean}")
    print(f"📊 Actual mean: {actual_mean}")
    print(f"🎯 Expected std: {expected_std}")
    print(f"📊 Actual std: {actual_std}")
    
    return tensor.unsqueeze(0), image

def debug_model_prediction(model, input_tensor, device):
    """Debug the model prediction process"""
    print(f"\n🔮 DEBUGGING MODEL PREDICTION")
    print("=" * 40)
    
    input_tensor = input_tensor.to(device)
    print(f"📥 Input tensor shape: {input_tensor.shape}")
    print(f"📊 Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    with torch.no_grad():
        # Raw model output (before sigmoid)
        raw_output = model(input_tensor)
        print(f"📤 Raw output shape: {raw_output.shape}")
        print(f"📊 Raw output range: [{raw_output.min():.3f}, {raw_output.max():.3f}]")
        print(f"📈 Raw output mean: {raw_output.mean():.3f}")
        print(f"📊 Raw output std: {raw_output.std():.3f}")
        
        # After sigmoid
        sigmoid_output = torch.sigmoid(raw_output)
        print(f"🎯 Sigmoid output range: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")
        print(f"📈 Sigmoid output mean: {sigmoid_output.mean():.3f}")
        
        # Check how many pixels are above different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            count = (sigmoid_output > thresh).sum().item()
            percentage = (count / sigmoid_output.numel()) * 100
            print(f"🎯 Pixels > {thresh}: {count} ({percentage:.2f}%)")
    
    return raw_output, sigmoid_output

def create_debug_visualization(original_image, raw_output, sigmoid_output, image_path):
    """Create detailed debug visualization"""
    print(f"\n🎨 CREATING DEBUG VISUALIZATION")
    print("=" * 40)
    
    # Convert to numpy
    raw_np = raw_output.squeeze().cpu().numpy()
    sigmoid_np = sigmoid_output.squeeze().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Debug Analysis: {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
    
    # Row 1: Original and raw outputs
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(raw_np, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[0, 1].set_title(f'Raw Output\nRange: [{raw_np.min():.2f}, {raw_np.max():.2f}]')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 2].imshow(sigmoid_np, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Sigmoid Output\nRange: [{sigmoid_np.min():.3f}, {sigmoid_np.max():.3f}]')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Histogram of sigmoid values
    axes[0, 3].hist(sigmoid_np.flatten(), bins=50, alpha=0.7)
    axes[0, 3].set_title('Sigmoid Value Distribution')
    axes[0, 3].set_xlabel('Sigmoid Value')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
    axes[0, 3].legend()
    
    # Row 2: Different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    for i, thresh in enumerate(thresholds):
        binary_mask = (sigmoid_np > thresh).astype(np.uint8)
        positive_pixels = binary_mask.sum()
        percentage = (positive_pixels / binary_mask.size) * 100
        
        axes[1, i].imshow(binary_mask, cmap='gray')
        axes[1, i].set_title(f'Threshold {thresh}\n{positive_pixels} pixels ({percentage:.2f}%)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    debug_path = "DEBUG_PREDICTION_ANALYSIS.png"
    plt.savefig(debug_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Debug visualization saved: {debug_path}")
    
    return debug_path

def main():
    """Main debug function"""
    print("🐛 PREDICTION DEBUG ANALYSIS")
    print("=" * 50)
    print("This will help identify why predictions are poor")
    print()
    
    # Parameters
    model_path = "models_20250609_105424/best_model.pth"
    encoder = "efficientnet-b5"
    image_size = 640
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test image
    test_dir = "Dataset_no_preprocessing/test"
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_image_path = os.path.join(test_dir, test_images[0])
    
    print(f"🖼️  Test image: {test_images[0]}")
    print(f"🖥️  Device: {device}")
    print()
    
    try:
        # 1. Debug model loading
        model = load_model_debug(model_path, device, encoder)
        
        # 2. Debug preprocessing
        input_tensor, original_image = debug_preprocessing(test_image_path, image_size)
        
        # 3. Debug prediction
        raw_output, sigmoid_output = debug_model_prediction(model, input_tensor, device)
        
        # 4. Create visualization
        debug_path = create_debug_visualization(original_image, raw_output, sigmoid_output, test_image_path)
        
        print(f"\n🎯 SUMMARY:")
        print("=" * 15)
        print(f"✅ Model loaded successfully")
        print(f"✅ Preprocessing completed")
        print(f"✅ Prediction generated")
        print(f"✅ Debug visualization created: {debug_path}")
        
        # Key insights
        sigmoid_np = sigmoid_output.squeeze().cpu().numpy()
        max_confidence = sigmoid_np.max()
        mean_confidence = sigmoid_np.mean()
        pixels_above_half = (sigmoid_np > 0.5).sum()
        
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   📊 Max confidence: {max_confidence:.4f}")
        print(f"   📈 Mean confidence: {mean_confidence:.4f}")
        print(f"   🎯 Pixels > 0.5: {pixels_above_half}")
        
        if max_confidence < 0.1:
            print(f"   ⚠️  VERY LOW CONFIDENCE - Model might not be working properly")
        elif max_confidence < 0.5:
            print(f"   ⚠️  LOW CONFIDENCE - Consider lower threshold")
        elif pixels_above_half == 0:
            print(f"   ⚠️  NO PIXELS ABOVE 0.5 - Try threshold 0.1-0.3")
        else:
            print(f"   ✅ Model seems to be working")
            
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
