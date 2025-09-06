#!/usr/bin/env python3
"""
Test script for Grad-CAM implementation.
This script tests the core functionality and verifies everything works correctly.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradcam_unet import (
    UNetGradCAM,
    load_model,
    preprocess_image_for_gradcam
)


def test_gradcam_implementation():
    """Test the Grad-CAM implementation."""
    
    print("🧪 Testing Grad-CAM Implementation")
    print("=" * 50)
    
    # Configuration
    MODEL_PATH = "models_20250609_105424/best_model.pth"
    DATA_DIR = "Dataset_no_preprocessing"
    OUTPUT_DIR = "test_gradcam_results"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load model
        print("📥 Loading model...")
        model = load_model(MODEL_PATH, device, 'efficientnet-b5')
        print("✅ Model loaded successfully!")
        
        # Initialize Grad-CAM
        print("🔬 Initializing Grad-CAM...")
        gradcam = UNetGradCAM(model, device)
        print("✅ Grad-CAM initialized!")
        
        # Print available target layers
        print(f"🎯 Available target layers: {list(gradcam.target_layers.keys())}")
        
        # Find a test image
        test_images_dir = os.path.join(DATA_DIR, "test")
        if not os.path.exists(test_images_dir):
            print(f"❌ Test images directory not found: {test_images_dir}")
            return False
        
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"❌ No image files found in {test_images_dir}")
            return False
        
        # Use first image for testing
        test_image_path = os.path.join(test_images_dir, image_files[0])
        print(f"📸 Testing with image: {image_files[0]}")
        
        # Preprocess image
        print("🔄 Preprocessing image...")
        input_tensor, original_image = preprocess_image_for_gradcam(test_image_path, 640)
        input_tensor = input_tensor.to(device)
        print(f"✅ Image preprocessed: {input_tensor.shape}")
        
        # Get model prediction
        print("🔮 Getting model prediction...")
        with torch.no_grad():
            prediction = model(input_tensor)
        print(f"✅ Prediction obtained: {prediction.shape}")
        
        # Test different target layers
        test_layers = ['encoder_last', 'encoder_mid']
        available_layers = [layer for layer in test_layers if layer in gradcam.target_layers]
        
        if not available_layers:
            available_layers = [list(gradcam.target_layers.keys())[0]]
        
        print(f"🧮 Testing layers: {available_layers}")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        for layer in available_layers:
            print(f"\n🔍 Testing layer: {layer}")
            
            # Test Grad-CAM
            try:
                heatmap = gradcam.compute_gradcam(
                    input_tensor=input_tensor,
                    target_layer_name=layer,
                    method='gradcam'
                )
                print(f"✅ Grad-CAM computed: {heatmap.shape}")
                print(f"   Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
                
                # Test visualization
                fig = gradcam.create_visualization(
                    input_tensor=input_tensor,
                    gradcam_heatmap=heatmap,
                    prediction=prediction,
                    title=f'Test Grad-CAM - {layer}'
                )
                
                # Save visualization
                save_path = os.path.join(OUTPUT_DIR, f'test_gradcam_{layer}.png')
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ Visualization saved: {save_path}")
                
                # Test analysis
                stats = gradcam.analyze_attention_patterns(
                    input_tensor=input_tensor,
                    gradcam_heatmap=heatmap,
                    prediction=prediction
                )
                print(f"✅ Analysis completed:")
                print(f"   Attention Mean: {stats['attention_mean']:.6f}")
                print(f"   Attention Std: {stats['attention_std']:.6f}")
                if 'attention_ratio' in stats:
                    print(f"   Attention Ratio: {stats['attention_ratio']:.4f}")
                
            except Exception as e:
                print(f"❌ Error with layer {layer}: {str(e)}")
                continue
        
        # Test method comparison
        print(f"\n🔍 Testing method comparison...")
        try:
            methods = ['gradcam', 'gradcam++']
            method_results = gradcam.compare_methods(
                input_tensor=input_tensor,
                target_layer_name=available_layers[0],
                methods=methods
            )
            
            print(f"✅ Method comparison completed:")
            for method, heatmap in method_results.items():
                print(f"   {method}: {heatmap.shape}, range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
            
            # Create comparison plot
            comp_fig = gradcam.create_method_comparison_plot(
                input_tensor=input_tensor,
                method_results=method_results,
                prediction=prediction,
                save_path=os.path.join(OUTPUT_DIR, 'test_method_comparison.png')
            )
            plt.close(comp_fig)
            print(f"✅ Method comparison plot saved")
            
        except Exception as e:
            print(f"❌ Error in method comparison: {str(e)}")
        
        # Save test statistics
        stats_file = os.path.join(OUTPUT_DIR, 'test_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Grad-CAM Test Results\n")
            f.write("====================\n\n")
            f.write(f"Image: {test_image_path}\n")
            f.write(f"Model: {MODEL_PATH}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Available Layers: {list(gradcam.target_layers.keys())}\n")
            f.write(f"Tested Layers: {available_layers}\n")
        
        print(f"✅ Statistics saved to {stats_file}")
        
        print("\n🎉 Test completed successfully!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_accessibility():
    """Test which layers are accessible in the model."""
    
    print("\n🔍 Testing Layer Accessibility")
    print("=" * 40)
    
    MODEL_PATH = "models_20250609_105424/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model
        model = load_model(MODEL_PATH, device, 'efficientnet-b5')
        
        # Initialize Grad-CAM
        gradcam = UNetGradCAM(model, device)
        
        print("📋 Model Structure Analysis:")
        print(f"   Available target layers: {len(gradcam.target_layers)}")
        
        for name, layer in gradcam.target_layers.items():
            print(f"   - {name}: {type(layer).__name__}")
        
        # Test model components
        print("\n🔍 Model Components:")
        if hasattr(model, 'encoder'):
            print(f"   ✅ Encoder found: {type(model.encoder).__name__}")
            if hasattr(model.encoder, '_blocks'):
                print(f"      - Blocks: {len(model.encoder._blocks)}")
        
        if hasattr(model, 'decoder'):
            print(f"   ✅ Decoder found: {type(model.decoder).__name__}")
        
        if hasattr(model, 'segmentation_head'):
            print(f"   ✅ Segmentation head found: {type(model.segmentation_head).__name__}")
        
        print("✅ Layer accessibility test completed!")
        
    except Exception as e:
        print(f"❌ Layer accessibility test failed: {str(e)}")


if __name__ == "__main__":
    # Run tests
    success = test_gradcam_implementation()
    
    if success:
        test_layer_accessibility()
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")
