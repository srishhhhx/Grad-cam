#!/usr/bin/env python3
"""
Diagnostic script to understand prediction issues
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def check_model_and_config():
    """Check model file and training configuration"""
    print("🔍 DIAGNOSTIC REPORT")
    print("=" * 50)
    
    # Check model file
    model_path = "models_20250609_105424/best_model.pth"
    print(f"\n📁 Model File Check:")
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"   ✅ Model exists: {model_path}")
        print(f"   📊 Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    else:
        print(f"   ❌ Model NOT found: {model_path}")
        return False
    
    # Check training config
    config_path = "results_20250609_105424/training_config.json"
    print(f"\n📋 Training Configuration:")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"   ✅ Config exists: {config_path}")
        print(f"   🏗️  Architecture: {config['model']['architecture']}")
        print(f"   🧠 Encoder: {config['model']['encoder']}")
        print(f"   📐 Image Size: {config['model']['image_size']}")
        print(f"   🎯 Best Dice: {config['training']['best_dice']:.4f}")
        print(f"   📈 Best Epoch: {config['training']['best_epoch']}")
    else:
        print(f"   ❌ Config NOT found: {config_path}")
        return False
    
    # Check test images
    test_dir = "Dataset_no_preprocessing/test"
    print(f"\n🖼️  Test Images:")
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        print(f"   ✅ Test directory exists: {test_dir}")
        print(f"   📊 Number of test images: {len(test_images)}")
        if test_images:
            first_image = os.path.join(test_dir, test_images[0])
            print(f"   🖼️  First image: {test_images[0]}")
            
            # Check image properties
            img = cv2.imread(first_image)
            if img is not None:
                print(f"   📐 Image shape: {img.shape}")
                print(f"   🎨 Image dtype: {img.dtype}")
                print(f"   📊 Pixel range: [{img.min()}, {img.max()}]")
            else:
                print(f"   ❌ Could not load image: {first_image}")
    else:
        print(f"   ❌ Test directory NOT found: {test_dir}")
        return False
    
    return True

def check_prediction_results():
    """Check the prediction results"""
    print(f"\n🔮 Prediction Results Check:")
    
    results_dir = "corrected_prediction_results"
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"   ✅ Results directory exists: {results_dir}")
        print(f"   📁 Files generated: {len(files)}")
        for file in files:
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"      📄 {file} ({size:,} bytes)")
    else:
        print(f"   ❌ Results directory NOT found: {results_dir}")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print(f"\n📦 Dependencies Check:")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('segmentation_models_pytorch', 'SMP'),
        ('cv2', 'OpenCV'),
        ('albumentations', 'Albumentations'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name} available")
        except ImportError:
            print(f"   ❌ {name} NOT available")
            all_good = False
    
    return all_good

def analyze_training_results():
    """Analyze the training results to understand expected performance"""
    print(f"\n📈 Training Results Analysis:")
    
    # Check metrics summary
    metrics_path = "results_20250609_105424/metrics_summary.csv"
    if os.path.exists(metrics_path):
        print(f"   ✅ Metrics file exists: {metrics_path}")
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
            print(f"   📊 Metrics Summary:")
            for line in lines:
                if line.strip():
                    print(f"      {line.strip()}")
    else:
        print(f"   ❌ Metrics file NOT found: {metrics_path}")
    
    # Check if training predictions exist
    training_pred_path = "results_20250609_105424/best_predictions.png"
    if os.path.exists(training_pred_path):
        print(f"   ✅ Training predictions exist: {training_pred_path}")
        size = os.path.getsize(training_pred_path)
        print(f"   📊 Size: {size:,} bytes")
    else:
        print(f"   ❌ Training predictions NOT found: {training_pred_path}")

def suggest_solutions():
    """Suggest potential solutions"""
    print(f"\n💡 POTENTIAL SOLUTIONS:")
    print("=" * 30)
    
    solutions = [
        "1. 🔧 Verify model architecture matches exactly (UNet++ with efficientnet-b5)",
        "2. 📐 Ensure image size is exactly 640x640 (not 960x960)",
        "3. 🎨 Check image preprocessing (normalization, pixel range)",
        "4. 🧠 Verify model weights loaded correctly",
        "5. 📊 Check if model is in evaluation mode",
        "6. 🔍 Compare with training validation images",
        "7. 🎯 Try different threshold values (0.3, 0.5, 0.7)",
        "8. 📱 Test on multiple images from test set",
        "9. 🔄 Re-run prediction with verbose output",
        "10. 📋 Check if using same transforms as training"
    ]
    
    for solution in solutions:
        print(f"   {solution}")

def main():
    """Main diagnostic function"""
    print("🩺 U-Net Prediction Diagnostic Tool")
    print("This tool will help identify why predictions aren't working")
    print()
    
    # Run all checks
    model_ok = check_model_and_config()
    deps_ok = check_dependencies()
    results_ok = check_prediction_results()
    
    # Analyze training results
    analyze_training_results()
    
    # Overall status
    print(f"\n🎯 OVERALL STATUS:")
    print("=" * 20)
    print(f"   Model & Config: {'✅' if model_ok else '❌'}")
    print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   Prediction Results: {'✅' if results_ok else '❌'}")
    
    if model_ok and deps_ok and results_ok:
        print(f"\n🎉 All checks passed! The issue might be with:")
        print(f"   - Prediction quality/accuracy")
        print(f"   - Visualization/display")
        print(f"   - Threshold settings")
        print(f"   - Expected vs actual results")
    else:
        print(f"\n⚠️  Some issues found. Check the details above.")
    
    # Suggest solutions
    suggest_solutions()

if __name__ == "__main__":
    main()
