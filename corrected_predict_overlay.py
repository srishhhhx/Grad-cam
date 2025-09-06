#!/usr/bin/env python3
"""
Corrected prediction script that exactly matches the training configuration
Model: efficientnet-b5, image_size=640, proper preprocessing
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# Add the UNET-model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'UNET-model'))

def predict_with_exact_config():
    """Run prediction with exact training configuration"""
    
    # Exact parameters from training config
    model_path = "models_20250609_105424/best_model.pth"
    encoder = "efficientnet-b5"
    image_size = 640  # Exact match from training
    threshold = 0.5
    
    # Find first test image
    test_dir = "Dataset_no_preprocessing/test"
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    if not test_images:
        print("No test images found!")
        return
    
    test_image = os.path.join(test_dir, test_images[0])
    output_dir = "corrected_prediction_results"
    
    print(f"Using exact training configuration:")
    print(f"  Model: {model_path}")
    print(f"  Encoder: {encoder}")
    print(f"  Image size: {image_size}")
    print(f"  Test image: {test_image}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run prediction with exact parameters
    cmd = [
        "python", "UNET-model/predict.py",
        "--model_path", model_path,
        "--single_image", test_image,
        "--output_dir", output_dir,
        "--encoder", encoder,
        "--image_size", str(image_size),
        "--threshold", str(threshold)
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()
    
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Prediction completed successfully!")
            print("\nOutput:")
            print(result.stdout)
            
            # List generated files
            if os.path.exists(output_dir):
                print(f"\n📁 Generated files in {output_dir}:")
                for file in sorted(os.listdir(output_dir)):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f"  📄 {file} ({size:,} bytes)")
                
                # Try to display the result if it exists
                result_image = os.path.join(output_dir, "single_prediction.png")
                if os.path.exists(result_image):
                    print(f"\n🖼️  Main result saved as: {result_image}")
                    
                    # Also create a simple overlay visualization
                    create_simple_overlay(test_image, output_dir)
            
        else:
            print("❌ Prediction failed!")
            print("\nError output:")
            print(result.stderr)
            print("\nStandard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Prediction timed out after 5 minutes")
    except Exception as e:
        print(f"💥 Error running prediction: {e}")

def create_simple_overlay(original_image_path, output_dir):
    """Create a simple overlay visualization"""
    try:
        # Load original image
        original = cv2.imread(original_image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        print(f"\n🎨 Creating overlay visualization...")
        print(f"   Original image shape: {original.shape}")
        
        # Create a simple figure showing the original image
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(original)
        ax.set_title(f'Original Test Image\n{os.path.basename(original_image_path)}')
        ax.axis('off')
        
        overlay_path = os.path.join(output_dir, "original_image.png")
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Original image saved as: {overlay_path}")
        
    except Exception as e:
        print(f"   Error creating overlay: {e}")

def main():
    parser = argparse.ArgumentParser(description='Corrected prediction with exact training config')
    parser.add_argument('--show_config', action='store_true', help='Show training configuration')
    
    args = parser.parse_args()
    
    if args.show_config:
        print("Training Configuration (from results_20250609_105424/training_config.json):")
        print("=" * 60)
        config_path = "results_20250609_105424/training_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                print(json.dumps(config, indent=2))
        else:
            print("Config file not found!")
        return
    
    print("🔬 Corrected U-Net Prediction")
    print("=" * 40)
    print("This script uses the EXACT training configuration")
    print("to ensure predictions match the training results.")
    print()
    
    predict_with_exact_config()

if __name__ == "__main__":
    main()
