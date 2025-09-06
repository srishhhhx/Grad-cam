#!/usr/bin/env python3
"""
Simple script to run prediction and create overlay visualization
Uses the existing predict.py functionality with EfficientNet-B5
"""

import os
import subprocess
import sys
import glob

def run_prediction_with_overlay():
    """Run prediction on a test image and create overlay visualization"""
    
    # Find the first test image
    test_dir = "Dataset_no_preprocessing/test"
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    if not test_images:
        print("No test images found!")
        return
    
    # Use the first test image
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    # Find the best model
    model_path = "models_20250609_105424/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Create output directory
    output_dir = "prediction_overlay_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run prediction using the existing predict.py script with correct parameters
    # Model was trained with image_size=640 and efficientnet-b5
    cmd = [
        sys.executable,
        "UNET-model/predict.py",
        "--model_path", model_path,
        "--single_image", test_image,
        "--output_dir", output_dir,
        "--encoder", "efficientnet-b5",
        "--image_size", "640",  # Match training configuration!
        "--threshold", "0.5"
    ]
    
    print("Running prediction...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("Prediction completed successfully!")
            print("Output:")
            print(result.stdout)
            
            # List the generated files
            if os.path.exists(output_dir):
                print(f"\nGenerated files in {output_dir}:")
                for file in os.listdir(output_dir):
                    print(f"  - {file}")
            
        else:
            print("Prediction failed!")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("Prediction timed out after 5 minutes")
    except Exception as e:
        print(f"Error running prediction: {e}")

if __name__ == "__main__":
    print("Simple Prediction and Overlay Script")
    print("====================================")
    run_prediction_with_overlay()
