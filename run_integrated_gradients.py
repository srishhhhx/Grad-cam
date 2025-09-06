#!/usr/bin/env python3
"""
Script to run Integrated Gradients analysis on the best U-Net++ EfficientNet-B5 model.

This script provides an easy way to run Integrated Gradients analysis on your trained model
with predefined settings for the best model found in your experiments.
"""

import os
import sys
import torch
from pathlib import Path

# Fix OpenMP library conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrated_gradients import (
    run_integrated_gradients_analysis,
    batch_integrated_gradients_analysis
)


def main():
    """Main function to run Integrated Gradients analysis."""
    
    # Configuration for the best model
    BEST_MODEL_PATH = "models_20250609_105424/best_model.pth"
    DATA_DIR = "Dataset_no_preprocessing"
    OUTPUT_DIR = "integrated_gradients_results"
    
    # Model configuration (from the best model's config)
    ENCODER_NAME = "efficientnet-b5"
    IMAGE_SIZE = 640
    
    # Integrated Gradients configuration
    BASELINE_TYPE = "zero"  # Options: 'zero', 'random', 'blur', 'mean'
    NUM_STEPS = 50
    NUM_SAMPLES = 10  # For batch analysis
    
    print("🔬 Integrated Gradients Analysis for U-Net++ EfficientNet-B5")
    print("=" * 60)
    print(f"📁 Model: {BEST_MODEL_PATH}")
    print(f"📊 Model Performance: Dice Score = 0.8945")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"❌ Model file not found: {BEST_MODEL_PATH}")
        print("Please make sure the model file exists.")
        return
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please make sure the dataset directory exists.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ask user for analysis mode
    print("\nSelect analysis mode:")
    print("1. Single image analysis")
    print("2. Batch analysis (multiple test images)")
    print("3. Both")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    try:
        if choice in ['1', '3']:
            # Single image analysis
            print(f"\n🔍 Running Single Image Analysis...")
            
            # Get list of test images
            test_images_dir = os.path.join(DATA_DIR, "test")
            if os.path.exists(test_images_dir):
                image_files = [f for f in os.listdir(test_images_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_files:
                    # Use the first test image as example
                    sample_image = os.path.join(test_images_dir, image_files[0])
                    print(f"📸 Analyzing: {sample_image}")
                    
                    single_output_dir = os.path.join(OUTPUT_DIR, "single_analysis")
                    
                    results = run_integrated_gradients_analysis(
                        model_path=BEST_MODEL_PATH,
                        image_path=sample_image,
                        output_dir=single_output_dir,
                        encoder_name=ENCODER_NAME,
                        image_size=IMAGE_SIZE,
                        baseline_type=BASELINE_TYPE,
                        num_steps=NUM_STEPS
                    )
                    
                    print(f"\n✅ Single image analysis complete!")
                    print(f"📊 Results:")
                    stats = results['statistics']
                    print(f"   - Attribution Mean: {stats['attribution_mean']:.6f}")
                    print(f"   - Attribution Std: {stats['attribution_std']:.6f}")
                    print(f"   - Prediction Ratio: {stats['prediction_ratio']:.4f}")
                    print(f"   - Attribution Ratio: {stats['attr_ratio']:.4f}")
                    print(f"📁 Output saved to: {single_output_dir}")
                else:
                    print(f"❌ No image files found in {test_images_dir}")
            else:
                print(f"❌ Test images directory not found: {test_images_dir}")
        
        if choice in ['2', '3']:
            # Batch analysis
            print(f"\n🔄 Running Batch Analysis on {NUM_SAMPLES} samples...")
            
            batch_output_dir = os.path.join(OUTPUT_DIR, "batch_analysis")
            
            results = batch_integrated_gradients_analysis(
                model_path=BEST_MODEL_PATH,
                data_dir=DATA_DIR,
                output_dir=batch_output_dir,
                num_samples=NUM_SAMPLES,
                encoder_name=ENCODER_NAME,
                image_size=IMAGE_SIZE,
                baseline_type=BASELINE_TYPE,
                num_steps=NUM_STEPS
            )
            
            print(f"\n✅ Batch analysis complete!")
            print(f"📊 Processed {len(results)} samples")
            print(f"📁 Output saved to: {batch_output_dir}")
            
            # Show summary statistics
            if results:
                all_stats = [r['statistics'] for r in results]
                avg_attr_mean = sum(s['attribution_mean'] for s in all_stats) / len(all_stats)
                avg_pred_ratio = sum(s['prediction_ratio'] for s in all_stats) / len(all_stats)
                avg_attr_ratio = sum(s['attr_ratio'] for s in all_stats) / len(all_stats)
                
                print(f"📈 Average Statistics:")
                print(f"   - Attribution Mean: {avg_attr_mean:.6f}")
                print(f"   - Prediction Ratio: {avg_pred_ratio:.4f}")
                print(f"   - Attribution Ratio: {avg_attr_ratio:.4f}")
        
        print(f"\n🎉 Analysis complete! Check the results in: {OUTPUT_DIR}")
        print("\n📋 What the results show:")
        print("   - Attribution maps highlight which pixels the model focuses on")
        print("   - Higher attribution values indicate more important pixels for the prediction")
        print("   - Compare attribution maps with predictions to understand model behavior")
        print("   - Channel-wise analysis shows how different color channels contribute")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check the error message and try again.")
        raise


if __name__ == "__main__":
    main()
