#!/usr/bin/env python3
"""
Simple script to run Grad-CAM analysis on the best U-Net++ EfficientNet-B5 model.

This script provides an easy way to run Grad-CAM analysis on your trained model
with predefined settings for the best model found in your experiments.
"""

import os
import sys
import torch

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradcam_unet import (
    run_gradcam_analysis,
    batch_gradcam_analysis
)


def main():
    """Main function to run Grad-CAM analysis."""
    
    # Configuration for the best model
    BEST_MODEL_PATH = "models_20250609_105424/best_model.pth"
    DATA_DIR = "Dataset_no_preprocessing"
    OUTPUT_DIR = "gradcam_results"
    
    # Model configuration (from the best model's config)
    ENCODER_NAME = "efficientnet-b5"
    IMAGE_SIZE = 640
    
    # Grad-CAM configuration
    TARGET_LAYER = "encoder_last"  # Options: encoder_last, encoder_mid, encoder_early, decoder_last, seg_head
    METHODS = ["gradcam", "gradcam++"]  # Options: gradcam, gradcam++, scorecam, layercam
    NUM_SAMPLES = 10  # For batch analysis
    
    print("🔬 Grad-CAM Analysis for U-Net++ EfficientNet-B5")
    print("=" * 60)
    print(f"📁 Model: {BEST_MODEL_PATH}")
    print(f"📊 Model Performance: Dice Score = 0.8945")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🎯 Target Layer: {TARGET_LAYER}")
    print(f"🔍 Methods: {', '.join(METHODS)}")
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
                    
                    results = run_gradcam_analysis(
                        model_path=BEST_MODEL_PATH,
                        image_path=sample_image,
                        output_dir=single_output_dir,
                        encoder_name=ENCODER_NAME,
                        image_size=IMAGE_SIZE,
                        target_layer=TARGET_LAYER,
                        methods=METHODS
                    )
                    
                    print(f"\n✅ Single image analysis complete!")
                    print(f"📊 Results:")
                    for method, result in results['results'].items():
                        stats = result['statistics']
                        print(f"   {method.upper()}:")
                        print(f"     - Attention Mean: {stats['attention_mean']:.6f}")
                        print(f"     - Attention Std: {stats['attention_std']:.6f}")
                        if 'attention_ratio' in stats:
                            print(f"     - Attention Ratio: {stats['attention_ratio']:.4f}")
                            print(f"     - Prediction Ratio: {stats['prediction_ratio']:.4f}")
                    
                    print(f"📁 Output saved to: {single_output_dir}")
                    print(f"🎯 Available target layers: {results['target_layers']}")
                else:
                    print(f"❌ No image files found in {test_images_dir}")
            else:
                print(f"❌ Test images directory not found: {test_images_dir}")
        
        if choice in ['2', '3']:
            # Batch analysis
            print(f"\n🔄 Running Batch Analysis on {NUM_SAMPLES} samples...")
            
            batch_output_dir = os.path.join(OUTPUT_DIR, "batch_analysis")
            
            results = batch_gradcam_analysis(
                model_path=BEST_MODEL_PATH,
                data_dir=DATA_DIR,
                output_dir=batch_output_dir,
                num_samples=NUM_SAMPLES,
                encoder_name=ENCODER_NAME,
                image_size=IMAGE_SIZE,
                target_layer=TARGET_LAYER,
                methods=METHODS
            )
            
            print(f"\n✅ Batch analysis complete!")
            print(f"📊 Processed {len(results)} samples")
            print(f"📁 Output saved to: {batch_output_dir}")
            
            # Show summary statistics
            if results:
                print(f"📈 Summary Statistics:")
                for method in METHODS:
                    method_stats = []
                    for result in results:
                        if method in result['results']:
                            method_stats.append(result['results'][method]['statistics'])
                    
                    if method_stats:
                        avg_attention_mean = sum(s['attention_mean'] for s in method_stats) / len(method_stats)
                        avg_attention_ratio = sum(s.get('attention_ratio', 0) for s in method_stats) / len(method_stats)
                        avg_pred_ratio = sum(s.get('prediction_ratio', 0) for s in method_stats) / len(method_stats)
                        
                        print(f"   {method.upper()}:")
                        print(f"     - Average Attention Mean: {avg_attention_mean:.6f}")
                        print(f"     - Average Attention Ratio: {avg_attention_ratio:.4f}")
                        print(f"     - Average Prediction Ratio: {avg_pred_ratio:.4f}")
        
        print(f"\n🎉 Analysis complete! Check the results in: {OUTPUT_DIR}")
        print("\n📋 What the results show:")
        print("   - Grad-CAM heatmaps highlight which features the model focuses on")
        print("   - Higher attention values indicate more important features for prediction")
        print("   - Compare different methods (Grad-CAM vs Grad-CAM++) for robust interpretation")
        print("   - Attention ratio shows how much more the model focuses on predicted vs background regions")
        print("   - Different target layers reveal different levels of feature importance")
        
        print("\n🎯 Target Layer Options:")
        print("   - encoder_last: Final encoder features (most detailed)")
        print("   - encoder_mid: Mid-level encoder features")
        print("   - encoder_early: Early encoder features (more general)")
        print("   - decoder_last: Final decoder features")
        print("   - seg_head: Segmentation head (final predictions)")
        
        print("\n🔍 Method Comparison:")
        print("   - Grad-CAM: Standard gradient-based attention")
        print("   - Grad-CAM++: Improved version with better localization")
        print("   - Score-CAM: Gradient-free method using forward passes")
        print("   - Layer-CAM: Layer-wise attention visualization")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check the error message and try again.")
        raise


if __name__ == "__main__":
    main()
