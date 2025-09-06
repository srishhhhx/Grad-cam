#!/usr/bin/env python3
"""
Start U-Net Training with Enhanced 1:3 Augmentation DataLoader
This script provides easy configuration and monitoring for training
"""

import os
import sys
import subprocess
import torch
import time
from datetime import datetime

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if dataset exists
    dataset_path = "Dataset_no_preprocessing"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    # Check if enhanced dataloader exists
    if not os.path.exists("enhanced_dataloader.py"):
        print("❌ Enhanced dataloader not found")
        return False
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  No GPU available - training will use CPU (much slower)")
    
    # Check disk space
    import shutil
    free_space = shutil.disk_usage('.').free / 1e9
    print(f"💾 Free disk space: {free_space:.1f} GB")
    
    if free_space < 5:
        print("⚠️  Low disk space - consider freeing up space for model checkpoints")
    
    print("✅ Requirements check completed")
    return True

def get_training_config():
    """Get training configuration from user"""
    print("\n🔧 Training Configuration")
    print("=" * 40)
    
    configs = {
        "quick_test": {
            "description": "Quick test run (2 epochs, small batch)",
            "args": [
                "--epochs", "2",
                "--batch_size", "1",
                "--num_workers", "0",
                "--image_size", "512",
                "--encoder", "efficientnet-b0"
            ]
        },
        "standard": {
            "description": "Standard training (recommended)",
            "args": [
                "--epochs", "30",
                "--batch_size", "2",
                "--num_workers", "2",
                "--image_size", "640",
                "--encoder", "efficientnet-b3"
            ]
        },
        "high_quality": {
            "description": "High quality training (longer, better results)",
            "args": [
                "--epochs", "50",
                "--batch_size", "2",
                "--num_workers", "2",
                "--image_size", "640",
                "--encoder", "efficientnet-b5"
            ]
        },
        "gpu_optimized": {
            "description": "GPU optimized (if you have good GPU)",
            "args": [
                "--epochs", "40",
                "--batch_size", "4",
                "--num_workers", "4",
                "--image_size", "640",
                "--encoder", "efficientnet-b3"
            ]
        }
    }
    
    print("Available configurations:")
    for i, (key, config) in enumerate(configs.items(), 1):
        print(f"  {i}. {key}: {config['description']}")
    
    while True:
        try:
            choice = input("\nSelect configuration (1-4) or 'custom': ").strip()
            
            if choice == 'custom':
                print("\nCustom configuration:")
                epochs = input("Number of epochs (default: 30): ").strip() or "30"
                batch_size = input("Batch size (default: 2): ").strip() or "2"
                image_size = input("Image size (default: 640): ").strip() or "640"
                encoder = input("Encoder (efficientnet-b0/b3/b5, default: b3): ").strip() or "efficientnet-b3"
                
                return [
                    "--epochs", epochs,
                    "--batch_size", batch_size,
                    "--image_size", image_size,
                    "--encoder", encoder,
                    "--num_workers", "2"
                ]
            
            choice_idx = int(choice) - 1
            config_keys = list(configs.keys())
            
            if 0 <= choice_idx < len(config_keys):
                selected_config = configs[config_keys[choice_idx]]
                print(f"\nSelected: {config_keys[choice_idx]} - {selected_config['description']}")
                return selected_config['args']
            else:
                print("Invalid choice. Please select 1-4 or 'custom'")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'custom'")

def start_training(config_args):
    """Start the training process"""
    print("\n🚀 Starting U-Net Training with 1:3 Augmentation")
    print("=" * 60)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare command
    cmd = [
        "python", "UNET-model/train_unetpp.py",
        "--data_dir", "Dataset_no_preprocessing",
        "--model_dir", f"models_{timestamp}",
        "--results_dir", f"results_{timestamp}",
        "--logs_dir", f"logs_{timestamp}"
    ] + config_args
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    # Ask for confirmation
    confirm = input("Start training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print(f"\n🎯 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Dataset: 1:3 augmentation (4x data expansion)")
    print("⚖️  Class balancing: Weighted sampling enabled")
    print("🛡️  Negative samples: Included for better specificity")
    print()
    
    # Start training
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Total time: {training_time/3600:.2f} hours")
        print(f"📁 Results saved to: results_{timestamp}/")
        print(f"🏆 Models saved to: models_{timestamp}/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code: {e.returncode}")
        print("Check the error messages above for details.")
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        print("Partial results may be available in the output directories.")

def monitor_training():
    """Provide training monitoring tips"""
    print("\n📊 Training Monitoring Tips:")
    print("=" * 40)
    print("1. Watch for GPU memory usage: nvidia-smi")
    print("2. Monitor training progress in the terminal output")
    print("3. Check results/ directory for visualizations")
    print("4. Best model will be saved automatically")
    print("5. Training metrics will be plotted and saved")
    print()
    print("Expected training time:")
    print("  - Quick test: ~5-10 minutes")
    print("  - Standard: ~2-4 hours")
    print("  - High quality: ~4-8 hours")
    print()
    print("With 1:3 augmentation, you get:")
    print("  - 4x more training data per epoch")
    print("  - Better generalization")
    print("  - Improved class balance")
    print("  - More robust features")

def main():
    """Main function"""
    print("🚀 U-Net Training with Enhanced 1:3 Augmentation")
    print("=" * 60)
    print("This script will train a U-Net++ model for psoriasis segmentation")
    print("using the enhanced dataloader with 1:3 augmentation ratio.")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return
    
    # Get configuration
    config_args = get_training_config()
    
    # Show monitoring tips
    monitor_training()
    
    # Start training
    start_training(config_args)

if __name__ == "__main__":
    main()
