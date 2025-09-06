#!/usr/bin/env python3
"""
Comprehensive UNET++ Training and Evaluation Script for Psoriasis Segmentation
This script provides an easy interface to run training and evaluation experiments.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def run_training(args):
    """Run training with specified parameters"""
    print("🚀 Starting UNET++ Training...")
    print(f"📊 Dataset: {args.data_dir}")
    print(f"🏗️  Architecture: UNet++ with {args.encoder} encoder")
    print(f"📐 Image size: {args.image_size}x{args.image_size}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"📚 Epochs: {args.epochs}")
    print(f"⚡ Learning rate: {args.lr}")
    print("-" * 50)
    
    # Prepare training command
    cmd = [
        'python', 'train_unetpp.py',
        '--data_dir', args.data_dir,
        '--model_dir', args.model_dir,
        '--results_dir', args.results_dir,
        '--encoder', args.encoder,
        '--image_size', str(args.image_size),
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--patience', str(args.patience),
        '--loss_function', args.loss_function,
        '--scheduler', args.scheduler,
        '--num_workers', str(args.num_workers),
        '--seed', str(args.seed)
    ]
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False

def run_evaluation(args):
    """Run evaluation on trained model"""
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    
    if not os.path.exists(best_model_path):
        print(f"❌ Model not found at {best_model_path}")
        print("Please run training first or specify correct model path")
        return False
    
    print("🔬 Starting Model Evaluation...")
    print(f"📊 Model: {best_model_path}")
    print(f"📐 Threshold: {args.threshold}")
    print("-" * 50)
    
    # Prepare evaluation command
    cmd = [
        'python', 'predict.py',
        '--model_path', best_model_path,
        '--data_dir', args.data_dir,
        '--output_dir', args.prediction_dir,
        '--encoder', args.encoder,
        '--image_size', str(args.image_size),
        '--batch_size', str(args.batch_size),
        '--threshold', str(args.threshold),
        '--num_workers', str(args.num_workers)
    ]
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with error: {e}")
        return False

def run_single_prediction(args):
    """Run prediction on single image"""
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    
    if not os.path.exists(best_model_path):
        print(f"❌ Model not found at {best_model_path}")
        return False
    
    if not os.path.exists(args.single_image):
        print(f"❌ Image not found at {args.single_image}")
        return False
    
    print(f"🖼️  Predicting on single image: {args.single_image}")
    print("-" * 50)
    
    # Prepare prediction command
    cmd = [
        'python', 'predict.py',
        '--model_path', best_model_path,
        '--data_dir', args.data_dir,
        '--output_dir', args.prediction_dir,
        '--encoder', args.encoder,
        '--image_size', str(args.image_size),
        '--threshold', str(args.threshold),
        '--single_image', args.single_image
    ]
    
    # Run prediction
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Prediction completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Prediction failed with error: {e}")
        return False

def create_experiment_summary(args):
    """Create experiment summary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = f"""
# UNET++ Psoriasis Segmentation Experiment
## Experiment ID: {timestamp}

### Configuration:
- **Dataset**: {args.data_dir}
- **Architecture**: UNet++ with {args.encoder} encoder
- **Image Size**: {args.image_size}x{args.image_size}
- **Batch Size**: {args.batch_size}
- **Epochs**: {args.epochs}
- **Learning Rate**: {args.lr}
- **Loss Function**: {args.loss_function}
- **Scheduler**: {args.scheduler}

### Directories:
- **Models**: {args.model_dir}
- **Results**: {args.results_dir}
- **Predictions**: {args.prediction_dir}

### Commands Run:
"""
    
    if args.mode in ['train', 'all']:
        summary += "1. Training: ✅\n"
    
    if args.mode in ['eval', 'all']:
        summary += "2. Evaluation: ✅\n"
    
    if args.single_image:
        summary += f"3. Single Prediction: {args.single_image} ✅\n"
    
    summary += f"\n### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Save summary
    summary_path = os.path.join(args.results_dir, f'experiment_summary_{timestamp}.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"📄 Experiment summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description='UNET++ Psoriasis Segmentation Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training + evaluation
  python run_experiment.py --mode all
  
  # Training only
  python run_experiment.py --mode train --epochs 50
  
  # Evaluation only
  python run_experiment.py --mode eval
  
  # Single image prediction
  python run_experiment.py --mode predict --single_image path/to/image.jpg
  
  # Custom configuration
  python run_experiment.py --mode all --encoder efficientnet-b5 --batch_size 16 --lr 2e-4
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'eval', 'predict', 'all'],
                       help='Experiment mode')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save training results')
    parser.add_argument('--prediction_dir', type=str, default='predictions',
                       help='Directory to save predictions')
    
    # Model parameters
    parser.add_argument('--encoder', type=str, default='efficientnet-b3',
                       choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3', 
                               'efficientnet-b5', 'densenet121', 'densenet201'],
                       help='Encoder architecture')
    parser.add_argument('--image_size', type=int, default=960,
                       help='Input image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--loss_function', type=str, default='combined',
                       choices=['combined', 'dice', 'focal', 'tversky', 'bce'],
                       help='Loss function')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    parser.add_argument('--single_image', type=str,
                       help='Path to single image for prediction')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.prediction_dir, exist_ok=True)
    
    print("🧠 UNET++ Psoriasis Segmentation Experiment")
    print("=" * 50)
    
    success = True
    
    # Run based on mode
    if args.mode in ['train', 'all']:
        success &= run_training(args)
    
    if args.mode in ['eval', 'all'] and success:
        success &= run_evaluation(args)
    
    if args.mode == 'predict':
        if args.single_image:
            success &= run_single_prediction(args)
        else:
            success &= run_evaluation(args)
    
    # Create experiment summary
    if success:
        create_experiment_summary(args)
        
        print("\n" + "=" * 50)
        print("🎉 Experiment completed successfully!")
        print(f"📁 Check results in: {args.results_dir}")
        if args.mode in ['eval', 'all', 'predict']:
            print(f"📊 Check predictions in: {args.prediction_dir}")
        print("=" * 50)
    else:
        print("\n❌ Experiment failed. Check error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main() 