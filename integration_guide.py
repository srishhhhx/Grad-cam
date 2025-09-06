"""
Integration Guide: How to use Enhanced DataLoader with existing U-Net training

This file shows how to modify the existing train_unetpp.py to use the enhanced dataloader
"""

# Example of how to modify the existing training script

def modified_train_model_example():
    """
    Example showing how to integrate enhanced dataloader into existing training script
    """
    
    # 1. Import the enhanced dataloader
    from enhanced_dataloader import get_enhanced_loaders
    
    # 2. Replace the existing get_loaders call in train_unetpp.py
    # OLD CODE (line ~200-206 in train_unetpp.py):
    """
    loaders = get_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    """
    
    # NEW CODE:
    """
    loaders = get_enhanced_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation_level='medium',  # 'light', 'medium', 'heavy'
        use_weighted_sampling=True,   # Enable for class imbalance
        cache_images=False,           # Enable for small datasets
        validate_data=True            # Validate data integrity
    )
    """


def create_modified_training_script():
    """
    Create a modified version of the training script with enhanced dataloader
    """
    
    modified_script = '''
# Modified train_unetpp.py with Enhanced DataLoader
import os
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from datetime import datetime
import json

# Disable albumentations update check warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# MODIFIED: Import enhanced dataloader instead of original
from enhanced_dataloader import get_enhanced_loaders

from utils.metrics import (
    CombinedLoss, DiceLoss, FocalLoss, TverskyLoss,
    dice_coef, iou_score, pixel_accuracy, sensitivity_specificity,
    MetricsTracker, visualize_predictions, create_confusion_matrix_plot
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_model(args):
    """Main training function with enhanced dataloader"""
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # MODIFIED: Initialize enhanced loaders
    print("Loading datasets with enhanced dataloader...")
    loaders = get_enhanced_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation_level=args.augmentation_level,  # NEW PARAMETER
        use_weighted_sampling=args.use_weighted_sampling,  # NEW PARAMETER
        cache_images=args.cache_images,  # NEW PARAMETER
        validate_data=args.validate_data  # NEW PARAMETER
    )
    
    # Rest of the training code remains the same...
    # [Continue with existing training logic]

def parse_args():
    parser = argparse.ArgumentParser(description='Train a UNet++ model with enhanced dataloader')
    
    # Existing parameters...
    parser.add_argument('--data_dir', type=str, default='Dataset_no_preprocessing', 
                       help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory to save logs')
    
    # Model parameters
    parser.add_argument('--encoder', type=str, default='efficientnet-b3', 
                       choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3', 
                               'efficientnet-b5', 'densenet121', 'densenet201'],
                       help='Encoder architecture')
    parser.add_argument('--image_size', type=int, default=640, help='Input image size')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # NEW PARAMETERS for enhanced dataloader
    parser.add_argument('--augmentation_level', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Augmentation intensity level')
    parser.add_argument('--use_weighted_sampling', action='store_true',
                       help='Use weighted sampling for class balance')
    parser.add_argument('--cache_images', action='store_true',
                       help='Cache images in memory (for small datasets)')
    parser.add_argument('--validate_data', action='store_true', default=True,
                       help='Validate data integrity during loading')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
'''
    
    return modified_script


def recommended_configurations():
    """
    Recommended configurations for different scenarios
    """
    
    configs = {
        "small_dataset": {
            "description": "For datasets with < 1000 images",
            "config": {
                "batch_size": 8,
                "augmentation_level": "heavy",
                "use_weighted_sampling": True,
                "cache_images": True,
                "validate_data": True,
                "image_size": 512
            }
        },
        
        "medium_dataset": {
            "description": "For datasets with 1000-5000 images",
            "config": {
                "batch_size": 4,
                "augmentation_level": "medium",
                "use_weighted_sampling": True,
                "cache_images": False,
                "validate_data": True,
                "image_size": 640
            }
        },
        
        "large_dataset": {
            "description": "For datasets with > 5000 images",
            "config": {
                "batch_size": 2,
                "augmentation_level": "light",
                "use_weighted_sampling": False,
                "cache_images": False,
                "validate_data": False,
                "image_size": 768
            }
        },
        
        "high_resolution": {
            "description": "For high-resolution training",
            "config": {
                "batch_size": 1,
                "augmentation_level": "light",
                "use_weighted_sampling": False,
                "cache_images": False,
                "validate_data": True,
                "image_size": 960
            }
        },
        
        "quick_experiment": {
            "description": "For quick experiments and debugging",
            "config": {
                "batch_size": 2,
                "augmentation_level": "light",
                "use_weighted_sampling": False,
                "cache_images": False,
                "validate_data": False,
                "image_size": 256
            }
        }
    }
    
    return configs


def print_integration_instructions():
    """Print step-by-step integration instructions"""
    
    print("🔧 Enhanced DataLoader Integration Guide")
    print("=" * 50)
    
    print("\n📋 Step 1: Install Dependencies")
    print("-" * 30)
    print("Make sure you have all required packages:")
    print("pip install torch torchvision albumentations opencv-python pycocotools")
    
    print("\n📋 Step 2: Copy Enhanced DataLoader")
    print("-" * 30)
    print("1. Copy 'enhanced_dataloader.py' to your project directory")
    print("2. Make sure it's in the same directory as your training script")
    
    print("\n📋 Step 3: Modify Training Script")
    print("-" * 30)
    print("Replace the import in your training script:")
    print("OLD: from utils.dataset import get_loaders")
    print("NEW: from enhanced_dataloader import get_enhanced_loaders")
    
    print("\n📋 Step 4: Update Function Call")
    print("-" * 30)
    print("Replace the get_loaders call with:")
    print("""
loaders = get_enhanced_loaders(
    root_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    image_size=args.image_size,
    augmentation_level='medium',
    use_weighted_sampling=True,
    cache_images=False,
    validate_data=True
)
""")
    
    print("\n📋 Step 5: Add New Arguments (Optional)")
    print("-" * 30)
    print("Add these arguments to your argument parser:")
    print("""
parser.add_argument('--augmentation_level', default='medium', 
                   choices=['light', 'medium', 'heavy'])
parser.add_argument('--use_weighted_sampling', action='store_true')
parser.add_argument('--cache_images', action='store_true')
""")
    
    print("\n📋 Step 6: Test the Integration")
    print("-" * 30)
    print("Run the test script to verify everything works:")
    print("python test_enhanced_dataloader.py")
    
    print("\n📋 Step 7: Choose Configuration")
    print("-" * 30)
    print("Select appropriate configuration based on your dataset size:")
    
    configs = recommended_configurations()
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  {config['description']}")
        for key, value in config['config'].items():
            print(f"  {key}: {value}")
    
    print("\n✅ Integration Complete!")
    print("Your enhanced dataloader is ready for training!")


if __name__ == "__main__":
    print_integration_instructions()
