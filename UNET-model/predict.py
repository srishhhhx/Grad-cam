import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
from datetime import datetime

import segmentation_models_pytorch as smp
from utils.dataset import get_loaders, PsoriasisDataset, get_transforms
from utils.metrics import (
    dice_coef, iou_score, pixel_accuracy, sensitivity_specificity,
    visualize_predictions, create_confusion_matrix_plot,
    MetricsTracker
)

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path, device, encoder_name='efficientnet-b3'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with same architecture
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # No pretrained weights when loading from checkpoint
        in_channels=3,
        classes=1,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Model validation metrics: {checkpoint['metrics']}")
    
    return model, checkpoint

def predict_single_image(model, image_path, device, image_size=960, threshold=0.5):
    """Predict on a single image"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Resize image
    image = cv2.resize(image, (image_size, image_size))
    
    # Apply transforms
    transform = get_transforms(image_size=image_size, is_training=False)
    transformed = transform(image=image, mask=np.zeros((image_size, image_size)))
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.cpu().numpy()[0, 0]
    
    # Resize prediction back to original size
    prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]))
    
    # Create binary mask
    binary_mask = (prediction > threshold).astype(np.uint8)
    
    return prediction, binary_mask, image

def evaluate_test_set(model, test_loader, device, save_dir, threshold=0.5):
    """Comprehensive evaluation on test set"""
    model.eval()
    
    all_dice_scores = []
    all_iou_scores = []
    all_pixel_accuracies = []
    all_sensitivities = []
    all_specificities = []
    all_precisions = []
    
    detailed_results = []
    sample_predictions = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images = batch['image'].to(device)
            masks = batch['mask'].float().to(device)
            
            # Ensure masks have correct dimensions [B, 1, H, W]
            if len(masks.shape) == 3:  # [B, H, W]
                masks = masks.unsqueeze(1)  # [B, 1, H, W]
            filenames = batch['filename']
            
            # Predict
            predictions = model(images)
            
            # Calculate metrics for each image in batch
            for i in range(images.shape[0]):
                img = images[i:i+1]
                mask = masks[i:i+1]
                pred = predictions[i:i+1]
                filename = filenames[i]
                
                # Calculate metrics
                dice = dice_coef(pred, mask, threshold)
                iou = iou_score(pred, mask, threshold)
                pixel_acc = pixel_accuracy(pred, mask, threshold)
                sensitivity, specificity, precision = sensitivity_specificity(pred, mask, threshold)
                
                # Store metrics
                all_dice_scores.append(dice)
                all_iou_scores.append(iou)
                all_pixel_accuracies.append(pixel_acc)
                all_sensitivities.append(sensitivity)
                all_specificities.append(specificity)
                all_precisions.append(precision)
                
                # Store detailed results
                detailed_results.append({
                    'filename': filename,
                    'dice': dice,
                    'iou': iou,
                    'pixel_accuracy': pixel_acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision
                })
                
                # Store sample predictions for visualization
                if len(sample_predictions) < 12:  # Save first 12 samples
                    sample_predictions.append({
                        'image': img,
                        'mask': mask,
                        'prediction': pred,
                        'filename': filename
                    })
    
    # Calculate summary statistics
    summary_stats = {
        'dice': {
            'mean': np.mean(all_dice_scores),
            'std': np.std(all_dice_scores),
            'min': np.min(all_dice_scores),
            'max': np.max(all_dice_scores),
            'median': np.median(all_dice_scores)
        },
        'iou': {
            'mean': np.mean(all_iou_scores),
            'std': np.std(all_iou_scores),
            'min': np.min(all_iou_scores),
            'max': np.max(all_iou_scores),
            'median': np.median(all_iou_scores)
        },
        'pixel_accuracy': {
            'mean': np.mean(all_pixel_accuracies),
            'std': np.std(all_pixel_accuracies),
            'min': np.min(all_pixel_accuracies),
            'max': np.max(all_pixel_accuracies),
            'median': np.median(all_pixel_accuracies)
        },
        'sensitivity': {
            'mean': np.mean(all_sensitivities),
            'std': np.std(all_sensitivities),
            'min': np.min(all_sensitivities),
            'max': np.max(all_sensitivities),
            'median': np.median(all_sensitivities)
        },
        'specificity': {
            'mean': np.mean(all_specificities),
            'std': np.std(all_specificities),
            'min': np.min(all_specificities),
            'max': np.max(all_specificities),
            'median': np.median(all_specificities)
        },
        'precision': {
            'mean': np.mean(all_precisions),
            'std': np.std(all_precisions),
            'min': np.min(all_precisions),
            'max': np.max(all_precisions),
            'median': np.median(all_precisions)
        }
    }
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    
    for metric, stats in summary_stats.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Save detailed results
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(os.path.join(save_dir, 'detailed_test_results.csv'), index=False)
    
    # Save summary statistics
    with open(os.path.join(save_dir, 'test_summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create and save visualizations
    create_test_visualizations(
        df_results, summary_stats, sample_predictions, save_dir, threshold
    )
    
    return summary_stats, detailed_results, sample_predictions

def create_test_visualizations(df_results, summary_stats, sample_predictions, save_dir, threshold):
    """Create comprehensive visualizations for test results"""
    
    # 1. Metrics distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test Set Metrics Distribution', fontsize=16, fontweight='bold')
    
    metrics = ['dice', 'iou', 'pixel_accuracy', 'sensitivity', 'specificity', 'precision']
    metric_names = ['Dice Score', 'IoU Score', 'Pixel Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row, col = i // 3, i % 3
        
        # Histogram
        axes[row, col].hist(df_results[metric], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col].axvline(summary_stats[metric]['mean'], color='red', linestyle='--', 
                              label=f'Mean: {summary_stats[metric]["mean"]:.3f}')
        axes[row, col].axvline(summary_stats[metric]['median'], color='green', linestyle='--', 
                              label=f'Median: {summary_stats[metric]["median"]:.3f}')
        axes[row, col].set_title(f'{name} Distribution')
        axes[row, col].set_xlabel(name)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Box plots
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    data_for_boxplot = [df_results[metric] for metric in metrics]
    box_plot = ax.boxplot(data_for_boxplot, labels=metric_names, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Test Set Metrics - Box Plot Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Correlation matrix
    correlation_matrix = df_results[metrics].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={"shrink": .8})
    plt.title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Sample predictions visualization
    if sample_predictions:
        n_samples = min(12, len(sample_predictions))
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Sample Test Predictions', fontsize=16, fontweight='bold')
        
        for i in range(n_samples):
            sample = sample_predictions[i]
            
            # Denormalize image
            img = sample['image'][0].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            mask = sample['mask'][0, 0].cpu().numpy()
            pred = torch.sigmoid(sample['prediction'][0, 0]).cpu().numpy()
            pred_binary = (pred > threshold).astype(np.float32)
            
            # Calculate metrics for this sample
            dice = dice_coef(sample['prediction'], sample['mask'], threshold)
            iou = iou_score(sample['prediction'], sample['mask'], threshold)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original\n{sample["filename"]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction (Raw)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_binary, cmap='gray')
            axes[i, 3].set_title(f'Binary\nDice: {dice:.3f}, IoU: {iou:.3f}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Performance by image (sorted by Dice score)
    df_sorted = df_results.sort_values('dice', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot top and bottom performers
    n_show = min(20, len(df_sorted))
    
    # Top performers
    top_performers = df_sorted.head(n_show)
    x_pos = range(len(top_performers))
    
    ax1.bar(x_pos, top_performers['dice'], alpha=0.7, color='green', label='Dice')
    ax1.bar(x_pos, top_performers['iou'], alpha=0.7, color='blue', label='IoU')
    ax1.set_title(f'Top {n_show} Performing Images (by Dice Score)')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{fn[:15]}...' if len(fn) > 15 else fn 
                        for fn in top_performers['filename']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom performers
    bottom_performers = df_sorted.tail(n_show)
    x_pos = range(len(bottom_performers))
    
    ax2.bar(x_pos, bottom_performers['dice'], alpha=0.7, color='red', label='Dice')
    ax2.bar(x_pos, bottom_performers['iou'], alpha=0.7, color='orange', label='IoU')
    ax2.set_title(f'Bottom {n_show} Performing Images (by Dice Score)')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Images')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{fn[:15]}...' if len(fn) > 15 else fn 
                        for fn in bottom_performers['filename']], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_ranking.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_report(args, model_info, test_stats, save_dir):
    """Create a comprehensive prediction report"""
    report = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'test_dataset': args.data_dir,
            'threshold': args.threshold,
            'image_size': args.image_size
        },
        'model_info': model_info,
        'test_statistics': test_stats,
        'summary': {
            'total_images': len(os.listdir(os.path.join(args.data_dir, 'test'))) - 1,  # -1 for annotations file
            'mean_dice': test_stats['dice']['mean'],
            'mean_iou': test_stats['iou']['mean'],
            'mean_pixel_accuracy': test_stats['pixel_accuracy']['mean']
        }
    }
    
    with open(os.path.join(save_dir, 'prediction_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPrediction report saved to: {os.path.join(save_dir, 'prediction_report.json')}")

def main():
    parser = argparse.ArgumentParser(description='Predict using trained UNet++ model')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--encoder', type=str, default='efficientnet-b3', help='Encoder architecture')
    parser.add_argument('--image_size', type=int, default=960, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--single_image', type=str, help='Path to single image for prediction')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.model_path, device, args.encoder)
    
    if args.single_image:
        # Predict on single image
        print(f"Predicting on single image: {args.single_image}")
        prediction, binary_mask, original_image = predict_single_image(
            model, args.single_image, device, args.image_size, args.threshold
        )
        
        # Visualize result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.resize(original_image, (args.image_size, args.image_size)))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction (Probability)')
        axes[1].axis('off')
        
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title(f'Binary Mask (threshold={args.threshold})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'single_prediction.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        # Evaluate on test set
        print("Loading test dataset...")
        loaders = get_loaders(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
        
        # Evaluate
        test_stats, detailed_results, sample_predictions = evaluate_test_set(
            model, loaders['test'], device, args.output_dir, args.threshold
        )
        
        # Create comprehensive report
        model_info = {
            'architecture': 'UNet++',
            'encoder': args.encoder,
            'checkpoint_epoch': checkpoint['epoch'],
            'checkpoint_metrics': checkpoint.get('metrics', {}),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
        
        create_prediction_report(args, model_info, test_stats, args.output_dir)
        
        print(f"\nAll results saved to: {args.output_dir}")
        print("\nFiles created:")
        for file in os.listdir(args.output_dir):
            print(f"  - {file}")

if __name__ == '__main__':
    main() 