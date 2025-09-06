import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score
import cv2
from typing import Dict, List, Tuple
import pandas as pd

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class IoULoss(nn.Module):
    """IoU Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss"""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        TP = (predictions * targets).sum()
        FP = ((1 - targets) * predictions).sum()
        FN = (targets * (1 - predictions)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    def __init__(self, dice_weight=0.3, focal_weight=0.4, tversky_weight=0.3, alpha=1, gamma=2):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.tversky_loss = TverskyLoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        tversky = self.tversky_loss(predictions, targets)
        
        combined = (self.dice_weight * dice + 
                   self.focal_weight * focal + 
                   self.tversky_weight * tversky)
        
        return combined

def dice_coef(predictions, targets, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice.item()

def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """Calculate IoU score"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()

def pixel_accuracy(predictions, targets, threshold=0.5):
    """Calculate pixel accuracy"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = targets.float()
    
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    accuracy = correct / total
    
    return accuracy.item()

def sensitivity_specificity(predictions, targets, threshold=0.5):
    """Calculate sensitivity (recall) and specificity"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    targets = targets.float()
    
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    TP = ((predictions == 1) & (targets == 1)).float().sum()
    TN = ((predictions == 0) & (targets == 0)).float().sum()
    FP = ((predictions == 1) & (targets == 0)).float().sum()
    FN = ((predictions == 0) & (targets == 1)).float().sum()
    
    sensitivity = TP / (TP + FN + 1e-6)  # Recall
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    
    return sensitivity.item(), specificity.item(), precision.item()

def hausdorff_distance(predictions, targets, threshold=0.5):
    """Calculate Hausdorff distance"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Convert to numpy
    pred_np = predictions.cpu().numpy().astype(np.uint8)
    target_np = targets.cpu().numpy().astype(np.uint8)
    
    hausdorff_distances = []
    
    for i in range(pred_np.shape[0]):
        pred_contours, _ = cv2.findContours(pred_np[i, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_contours, _ = cv2.findContours(target_np[i, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(pred_contours) == 0 or len(target_contours) == 0:
            hausdorff_distances.append(float('inf'))
            continue
        
        # Get largest contours
        pred_contour = max(pred_contours, key=cv2.contourArea)
        target_contour = max(target_contours, key=cv2.contourArea)
        
        # Calculate Hausdorff distance (simplified)
        try:
            distances = []
            for point in pred_contour:
                min_dist = min([cv2.pointPolygonTest(target_contour, tuple(point[0]), True) 
                               for _ in range(len(target_contour))])
                distances.append(abs(min_dist))
            
            hausdorff_distances.append(max(distances) if distances else float('inf'))
        except:
            hausdorff_distances.append(float('inf'))
    
    return np.mean([d for d in hausdorff_distances if d != float('inf')])

class MetricsTracker:
    """Track and visualize training metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'train_pixel_acc': [], 'val_pixel_acc': [],
            'train_sensitivity': [], 'val_sensitivity': [],
            'train_specificity': [], 'val_specificity': [],
            'train_precision': [], 'val_precision': [],
            'learning_rates': [],
            'epochs': []
        }
    
    def update(self, epoch, metrics_dict):
        """Update metrics for current epoch"""
        self.metrics['epochs'].append(epoch)
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot_metrics(self, save_path=None):
        """Plot comprehensive training metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(self.metrics['epochs'], self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.metrics['epochs'], self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice Score
        axes[0, 1].plot(self.metrics['epochs'], self.metrics['train_dice'], 'b-', label='Train Dice', linewidth=2)
        axes[0, 1].plot(self.metrics['epochs'], self.metrics['val_dice'], 'r-', label='Val Dice', linewidth=2)
        axes[0, 1].set_title('Dice Score Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU Score
        axes[1, 0].plot(self.metrics['epochs'], self.metrics['train_iou'], 'b-', label='Train IoU', linewidth=2)
        axes[1, 0].plot(self.metrics['epochs'], self.metrics['val_iou'], 'r-', label='Val IoU', linewidth=2)
        axes[1, 0].set_title('IoU Score Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Pixel Accuracy
        axes[1, 1].plot(self.metrics['epochs'], self.metrics['train_pixel_acc'], 'b-', label='Train Pixel Acc', linewidth=2)
        axes[1, 1].plot(self.metrics['epochs'], self.metrics['val_pixel_acc'], 'r-', label='Val Pixel Acc', linewidth=2)
        axes[1, 1].set_title('Pixel Accuracy Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pixel Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Sensitivity & Specificity
        axes[2, 0].plot(self.metrics['epochs'], self.metrics['train_sensitivity'], 'b-', label='Train Sensitivity', linewidth=2)
        axes[2, 0].plot(self.metrics['epochs'], self.metrics['val_sensitivity'], 'r-', label='Val Sensitivity', linewidth=2)
        axes[2, 0].plot(self.metrics['epochs'], self.metrics['train_specificity'], 'g-', label='Train Specificity', linewidth=2)
        axes[2, 0].plot(self.metrics['epochs'], self.metrics['val_specificity'], 'orange', label='Val Specificity', linewidth=2)
        axes[2, 0].set_title('Sensitivity & Specificity Over Time')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if self.metrics['learning_rates']:
            axes[2, 1].plot(self.metrics['epochs'], self.metrics['learning_rates'], 'purple', linewidth=2)
            axes[2, 1].set_title('Learning Rate Over Time')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Learning Rate')
            axes[2, 1].set_yscale('log')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        plt.show()
    
    def get_summary_table(self):
        """Get summary statistics table"""
        if not self.metrics['epochs']:
            return None
        
        latest_metrics = {
            'Metric': ['Loss', 'Dice Score', 'IoU Score', 'Pixel Accuracy', 'Sensitivity', 'Specificity', 'Precision'],
            'Train (Latest)': [
                self.metrics['train_loss'][-1] if self.metrics['train_loss'] else 0,
                self.metrics['train_dice'][-1] if self.metrics['train_dice'] else 0,
                self.metrics['train_iou'][-1] if self.metrics['train_iou'] else 0,
                self.metrics['train_pixel_acc'][-1] if self.metrics['train_pixel_acc'] else 0,
                self.metrics['train_sensitivity'][-1] if self.metrics['train_sensitivity'] else 0,
                self.metrics['train_specificity'][-1] if self.metrics['train_specificity'] else 0,
                self.metrics['train_precision'][-1] if self.metrics['train_precision'] else 0,
            ],
            'Validation (Latest)': [
                self.metrics['val_loss'][-1] if self.metrics['val_loss'] else 0,
                self.metrics['val_dice'][-1] if self.metrics['val_dice'] else 0,
                self.metrics['val_iou'][-1] if self.metrics['val_iou'] else 0,
                self.metrics['val_pixel_acc'][-1] if self.metrics['val_pixel_acc'] else 0,
                self.metrics['val_sensitivity'][-1] if self.metrics['val_sensitivity'] else 0,
                self.metrics['val_specificity'][-1] if self.metrics['val_specificity'] else 0,
                self.metrics['val_precision'][-1] if self.metrics['val_precision'] else 0,
            ],
            'Best Validation': [
                min(self.metrics['val_loss']) if self.metrics['val_loss'] else 0,
                max(self.metrics['val_dice']) if self.metrics['val_dice'] else 0,
                max(self.metrics['val_iou']) if self.metrics['val_iou'] else 0,
                max(self.metrics['val_pixel_acc']) if self.metrics['val_pixel_acc'] else 0,
                max(self.metrics['val_sensitivity']) if self.metrics['val_sensitivity'] else 0,
                max(self.metrics['val_specificity']) if self.metrics['val_specificity'] else 0,
                max(self.metrics['val_precision']) if self.metrics['val_precision'] else 0,
            ]
        }
        
        return pd.DataFrame(latest_metrics)

def visualize_predictions(images, masks, predictions, num_samples=4, save_path=None):
    """Visualize predictions alongside ground truth"""
    batch_size = images.shape[0]
    num_samples = min(num_samples, batch_size)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Get mask and prediction
        mask = masks[i, 0].cpu().numpy()
        pred = torch.sigmoid(predictions[i, 0]).cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction (Raw)')
        axes[i, 2].axis('off')
        
        # Binary prediction
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title('Prediction (Binary)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.show()

def create_confusion_matrix_plot(predictions, targets, threshold=0.5, save_path=None):
    """Create confusion matrix visualization"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    
    # Flatten and convert to numpy
    pred_flat = predictions.cpu().numpy().flatten()
    target_flat = targets.cpu().numpy().flatten()
    
    # Create confusion matrix
    cm = confusion_matrix(target_flat, pred_flat)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Psoriasis'], 
                yticklabels=['Background', 'Psoriasis'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm 