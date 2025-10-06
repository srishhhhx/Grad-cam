#!/usr/bin/env python3
"""
Robust Grad-CAM Implementation for U-Net++ EfficientNet-B5 Segmentation Model

This implementation provides comprehensive Grad-CAM visualization for understanding
which features the model focuses on when making segmentation predictions.

Features:
- Multiple target layer options (encoder stages, decoder stages, bottleneck)
- Batch processing capabilities
- Multiple Grad-CAM variants (Grad-CAM, Grad-CAM++, Score-CAM)
- Comprehensive visualizations with overlays
- Statistical analysis of attention patterns
- Export capabilities for research and clinical use
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple, Union
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import model and data loading utilities
import segmentation_models_pytorch as smp
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "UNET-model"))
from utils.enhanced_dataloader import get_1_to_3_augmentation_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Try to import pytorch-grad-cam, install if not available
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    print("⚠️  pytorch-grad-cam not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grad-cam"])
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    GRADCAM_AVAILABLE = True


class UNetGradCAM:
    """
    Comprehensive Grad-CAM implementation for U-Net++ segmentation models.
    
    This class provides multiple Grad-CAM variants and visualization options
    specifically designed for medical image segmentation tasks.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize Grad-CAM analyzer.
        
        Args:
            model: Trained U-Net++ model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Get available target layers
        self.target_layers = self._get_target_layers()
        
    def _get_target_layers(self) -> Dict[str, torch.nn.Module]:
        """
        Get available target layers from the U-Net++ model.
        
        Returns:
            Dictionary mapping layer names to layer modules
        """
        layers = {}
        
        # Encoder layers (EfficientNet-B5)
        if hasattr(self.model, 'encoder'):
            # Last encoder block (most commonly used)
            if hasattr(self.model.encoder, '_blocks') and len(self.model.encoder._blocks) > 0:
                layers['encoder_last'] = self.model.encoder._blocks[-1]
                layers['encoder_mid'] = self.model.encoder._blocks[len(self.model.encoder._blocks)//2]
                layers['encoder_early'] = self.model.encoder._blocks[2] if len(self.model.encoder._blocks) > 2 else self.model.encoder._blocks[0]
            
            # Alternative encoder access patterns
            elif hasattr(self.model.encoder, 'features'):
                layers['encoder_last'] = self.model.encoder.features[-1]
            elif hasattr(self.model.encoder, 'layer4'):
                layers['encoder_last'] = self.model.encoder.layer4
        
        # Decoder layers
        if hasattr(self.model, 'decoder'):
            decoder_blocks = []
            for name, module in self.model.decoder.named_children():
                if 'block' in name.lower() or 'conv' in name.lower():
                    decoder_blocks.append(module)
            
            if decoder_blocks:
                layers['decoder_last'] = decoder_blocks[-1]
                if len(decoder_blocks) > 1:
                    layers['decoder_mid'] = decoder_blocks[len(decoder_blocks)//2]
        
        # Segmentation head
        if hasattr(self.model, 'segmentation_head'):
            layers['seg_head'] = self.model.segmentation_head[0] if isinstance(self.model.segmentation_head, torch.nn.Sequential) else self.model.segmentation_head
        
        return layers
    
    def compute_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_layer_name: str = 'encoder_last',
        method: str = 'gradcam',
        target_class: int = 1
    ) -> np.ndarray:
        """
        Compute Grad-CAM for the given input.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_mask: Target segmentation mask [1, H, W] (optional)
            target_layer_name: Name of target layer
            method: Grad-CAM method ('gradcam', 'gradcam++', 'scorecam', 'layercam')
            target_class: Target class for visualization
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        if target_layer_name not in self.target_layers:
            print(f"⚠️  Layer '{target_layer_name}' not found. Available layers: {list(self.target_layers.keys())}")
            target_layer_name = 'encoder_last'
        
        target_layer = self.target_layers[target_layer_name]
        
        # Select Grad-CAM method (updated API without use_cuda parameter)
        if method == 'gradcam':
            cam = GradCAM(model=self.model, target_layers=[target_layer])
        elif method == 'gradcam++':
            cam = GradCAMPlusPlus(model=self.model, target_layers=[target_layer])
        elif method == 'scorecam':
            cam = ScoreCAM(model=self.model, target_layers=[target_layer])
        elif method == 'layercam':
            cam = LayerCAM(model=self.model, target_layers=[target_layer])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Prepare targets for binary segmentation
        if target_mask is not None:
            # Use provided mask as target
            mask_np = target_mask.squeeze().cpu().numpy()
            # For binary segmentation, use category=None and just the mask
            targets = [SemanticSegmentationTarget(category=None, mask=mask_np)]
        else:
            # Use model prediction as target
            with torch.no_grad():
                prediction = self.model(input_tensor)
                pred_mask = torch.sigmoid(prediction) > 0.5
                mask_np = pred_mask.squeeze().cpu().numpy()
                targets = [SemanticSegmentationTarget(category=None, mask=mask_np)]
        
        # Compute Grad-CAM
        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0]  # Return first (and only) result
        except Exception as e:
            print(f"❌ Error computing Grad-CAM: {str(e)}")
            # Return zeros as fallback
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
    
    def create_visualization(
        self,
        input_tensor: torch.Tensor,
        gradcam_heatmap: np.ndarray,
        prediction: Optional[torch.Tensor] = None,
        ground_truth: Optional[torch.Tensor] = None,
        title: str = "Grad-CAM Visualization"
    ) -> plt.Figure:
        """
        Create comprehensive Grad-CAM visualization.
        
        Args:
            input_tensor: Original input tensor [1, 3, H, W]
            gradcam_heatmap: Grad-CAM heatmap
            prediction: Model prediction [1, 1, H, W]
            ground_truth: Ground truth mask [1, H, W]
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Prepare image for visualization
        image_np = input_tensor.squeeze().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        
        # Denormalize image (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Create overlay
        overlay = show_cam_on_image(image_np, gradcam_heatmap, use_rgb=True)
        
        # Determine subplot layout (removed prediction column)
        n_cols = 3  # Original, Heatmap, Overlay
        if ground_truth is not None:
            n_cols += 1

        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

        # Row 1: Original visualizations
        col = 0

        # Original image
        axes[0, col].imshow(image_np)
        axes[0, col].set_title('Original Image')
        axes[0, col].axis('off')
        col += 1

        # Grad-CAM heatmap
        im1 = axes[0, col].imshow(gradcam_heatmap, cmap='jet', alpha=0.8)
        axes[0, col].set_title('Grad-CAM Heatmap')
        axes[0, col].axis('off')
        plt.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)
        col += 1

        # Grad-CAM overlay
        axes[0, col].imshow(overlay)
        axes[0, col].set_title('Grad-CAM Overlay')
        axes[0, col].axis('off')
        col += 1

        # Ground truth (if available)
        if ground_truth is not None:
            gt_np = ground_truth.squeeze().cpu().numpy()
            axes[0, col].imshow(gt_np, cmap='gray')
            axes[0, col].set_title('Ground Truth')
            axes[0, col].axis('off')
        
        # Row 2: Combined visualizations
        col = 0

        # Image with Grad-CAM overlay (simplified)
        axes[1, col].imshow(image_np)
        axes[1, col].imshow(gradcam_heatmap, cmap='jet', alpha=0.4)
        axes[1, col].set_title('Enhanced Overlay')
        axes[1, col].axis('off')
        col += 1

        # Attention intensity visualization
        axes[1, col].imshow(gradcam_heatmap, cmap='hot')
        axes[1, col].set_title('Attention Intensity')
        axes[1, col].axis('off')
        col += 1

        # Attention statistics visualization
        attention_stats = {
            'Mean': gradcam_heatmap.mean(),
            'Max': gradcam_heatmap.max(),
            'Std': gradcam_heatmap.std()
        }

        categories = list(attention_stats.keys())
        values = list(attention_stats.values())

        bars = axes[1, col].bar(categories, values, color=['blue', 'red', 'green'], alpha=0.7)
        axes[1, col].set_title('Attention Statistics')
        axes[1, col].set_ylabel('Attention Value')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
        col += 1
        
        # Fill remaining subplots if needed
        while col < n_cols:
            axes[1, col].axis('off')
            col += 1
        
        plt.suptitle(title, fontsize=16, y=0.95)
        plt.tight_layout()
        
        return fig

    def analyze_attention_patterns(
        self,
        input_tensor: torch.Tensor,
        gradcam_heatmap: np.ndarray,
        prediction: Optional[torch.Tensor] = None,
        ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Analyze attention patterns and compute statistics.

        Args:
            input_tensor: Input tensor
            gradcam_heatmap: Grad-CAM heatmap
            prediction: Model prediction
            ground_truth: Ground truth mask

        Returns:
            Dictionary with analysis statistics
        """
        stats = {
            'attention_mean': float(gradcam_heatmap.mean()),
            'attention_std': float(gradcam_heatmap.std()),
            'attention_max': float(gradcam_heatmap.max()),
            'attention_min': float(gradcam_heatmap.min()),
        }

        if prediction is not None:
            pred_mask = torch.sigmoid(prediction) > 0.5
            pred_np = pred_mask.squeeze().cpu().numpy()

            # Attention in predicted regions vs background
            if pred_np.sum() > 0:
                stats['attention_in_prediction'] = float(gradcam_heatmap[pred_np].mean())
                stats['attention_in_background'] = float(gradcam_heatmap[~pred_np].mean())
                stats['attention_ratio'] = stats['attention_in_prediction'] / (stats['attention_in_background'] + 1e-8)
            else:
                stats['attention_in_prediction'] = 0.0
                stats['attention_in_background'] = float(gradcam_heatmap.mean())
                stats['attention_ratio'] = 0.0

            stats['prediction_area'] = float(pred_np.sum())
            stats['prediction_ratio'] = float(pred_np.mean())

        if ground_truth is not None:
            gt_np = ground_truth.squeeze().cpu().numpy()

            # Attention in ground truth regions
            if gt_np.sum() > 0:
                stats['attention_in_gt'] = float(gradcam_heatmap[gt_np > 0.5].mean())
                stats['attention_in_gt_bg'] = float(gradcam_heatmap[gt_np <= 0.5].mean())
                stats['gt_attention_ratio'] = stats['attention_in_gt'] / (stats['attention_in_gt_bg'] + 1e-8)
            else:
                stats['attention_in_gt'] = 0.0
                stats['attention_in_gt_bg'] = float(gradcam_heatmap.mean())
                stats['gt_attention_ratio'] = 0.0

            stats['gt_area'] = float(gt_np.sum())
            stats['gt_ratio'] = float(gt_np.mean())

        return stats

    def compare_methods(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_layer_name: str = 'encoder_last',
        methods: List[str] = ['gradcam', 'gradcam++', 'scorecam']
    ) -> Dict[str, np.ndarray]:
        """
        Compare different Grad-CAM methods.

        Args:
            input_tensor: Input tensor
            target_mask: Target mask
            target_layer_name: Target layer name
            methods: List of methods to compare

        Returns:
            Dictionary mapping method names to heatmaps
        """
        results = {}

        for method in methods:
            try:
                print(f"🔍 Computing {method.upper()}...")
                heatmap = self.compute_gradcam(
                    input_tensor=input_tensor,
                    target_mask=target_mask,
                    target_layer_name=target_layer_name,
                    method=method
                )
                results[method] = heatmap
            except Exception as e:
                print(f"❌ Error with {method}: {str(e)}")
                results[method] = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        return results

    def create_method_comparison_plot(
        self,
        input_tensor: torch.Tensor,
        method_results: Dict[str, np.ndarray],
        prediction: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comparison plot of different Grad-CAM methods.

        Args:
            input_tensor: Input tensor
            method_results: Dictionary of method results
            prediction: Model prediction (not displayed)
            save_path: Path to save plot

        Returns:
            Matplotlib figure
        """
        n_methods = len(method_results)
        # Only create 2 rows x n_methods columns (removed +1 for prediction)
        fig, axes = plt.subplots(2, n_methods, figsize=(4*n_methods, 8))

        # Prepare image
        image_np = input_tensor.squeeze().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)

        # Method results (no original image or prediction columns)
        for i, (method, heatmap) in enumerate(method_results.items()):
            # Heatmap
            im = axes[0, i].imshow(heatmap, cmap='jet')
            axes[0, i].set_title(f'{method.upper()} Heatmap')
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Overlay
            overlay = show_cam_on_image(image_np, heatmap, use_rgb=True)
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'{method.upper()} Overlay')
            axes[1, i].axis('off')

        plt.suptitle('Grad-CAM Method Comparison', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Method comparison saved to {save_path}")

        return fig


def load_model(model_path: str, device: torch.device, encoder_name: str = 'efficientnet-b5') -> torch.nn.Module:
    """
    Load trained U-Net++ model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        encoder_name: Encoder architecture name

    Returns:
        Loaded model
    """
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

    print(f"✅ Model loaded successfully from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"📊 Model validation metrics: {checkpoint['metrics']}")

    return model


def preprocess_image_for_gradcam(image_path: str, image_size: int = 640) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for Grad-CAM analysis.

    Args:
        image_path: Path to input image
        image_size: Target image size

    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    # Define preprocessing transform
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Apply transform
    transformed = transform(image=image)
    tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

    return tensor, original_image


def run_gradcam_analysis(
    model_path: str,
    image_path: str,
    output_dir: str,
    encoder_name: str = 'efficientnet-b5',
    image_size: int = 640,
    target_layer: str = 'encoder_last',
    methods: List[str] = ['gradcam', 'gradcam++'],
    device: Optional[torch.device] = None
) -> Dict:
    """
    Run comprehensive Grad-CAM analysis on a single image.

    Args:
        model_path: Path to trained model checkpoint
        image_path: Path to input image
        output_dir: Directory to save results
        encoder_name: Model encoder architecture
        image_size: Input image size
        target_layer: Target layer for Grad-CAM
        methods: List of Grad-CAM methods to use
        device: Device to run on

    Returns:
        Dictionary with analysis results
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    print(f"🔬 Running Grad-CAM Analysis")
    print(f"📁 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")

    # Load model
    model = load_model(model_path, device, encoder_name)

    # Initialize Grad-CAM
    gradcam = UNetGradCAM(model, device)

    print(f"🎯 Available target layers: {list(gradcam.target_layers.keys())}")

    # Preprocess image
    print("🔄 Preprocessing image...")
    input_tensor, original_image = preprocess_image_for_gradcam(image_path, image_size)
    input_tensor = input_tensor.to(device)

    # Get model prediction
    print("🔮 Getting model prediction...")
    with torch.no_grad():
        prediction = model(input_tensor)

    # Compare different methods
    print(f"🧮 Computing Grad-CAM with methods: {methods}")
    method_results = gradcam.compare_methods(
        input_tensor=input_tensor,
        target_layer_name=target_layer,
        methods=methods
    )

    # Analyze each method
    results = {}
    image_name = Path(image_path).stem

    for method, heatmap in method_results.items():
        print(f"📊 Analyzing {method.upper()}...")

        # Create individual visualization
        fig = gradcam.create_visualization(
            input_tensor=input_tensor,
            gradcam_heatmap=heatmap,
            prediction=prediction,
            title=f'{method.upper()} Analysis - {image_name}'
        )

        # Save visualization
        viz_path = os.path.join(output_dir, f'{image_name}_{method}_analysis.png')
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Analyze attention patterns
        stats = gradcam.analyze_attention_patterns(
            input_tensor=input_tensor,
            gradcam_heatmap=heatmap,
            prediction=prediction
        )

        results[method] = {
            'heatmap': heatmap,
            'statistics': stats,
            'visualization_path': viz_path
        }

    # Create method comparison
    print("🔍 Creating method comparison...")
    comparison_path = os.path.join(output_dir, f'{image_name}_method_comparison.png')
    comp_fig = gradcam.create_method_comparison_plot(
        input_tensor=input_tensor,
        method_results=method_results,
        prediction=prediction,
        save_path=comparison_path
    )
    plt.close(comp_fig)

    # Save comprehensive report
    report_path = os.path.join(output_dir, f'{image_name}_gradcam_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Grad-CAM Analysis Report\n")
        f.write(f"========================\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Target Layer: {target_layer}\n")
        f.write(f"Methods: {', '.join(methods)}\n\n")

        for method, result in results.items():
            stats = result['statistics']
            f.write(f"{method.upper()} Results:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Attention Mean: {stats['attention_mean']:.6f}\n")
            f.write(f"Attention Std: {stats['attention_std']:.6f}\n")
            f.write(f"Attention Max: {stats['attention_max']:.6f}\n")
            f.write(f"Attention Min: {stats['attention_min']:.6f}\n")

            if 'attention_ratio' in stats:
                f.write(f"Attention Ratio (Pred/BG): {stats['attention_ratio']:.4f}\n")
                f.write(f"Prediction Ratio: {stats['prediction_ratio']:.4f}\n")

            f.write(f"\n")

        f.write(f"Interpretation Guide:\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"- Attention Ratio > 2.0: Good focus on predicted regions\n")
        f.write(f"- Attention Ratio < 1.0: May be focusing on irrelevant areas\n")
        f.write(f"- Higher attention values indicate stronger feature importance\n")
        f.write(f"- Compare different methods for robust interpretation\n")

    print(f"✅ Analysis complete! Results saved to {output_dir}")

    return {
        'results': results,
        'comparison_path': comparison_path,
        'report_path': report_path,
        'target_layers': list(gradcam.target_layers.keys())
    }


def batch_gradcam_analysis(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_samples: int = 10,
    encoder_name: str = 'efficientnet-b5',
    image_size: int = 640,
    target_layer: str = 'encoder_last',
    methods: List[str] = ['gradcam', 'gradcam++']
) -> List[Dict]:
    """
    Run Grad-CAM analysis on multiple test images.

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to dataset directory
        output_dir: Directory to save results
        num_samples: Number of test samples to analyze
        encoder_name: Model encoder architecture
        image_size: Input image size
        target_layer: Target layer for Grad-CAM
        methods: List of Grad-CAM methods to use

    Returns:
        List of analysis results for each image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🔬 Running Batch Grad-CAM Analysis")
    print(f"📁 Dataset: {data_dir}")
    print(f"🔢 Samples: {num_samples}")

    # Load test data
    print("📂 Loading test dataset...")
    loaders = get_1_to_3_augmentation_loaders(
        root_dir=data_dir,
        batch_size=1,
        num_workers=0,
        image_size=image_size,
        use_weighted_sampling=False,
        cache_images=False,
        validate_data=False,
        include_negative_samples=True,
        deterministic_augmentation=True
    )
    test_loader = loaders['test']

    # Load model
    model = load_model(model_path, device, encoder_name)
    gradcam = UNetGradCAM(model, device)

    results = []
    sample_count = 0

    print(f"🔄 Processing {num_samples} samples...")

    for batch_idx, batch in enumerate(test_loader):
        if sample_count >= num_samples:
            break

        # Get sample info
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Get image name if available
        image_name = f'sample_{sample_count:03d}'
        if 'image_name' in batch:
            image_name = batch['image_name'][0]
        elif 'filename' in batch:
            image_name = Path(batch['filename'][0]).stem

        print(f"\n📸 Processing {sample_count + 1}/{num_samples}: {image_name}")

        # Get prediction
        with torch.no_grad():
            prediction = model(images)

        # Create sample output directory
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_count:03d}_{image_name}')
        os.makedirs(sample_output_dir, exist_ok=True)

        # Compare methods
        method_results = gradcam.compare_methods(
            input_tensor=images,
            target_mask=masks,
            target_layer_name=target_layer,
            methods=methods
        )

        # Analyze each method
        sample_results = {}
        for method, heatmap in method_results.items():
            # Create visualization with ground truth
            fig = gradcam.create_visualization(
                input_tensor=images,
                gradcam_heatmap=heatmap,
                prediction=prediction,
                ground_truth=masks,
                title=f'{method.upper()} - {image_name}'
            )

            viz_path = os.path.join(sample_output_dir, f'{method}_analysis.png')
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Analyze patterns
            stats = gradcam.analyze_attention_patterns(
                input_tensor=images,
                gradcam_heatmap=heatmap,
                prediction=prediction,
                ground_truth=masks
            )

            sample_results[method] = {
                'statistics': stats,
                'visualization_path': viz_path
            }

        # Create method comparison
        comparison_path = os.path.join(sample_output_dir, 'method_comparison.png')
        comp_fig = gradcam.create_method_comparison_plot(
            input_tensor=images,
            method_results=method_results,
            prediction=prediction,
            save_path=comparison_path
        )
        plt.close(comp_fig)

        results.append({
            'sample_id': sample_count,
            'image_name': image_name,
            'results': sample_results,
            'output_dir': sample_output_dir
        })

        sample_count += 1

    # Create batch summary
    summary_path = os.path.join(output_dir, 'batch_gradcam_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Batch Grad-CAM Analysis Summary\n")
        f.write(f"===============================\n\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Target Layer: {target_layer}\n")
        f.write(f"Methods: {', '.join(methods)}\n\n")

        # Aggregate statistics
        for method in methods:
            method_stats = []
            for result in results:
                if method in result['results']:
                    method_stats.append(result['results'][method]['statistics'])

            if method_stats:
                f.write(f"{method.upper()} Average Statistics:\n")
                f.write(f"-" * 30 + "\n")

                avg_attention_mean = np.mean([s['attention_mean'] for s in method_stats])
                avg_attention_ratio = np.mean([s.get('attention_ratio', 0) for s in method_stats])
                avg_pred_ratio = np.mean([s.get('prediction_ratio', 0) for s in method_stats])

                f.write(f"Average Attention Mean: {avg_attention_mean:.6f}\n")
                f.write(f"Average Attention Ratio: {avg_attention_ratio:.4f}\n")
                f.write(f"Average Prediction Ratio: {avg_pred_ratio:.4f}\n\n")

    print(f"\n✅ Batch analysis complete! Results saved to {output_dir}")
    print(f"📊 Summary report: {summary_path}")

    return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Grad-CAM Analysis for U-Net++ Segmentation Model')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                       help='Analysis mode: single image or batch processing')

    # Input arguments
    parser.add_argument('--image_path', type=str,
                       help='Path to single image (required for single mode)')
    parser.add_argument('--data_dir', type=str,
                       help='Path to dataset directory (required for batch mode)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                       help='Directory to save results')

    # Model arguments
    parser.add_argument('--encoder', type=str, default='efficientnet-b5',
                       choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3',
                               'efficientnet-b5', 'densenet121', 'densenet201'],
                       help='Encoder architecture')
    parser.add_argument('--image_size', type=int, default=640,
                       help='Input image size')

    # Grad-CAM arguments
    parser.add_argument('--target_layer', type=str, default='encoder_last',
                       choices=['encoder_last', 'encoder_mid', 'encoder_early', 'decoder_last', 'decoder_mid', 'seg_head'],
                       help='Target layer for Grad-CAM')
    parser.add_argument('--methods', type=str, nargs='+', default=['gradcam', 'gradcam++'],
                       choices=['gradcam', 'gradcam++', 'scorecam', 'layercam'],
                       help='Grad-CAM methods to use')

    # Batch mode arguments
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to analyze in batch mode')

    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for computation')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'single' and not args.image_path:
        parser.error("--image_path is required for single mode")
    if args.mode == 'batch' and not args.data_dir:
        parser.error("--data_dir is required for batch mode")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"🚀 Starting Grad-CAM Analysis")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Methods: {args.methods}")
    print(f"Target Layer: {args.target_layer}")

    try:
        if args.mode == 'single':
            # Single image analysis
            results = run_gradcam_analysis(
                model_path=args.model_path,
                image_path=args.image_path,
                output_dir=args.output_dir,
                encoder_name=args.encoder,
                image_size=args.image_size,
                target_layer=args.target_layer,
                methods=args.methods,
                device=device
            )

            print(f"\n📊 Analysis Results:")
            for method, result in results['results'].items():
                stats = result['statistics']
                print(f"{method.upper()}:")
                print(f"  - Attention Mean: {stats['attention_mean']:.6f}")
                print(f"  - Attention Std: {stats['attention_std']:.6f}")
                if 'attention_ratio' in stats:
                    print(f"  - Attention Ratio: {stats['attention_ratio']:.4f}")
                    print(f"  - Prediction Ratio: {stats['prediction_ratio']:.4f}")

            print(f"\n🎯 Available target layers: {results['target_layers']}")

        else:
            # Batch analysis
            results = batch_gradcam_analysis(
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                encoder_name=args.encoder,
                image_size=args.image_size,
                target_layer=args.target_layer,
                methods=args.methods
            )

            print(f"\n📊 Batch Analysis Complete:")
            print(f"- Processed {len(results)} samples")
            print(f"- Results saved to {args.output_dir}")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
