import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model and data loading utilities
import segmentation_models_pytorch as smp
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "UNET-model"))
from utils.enhanced_dataloader import get_1_to_3_augmentation_loaders
import albumentations as A
from albumentations.pytorch import ToTensorV2


class IntegratedGradients:
    """
    Integrated Gradients implementation for segmentation models.
    
    This class implements the Integrated Gradients method for explaining
    predictions of deep neural networks, specifically adapted for segmentation tasks.
    
    Reference: Sundararajan, M., Taly, A., & Yan, Q. (2017). 
    Axiomatic attribution for deep networks. ICML.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize Integrated Gradients explainer.
        
        Args:
            model: The trained segmentation model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def _get_gradients(self, inputs: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Compute gradients of the model output with respect to inputs.
        
        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            target_class: Target class for gradient computation (None for segmentation)
            
        Returns:
            Gradients tensor of same shape as inputs
        """
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # For segmentation, we typically want gradients w.r.t. the entire output
        if target_class is None:
            # Sum over all spatial locations and apply sigmoid
            score = torch.sigmoid(outputs).sum()
        else:
            # For specific class (if needed)
            score = outputs[:, target_class].sum()
        
        # Backward pass
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=inputs,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return gradients
    
    def _generate_baseline(self, inputs: torch.Tensor, baseline_type: str = 'zero') -> torch.Tensor:
        """
        Generate baseline for integrated gradients computation.
        
        Args:
            inputs: Input tensor
            baseline_type: Type of baseline ('zero', 'random', 'blur', 'mean')
            
        Returns:
            Baseline tensor of same shape as inputs
        """
        if baseline_type == 'zero':
            return torch.zeros_like(inputs)
        elif baseline_type == 'random':
            return torch.randn_like(inputs) * 0.1
        elif baseline_type == 'blur':
            # Apply Gaussian blur as baseline
            baseline = inputs.clone()
            for i in range(baseline.shape[0]):
                for c in range(baseline.shape[1]):
                    img_np = baseline[i, c].cpu().numpy()
                    blurred = cv2.GaussianBlur(img_np, (15, 15), 0)
                    baseline[i, c] = torch.from_numpy(blurred)
            return baseline
        elif baseline_type == 'mean':
            # Use mean pixel values as baseline
            mean_vals = inputs.mean(dim=(2, 3), keepdim=True)
            return mean_vals.expand_as(inputs)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    def compute_integrated_gradients(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
        baseline_type: str = 'zero',
        num_steps: int = 50,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            target_class: Target class for attribution (None for segmentation)
            baseline_type: Type of baseline to use
            num_steps: Number of integration steps
            batch_size: Batch size for gradient computation
            
        Returns:
            Attribution tensor of same shape as inputs
        """
        # Generate baseline
        baseline = self._generate_baseline(inputs, baseline_type)
        
        # Compute the difference between input and baseline
        input_diff = inputs - baseline
        
        # Initialize attributions
        attributions = torch.zeros_like(inputs)
        
        # Compute gradients for interpolated inputs
        for i in tqdm(range(0, num_steps, batch_size), desc="Computing Integrated Gradients"):
            # Create batch of interpolated inputs
            batch_end = min(i + batch_size, num_steps)
            current_batch_size = batch_end - i
            
            # Generate alpha values for this batch
            alphas = torch.linspace(i / num_steps, batch_end / num_steps, current_batch_size + 1)[:-1]
            alphas = alphas.view(-1, 1, 1, 1).to(self.device)
            
            # Create interpolated inputs
            interpolated_inputs = baseline.unsqueeze(0) + alphas * input_diff.unsqueeze(0)
            interpolated_inputs = interpolated_inputs.view(-1, *inputs.shape[1:])
            
            # Compute gradients
            gradients = self._get_gradients(interpolated_inputs, target_class)
            
            # Reshape gradients back to batch format
            gradients = gradients.view(current_batch_size, *inputs.shape)
            
            # Accumulate attributions (Riemann sum approximation)
            attributions += gradients.sum(dim=0) / num_steps
        
        # Multiply by input difference to get final attributions
        attributions = attributions * input_diff
        
        return attributions


class SegmentationModelWrapper:
    """
    Wrapper for segmentation models to handle Integrated Gradients computation.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """
        Initialize model wrapper.
        
        Args:
            model: The segmentation model
            target_layer: Specific layer to analyze (None for final output)
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = {}
        self.gradients = {}
        
        if target_layer:
            self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layer."""
        def forward_hook(module, input_tensor, output):
            self.activations[self.target_layer] = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients[self.target_layer] = grad_output[0]
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)


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


def preprocess_image(image_path: str, image_size: int = 640) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model input.

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


def create_attribution_visualization(
    original_image: np.ndarray,
    attributions: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Integrated Gradients Attribution"
) -> plt.Figure:
    """
    Create comprehensive visualization of Integrated Gradients attributions.

    Args:
        original_image: Original input image (H, W, 3)
        attributions: Attribution tensor (C, H, W)
        prediction: Model prediction (1, H, W)
        ground_truth: Ground truth mask (H, W) - optional
        save_path: Path to save visualization
        title: Title for the plot

    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = torch.sigmoid(prediction).cpu().numpy()

    # Resize original image to match attribution size
    attr_h, attr_w = attributions.shape[-2:]
    original_resized = cv2.resize(original_image, (attr_w, attr_h))

    # Compute attribution magnitude across channels
    attribution_magnitude = np.sqrt(np.sum(attributions**2, axis=0))

    # Normalize attribution for visualization
    attr_norm = (attribution_magnitude - attribution_magnitude.min()) / \
                (attribution_magnitude.max() - attribution_magnitude.min() + 1e-8)

    # Create figure
    n_cols = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

    # Row 1: Original image, Attribution heatmap, Prediction
    axes[0, 0].imshow(original_resized)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Attribution heatmap
    im1 = axes[0, 1].imshow(attr_norm, cmap='hot', alpha=0.8)
    axes[0, 1].set_title('Attribution Magnitude')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Prediction
    axes[0, 2].imshow(prediction[0], cmap='gray')
    axes[0, 2].set_title('Model Prediction')
    axes[0, 2].axis('off')

    # Ground truth (if available)
    if ground_truth is not None:
        axes[0, 3].imshow(ground_truth, cmap='gray')
        axes[0, 3].set_title('Ground Truth')
        axes[0, 3].axis('off')

    # Row 2: Overlays
    # Attribution overlay on original image
    axes[1, 0].imshow(original_resized)
    axes[1, 0].imshow(attr_norm, cmap='hot', alpha=0.4)
    axes[1, 0].set_title('Attribution Overlay')
    axes[1, 0].axis('off')

    # Prediction overlay on original image
    axes[1, 1].imshow(original_resized)
    pred_mask = prediction[0] > 0.5
    axes[1, 1].imshow(pred_mask, cmap='Reds', alpha=0.4)
    axes[1, 1].set_title('Prediction Overlay')
    axes[1, 1].axis('off')

    # Combined attribution and prediction
    axes[1, 2].imshow(original_resized)
    axes[1, 2].imshow(attr_norm, cmap='hot', alpha=0.3)
    axes[1, 2].imshow(pred_mask, cmap='Blues', alpha=0.3)
    axes[1, 2].set_title('Combined Overlay')
    axes[1, 2].axis('off')

    # Attribution vs Prediction comparison (if ground truth available)
    if ground_truth is not None:
        gt_mask = ground_truth > 0.5
        axes[1, 3].imshow(original_resized)
        axes[1, 3].imshow(gt_mask, cmap='Greens', alpha=0.4)
        axes[1, 3].set_title('Ground Truth Overlay')
        axes[1, 3].axis('off')

    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    return fig


def analyze_attribution_statistics(attributions: torch.Tensor, prediction: torch.Tensor) -> dict:
    """
    Analyze attribution statistics and relationship with predictions.

    Args:
        attributions: Attribution tensor (C, H, W)
        prediction: Model prediction (1, H, W)

    Returns:
        Dictionary with analysis results
    """
    # Convert to numpy
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = torch.sigmoid(prediction).cpu().numpy()

    # Compute attribution magnitude
    attr_magnitude = np.sqrt(np.sum(attributions**2, axis=0))

    # Get prediction mask
    pred_mask = prediction[0] > 0.5

    # Statistics
    stats = {
        'attribution_mean': float(attr_magnitude.mean()),
        'attribution_std': float(attr_magnitude.std()),
        'attribution_max': float(attr_magnitude.max()),
        'attribution_min': float(attr_magnitude.min()),
        'prediction_area': float(pred_mask.sum()),
        'total_pixels': int(pred_mask.size),
        'prediction_ratio': float(pred_mask.sum() / pred_mask.size),
    }

    # Attribution in predicted regions vs background
    if pred_mask.sum() > 0:
        stats['attr_in_prediction'] = float(attr_magnitude[pred_mask].mean())
        stats['attr_in_background'] = float(attr_magnitude[~pred_mask].mean())
        stats['attr_ratio'] = stats['attr_in_prediction'] / (stats['attr_in_background'] + 1e-8)
    else:
        stats['attr_in_prediction'] = 0.0
        stats['attr_in_background'] = float(attr_magnitude.mean())
        stats['attr_ratio'] = 0.0

    return stats


def create_channel_wise_attribution_plot(
    attributions: torch.Tensor,
    channel_names: List[str] = ['Red', 'Green', 'Blue'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create channel-wise attribution visualization.

    Args:
        attributions: Attribution tensor (C, H, W)
        channel_names: Names for each channel
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()

    n_channels = attributions.shape[0]
    fig, axes = plt.subplots(2, n_channels, figsize=(4*n_channels, 8))

    for i in range(n_channels):
        # Raw attribution
        im1 = axes[0, i].imshow(attributions[i], cmap='RdBu_r',
                               vmin=-np.abs(attributions[i]).max(),
                               vmax=np.abs(attributions[i]).max())
        axes[0, i].set_title(f'{channel_names[i]} Channel Attribution')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Absolute attribution
        im2 = axes[1, i].imshow(np.abs(attributions[i]), cmap='hot')
        axes[1, i].set_title(f'{channel_names[i]} Magnitude')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.suptitle('Channel-wise Attribution Analysis', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Channel-wise plot saved to {save_path}")

    return fig


def run_integrated_gradients_analysis(
    model_path: str,
    image_path: str,
    output_dir: str,
    encoder_name: str = 'efficientnet-b5',
    image_size: int = 640,
    baseline_type: str = 'zero',
    num_steps: int = 50,
    device: Optional[torch.device] = None
) -> dict:
    """
    Run complete Integrated Gradients analysis on a single image.

    Args:
        model_path: Path to trained model checkpoint
        image_path: Path to input image
        output_dir: Directory to save results
        encoder_name: Model encoder architecture
        image_size: Input image size
        baseline_type: Type of baseline for IG computation
        num_steps: Number of integration steps
        device: Device to run on

    Returns:
        Dictionary with analysis results
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    print(f"🔬 Running Integrated Gradients Analysis")
    print(f"📁 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️  Device: {device}")

    # Load model
    model = load_model(model_path, device, encoder_name)

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model, device)

    # Preprocess image
    print("🔄 Preprocessing image...")
    input_tensor, original_image = preprocess_image(image_path, image_size)
    input_tensor = input_tensor.to(device)

    # Get model prediction
    print("🔮 Getting model prediction...")
    with torch.no_grad():
        prediction = model(input_tensor)

    # Compute Integrated Gradients
    print(f"🧮 Computing Integrated Gradients (steps={num_steps}, baseline={baseline_type})...")
    attributions = ig.compute_integrated_gradients(
        input_tensor,
        baseline_type=baseline_type,
        num_steps=num_steps,
        batch_size=8
    )

    # Remove batch dimension
    attributions = attributions.squeeze(0)
    prediction = prediction.squeeze(0)

    # Analyze statistics
    print("📊 Analyzing attribution statistics...")
    stats = analyze_attribution_statistics(attributions, prediction)

    # Create visualizations
    print("🎨 Creating visualizations...")

    # Main attribution visualization
    image_name = Path(image_path).stem
    main_viz_path = os.path.join(output_dir, f'{image_name}_integrated_gradients.png')
    fig1 = create_attribution_visualization(
        original_image, attributions, prediction,
        save_path=main_viz_path,
        title=f'Integrated Gradients Analysis - {image_name}'
    )
    plt.close(fig1)

    # Channel-wise analysis
    channel_viz_path = os.path.join(output_dir, f'{image_name}_channel_analysis.png')
    fig2 = create_channel_wise_attribution_plot(
        attributions,
        save_path=channel_viz_path
    )
    plt.close(fig2)

    # Save statistics
    stats_path = os.path.join(output_dir, f'{image_name}_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Integrated Gradients Analysis Results\n")
        f.write(f"=====================================\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Baseline: {baseline_type}\n")
        f.write(f"Integration Steps: {num_steps}\n\n")
        f.write(f"Attribution Statistics:\n")
        f.write(f"- Mean: {stats['attribution_mean']:.6f}\n")
        f.write(f"- Std: {stats['attribution_std']:.6f}\n")
        f.write(f"- Max: {stats['attribution_max']:.6f}\n")
        f.write(f"- Min: {stats['attribution_min']:.6f}\n\n")
        f.write(f"Prediction Statistics:\n")
        f.write(f"- Predicted Area: {stats['prediction_area']:.0f} pixels\n")
        f.write(f"- Total Pixels: {stats['total_pixels']}\n")
        f.write(f"- Prediction Ratio: {stats['prediction_ratio']:.4f}\n\n")
        f.write(f"Attribution Analysis:\n")
        f.write(f"- Attribution in Prediction: {stats['attr_in_prediction']:.6f}\n")
        f.write(f"- Attribution in Background: {stats['attr_in_background']:.6f}\n")
        f.write(f"- Attribution Ratio: {stats['attr_ratio']:.4f}\n")

    print(f"✅ Analysis complete! Results saved to {output_dir}")

    return {
        'attributions': attributions,
        'prediction': prediction,
        'statistics': stats,
        'visualizations': {
            'main': main_viz_path,
            'channels': channel_viz_path,
            'stats': stats_path
        }
    }


def batch_integrated_gradients_analysis(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_samples: int = 10,
    encoder_name: str = 'efficientnet-b5',
    image_size: int = 640,
    baseline_type: str = 'zero',
    num_steps: int = 50
) -> List[dict]:
    """
    Run Integrated Gradients analysis on multiple test images.

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to dataset directory
        output_dir: Directory to save results
        num_samples: Number of test samples to analyze
        encoder_name: Model encoder architecture
        image_size: Input image size
        baseline_type: Type of baseline for IG computation
        num_steps: Number of integration steps

    Returns:
        List of analysis results for each image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🔬 Running Batch Integrated Gradients Analysis")
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
    ig = IntegratedGradients(model, device)

    results = []
    sample_count = 0

    print(f"🔄 Processing {num_samples} samples...")

    for batch_idx, (images, masks, metadata) in enumerate(test_loader):
        if sample_count >= num_samples:
            break

        # Get sample info
        image_name = metadata['image_name'][0] if 'image_name' in metadata else f'sample_{batch_idx}'
        print(f"\n📸 Processing {sample_count + 1}/{num_samples}: {image_name}")

        # Move to device
        images = images.to(device)
        masks = masks.to(device)

        # Get prediction
        with torch.no_grad():
            prediction = model(images)

        # Compute Integrated Gradients
        attributions = ig.compute_integrated_gradients(
            images,
            baseline_type=baseline_type,
            num_steps=num_steps,
            batch_size=8
        )

        # Remove batch dimension
        attributions = attributions.squeeze(0)
        prediction = prediction.squeeze(0)
        masks = masks.squeeze(0)

        # Convert image for visualization
        image_np = images.squeeze(0).cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # Create sample output directory
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_count:03d}_{image_name}')
        os.makedirs(sample_output_dir, exist_ok=True)

        # Analyze statistics
        stats = analyze_attribution_statistics(attributions, prediction)

        # Create visualizations with ground truth
        main_viz_path = os.path.join(sample_output_dir, 'integrated_gradients.png')
        fig1 = create_attribution_visualization(
            image_np, attributions, prediction,
            ground_truth=masks.cpu().numpy(),
            save_path=main_viz_path,
            title=f'Integrated Gradients - {image_name}'
        )
        plt.close(fig1)

        # Channel-wise analysis
        channel_viz_path = os.path.join(sample_output_dir, 'channel_analysis.png')
        fig2 = create_channel_wise_attribution_plot(
            attributions,
            save_path=channel_viz_path
        )
        plt.close(fig2)

        # Save results
        result = {
            'sample_id': sample_count,
            'image_name': image_name,
            'statistics': stats,
            'output_dir': sample_output_dir
        }
        results.append(result)

        sample_count += 1

    # Create summary report
    summary_path = os.path.join(output_dir, 'batch_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Batch Integrated Gradients Analysis Summary\n")
        f.write(f"==========================================\n\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Baseline: {baseline_type}\n")
        f.write(f"Integration Steps: {num_steps}\n\n")

        # Aggregate statistics
        all_stats = [r['statistics'] for r in results]
        avg_stats = {}
        for key in all_stats[0].keys():
            if isinstance(all_stats[0][key], (int, float)):
                avg_stats[key] = np.mean([s[key] for s in all_stats])

        f.write(f"Average Statistics Across All Samples:\n")
        for key, value in avg_stats.items():
            f.write(f"- {key}: {value:.6f}\n")

    print(f"\n✅ Batch analysis complete! Results saved to {output_dir}")
    print(f"📊 Summary report: {summary_path}")

    return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Integrated Gradients Analysis for U-Net++ Segmentation Model')

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
    parser.add_argument('--output_dir', type=str, default='integrated_gradients_results',
                       help='Directory to save results')

    # Model arguments
    parser.add_argument('--encoder', type=str, default='efficientnet-b5',
                       choices=['resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b3',
                               'efficientnet-b5', 'densenet121', 'densenet201'],
                       help='Encoder architecture')
    parser.add_argument('--image_size', type=int, default=640,
                       help='Input image size')

    # Integrated Gradients arguments
    parser.add_argument('--baseline', type=str, default='zero',
                       choices=['zero', 'random', 'blur', 'mean'],
                       help='Baseline type for Integrated Gradients')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of integration steps')

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

    print(f"🚀 Starting Integrated Gradients Analysis")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")

    try:
        if args.mode == 'single':
            # Single image analysis
            results = run_integrated_gradients_analysis(
                model_path=args.model_path,
                image_path=args.image_path,
                output_dir=args.output_dir,
                encoder_name=args.encoder,
                image_size=args.image_size,
                baseline_type=args.baseline,
                num_steps=args.num_steps,
                device=device
            )

            print(f"\n📊 Analysis Results:")
            print(f"- Attribution Mean: {results['statistics']['attribution_mean']:.6f}")
            print(f"- Attribution Std: {results['statistics']['attribution_std']:.6f}")
            print(f"- Prediction Ratio: {results['statistics']['prediction_ratio']:.4f}")
            print(f"- Attribution Ratio: {results['statistics']['attr_ratio']:.4f}")

        else:
            # Batch analysis
            results = batch_integrated_gradients_analysis(
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                encoder_name=args.encoder,
                image_size=args.image_size,
                baseline_type=args.baseline,
                num_steps=args.num_steps
            )

            print(f"\n📊 Batch Analysis Complete:")
            print(f"- Processed {len(results)} samples")
            print(f"- Results saved to {args.output_dir}")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
