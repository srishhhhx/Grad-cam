import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

import segmentation_models_pytorch as smp
from utils.dataset import get_transforms


def load_model(model_path, device, encoder_name="efficientnet-b5"):
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    if "metrics" in checkpoint:
        print(f"Model validation metrics: {checkpoint['metrics']}")

    return model, checkpoint


def predict_single_image(model, image_path, device, image_size=960, threshold=0.5):
    """Predict on a single image"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]

    # Resize image
    image_resized = cv2.resize(image, (image_size, image_size))

    # Apply transforms
    transform = get_transforms(image_size=image_size, is_training=False)
    transformed = transform(
        image=image_resized, mask=np.zeros((image_size, image_size))
    )
    input_tensor = transformed["image"].unsqueeze(0).to(device)

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


def create_overlay(original_image, mask, alpha=0.5):
    """Create an overlay of the mask on the original image"""
    # Create a colored mask (red for detected regions)
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :, 0] = mask * 255  # Red channel

    # Blend the original image with the colored mask
    overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)

    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Predict and overlay segmentation mask on image"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to input image (if not provided, uses first test image)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prediction_overlay_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Prediction threshold"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3, help="Overlay transparency (0-1)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Input image size for model (must match training!)",
    )
    parser.add_argument(
        "--encoder", type=str, default="efficientnet-b5", help="Encoder architecture"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If no image path provided, use the first test image
    if args.image_path is None:
        test_dir = "Dataset_no_preprocessing/test"
        test_images = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
        if not test_images:
            print("No test images found!")
            return
        args.image_path = os.path.join(test_dir, test_images[0])
        print(f"Using test image: {args.image_path}")

    # Load model
    model, checkpoint = load_model(args.model_path, device, args.encoder)

    # Predict on image
    print(f"Predicting on image: {args.image_path}")
    prediction, binary_mask, original_image = predict_single_image(
        model, args.image_path, device, args.image_size, args.threshold
    )

    # Create overlay
    overlay = create_overlay(original_image, binary_mask, args.alpha)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Segmentation Results for {os.path.basename(args.image_path)}", fontsize=16
    )

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Prediction probability map
    im1 = axes[0, 1].imshow(prediction, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title("Prediction Probability Map")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Binary mask
    axes[1, 0].imshow(binary_mask, cmap="gray")
    axes[1, 0].set_title(f"Binary Mask (threshold={args.threshold})")
    axes[1, 0].axis("off")

    # Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Overlay (alpha={args.alpha})")
    axes[1, 1].axis("off")

    plt.tight_layout()

    # Save the result
    output_path = os.path.join(
        args.output_dir, f"prediction_overlay_{os.path.basename(args.image_path)}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Results saved to: {output_path}")

    # Also save individual images
    cv2.imwrite(
        os.path.join(args.output_dir, f"original_{os.path.basename(args.image_path)}"),
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(args.output_dir, f"overlay_{os.path.basename(args.image_path)}"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(args.output_dir, f"mask_{os.path.basename(args.image_path)}"),
        binary_mask * 255,
    )

    # Display the plot
    plt.show()

    # Print some statistics
    mask_area = np.sum(binary_mask)
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    coverage_percentage = (mask_area / total_area) * 100

    print(f"\nPrediction Statistics:")
    print(f"Image size: {original_image.shape[:2]}")
    print(f"Detected area: {mask_area} pixels")
    print(f"Total area: {total_area} pixels")
    print(f"Coverage: {coverage_percentage:.2f}%")
    print(f"Max prediction confidence: {np.max(prediction):.3f}")
    print(f"Mean prediction confidence: {np.mean(prediction):.3f}")


if __name__ == "__main__":
    main()
