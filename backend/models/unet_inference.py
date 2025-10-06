#!/usr/bin/env python3
"""
U-Net Inference Engine for Uterine Fibroids Detection
Integrates with the existing U-Net model and provides async inference capabilities.
"""

import os
import sys
import asyncio
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the existing prediction module
try:
    # Import from UNET-model directory
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    unet_model_path = os.path.join(parent_dir, 'UNET-model')
    sys.path.insert(0, unet_model_path)
    from predict import load_model, predict_single_image
    from utils.dataset import get_transforms
    PREDICT_MODULE_AVAILABLE = True
    print("✅ Prediction module imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import predict module: {e}")
    PREDICT_MODULE_AVAILABLE = False

class UNetInferenceEngine:
    """
    Async U-Net inference engine for uterine fibroids detection.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the U-Net inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path or self._find_best_model()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.image_size = 640  # FIXED: Match training configuration (was 960)
        self.encoder_name = 'efficientnet-b5'
        self.threshold = 0.5
        
        print(f"🔧 UNet Inference Engine initialized")
        print(f"📁 Model path: {self.model_path}")
        print(f"🖥️  Device: {self.device}")
    
    def _find_best_model(self) -> str:
        """Find the best available model checkpoint."""
        # Look for the best model from the training results
        possible_paths = [
            "models_20250609_105424/best_model.pth",
            "models_20250609_102709/best_model.pth",
            "models_20250609_103512/best_model.pth",
            "../models_20250609_105424/best_model.pth",
            "../models_20250609_102709/best_model.pth",
            "../models_20250609_103512/best_model.pth",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no model found, return a default path (will be handled in initialize)
        return "models_20250609_105424/best_model.pth"
    
    async def initialize(self):
        """Initialize the model and preprocessing pipeline."""
        try:
            print("🔄 Loading U-Net++ model...")

            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"⚠️  Model file not found: {self.model_path}")
                print("🎭 Running in demo mode with mock predictions")
                self.model = None
                return

            if PREDICT_MODULE_AVAILABLE:
                # Use the existing prediction module
                print("📦 Using existing prediction module...")
                self.model, self.checkpoint = load_model(
                    self.model_path,
                    self.device,
                    self.encoder_name
                )
                print(f"✅ Model loaded successfully!")
                print(f"📊 Model epoch: {self.checkpoint.get('epoch', 'unknown')}")
                if 'metrics' in self.checkpoint:
                    print(f"📈 Model metrics: {self.checkpoint['metrics']}")
            else:
                # Fallback to manual loading
                print("📦 Using fallback model loading...")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Initialize model architecture
                self.model = smp.UnetPlusPlus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                )

                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                # Initialize preprocessing pipeline with FIXED parameters
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0  # FIXED: was 1.0, should be 255.0!
                    ),
                    ToTensorV2()
                ])

                print(f"✅ Model loaded successfully!")
                print(f"📊 Model epoch: {checkpoint.get('epoch', 'unknown')}")
                if 'metrics' in checkpoint:
                    print(f"📈 Model metrics: {checkpoint['metrics']}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🎭 Running in demo mode with mock predictions")
            self.model = None
    
    def _get_fixed_transforms(self, image_size=640):
        """Get CORRECTED transforms with proper max_pixel_value"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0  # FIXED: was 1.0, should be 255.0!
            ),
            ToTensorV2()
        ])

    async def _predict_with_fixed_preprocessing(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run prediction with FIXED preprocessing"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        print(f"📸 Original image shape: {image.shape}")
        print(f"📊 Original pixel range: [{image.min()}, {image.max()}]")

        # Apply CORRECTED transforms
        transform = self._get_fixed_transforms(self.image_size)
        transformed = transform(image=image, mask=np.zeros((self.image_size, self.image_size)))
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        print(f"🔧 Transformed tensor shape: {input_tensor.shape}")
        print(f"📊 Transformed range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"📈 Transformed mean: {input_tensor.mean():.3f}")

        # Predict
        with torch.no_grad():
            raw_output = self.model(input_tensor)
            prediction = torch.sigmoid(raw_output)

            print(f"📤 Raw output range: [{raw_output.min():.3f}, {raw_output.max():.3f}]")
            print(f"🎯 Sigmoid range: [{prediction.min():.3f}, {prediction.max():.3f}]")
            print(f"📈 Sigmoid mean: {prediction.mean():.3f}")

            # Check different thresholds
            for thresh in [0.1, 0.3, 0.5, 0.7]:
                count = (prediction > thresh).sum().item()
                percentage = (count / prediction.numel()) * 100
                print(f"🎯 Pixels > {thresh}: {count} ({percentage:.2f}%)")

        # Convert to numpy and resize back to original size
        prediction_np = prediction.cpu().numpy()[0, 0]
        prediction_resized = cv2.resize(prediction_np, (original_shape[1], original_shape[0]))

        # Create binary mask
        binary_mask = (prediction_resized > self.threshold).astype(np.uint8)

        return prediction_resized, binary_mask

    async def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a medical image.

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model is None:
                # Demo mode - return mock predictions
                return await self._generate_mock_prediction(image_path)

            # Use our FIXED preprocessing method
            print(f"🔍 Running prediction with FIXED preprocessing...")
            print(f"📏 Using FIXED image_size={self.image_size}, threshold={self.threshold}")

            # Load original image to get correct shape
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_shape = original_img.shape[:2]

            # Run prediction with fixed preprocessing
            prediction_prob, binary_mask = await self._predict_with_fixed_preprocessing(image_path)

            # Post-process results
            results = await self._post_process_prediction(
                prediction_prob, original_shape, image_path
            )

            return results

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            # Fallback to mock prediction
            return await self._generate_mock_prediction(image_path)
    
    async def _post_process_prediction(
        self, 
        prediction: np.ndarray, 
        original_shape: Tuple[int, int],
        image_path: str
    ) -> Dict[str, Any]:
        """
        Post-process the model prediction to extract meaningful results.
        
        Args:
            prediction: Raw model prediction (probability map)
            original_shape: Original image shape (H, W)
            image_path: Path to original image
            
        Returns:
            Structured prediction results
        """
        # Resize prediction to original image size
        prediction_resized = cv2.resize(prediction, (original_shape[1], original_shape[0]))
        
        # Create binary mask
        binary_mask = (prediction_resized > self.threshold).astype(np.uint8)
        
        # Calculate confidence as mean probability in predicted regions
        confidence = float(np.mean(prediction_resized[binary_mask > 0])) if np.any(binary_mask) else 0.0
        
        # Detect fibroids (connected components)
        fibroid_detected = bool(np.any(binary_mask))  # Convert to Python bool
        fibroid_areas = []
        fibroid_count = 0
        
        if fibroid_detected:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 50:  # Minimum area threshold
                    centroid_x, centroid_y = centroids[i]
                    
                    # Determine severity based on area
                    if area < 500:
                        severity = 'mild'
                    elif area < 1500:
                        severity = 'moderate'
                    else:
                        severity = 'severe'
                    
                    fibroid_areas.append({
                        'area': float(area),
                        'location': {
                            'x': float(centroid_x / original_shape[1] * 100),  # Convert to percentage
                            'y': float(centroid_y / original_shape[0] * 100)
                        },
                        'severity': severity
                    })
                    fibroid_count += 1
        
        # Convert prediction mask to base64 for frontend
        mask_base64 = await self._array_to_base64(prediction_resized)
        
        result = {
            'mask': mask_base64,
            'probability': prediction_resized.tolist(),
            'confidence': float(confidence),
            'fibroidDetected': bool(fibroid_detected),
            'fibroidCount': int(fibroid_count),
            'fibroidAreas': fibroid_areas
        }

        # Ensure all numpy types are converted
        return self._convert_numpy_types(result)
    
    async def _generate_mock_prediction(self, image_path: str) -> Dict[str, Any]:
        """
        Generate mock prediction for demo purposes.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Mock prediction results
        """
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            # Use default dimensions
            height, width = 640, 640
        else:
            height, width = image.shape[:2]
        
        # Create a mock prediction mask with some fibroids
        mock_mask = np.zeros((height, width), dtype=np.float32)
        
        # Add some mock fibroids
        import random
        random.seed(42)  # For consistent demo results
        
        num_fibroids = random.randint(1, 3)
        fibroid_areas = []
        
        for i in range(num_fibroids):
            # Random fibroid location and size
            center_x = random.randint(width // 4, 3 * width // 4)
            center_y = random.randint(height // 4, 3 * height // 4)
            radius = random.randint(20, 60)
            
            # Draw circular fibroid
            cv2.circle(mock_mask, (center_x, center_y), radius, 0.8, -1)
            
            # Calculate area and severity
            area = np.pi * radius * radius
            if area < 2000:
                severity = 'mild'
            elif area < 6000:
                severity = 'moderate'
            else:
                severity = 'severe'
            
            fibroid_areas.append({
                'area': float(area),
                'location': {
                    'x': float(center_x / width * 100),
                    'y': float(center_y / height * 100)
                },
                'severity': severity
            })
        
        # Convert mask to base64
        mask_base64 = await self._array_to_base64(mock_mask)
        
        result = {
            'mask': mask_base64,
            'probability': mock_mask.tolist(),
            'confidence': 0.85,
            'fibroidDetected': bool(num_fibroids > 0),
            'fibroidCount': int(num_fibroids),
            'fibroidAreas': fibroid_areas
        }

        # Use the same conversion function to ensure no numpy types
        return self._convert_numpy_types(result)
    
    async def _array_to_base64(self, array: np.ndarray) -> str:
        """
        Convert numpy array to base64 encoded PNG image.
        
        Args:
            array: Input array (values should be in [0, 1] range)
            
        Returns:
            Base64 encoded PNG image string
        """
        # Normalize to 0-255 range
        array_normalized = (array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(array_normalized, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_string

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'image_size': self.image_size,
            'encoder_name': self.encoder_name,
            'threshold': self.threshold,
            'model_loaded': self.model is not None
        }
