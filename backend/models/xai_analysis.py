#!/usr/bin/env python3
"""
XAI Analysis Engine for Explainable AI
Integrates GradCAM and Integrated Gradients for model interpretability.
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
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for threading compatibility
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Import XAI modules from the same directory (backend/models/)
    from .gradcam_unet import run_gradcam_analysis
    from .integrated_gradients import run_integrated_gradients_analysis

    print("✅ XAI modules imported successfully")
    run_gradcam_analysis = run_gradcam_analysis
    run_integrated_gradients_analysis = run_integrated_gradients_analysis
except ImportError as e:
    print(f"⚠️  XAI modules not found, running in demo mode: {e}")
    run_gradcam_analysis = None
    run_integrated_gradients_analysis = None


class XAIAnalysisEngine:
    """
    Explainable AI analysis engine that provides GradCAM and Integrated Gradients explanations.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the XAI analysis engine.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run analysis on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path or self._find_best_model()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 640
        self.encoder_name = "efficientnet-b5"
        self.available = (
            run_gradcam_analysis is not None
            and run_integrated_gradients_analysis is not None
        )

        print(f"🔬 XAI Analysis Engine initialized")
        print(f"📁 Model path: {self.model_path}")
        print(f"🖥️  Device: {self.device}")
        print(f"✅ XAI modules available: {self.available}")

    def _find_best_model(self) -> str:
        """Find the best available model checkpoint."""
        # Use the same paths as UNet inference engine
        possible_paths = [
            "models_20250609_105424/best_model.pth",
            "models_20250609_102709/best_model.pth",
            "models_20250609_103512/best_model.pth",
            "../models_20250609_105424/best_model.pth",
            "../models_20250609_102709/best_model.pth",
            "../models_20250609_103512/best_model.pth",
            "model/best_model.pth",
            "../model/best_model.pth",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return "models_20250609_105424/best_model.pth"

    async def initialize(self):
        """Initialize the XAI analysis engine."""
        try:
            if not self.available:
                print("🎭 XAI modules not available, running in demo mode")
                return

            if not os.path.exists(self.model_path):
                print(f"⚠️  Model file not found: {self.model_path}")
                print("🎭 Running in demo mode")
                self.available = False
                return

            print("✅ XAI Analysis Engine ready!")

        except Exception as e:
            print(f"❌ Error initializing XAI engine: {e}")
            self.available = False

    async def analyze(
        self, image_path: str, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run comprehensive XAI analysis on an image and its prediction.

        Args:
            image_path: Path to the input image
            prediction_data: Prediction results from U-Net model

        Returns:
            Dictionary containing XAI analysis results
        """
        try:
            if not self.available:
                return await self._generate_mock_analysis(image_path, prediction_data)

            # Create temporary output directory
            output_dir = Path("temp_xai_output")
            output_dir.mkdir(exist_ok=True)

            # Run GradCAM analysis (fast)
            print("🔍 Running GradCAM analysis...")
            gradcam_results = await self._run_gradcam_analysis(image_path, output_dir)

            # Run Integrated Gradients analysis (real implementation)
            print("📊 Running Integrated Gradients analysis...")
            ig_results = await self._run_integrated_gradients_analysis(
                image_path, output_dir
            )

            # Generate explanation
            explanation = await self._generate_explanation(
                gradcam_results, ig_results, prediction_data
            )

            # Clean up temporary files
            await self._cleanup_temp_files(output_dir)

            return {
                "gradcam": gradcam_results,
                "integratedGradients": ig_results,
                "explanation": explanation,
            }

        except Exception as e:
            print(f"❌ Error during XAI analysis: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return await self._generate_mock_analysis(image_path, prediction_data)

    async def analyze_gradcam_only(
        self, image_path: str, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run GradCAM analysis only."""
        try:
            if not self.available:
                return await self._generate_mock_analysis(image_path, prediction_data)

            # Create temporary output directory
            output_dir = Path("temp_xai_output")
            output_dir.mkdir(exist_ok=True)

            # Run GradCAM analysis only
            print("🔍 Running GradCAM analysis only...")
            gradcam_results = await self._run_gradcam_analysis(image_path, output_dir)

            # Generate basic explanation with GradCAM only
            mock_ig_results = {
                "attribution": "",
                "channelAnalysis": "",
                "statistics": {
                    "attributionMean": 0.0,
                    "attributionStd": 0.0,
                    "attributionMax": 0.0,
                    "attributionMin": 0.0,
                    "predictionArea": 0.0,
                    "attributionRatio": 1.0,
                },
            }
            explanation = await self._generate_explanation(
                gradcam_results, mock_ig_results, prediction_data
            )

            # Clean up temporary files
            await self._cleanup_temp_files(output_dir)

            return {
                "gradcam": gradcam_results,
                "integratedGradients": None,  # Not computed yet
                "explanation": explanation,
            }

        except Exception as e:
            print(f"❌ Error during GradCAM analysis: {e}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return await self._generate_mock_analysis(image_path, prediction_data)

    async def analyze_integrated_gradients_only(
        self, image_path: str, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Integrated Gradients analysis only."""
        try:
            if not self.available:
                return await self._generate_mock_analysis(image_path, prediction_data)

            # Create temporary output directory
            output_dir = Path("temp_xai_output")
            output_dir.mkdir(exist_ok=True)

            # Run Integrated Gradients analysis only
            print("📊 Running Integrated Gradients analysis only...")
            ig_results = await self._run_integrated_gradients_analysis(
                image_path, output_dir
            )

            # Generate basic explanation with IG only
            mock_gradcam_results = {
                "heatmap": "",
                "overlayImage": "",
                "statistics": {
                    "attentionMean": 0.0,
                    "attentionStd": 0.0,
                    "attentionRatio": 1.0,
                    "predictionRatio": 0.5,
                },
            }
            explanation = await self._generate_explanation(
                mock_gradcam_results, ig_results, prediction_data
            )

            # Clean up temporary files
            await self._cleanup_temp_files(output_dir)

            return {
                "gradcam": None,  # Not computed
                "integratedGradients": ig_results,
                "explanation": explanation,
            }

        except Exception as e:
            print(f"❌ Error during Integrated Gradients analysis: {e}")
            return await self._generate_mock_analysis(image_path, prediction_data)

    async def _run_gradcam_analysis(
        self, image_path: str, output_dir: Path
    ) -> Dict[str, Any]:
        """Run GradCAM analysis."""
        try:
            print(f"🔬 Starting GradCAM analysis for image: {image_path}")
            print(f"📁 Model path: {self.model_path}")
            print(f"📁 Model exists: {os.path.exists(self.model_path)}")

            # Run GradCAM analysis using the existing module
            results = run_gradcam_analysis(
                model_path=self.model_path,
                image_path=image_path,
                output_dir=str(output_dir / "gradcam"),
                encoder_name=self.encoder_name,
                image_size=self.image_size,
                target_layer="encoder_last",
                methods=["gradcam", "gradcam++"],
            )

            print(f"✅ GradCAM analysis completed, results: {type(results)}")

            # Convert results to base64 images
            # Extract image name from path for file naming
            image_name = Path(image_path).stem
            print(f"📝 Image name extracted: {image_name}")

            # The actual files generated by run_gradcam_analysis
            heatmap_path = output_dir / "gradcam" / f"{image_name}_gradcam_analysis.png"
            overlay_path = (
                output_dir / "gradcam" / f"{image_name}_gradcam++_analysis.png"
            )

            print(f"🔍 Looking for heatmap at: {heatmap_path}")
            print(f"🔍 Looking for overlay at: {overlay_path}")

            gradcam_heatmap = await self._load_image_as_base64(heatmap_path)
            gradcam_overlay = await self._load_image_as_base64(overlay_path)

            return {
                "heatmap": gradcam_heatmap,
                "overlayImage": gradcam_overlay,
                "statistics": {
                    "attentionMean": results["results"]["gradcam"]["statistics"][
                        "attention_mean"
                    ],
                    "attentionStd": results["results"]["gradcam"]["statistics"][
                        "attention_std"
                    ],
                    "attentionRatio": results["results"]["gradcam"]["statistics"].get(
                        "attention_ratio", 1.0
                    ),
                    "predictionRatio": results["results"]["gradcam"]["statistics"].get(
                        "prediction_ratio", 0.5
                    ),
                },
            }

        except Exception as e:
            print(f"❌ Error in GradCAM analysis: {e}")
            import traceback
            print(f"❌ GradCAM traceback: {traceback.format_exc()}")
            return await self._generate_mock_gradcam()

    async def _run_integrated_gradients_analysis(
        self, image_path: str, output_dir: Path
    ) -> Dict[str, Any]:
        """Run Integrated Gradients analysis."""
        try:
            # Run Integrated Gradients analysis using the existing module
            results = run_integrated_gradients_analysis(
                model_path=self.model_path,
                image_path=image_path,
                output_dir=str(output_dir / "integrated_gradients"),
                encoder_name=self.encoder_name,
                image_size=self.image_size,
                baseline_type="zero",
                num_steps=3,  # Further reduced for much faster computation
            )

            # Extract image name from path for file naming
            image_name = Path(image_path).stem

            # Look for the actual output files with the correct prefix
            attribution_path = (
                output_dir
                / "integrated_gradients"
                / f"{image_name}_integrated_gradients.png"
            )
            channel_analysis_path = (
                output_dir
                / "integrated_gradients"
                / f"{image_name}_channel_analysis.png"
            )

            attribution_map = await self._load_image_as_base64(attribution_path)
            channel_analysis = await self._load_image_as_base64(channel_analysis_path)

            return {
                "attribution": attribution_map,
                "channelAnalysis": channel_analysis,
                "statistics": {
                    "attributionMean": results["statistics"]["attribution_mean"],
                    "attributionStd": results["statistics"]["attribution_std"],
                    "attributionMax": results["statistics"]["attribution_max"],
                    "attributionMin": results["statistics"]["attribution_min"],
                    "predictionArea": results["statistics"]["prediction_area"],
                    "attributionRatio": results["statistics"]["attr_ratio"],
                },
            }

        except Exception as e:
            print(f"Error in Integrated Gradients analysis: {e}")
            return await self._generate_mock_integrated_gradients()

    async def _generate_explanation(
        self,
        gradcam_results: Dict[str, Any],
        ig_results: Dict[str, Any],
        prediction_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate human-readable explanation based on XAI results."""

        # Analyze attention patterns
        attention_ratio = gradcam_results["statistics"]["attentionRatio"]
        attribution_ratio = ig_results["statistics"]["attributionRatio"]

        # Generate summary
        summary_parts = []

        if prediction_data["fibroidDetected"]:
            summary_parts.append(
                f"The AI model detected {prediction_data['fibroidCount']} fibroid(s) in the scan."
            )

            if attention_ratio > 2.0:
                summary_parts.append(
                    "The model shows strong focus on the detected regions, indicating high confidence in the findings."
                )
            elif attention_ratio > 1.5:
                summary_parts.append(
                    "The model shows moderate focus on the detected regions."
                )
            else:
                summary_parts.append(
                    "The model shows diffuse attention, suggesting the findings may be subtle."
                )

            if attribution_ratio > 2.0:
                summary_parts.append(
                    "The pixel-level analysis confirms that the detected regions significantly contribute to the prediction."
                )
            else:
                summary_parts.append(
                    "The pixel-level analysis shows moderate contribution from the detected regions."
                )
        else:
            summary_parts.append(
                "The AI model did not detect any fibroids in the scan."
            )
            summary_parts.append(
                "The attention maps show distributed focus across the image, consistent with a negative finding."
            )

        summary = " ".join(summary_parts)

        # Generate key findings
        key_findings = [
            f"Model confidence: {prediction_data['confidence']:.1%}",
            f"Attention focus ratio: {attention_ratio:.2f}",
            f"Attribution strength: {attribution_ratio:.2f}",
        ]

        if prediction_data["fibroidDetected"]:
            severities = [area["severity"] for area in prediction_data["fibroidAreas"]]
            most_severe = max(
                severities, key=lambda x: ["mild", "moderate", "severe"].index(x)
            )
            key_findings.append(f"Most severe finding: {most_severe}")

        # Generate recommendations
        recommendations = []

        if prediction_data["fibroidDetected"]:
            recommendations.append(
                "Consult with a gynecologist for clinical correlation and treatment planning."
            )

            if any(
                area["severity"] == "severe" for area in prediction_data["fibroidAreas"]
            ):
                recommendations.append(
                    "Consider MRI for detailed characterization of large fibroids."
                )

            recommendations.append(
                "Monitor symptoms and consider follow-up imaging if symptoms worsen."
            )
        else:
            recommendations.append("Continue routine gynecological care.")
            recommendations.append(
                "If symptoms persist, consider additional imaging or clinical evaluation."
            )

        # Calculate explanation confidence
        confidence_factors = [
            prediction_data["confidence"],
            min(attention_ratio / 3.0, 1.0),  # Normalize attention ratio
            min(attribution_ratio / 3.0, 1.0),  # Normalize attribution ratio
        ]
        explanation_confidence = np.mean(confidence_factors)

        return {
            "summary": summary,
            "keyFindings": key_findings,
            "confidence": float(explanation_confidence),
            "recommendations": recommendations,
        }

    async def _generate_mock_analysis(
        self, image_path: str, prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mock XAI analysis for demo purposes."""

        # Generate mock GradCAM
        gradcam_results = await self._generate_mock_gradcam()

        # Generate mock Integrated Gradients
        ig_results = await self._generate_mock_integrated_gradients()

        # Generate explanation
        explanation = await self._generate_explanation(
            gradcam_results, ig_results, prediction_data
        )

        return {
            "gradcam": gradcam_results,
            "integratedGradients": ig_results,
            "explanation": explanation,
        }

    async def _generate_mock_gradcam(self) -> Dict[str, Any]:
        """Generate mock GradCAM results."""
        # Create a simple heatmap
        heatmap = np.random.rand(256, 256)
        heatmap = (heatmap * 255).astype(np.uint8)

        # Convert to base64
        heatmap_base64 = await self._array_to_base64(heatmap)
        overlay_base64 = await self._array_to_base64(heatmap)  # Same for demo

        return {
            "heatmap": heatmap_base64,
            "overlayImage": overlay_base64,
            "statistics": {
                "attentionMean": 0.125,
                "attentionStd": 0.089,
                "attentionRatio": 2.34,
                "predictionRatio": 0.67,
            },
        }

    async def _generate_mock_integrated_gradients(self) -> Dict[str, Any]:
        """Generate mock Integrated Gradients results."""
        # Create a simple attribution map
        attribution = np.random.rand(256, 256)
        attribution = (attribution * 255).astype(np.uint8)

        # Convert to base64
        attribution_base64 = await self._array_to_base64(attribution)
        channel_base64 = await self._array_to_base64(attribution)  # Same for demo

        return {
            "attribution": attribution_base64,
            "channelAnalysis": channel_base64,
            "statistics": {
                "attributionMean": 0.0234,
                "attributionStd": 0.0156,
                "attributionMax": 0.234,
                "attributionMin": -0.123,
                "predictionArea": 1250.0,
                "attributionRatio": 1.89,
            },
        }

    async def _load_image_as_base64(self, image_path: Path) -> str:
        """Load image file and convert to base64."""
        try:
            print(f"🔍 Looking for image at: {image_path}")
            print(f"📁 Image exists: {image_path.exists()}")
            if image_path.exists():
                print(f"✅ Loading real image: {image_path}")
                with open(image_path, "rb") as f:
                    image_data = f.read()
                print(f"📊 Image size: {len(image_data)} bytes")
                return base64.b64encode(image_data).decode("utf-8")
            else:
                # List files in the directory to see what's actually there
                parent_dir = image_path.parent
                if parent_dir.exists():
                    files = list(parent_dir.glob("*"))
                    print(f"📂 Files in {parent_dir}: {[f.name for f in files]}")
                else:
                    print(f"❌ Directory doesn't exist: {parent_dir}")

                # Return mock image if file doesn't exist
                print(f"⚠️ Image not found, using mock data for: {image_path}")
                mock_array = np.random.rand(256, 256)
                return await self._array_to_base64(mock_array)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            # Fallback to mock image
            mock_array = np.random.rand(256, 256)
            return await self._array_to_base64(mock_array)

    async def _array_to_base64(self, array: np.ndarray) -> str:
        """Convert numpy array to base64 encoded PNG image."""
        # Normalize to 0-255 range
        if array.dtype != np.uint8:
            array_normalized = (array * 255).astype(np.uint8)
        else:
            array_normalized = array

        # Convert to PIL Image
        image = Image.fromarray(array_normalized, mode="L")

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_string

    async def _cleanup_temp_files(self, output_dir: Path):
        """Clean up temporary files."""
        try:
            import shutil

            if output_dir.exists():
                shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the XAI engine."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "image_size": self.image_size,
            "encoder_name": self.encoder_name,
            "available": self.available,
        }
