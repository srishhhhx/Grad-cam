#!/usr/bin/env python3
"""
Test script to isolate GradCAM issues
"""

import os
import sys
from pathlib import Path

# Add the models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

def test_gradcam():
    """Test GradCAM analysis in isolation"""
    
    print("🧪 Testing GradCAM Analysis...")
    
    # Test 1: Check if we can import the modules
    try:
        from gradcam_unet import run_gradcam_analysis
        print("✅ GradCAM module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GradCAM module: {e}")
        return False
    
    # Test 2: Check if model file exists
    model_path = "../models_20250609_105424/best_model.pth"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Test 3: Check if we have a test image
    test_image_dir = "../DEMO_ULTRASOUND_IMAGES"
    if os.path.exists(test_image_dir):
        test_images = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
        if test_images:
            test_image = os.path.join(test_image_dir, test_images[0])
            print(f"✅ Test image found: {test_image}")
        else:
            print("❌ No test images found")
            return False
    else:
        print("❌ Test image directory not found")
        return False
    
    # Test 4: Try to run GradCAM analysis
    try:
        print("🔬 Running GradCAM analysis...")
        output_dir = "test_gradcam_output"
        os.makedirs(output_dir, exist_ok=True)
        
        results = run_gradcam_analysis(
            model_path=model_path,
            image_path=test_image,
            output_dir=output_dir,
            encoder_name="efficientnet-b5",
            image_size=640,
            target_layer="encoder_last",
            methods=["gradcam"],
        )
        
        print(f"✅ GradCAM analysis completed!")
        print(f"📊 Results type: {type(results)}")
        print(f"📊 Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        print(f"❌ GradCAM analysis failed: {e}")
        import traceback
        print(f"❌ Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_gradcam()
    if success:
        print("\n🎉 GradCAM test passed!")
    else:
        print("\n💥 GradCAM test failed!")
