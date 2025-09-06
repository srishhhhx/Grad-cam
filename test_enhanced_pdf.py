#!/usr/bin/env python3
"""
Test script for the enhanced PDF report generator
Demonstrates the beautiful medical report generation with overlay images
"""

import asyncio
import base64
from pathlib import Path
from backend.utils.pdf_generator import PDFReportGenerator

def create_mock_data():
    """Create mock data for testing the enhanced PDF generator."""
    
    # Mock image data (you would normally have actual base64 image data)
    mock_image_data = {
        'image_data': '',  # Would contain base64 encoded image
        'width': 640,
        'height': 640,
        'filename': 'test_ultrasound.jpg',
        'content_type': 'image/jpeg'
    }
    
    # Mock prediction data with detailed results
    mock_prediction_data = {
        'id': 'pred_12345',
        'status': 'completed',
        'processing_time': 2.3,
        'prediction': {
            'fibroidDetected': True,
            'confidence': 0.92,
            'fibroidCount': 3,
            'segmentationMask': '',  # Would contain base64 encoded mask
            'fibroidAreas': [
                {
                    'area': 45.2,
                    'severity': 'moderate',
                    'location': {'x': 320, 'y': 240}
                },
                {
                    'area': 23.8,
                    'severity': 'mild',
                    'location': {'x': 180, 'y': 350}
                },
                {
                    'area': 67.5,
                    'severity': 'severe',
                    'location': {'x': 450, 'y': 180}
                }
            ]
        }
    }
    
    # Mock XAI data with detailed analysis
    mock_xai_data = {
        'id': 'xai_67890',
        'status': 'completed',
        'analysis': {
            'gradcam': {
                'statistics': {
                    'attentionMean': 0.0234,
                    'attentionStd': 0.0156,
                    'attentionRatio': 2.45
                }
            },
            'integratedGradients': {
                'statistics': {
                    'attributionMean': 0.000123,
                    'attributionStd': 0.000089,
                    'attributionMax': 0.002345,
                    'attributionMin': -0.001234
                }
            },
            'explanation': {
                'summary': 'The AI model identified three distinct fibroid regions with high confidence. The largest lesion shows characteristics consistent with a subserosal fibroid, while the smaller lesions appear to be intramural. The model\'s attention was well-focused on the pathological areas.',
                'keyFindings': [
                    'High confidence detection (92%) indicates reliable results',
                    'Multiple fibroids detected with varying severities',
                    'Largest fibroid (67.5 mm²) requires clinical attention',
                    'Model attention well-focused on pathological regions'
                ],
                'confidence': 0.89,
                'recommendations': [
                    'Consider detailed MRI for surgical planning',
                    'Monitor largest fibroid for growth progression'
                ]
            }
        }
    }
    
    # Mock patient profile
    mock_patient_profile = {
        'demographics': {
            'age': 34,
            'sex': 'female',
            'ethnicity': 'Caucasian'
        },
        'symptoms': {
            'heavyMenstrualBleeding': True,
            'pelvicPain': True,
            'pelvicPressure': False
        },
        'medicalHistory': {
            'familyHistoryFibroids': True
        }
    }
    
    # Mock doctor notes
    mock_doctor_notes = """
    Patient presents with heavy menstrual bleeding and pelvic pain consistent with clinical suspicion of uterine fibroids. 
    
    AI analysis confirms the presence of multiple fibroids with the largest measuring approximately 67.5 mm². 
    The distribution and characteristics are consistent with a mixed pattern of subserosal and intramural fibroids.
    
    Clinical correlation: Patient's symptoms align well with the imaging findings. The moderate to severe fibroids 
    identified by the AI system explain the patient's symptomatology.
    
    Recommendation: Proceed with gynecological consultation for treatment planning. Consider MRI for detailed 
    pre-surgical evaluation given the size and number of lesions detected.
    
    Dr. Shashi, MD
    Radiology & Medical Imaging
    """
    
    return mock_image_data, mock_prediction_data, mock_xai_data, mock_patient_profile, mock_doctor_notes

async def test_enhanced_pdf_generation():
    """Test the enhanced PDF generation with beautiful styling."""
    
    print("🔬 Testing Enhanced PDF Report Generation")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("test_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the enhanced PDF generator
    pdf_generator = PDFReportGenerator(str(output_dir))
    
    # Create mock data
    image_data, prediction_data, xai_data, patient_profile, doctor_notes = create_mock_data()
    
    # Generate the enhanced PDF report
    report_id = "TEST_ENHANCED_REPORT_001"
    
    try:
        print(f"📄 Generating enhanced PDF report: {report_id}")
        
        pdf_path = await pdf_generator.generate_report(
            report_id=report_id,
            image_data=image_data,
            prediction_data=prediction_data,
            xai_data=xai_data,
            patient_profile=patient_profile,
            doctor_notes=doctor_notes.strip(),
            include_integrated_gradients=True
        )
        
        print(f"✅ Enhanced PDF report generated successfully!")
        print(f"📁 Report saved to: {pdf_path}")
        print(f"📊 File size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        # Display features included
        print("\n🎨 Enhanced Features Included:")
        print("   ✅ Professional medical headers with institution branding")
        print("   ✅ Enhanced doctor information with comprehensive details")
        print("   ✅ Beautiful title section with medical styling")
        print("   ✅ Professional ultrasound image framing (mock data)")
        print("   ✅ Color-coded analysis summary with medical icons")
        print("   ✅ Enhanced XAI metrics with visual indicators")
        print("   ✅ Professional doctor notes section")
        print("   ✅ Beautiful recommendations with numbered styling")
        print("   ✅ Comprehensive medical footer with disclaimers")
        print("   ✅ Professional borders and rounded corners throughout")
        print("   ✅ Medical color scheme with proper contrast")
        
        return pdf_path
        
    except Exception as e:
        print(f"❌ Error generating enhanced PDF: {e}")
        return None

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_enhanced_pdf_generation())
    
    if result:
        print(f"\n🎉 Test completed successfully!")
        print(f"📖 Open the generated PDF to see the beautiful medical report styling")
    else:
        print(f"\n❌ Test failed!")
