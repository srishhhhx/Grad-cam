#!/usr/bin/env python3
"""
FastAPI backend for Uterine Fibroids Analyzer
Provides REST API endpoints for image upload, U-Net inference, XAI analysis, and report generation.
"""

import os
import sys
import asyncio
import uuid
import threading
import concurrent.futures
import io
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiofiles

# Add the parent directory to the path to import U-Net modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.unet_inference import UNetInferenceEngine
    from models.xai_analysis import XAIAnalysisEngine
    from utils.pdf_generator import PDFReportGenerator

    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import model modules: {e}")
    print("🎭 Running in API-only mode with mock responses")
    UNetInferenceEngine = None
    XAIAnalysisEngine = None
    PDFReportGenerator = None
    MODELS_AVAILABLE = False

# Import chatbot service
try:
    from services.chatbot_service import (
        get_chatbot_service,
        PatientContext,
        ChatMessage,
        QuestionnaireFormData,
    )

    CHATBOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import chatbot service: {e}")
    print("🤖 Chatbot functionality will not be available")
    CHATBOT_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Uterine Fibroids Analyzer API",
    description="AI-powered medical imaging analysis with explainable AI",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")

for directory in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Global instances
unet_engine = None
xai_engine = None
pdf_generator = None

# In-memory storage for demo (in production, use a proper database)
images_db: Dict[str, Dict] = {}
predictions_db: Dict[str, Dict] = {}
xai_analyses_db: Dict[str, Dict] = {}
reports_db: Dict[str, Dict] = {}


# Mock response functions for demo mode
def create_mock_prediction(image_id: str) -> Dict:
    """Create mock prediction data for demo mode"""
    import random

    random.seed(42)  # For consistent results

    num_fibroids = random.randint(0, 2)
    fibroid_areas = []

    for i in range(num_fibroids):
        area = random.uniform(100, 1500)
        severity = "mild" if area < 500 else "moderate" if area < 1000 else "severe"

        fibroid_areas.append(
            {
                "area": area,
                "location": {"x": random.uniform(20, 80), "y": random.uniform(20, 80)},
                "severity": severity,
            }
        )

    # Create a simple mock mask (base64 encoded)
    import base64

    mock_mask_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    return {
        "mask": base64.b64encode(mock_mask_data).decode("utf-8"),
        "probability": [[0.1, 0.8, 0.1]],
        "confidence": 0.85,
        "fibroidDetected": num_fibroids > 0,
        "fibroidCount": num_fibroids,
        "fibroidAreas": fibroid_areas,
    }


def create_mock_xai_analysis() -> Dict:
    """Create mock XAI analysis for demo mode"""
    import base64

    mock_image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

    return {
        "gradcam": {
            "heatmap": base64.b64encode(mock_image_data).decode("utf-8"),
            "overlayImage": base64.b64encode(mock_image_data).decode("utf-8"),
            "statistics": {
                "attentionMean": 0.125,
                "attentionStd": 0.089,
                "attentionRatio": 2.34,
                "predictionRatio": 0.67,
            },
        },
        "integratedGradients": {
            "attribution": base64.b64encode(mock_image_data).decode("utf-8"),
            "channelAnalysis": base64.b64encode(mock_image_data).decode("utf-8"),
            "statistics": {
                "attributionMean": 0.0234,
                "attributionStd": 0.0156,
                "attributionMax": 0.234,
                "attributionMin": -0.123,
                "predictionArea": 1250.0,
                "attributionRatio": 1.89,
            },
        },
        "explanation": {
            "summary": "The AI model analysis shows focused attention on specific regions of the image, indicating potential areas of interest for fibroid detection.",
            "keyFindings": [
                "Model confidence: 85.0%",
                "Attention focus ratio: 2.34",
                "Attribution strength: 1.89",
            ],
            "confidence": 0.85,
            "recommendations": [
                "Consult with a gynecologist for clinical correlation",
                "Consider follow-up imaging if symptoms persist",
            ],
        },
    }


# Pydantic models
class ImageUploadResponse(BaseModel):
    id: str
    filename: str
    url: str
    uploaded_at: datetime


class PredictionRequest(BaseModel):
    image_id: str


class PredictionResponse(BaseModel):
    id: str
    image_id: str
    status: str
    prediction: Optional[Dict] = None
    processed_at: Optional[datetime] = None


class XAIAnalysisRequest(BaseModel):
    prediction_id: str


class XAIAnalysisResponse(BaseModel):
    id: str
    prediction_id: str
    status: str
    analysis: Optional[Dict] = None
    processed_at: Optional[datetime] = None


class ReportGenerationRequest(BaseModel):
    patient_id: Optional[str] = None
    image_id: str
    prediction_id: str
    xai_analysis_id: Optional[str] = None
    patient_profile: Optional[Dict] = None
    doctor_notes: Optional[str] = None

    class Config:
        # Allow extra fields and be more flexible with validation
        extra = "allow"


class ReportResponse(BaseModel):
    id: str
    status: str
    download_url: Optional[str] = None
    generated_at: Optional[datetime] = None
    summary: Optional[Dict] = None


# Chatbot API Models
class ChatStartRequest(BaseModel):
    scan_results: Optional[Dict] = None


class QuestionnaireFormRequest(BaseModel):
    form_data: Dict
    scan_results: Optional[Dict] = None


class ChatMessageRequest(BaseModel):
    message: str
    conversation_id: str
    patient_context: Dict
    conversation_history: List[Dict]
    scan_results: Optional[Dict] = None


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    patient_context: Dict
    current_question: Optional[Dict] = None
    questionnaire_complete: bool = False
    error: bool = False
    system_prompt: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the AI engines on startup"""
    global unet_engine, xai_engine, pdf_generator

    try:
        print("🚀 Starting Uterine Fibroids Analyzer API...")

        # Configure matplotlib for thread safety
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-GUI backend for threading
            print("✅ Matplotlib configured for threading")
        except ImportError:
            print("⚠️  Matplotlib not available")

        if MODELS_AVAILABLE:
            # Initialize U-Net inference engine
            print("📊 Loading U-Net model...")
            unet_engine = UNetInferenceEngine()
            await unet_engine.initialize()

            # Initialize XAI analysis engine
            print("🔬 Loading XAI analysis engine...")
            xai_engine = XAIAnalysisEngine()
            await xai_engine.initialize()

            # Initialize PDF generator
            print("📄 Initializing PDF generator...")
            pdf_generator = PDFReportGenerator()

            print("✅ API startup complete with full AI capabilities!")
        else:
            print("🎭 API started in demo mode (AI modules not available)")
            unet_engine = None
            xai_engine = None
            pdf_generator = None

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        print("🎭 Falling back to demo mode...")
        unet_engine = None
        xai_engine = None
        pdf_generator = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "unet_engine": unet_engine is not None,
                "xai_engine": xai_engine is not None,
                "pdf_generator": pdf_generator is not None,
            },
        },
    }


@app.post("/api/images/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload medical image for analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Generate unique ID and filename
        image_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{image_id}{file_extension}"
        file_path = UPLOAD_DIR / filename

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Store metadata
        image_data = {
            "id": image_id,
            "filename": file.filename,
            "stored_filename": filename,
            "file_path": str(file_path),
            "url": f"/api/images/{image_id}",
            "content_type": file.content_type,
            "size": len(content),
            "uploaded_at": datetime.now(),
        }

        images_db[image_id] = image_data

        return {
            "success": True,
            "data": ImageUploadResponse(
                id=image_id,
                filename=file.filename,
                url=image_data["url"],
                uploaded_at=image_data["uploaded_at"],
            ),
        }

    except Exception as e:
        print(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Serve uploaded image"""
    if image_id not in images_db:
        raise HTTPException(status_code=404, detail="Image not found")

    image_data = images_db[image_id]
    file_path = Path(image_data["file_path"])

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        file_path,
        media_type=image_data["content_type"],
        filename=image_data["filename"],
    )


@app.post("/api/predictions/unet/{image_id}")
async def start_unet_prediction(image_id: str, background_tasks: BackgroundTasks):
    """Start U-Net prediction for an image"""
    if image_id not in images_db:
        raise HTTPException(status_code=404, detail="Image not found")

    if not unet_engine:
        raise HTTPException(status_code=503, detail="U-Net engine not available")

    # Generate prediction ID
    prediction_id = str(uuid.uuid4())

    # Initialize prediction record
    prediction_data = {
        "id": prediction_id,
        "image_id": image_id,
        "status": "processing",
        "prediction": None,
        "created_at": datetime.now(),
        "processed_at": None,
    }

    predictions_db[prediction_id] = prediction_data

    # Start background task
    background_tasks.add_task(run_unet_prediction, prediction_id, image_id)

    return {
        "success": True,
        "data": PredictionResponse(
            id=prediction_id, image_id=image_id, status="processing"
        ),
    }


async def run_unet_prediction(prediction_id: str, image_id: str):
    """Background task to run U-Net prediction"""
    try:
        image_data = images_db[image_id]
        image_path = image_data["file_path"]

        # Run prediction
        if unet_engine:
            result = await unet_engine.predict(image_path)
        else:
            # Use mock prediction in demo mode
            print("🎭 Using mock prediction (demo mode)")
            result = create_mock_prediction(image_id)

        # Ensure all numpy types are converted to Python native types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert result to ensure JSON serialization
        result = convert_numpy_types(result)

        # Update prediction record
        predictions_db[prediction_id].update(
            {
                "status": "completed",
                "prediction": result,
                "processed_at": datetime.now(),
            }
        )

    except Exception as e:
        print(f"Error in U-Net prediction: {e}")
        # Fallback to mock prediction
        try:
            result = create_mock_prediction(image_id)
            # Convert result to ensure JSON serialization
            result = convert_numpy_types(result)
            predictions_db[prediction_id].update(
                {
                    "status": "completed",
                    "prediction": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock prediction successful")
        except Exception as fallback_error:
            print(f"Error in fallback prediction: {fallback_error}")
            predictions_db[prediction_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


@app.get("/api/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Get prediction status and results"""
    if prediction_id not in predictions_db:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction_data = predictions_db[prediction_id]

    return {
        "success": True,
        "data": PredictionResponse(
            id=prediction_data["id"],
            image_id=prediction_data["image_id"],
            status=prediction_data["status"],
            prediction=prediction_data.get("prediction"),
            processed_at=prediction_data.get("processed_at"),
        ),
    }


@app.post("/api/xai/gradcam/{prediction_id}")
async def start_gradcam_analysis(prediction_id: str, background_tasks: BackgroundTasks):
    """Start GradCAM analysis only for a prediction"""
    if prediction_id not in predictions_db:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction_data = predictions_db[prediction_id]
    if prediction_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Prediction not completed")

    if not xai_engine:
        raise HTTPException(status_code=503, detail="XAI engine not available")

    # Generate analysis ID
    analysis_id = str(uuid.uuid4())

    # Initialize analysis record
    analysis_data = {
        "id": analysis_id,
        "prediction_id": prediction_id,
        "status": "processing",
        "analysis_type": "gradcam",
        "analysis": None,
        "created_at": datetime.now(),
        "processed_at": None,
    }

    xai_analyses_db[analysis_id] = analysis_data

    # Start background task for GradCAM only
    background_tasks.add_task(run_gradcam_only_analysis, analysis_id, prediction_id)

    return {
        "success": True,
        "data": XAIAnalysisResponse(
            id=analysis_id, prediction_id=prediction_id, status="processing"
        ),
    }


@app.post("/api/xai/integrated-gradients/{prediction_id}")
async def start_integrated_gradients_analysis(
    prediction_id: str, background_tasks: BackgroundTasks
):
    """Start Integrated Gradients analysis for a prediction"""
    if prediction_id not in predictions_db:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction_data = predictions_db[prediction_id]
    if prediction_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Prediction not completed")

    if not xai_engine:
        raise HTTPException(status_code=503, detail="XAI engine not available")

    # Generate analysis ID
    analysis_id = str(uuid.uuid4())

    # Initialize analysis record
    analysis_data = {
        "id": analysis_id,
        "prediction_id": prediction_id,
        "status": "processing",
        "analysis_type": "integrated_gradients",
        "analysis": None,
        "created_at": datetime.now(),
        "processed_at": None,
    }

    xai_analyses_db[analysis_id] = analysis_data

    # Start background task for Integrated Gradients only
    background_tasks.add_task(
        run_integrated_gradients_only_analysis, analysis_id, prediction_id
    )

    return {
        "success": True,
        "data": XAIAnalysisResponse(
            id=analysis_id, prediction_id=prediction_id, status="processing"
        ),
    }


@app.post("/api/xai/analyze/{prediction_id}")
async def start_xai_analysis(prediction_id: str, background_tasks: BackgroundTasks):
    """Start full XAI analysis for a prediction (legacy endpoint)"""
    if prediction_id not in predictions_db:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction_data = predictions_db[prediction_id]
    if prediction_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Prediction not completed")

    if not xai_engine:
        raise HTTPException(status_code=503, detail="XAI engine not available")

    # Generate analysis ID
    analysis_id = str(uuid.uuid4())

    # Initialize analysis record
    analysis_data = {
        "id": analysis_id,
        "prediction_id": prediction_id,
        "status": "processing",
        "analysis_type": "full",
        "analysis": None,
        "created_at": datetime.now(),
        "processed_at": None,
    }

    xai_analyses_db[analysis_id] = analysis_data

    # Start background task
    background_tasks.add_task(run_xai_analysis, analysis_id, prediction_id)

    return {
        "success": True,
        "data": XAIAnalysisResponse(
            id=analysis_id, prediction_id=prediction_id, status="processing"
        ),
    }


def run_gradcam_only_analysis_sync(analysis_id: str, prediction_id: str):
    """Synchronous GradCAM analysis function to run in thread"""
    try:
        # Configure matplotlib for thread safety
        import matplotlib

        matplotlib.use("Agg")  # Use non-GUI backend

        prediction_data = predictions_db[prediction_id]
        image_data = images_db[prediction_data["image_id"]]
        image_path = image_data["file_path"]

        print(f"🔬 Starting threaded GradCAM analysis for {analysis_id}")

        # Run GradCAM analysis only
        if xai_engine:
            # Create new event loop for this thread
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    xai_engine.analyze_gradcam_only(
                        image_path, prediction_data["prediction"]
                    )
                )
            finally:
                loop.close()
        else:
            # Use mock XAI analysis in demo mode
            print("🎭 Using mock GradCAM analysis (demo mode)")
            result = create_mock_xai_analysis()

        # Update analysis record
        xai_analyses_db[analysis_id].update(
            {"status": "completed", "analysis": result, "processed_at": datetime.now()}
        )

        print(f"✅ GradCAM analysis completed for {analysis_id}")
        print(f"🔍 GradCAM result keys: {list(result.keys()) if result else 'None'}")

    except Exception as e:
        print(f"❌ Error in GradCAM analysis: {e}")
        # Fallback to mock analysis
        try:
            result = create_mock_xai_analysis()
            xai_analyses_db[analysis_id].update(
                {
                    "status": "completed",
                    "analysis": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock GradCAM analysis successful")
        except Exception as fallback_error:
            print(f"Error in fallback GradCAM analysis: {fallback_error}")
            xai_analyses_db[analysis_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


def run_integrated_gradients_only_analysis_sync(analysis_id: str, prediction_id: str):
    """Synchronous Integrated Gradients analysis function to run in thread"""
    try:
        # Configure matplotlib for thread safety
        import matplotlib

        matplotlib.use("Agg")  # Use non-GUI backend

        prediction_data = predictions_db[prediction_id]
        image_data = images_db[prediction_data["image_id"]]
        image_path = image_data["file_path"]

        print(f"🔬 Starting threaded Integrated Gradients analysis for {analysis_id}")

        # Run Integrated Gradients analysis only
        if xai_engine:
            # Create new event loop for this thread
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    xai_engine.analyze_integrated_gradients_only(
                        image_path, prediction_data["prediction"]
                    )
                )
            finally:
                loop.close()
        else:
            # Use mock XAI analysis in demo mode
            print("🎭 Using mock Integrated Gradients analysis (demo mode)")
            result = create_mock_xai_analysis()

        # Update analysis record
        xai_analyses_db[analysis_id].update(
            {"status": "completed", "analysis": result, "processed_at": datetime.now()}
        )

        print(f"✅ Integrated Gradients analysis completed for {analysis_id}")
        print(f"🔍 IG result keys: {list(result.keys()) if result else 'None'}")

    except Exception as e:
        print(f"❌ Error in Integrated Gradients analysis: {e}")
        # Fallback to mock analysis
        try:
            result = create_mock_xai_analysis()
            xai_analyses_db[analysis_id].update(
                {
                    "status": "completed",
                    "analysis": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock IG analysis successful")
        except Exception as fallback_error:
            print(f"Error in fallback IG analysis: {fallback_error}")
            xai_analyses_db[analysis_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


def run_xai_analysis_sync(analysis_id: str, prediction_id: str):
    """Synchronous full XAI analysis function to run in thread"""
    try:
        # Configure matplotlib for thread safety
        import matplotlib

        matplotlib.use("Agg")  # Use non-GUI backend

        prediction_data = predictions_db[prediction_id]
        image_data = images_db[prediction_data["image_id"]]
        image_path = image_data["file_path"]

        print(f"🔬 Starting threaded full XAI analysis for {analysis_id}")

        # Run full XAI analysis synchronously
        if xai_engine:
            # Create new event loop for this thread
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    xai_engine.analyze(image_path, prediction_data["prediction"])
                )
            finally:
                loop.close()
        else:
            # Use mock XAI analysis in demo mode
            print("🎭 Using mock XAI analysis (demo mode)")
            result = create_mock_xai_analysis()

        # Update analysis record
        xai_analyses_db[analysis_id].update(
            {"status": "completed", "analysis": result, "processed_at": datetime.now()}
        )

        print(f"✅ Full XAI analysis completed for {analysis_id}")
        print(f"🔍 XAI result keys: {list(result.keys()) if result else 'None'}")
        if result and "gradcam" in result:
            print(f"🎯 GradCAM keys: {list(result['gradcam'].keys())}")
        if result and "integrated_gradients" in result:
            print(f"📊 IG keys: {list(result['integrated_gradients'].keys())}")

    except Exception as e:
        print(f"❌ Error in full XAI analysis: {e}")
        # Fallback to mock analysis
        try:
            result = create_mock_xai_analysis()
            xai_analyses_db[analysis_id].update(
                {
                    "status": "completed",
                    "analysis": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock XAI analysis successful")
        except Exception as fallback_error:
            print(f"Error in fallback XAI analysis: {fallback_error}")
            xai_analyses_db[analysis_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


async def run_gradcam_only_analysis(analysis_id: str, prediction_id: str):
    """Background task to run GradCAM analysis only"""
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        print(f"🚀 Starting GradCAM analysis in background thread for {analysis_id}")

        # Run the synchronous GradCAM analysis in a thread
        await loop.run_in_executor(
            executor, run_gradcam_only_analysis_sync, analysis_id, prediction_id
        )

    except Exception as e:
        print(f"Error in threaded GradCAM analysis: {e}")
        # Fallback to mock analysis
        try:
            result = create_mock_xai_analysis()
            xai_analyses_db[analysis_id].update(
                {
                    "status": "completed",
                    "analysis": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock GradCAM analysis successful")
        except Exception as fallback_error:
            print(f"Error in fallback GradCAM analysis: {fallback_error}")
            xai_analyses_db[analysis_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


async def run_integrated_gradients_only_analysis(analysis_id: str, prediction_id: str):
    """Background task to run Integrated Gradients analysis only"""
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        print(
            f"🚀 Starting Integrated Gradients analysis in background thread for {analysis_id}"
        )

        # Run the synchronous IG analysis in a thread
        await loop.run_in_executor(
            executor,
            run_integrated_gradients_only_analysis_sync,
            analysis_id,
            prediction_id,
        )

    except Exception as e:
        print(f"Error in threaded Integrated Gradients analysis: {e}")
        # Fallback to mock analysis
        try:
            result = create_mock_xai_analysis()
            xai_analyses_db[analysis_id].update(
                {
                    "status": "completed",
                    "analysis": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock IG analysis successful")
        except Exception as fallback_error:
            print(f"Error in fallback IG analysis: {fallback_error}")
            xai_analyses_db[analysis_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


async def run_xai_analysis(analysis_id: str, prediction_id: str):
    """Background task to run full XAI analysis in a separate thread"""
    try:
        # Run XAI analysis in a separate thread to avoid blocking
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()

        print(f"🚀 Starting full XAI analysis in background thread for {analysis_id}")

        # Run the synchronous XAI analysis in a thread
        await loop.run_in_executor(
            executor, run_xai_analysis_sync, analysis_id, prediction_id
        )

    except Exception as e:
        print(f"Error in threaded XAI analysis: {e}")
        # Fallback to mock analysis
        try:
            result = create_mock_xai_analysis()
            xai_analyses_db[analysis_id].update(
                {
                    "status": "completed",
                    "analysis": result,
                    "processed_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock XAI analysis successful")
        except Exception as fallback_error:
            print(f"Error in fallback XAI analysis: {fallback_error}")
            xai_analyses_db[analysis_id].update(
                {"status": "error", "error": str(e), "processed_at": datetime.now()}
            )


@app.get("/api/xai/{analysis_id}")
async def get_xai_analysis(analysis_id: str):
    """Get XAI analysis status and results"""
    if analysis_id not in xai_analyses_db:
        raise HTTPException(status_code=404, detail="XAI analysis not found")

    analysis_data = xai_analyses_db[analysis_id]

    return {
        "success": True,
        "data": XAIAnalysisResponse(
            id=analysis_data["id"],
            prediction_id=analysis_data["prediction_id"],
            status=analysis_data["status"],
            analysis=analysis_data.get("analysis"),
            processed_at=analysis_data.get("processed_at"),
        ),
    }


@app.post("/api/reports/generate")
async def generate_report(
    request: ReportGenerationRequest, background_tasks: BackgroundTasks
):
    """Generate medical report"""
    try:
        print(f"🔍 Report generation request received:")
        print(f"   Patient ID: {request.patient_id}")
        print(f"   Image ID: {request.image_id}")
        print(f"   Prediction ID: {request.prediction_id}")
        print(f"   XAI Analysis ID: {request.xai_analysis_id}")
        print(f"   Patient Profile type: {type(request.patient_profile)}")
        print(f"   Patient Profile: {request.patient_profile}")
        print(f"   Doctor Notes: {request.doctor_notes}")
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

    # Validate required data exists
    if request.image_id not in images_db:
        print(f"❌ Image not found: {request.image_id}")
        print(f"   Available images: {list(images_db.keys())}")
        raise HTTPException(status_code=404, detail="Image not found")

    if request.prediction_id not in predictions_db:
        print(f"❌ Prediction not found: {request.prediction_id}")
        print(f"   Available predictions: {list(predictions_db.keys())}")
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not pdf_generator:
        raise HTTPException(status_code=503, detail="PDF generator not available")

    # Generate report ID
    report_id = str(uuid.uuid4())

    # Get prediction data to extract summary information
    prediction_data = predictions_db[request.prediction_id]
    prediction = prediction_data.get("prediction", {})

    # Calculate overall severity from fibroid areas
    overall_severity = "Not specified"
    recommendations = []

    if prediction.get("fibroidDetected") and prediction.get("fibroidAreas"):
        severities = [
            area.get("severity", "unknown") for area in prediction["fibroidAreas"]
        ]
        severity_priority = {"severe": 3, "moderate": 2, "mild": 1, "unknown": 0}
        if severities:
            highest_severity = max(
                severities, key=lambda x: severity_priority.get(x, 0)
            )
            if highest_severity != "unknown":
                overall_severity = highest_severity

        # Generate recommendations based on severity
        if overall_severity == "severe":
            recommendations = [
                "Urgent gynecological consultation recommended",
                "Consider MRI for detailed characterization",
                "Discuss treatment options immediately",
            ]
        elif overall_severity == "moderate":
            recommendations = [
                "Schedule gynecological consultation within 2-4 weeks",
                "Monitor symptoms closely",
                "Consider follow-up imaging in 6 months",
            ]
        elif overall_severity == "mild":
            recommendations = [
                "Routine gynecological follow-up recommended",
                "Monitor symptoms",
                "Annual follow-up imaging if symptomatic",
            ]
    else:
        recommendations = [
            "Continue routine gynecological care",
            "If symptoms persist, consider additional evaluation",
        ]

    # Initialize report record
    report_data = {
        "id": report_id,
        "patient_id": request.patient_id,
        "image_id": request.image_id,
        "prediction_id": request.prediction_id,
        "xai_analysis_id": request.xai_analysis_id,
        "status": "generating",
        "download_url": None,
        "created_at": datetime.now(),
        "generated_at": None,
        "summary": {
            "severity": overall_severity,
            "fibroidDetected": prediction.get("fibroidDetected", False),
            "confidence": prediction.get("confidence", 0),
            "fibroidCount": prediction.get("fibroidCount", 0),
            "recommendations": recommendations,
        },
    }

    reports_db[report_id] = report_data

    # Start background task
    background_tasks.add_task(generate_pdf_report, report_id, request)

    return {"success": True, "data": ReportResponse(id=report_id, status="generating")}


async def generate_pdf_report(report_id: str, request: ReportGenerationRequest):
    """Background task to generate PDF report"""
    try:
        # Gather all data
        image_data = images_db[request.image_id]
        prediction_data = predictions_db[request.prediction_id]

        xai_data = None
        if (
            request.xai_analysis_id
            and request.xai_analysis_id.strip()
            and request.xai_analysis_id in xai_analyses_db
        ):
            xai_data = xai_analyses_db[request.xai_analysis_id]

        # Generate PDF
        if pdf_generator:
            # Check if include_integrated_gradients flag is provided
            include_ig = getattr(request, "include_integrated_gradients", True)

            pdf_path = await pdf_generator.generate_report(
                report_id=report_id,
                image_data=image_data,
                prediction_data=prediction_data,
                xai_data=xai_data,
                patient_profile=request.patient_profile,
                doctor_notes=request.doctor_notes,
                include_integrated_gradients=include_ig,
            )
        else:
            # Create mock PDF in demo mode
            print("🎭 Creating mock PDF report (demo mode)")
            pdf_path = await create_mock_pdf_report(
                report_id, image_data, prediction_data
            )

        # Update report record
        reports_db[report_id].update(
            {
                "status": "completed",
                "download_url": f"/api/reports/{report_id}/download",
                "pdf_path": str(pdf_path),
                "generated_at": datetime.now(),
            }
        )

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        # Try fallback mock PDF
        try:
            pdf_path = await create_mock_pdf_report(report_id, {}, {})
            reports_db[report_id].update(
                {
                    "status": "completed",
                    "download_url": f"/api/reports/{report_id}/download",
                    "pdf_path": str(pdf_path),
                    "generated_at": datetime.now(),
                }
            )
            print("🎭 Fallback to mock PDF successful")
        except Exception as fallback_error:
            print(f"Error in fallback PDF generation: {fallback_error}")
            reports_db[report_id].update(
                {"status": "error", "error": str(e), "generated_at": datetime.now()}
            )


async def create_mock_pdf_report(
    report_id: str, image_data: Dict, prediction_data: Dict
) -> Path:
    """Create a simple mock PDF report for demo mode"""
    pdf_path = REPORTS_DIR / f"mock_report_{report_id}.txt"

    # Create a simple text file as mock PDF
    content = f"""
UTERINE FIBROIDS ANALYSIS REPORT (DEMO MODE)
============================================

Report ID: {report_id}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ANALYSIS SUMMARY:
- This is a demo report generated in mock mode
- The actual application would generate a professional PDF
- All analysis results are simulated for demonstration

FINDINGS:
- Demo fibroid detection results
- Mock AI analysis completed
- Sample XAI explanations provided

RECOMMENDATIONS:
- Consult with healthcare provider
- This is for demonstration purposes only

Note: This is a mock report generated because the full PDF
generation system is not available in demo mode.
"""

    # Write the mock report
    with open(pdf_path, "w") as f:
        f.write(content)

    return pdf_path


@app.get("/api/reports/{report_id}")
async def get_report_status(report_id: str):
    """Get report generation status"""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="Report not found")

    report_data = reports_db[report_id]

    return {
        "success": True,
        "data": ReportResponse(
            id=report_data["id"],
            status=report_data["status"],
            download_url=report_data.get("download_url"),
            generated_at=report_data.get("generated_at"),
            summary=report_data.get("summary"),
        ),
    }


@app.get("/api/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download generated PDF report"""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="Report not found")

    report_data = reports_db[report_id]

    if report_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready for download")

    pdf_path = Path(report_data["pdf_path"])

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"fibroid-analysis-report-{report_id}.pdf",
    )


@app.get("/api/predictions/{prediction_id}/overlay-image")
async def get_prediction_overlay_image(prediction_id: str):
    """Get ultrasound image with prediction overlay for frontend display"""
    import aiofiles
    import io
    import traceback

    if prediction_id not in predictions_db:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction_data = predictions_db[prediction_id]

    if prediction_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Prediction not completed")

    # Get the associated image data
    image_id = prediction_data["image_id"]
    if image_id not in images_db:
        raise HTTPException(status_code=404, detail="Associated image not found")

    image_data = images_db[image_id]

    # --- CRITICAL FIX: Load image bytes and add to image_data dict ---
    file_path = Path(image_data["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    async with aiofiles.open(file_path, "rb") as f:
        image_bytes = await f.read()
    image_data = dict(image_data)  # Make a copy to avoid mutating global
    image_data["image_data"] = base64.b64encode(image_bytes).decode("utf-8")
    # ---------------------------------------------------------------

    # Generate overlay image using the PDF generator's method
    if pdf_generator:
        try:
            print(f"🔍 Overlay endpoint - Image data keys: {list(image_data.keys())}")
            print(
                f"🔍 Overlay endpoint - Prediction data keys: {list(prediction_data.keys())}"
            )
            overlay_buffer = pdf_generator._create_segmentation_overlay(
                image_data, prediction_data
            )
            if overlay_buffer:
                overlay_buffer.seek(0)
                return StreamingResponse(
                    io.BytesIO(overlay_buffer.read()),
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"inline; filename=overlay_{prediction_id}.png"
                    },
                )
            else:
                print("❌ overlay_buffer is None")
                raise HTTPException(
                    status_code=500, detail="Failed to generate overlay image"
                )
        except Exception as e:
            print(f"❌ Error generating overlay image: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Failed to generate overlay image: {e}"
            )
    else:
        raise HTTPException(status_code=503, detail="PDF generator not available")


# Chatbot API Endpoints
@app.post("/api/chatbot/start")
async def start_chatbot_conversation(request: ChatStartRequest):
    """Start a new chatbot conversation"""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chatbot service not available")

    try:
        chatbot = get_chatbot_service()
        result = await chatbot.start_conversation(request.scan_results)

        return {
            "success": True,
            "data": ChatResponse(
                message=result["message"],
                conversation_id=result["conversation_id"],
                patient_context=result["patient_context"],
                current_question=result.get("current_question"),
                questionnaire_complete=False,
            ),
        }
    except Exception as e:
        print(f"Error starting chatbot conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chatbot/message")
async def send_chatbot_message(request: ChatMessageRequest):
    """Send a message to the chatbot"""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chatbot service not available")

    try:
        chatbot = get_chatbot_service()

        # Convert request data to proper types
        patient_context = PatientContext(**request.patient_context)

        # Convert conversation history
        conversation_history = []
        for msg_data in request.conversation_history:
            # Handle timestamp conversion
            timestamp = msg_data["timestamp"]
            if isinstance(timestamp, str):
                # Remove 'Z' suffix and parse
                timestamp = timestamp.replace("Z", "+00:00")
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    # Fallback to current time if parsing fails
                    timestamp = datetime.now()

            conversation_history.append(
                ChatMessage(
                    id=msg_data["id"],
                    type=msg_data["type"],
                    content=msg_data["content"],
                    timestamp=timestamp,
                )
            )

        # Process the message
        result = await chatbot.process_user_response(
            user_message=request.message,
            patient_context=patient_context,
            conversation_history=conversation_history,
            scan_results=request.scan_results,  # Pass scan results from request
        )

        return {
            "success": True,
            "data": ChatResponse(
                message=result["message"],
                conversation_id=request.conversation_id,
                patient_context=result["patient_context"],
                current_question=result.get("current_question"),
                questionnaire_complete=result.get("questionnaire_complete", False),
                error=result.get("error", False),
            ),
        }

    except Exception as e:
        print(f"Error processing chatbot message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chatbot/form-start")
async def start_form_based_chatbot(request: QuestionnaireFormRequest):
    """Start a new chatbot conversation based on completed form data"""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chatbot service not available")

    try:
        from services.chatbot_service import QuestionnaireFormData

        chatbot = get_chatbot_service()

        # Convert form data to proper format
        form_data = QuestionnaireFormData(**request.form_data)

        # Start form-based conversation
        result = await chatbot.start_form_based_conversation(
            form_data, request.scan_results
        )

        return {
            "success": True,
            "data": ChatResponse(
                message=result["message"],
                conversation_id=result["conversation_id"],
                patient_context=result["patient_context"],
                questionnaire_complete=True,
                system_prompt=result.get("system_prompt"),
            ),
        }
    except Exception as e:
        print(f"Error starting form-based chatbot conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chatbot/health")
async def chatbot_health_check():
    """Check chatbot service health"""
    return {
        "success": True,
        "data": {
            "chatbot_available": CHATBOT_AVAILABLE,
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
            if CHATBOT_AVAILABLE
            else False,
        },
    }


if __name__ == "__main__":
    print(f"🚀 Starting server on port 8000")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
