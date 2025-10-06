# Uterine Fibroids Analyzer Web Application

## Contents

1. [Introduction](#1-introduction)
2. [Demo Video](#2-demo-video)
3. [Model Summary](#3-model-summary)
4. [Features](#4-features)
5. [Architecture Diagram](#5-architecture-diagram)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [How to Run the App](#8-how-to-run-the-app)
9. [Difficulties Faced](#9-difficulties-faced)
10. [Future Improvements](#10-future-improvements)

---

## 1. Introduction

The Uterine Fibroids Analyzer is an AI-powered medical imaging web application that assists healthcare professionals and patients in detecting and analyzing uterine fibroids from ultrasound images. The system combines advanced deep learning techniques(UNet,UNet++) with explainable AI(GradCam,Integrated Gradients) to provide accurate segmentation results and interpretable insights through an intuitive web interface(React,FastAPI).

---

## 2. Demo Video

<!-- Demo video will be added here -->
[*Demo video placeholder - to be added*](https://github.com/user-attachments/assets/f90d4572-6981-4a74-b862-5c41dcc53863)

---

## 3. Model Summary

### U-Net++ Architecture with EfficientNet-B5 Encoder

The core AI model is a U-Net++ architecture with EfficientNet-B5 encoder, specifically trained for uterine fibroid segmentation from ultrasound images.

**Key Performance Metrics:**
- **Dice Score**: 0.8944 (89.44% segmentation accuracy)
- **IoU (Intersection over Union)**: 0.8135 (81.35% overlap accuracy)
- **Pixel Accuracy**: 99.72% (overall pixel classification accuracy)
- **Sensitivity**: 89.11% (true positive rate for fibroid detection)
- **Specificity**: 99.90% (true negative rate for healthy tissue)
- **Precision**: 91.00% (positive predictive value)


For detailed model training results, architecture details, and comprehensive evaluation metrics, see: [UNET-model/README.md](UNET-model/README.md)

---

## 4. Features

### **Doctor Interface**
- **Image Upload & Analysis**: Secure upload of patient ultrasound images with real-time AI-powered segmentation
- **U-Net++ Segmentation**: State-of-the-art deep learning model for precise fibroid detection and boundary delineation
- **Explainable AI (XAI)**: 
  - **GradCAM Analysis**: Visual attention heatmaps showing model focus areas
  - **Integrated Gradients**: Pixel-level attribution maps for detailed interpretability
- **Comprehensive Analytics**: Detailed metrics including Dice score, IoU, sensitivity, specificity, and precision
- **PDF Report Generation**: Professional medical reports with patient data, analysis results, and XAI visualizations
- **Clinical Notes**: Add personalized observations and recommendations to reports
- **Real-time Processing**: Asynchronous analysis with progress tracking and status updates

### **Patient Interface**
- **User-Friendly Upload**: Simplified interface for personal ultrasound scan uploads
- **AI Health Chatbot**: Intelligent conversational assistant powered by Gemini AI for health guidance
- **Interactive Questionnaire**: Comprehensive health assessment with dynamic question flow
- **Personalized Reports**: Easy-to-understand analysis results with visual explanations
- **Health Consultation**: Get answers to fibroid-related questions and lifestyle recommendations

### **Technical Features**
- **Asynchronous Processing**: Non-blocking AI inference with polling-based status updates
- **Progress Tracking**: Real-time progress indicators for all analysis stages
- **Error Handling**: Robust error management with graceful fallbacks
- **Responsive Design**: Mobile-first design that works across all devices
- **Modern UI/UX**: Clean, intuitive interface with smooth animations and transitions

---

## 5. Architecture Diagram

<!-- Architecture diagram will be added here -->
*Architecture diagram placeholder - to be added*

---

## 6. Tech Stack

### **Frontend**
- React 18
- TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Axios

### **Backend**
- Python 3.11
- FastAPI
- PyTorch
- Segmentation Models PyTorch
- PyTorch GradCAM
- OpenCV
- NumPy
- Matplotlib
- ReportLab
- Google Gemini AI

### **AI/ML Stack**
- U-Net++
- EfficientNet-B5
- Albumentations
- GradCAM & GradCAM++
- Integrated Gradients

### **Development Tools**
- npm/pip
- Uvicorn
- Git

---

## 7. Project Structure

```
UT_webapp 2/
├── frontend/                    # React TypeScript frontend
│   ├── src/
│   │   ├── components/         
│   │   │   ├── ImageUpload.tsx     
│   │   │   ├── PredictionViewer.tsx 
│   │   │   ├── GradCAMSlideshow.tsx 
│   │   │   ├── PatientInterface.tsx 
│   │   │   ├── DoctorInterface.tsx 
│   │   │   └── ChatInterface.tsx    
│   │   ├── utils/
│   │   │   └── api.ts              # API communication layer
│   │   ├── types/
│   │   │   └── index.ts           
│   │   └── App.tsx                 
│   ├── package.json
│   └── vite.config.ts            
│
├── backend/                     # FastAPI Python backend
│   ├── models/                  # AI model implementations
│   │   ├── unet_inference.py       # U-Net segmentation engine
│   │   ├── xai_analysis.py         # XAI analysis coordinator
│   │   ├── gradcam_unet.py         # GradCAM implementation
│   │   └── integrated_gradients.py # Integrated Gradients implementation
│   ├── services/               
│   │   └── chatbot_service.py      # Gemini AI chatbot service
│   ├── utils/                  
│   │   └── pdf_generator.py        # PDF report generation
│   ├── main.py                     # FastAPI application entry point
│   └── requirements.txt           
│
├── UNET-model/                  # Model training and utilities
│   ├── utils/                   
│   │   ├── dataset.py              
│   │   ├── enhanced_dataloader.py  
│   │   └── metrics.py              
│   ├── train_unetpp.py            
│   ├── predict.py                
│   └── predict_and_overlay.py    
│
├── models_20250609_105424/      # Trained model weights
│   └── best_model.pth             # Best performing model checkpoint
│
└── README.md                      
```

---

## 8. How to Run the App

### **Prerequisites**
- Python 3.11 or higher
- Node.js 18 or higher
- At least 4GB RAM (for model inference)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/srishhhhx/Grad-cam.git
cd UT_webapp\ 2
```

### **Step 2: Backend Setup**
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env  # Configure Gemini API key if using chatbot
```

### **Step 3: Frontend Setup**
```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install Node.js dependencies
npm install
```

### **Step 4: Start the Application**

**Terminal 1 - Backend Server:**
```bash
cd backend
python main.py
# Server will start at http://localhost:8000
```

**Terminal 2 - Frontend Development Server:**
```bash
cd frontend
npm run dev
# Application will open at http://localhost:3000
```
### **Step 5 : Download the Unet model from the provided link and place it in the models_20250609_105424 folder**:

Download the model from here: https://drive.google.com/file/d/1iGERcpQW3reazoFDjXG-QJnLMYNmoHAP/view?usp=sharing


### **Step 6: Access the Application**
- Open your browser and navigate to `http://localhost:3000`
- Choose **Doctor Interface** for clinical analysis
- Choose **Patient Interface** for patient-focused experience
- Upload an ultrasound image to begin analysis

### **API Documentation**
- FastAPI auto-generated docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

---

## 9. Difficulties Faced

### **Technical Challenges**

#### **Model Integration & Path Management**
- **Issue**: Complex import path resolution between backend and UNET-model directories
- **Solution**: Implemented dynamic path resolution and reorganized project structure for cleaner imports
- **Impact**: Improved code maintainability and reduced import errors

#### **Large File Handling**
- **Issue**: 358MB model file exceeded GitHub's 100MB limit, causing push failures
- **Solution**: Implemented proper .gitignore patterns and Git history cleanup
- **Impact**: Successful repository management without compromising model availability

#### **Asynchronous Processing**
- **Issue**: Long-running AI inference blocking the web interface
- **Solution**: Implemented polling-based asynchronous processing with progress tracking
- **Impact**: Improved user experience with real-time progress updates

#### **XAI Analysis Complexity**
- **Issue**: GradCAM and Integrated Gradients analysis failing silently in production
- **Solution**: Added comprehensive error handling, logging, and fallback mechanisms
- **Impact**: Robust XAI analysis with detailed debugging capabilities

### **UI/UX Challenges**

#### **Cross-Platform Compatibility**
- **Issue**: Inconsistent behavior across different browsers and devices
- **Solution**: Implemented responsive design with extensive cross-browser testing
- **Impact**: Consistent user experience across all platforms

#### **Medical Data Visualization**
- **Issue**: Complex medical imaging data difficult to interpret for non-experts
- **Solution**: Created intuitive visualizations with explanatory tooltips and guides
- **Impact**: Improved accessibility for both medical professionals and patients

### **Performance & Security**

#### **Memory Management**
- **Issue**: Large model and image processing causing memory issues
- **Solution**: Implemented efficient memory management and garbage collection
- **Impact**: Stable performance even with large ultrasound images

#### **Error Handling**
- **Issue**: Cryptic error messages and poor error recovery
- **Solution**: Comprehensive error handling with user-friendly messages
- **Impact**: Better user experience and easier debugging

---

## 10. Future Improvements

#### **Performance Optimization**
- **GPU Acceleration**: Implement CUDA support for faster model inference
- **Model Quantization**: Reduce model size while maintaining accuracy
- **Caching System**: Implement Redis for caching frequent analysis results
- **CDN Integration**: Use content delivery networks for faster asset loading

### **Advanced AI Features**

#### **Model Improvements**
- **Multi-class Segmentation**: Detect different types of fibroids and abnormalities
- **3D Analysis**: Support for 3D ultrasound and MRI imaging
- **Temporal Analysis**: Track fibroid growth over time with multiple scans
- **Uncertainty Quantification**: Provide confidence intervals for predictions

#### **Enhanced XAI**
- **LIME Integration**: Add Local Interpretable Model-agnostic Explanations
- **SHAP Values**: Implement SHapley Additive exPlanations for feature importance
- **Counterfactual Explanations**: Show what changes would alter the diagnosis
- **Interactive Explanations**: Allow users to explore different explanation methods

### **Clinical Integration**

#### **Healthcare System Integration**
- **DICOM Support**: Full integration with medical imaging standards
- **HL7 FHIR**: Healthcare data exchange protocol implementation
- **EHR Integration**: Connect with Electronic Health Record systems
- **PACS Integration**: Picture Archiving and Communication Systems support

