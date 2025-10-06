# 🩺 Uterine Fibroids Analyzer Web Application

## 📋 Contents

1. [Introduction](#1-introduction)
2. [Demo Video](#2-demo-video)
3. [Features](#3-features)
4. [Architecture Diagram](#4-architecture-diagram)
5. [Tech Stack](#5-tech-stack)
6. [Project Structure](#6-project-structure)
7. [How to Run the App](#7-how-to-run-the-app)
8. [Difficulties Faced](#8-difficulties-faced)
9. [Future Improvements](#9-future-improvements)

---

## 1. Introduction

The Uterine Fibroids Analyzer is an AI-powered medical imaging web application that assists healthcare professionals and patients in detecting and analyzing uterine fibroids from ultrasound images. The system combines advanced deep learning techniques with explainable AI to provide accurate segmentation results and interpretable insights through an intuitive web interface.

---

## 2. Demo Video

<!-- Demo video will be added here -->
[*Demo video placeholder - to be added*](https://github.com/user-attachments/assets/f90d4572-6981-4a74-b862-5c41dcc53863)

---

## 3. Features

### 🏥 **Doctor Interface**
- **🖼️ Image Upload & Analysis**: Secure upload of patient ultrasound images with real-time AI-powered segmentation
- **🔍 U-Net++ Segmentation**: State-of-the-art deep learning model for precise fibroid detection and boundary delineation
- **🧠 Explainable AI (XAI)**: 
  - **GradCAM Analysis**: Visual attention heatmaps showing model focus areas
  - **Integrated Gradients**: Pixel-level attribution maps for detailed interpretability
- **📊 Comprehensive Analytics**: Detailed metrics including Dice score, IoU, sensitivity, specificity, and precision
- **📄 PDF Report Generation**: Professional medical reports with patient data, analysis results, and XAI visualizations
- **📝 Clinical Notes**: Add personalized observations and recommendations to reports
- **⚡ Real-time Processing**: Asynchronous analysis with progress tracking and status updates

### 👩‍⚕️ **Patient Interface**
- **📱 User-Friendly Upload**: Simplified interface for personal ultrasound scan uploads
- **🤖 AI Health Chatbot**: Intelligent conversational assistant powered by Gemini AI for health guidance
- **📋 Interactive Questionnaire**: Comprehensive health assessment with dynamic question flow
- **📊 Personalized Reports**: Easy-to-understand analysis results with visual explanations
- **💬 Health Consultation**: Get answers to fibroid-related questions and lifestyle recommendations

### 🔧 **Technical Features**
- **🔄 Asynchronous Processing**: Non-blocking AI inference with polling-based status updates
- **📈 Progress Tracking**: Real-time progress indicators for all analysis stages
- **🛡️ Error Handling**: Robust error management with graceful fallbacks
- **📱 Responsive Design**: Mobile-first design that works across all devices
- **🎨 Modern UI/UX**: Clean, intuitive interface with smooth animations and transitions

---

## 4. Architecture Diagram

<!-- Architecture diagram will be added here -->
*Architecture diagram placeholder - to be added*

---

## 5. Tech Stack

### **Frontend**
- **⚛️ React 18** - Modern component-based UI framework
- **📘 TypeScript** - Type-safe JavaScript for better development experience
- **⚡ Vite** - Fast build tool and development server
- **🎨 Tailwind CSS** - Utility-first CSS framework for rapid styling
- **🎭 Framer Motion** - Production-ready motion library for React
- **📡 Axios** - Promise-based HTTP client for API communication

### **Backend**
- **🐍 Python 3.11** - Core programming language
- **⚡ FastAPI** - Modern, fast web framework for building APIs
- **🧠 PyTorch** - Deep learning framework for model inference
- **🔬 Segmentation Models PyTorch** - Pre-trained segmentation architectures
- **👁️ PyTorch GradCAM** - Explainable AI visualization library
- **🖼️ OpenCV** - Computer vision and image processing
- **📊 NumPy** - Numerical computing library
- **🎨 Matplotlib** - Plotting and visualization
- **📄 ReportLab** - PDF generation and document creation
- **🤖 Google Gemini AI** - Conversational AI for chatbot functionality

### **AI/ML Stack**
- **🏗️ U-Net++** - Advanced segmentation architecture
- **🔧 EfficientNet-B5** - Efficient convolutional neural network encoder
- **📐 Albumentations** - Image augmentation library
- **🔍 GradCAM & GradCAM++** - Gradient-based attention visualization
- **📊 Integrated Gradients** - Attribution method for model interpretability

### **Development Tools**
- **📦 npm/pip** - Package managers
- **🔄 Uvicorn** - ASGI server for FastAPI
- **🛠️ Git** - Version control system

---

## 6. Project Structure

```
UT_webapp 2/
├── 📁 frontend/                    # React TypeScript frontend
│   ├── 📁 src/
│   │   ├── 📁 components/          # Reusable UI components
│   │   │   ├── ImageUpload.tsx     # Image upload component
│   │   │   ├── PredictionViewer.tsx # Results visualization
│   │   │   ├── GradCAMSlideshow.tsx # XAI analysis display
│   │   │   ├── PatientInterface.tsx # Patient-specific UI
│   │   │   ├── DoctorInterface.tsx  # Doctor-specific UI
│   │   │   └── ChatInterface.tsx    # Chatbot integration
│   │   ├── 📁 utils/
│   │   │   └── api.ts              # API communication layer
│   │   ├── 📁 types/
│   │   │   └── index.ts            # TypeScript type definitions
│   │   └── App.tsx                 # Main application component
│   ├── package.json                # Frontend dependencies
│   └── vite.config.ts             # Vite configuration
│
├── 📁 backend/                     # FastAPI Python backend
│   ├── 📁 models/                  # AI model implementations
│   │   ├── unet_inference.py       # U-Net segmentation engine
│   │   ├── xai_analysis.py         # XAI analysis coordinator
│   │   ├── gradcam_unet.py         # GradCAM implementation
│   │   └── integrated_gradients.py # Integrated Gradients implementation
│   ├── 📁 services/                # Business logic services
│   │   └── chatbot_service.py      # Gemini AI chatbot service
│   ├── 📁 utils/                   # Utility functions
│   │   └── pdf_generator.py        # PDF report generation
│   ├── main.py                     # FastAPI application entry point
│   └── requirements.txt            # Python dependencies
│
├── 📁 UNET-model/                  # Model training and utilities
│   ├── 📁 utils/                   # Training utilities
│   │   ├── dataset.py              # Dataset handling
│   │   ├── enhanced_dataloader.py  # Advanced data loading
│   │   └── metrics.py              # Training metrics
│   ├── train_unetpp.py            # Model training script
│   ├── predict.py                 # Standalone prediction script
│   └── predict_and_overlay.py     # Prediction with visualization
│
├── 📁 models_20250609_105424/      # Trained model weights
│   └── best_model.pth             # Best performing model checkpoint
│
└── README.md                      # This file
```

---

## 7. How to Run the App

### **Prerequisites**
- 🐍 Python 3.11 or higher
- 📦 Node.js 18 or higher
- 💾 At least 4GB RAM (for model inference)
- 🖥️ Modern web browser (Chrome, Firefox, Safari, Edge)

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

### **Step 5: Access the Application**
- 🌐 Open your browser and navigate to `http://localhost:3000`
- 👩‍⚕️ Choose **Doctor Interface** for clinical analysis
- 👤 Choose **Patient Interface** for patient-focused experience
- 📤 Upload an ultrasound image to begin analysis

### **API Documentation**
- 📚 FastAPI auto-generated docs: `http://localhost:8000/docs`
- 🔍 Alternative docs: `http://localhost:8000/redoc`

---

## 8. Difficulties Faced

### **🔧 Technical Challenges**

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

### **🎨 UI/UX Challenges**

#### **Cross-Platform Compatibility**
- **Issue**: Inconsistent behavior across different browsers and devices
- **Solution**: Implemented responsive design with extensive cross-browser testing
- **Impact**: Consistent user experience across all platforms

#### **Medical Data Visualization**
- **Issue**: Complex medical imaging data difficult to interpret for non-experts
- **Solution**: Created intuitive visualizations with explanatory tooltips and guides
- **Impact**: Improved accessibility for both medical professionals and patients

### **🔒 Performance & Security**

#### **Memory Management**
- **Issue**: Large model and image processing causing memory issues
- **Solution**: Implemented efficient memory management and garbage collection
- **Impact**: Stable performance even with large ultrasound images

#### **Error Handling**
- **Issue**: Cryptic error messages and poor error recovery
- **Solution**: Comprehensive error handling with user-friendly messages
- **Impact**: Better user experience and easier debugging

---

## 9. Future Improvements

### **🚀 Short-term Enhancements**

#### **Performance Optimization**
- **GPU Acceleration**: Implement CUDA support for faster model inference
- **Model Quantization**: Reduce model size while maintaining accuracy
- **Caching System**: Implement Redis for caching frequent analysis results
- **CDN Integration**: Use content delivery networks for faster asset loading

#### **User Experience**
- **Batch Processing**: Allow multiple image uploads and batch analysis
- **Real-time Collaboration**: Enable multiple doctors to review cases simultaneously
- **Mobile App**: Develop native mobile applications for iOS and Android
- **Offline Mode**: Enable basic functionality without internet connectivity

### **🔬 Advanced AI Features**

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

### **🏥 Clinical Integration**

#### **Healthcare System Integration**
- **DICOM Support**: Full integration with medical imaging standards
- **HL7 FHIR**: Healthcare data exchange protocol implementation
- **EHR Integration**: Connect with Electronic Health Record systems
- **PACS Integration**: Picture Archiving and Communication Systems support

#### **Regulatory Compliance**
- **HIPAA Compliance**: Ensure patient data privacy and security
- **FDA Approval Process**: Prepare for medical device regulatory approval
- **Clinical Validation**: Conduct extensive clinical trials and validation studies
- **Audit Trails**: Comprehensive logging for regulatory compliance

### **🌐 Platform & Infrastructure**

#### **Scalability**
- **Microservices Architecture**: Break down monolithic backend into microservices
- **Container Orchestration**: Implement Kubernetes for scalable deployment
- **Load Balancing**: Handle high traffic with intelligent load distribution
- **Auto-scaling**: Automatic resource scaling based on demand

#### **Database & Storage**
- **PostgreSQL Integration**: Replace in-memory storage with robust database
- **Cloud Storage**: Implement AWS S3 or Google Cloud Storage for images
- **Data Backup**: Automated backup and disaster recovery systems
- **Data Analytics**: Implement analytics dashboard for usage insights

### **🔐 Security & Privacy**

#### **Advanced Security**
- **Multi-factor Authentication**: Enhanced user authentication systems
- **Role-based Access Control**: Granular permissions for different user types
- **End-to-end Encryption**: Secure data transmission and storage
- **Security Auditing**: Regular security assessments and penetration testing

#### **Privacy Features**
- **Data Anonymization**: Automatic removal of patient identifiers
- **Consent Management**: Comprehensive patient consent tracking
- **Data Retention Policies**: Automated data lifecycle management
- **Privacy Dashboard**: User control over personal data usage

### **📊 Analytics & Monitoring**

#### **Business Intelligence**
- **Usage Analytics**: Detailed insights into application usage patterns
- **Performance Monitoring**: Real-time application performance tracking
- **Error Tracking**: Comprehensive error monitoring and alerting
- **A/B Testing**: Experiment with different UI/UX approaches

#### **Clinical Analytics**
- **Population Health Insights**: Aggregate analysis across patient populations
- **Treatment Outcome Tracking**: Long-term patient outcome monitoring
- **Research Data Export**: Support for clinical research and studies
- **Quality Metrics**: Track diagnostic accuracy and clinical outcomes
