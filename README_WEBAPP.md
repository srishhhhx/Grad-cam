# Uterine Fibroids Analyzer - React TypeScript Web Application

A modern, AI-powered web application for uterine fibroids detection using U-Net++ deep learning models with explainable AI features.

## 🌟 Features

### Doctor Interface
- **Step 1: Upload Scan** - Secure medical image upload with validation
- **Step 2: AI Analysis** - U-Net++ model inference for fibroid detection
- **Step 3: XAI Explanation** - GradCAM and Integrated Gradients explanations
- **Step 4: Generate Report** - Comprehensive PDF reports with clinical insights

### Patient Interface
- **Step 1: Upload Scan** - User-friendly image upload
- **Step 2: AI Analysis** - Automated fibroid detection results
- **Step 3: Health Chat** - Interactive questionnaire for health information collection
- **Step 4: Download Report** - Personalized health reports

### Key Technologies
- **Frontend**: React 18 + TypeScript + Tailwind CSS + Framer Motion
- **Backend**: FastAPI + PyTorch + U-Net++ + XAI modules
- **UI Design**: Glassmorphic design with medical-grade aesthetics
- **AI Models**: U-Net++ with EfficientNet-B5 encoder
- **XAI**: GradCAM and Integrated Gradients for model interpretability

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+ and pip
- CUDA-compatible GPU (optional, for faster inference)

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python main.py
```

The backend API will be available at `http://localhost:8000`

## 📁 Project Structure

```
Downloads/u-net/
├── frontend/                 # React TypeScript frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── DoctorInterface.tsx
│   │   │   ├── PatientInterface.tsx
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── PredictionViewer.tsx
│   │   │   ├── XAIExplanation.tsx
│   │   │   ├── ChatInterface.tsx
│   │   │   └── PDFReport.tsx
│   │   ├── types/           # TypeScript type definitions
│   │   ├── utils/           # API utilities
│   │   └── styles/          # CSS and styling
│   ├── package.json
│   └── vite.config.ts
├── backend/                 # FastAPI backend
│   ├── main.py             # FastAPI application
│   ├── models/             # AI model interfaces
│   │   ├── unet_inference.py
│   │   └── xai_analysis.py
│   ├── utils/              # Utilities
│   │   └── pdf_generator.py
│   └── requirements.txt
├── gradcam_unet.py         # GradCAM implementation
├── integrated_gradients.py # Integrated Gradients implementation
├── run_gradcam.py          # GradCAM runner script
├── UNET-model/             # U-Net model files
└── models_*/               # Trained model checkpoints
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Model Configuration
MODEL_PATH=models_20250609_105424/best_model.pth
DEVICE=cuda  # or cpu
IMAGE_SIZE=640
ENCODER_NAME=efficientnet-b5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# File Storage
UPLOAD_DIR=uploads
RESULTS_DIR=results
REPORTS_DIR=reports
```

### Frontend Configuration

Update `vite.config.ts` for API proxy settings:

```typescript
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

## 🎨 UI Design Features

### Glassmorphic Design
- Semi-transparent backgrounds with backdrop blur
- Subtle shadows and borders
- Smooth animations and transitions
- Medical-grade color palette

### Responsive Layout
- Mobile-first design approach
- Adaptive layouts for different screen sizes
- Touch-friendly interface elements

### Accessibility
- WCAG 2.1 compliant color contrasts
- Keyboard navigation support
- Screen reader friendly markup
- Focus indicators and states

## 🤖 AI Model Integration

### U-Net++ Model
- **Architecture**: U-Net++ with EfficientNet-B5 encoder
- **Input Size**: 640x640 pixels
- **Output**: Segmentation mask for fibroid detection
- **Performance**: Dice Score ~0.89 on validation set

### Explainable AI (XAI)
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **Integrated Gradients**: Attribution-based explanations
- **Visualization**: Heatmaps and overlay images
- **Statistics**: Attention ratios and attribution metrics

## 📊 API Endpoints

### Image Management
- `POST /api/images/upload` - Upload medical image
- `GET /api/images/{image_id}` - Retrieve uploaded image

### AI Analysis
- `POST /api/predictions/unet/{image_id}` - Start U-Net prediction
- `GET /api/predictions/{prediction_id}` - Get prediction status
- `POST /api/xai/analyze/{prediction_id}` - Start XAI analysis
- `GET /api/xai/{analysis_id}` - Get XAI analysis results

### Report Generation
- `POST /api/reports/generate` - Generate PDF report
- `GET /api/reports/{report_id}` - Get report status
- `GET /api/reports/{report_id}/download` - Download PDF report

### Health Check
- `GET /api/health` - API health status

## 🔒 Security Features

### Data Protection
- Secure file upload with validation
- Temporary file storage with automatic cleanup
- No permanent storage of medical images
- HIPAA-compliant data handling practices

### API Security
- CORS protection
- Request rate limiting
- Input validation and sanitization
- Error handling without information leakage

## 🧪 Testing

### Frontend Testing
```bash
cd frontend
npm run test
npm run test:coverage
```

### Backend Testing
```bash
cd backend
python -m pytest tests/
python -m pytest --cov=. tests/
```

## 📦 Production Deployment

### Frontend Build
```bash
cd frontend
npm run build
```

### Backend Production
```bash
cd backend
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- U-Net++ architecture by Zhou et al.
- EfficientNet by Tan & Le
- GradCAM by Selvaraju et al.
- Integrated Gradients by Sundararajan et al.
- React and FastAPI communities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**⚠️ Medical Disclaimer**: This application is for research and educational purposes. All AI-generated results should be validated by qualified medical professionals. This tool is not intended for clinical diagnosis or treatment decisions.
