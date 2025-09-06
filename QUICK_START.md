# 🚀 Quick Start Guide - Uterine Fibroids Analyzer

## Prerequisites Installation

### 1. Install Node.js and npm
- **Download**: https://nodejs.org/ (LTS version recommended)
- **Verify installation**:
  ```bash
  node --version  # Should show v18.x.x or higher
  npm --version   # Should show 9.x.x or higher
  ```

### 2. Install Python
- **Download**: https://python.org/ (3.8+ required)
- **Verify installation**:
  ```bash
  python --version  # Should show 3.8.x or higher
  # or
  python3 --version
  ```

## 🎯 One-Command Start

Once you have Node.js and Python installed, navigate to the project directory and run:

```bash
# Install all dependencies and start the application
npm run install:all && npm start
```

**That's it!** This single command will:
1. Install frontend dependencies (React, TypeScript, etc.)
2. Install backend dependencies (FastAPI, PyTorch, etc.)
3. Start both frontend and backend servers simultaneously

## 🌐 Access Points

After running `npm start`, the application will be available at:

- **🎨 Frontend (React App)**: http://localhost:3000
- **🔧 Backend API**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/api/docs

## 📋 Step-by-Step Alternative

If you prefer to run commands separately:

### Step 1: Install Dependencies
```bash
# Install frontend dependencies
npm run install:frontend

# Install backend dependencies
npm run install:backend
```

### Step 2: Start the Application
```bash
# Start both frontend and backend
npm start

# OR start them separately in different terminals:
# Terminal 1 (Frontend):
npm run start:frontend

# Terminal 2 (Backend):
npm run start:backend
```

## 🎭 Demo Mode

The application includes a **demo mode** that works without trained models:
- ✅ Mock AI predictions
- ✅ Sample XAI visualizations  
- ✅ Full UI functionality
- ✅ PDF report generation

## 🔧 Troubleshooting

### If `npm install` fails:
```bash
# Clear cache and retry
npm cache clean --force
npm run install:frontend --legacy-peer-deps
```

### If Python dependencies fail:
```bash
# Upgrade pip first
pip install --upgrade pip
# Then retry
npm run install:backend
```

### If ports are already in use:
- Frontend (3000): Edit `frontend/vite.config.ts`
- Backend (8000): Edit `backend/main.py`

## 🎯 What You'll See

### 1. Landing Page
- Choose between **Doctor Interface** or **Patient Interface**
- Modern glassmorphic design
- API status indicator

### 2. Doctor Interface (4 Steps)
1. **📤 Upload Scan**: Drag & drop medical images
2. **🤖 AI Analysis**: U-Net++ model inference
3. **🔍 XAI Explanation**: GradCAM + Integrated Gradients
4. **📄 Generate Report**: Professional PDF reports

### 3. Patient Interface (4 Steps)
1. **📤 Upload Scan**: User-friendly image upload
2. **🤖 AI Analysis**: Clear results presentation  
3. **💬 Health Chat**: Interactive medical questionnaire
4. **📥 Download Report**: Personalized health reports

## 🛠 Development Commands

```bash
# Start development servers
npm start

# Build for production
npm run build

# Install all dependencies
npm run install:all

# Setup with shell scripts (alternative)
npm run setup        # Mac/Linux
npm run setup:windows # Windows
```

## 🎨 Features Highlights

- **🎭 Glassmorphic UI**: Modern, medical-grade design
- **📱 Responsive**: Works on desktop, tablet, mobile
- **♿ Accessible**: WCAG compliant, keyboard navigation
- **🔒 Secure**: HIPAA-compliant data handling
- **🤖 AI-Powered**: U-Net++ with explainable AI
- **📊 Interactive**: Real-time progress, animations
- **📄 Professional**: Medical-grade PDF reports

## 🆘 Need Help?

1. **Check the logs** in your terminal for error messages
2. **Open browser DevTools** (F12) to see frontend errors
3. **Visit API docs** at http://localhost:8000/api/docs
4. **Check README_WEBAPP.md** for detailed documentation

---

**🏥 Medical Disclaimer**: This application is for research and educational purposes. All AI-generated results should be validated by qualified medical professionals.
