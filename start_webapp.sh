#!/bin/bash

# Uterine Fibroids Analyzer - Web Application Startup Script
# This script helps you start both the frontend and backend services

set -e

echo "🚀 Starting Uterine Fibroids Analyzer Web Application"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
check_nodejs() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js found: $NODE_VERSION"
    else
        print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python found: $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version)
        print_success "Python found: $PYTHON_VERSION"
    else
        print_error "Python is not installed. Please install Python 3.8+ from https://python.org/"
        exit 1
    fi
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    if [ ! -d "frontend" ]; then
        print_error "Frontend directory not found!"
        exit 1
    fi
    
    cd frontend
    
    if [ ! -f "package.json" ]; then
        print_error "package.json not found in frontend directory!"
        exit 1
    fi
    
    if [ ! -d "node_modules" ]; then
        print_status "Installing frontend dependencies..."
        npm install
        print_success "Frontend dependencies installed"
    else
        print_status "Frontend dependencies already installed"
    fi
    
    cd ..
}

# Setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    if [ ! -d "backend" ]; then
        print_error "Backend directory not found!"
        exit 1
    fi
    
    cd backend
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in backend directory!"
        exit 1
    fi
    
    # Check if we're already in the medical_imaging environment
    if [[ "$CONDA_DEFAULT_ENV" == "medical_imaging" ]]; then
        print_status "Using active conda environment: medical_imaging"

        # Check if dependencies are already installed
        if [ ! -f ".dependencies_installed" ]; then
            print_status "Installing backend dependencies..."
            pip install -r requirements.txt
            # Create marker file to indicate dependencies are installed
            touch .dependencies_installed
            print_success "Backend dependencies installed"
        else
            print_status "Backend dependencies already installed"
        fi
    else
        print_error "Please activate the 'medical_imaging' conda environment first:"
        print_error "conda activate medical_imaging"
        print_error "Then run this script again."
        exit 1
    fi
    
    cd ..
}

# Start backend server
start_backend() {
    print_status "Starting backend server..."
    
    cd backend
    # Environment should already be activated from setup
    
    # Check if model files exist
    if [ ! -f "../models_20250609_105424/best_model.pth" ] && [ ! -f "../models_20250609_102709/best_model.pth" ]; then
        print_warning "No trained model found. Running in demo mode with mock predictions."
    fi
    
    # Start FastAPI server in background
    python main.py &
    BACKEND_PID=$!
    
    print_success "Backend server started (PID: $BACKEND_PID)"
    print_status "Backend API available at: http://localhost:8000"
    print_status "API documentation at: http://localhost:8000/api/docs"
    
    cd ..
    
    # Wait for backend to start
    sleep 3
}

# Start frontend server
start_frontend() {
    print_status "Starting frontend server..."
    
    cd frontend
    
    # Start React development server in background
    npm run dev &
    FRONTEND_PID=$!
    
    print_success "Frontend server started (PID: $FRONTEND_PID)"
    print_status "Frontend application available at: http://localhost:3000"
    
    cd ..
    
    # Wait for frontend to start
    sleep 3
}

# Cleanup function
cleanup() {
    print_status "Shutting down servers..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend server stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend server stopped"
    fi
    
    print_success "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_status "Checking system requirements..."
    check_nodejs
    check_python
    
    print_status "Setting up application..."
    setup_frontend
    setup_backend
    
    print_status "Starting services..."
    start_backend
    start_frontend
    
    echo ""
    echo "🎉 Application started successfully!"
    echo "=================================================="
    echo "Frontend: http://localhost:3000"
    echo "Backend:  http://localhost:8000"
    echo "API Docs: http://localhost:8000/api/docs"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Keep script running
    while true; do
        sleep 1
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
