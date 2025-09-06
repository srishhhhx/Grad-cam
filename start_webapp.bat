@echo off
REM Uterine Fibroids Analyzer - Web Application Startup Script for Windows
REM This script helps you start both the frontend and backend services

echo 🚀 Starting Uterine Fibroids Analyzer Web Application
echo ==================================================

REM Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo [SUCCESS] Node.js found: %NODE_VERSION%

REM Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.8+ from https://python.org/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python found: %PYTHON_VERSION%

REM Setup frontend
echo [INFO] Setting up frontend...

if not exist "frontend" (
    echo [ERROR] Frontend directory not found!
    pause
    exit /b 1
)

cd frontend

if not exist "package.json" (
    echo [ERROR] package.json not found in frontend directory!
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [INFO] Installing frontend dependencies...
    call npm install
    echo [SUCCESS] Frontend dependencies installed
) else (
    echo [INFO] Frontend dependencies already installed
)

cd ..

REM Setup backend
echo [INFO] Setting up backend...

if not exist "backend" (
    echo [ERROR] Backend directory not found!
    pause
    exit /b 1
)

cd backend

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in backend directory!
    pause
    exit /b 1
)

REM Check if we're already in the medical_imaging environment
if "%CONDA_DEFAULT_ENV%"=="medical_imaging" (
    echo [INFO] Using active conda environment: medical_imaging

    if not exist ".dependencies_installed" (
        echo [INFO] Installing backend dependencies...
        pip install -r requirements.txt
        REM Create marker file to indicate dependencies are installed
        echo. > .dependencies_installed
        echo [SUCCESS] Backend dependencies installed
    ) else (
        echo [INFO] Backend dependencies already installed
    )
) else (
    echo [ERROR] Please activate the 'medical_imaging' conda environment first:
    echo [ERROR] conda activate medical_imaging
    echo [ERROR] Then run this script again.
    pause
    exit /b 1
)

cd ..

REM Check if model files exist
if not exist "models_20250609_105424\best_model.pth" (
    if not exist "models_20250609_102709\best_model.pth" (
        echo [WARNING] No trained model found. Running in demo mode with mock predictions.
    )
)

REM Start backend server
echo [INFO] Starting backend server...
cd backend
start "Backend Server" cmd /k "python main.py"
cd ..

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo [INFO] Starting frontend server...
cd frontend
start "Frontend Server" cmd /k "npm run dev"
cd ..

REM Wait for frontend to start
timeout /t 3 /nobreak >nul

echo.
echo 🎉 Application started successfully!
echo ==================================================
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/api/docs
echo.
echo Both servers are running in separate windows.
echo Close the server windows to stop the services.
echo.

pause
