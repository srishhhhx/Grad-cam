# Uterine Fibroids Analyzer

## 📋 Table of Contents

- [Introduction](#introduction)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction

The Uterine Fibroids Analyzer is an AI-powered medical imaging analysis platform designed to assist doctors and patients in the detection and understanding of uterine fibroids. The application leverages a powerful U-Net++ model with an EfficientNet-B5 encoder for accurate segmentation of uterine fibroids from ultrasound images. It provides a user-friendly interface for both doctors and patients, with features such as image upload, AI analysis, explainable AI (XAI) insights, and comprehensive PDF report generation.

## Core Features

### For Doctors

-   **Image Upload:** Securely upload patient ultrasound images for analysis.
-   **AI-Powered Segmentation:** Utilize a state-of-the-art U-Net++ model to automatically segment and identify uterine fibroids.
-   **Explainable AI (XAI):** Gain insights into the model's decision-making process with GradCAM and Integrated Gradients analysis.
-   **Comprehensive PDF Reports:** Generate detailed, professional reports including patient information, analysis results, and XAI visualizations.
-   **Doctor's Notes:** Add clinical notes and observations to the generated reports.

### For Patients

-   **Image Upload:** Easily upload personal ultrasound scans for analysis.
-   **AI Analysis:** Receive AI-powered analysis of the uploaded scan.
-   **Interactive Health Chat:** Engage with an intelligent chatbot to answer health-related questions and provide personalized guidance.
-   **Personalized Reports:** Download a simplified, easy-to-understand report of the analysis.

## System Architecture

The application is built with a modern, decoupled architecture consisting of a FastAPI backend and a React frontend.

-   **Frontend:** A responsive and interactive user interface built with React, TypeScript, and Tailwind CSS. It provides separate interfaces for doctors and patients, ensuring a tailored user experience.
-   **Backend:** A robust RESTful API powered by FastAPI. It handles image uploads, runs the U-Net inference and XAI analysis, and generates PDF reports.
-   **AI Model:** A U-Net++ model with an EfficientNet-B5 encoder, trained for uterine fibroid segmentation. The model is integrated into the backend for seamless analysis.
-   **Chatbot Service:** A Gemini-powered chatbot service provides an intelligent conversational interface for patients.

## Technology Stack

-   **Backend:** Python, FastAPI, PyTorch, OpenCV, ReportLab
-   **Frontend:** React, TypeScript, Vite, Tailwind CSS, Framer Motion
-   **AI Model:** U-Net++, EfficientNet-B5
-   **Database:** In-memory dictionaries for demonstration purposes (can be replaced with a production-ready database).

## Getting Started

### Prerequisites

-   Python 3.8+
-   Node.js 14+
-   `pip` and `npm`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Uterine_fibroids.git
    cd Uterine_fibroids
    ```

2.  **Backend Setup:**

    ```bash
    cd backend
    pip install -r requirements.txt
    ```

3.  **Frontend Setup:**

    ```bash
    cd ../frontend
    npm install
    ```

### Running the Application

1.  **Start the backend server:**

    ```bash
    cd backend
    uvicorn main:app --reload
    ```

    The backend will be available at `http://localhost:8000`.

2.  **Start the frontend development server:**

    ```bash
    cd ../frontend
    npm run dev
    ```

    The frontend will be available at `http://localhost:3000`.

## Usage

### Doctor Interface

1.  Select the "Doctor Interface" on the home screen.
2.  Upload a patient's ultrasound image.
3.  Run the AI analysis to get the segmentation results.
4.  Perform XAI analysis (GradCAM and Integrated Gradients) to understand the model's predictions.
5.  Generate a comprehensive PDF report, with the option to add clinical notes.

### Patient Interface

1.  Select the "Patient Interface" on the home screen.
2.  Upload your ultrasound scan.
3.  Run the AI analysis to see the results.
4.  Engage with the health chatbot to get more information and guidance.
5.  Download a personalized report.

## Future Improvements

-   **Database Integration:** Replace the in-memory databases with a robust database system like PostgreSQL or MongoDB.
-   **User Authentication:** Implement a secure user authentication and authorization system.
-   **Enhanced XAI Visualizations:** Improve the XAI visualizations for better interpretability.
-   **Model Improvement:** Continuously train and improve the U-Net model with more data.
-   **Deployment:** Deploy the application to a cloud platform like AWS, Google Cloud, or Azure.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
