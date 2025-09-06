import axios, { AxiosError } from 'axios';
import {
  ApiResponse,
  MedicalImage,
  UNetPrediction,
  XAIAnalysis,
  PatientProfile,
  MedicalReport,
  ChatSession,
  UploadProgress
} from '../types';

// Type guard to check if error is an AxiosError
const isAxiosError = (error: unknown): error is AxiosError => {
  return axios.isAxiosError(error);
};

// Type for validation error details
interface ValidationErrorDetail {
  loc?: string[];
  msg: string;
  type?: string;
}

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api',
  timeout: 600000, // 10 minutes for XAI analysis (increased from 5 minutes)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens if needed
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// Image Upload API
export const uploadImage = async (
  file: File,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<MedicalImage>> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/images/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentage = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress({
            percentage,
            status: 'uploading',
            message: `Uploading... ${percentage}%`,
          });
        }
      },
    });

    // Transform backend response to MedicalImage format
    const backendData = response.data.data; // Extract the data from { success: true, data: ... }
    const medicalImage: MedicalImage = {
      id: backendData.id,
      file: file, // Keep the original File object
      url: backendData.url,
      uploadedAt: new Date(backendData.uploaded_at)
    };

    return {
      success: true,
      data: medicalImage
    };
  } catch (error) {
    console.error('Upload API error:', error);

    // Check if it's an AxiosError
    if (isAxiosError(error)) {
      // Check for specific error types
      if (error.code === 'ERR_NETWORK' || error.code === 'ERR_CONNECTION_REFUSED') {
        throw new Error('Network Error - Backend server not reachable');
      }

      if (error.response) {
        // Server responded with error status
        const status = error.response.status;
        const data = error.response.data as any;
        const message = data?.message || data?.detail || 'Server error';
        throw new Error(`Server error (${status}): ${message}`);
      }
    }

    throw new Error('Failed to upload image - Please check if backend is running');
  }
};

// U-Net Prediction API
export const runUNetPrediction = async (
  imageId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<UNetPrediction>> => {
  try {
    // Start prediction
    const response = await api.post(`/predictions/unet/${imageId}`);

    if (onProgress) {
      onProgress({
        percentage: 50,
        status: 'processing',
        message: 'Running U-Net inference...',
      });
    }

    // Poll for completion (in a real app, you might use WebSockets)
    const predictionId = response.data.data.id;
    return await pollPredictionStatus(predictionId, onProgress);
  } catch (error) {
    console.error('U-Net prediction error:', error);
    throw new Error('Failed to run U-Net prediction');
  }
};

// Poll prediction status
const pollPredictionStatus = async (
  predictionId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<UNetPrediction>> => {
  const maxAttempts = 60; // 5 minutes with 5-second intervals
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const response = await api.get(`/predictions/${predictionId}`);
      const prediction = response.data.data;

      if (prediction.status === 'completed') {
        if (onProgress) {
          onProgress({
            percentage: 100,
            status: 'completed',
            message: 'Prediction completed!',
          });
        }
        return response.data;
      } else if (prediction.status === 'error') {
        throw new Error('Prediction failed');
      }

      // Update progress
      if (onProgress) {
        const progress = Math.min(50 + (attempts / maxAttempts) * 40, 90);
        onProgress({
          percentage: progress,
          status: 'processing',
          message: 'Processing image...',
        });
      }

      // Wait 5 seconds before next poll
      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    } catch (error) {
      console.error('Prediction status check error:', error);
      throw new Error('Failed to check prediction status');
    }
  }

  throw new Error('Prediction timeout');
};

// XAI Analysis API
export const runXAIAnalysis = async (
  predictionId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<XAIAnalysis>> => {
  try {
    const response = await api.post(`/xai/analyze/${predictionId}`);

    if (onProgress) {
      onProgress({
        percentage: 30,
        status: 'processing',
        message: 'Running GradCAM analysis...',
      });
    }

    // Poll for XAI completion
    const analysisId = response.data.data.id;
    return await pollXAIStatus(analysisId, onProgress);
  } catch (error) {
    console.error('XAI analysis error:', error);
    throw new Error('Failed to run XAI analysis');
  }
};

// Poll XAI analysis status
const pollXAIStatus = async (
  analysisId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<XAIAnalysis>> => {
  const maxAttempts = 120; // Increased from 60 to 120 (10 minutes with 5-second intervals)
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const response = await api.get(`/xai/${analysisId}`);
      const analysis = response.data.data;

      if (analysis.status === 'completed') {
        if (onProgress) {
          onProgress({
            percentage: 100.0,
            status: 'completed',
            message: 'XAI analysis completed!',
          });
        }
        // Return the analysis data, not the wrapper
        return {
          success: true,
          data: analysis.analysis
        };
      } else if (analysis.status === 'error') {
        throw new Error('XAI analysis failed');
      }

      // Update progress
      if (onProgress) {
        const baseProgress = 30;
        const additionalProgress = (attempts / maxAttempts) * 60;
        const progress = Math.min(baseProgress + additionalProgress, 90);
        onProgress({
          percentage: Math.round(progress * 10) / 10, // Round to 1 decimal place
          status: 'processing',
          message: attempts < 20 ? 'Running GradCAM...' : 'Running Integrated Gradients...',
        });
      }

      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    } catch (error) {
      console.error('XAI analysis status check error:', error);
      throw new Error('Failed to check XAI analysis status');
    }
  }

  throw new Error('XAI analysis timeout');
};

// GradCAM Analysis API
export const runGradCAMAnalysis = async (
  predictionId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<XAIAnalysis>> => {
  try {
    const response = await api.post(`/xai/gradcam/${predictionId}`);

    if (onProgress) {
      onProgress({
        percentage: 10.0,
        status: 'processing',
        message: 'Starting GradCAM analysis...',
      });
    }

    // Poll for GradCAM completion
    const analysisId = response.data.data.id;
    return await pollGradCAMStatus(analysisId, onProgress);
  } catch (error) {
    console.error('GradCAM analysis error:', error);
    throw new Error('Failed to run GradCAM analysis');
  }
};

// Integrated Gradients Analysis API
export const runIntegratedGradientsAnalysis = async (
  predictionId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<XAIAnalysis>> => {
  try {
    const response = await api.post(`/xai/integrated-gradients/${predictionId}`);

    if (onProgress) {
      onProgress({
        percentage: 10.0,
        status: 'processing',
        message: 'Starting Integrated Gradients analysis...',
      });
    }

    // Poll for IG completion
    const analysisId = response.data.data.id;
    return await pollIntegratedGradientsStatus(analysisId, onProgress);
  } catch (error) {
    console.error('Integrated Gradients analysis error:', error);
    throw new Error('Failed to run Integrated Gradients analysis');
  }
};

// Poll GradCAM analysis status
const pollGradCAMStatus = async (
  analysisId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<XAIAnalysis>> => {
  const maxAttempts = 24; // 2 minutes max for GradCAM
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const response = await api.get(`/xai/${analysisId}`);
      const analysis = response.data.data;

      if (analysis.status === 'completed') {
        if (onProgress) {
          onProgress({
            percentage: 100.0,
            status: 'completed',
            message: 'GradCAM analysis completed!',
          });
        }
        // Return the analysis data, not the wrapper
        return {
          success: true,
          data: analysis.analysis
        };
      } else if (analysis.status === 'error') {
        throw new Error('GradCAM analysis failed');
      }

      // Update progress
      if (onProgress) {
        const progress = Math.min(90, 10 + (attempts / maxAttempts) * 80);
        onProgress({
          percentage: Math.round(progress * 10) / 10, // Round to 1 decimal place
          status: 'processing',
          message: 'Computing GradCAM heatmaps...',
        });
      }

      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    } catch (error) {
      console.error('Error polling GradCAM status:', error);
      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    }
  }

  throw new Error('GradCAM analysis timeout');
};

// Poll Integrated Gradients analysis status
const pollIntegratedGradientsStatus = async (
  analysisId: string,
  onProgress?: (progress: UploadProgress) => void
): Promise<ApiResponse<XAIAnalysis>> => {
  const maxAttempts = 60; // 5 minutes max for IG
  let attempts = 0;

  while (attempts < maxAttempts) {
    try {
      const response = await api.get(`/xai/${analysisId}`);
      const analysis = response.data.data;

      if (analysis.status === 'completed') {
        if (onProgress) {
          onProgress({
            percentage: 100.0,
            status: 'completed',
            message: 'Integrated Gradients analysis completed!',
          });
        }
        // Return the analysis data, not the wrapper
        return {
          success: true,
          data: analysis.analysis
        };
      } else if (analysis.status === 'error') {
        throw new Error('Integrated Gradients analysis failed');
      }

      // Update progress
      if (onProgress) {
        const progress = Math.min(90, 10 + (attempts / maxAttempts) * 80);
        onProgress({
          percentage: Math.round(progress * 10) / 10, // Round to 1 decimal place
          status: 'processing',
          message: 'Computing pixel-level attributions...',
        });
      }

      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    } catch (error) {
      console.error('Error polling IG status:', error);
      await new Promise(resolve => setTimeout(resolve, 5000));
      attempts++;
    }
  }

  throw new Error('Integrated Gradients analysis timeout');
};

// Patient Profile API
export const savePatientProfile = async (
  profile: Partial<PatientProfile>
): Promise<ApiResponse<PatientProfile>> => {
  try {
    const response = await api.post('/patients/profile', profile);
    return response.data;
  } catch (error) {
    console.error('Save patient profile error:', error);
    throw new Error('Failed to save patient profile');
  }
};

export const getPatientProfile = async (
  patientId: string
): Promise<ApiResponse<PatientProfile>> => {
  try {
    const response = await api.get(`/patients/${patientId}`);
    return response.data;
  } catch (error) {
    console.error('Get patient profile error:', error);
    throw new Error('Failed to get patient profile');
  }
};

// Chat API
export const createChatSession = async (): Promise<ApiResponse<ChatSession>> => {
  try {
    const response = await api.post('/chat/session');
    return response.data;
  } catch (error) {
    console.error('Create chat session error:', error);
    throw new Error('Failed to create chat session');
  }
};

export const sendChatMessage = async (
  sessionId: string,
  message: string
): Promise<ApiResponse<{ response: string; profileUpdate?: Partial<PatientProfile> }>> => {
  try {
    const response = await api.post(`/chat/${sessionId}/message`, { message });
    return response.data;
  } catch (error) {
    console.error('Send chat message error:', error);
    throw new Error('Failed to send chat message');
  }
};

// Report Generation API
export const generateReport = async (
  reportData: {
    patient_id?: string;
    image_id: string;
    prediction_id: string;
    xai_analysis_id?: string;
    patient_profile?: PatientProfile;
    doctor_notes?: string;
    include_integrated_gradients?: boolean;
  }
): Promise<ApiResponse<MedicalReport>> => {
  try {
    // Clean up the data to ensure proper format
    const cleanedData = {
      ...reportData,
      xai_analysis_id: reportData.xai_analysis_id || null, // Convert undefined/empty to null
    };

    const response = await api.post('/reports/generate', cleanedData);
    return response.data;
  } catch (error) {
    console.error('Generate report API error:', error);

    if (isAxiosError(error) && error.response) {
      const status = error.response.status;
      let message = 'Server error';

      // Try to extract error details
      const data = error.response.data as any;
      if (data) {
        if (typeof data === 'string') {
          message = data;
        } else if (data.detail) {
          if (Array.isArray(data.detail)) {
            // Pydantic validation errors
            message = data.detail.map((err: ValidationErrorDetail) => `${err.loc?.join('.')}: ${err.msg}`).join(', ');
          } else {
            message = data.detail;
          }
        } else if (data.message) {
          message = data.message;
        } else {
          message = JSON.stringify(data);
        }
      }

      throw new Error(`Report generation failed (${status}): ${message}`);
    }
    throw new Error('Failed to generate report');
  }
};

export const downloadReport = async (reportId: string): Promise<Blob> => {
  try {
    const response = await api.get(`/reports/${reportId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  } catch (error) {
    console.error('Download report error:', error);
    throw new Error('Failed to download report');
  }
};

// Get ultrasound image with prediction overlay
export const getPredictionOverlayImage = async (predictionId: string): Promise<string> => {
  try {
    const response = await api.get(`/predictions/${predictionId}/overlay-image`, {
      responseType: 'blob'
    });

    // Convert blob to data URL
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(response.data);
    });
  } catch (error) {
    console.error('Get overlay image API error:', error);
    throw new Error('Failed to fetch overlay image');
  }
};

// Chatbot API
export const startChatbotConversation = async (
  scanResults?: any
): Promise<ApiResponse<{
  message: string;
  conversation_id: string;
  patient_context: any;
  current_question?: any;
  questionnaire_complete: boolean;
}>> => {
  try {
    const response = await api.post('/chatbot/start', {
      scan_results: scanResults
    });
    return response.data;
  } catch (error) {
    console.error('Start chatbot conversation error:', error);
    throw new Error('Failed to start chatbot conversation');
  }
};

export const sendChatbotMessage = async (
  message: string,
  conversationId: string,
  patientContext: any,
  conversationHistory: any[],
  scanResults?: any
): Promise<ApiResponse<{
  message: string;
  conversation_id: string;
  patient_context: any;
  current_question?: any;
  questionnaire_complete: boolean;
  error: boolean;
}>> => {
  try {
    const response = await api.post('/chatbot/message', {
      message,
      conversation_id: conversationId,
      patient_context: patientContext,
      conversation_history: conversationHistory,
      scan_results: scanResults
    });
    return response.data;
  } catch (error) {
    console.error('Send chatbot message error:', error);
    throw new Error('Failed to send chatbot message');
  }
};

export const startFormBasedChatbot = async (
  formData: any,
  scanResults?: any
): Promise<ApiResponse<{
  message: string;
  conversation_id: string;
  patient_context: any;
  questionnaire_complete: boolean;
  system_prompt?: string;
}>> => {
  try {
    const response = await api.post('/chatbot/form-start', {
      form_data: formData,
      scan_results: scanResults
    });
    return response.data;
  } catch (error) {
    console.error('Start form-based chatbot error:', error);
    throw new Error('Failed to start form-based chatbot conversation');
  }
};

export const chatbotHealthCheck = async (): Promise<ApiResponse<{
  chatbot_available: boolean;
  gemini_configured: boolean;
}>> => {
  try {
    const response = await api.get('/chatbot/health');
    return response.data;
  } catch (error) {
    console.error('Chatbot health check error:', error);
    throw new Error('Chatbot health check failed');
  }
};

// Health check
export const healthCheck = async (): Promise<ApiResponse<{ status: string }>> => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check error:', error);
    throw new Error('Health check failed');
  }
};

export default api;
