// User Types
export interface User {
  id: string;
  type: 'doctor' | 'patient';
  name: string;
  email: string;
}

// Medical Image Types
export interface MedicalImage {
  id: string;
  file: File;
  url: string;
  imageData?: string; // Base64 encoded image data
  uploadedAt: Date;
  patientId?: string;
}

// U-Net Prediction Types
export interface UNetPrediction {
  id: string;
  imageId: string;
  fibroidDetected: boolean;
  fibroidCount: number;
  confidence: number;
  severity: 'mild' | 'moderate' | 'severe';
  prediction: {
    mask: string; // Base64 encoded image
    probability: number[][];
    confidence: number;
    fibroidDetected: boolean;
    fibroidCount: number;
    fibroidAreas: Array<{
      area: number;
      location: { x: number; y: number };
      severity: 'mild' | 'moderate' | 'severe';
    }>;
  };
  processedAt: Date;
}

// XAI (Explainable AI) Types
export interface XAIAnalysis {
  id: string;
  predictionId: string;
  gradcam: {
    heatmap: string; // Base64 encoded heatmap
    overlayImage: string; // Base64 encoded overlay
    statistics: {
      attentionMean: number;
      attentionStd: number;
      attentionRatio: number;
      predictionRatio: number;
    };
  };
  integratedGradients: {
    attribution: string; // Base64 encoded attribution map
    channelAnalysis: string; // Base64 encoded channel-wise analysis
    statistics: {
      attributionMean: number;
      attributionStd: number;
      attributionMax: number;
      attributionMin: number;
      predictionArea: number;
      attributionRatio: number;
    };
  };
  explanation: {
    summary: string;
    keyFindings: string[];
    confidence: number;
    recommendations: string[];
  };
}

// Patient Information Types
export interface PatientDemographics {
  age: number;
  sex: 'female' | 'male';
  ethnicity?: string;
}

export interface MenstrualHistory {
  ageOfMenarche: number;
  menstrualRegularity: 'regular' | 'irregular';
  cycleLength: number;
  bleedingDuration: number;
  heavyPeriods: boolean;
  hasClots: boolean;
  hasFlooding: boolean;
}

export interface Symptoms {
  pelvicPain: boolean;
  pelvicPressure: boolean;
  heavyMenstrualBleeding: boolean;
  frequentUrination: boolean;
  constipation: boolean;
  painDuringIntercourse: boolean;
  fatigue: boolean;
  anemia: boolean;
}

export interface ReproductiveHistory {
  pregnancies: number;
  liveBirths: number;
  miscarriages: number;
  abortions: number;
  fertilityIssues: boolean;
}

export interface MedicalHistory {
  familyHistoryFibroids: boolean;
  pastFibroidDiagnosis: boolean;
  pastFibroidTreatments: string[];
  hormonalTreatmentHistory: string[];
}

export interface LifestyleFactors {
  bmi: number;
  weight: number;
  height: number;
  dietHighRedMeat: boolean;
  dietLowVegetables: boolean;
  exerciseRoutine: 'none' | 'light' | 'moderate' | 'intense';
  alcoholConsumption: 'none' | 'light' | 'moderate' | 'heavy';
  smokingHistory: 'never' | 'former' | 'current';
}

export interface PatientProfile {
  id: string;
  demographics: PatientDemographics;
  menstrualHistory: MenstrualHistory;
  symptoms: Symptoms;
  reproductiveHistory: ReproductiveHistory;
  medicalHistory: MedicalHistory;
  lifestyleFactors: LifestyleFactors;
  riskFactors?: string[]; // Additional risk factors
  createdAt: Date;
  updatedAt: Date;
}

// Chat Types
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'bot';
  content: string;
  timestamp: Date;
}

export interface ChatSession {
  id: string;
  patientId: string;
  messages: ChatMessage[];
  patientProfile?: Partial<PatientProfile>;
  isComplete: boolean;
  startedAt?: Date; // Optional start time
  createdAt: Date;
}

// Report Types
export interface MedicalReport {
  id: string;
  patientId: string;
  imageId: string;
  predictionId: string;
  xaiAnalysisId: string;
  patientProfile?: PatientProfile;
  summary: {
    fibroidDetected: boolean;
    severity: 'mild' | 'moderate' | 'severe';
    recommendations: string[];
    followUpRequired: boolean;
  };
  generatedAt: Date;
  generatedBy: string; // doctor ID
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Step Types for UI
export type DoctorStep = 1 | 2 | 3 | 4;
export type PatientStep = 1 | 2 | 3 | 4;

export interface StepStatus {
  completed: boolean;
  current: boolean;
  data?: any;
}

// Analysis Status
export type AnalysisStatus = 'idle' | 'uploading' | 'processing' | 'analyzing' | 'completed' | 'error';

// File Upload Types
export interface UploadProgress {
  percentage: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  message?: string;
}

// GradCAM Slideshow Types
export interface SlideData {
  id: string;
  title: string;
  image: string;
  description: string;
  method: 'gradcam' | 'gradcam++';
}
