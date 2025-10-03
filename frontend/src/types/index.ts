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
  pdf: {
    structure: PDFReportStructure;
    formatting: PDFReportFormatting;
    status: PDFGenerationStatus;
    preview?: {
      dataUrl: string;
      pages: string[];
    };
  };
}

// Enhanced PDF Report Types
export interface PDFStyles {
  text: {
    fontSize: number;
    fontFamily: string;
    color: string;
    lineHeight: number;
    alignment: 'left' | 'center' | 'right' | 'justify';
  };
  section: {
    marginTop: number;
    marginBottom: number;
    padding: number;
    backgroundColor?: string;
  };
  table: {
    fontSize: number;
    cellPadding: number;
    borderColor: string;
    headerBackground: string;
  };
}

export interface PDFContentBlock {
  type: 'text' | 'table' | 'image' | 'chart' | 'separator';
  content: string | string[] | Record<string, any>;
  style: Partial<PDFStyles>;
  metadata?: {
    id: string;
    className?: string;
    renderHint?: string;
  };
}

export interface PDFPreviewState {
  loading: boolean;
  error?: string;
  currentPage: number;
  totalPages: number;
  scale: number;
  rotation: number;
}

// PDF Report Types
export interface PDFReportFormatting {
  fonts: {
    header: string;
    body: string;
    emphasis: string;
  };
  colors: {
    primary: string;
    secondary: string;
    warning: string;
    success: string;
    error: string;
  };
  spacing: {
    lineHeight: number;
    paragraphSpacing: number;
    sectionSpacing: number;
  };
}

export interface PDFReportSection {
  title: string;
  content: string | string[];
  style?: 'normal' | 'emphasis' | 'warning';
  formatting?: Partial<PDFReportFormatting>;
}

export interface PDFReportStructure {
  header: {
    title: string;
    subtitle: string;
    logo?: string;
    reportId: string;
    date: Date;
  };
  physicianInfo: {
    name: string;
    specialization: string;
    institution: string;
    contact: string;
  };
  aiModelInfo: {
    name: string;
    version: string;
    confidence: number;
  };
  scanInfo: {
    type: string;
    resolution?: string;
    date: Date;
    status: string;
  };
  findings: {
    detected: boolean;
    confidence: number;
    count: number;
    totalArea: number;
    severity: 'mild' | 'moderate' | 'severe';
    details: Array<{
      id: number;
      severity: string;
      area: number;
      location: [number, number];
    }>;
  };
  recommendations: string[];
  disclaimer: string;
  footer: {
    institution: string;
    timestamp: Date;
    contactInfo: string;
  };
  metadata: {
    version: string;
    generator: string;
    created: Date;
    modified: Date;
    author: string;
    keywords: string[];
  };
  styling: PDFStyles;
  preview?: PDFPreviewState;
  content: PDFContentBlock[];
  rendering: {
    pageSize: 'A4' | 'Letter' | 'Legal';
    orientation: 'portrait' | 'landscape';
    margins: {
      top: number;
      bottom: number;
      left: number;
      right: number;
    };
    template?: string;
    watermark?: string;
  };
}

// PDF Generation Status
export interface PDFGenerationStatus {
  status: 'idle' | 'generating' | 'completed' | 'error';
  progress: number;
  currentStep: string;
  error?: string;
  preview?: {
    url: string;
    blob?: Blob;
    base64?: string;
  };
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
