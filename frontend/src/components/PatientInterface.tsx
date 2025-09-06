import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Brain,
  MessageCircle,
  FileText,
  LogOut,
  CheckCircle,
  Clock,
  Users,
  Activity,
  Heart
} from 'lucide-react';
import { User, PatientStep, MedicalImage, UNetPrediction, PatientProfile, ChatSession, MedicalReport, UploadProgress } from '../types';
import ImageUpload from './ImageUpload';
import PredictionViewer from './PredictionViewer';
import FormBasedChatInterface from './FormBasedChatInterface';
import PDFReport from './PDFReport';
import { runUNetPrediction, generateReport } from '../utils/api';

interface PatientInterfaceProps {
  user: User;
  onLogout: () => void;
}

const PatientInterface: React.FC<PatientInterfaceProps> = ({ user, onLogout }) => {
  const [currentStep, setCurrentStep] = useState<PatientStep>(1);
  const [uploadedImage, setUploadedImage] = useState<MedicalImage | null>(null);
  const [prediction, setPrediction] = useState<UNetPrediction | null>(null);
  const [patientProfile, setPatientProfile] = useState<PatientProfile | null>(null);
  const [, setChatSession] = useState<ChatSession | null>(null);
  const [report, setReport] = useState<MedicalReport | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const steps = [
    { number: 1, title: 'Upload Scan', icon: Upload, description: 'Upload your medical scan' },
    { number: 2, title: 'AI Analysis', icon: Brain, description: 'Get AI analysis results' },
    { number: 3, title: 'Health Chat', icon: MessageCircle, description: 'Interactive health questionnaire' },
    { number: 4, title: 'Download Report', icon: FileText, description: 'Get your personalized report' },
  ];

  const getStepStatus = (stepNumber: PatientStep) => {
    if (stepNumber < currentStep) return 'completed';
    if (stepNumber === currentStep) return 'current';
    return 'pending';
  };

  const handleImageUpload = (image: MedicalImage) => {
    setUploadedImage(image);
    setCurrentStep(2);
    setError(null);
  };

  const handleRunPrediction = async () => {
    if (!uploadedImage) return;

    setIsProcessing(true);
    setError(null);

    try {
      const response = await runUNetPrediction(uploadedImage.id, setProcessingProgress);

      if (response.success && response.data) {
        setPrediction(response.data);
        // Stay on Step 2 to show the prediction results
        // User will manually proceed to Step 3 if needed
      } else {
        throw new Error(response.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error instanceof Error ? error.message : 'Analysis failed');
    } finally {
      setIsProcessing(false);
      setProcessingProgress(null);
    }
  };

  const handleChatComplete = (profile: PatientProfile, session: ChatSession) => {
    setPatientProfile(profile);
    setChatSession(session);
    setCurrentStep(4);
  };

  const handleGenerateReport = async () => {
    if (!uploadedImage || !prediction) return;

    setIsProcessing(true);
    setError(null);

    try {
      const response = await generateReport({
        patient_id: user.id,
        image_id: uploadedImage.id,
        prediction_id: prediction.id,
        xai_analysis_id: undefined, // For patient interface, we don't include XAI
        patient_profile: patientProfile || undefined,
      });

      if (response.success && response.data) {
        setReport(response.data);
      } else {
        throw new Error(response.error || 'Report generation failed');
      }
    } catch (error) {
      console.error('Report generation error:', error);
      setError(error instanceof Error ? error.message : 'Report generation failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetWorkflow = () => {
    setCurrentStep(1);
    setUploadedImage(null);
    setPrediction(null);
    setPatientProfile(null);
    setChatSession(null);
    setReport(null);
    setError(null);
    setIsProcessing(false);
    setProcessingProgress(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Header */}
      <header className="medical-glass border-b border-white/20 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gradient-to-br from-medical-500 to-medical-600 rounded-xl flex items-center justify-center">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-medical-800">Patient Portal</h1>
                <p className="text-sm text-medical-600">Welcome, {user.name}</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={resetWorkflow}
                className="btn-secondary text-sm"
                disabled={isProcessing}
              >
                New Analysis
              </button>
              <button
                onClick={onLogout}
                className="flex items-center space-x-2 text-medical-600 hover:text-medical-800 transition-colors"
              >
                <LogOut className="w-5 h-5" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => {
              const status = getStepStatus(step.number as PatientStep);
              const Icon = step.icon;

              return (
                <React.Fragment key={step.number}>
                  <div className="flex flex-col items-center">
                    <motion.div
                      className={`step-indicator ${status === 'completed' ? 'step-completed' :
                        status === 'current' ? 'step-current' : 'step-pending'
                        }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {status === 'completed' ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : status === 'current' && isProcessing ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        >
                          <Activity className="w-5 h-5" />
                        </motion.div>
                      ) : (
                        <Icon className="w-5 h-5" />
                      )}
                    </motion.div>
                    <div className="mt-2 text-center">
                      <p className="text-sm font-medium text-medical-800">{step.title}</p>
                      <p className="text-xs text-medical-600">{step.description}</p>
                    </div>
                  </div>

                  {index < steps.length - 1 && (
                    <div className="flex-1 h-px bg-gradient-to-r from-medical-200 to-medical-300 mx-4" />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mb-6 bg-red-50/50 border border-red-200/50 rounded-xl p-4"
            >
              <div className="flex items-center text-red-700">
                <Clock className="w-5 h-5 mr-2" />
                <span className="font-medium">Error</span>
              </div>
              <p className="text-red-600 text-sm mt-1">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Processing Progress */}
        <AnimatePresence>
          {processingProgress && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mb-6 medical-glass p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="w-8 h-8 border-4 border-primary-200 border-t-primary-600 rounded-full"
                  />
                  <div>
                    <h3 className="font-semibold text-medical-800">Processing</h3>
                    <p className="text-sm text-medical-600">{processingProgress.message}</p>
                  </div>
                </div>
                <span className="text-lg font-bold text-primary-600">
                  {processingProgress.percentage.toFixed(1)}%
                </span>
              </div>
              <div className="progress-bar">
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${processingProgress.percentage}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step Content */}
        <AnimatePresence mode="wait">
          {currentStep === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              className="medical-glass p-8"
            >
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 1: Upload Your Medical Scan</h2>
              <div className="mb-6 bg-blue-50/30 border border-blue-200/30 rounded-xl p-4">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Users className="w-4 h-4 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-medium text-blue-800 mb-1">Privacy & Security</h3>
                    <p className="text-blue-700 text-sm">
                      Your medical images are processed securely and are not stored permanently.
                      All analysis is performed with the highest privacy standards.
                    </p>
                  </div>
                </div>
              </div>
              <ImageUpload
                onImageUploaded={handleImageUpload}
                onError={setError}
                disabled={isProcessing}
              />
            </motion.div>
          )}

          {currentStep === 2 && uploadedImage && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              className="medical-glass p-8"
            >
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 2: AI Analysis Results</h2>
              {!prediction ? (
                <div className="text-center">
                  <div className="mb-6 bg-amber-50/30 border border-amber-200/30 rounded-xl p-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                        <Brain className="w-4 h-4 text-amber-600" />
                      </div>
                      <div>
                        <h3 className="font-medium text-amber-800 mb-1">AI Analysis</h3>
                        <p className="text-amber-700 text-sm">
                          Our advanced U-Net++ model will analyze your scan for uterine fibroids.
                          This process typically takes 1-2 minutes.
                        </p>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={handleRunPrediction}
                    disabled={isProcessing}
                    className="btn-primary"
                  >
                    {isProcessing ? 'Analyzing Your Scan...' : 'Start AI Analysis'}
                  </button>
                </div>
              ) : (
                <div>
                  <PredictionViewer
                    image={uploadedImage}
                    prediction={prediction}
                    onNext={prediction.prediction.fibroidDetected ? () => setCurrentStep(3) : undefined}
                  />
                  {prediction.prediction.fibroidDetected && (
                    <div className="mt-6 text-center">
                      <button
                        onClick={() => setCurrentStep(3)}
                        className="btn-primary"
                      >
                        Continue to Health Chat
                      </button>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {currentStep === 3 && prediction && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              className="medical-glass p-8"
            >
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 3: Health Information Chat</h2>
              <FormBasedChatInterface
                prediction={prediction}
                onComplete={handleChatComplete}
              />
            </motion.div>
          )}

          {currentStep === 4 && prediction && (
            <motion.div
              key="step4"
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              className="medical-glass p-8"
            >
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 4: Your Personalized Report</h2>
              <div className="mb-6 bg-green-50/30 border border-green-200/30 rounded-xl p-4">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                    <FileText className="w-4 h-4 text-green-600" />
                  </div>
                  <div>
                    <h3 className="font-medium text-green-800 mb-1">Your Report</h3>
                    <p className="text-green-700 text-sm">
                      Based on your scan analysis and health information, we've created a personalized report
                      with recommendations and next steps.
                    </p>
                  </div>
                </div>
              </div>

              {!report ? (
                <div className="text-center">
                  <button
                    onClick={handleGenerateReport}
                    disabled={isProcessing}
                    className="btn-primary"
                  >
                    {isProcessing ? 'Generating Your Report...' : 'Generate My Report'}
                  </button>
                </div>
              ) : (
                <PDFReport
                  image={uploadedImage!}
                  prediction={prediction}
                  xaiAnalysis={null as any} // Patient interface doesn't include XAI
                  report={report}
                  onGenerateReport={() => { }}
                  isGenerating={false}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default PatientInterface;
