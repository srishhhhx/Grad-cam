import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Brain,
  Eye,
  FileText,
  LogOut,
  CheckCircle,
  Clock,
  Stethoscope,
  Activity,
  TrendingUp
} from 'lucide-react';
import { User, DoctorStep, MedicalImage, UNetPrediction, XAIAnalysis, MedicalReport, UploadProgress } from '../types';
import ImageUpload from './ImageUpload';
import PredictionViewer from './PredictionViewer';
import PDFReport from './PDFReport';
import GradCAMSlideshow from './GradCAMSlideshow';
import { runUNetPrediction, runGradCAMAnalysis, runIntegratedGradientsAnalysis, generateReport } from '../utils/api';

interface DoctorInterfaceProps {
  user: User;
  onLogout: () => void;
}

const DoctorInterface: React.FC<DoctorInterfaceProps> = ({ user, onLogout }) => {
  const [currentStep, setCurrentStep] = useState<DoctorStep>(1);
  const [uploadedImage, setUploadedImage] = useState<MedicalImage | null>(null);
  const [prediction, setPrediction] = useState<UNetPrediction | null>(null);
  const [xaiAnalysis, setXaiAnalysis] = useState<XAIAnalysis | null>(null);
  const [gradcamCompleted, setGradcamCompleted] = useState(false);
  const [integratedGradientsCompleted, setIntegratedGradientsCompleted] = useState(false);
  const [isRunningGradcam, setIsRunningGradcam] = useState(false);
  const [isRunningIG, setIsRunningIG] = useState(false);
  const [report, setReport] = useState<MedicalReport | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Debug logging
  console.log('🔍 DoctorInterface render - currentStep:', currentStep, 'uploadedImage:', !!uploadedImage);
  console.log('🔍 uploadedImage details:', uploadedImage);
  console.log('🔍 All state:', { currentStep, uploadedImage: !!uploadedImage, prediction: !!prediction, xaiAnalysis: !!xaiAnalysis });

  const steps = [
    { number: 1, title: 'Upload Scan', icon: Upload, description: 'Upload medical imaging scan' },
    { number: 2, title: 'AI Analysis', icon: Brain, description: 'U-Net model prediction' },
    { number: 3, title: 'XAI Explanation', icon: Eye, description: 'Explainable AI insights' },
    { number: 4, title: 'Generate Report', icon: FileText, description: 'Download PDF report' },
  ];

  const getStepStatus = (stepNumber: DoctorStep) => {
    if (stepNumber < currentStep) return 'completed';
    if (stepNumber === currentStep) return 'current';
    return 'pending';
  };

  const handleImageUpload = (image: MedicalImage) => {
    console.log('🔄 Image uploaded:', image);
    console.log('🔄 Setting current step to 2');
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
        // User will manually proceed to Step 3 if fibroids are detected
      } else {
        throw new Error(response.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setIsProcessing(false);
      setProcessingProgress(null);
    }
  };

  const handleRunGradCAM = async () => {
    if (!prediction) return;

    setIsRunningGradcam(true);
    setError(null);

    try {
      console.log('🎯 Starting GradCAM analysis using /api/xai/gradcam/ endpoint...');
      const response = await runGradCAMAnalysis(prediction.id, setProcessingProgress);

      if (response.success && response.data) {
        // Check if response.data has an 'analysis' property (nested structure)
        const analysisData = (response.data as any).analysis || response.data;
        setXaiAnalysis(analysisData);
        setGradcamCompleted(true);
        console.log('✅ GradCAM completed:', response.data);
        console.log('🔍 Full response structure:', JSON.stringify(response, null, 2));
        console.log('🎯 Analysis data:', analysisData);
        console.log('🖼️ Has gradcam property:', !!analysisData.gradcam);
        console.log('🔥 Has heatmap:', !!analysisData.gradcam?.heatmap);
      } else {
        throw new Error(response.error || 'GradCAM analysis failed');
      }
    } catch (error) {
      console.error('GradCAM analysis error:', error);
      setError(error instanceof Error ? error.message : 'GradCAM analysis failed');
    } finally {
      setIsRunningGradcam(false);
      setProcessingProgress(null);
    }
  };

  const handleRunIntegratedGradients = async () => {
    if (!prediction || !gradcamCompleted) return;

    setIsRunningIG(true);
    setError(null);

    try {
      console.log('📊 Starting Integrated Gradients analysis using /api/xai/integrated-gradients/ endpoint...');
      const response = await runIntegratedGradientsAnalysis(prediction.id, setProcessingProgress);

      if (response.success && response.data) {
        // Check if response.data has an 'analysis' property (nested structure)
        const analysisData = (response.data as any).analysis || response.data;
        setXaiAnalysis(analysisData);
        setIntegratedGradientsCompleted(true);
        console.log('✅ Integrated Gradients completed:', response.data);
        console.log('🔍 IG analysis data:', analysisData);
      } else {
        throw new Error(response.error || 'Integrated Gradients analysis failed');
      }
    } catch (error) {
      console.error('Integrated Gradients analysis error:', error);
      setError(error instanceof Error ? error.message : 'Integrated Gradients analysis failed');
    } finally {
      setIsRunningIG(false);
      setProcessingProgress(null);
    }
  };

  const handleGenerateReport = async (doctorNotes?: string) => {
    if (!uploadedImage || !prediction || !xaiAnalysis) return;

    setIsProcessing(true);
    setError(null);

    try {
      const response = await generateReport({
        image_id: uploadedImage.id,
        prediction_id: prediction.id,
        xai_analysis_id: xaiAnalysis.id,
        doctor_notes: doctorNotes,
        include_integrated_gradients: integratedGradientsCompleted, // Flag to indicate IG availability
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
    setXaiAnalysis(null);
    setGradcamCompleted(false);
    setIntegratedGradientsCompleted(false);
    setIsRunningGradcam(false);
    setIsRunningIG(false);
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
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-medical-800">Doctor Interface</h1>
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
              const status = getStepStatus(step.number as DoctorStep);
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
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 1: Upload Medical Scan</h2>
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
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 2: AI Analysis</h2>
              {!uploadedImage ? (
                <div className="text-center">
                  <p className="text-red-600 mb-6">
                    ⚠️ No uploaded image found. Please go back to Step 1.
                  </p>
                  <button
                    onClick={() => setCurrentStep(1)}
                    className="btn-secondary"
                  >
                    Back to Upload
                  </button>
                </div>
              ) : !prediction ? (
                <div className="text-center">
                  <p className="text-medical-600 mb-6">
                    Run U-Net++ model analysis to detect uterine fibroids
                  </p>
                  <p className="text-sm text-green-600 mb-4">
                    ✅ Image uploaded: {uploadedImage?.file?.name || 'Unknown file'}
                  </p>
                  <button
                    onClick={handleRunPrediction}
                    disabled={isProcessing}
                    className="btn-primary"
                  >
                    {isProcessing ? 'Running Analysis...' : 'Run AI Analysis'}
                  </button>
                </div>
              ) : (
                <PredictionViewer
                  image={uploadedImage}
                  prediction={prediction}
                  onNext={() => {
                    setCurrentStep(3);
                    // Don't auto-start XAI - let user choose when to run each method
                  }}
                />
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
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 3: Explainable AI</h2>

              <div className="space-y-6">
                {/* Introduction */}
                <div className="bg-blue-50/30 border border-blue-200/30 rounded-xl p-4">
                  <h3 className="font-medium text-blue-800 mb-2">Progressive XAI Analysis</h3>
                  <p className="text-blue-700 text-sm">
                    Run each analysis method separately to understand how the AI makes its predictions.
                  </p>
                </div>

                {/* GradCAM Section */}
                <div className="medical-glass p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h4 className="font-semibold text-medical-800 flex items-center">
                        <Eye className="w-5 h-5 mr-2" />
                        GradCAM Analysis
                      </h4>
                      <p className="text-medical-600 text-sm">
                        Shows which regions the model focuses on (Fast: ~30 seconds)
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {gradcamCompleted && (
                        <span className="text-green-600 text-sm flex items-center">
                          <CheckCircle className="w-4 h-4 mr-1" />
                          Completed
                        </span>
                      )}
                      <button
                        onClick={handleRunGradCAM}
                        disabled={isRunningGradcam || gradcamCompleted}
                        className={`btn-primary ${gradcamCompleted ? 'opacity-50' : ''}`}
                      >
                        {isRunningGradcam ? (
                          <div className="flex items-center">
                            <motion.div
                              animate={{ rotate: 360 }}
                              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                              className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                            />
                            Running...
                          </div>
                        ) : gradcamCompleted ? (
                          'Completed'
                        ) : (
                          'Run GradCAM'
                        )}
                      </button>
                    </div>
                  </div>

                  {isRunningGradcam && (
                    <div className="bg-blue-50/30 border border-blue-200/30 rounded-lg p-3">
                      <p className="text-blue-700 text-sm">
                        🎯 Generating GradCAM heatmaps and overlays...
                      </p>
                    </div>
                  )}
                </div>

                {/* Integrated Gradients Section */}
                <div className="medical-glass p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h4 className="font-semibold text-medical-800 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2" />
                        Integrated Gradients Analysis
                      </h4>
                      <p className="text-medical-600 text-sm">
                        Pixel-level attribution analysis (Slower: ~2-5 minutes)
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {integratedGradientsCompleted && (
                        <span className="text-green-600 text-sm flex items-center">
                          <CheckCircle className="w-4 h-4 mr-1" />
                          Completed
                        </span>
                      )}
                      <button
                        onClick={handleRunIntegratedGradients}
                        disabled={!gradcamCompleted || isRunningIG || integratedGradientsCompleted}
                        className={`btn-primary ${!gradcamCompleted ? 'opacity-50' : integratedGradientsCompleted ? 'opacity-50' : ''}`}
                      >
                        {isRunningIG ? (
                          <div className="flex items-center">
                            <motion.div
                              animate={{ rotate: 360 }}
                              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                              className="w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"
                            />
                            Running...
                          </div>
                        ) : integratedGradientsCompleted ? (
                          'Completed'
                        ) : !gradcamCompleted ? (
                          'Run GradCAM First'
                        ) : (
                          'Run Integrated Gradients'
                        )}
                      </button>
                    </div>
                  </div>

                  {isRunningIG && (
                    <div className="bg-blue-50/30 border border-blue-200/30 rounded-lg p-3">
                      <p className="text-blue-700 text-sm">
                        📊 Computing pixel-level attributions (this may take several minutes)...
                      </p>
                    </div>
                  )}

                  {!gradcamCompleted && (
                    <div className="bg-amber-50/30 border border-amber-200/30 rounded-lg p-3">
                      <p className="text-amber-700 text-sm">
                        ⚠️ Please complete GradCAM analysis first
                      </p>
                    </div>
                  )}
                </div>

                {/* GradCAM Results */}
                {xaiAnalysis && gradcamCompleted && (
                  <div className="medical-glass p-6">
                    <div className="flex items-center justify-between mb-6">
                      <h4 className="font-semibold text-medical-800 flex items-center">
                        <Eye className="w-5 h-5 mr-2" />
                        GradCAM Analysis Results
                      </h4>
                      <span className="text-green-600 text-sm flex items-center">
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Completed
                      </span>
                    </div>

                    {/* Interactive GradCAM Slideshow */}
                    {xaiAnalysis.gradcam ? (
                      <GradCAMSlideshow gradcamData={xaiAnalysis.gradcam} />
                    ) : (
                      <div className="bg-yellow-50/30 border border-yellow-200/30 rounded-lg p-4">
                        <p className="text-yellow-700 text-sm">
                          ⚠️ GradCAM analysis completed but no visualization data found. This might be a data structure issue.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Integrated Gradients Results */}
                {xaiAnalysis && integratedGradientsCompleted && xaiAnalysis.integratedGradients && (
                  <div className="medical-glass p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="font-semibold text-medical-800 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2" />
                        Integrated Gradients Results
                      </h4>
                      <span className="text-green-600 text-sm flex items-center">
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Completed
                      </span>
                    </div>

                    {/* IG Visualizations */}
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Attribution */}
                        <div className="relative">
                          <img
                            src={`data:image/png;base64,${xaiAnalysis.integratedGradients.attribution}`}
                            alt="Integrated Gradients attribution"
                            className="w-full h-auto rounded-lg"
                            onError={(e) => {
                              console.error('Failed to load IG attribution');
                              e.currentTarget.style.display = 'none';
                            }}
                          />
                          <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                            Attribution
                          </div>
                        </div>

                        {/* Channel Analysis */}
                        {xaiAnalysis.integratedGradients.channelAnalysis && (
                          <div className="relative">
                            <img
                              src={`data:image/png;base64,${xaiAnalysis.integratedGradients.channelAnalysis}`}
                              alt="Channel analysis"
                              className="w-full h-auto rounded-lg"
                              onError={(e) => {
                                console.error('Failed to load IG channel analysis');
                                e.currentTarget.style.display = 'none';
                              }}
                            />
                            <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                              Channel Analysis
                            </div>
                          </div>
                        )}
                      </div>

                      {/* IG Statistics */}
                      {xaiAnalysis.integratedGradients.statistics && (
                        <div className="bg-gray-50/30 border border-gray-200/30 rounded-lg p-4">
                          <h5 className="font-medium text-medical-700 mb-3">Integrated Gradients Statistics</h5>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-medical-600">Attribution Mean:</span>
                              <span className="font-medium text-medical-800">
                                {xaiAnalysis.integratedGradients.statistics.attributionMean?.toFixed(6) || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-medical-600">Attribution Std:</span>
                              <span className="font-medium text-medical-800">
                                {xaiAnalysis.integratedGradients.statistics.attributionStd?.toFixed(6) || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-medical-600">Attribution Max:</span>
                              <span className="font-medium text-medical-800">
                                {xaiAnalysis.integratedGradients.statistics.attributionMax?.toFixed(6) || 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-medical-600">Attribution Min:</span>
                              <span className="font-medium text-medical-800">
                                {xaiAnalysis.integratedGradients.statistics.attributionMin?.toFixed(6) || 'N/A'}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Next Step Button - Allow proceeding after GradCAM */}
                {gradcamCompleted && (
                  <div className="medical-glass p-6 text-center">
                    <h4 className="font-semibold text-medical-800 mb-4">
                      {integratedGradientsCompleted ? 'Complete Analysis!' : 'GradCAM Analysis Complete!'}
                    </h4>
                    <p className="text-medical-600 mb-6">
                      {integratedGradientsCompleted
                        ? 'Both GradCAM and Integrated Gradients analyses have been completed. You can now generate a comprehensive medical report.'
                        : 'GradCAM analysis is complete. You can generate a report now or run Integrated Gradients for additional insights.'
                      }
                    </p>
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                      <button
                        onClick={() => setCurrentStep(4)}
                        className="btn-primary"
                      >
                        Generate Medical Report
                      </button>
                      {!integratedGradientsCompleted && (
                        <button
                          onClick={handleRunIntegratedGradients}
                          disabled={isRunningIG}
                          className="btn-secondary"
                        >
                          {isRunningIG ? 'Running IG...' : 'Add Integrated Gradients'}
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {/* Guidance Messages */}
                {gradcamCompleted && !integratedGradientsCompleted && (
                  <div className="bg-blue-50/30 border border-blue-200/30 rounded-lg p-4">
                    <p className="text-blue-700 text-sm">
                      💡 <strong>Optional:</strong> Run Integrated Gradients analysis for pixel-level attribution insights, or proceed directly to report generation with GradCAM results.
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {currentStep === 4 && xaiAnalysis && gradcamCompleted && (
            <motion.div
              key="step4"
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              className="medical-glass p-8"
            >
              <h2 className="text-2xl font-bold text-medical-800 mb-6">Step 4: Generate Report</h2>
              <PDFReport
                image={uploadedImage!}
                prediction={prediction!}
                xaiAnalysis={xaiAnalysis}
                report={report}
                onGenerateReport={handleGenerateReport}
                isGenerating={isProcessing}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default DoctorInterface;
