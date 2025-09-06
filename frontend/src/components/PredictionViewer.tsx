import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Download,
  AlertTriangle,
  CheckCircle,
  Info,
  ArrowRight
} from 'lucide-react';
import { MedicalImage, UNetPrediction } from '../types';

interface PredictionViewerProps {
  image: MedicalImage;
  prediction: UNetPrediction;
  onNext?: () => void;
}

const PredictionViewer: React.FC<PredictionViewerProps> = ({
  image,
  prediction,
  onNext
}) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [selectedView, setSelectedView] = useState<'original' | 'prediction' | 'overlay'>('overlay');
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5));
  const handleRotate = () => setRotation(prev => (prev + 90) % 360);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'mild': return 'text-green-600 bg-green-100/50';
      case 'moderate': return 'text-yellow-600 bg-yellow-100/50';
      case 'severe': return 'text-red-600 bg-red-100/50';
      default: return 'text-gray-600 bg-gray-100/50';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'mild': return <CheckCircle className="w-4 h-4" />;
      case 'moderate': return <Info className="w-4 h-4" />;
      case 'severe': return <AlertTriangle className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  // If no fibroids detected, show cheerful message
  if (!prediction.prediction.fibroidDetected) {
    return (
      <div className="space-y-6">
        {/* Cheerful No Fibroids Message */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="medical-glass p-8 text-center"
        >
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center">
              <CheckCircle className="w-12 h-12 text-green-600" />
            </div>
          </div>
          <h3 className="text-2xl font-bold text-green-700 mb-4">
            🎉 Great News! No Fibroids Detected
          </h3>
          <p className="text-lg text-medical-700 mb-4">
            Your scan shows no abnormalities indicating uterine fibroids.
          </p>
          <p className="text-medical-600 mb-6">
            The AI analysis found no signs of fibroids in your medical image.
            This is a positive result that suggests healthy uterine tissue.
          </p>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
            <div className="flex items-center justify-center space-x-4 text-sm text-green-800">
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 mr-2" />
                <span>AI Confidence: {(prediction.prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 mr-2" />
                <span>Analysis Complete</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Image Display for No Fibroids */}
        <div className="medical-glass p-6">
          <h3 className="text-lg font-semibold text-medical-800 mb-4">Your Scan Results</h3>
          <div className="relative bg-black/5 rounded-xl overflow-hidden" style={{ height: '400px' }}>
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                style={{
                  transform: `scale(${zoom}) rotate(${rotation}deg)`,
                }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className="relative"
              >
                <img
                  src={image.url}
                  alt="Medical scan - No fibroids detected"
                  className="max-w-full max-h-full object-contain"
                />
              </motion.div>
            </div>

            {/* Clean Scan Overlay */}
            <div className="absolute top-4 left-4 bg-green-600/90 text-white px-4 py-2 rounded-lg text-sm font-medium">
              ✅ Clean Scan - No Abnormalities
            </div>

            {/* Controls */}
            <div className="absolute top-4 right-4 flex items-center space-x-1 bg-white/20 rounded-lg p-1">
              <button
                onClick={handleZoomOut}
                className="p-2 hover:bg-white/30 rounded transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4 text-medical-600" />
              </button>
              <span className="text-sm text-medical-600 px-2">
                {Math.round(zoom * 100)}%
              </span>
              <button
                onClick={handleZoomIn}
                className="p-2 hover:bg-white/30 rounded transition-colors"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4 text-medical-600" />
              </button>
              <button
                onClick={handleRotate}
                className="p-2 hover:bg-white/30 rounded transition-colors"
                title="Rotate"
              >
                <RotateCw className="w-4 h-4 text-medical-600" />
              </button>
            </div>
          </div>
        </div>

        {/* Next Steps for Clean Scan */}
        <div className="medical-glass p-6">
          <h3 className="text-lg font-semibold text-medical-800 mb-4">What This Means</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-medium text-green-800 mb-2">✅ Healthy Results</h4>
              <p className="text-sm text-green-700">
                No uterine fibroids were detected in your scan, indicating healthy uterine tissue.
              </p>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-medium text-blue-800 mb-2">📋 Next Steps</h4>
              <p className="text-sm text-blue-700">
                Continue with regular check-ups as recommended by your healthcare provider.
              </p>
            </div>
          </div>
        </div>

        {/* Action Buttons for Clean Scan */}
        <div className="flex items-center justify-between">
          <button className="btn-secondary">
            <Download className="w-4 h-4 mr-2" />
            Download Clean Scan Report
          </button>

          <button
            onClick={() => window.location.reload()}
            className="btn-primary bg-green-600 hover:bg-green-700"
          >
            Analyze Another Scan
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Analysis Summary */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="medical-glass p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-medical-600">Detection Status</span>
            {prediction.prediction.fibroidDetected ? (
              <AlertTriangle className="w-5 h-5 text-red-500" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green-500" />
            )}
          </div>
          <p className="text-lg font-bold text-medical-800">
            {prediction.prediction.fibroidDetected ? 'Fibroids Detected' : 'No Fibroids Detected'}
          </p>
          <p className="text-sm text-medical-600">
            Confidence: {(prediction.prediction.confidence * 100).toFixed(1)}%
          </p>
        </div>

        <div className="medical-glass p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-medical-600">Fibroid Count</span>
            <Info className="w-5 h-5 text-primary-500" />
          </div>
          <p className="text-lg font-bold text-medical-800">
            {prediction.prediction.fibroidCount}
          </p>
          <p className="text-sm text-medical-600">
            {prediction.prediction.fibroidCount === 0 ? 'None detected' :
              prediction.prediction.fibroidCount === 1 ? 'Single fibroid' : 'Multiple fibroids'}
          </p>
        </div>

        <div className="medical-glass p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-medical-600">Overall Severity</span>
            {prediction.prediction.fibroidAreas.length > 0 &&
              getSeverityIcon(prediction.prediction.fibroidAreas[0].severity)}
          </div>
          <p className="text-lg font-bold text-medical-800">
            {prediction.prediction.fibroidAreas.length > 0 ?
              prediction.prediction.fibroidAreas[0].severity.charAt(0).toUpperCase() +
              prediction.prediction.fibroidAreas[0].severity.slice(1) : 'N/A'}
          </p>
          <p className="text-sm text-medical-600">
            Based on size and location
          </p>
        </div>
      </div>

      {/* Image Viewer */}
      <div className="medical-glass p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-medical-800">Medical Image Analysis</h3>
            {prediction.prediction.fibroidDetected && (
              <p className="text-sm text-medical-600 mt-1">
                💡 Switch to "Overlay" view to see AI-detected fibroids highlighted on your scan
              </p>
            )}
          </div>
          <div className="flex items-center space-x-2">
            {/* View Toggle */}
            <div className="flex bg-white/20 rounded-lg p-1">
              {[
                { key: 'original', label: 'Original' },
                { key: 'prediction', label: 'Prediction' },
                { key: 'overlay', label: 'Overlay' }
              ].map((view) => (
                <button
                  key={view.key}
                  onClick={() => setSelectedView(view.key as any)}
                  className={`px-3 py-1 text-sm rounded transition-all duration-200 ${selectedView === view.key
                    ? 'bg-primary-500 text-white shadow-md'
                    : 'text-medical-600 hover:bg-white/30'
                    }`}
                >
                  {view.label}
                </button>
              ))}
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-2">
              {/* Overlay Toggle */}
              {selectedView === 'overlay' && (
                <div className="flex items-center space-x-2 bg-white/20 rounded-lg p-2">
                  <button
                    onClick={() => setShowOverlay(!showOverlay)}
                    className={`p-1 rounded transition-colors ${showOverlay ? 'bg-primary-500 text-white' : 'text-medical-600 hover:bg-white/30'
                      }`}
                    title="Toggle Overlay"
                  >
                    {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                  </button>
                  {showOverlay && (
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                      className="w-16 h-2 bg-white/30 rounded-lg appearance-none cursor-pointer"
                      title="Overlay Opacity"
                    />
                  )}
                </div>
              )}

              {/* Zoom and Rotate Controls */}
              <div className="flex items-center space-x-1 bg-white/20 rounded-lg p-1">
                <button
                  onClick={handleZoomOut}
                  className="p-2 hover:bg-white/30 rounded transition-colors"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-4 h-4 text-medical-600" />
                </button>
                <span className="text-sm text-medical-600 px-2">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  onClick={handleZoomIn}
                  className="p-2 hover:bg-white/30 rounded transition-colors"
                  title="Zoom In"
                >
                  <ZoomIn className="w-4 h-4 text-medical-600" />
                </button>
                <button
                  onClick={handleRotate}
                  className="p-2 hover:bg-white/30 rounded transition-colors"
                  title="Rotate"
                >
                  <RotateCw className="w-4 h-4 text-medical-600" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Image Display */}
        <div className="relative bg-black/5 rounded-xl overflow-hidden" style={{ height: '500px' }}>
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              style={{
                transform: `scale(${zoom}) rotate(${rotation}deg)`,
              }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="relative"
            >
              {/* Original Image - Always render as base */}
              <img
                src={image.url}
                alt="Medical scan"
                className="max-w-full max-h-full object-contain block"
                style={{
                  display: selectedView === 'prediction' ? 'none' : 'block'
                }}
              />

              {/* Prediction Mask Only */}
              {selectedView === 'prediction' && (
                <img
                  src={`data:image/png;base64,${prediction.prediction.mask}`}
                  alt="AI Prediction mask"
                  className="max-w-full max-h-full object-contain block"
                />
              )}

              {/* Segmentation Overlay - Enhanced visibility */}
              {selectedView === 'overlay' && prediction.prediction.mask && showOverlay && (
                <>
                  {/* Primary overlay - Red/Orange for fibroids */}
                  <div className="absolute inset-0 pointer-events-none">
                    <img
                      src={`data:image/png;base64,${prediction.prediction.mask}`}
                      alt="Segmentation overlay"
                      className="w-full h-full object-contain"
                      style={{
                        mixBlendMode: 'multiply',
                        opacity: overlayOpacity,
                        filter: 'hue-rotate(0deg) saturate(3) brightness(2) contrast(2)',
                        background: 'transparent'
                      }}
                    />
                  </div>

                  {/* Secondary overlay - Enhanced contrast */}
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      background: `url(data:image/png;base64,${prediction.prediction.mask}) center/contain no-repeat`,
                      mixBlendMode: 'screen',
                      opacity: overlayOpacity * 0.5,
                      filter: 'hue-rotate(15deg) saturate(2) brightness(1.8)',
                    }}
                  />
                </>
              )}
            </motion.div>
          </div>

          {/* Image Info Overlay */}
          <div className="absolute top-4 left-4 bg-black/50 text-white px-3 py-2 rounded-lg text-sm">
            {selectedView === 'original' && 'Original Scan'}
            {selectedView === 'prediction' && 'AI Prediction'}
            {selectedView === 'overlay' && 'Overlay View'}
          </div>

          {/* Fibroid Markers */}
          {selectedView !== 'prediction' && prediction.prediction.fibroidAreas.map((fibroid, index) => (
            <motion.div
              key={index}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: index * 0.2 }}
              className="absolute"
              style={{
                left: `${fibroid.location.x}%`,
                top: `${fibroid.location.y}%`,
                transform: 'translate(-50%, -50%)',
              }}
            >
              <div className={`w-6 h-6 rounded-full border-2 border-white shadow-lg ${fibroid.severity === 'mild' ? 'bg-green-500' :
                fibroid.severity === 'moderate' ? 'bg-yellow-500' : 'bg-red-500'
                }`}>
                <div className="w-full h-full rounded-full animate-ping opacity-75 bg-current" />
              </div>
              <div className="absolute top-8 left-1/2 transform -translate-x-1/2 bg-black/75 text-white px-2 py-1 rounded text-xs whitespace-nowrap">
                {fibroid.severity} • {fibroid.area.toFixed(1)}mm²
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Detailed Findings */}
      {prediction.prediction.fibroidAreas.length > 0 && (
        <div className="medical-glass p-6">
          <h3 className="text-lg font-semibold text-medical-800 mb-4">Detailed Findings</h3>
          <div className="space-y-3">
            {prediction.prediction.fibroidAreas.map((fibroid, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-4 bg-white/20 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-4 h-4 rounded-full ${fibroid.severity === 'mild' ? 'bg-green-500' :
                    fibroid.severity === 'moderate' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                  <div>
                    <p className="font-medium text-medical-800">
                      Fibroid #{index + 1}
                    </p>
                    <p className="text-sm text-medical-600">
                      Location: ({fibroid.location.x.toFixed(1)}, {fibroid.location.y.toFixed(1)})
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-sm font-medium px-2 py-1 rounded ${getSeverityColor(fibroid.severity)}`}>
                    {fibroid.severity.charAt(0).toUpperCase() + fibroid.severity.slice(1)}
                  </p>
                  <p className="text-sm text-medical-600 mt-1">
                    {fibroid.area.toFixed(1)} mm²
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center justify-between">
        <button className="btn-secondary">
          <Download className="w-4 h-4 mr-2" />
          Download Results
        </button>

        {onNext && (
          <button onClick={onNext} className="btn-primary">
            Continue to XAI Analysis
            <ArrowRight className="w-4 h-4 ml-2" />
          </button>
        )}
      </div>
    </div>
  );
};

export default PredictionViewer;
