import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Eye,
  Brain,
  TrendingUp,
  Info,
  ArrowRight,
  Lightbulb,
  Target,
  BarChart3,
  Zap
} from 'lucide-react';
import { XAIAnalysis } from '../types';

interface XAIExplanationProps {
  analysis: XAIAnalysis;
  onNext?: () => void;
}

const XAIExplanation: React.FC<XAIExplanationProps> = ({ analysis, onNext }) => {
  const [selectedMethod, setSelectedMethod] = useState<'gradcam' | 'integrated_gradients'>('gradcam');
  const [showStatistics, setShowStatistics] = useState(false);

  // Debug logging
  console.log('🔍 XAI Analysis received:', analysis);
  console.log('🎯 GradCAM data:', analysis?.gradcam);
  console.log('📊 IG data:', analysis?.integratedGradients);

  const methods = [
    {
      key: 'gradcam',
      title: 'GradCAM',
      icon: Eye,
      description: 'Gradient-weighted Class Activation Mapping',
      explanation: 'Shows which regions the model focuses on when making predictions. Warmer colors indicate higher importance.',
    },
    {
      key: 'integrated_gradients',
      title: 'Integrated Gradients',
      icon: TrendingUp,
      description: 'Attribution-based explanation method',
      explanation: 'Reveals how each pixel contributes to the final prediction by integrating gradients along a path.',
    },
  ];

  const currentMethod = methods.find(m => m.key === selectedMethod)!;

  return (
    <div className="space-y-6">
      {/* Method Selection */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-medical-800 mb-2">
            Explainable AI Analysis
          </h3>
          <p className="text-medical-600">
            Understanding how the AI model makes its predictions
          </p>
        </div>

        <div className="flex bg-white/20 rounded-lg p-1">
          {methods.map((method) => {
            const Icon = method.icon;
            return (
              <button
                key={method.key}
                onClick={() => setSelectedMethod(method.key as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded transition-all duration-200 ${selectedMethod === method.key
                  ? 'bg-primary-500 text-white shadow-md'
                  : 'text-medical-600 hover:bg-white/30'
                  }`}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{method.title}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Method Explanation */}
      <motion.div
        key={selectedMethod}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="xai-explanation"
      >
        <div className="flex items-start space-x-3">
          <div className="w-8 h-8 bg-amber-100 rounded-lg flex items-center justify-center flex-shrink-0">
            <Lightbulb className="w-5 h-5 text-amber-600" />
          </div>
          <div>
            <h4 className="font-semibold text-medical-800 mb-1">
              {currentMethod.title} - {currentMethod.description}
            </h4>
            <p className="text-medical-600 text-sm">
              {currentMethod.explanation}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Visualization */}
      <div className="medical-glass p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-medical-800">
            {currentMethod.title} Visualization
          </h4>
          <button
            onClick={() => setShowStatistics(!showStatistics)}
            className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 transition-colors"
          >
            <BarChart3 className="w-4 h-4" />
            <span className="text-sm">
              {showStatistics ? 'Hide' : 'Show'} Statistics
            </span>
          </button>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Main Visualization */}
          <div className="space-y-4">
            <div className="xai-heatmap">
              {selectedMethod === 'gradcam' && analysis.gradcam?.heatmap ? (
                <img
                  src={`data:image/png;base64,${analysis.gradcam.heatmap}`}
                  alt="GradCAM heatmap"
                  className="w-full h-auto"
                  onError={(e) => {
                    console.error('Failed to load GradCAM heatmap');
                    e.currentTarget.style.display = 'none';
                  }}
                />
              ) : selectedMethod === 'integrated_gradients' && analysis.integratedGradients?.attribution ? (
                <img
                  src={`data:image/png;base64,${analysis.integratedGradients.attribution}`}
                  alt="Integrated Gradients attribution"
                  className="w-full h-auto"
                  onError={(e) => {
                    console.error('Failed to load IG attribution');
                    e.currentTarget.style.display = 'none';
                  }}
                />
              ) : (
                <div className="w-full h-64 bg-gray-200 rounded-lg flex items-center justify-center">
                  <p className="text-gray-500">
                    {selectedMethod === 'gradcam' ? 'GradCAM' : 'Integrated Gradients'} visualization loading...
                  </p>
                </div>
              )}
              <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                {currentMethod.title}
              </div>
            </div>

            {selectedMethod === 'gradcam' && analysis.gradcam?.overlayImage && (
              <div className="xai-heatmap">
                <img
                  src={`data:image/png;base64,${analysis.gradcam.overlayImage}`}
                  alt="GradCAM overlay"
                  className="w-full h-auto"
                  onError={(e) => {
                    console.error('Failed to load GradCAM overlay');
                    e.currentTarget.style.display = 'none';
                  }}
                />
                <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                  Overlay
                </div>
              </div>
            )}

            {selectedMethod === 'integrated_gradients' && analysis.integratedGradients?.channelAnalysis && (
              <div className="xai-heatmap">
                <img
                  src={`data:image/png;base64,${analysis.integratedGradients.channelAnalysis}`}
                  alt="Channel analysis"
                  className="w-full h-auto"
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

          {/* Statistics and Insights */}
          <div className="space-y-4">
            <AnimatePresence>
              {showStatistics && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-white/20 rounded-xl p-4"
                >
                  <h5 className="font-semibold text-medical-800 mb-3 flex items-center">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Statistical Analysis
                  </h5>

                  {selectedMethod === 'gradcam' && analysis.gradcam?.statistics && (
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attention Mean:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.gradcam.statistics.attentionMean?.toFixed(6) || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attention Std:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.gradcam.statistics.attentionStd.toFixed(6)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attention Ratio:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.gradcam.statistics.attentionRatio.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Prediction Ratio:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.gradcam.statistics.predictionRatio.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  )}

                  {selectedMethod === 'integrated_gradients' && analysis.integratedGradients?.statistics && (
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attribution Mean:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.integratedGradients.statistics.attributionMean?.toFixed(6) || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attribution Std:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.integratedGradients.statistics.attributionStd.toFixed(6)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attribution Max:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.integratedGradients.statistics.attributionMax.toFixed(6)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Prediction Area:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.integratedGradients.statistics.predictionArea.toFixed(0)} pixels
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-medical-600">Attribution Ratio:</span>
                        <span className="font-medium text-medical-800">
                          {analysis.integratedGradients.statistics.attributionRatio.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Key Insights */}
            <div className="bg-white/20 rounded-xl p-4">
              <h5 className="font-semibold text-medical-800 mb-3 flex items-center">
                <Target className="w-4 h-4 mr-2" />
                Key Insights
              </h5>
              <div className="space-y-2">
                {analysis.explanation.keyFindings.map((finding, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-start space-x-2"
                  >
                    <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0" />
                    <p className="text-medical-700 text-sm">{finding}</p>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Confidence Score */}
            <div className="bg-gradient-to-r from-primary-50/50 to-primary-100/30 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-medical-800 flex items-center">
                  <Zap className="w-4 h-4 mr-2" />
                  Explanation Confidence
                </span>
                <span className="text-2xl font-bold text-primary-600">
                  {(analysis.explanation.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="progress-bar">
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${analysis.explanation.confidence * 100}%` }}
                  transition={{ duration: 1, delay: 0.5 }}
                />
              </div>
              <p className="text-medical-600 text-sm mt-2">
                How confident the AI is in its explanation
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary and Recommendations */}
      <div className="medical-glass p-6">
        <h4 className="font-semibold text-medical-800 mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2" />
          AI Explanation Summary
        </h4>

        <div className="bg-white/20 rounded-xl p-4 mb-4">
          <p className="text-medical-700 leading-relaxed">
            {analysis.explanation.summary}
          </p>
        </div>

        {analysis.explanation.recommendations.length > 0 && (
          <div>
            <h5 className="font-medium text-medical-800 mb-3">Clinical Recommendations:</h5>
            <div className="space-y-2">
              {analysis.explanation.recommendations.map((recommendation, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-start space-x-3 p-3 bg-blue-50/30 rounded-lg"
                >
                  <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <p className="text-medical-700 text-sm">{recommendation}</p>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-medical-600">
          <p>XAI analysis provides transparency into AI decision-making</p>
          <p>Use these insights to validate and understand the model's predictions</p>
        </div>

        {onNext && (
          <button onClick={onNext} className="btn-primary">
            Generate Report
            <ArrowRight className="w-4 h-4 ml-2" />
          </button>
        )}
      </div>
    </div>
  );
};

export default XAIExplanation;
