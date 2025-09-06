import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Maximize2, X, Eye, Zap } from 'lucide-react';
import { SlideData } from '../types';

interface GradCAMData {
  heatmap: string;
  overlayImage: string;
  statistics: {
    attentionMean: number;
    attentionStd: number;
    attentionRatio: number;
  };
}

interface GradCAMSlideshowProps {
  gradcamData: GradCAMData;
}



const GradCAMSlideshow: React.FC<GradCAMSlideshowProps> = ({ gradcamData }) => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [expandedImage, setExpandedImage] = useState<string | null>(null);

  // Create slide data based on available images
  // We have heatmap (gradcam) and overlayImage (gradcam++) from the backend
  // Model prediction images have been removed for cleaner visualization
  const slides: SlideData[] = [
    // GradCAM Set (using heatmap)
    {
      id: 'gradcam-heatmap',
      title: 'GradCAM Analysis',
      image: gradcamData.heatmap,
      description: 'Standard GradCAM attention heatmap showing model focus areas without prediction overlay',
      method: 'gradcam' as const
    },
    // GradCAM++ Set (using overlayImage)
    {
      id: 'gradcam++-overlay',
      title: 'GradCAM++ Enhanced',
      image: gradcamData.overlayImage,
      description: 'GradCAM++ enhanced attention visualization with improved localization accuracy',
      method: 'gradcam++' as const
    }
  ].filter(slide => slide.image); // Only include slides with valid images

  const currentMethod = slides[currentSlide]?.method;
  const methodSlides = slides.filter(slide => slide.method === currentMethod);
  const currentMethodIndex = methodSlides.findIndex(slide => slide.id === slides[currentSlide].id);

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const goToSlide = (index: number) => {
    setCurrentSlide(index);
  };

  const expandImage = (image: string) => {
    setExpandedImage(image);
  };

  const closeExpanded = () => {
    setExpandedImage(null);
  };

  return (
    <div className="space-y-6">
      {/* Method Indicator */}
      <div className="flex items-center justify-center space-x-6 mb-2">
        {slides.map((slide, index) => (
          <motion.div
            key={slide.id}
            className={`px-6 py-3 rounded-full cursor-pointer transition-all duration-300 ${index === currentSlide
              ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white shadow-lg scale-105'
              : 'bg-white/30 backdrop-blur-sm border border-white/40 text-medical-700 hover:bg-white/40'
              }`}
            onClick={() => goToSlide(index)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="font-medium flex items-center">
              {slide.method === 'gradcam' ? (
                <Eye className="w-4 h-4 mr-2" />
              ) : (
                <Zap className="w-4 h-4 mr-2" />
              )}
              {slide.method === 'gradcam' ? 'GradCAM' : 'GradCAM++'}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Current Method Description */}
      <div className="text-center mb-4">
        <p className="text-sm text-medical-600">
          {slides[currentSlide]?.method === 'gradcam'
            ? 'Standard gradient-based attention visualization'
            : 'Enhanced gradient-based visualization with improved localization'
          }
        </p>
      </div>

      {/* Main Slideshow */}
      <div className="relative">
        {/* Glassmorphism Container */}
        <div className="bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl p-6 shadow-xl">
          <div className="relative overflow-hidden rounded-xl">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentSlide}
                initial={{ opacity: 0, x: 300 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -300 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
                className="relative"
              >
                <div
                  className="relative cursor-pointer group"
                  onClick={() => expandImage(slides[currentSlide].image)}
                >
                  <img
                    src={`data:image/png;base64,${slides[currentSlide].image}`}
                    alt={slides[currentSlide].title}
                    className="w-full h-80 object-cover rounded-lg transition-transform duration-300 group-hover:scale-105"
                    onError={(e) => {
                      console.error(`Failed to load image: ${slides[currentSlide].title}`);
                      e.currentTarget.style.display = 'none';
                    }}
                  />

                  {/* Hover Overlay */}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-300 rounded-lg flex items-center justify-center">
                    <Maximize2 className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  </div>

                  {/* Image Label */}
                  <div className="absolute top-3 right-3 bg-black/70 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm font-medium">
                    {slides[currentSlide].title}
                  </div>

                  {/* Method Badge */}
                  <div className={`absolute top-3 left-3 px-3 py-1 rounded-full text-xs font-medium ${currentMethod === 'gradcam'
                    ? 'bg-blue-500/80 text-white'
                    : 'bg-purple-500/80 text-white'
                    }`}>
                    {currentMethod.toUpperCase()}
                  </div>
                </div>

                {/* Description */}
                <div className="mt-4 text-center">
                  <h3 className="text-lg font-semibold text-medical-800 mb-2">
                    {slides[currentSlide].title}
                  </h3>
                  <p className="text-medical-600 text-sm">
                    {slides[currentSlide].description}
                  </p>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center justify-between mt-6">
            <button
              onClick={prevSlide}
              className="p-2 rounded-full bg-white/30 backdrop-blur-sm border border-white/40 hover:bg-white/40 transition-all duration-200"
            >
              <ChevronLeft className="w-5 h-5 text-medical-700" />
            </button>

            {/* Slide Indicators */}
            <div className="flex space-x-2">
              {slides.map((_, index) => (
                <button
                  key={index}
                  onClick={() => goToSlide(index)}
                  className={`w-3 h-3 rounded-full transition-all duration-200 ${index === currentSlide
                    ? 'bg-primary-500 scale-125'
                    : 'bg-white/40 hover:bg-white/60'
                    }`}
                />
              ))}
            </div>

            <button
              onClick={nextSlide}
              className="p-2 rounded-full bg-white/30 backdrop-blur-sm border border-white/40 hover:bg-white/40 transition-all duration-200"
            >
              <ChevronRight className="w-5 h-5 text-medical-700" />
            </button>
          </div>

          {/* Method Progress */}
          <div className="mt-4 text-center">
            <span className="text-sm text-medical-600">
              {currentMethodIndex + 1} of {methodSlides.length} - {currentMethod.toUpperCase()} Analysis
            </span>
          </div>
        </div>
      </div>

      {/* Expanded Image Modal */}
      <AnimatePresence>
        {expandedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={closeExpanded}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="relative max-w-4xl max-h-full"
              onClick={(e) => e.stopPropagation()}
            >
              <img
                src={`data:image/png;base64,${expandedImage}`}
                alt="Expanded view"
                className="max-w-full max-h-full object-contain rounded-lg"
              />
              <button
                onClick={closeExpanded}
                className="absolute top-4 right-4 p-2 bg-black/50 hover:bg-black/70 rounded-full text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Statistics Section */}
      {gradcamData.statistics && (
        <div className="bg-white/20 backdrop-blur-md border border-white/30 rounded-2xl p-6">
          <h5 className="font-medium text-medical-700 mb-4 flex items-center">
            <Zap className="w-4 h-4 mr-2" />
            Analysis Statistics
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-3 bg-white/20 rounded-lg">
              <div className="text-2xl font-bold text-primary-600">
                {gradcamData.statistics.attentionMean?.toFixed(3) || 'N/A'}
              </div>
              <div className="text-sm text-medical-600">Attention Mean</div>
            </div>
            <div className="text-center p-3 bg-white/20 rounded-lg">
              <div className="text-2xl font-bold text-primary-600">
                {gradcamData.statistics.attentionStd?.toFixed(3) || 'N/A'}
              </div>
              <div className="text-sm text-medical-600">Attention Std</div>
            </div>
            <div className="text-center p-3 bg-white/20 rounded-lg">
              <div className="text-2xl font-bold text-primary-600">
                {gradcamData.statistics.attentionRatio?.toFixed(2) || 'N/A'}
              </div>
              <div className="text-sm text-medical-600">Attention Ratio</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GradCAMSlideshow;
