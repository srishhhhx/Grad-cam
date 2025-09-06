import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, CheckCircle, AlertCircle } from 'lucide-react';
import { MedicalImage, UploadProgress } from '../types';
import { uploadImage } from '../utils/api';

interface ImageUploadProps {
  onImageUploaded: (image: MedicalImage) => void;
  onError: (error: string) => void;
  disabled?: boolean;
  maxSize?: number; // in MB
  acceptedFormats?: string[];
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageUploaded,
  onError,
  disabled = false,
  maxSize = 10,
  acceptedFormats = ['image/jpeg', 'image/png', 'image/dicom']
}) => {
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [uploadedImage, setUploadedImage] = useState<MedicalImage | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];

    // Validate file size
    if (file.size > maxSize * 1024 * 1024) {
      onError(`File size must be less than ${maxSize}MB`);
      return;
    }

    // Create preview URL
    const preview = URL.createObjectURL(file);
    setPreviewUrl(preview);

    try {
      // Upload the image
      const response = await uploadImage(file, setUploadProgress);

      if (response.success && response.data) {
        setUploadedImage(response.data);
        onImageUploaded(response.data);
        setUploadProgress({
          percentage: 100,
          status: 'completed',
          message: 'Upload completed successfully!'
        });
      } else {
        throw new Error(response.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);

      // Check if it's a network error (backend not running)
      if (error instanceof Error && (error.message.includes('Network Error') || error.message.includes('ERR_CONNECTION_REFUSED'))) {
        // Fallback to demo mode
        console.log('Backend not available, using demo mode');
        const demoImage: MedicalImage = {
          id: `demo_${Date.now()}`,
          file: file,
          url: preview,
          uploadedAt: new Date()
        };

        setUploadedImage(demoImage);
        onImageUploaded(demoImage);
        setUploadProgress({
          percentage: 100,
          status: 'completed',
          message: 'Upload completed (Demo Mode)'
        });
        return;
      }

      // Handle other errors
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      onError(`${errorMessage}. Make sure the backend server is running on port 8000.`);
      setUploadProgress({
        percentage: 0,
        status: 'error',
        message: 'Upload failed - Backend connection error'
      });
      // Clear preview on error
      setPreviewUrl(null);
    }
  }, [maxSize, onImageUploaded, onError]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: acceptedFormats.reduce((acc, format) => ({ ...acc, [format]: [] }), {}),
    maxFiles: 1,
    disabled: disabled || uploadProgress?.status === 'uploading',
  });

  const clearUpload = () => {
    setUploadedImage(null);
    setUploadProgress(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  };

  const getUploadZoneClasses = () => {
    let classes = 'upload-zone cursor-pointer transition-all duration-300 ';

    if (disabled || uploadProgress?.status === 'uploading') {
      classes += 'opacity-50 cursor-not-allowed ';
    } else if (isDragActive && !isDragReject) {
      classes += 'upload-zone-active ';
    } else if (isDragReject) {
      classes += 'border-red-400 bg-red-50/30 ';
    }

    return classes;
  };

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {!uploadedImage ? (
          <div {...getRootProps()} className={getUploadZoneClasses()}>
            <motion.div
              key="upload-zone"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full h-full"
            >
              <input {...getInputProps()} />

              <div className="flex flex-col items-center justify-center space-y-4">
                <motion.div
                  animate={{
                    scale: isDragActive ? 1.1 : 1,
                    rotate: isDragActive ? 5 : 0
                  }}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center"
                >
                  <Upload className="w-8 h-8 text-primary-600" />
                </motion.div>

                <div className="text-center">
                  <h3 className="text-lg font-semibold text-medical-800 mb-2">
                    {isDragActive ? 'Drop your medical scan here' : 'Upload Medical Scan'}
                  </h3>
                  <p className="text-medical-600 mb-4">
                    Drag and drop your scan, or click to browse
                  </p>
                  <div className="text-sm text-medical-500 space-y-1">
                    <p>Supported formats: JPEG, PNG, DICOM</p>
                    <p>Maximum size: {maxSize}MB</p>
                  </div>
                </div>

                {isDragReject && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center text-red-600 bg-red-50/50 px-3 py-2 rounded-lg"
                  >
                    <AlertCircle className="w-4 h-4 mr-2" />
                    <span className="text-sm">Invalid file type</span>
                  </motion.div>
                )}
              </div>
            </motion.div>
          </div>
        ) : (
          <motion.div
            key="uploaded-image"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="medical-glass p-6"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-medical-800">Upload Successful</h3>
                  <p className="text-sm text-medical-600">
                    {uploadedImage.file.name} • {(uploadedImage.file.size / 1024 / 1024).toFixed(2)}MB
                  </p>
                </div>
              </div>
              <button
                onClick={clearUpload}
                className="p-2 hover:bg-red-50 rounded-lg transition-colors duration-200"
                title="Remove image"
              >
                <X className="w-5 h-5 text-medical-500 hover:text-red-600" />
              </button>
            </div>

            {previewUrl && (
              <div className="mb-4">
                <div className="relative bg-black/5 rounded-xl overflow-hidden">
                  <img
                    src={previewUrl}
                    alt="Medical scan preview"
                    className="w-full h-64 object-contain"
                  />
                  <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                    Preview
                  </div>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center text-green-600">
                <CheckCircle className="w-4 h-4 mr-2" />
                <span>Ready for analysis</span>
              </div>
              <button
                onClick={clearUpload}
                className="text-primary-600 hover:text-primary-700 font-medium"
              >
                Upload Different Image
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload Progress */}
      <AnimatePresence>
        {uploadProgress && uploadProgress.status === 'uploading' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 medical-glass p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-medical-800">
                {uploadProgress.message}
              </span>
              <span className="text-sm text-medical-600">
                {uploadProgress.percentage.toFixed(1)}%
              </span>
            </div>
            <div className="progress-bar">
              <motion.div
                className="progress-fill"
                initial={{ width: 0 }}
                animate={{ width: `${uploadProgress.percentage}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Display */}
      <AnimatePresence>
        {uploadProgress?.status === 'error' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 bg-red-50/50 border border-red-200/50 rounded-xl p-4"
          >
            <div className="flex items-center text-red-700">
              <AlertCircle className="w-5 h-5 mr-2" />
              <span className="font-medium">Upload Failed</span>
            </div>
            <p className="text-red-600 text-sm mt-1">
              {uploadProgress.message}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUpload;
