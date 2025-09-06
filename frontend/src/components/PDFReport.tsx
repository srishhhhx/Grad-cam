import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  FileText,
  Download,
  Eye,
  Edit3,
  CheckCircle,
  Loader,
  Activity,
  Image as ImageIcon,
  Calendar,
  User
} from 'lucide-react';
import { MedicalImage, UNetPrediction, XAIAnalysis, MedicalReport } from '../types';
import { getPredictionOverlayImage } from '../utils/api';
import jsPDF from 'jspdf';


interface PDFReportProps {
  image: MedicalImage;
  prediction: UNetPrediction;
  xaiAnalysis: XAIAnalysis;
  report: MedicalReport | null;
  onGenerateReport: (doctorNotes?: string) => void;
  onResetReport?: () => void;
  isGenerating: boolean;
}

const PDFReport: React.FC<PDFReportProps> = ({
  image,
  prediction,
  xaiAnalysis,
  report,
  onGenerateReport,
  onResetReport,
  isGenerating
}) => {
  const [doctorNotes, setDoctorNotes] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [overlayImageUrl, setOverlayImageUrl] = useState<string | null>(null);
  const [loadingOverlay, setLoadingOverlay] = useState(false);

  // Fetch overlay image when component mounts or prediction changes
  useEffect(() => {
    const fetchOverlayImage = async () => {
      if (!prediction?.id) {
        // Use original image if no prediction
        setOverlayImageUrl(`data:image/jpeg;base64,${image.imageData}`);
        return;
      }

      setLoadingOverlay(true);
      try {
        const imageUrl = await getPredictionOverlayImage(prediction.id);
        setOverlayImageUrl(imageUrl);
      } catch (error) {
        console.error('Failed to fetch overlay image:', error);
        // Fallback to original image if overlay fails
        setOverlayImageUrl(`data:image/jpeg;base64,${image.imageData}`);
      } finally {
        setLoadingOverlay(false);
      }
    };

    // Set original image immediately, then try to fetch overlay
    setOverlayImageUrl(`data:image/jpeg;base64,${image.imageData}`);
    fetchOverlayImage();
  }, [prediction?.id, image.imageData]);

  const handleGenerateReport = () => {
    onGenerateReport(doctorNotes);
  };

  const handleDownloadPDF = async () => {
    if (!report) return;

    try {
      // Try to download the backend-generated PDF first
      try {
        const response = await fetch(`/api/reports/${report.id}/download`);
        if (response.ok) {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `Fibroid-Analysis-Report-${report.id.substring(0, 8)}.pdf`;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          return; // Success - exit early
        }
      } catch (backendError) {
        console.warn('Backend PDF download failed, falling back to frontend generation:', backendError);
      }

      // Fallback to frontend PDF generation
      // Create a new jsPDF instance with enhanced settings
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;

      // Enhanced color scheme matching backend
      const primaryColor = [30, 64, 175]; // Medical blue
      const secondaryColor = [71, 85, 105]; // Professional gray
      const accentColor = [59, 130, 246]; // Bright blue
      const successColor = [5, 150, 105]; // Medical green
      const warningColor = [217, 119, 6]; // Medical orange
      const errorColor = [220, 38, 38]; // Medical red

      // Professional header with medical styling
      pdf.setFillColor(248, 250, 252); // Light background
      pdf.rect(0, 0, pageWidth, 50, 'F');

      // Institution header
      pdf.setFontSize(20);
      pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.setFont('helvetica', 'bold');
      pdf.text('🏥 MEDICAL IMAGING CENTER', margin, 20);

      pdf.setFontSize(14);
      pdf.setTextColor(accentColor[0], accentColor[1], accentColor[2]);
      pdf.setFont('helvetica', 'normal');
      pdf.text('AI-Assisted Uterine Fibroids Analysis', margin, 30);

      pdf.setFontSize(11);
      pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.text('Advanced Neural Network Diagnostics', margin, 40);

      // Report info on the right
      pdf.setFontSize(10);
      pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.text(`📋 Report ID: ${report.id.substring(0, 8)}...`, pageWidth - 80, 20);
      pdf.text(`📅 Generated: ${new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      })}`, pageWidth - 80, 30);

      // Professional border
      pdf.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.setLineWidth(2);
      pdf.rect(margin - 5, 55, pageWidth - 2 * margin + 10, pageHeight - 110, 'S');

      let currentY = 70;

      // Enhanced title section
      pdf.setFillColor(239, 246, 255); // Light blue background
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 25, 'F');
      pdf.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 25, 'S');

      pdf.setFontSize(18);
      pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.setFont('helvetica', 'bold');
      pdf.text('🔬 UTERINE FIBROIDS ANALYSIS REPORT', margin + 10, currentY + 15);

      currentY += 35;

      // Doctor information section
      pdf.setFillColor(244, 244, 245); // Light gray background
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 20, 'F');
      pdf.setDrawColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 20, 'S');

      pdf.setFontSize(10);
      pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.setFont('helvetica', 'bold');
      pdf.text('👨‍⚕️ Attending Physician: Dr. Shashi', margin + 5, currentY + 8);
      pdf.text('🎓 Specialization: Radiology & Medical Imaging', margin + 5, currentY + 15);

      pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.setFont('helvetica', 'normal');
      pdf.text('🏥 Medical Imaging Center', pageWidth - 80, currentY + 8);
      pdf.text('🔬 AI Model: U-Net++ EfficientNet-B5', pageWidth - 80, currentY + 15);

      currentY += 30;

      // Enhanced ultrasound image section
      if (overlayImageUrl) {
        try {
          // Section header
          pdf.setFillColor(239, 246, 255);
          pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'F');
          pdf.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
          pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'S');

          pdf.setFontSize(12);
          pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
          pdf.setFont('helvetica', 'bold');
          pdf.text('🖼️ ULTRASOUND SCAN WITH AI SEGMENTATION OVERLAY', margin + 5, currentY + 10);

          currentY += 20;

          // Add the overlay image
          const imgWidth = 120;
          const imgHeight = 90;
          const imgX = (pageWidth - imgWidth) / 2;

          pdf.addImage(overlayImageUrl, 'PNG', imgX, currentY, imgWidth, imgHeight);

          // Image border
          pdf.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
          pdf.setLineWidth(1);
          pdf.rect(imgX - 2, currentY - 2, imgWidth + 4, imgHeight + 4, 'S');

          currentY += imgHeight + 10;

          // Image caption
          pdf.setFontSize(9);
          pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
          pdf.setFont('helvetica', 'italic');
          const caption = 'Figure 1: AI-Enhanced ultrasound analysis with segmentation overlay. Orange/red regions indicate detected fibroids.';
          const captionLines = pdf.splitTextToSize(caption, pageWidth - 2 * margin);
          pdf.text(captionLines, margin, currentY);
          currentY += captionLines.length * 4 + 10;

        } catch (error) {
          console.warn('Could not add overlay image to PDF:', error);
          currentY += 10;
        }
      }

      // Enhanced analysis summary
      pdf.setFillColor(240, 253, 244); // Light green background
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'F');
      pdf.setDrawColor(successColor[0], successColor[1], successColor[2]);
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'S');

      pdf.setFontSize(14);
      pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.setFont('helvetica', 'bold');
      pdf.text('📊 COMPREHENSIVE ANALYSIS SUMMARY', margin + 5, currentY + 10);

      currentY += 25;

      // Detection status with color coding
      const isDetected = prediction.prediction.fibroidDetected;
      const statusColor = isDetected ? errorColor : successColor;
      const statusText = isDetected ? '🔴 FIBROIDS DETECTED' : '🟢 NO FIBROIDS DETECTED';

      if (isDetected) {
        pdf.setFillColor(254, 242, 242); // Light red for detected
      } else {
        pdf.setFillColor(240, 253, 244); // Light green for not detected
      }
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 12, 'F');
      pdf.setDrawColor(statusColor[0], statusColor[1], statusColor[2]);
      pdf.rect(margin, currentY, pageWidth - 2 * margin, 12, 'S');

      pdf.setFontSize(12);
      pdf.setTextColor(statusColor[0], statusColor[1], statusColor[2]);
      pdf.setFont('helvetica', 'bold');
      pdf.text(statusText, margin + 10, currentY + 8);

      currentY += 20;

      // Analysis metrics in professional table format
      const confidence = prediction.prediction.confidence * 100;
      const confidenceColor = confidence >= 90 ? successColor : confidence >= 70 ? warningColor : errorColor;
      const confidenceIcon = confidence >= 90 ? '🟢' : confidence >= 70 ? '🟡' : '🔴';

      const metrics: Array<[string, string, number[]]> = [
        [
          `${confidenceIcon} AI Confidence:`,
          stripHtmlTags(`${confidence.toFixed(1)}%`),
          confidenceColor
        ],
        [
          '🔢 Fibroid Count:',
          stripHtmlTags(`${prediction.prediction.fibroidCount} detected`),
          secondaryColor
        ],
        [
          '⚠️ Overall Severity:',
          stripHtmlTags(report.summary?.severity?.toUpperCase() || 'NOT SPECIFIED'),
          report.summary?.severity === 'mild' ? successColor :
            report.summary?.severity === 'moderate' ? warningColor : errorColor
        ]
      ];

      metrics.forEach(([label, value, color], index) => {
        const y = currentY + (index * 12);

        // Alternating background
        if (index % 2 === 0) {
          pdf.setFillColor(248, 250, 252);
          pdf.rect(margin, y - 2, pageWidth - 2 * margin, 10, 'F');
        }

        pdf.setFontSize(10);
        pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        pdf.setFont('helvetica', 'bold');
        pdf.text(stripHtmlTags(label), margin + 5, y + 5);

        pdf.setTextColor(color[0], color[1], color[2]);
        pdf.setFont('helvetica', 'bold');
        pdf.text(stripHtmlTags(value), margin + 80, y + 5);
      });

      currentY += metrics.length * 12 + 15;

      // Enhanced fibroid details if any
      if (prediction.prediction.fibroidAreas.length > 0) {
        pdf.setFillColor(255, 247, 237); // Light orange background
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'F');
        pdf.setDrawColor(warningColor[0], warningColor[1], warningColor[2]);
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'S');

        pdf.setFontSize(12);
        pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        pdf.setFont('helvetica', 'bold');
        pdf.text('🔍 DETAILED FIBROID ANALYSIS', margin + 5, currentY + 10);

        currentY += 20;

        prediction.prediction.fibroidAreas.forEach((fibroid, index) => {
          const severityColor = fibroid.severity === 'mild' ? successColor :
            fibroid.severity === 'moderate' ? warningColor : errorColor;

          // Fibroid header
          pdf.setFillColor(250, 250, 250);
          pdf.rect(margin, currentY, pageWidth - 2 * margin, 8, 'F');
          pdf.setDrawColor(severityColor[0], severityColor[1], severityColor[2]);
          pdf.rect(margin, currentY, pageWidth - 2 * margin, 8, 'S');

          pdf.setFontSize(10);
          pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
          pdf.setFont('helvetica', 'bold');
          pdf.text(`🎯 Fibroid #${index + 1}`, margin + 5, currentY + 5);

          currentY += 12;

          // Fibroid details
          const details = [
            `📏 Area: ${fibroid.area.toFixed(1)} mm²`,
            `⚠️ Severity: ${fibroid.severity.toUpperCase()}`,
            `📍 Location: (${fibroid.location.x.toFixed(1)}, ${fibroid.location.y.toFixed(1)})`
          ];

          details.forEach((detail, detailIndex) => {
            pdf.setFontSize(9);
            pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
            pdf.setFont('helvetica', 'normal');
            pdf.text(detail, margin + 10, currentY + (detailIndex * 6));
          });

          currentY += 20;
        });
      }

      // Enhanced XAI explanation
      if (xaiAnalysis?.explanation?.summary) {
        pdf.setFillColor(248, 250, 255); // Light purple background
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'F');
        pdf.setDrawColor(139, 92, 246); // Purple border
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'S');

        pdf.setFontSize(12);
        pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        pdf.setFont('helvetica', 'bold');
        pdf.text('🤖 AI EXPLANATION & INSIGHTS', margin + 5, currentY + 10);

        currentY += 20;

        pdf.setFontSize(10);
        pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
        pdf.setFont('helvetica', 'normal');
        const explanationLines = pdf.splitTextToSize(stripHtmlTags(xaiAnalysis.explanation.summary), pageWidth - 2 * margin - 10);
        pdf.text(explanationLines, margin + 5, currentY);
        currentY += explanationLines.length * 4 + 15;
      }

      // Enhanced doctor notes if available
      if (doctorNotes.trim()) {
        pdf.setFillColor(255, 251, 235); // Light amber background
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'F');
        pdf.setDrawColor(warningColor[0], warningColor[1], warningColor[2]);
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'S');

        pdf.setFontSize(12);
        pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        pdf.setFont('helvetica', 'bold');
        pdf.text('👨‍⚕️ CLINICAL NOTES & PHYSICIAN OBSERVATIONS', margin + 5, currentY + 10);

        currentY += 20;

        pdf.setFontSize(10);
        pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
        pdf.setFont('helvetica', 'normal');
        const notesLines = pdf.splitTextToSize(stripHtmlTags(doctorNotes), pageWidth - 2 * margin - 10);
        pdf.text(notesLines, margin + 5, currentY);
        currentY += notesLines.length * 4 + 15;
      }

      // Enhanced recommendations
      if (report.summary?.recommendations?.length > 0) {
        pdf.setFillColor(240, 253, 250); // Light teal background
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'F');
        pdf.setDrawColor(20, 184, 166); // Teal border
        pdf.rect(margin, currentY, pageWidth - 2 * margin, 15, 'S');

        pdf.setFontSize(12);
        pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
        pdf.setFont('helvetica', 'bold');
        pdf.text('💡 CLINICAL RECOMMENDATIONS & NEXT STEPS', margin + 5, currentY + 10);

        currentY += 20;

        report.summary.recommendations.forEach((rec, index) => {
          pdf.setFontSize(10);
          pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
          pdf.setFont('helvetica', 'bold');
          pdf.text(`${index + 1}.`, margin + 5, currentY);

          pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
          pdf.setFont('helvetica', 'normal');
          const recLines = pdf.splitTextToSize(stripHtmlTags(rec), pageWidth - 2 * margin - 15);
          pdf.text(recLines, margin + 15, currentY);
          currentY += recLines.length * 4 + 8;
        });

        currentY += 10;
      }

      // Enhanced footer with medical disclaimer
      const footerY = pageHeight - 40;

      // Disclaimer section
      pdf.setFillColor(254, 242, 242); // Light red background
      pdf.rect(margin, footerY - 25, pageWidth - 2 * margin, 20, 'F');
      pdf.setDrawColor(errorColor[0], errorColor[1], errorColor[2]);
      pdf.rect(margin, footerY - 25, pageWidth - 2 * margin, 20, 'S');

      pdf.setFontSize(8);
      pdf.setTextColor(errorColor[0], errorColor[1], errorColor[2]);
      pdf.setFont('helvetica', 'bold');
      pdf.text('⚠️ IMPORTANT MEDICAL DISCLAIMER', margin + 5, footerY - 20);

      pdf.setFontSize(7);
      pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.setFont('helvetica', 'normal');
      const disclaimer = 'This report is generated by an AI-powered medical imaging analysis system. Results should be interpreted by qualified medical professionals and used in conjunction with clinical findings. This analysis is not a substitute for professional medical diagnosis or treatment.';
      const disclaimerLines = pdf.splitTextToSize(disclaimer, pageWidth - 2 * margin - 10);
      pdf.text(disclaimerLines, margin + 5, footerY - 15);

      // Final footer
      pdf.setFontSize(8);
      pdf.setTextColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.setFont('helvetica', 'italic');
      pdf.text('🏥 Medical Imaging Center - AI Analysis Division', margin, footerY);
      pdf.text(`📅 Generated: ${new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })}`, pageWidth - 80, footerY);

      // Save the PDF with enhanced filename
      const timestamp = new Date().toISOString().slice(0, 10);
      pdf.save(`Enhanced-Fibroid-Analysis-Report-${timestamp}-${report.id.substring(0, 8)}.pdf`);

    } catch (error) {
      console.error('Error generating enhanced PDF:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Report Generation */}
      {!report ? (
        <div className="medical-glass p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl flex items-center justify-center">
            <FileText className="w-8 h-8 text-white" />
          </div>

          <h3 className="text-xl font-bold text-medical-800 mb-4">
            Generate Medical Report
          </h3>

          <p className="text-medical-600 mb-6">
            Create a comprehensive PDF report with AI analysis results and explanations
          </p>

          {/* Doctor Notes */}
          <div className="max-w-2xl mx-auto mb-6">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-medium text-medical-700">
                Doctor Notes (Optional)
              </label>
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center space-x-1 text-primary-600 hover:text-primary-700 text-sm"
              >
                <Edit3 className="w-4 h-4" />
                <span>{isEditing ? 'Preview' : 'Edit'}</span>
              </button>
            </div>

            {isEditing ? (
              <textarea
                value={doctorNotes}
                onChange={(e) => setDoctorNotes(e.target.value)}
                placeholder="Add your clinical observations, additional findings, or recommendations..."
                className="input-glass w-full h-32 resize-none"
              />
            ) : (
              <div className="bg-white/20 rounded-xl p-4 min-h-[8rem] text-left">
                {doctorNotes.trim() ? (
                  <p className="text-medical-700 whitespace-pre-wrap">{doctorNotes}</p>
                ) : (
                  <p className="text-medical-500 italic">No additional notes added</p>
                )}
              </div>
            )}
          </div>

          <button
            onClick={handleGenerateReport}
            disabled={isGenerating}
            className="btn-primary"
          >
            {isGenerating ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-5 h-5 mr-2"
                >
                  <Activity className="w-full h-full" />
                </motion.div>
                Generating Report...
              </>
            ) : (
              <>
                <FileText className="w-5 h-5 mr-2" />
                Generate Report
              </>
            )}
          </button>
        </div>
      ) : (
        /* Report Generated */
        <div className="space-y-6">
          {/* Success Message */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="medical-glass p-6"
          >
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-medical-800">
                  Report Generated Successfully
                </h3>
                <p className="text-medical-600">
                  Your comprehensive medical report is ready for download
                </p>
              </div>
            </div>
          </motion.div>

          {/* Report Summary */}
          <div className="medical-glass p-6">
            <h4 className="font-semibold text-medical-800 mb-6 flex items-center">
              <FileText className="w-5 h-5 mr-2" />
              Report Summary
            </h4>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Ultrasound Preview */}
              <div className="lg:col-span-1">
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4">
                  <h5 className="font-medium text-medical-800 mb-3 flex items-center text-sm">
                    <ImageIcon className="w-4 h-4 mr-2 text-blue-600" />
                    Ultrasound Scan
                  </h5>

                  <div className="flex justify-center">
                    {loadingOverlay ? (
                      <div className="flex items-center justify-center w-32 h-24 bg-white rounded-lg border border-gray-200">
                        <Loader className="w-5 h-5 animate-spin text-blue-500" />
                      </div>
                    ) : overlayImageUrl ? (
                      <div className="relative group cursor-pointer">
                        <img
                          src={overlayImageUrl}
                          alt="Ultrasound Preview"
                          className="w-32 h-24 object-cover rounded-lg shadow-md border-2 border-white hover:shadow-lg transition-shadow"
                        />
                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 rounded-lg flex items-center justify-center">
                          <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                            <Eye className="w-5 h-5 text-white" />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center w-32 h-24 bg-gray-100 rounded-lg border border-gray-200">
                        <ImageIcon className="w-6 h-6 text-gray-400" />
                      </div>
                    )}
                  </div>

                  <p className="text-xs text-center text-gray-600 mt-2">
                    With AI segmentation overlay
                  </p>
                </div>
              </div>

              {/* Report Details */}
              <div className="lg:col-span-2 grid md:grid-cols-2 gap-4">
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <Calendar className="w-5 h-5 text-medical-500" />
                    <div>
                      <p className="text-sm text-medical-600">Generated</p>
                      <p className="font-medium text-medical-800">
                        {new Date(report.generatedAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <User className="w-5 h-5 text-medical-500" />
                    <div>
                      <p className="text-sm text-medical-600">Report ID</p>
                      <p className="font-medium text-medical-800 font-mono text-sm">
                        {report.id}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <Activity className="w-5 h-5 text-medical-500" />
                    <div>
                      <p className="text-sm text-medical-600">Detection Status</p>
                      <p className={`font-medium ${prediction.prediction.fibroidDetected ? 'text-orange-600' : 'text-green-600'}`}>
                        {prediction.prediction.fibroidDetected ? '⚠️ Fibroids Detected' : '✅ No Fibroids'}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <FileText className="w-5 h-5 text-medical-500" />
                    <div>
                      <p className="text-sm text-medical-600">Severity</p>
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${report.summary?.severity === 'mild' ? 'severity-mild' :
                        report.summary?.severity === 'moderate' ? 'severity-moderate' : 'severity-severe'
                        }`}>
                        {report.summary?.severity ?
                          report.summary.severity.charAt(0).toUpperCase() + report.summary.severity.slice(1) :
                          'Not specified'
                        }
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            {report.summary?.recommendations?.length > 0 && (
              <div className="mt-6">
                <h5 className="font-medium text-medical-800 mb-3">Key Recommendations:</h5>
                <div className="space-y-2">
                  {report.summary?.recommendations?.slice(0, 3).map((rec, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0" />
                      <p className="text-medical-700 text-sm">{rec}</p>
                    </div>
                  ))}
                  {(report.summary?.recommendations?.length || 0) > 3 && (
                    <p className="text-medical-500 text-sm italic">
                      +{(report.summary?.recommendations?.length || 0) - 3} more recommendations in full report
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setShowPreview(!showPreview)}
              className="btn-secondary"
            >
              <Eye className="w-4 h-4 mr-2" />
              {showPreview ? 'Hide Preview' : 'Preview Report'}
            </button>

            <div className="flex space-x-3">
              <button
                onClick={() => {
                  onResetReport?.();
                  setDoctorNotes('');
                }}
                className="btn-secondary"
              >
                Generate New Report
              </button>

              <button
                onClick={handleDownloadPDF}
                className="btn-primary"
              >
                <Download className="w-4 h-4 mr-2" />
                Download PDF
              </button>
            </div>
          </div>

          {/* Report Preview */}
          {showPreview && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="medical-glass p-6"
            >
              <h4 className="font-semibold text-medical-800 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Report Preview
              </h4>

              <div className="bg-white rounded-xl p-8 shadow-inner max-h-[600px] overflow-y-auto">
                <div className="space-y-6">
                  {/* Report Header */}
                  <div className="border-b-2 border-gray-200 pb-6">
                    <div className="text-center mb-4">
                      <h1 className="text-3xl font-bold text-gray-800 mb-2">
                        🏥 Uterine Fibroids Analysis Report
                      </h1>
                      <div className="text-gray-600 space-y-1">
                        <p className="text-lg">Generated: {new Date().toLocaleDateString('en-US', {
                          weekday: 'long',
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric'
                        })}</p>
                        <p className="font-mono text-sm bg-gray-100 px-3 py-1 rounded-full inline-block">
                          Report ID: {report.id}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Ultrasound Image Section */}
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6">
                    <h2 className="font-bold text-gray-800 mb-4 flex items-center text-lg">
                      <ImageIcon className="w-5 h-5 mr-2 text-blue-600" />
                      🔬 Ultrasound Scan with AI Analysis
                    </h2>

                    <div className="flex justify-center mb-4">
                      {loadingOverlay ? (
                        <div className="flex items-center justify-center w-96 h-72 bg-white rounded-lg border-2 border-dashed border-gray-300">
                          <div className="text-center">
                            <Loader className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-2" />
                            <p className="text-gray-500">Loading ultrasound image...</p>
                          </div>
                        </div>
                      ) : overlayImageUrl ? (
                        <div className="relative group">
                          <img
                            src={overlayImageUrl}
                            alt="Ultrasound with AI Prediction Overlay"
                            className="max-w-96 max-h-72 object-contain rounded-lg shadow-lg border-4 border-white"
                          />
                          <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all duration-300 rounded-lg flex items-center justify-center">
                            <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white px-3 py-1 rounded-full text-sm font-medium text-gray-700">
                              🎯 AI Segmentation Overlay
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center w-96 h-72 bg-gray-100 rounded-lg border-2 border-dashed border-gray-300">
                          <div className="text-center text-gray-500">
                            <ImageIcon className="w-12 h-12 mx-auto mb-2" />
                            <p>Ultrasound image unavailable</p>
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="text-center text-sm text-gray-600 bg-white rounded-lg p-3">
                      <p className="font-medium">🎨 Color-coded segmentation overlay shows detected fibroids</p>
                      <p>Orange/Red regions indicate areas of interest identified by the AI model</p>
                    </div>
                  </div>

                  {/* Analysis Summary */}
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6">
                    <h2 className="font-bold text-gray-800 mb-4 text-lg">📊 Analysis Summary</h2>
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <div className="bg-white rounded-lg p-3">
                          <p className="text-sm text-gray-600">Detection Status</p>
                          <p className={`font-bold text-lg ${prediction.prediction.fibroidDetected ? 'text-orange-600' : 'text-green-600'}`}>
                            {prediction.prediction.fibroidDetected ? '⚠️ Fibroids Detected' : '✅ No Fibroids Detected'}
                          </p>
                        </div>
                        <div className="bg-white rounded-lg p-3">
                          <p className="text-sm text-gray-600">AI Confidence</p>
                          <p className="font-bold text-lg text-blue-600">
                            {(prediction.prediction.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <div className="bg-white rounded-lg p-3">
                          <p className="text-sm text-gray-600">Fibroid Count</p>
                          <p className="font-bold text-lg text-purple-600">
                            {prediction.prediction.fibroidCount} detected
                          </p>
                        </div>
                        <div className="bg-white rounded-lg p-3">
                          <p className="text-sm text-gray-600">Overall Severity</p>
                          <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-bold ${report.summary?.severity === 'mild' ? 'bg-green-100 text-green-800' :
                            report.summary?.severity === 'moderate' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                            {report.summary?.severity ?
                              report.summary.severity.charAt(0).toUpperCase() + report.summary.severity.slice(1) :
                              'Not specified'
                            }
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* AI Explanation */}
                  {xaiAnalysis.explanation.summary && (
                    <div className="bg-gradient-to-br from-purple-50 to-violet-50 rounded-xl p-6">
                      <h2 className="font-bold text-gray-800 mb-4 text-lg">🤖 AI Explanation</h2>
                      <div className="bg-white rounded-lg p-4">
                        <p className="text-gray-700 leading-relaxed">{xaiAnalysis.explanation.summary}</p>
                      </div>
                    </div>
                  )}

                  {/* Doctor Notes */}
                  {doctorNotes.trim() && (
                    <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl p-6">
                      <h2 className="font-bold text-gray-800 mb-4 text-lg">👨‍⚕️ Doctor Notes</h2>
                      <div className="bg-white rounded-lg p-4">
                        <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">{doctorNotes}</p>
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {(report.summary?.recommendations?.length || 0) > 0 && (
                    <div className="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl p-6">
                      <h2 className="font-bold text-gray-800 mb-4 text-lg">💡 Recommendations</h2>
                      <div className="bg-white rounded-lg p-4">
                        <ol className="space-y-2">
                          {report.summary?.recommendations?.map((rec, index) => (
                            <li key={index} className="flex items-start">
                              <span className="flex-shrink-0 w-6 h-6 bg-teal-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3 mt-0.5">
                                {index + 1}
                              </span>
                              <span className="text-gray-700 leading-relaxed">{rec}</span>
                            </li>
                          ))}
                        </ol>
                      </div>
                    </div>
                  )}

                  {/* Footer */}
                  <div className="text-center pt-6 border-t-2 border-gray-200">
                    <p className="text-gray-500 text-sm">
                      🏥 Generated by Uterine Fibroids Analyzer - AI-Powered Medical Imaging System
                    </p>
                    <p className="text-gray-400 text-xs mt-1">
                      This report is for medical professional use only
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
};

// Add this utility function at the top of your PDFReport.tsx
function stripHtmlTags(str: string) {
  if (!str) return '';
  return str.replace(/<[^>]*>?/gm, '');
}

export default PDFReport;
