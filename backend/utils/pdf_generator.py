#!/usr/bin/env python3
"""
PDF Report Generator for Medical Analysis Reports
Generates comprehensive PDF reports with analysis results and visualizations.
"""

import os
import base64
import io
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from PIL import Image as PILImage, ImageDraw


class PDFReportGenerator:
    """
    Generates comprehensive PDF reports for medical analysis.
    """

    def __init__(self, reports_dir: str = "reports"):
        """
        Initialize the PDF generator.

        Args:
            reports_dir: Directory to save generated reports
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

        # Enhanced medical color scheme
        self.primary_color = HexColor("#1e40af")  # Deep medical blue
        self.secondary_color = HexColor("#475569")  # Professional gray
        self.accent_color = HexColor("#3b82f6")  # Bright blue
        self.success_color = HexColor("#059669")  # Medical green
        self.warning_color = HexColor("#d97706")  # Medical orange
        self.error_color = HexColor("#dc2626")  # Medical red
        self.light_bg = HexColor("#f8fafc")  # Light background
        self.border_color = HexColor("#e2e8f0")  # Subtle borders
        self.highlight_bg = HexColor("#eff6ff")  # Light blue highlight

        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        print("📄 PDF Report Generator initialized")

    def _setup_custom_styles(self):
        """Setup enhanced custom paragraph styles for beautiful medical reports."""

        # Enhanced title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=26,
                spaceAfter=20,
                spaceBefore=10,
                textColor=self.primary_color,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
                leading=30,
            )
        )

        # Professional subtitle style
        self.styles.add(
            ParagraphStyle(
                name="CustomSubtitle",
                parent=self.styles["Heading1"],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=8,
                textColor=self.accent_color,
                alignment=TA_CENTER,
                fontName="Helvetica",
                leading=20,
            )
        )

        # Enhanced section header style
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=self.primary_color,
                alignment=TA_LEFT,
                fontName="Helvetica-Bold",
                leading=20,
                borderWidth=0,
                borderPadding=0,
            )
        )

        # Professional body text style
        self.styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=self.styles["Normal"],
                fontSize=11,
                spaceAfter=8,
                spaceBefore=2,
                alignment=TA_LEFT,
                leading=16,
                textColor=self.secondary_color,
                fontName="Helvetica",
            )
        )

        # Enhanced highlight style
        self.styles.add(
            ParagraphStyle(
                name="Highlight",
                parent=self.styles["Normal"],
                fontSize=12,
                spaceAfter=10,
                spaceBefore=5,
                textColor=self.primary_color,
                alignment=TA_LEFT,
                fontName="Helvetica-Bold",
                leading=16,
            )
        )

        # New: Status badge style
        self.styles.add(
            ParagraphStyle(
                name="StatusBadge",
                parent=self.styles["Normal"],
                fontSize=14,
                spaceAfter=15,
                spaceBefore=10,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
                leading=18,
            )
        )

        # New: Caption style for images
        self.styles.add(
            ParagraphStyle(
                name="Caption",
                parent=self.styles["Normal"],
                fontSize=10,
                spaceAfter=12,
                spaceBefore=6,
                textColor=self.secondary_color,
                alignment=TA_CENTER,
                fontName="Helvetica-Oblique",
                leading=14,
            )
        )

        # New: Important note style
        self.styles.add(
            ParagraphStyle(
                name="ImportantNote",
                parent=self.styles["Normal"],
                fontSize=11,
                spaceAfter=10,
                spaceBefore=8,
                textColor=self.warning_color,
                alignment=TA_LEFT,
                fontName="Helvetica-Bold",
                leading=15,
            )
        )

    async def generate_report(
        self,
        report_id: str,
        image_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        xai_data: Optional[Dict[str, Any]] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        doctor_notes: Optional[str] = None,
        include_integrated_gradients: bool = True,
    ) -> Path:
        """
        Generate a comprehensive PDF report.

        Args:
            report_id: Unique report identifier
            image_data: Original image metadata
            prediction_data: U-Net prediction results
            xai_data: XAI analysis results (optional)
            patient_profile: Patient information (optional)
            doctor_notes: Additional doctor notes (optional)

        Returns:
            Path to the generated PDF file
        """

        # Create PDF file path
        pdf_path = self.reports_dir / f"report_{report_id}.pdf"

        # Create document with enhanced margins and styling
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=60,
            title=f"Uterine Fibroids Analysis Report - {report_id}",
            author="Medical Imaging Center - AI Analysis System",
            subject="Medical Imaging Analysis Report",
            creator="AI-Assisted Medical Imaging System",
        )

        # Build content
        story = []

        # Add medical header
        story.extend(self._build_medical_header(report_id))

        # Add ultrasound scan with segmentation overlay
        story.extend(self._build_scan_section(image_data, prediction_data))

        # Add patient information if available
        if patient_profile:
            story.extend(self._build_patient_section(patient_profile))

        # Add analysis summary with detailed metrics
        story.extend(
            self._build_enhanced_analysis_summary(
                prediction_data, xai_data, include_integrated_gradients
            )
        )

        # Add detailed findings
        story.extend(self._build_detailed_findings(prediction_data))

        # Add XAI explanation if available
        if xai_data:
            story.extend(
                self._build_xai_section(xai_data, include_integrated_gradients)
            )

        # Add doctor notes if available
        if doctor_notes:
            story.extend(self._build_doctor_notes(doctor_notes))

        # Add recommendations
        story.extend(self._build_recommendations(prediction_data, xai_data))

        # Add footer
        story.extend(self._build_footer())

        # Build PDF
        doc.build(story)

        print(f"📄 PDF report generated: {pdf_path}")
        return pdf_path

    def _build_medical_header(self, report_id: str) -> List:
        """Build beautiful comprehensive medical report header with enhanced styling."""
        content = []

        # Professional header with medical institution branding
        header_table_data = [
            ["🏥 MEDICAL IMAGING CENTER", "", ""],
            [
                "AI-Assisted Uterine Fibroids Analysis",
                "",
                f"📋 Report ID: {report_id[:8]}...",
            ],
            [
                "Advanced Neural Network Diagnostics",
                "",
                f"📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            ],
        ]

        header_table = Table(
            header_table_data, colWidths=[4.0 * inch, 1.0 * inch, 2.5 * inch]
        )
        header_table.setStyle(
            TableStyle(
                [
                    # Institution name styling - Bold and prominent
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 20),
                    ("TEXTCOLOR", (0, 0), (0, 0), self.primary_color),
                    # Subtitle styling - Professional
                    ("FONTNAME", (0, 1), (0, 2), "Helvetica"),
                    ("FONTSIZE", (0, 1), (0, 1), 14),
                    ("FONTSIZE", (0, 2), (0, 2), 11),
                    ("TEXTCOLOR", (0, 1), (0, 1), self.accent_color),
                    ("TEXTCOLOR", (0, 2), (0, 2), self.secondary_color),
                    # Report info styling - Clean and readable
                    ("FONTNAME", (2, 1), (2, 2), "Helvetica"),
                    ("FONTSIZE", (2, 1), (2, 2), 10),
                    ("TEXTCOLOR", (2, 1), (2, 2), self.secondary_color),
                    # Alignment
                    ("ALIGN", (0, 0), (0, 2), "LEFT"),
                    ("ALIGN", (2, 1), (2, 2), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    # Professional background and borders
                    ("BACKGROUND", (0, 0), (-1, -1), self.light_bg),
                    ("BOX", (0, 0), (-1, -1), 2, self.primary_color),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, self.border_color),
                    ("ROUNDEDCORNERS", [8, 8, 8, 8]),
                    # Enhanced spacing for readability
                    ("LEFTPADDING", (0, 0), (-1, -1), 15),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        content.append(header_table)
        content.append(Spacer(1, 25))

        # Enhanced Doctor Information with professional medical card styling
        doctor_info_table_data = [
            [
                "👨‍⚕️ Attending Physician:",
                "Dr. Shashi",
                "🎓 Specialization:",
                "Radiology & Medical Imaging",
            ],
            [
                "🏥 Institution:",
                "Medical Imaging Center",
                "📅 Date of Analysis:",
                datetime.now().strftime("%B %d, %Y"),
            ],
            [
                "📧 Contact:",
                "imaging@medcenter.com",
                "🔬 AI Model:",
                "U-Net++ with EfficientNet-B5",
            ],
        ]

        doctor_table = Table(
            doctor_info_table_data,
            colWidths=[1.9 * inch, 2.1 * inch, 1.6 * inch, 1.9 * inch],
        )
        doctor_table.setStyle(
            TableStyle(
                [
                    # Enhanced typography
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),  # Labels bold
                    ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),  # Labels bold
                    ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),  # Doctor name bold
                    # Professional color scheme
                    ("TEXTCOLOR", (0, 0), (0, -1), self.primary_color),
                    ("TEXTCOLOR", (2, 0), (2, -1), self.primary_color),
                    (
                        "TEXTCOLOR",
                        (1, 0),
                        (1, 0),
                        self.accent_color,
                    ),  # Doctor name highlighted
                    ("TEXTCOLOR", (1, 1), (1, -1), self.secondary_color),
                    ("TEXTCOLOR", (3, 0), (3, -1), self.secondary_color),
                    # Layout and alignment
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # Beautiful styling with gradient-like effect
                    ("BACKGROUND", (0, 0), (-1, -1), self.highlight_bg),
                    ("BOX", (0, 0), (-1, -1), 1.5, self.primary_color),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, self.border_color),
                    ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                    # Enhanced spacing for professional look
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        content.append(doctor_table)
        content.append(Spacer(1, 30))

        # Enhanced Main Title with professional medical styling
        title_table_data = [
            ["🔬 UTERINE FIBROIDS ANALYSIS REPORT"],
            ["AI-Assisted Medical Imaging Analysis"],
            ["Advanced Neural Network Diagnostics & Segmentation"],
        ]

        title_table = Table(title_table_data, colWidths=[7.5 * inch])
        title_table.setStyle(
            TableStyle(
                [
                    # Main title styling
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 24),
                    ("TEXTCOLOR", (0, 0), (0, 0), self.primary_color),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    # Subtitle styling
                    ("FONTNAME", (0, 1), (0, 1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (0, 1), 16),
                    ("TEXTCOLOR", (0, 1), (0, 1), self.accent_color),
                    ("ALIGN", (0, 1), (0, 1), "CENTER"),
                    # Description styling
                    ("FONTNAME", (0, 2), (0, 2), "Helvetica-Oblique"),
                    ("FONTSIZE", (0, 2), (0, 2), 12),
                    ("TEXTCOLOR", (0, 2), (0, 2), self.secondary_color),
                    ("ALIGN", (0, 2), (0, 2), "CENTER"),
                    # Professional background and borders
                    ("BACKGROUND", (0, 0), (-1, -1), self.light_bg),
                    ("BOX", (0, 0), (-1, -1), 2, self.primary_color),
                    ("ROUNDEDCORNERS", [10, 10, 10, 10]),
                    # Enhanced spacing
                    ("TOPPADDING", (0, 0), (-1, -1), 15),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                    ("LEFTPADDING", (0, 0), (-1, -1), 20),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                ]
            )
        )

        content.append(title_table)
        content.append(Spacer(1, 30))

        # Professional decorative separator
        content.append(
            HRFlowable(
                width="100%", thickness=3, lineCap="round", color=self.primary_color
            )
        )
        content.append(Spacer(1, 3))
        content.append(
            HRFlowable(
                width="60%", thickness=1, lineCap="round", color=self.accent_color
            )
        )
        content.append(Spacer(1, 30))

        return content

    def _create_segmentation_overlay(
        self, image_data: Dict[str, Any], prediction_data: Dict[str, Any]
    ) -> Optional[io.BytesIO]:
        """Create ultrasound image with segmentation overlay."""
        try:
            print(f"🔍 Creating overlay - Image data keys: {list(image_data.keys())}")
            print(
                f"🔍 Creating overlay - Prediction data keys: {list(prediction_data.keys())}"
            )

            # Get the original image
            original_image_b64 = image_data.get("image_data")
            if not original_image_b64:
                print("❌ No image_data found in image_data")
                return None

            # Decode base64 image
            image_bytes = base64.b64decode(original_image_b64)
            original_image = PILImage.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Get prediction mask if available
            prediction = prediction_data.get("prediction", {})
            mask_b64 = prediction.get("segmentationMask")

            if mask_b64:
                # Decode segmentation mask
                mask_bytes = base64.b64decode(mask_b64)
                mask_image = PILImage.open(io.BytesIO(mask_bytes))

                # Resize mask to match original image if needed
                if mask_image.size != original_image.size:
                    mask_image = mask_image.resize(
                        original_image.size, PILImage.Resampling.LANCZOS
                    )

                # Convert mask to RGBA for overlay
                if mask_image.mode != "RGBA":
                    mask_image = mask_image.convert("RGBA")

                # Create colored overlay (red/orange for fibroids)
                overlay = PILImage.new("RGBA", original_image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)

                # Convert mask to numpy array for processing
                mask_array = np.array(mask_image)

                # Create colored overlay where mask is positive
                if len(mask_array.shape) == 3:
                    mask_gray = np.mean(mask_array[:, :, :3], axis=2)
                else:
                    mask_gray = mask_array

                # Normalize mask values
                if mask_gray.max() > 1:
                    mask_gray = mask_gray / 255.0

                # Create overlay with varying intensity
                for y in range(mask_gray.shape[0]):
                    for x in range(mask_gray.shape[1]):
                        intensity = mask_gray[y, x]
                        if intensity > 0.1:  # Threshold for visibility
                            # Color intensity based on mask value
                            alpha = int(
                                intensity * 180
                            )  # Max 180 for semi-transparency
                            color = (
                                255,
                                int(165 * intensity),
                                0,
                                alpha,
                            )  # Orange to red gradient
                            overlay_draw.point((x, y), fill=color)

                # Composite the images
                original_rgba = original_image.convert("RGBA")
                result = PILImage.alpha_composite(original_rgba, overlay)
                result = result.convert("RGB")
            else:
                # No mask available, use original image
                result = original_image

            # Save to BytesIO
            img_buffer = io.BytesIO()
            result.save(img_buffer, format="PNG", dpi=(300, 300))
            img_buffer.seek(0)

            return img_buffer

        except Exception as e:
            print(f"Error creating segmentation overlay: {e}")
            return None

    def _build_scan_section(
        self, image_data: Dict[str, Any], prediction_data: Dict[str, Any]
    ) -> List:
        """Build beautiful ultrasound scan section with segmentation overlay."""
        content = []

        # Enhanced section header with icon-like styling
        content.append(
            Paragraph("🔬 Ultrasound Scan Analysis", self.styles["SectionHeader"])
        )
        content.append(Spacer(1, 10))

        # Enhanced scan information table with professional styling
        scan_info = [
            ["Scan Type:", "Uterine Ultrasound"],
            ["Analysis Model:", "U-Net++ with EfficientNet-B5 Encoder"],
            [
                "Image Resolution:",
                f"{image_data.get('width', 'N/A')} x {image_data.get('height', 'N/A')} pixels",
            ],
            ["Processing Date:", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ["Analysis Status:", "Completed Successfully"],
        ]

        scan_table = Table(scan_info, colWidths=[2.2 * inch, 4.3 * inch])
        scan_table.setStyle(
            TableStyle(
                [
                    # Enhanced typography
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    # Professional colors
                    ("TEXTCOLOR", (0, 0), (0, -1), self.primary_color),
                    ("TEXTCOLOR", (1, 0), (1, -1), self.secondary_color),
                    # Layout
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # Beautiful styling
                    ("BACKGROUND", (0, 0), (-1, -1), self.highlight_bg),
                    ("GRID", (0, 0), (-1, -1), 1, self.border_color),
                    ("ROUNDEDCORNERS", [3, 3, 3, 3]),
                    # Enhanced spacing
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )

        content.append(scan_table)
        content.append(Spacer(1, 20))

        # Enhanced ultrasound image display with professional medical framing
        overlay_image = self._create_segmentation_overlay(image_data, prediction_data)
        if overlay_image:
            try:
                # Create ReportLab Image with optimal medical viewing size
                img = Image(overlay_image, width=5.0 * inch, height=4.0 * inch)

                # Professional medical image frame with enhanced styling
                img_frame_data = [
                    ["🖼️ ULTRASOUND SCAN WITH AI SEGMENTATION OVERLAY"],  # Header
                    [""],  # Spacing
                    [img],  # Image
                    [""],  # Spacing
                    ["Figure 1: AI-Enhanced Ultrasound Analysis"],  # Caption
                ]

                img_table = Table(
                    img_frame_data,
                    colWidths=[7.0 * inch],
                    rowHeights=[
                        0.4 * inch,
                        0.1 * inch,
                        4.2 * inch,
                        0.1 * inch,
                        0.3 * inch,
                    ],
                )
                img_table.setStyle(
                    TableStyle(
                        [
                            # Header styling
                            ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (0, 0), 14),
                            ("TEXTCOLOR", (0, 0), (0, 0), self.primary_color),
                            ("ALIGN", (0, 0), (0, 0), "CENTER"),
                            ("BACKGROUND", (0, 0), (0, 0), self.highlight_bg),
                            # Image alignment
                            ("ALIGN", (0, 2), (0, 2), "CENTER"),
                            ("VALIGN", (0, 2), (0, 2), "MIDDLE"),
                            # Caption styling
                            ("FONTNAME", (0, 4), (0, 4), "Helvetica-Oblique"),
                            ("FONTSIZE", (0, 4), (0, 4), 11),
                            ("TEXTCOLOR", (0, 4), (0, 4), self.secondary_color),
                            ("ALIGN", (0, 4), (0, 4), "CENTER"),
                            # Professional border and background
                            ("BOX", (0, 0), (0, -1), 2, self.primary_color),
                            ("BACKGROUND", (0, 2), (0, 2), white),
                            ("BOX", (0, 2), (0, 2), 1, self.border_color),
                            ("ROUNDEDCORNERS", [8, 8, 8, 8]),
                            # Enhanced padding for medical presentation
                            ("LEFTPADDING", (0, 0), (0, -1), 15),
                            ("RIGHTPADDING", (0, 0), (0, -1), 15),
                            ("TOPPADDING", (0, 0), (0, 0), 12),
                            ("BOTTOMPADDING", (0, 4), (0, 4), 12),
                            ("TOPPADDING", (0, 2), (0, 2), 15),
                            ("BOTTOMPADDING", (0, 2), (0, 2), 15),
                        ]
                    )
                )

                content.append(img_table)
                content.append(Spacer(1, 20))

                # Enhanced technical details caption
                content.append(
                    Paragraph(
                        "<b>Technical Details:</b> Orange/red regions indicate fibroid tissue detected by the U-Net++ neural network. "
                        "The AI model uses EfficientNet-B5 encoder for feature extraction and provides pixel-level segmentation accuracy. "
                        "Color intensity correlates with prediction confidence levels.",
                        self.styles["Caption"],
                    )
                )
                content.append(Spacer(1, 20))

            except Exception as e:
                print(f"Error adding image to PDF: {e}")
                content.append(
                    Paragraph(
                        "⚠️ <b>Note:</b> Ultrasound image could not be displayed in this report.",
                        self.styles["ImportantNote"],
                    )
                )
                content.append(Spacer(1, 16))
        else:
            content.append(
                Paragraph(
                    "ℹ️ <b>Note:</b> Ultrasound image with segmentation overlay is not available for this analysis.",
                    self.styles["ImportantNote"],
                )
            )
            content.append(Spacer(1, 16))

        # Enhanced technical explanation with better formatting
        content.append(
            Paragraph(
                "<b>🔍 Segmentation Analysis Methodology:</b>", self.styles["Highlight"]
            )
        )
        content.append(
            Paragraph(
                "The AI model employs pixel-level analysis to identify potential fibroid regions within the ultrasound image. "
                "The colored overlay represents the model's confidence in detecting abnormal tissue, with color intensity "
                "directly corresponding to detection confidence levels. This advanced visualization technique enables "
                "precise localization of areas requiring clinical attention.",
                self.styles["CustomBody"],
            )
        )

        content.append(Spacer(1, 20))

        return content

    def _build_patient_section(self, patient_profile: Dict[str, Any]) -> List:
        """Build patient information section."""
        content = []

        content.append(Paragraph("Patient Information", self.styles["SectionHeader"]))

        # Extract patient data
        demographics = patient_profile.get("demographics", {})
        symptoms = patient_profile.get("symptoms", {})
        medical_history = patient_profile.get("medicalHistory", {})

        patient_info = []

        if demographics.get("age"):
            patient_info.append(["Age:", f"{demographics['age']} years"])

        if demographics.get("sex"):
            patient_info.append(["Sex:", demographics["sex"].title()])

        if demographics.get("ethnicity"):
            patient_info.append(["Ethnicity:", demographics["ethnicity"]])

        # Add relevant symptoms
        symptom_list = []
        if symptoms.get("heavyMenstrualBleeding"):
            symptom_list.append("Heavy menstrual bleeding")
        if symptoms.get("pelvicPain"):
            symptom_list.append("Pelvic pain")
        if symptoms.get("pelvicPressure"):
            symptom_list.append("Pelvic pressure")

        if symptom_list:
            patient_info.append(["Reported Symptoms:", ", ".join(symptom_list)])

        if medical_history.get("familyHistoryFibroids"):
            patient_info.append(["Family History:", "Positive for uterine fibroids"])

        if patient_info:
            table = Table(patient_info, colWidths=[2 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("TEXTCOLOR", (0, 0), (0, -1), self.secondary_color),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            content.append(table)

        content.append(Spacer(1, 16))

        return content

    def _build_enhanced_analysis_summary(
        self,
        prediction_data: Dict[str, Any],
        xai_data: Optional[Dict[str, Any]],
        include_integrated_gradients: bool,
    ) -> List:
        """Build beautiful enhanced analysis summary section with detailed metrics."""
        content = []

        content.append(
            Paragraph("📊 Comprehensive Analysis Summary", self.styles["SectionHeader"])
        )
        content.append(Spacer(1, 15))

        prediction = prediction_data.get("prediction", {})

        # Enhanced detection status with badge-like styling
        if prediction.get("fibroidDetected"):
            status_text = (
                f"🔴 <font color='{self.error_color}'><b>FIBROIDS DETECTED</b></font>"
            )
            status_bg_color = HexColor("#fef2f2")  # Light red background
        else:
            status_text = f"✅ <font color='{self.success_color}'><b>NO FIBROIDS DETECTED</b></font>"
            status_bg_color = HexColor("#f0fdf4")  # Light green background

        # Create status badge
        status_table = Table(
            [[Paragraph(status_text, self.styles["StatusBadge"])]], colWidths=[6 * inch]
        )
        status_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, 0), (-1, -1), status_bg_color),
                    ("BOX", (0, 0), (-1, -1), 2, self.border_color),
                    ("ROUNDEDCORNERS", [8, 8, 8, 8]),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("LEFTPADDING", (0, 0), (-1, -1), 20),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                ]
            )
        )

        content.append(status_table)
        content.append(Spacer(1, 20))

        # Enhanced Primary Analysis Metrics with icons
        content.append(
            Paragraph("🎯 <b>Primary Analysis Metrics</b>", self.styles["Highlight"])
        )
        content.append(Spacer(1, 10))

        confidence = prediction.get("confidence", 0) * 100
        fibroid_count = prediction.get("fibroidCount", 0)

        # Color-coded confidence level
        if confidence >= 90:
            confidence_color = self.success_color
            confidence_icon = "🟢"
        elif confidence >= 70:
            confidence_color = self.warning_color
            confidence_icon = "🟡"
        else:
            confidence_color = self.error_color
            confidence_icon = "🔴"

        primary_metrics = [
            [
                "🤖 Model Confidence:",
                Paragraph(
                    f"{confidence_icon} <font color='{confidence_color}'><b>{confidence:.1f}%</b></font>",
                    self.styles["CustomBody"],
                ),
            ],
            [
                "🔢 Number of Fibroids Detected:",
                Paragraph(f"<b>{fibroid_count}</b>", self.styles["CustomBody"]),
            ],
        ]

        # Add detailed fibroid information
        if fibroid_count > 0:
            fibroid_areas = prediction.get("fibroidAreas", [])
            if fibroid_areas:
                # Calculate size statistics
                sizes = [area.get("area", 0) for area in fibroid_areas]
                total_area = sum(sizes)
                avg_size = total_area / len(sizes) if sizes else 0
                largest_size = max(sizes) if sizes else 0

                # Severity analysis
                severities = [area.get("severity", "unknown") for area in fibroid_areas]
                severity_counts = {
                    "mild": severities.count("mild"),
                    "moderate": severities.count("moderate"),
                    "severe": severities.count("severe"),
                }
                most_severe = max(
                    severities,
                    key=lambda x: ["mild", "moderate", "severe"].index(x)
                    if x in ["mild", "moderate", "severe"]
                    else 0,
                )

                # Enhanced metrics with icons and color coding
                severity_color = (
                    self.success_color
                    if most_severe == "mild"
                    else (
                        self.warning_color
                        if most_severe == "moderate"
                        else self.error_color
                    )
                )
                severity_icon = (
                    "🟢"
                    if most_severe == "mild"
                    else ("🟡" if most_severe == "moderate" else "🔴")
                )

                primary_metrics.extend(
                    [
                        [
                            "📏 Total Affected Area:",
                            Paragraph(
                                f"<b>{total_area:.1f} mm²</b>",
                                self.styles["CustomBody"],
                            ),
                        ],
                        [
                            "📊 Average Fibroid Size:",
                            Paragraph(
                                f"<b>{avg_size:.1f} mm²</b>", self.styles["CustomBody"]
                            ),
                        ],
                        [
                            "📈 Largest Fibroid Size:",
                            Paragraph(
                                f"<b>{largest_size:.1f} mm²</b>",
                                self.styles["CustomBody"],
                            ),
                        ],
                        [
                            "⚠️ Highest Severity Level:",
                            Paragraph(
                                f"{severity_icon} <font color='{severity_color}'><b>{most_severe.title()}</b></font>",
                                self.styles["CustomBody"],
                            ),
                        ],
                        [
                            "📋 Severity Distribution:",
                            Paragraph(
                                f"<font color='{self.success_color}'>Mild: {severity_counts['mild']}</font>, <font color='{self.warning_color}'>Moderate: {severity_counts['moderate']}</font>, <font color='{self.error_color}'>Severe: {severity_counts['severe']}</font>",
                                self.styles["CustomBody"],
                            ),
                        ],
                    ]
                )

        primary_table = Table(primary_metrics, colWidths=[2.8 * inch, 3.7 * inch])
        primary_table.setStyle(
            TableStyle(
                [
                    # Enhanced typography
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    # Professional colors
                    ("TEXTCOLOR", (0, 0), (0, -1), self.primary_color),
                    ("TEXTCOLOR", (1, 0), (1, -1), self.secondary_color),
                    # Layout
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # Beautiful styling with alternating row colors
                    ("BACKGROUND", (0, 0), (-1, 0), self.highlight_bg),
                    ("BACKGROUND", (0, 2), (-1, 2), self.highlight_bg),
                    ("BACKGROUND", (0, 4), (-1, 4), self.highlight_bg),
                    ("BACKGROUND", (0, 1), (-1, 1), self.light_bg),
                    ("BACKGROUND", (0, 3), (-1, 3), self.light_bg),
                    # Professional borders
                    ("GRID", (0, 0), (-1, -1), 1, self.border_color),
                    ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                    # Enhanced spacing
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        content.append(primary_table)
        content.append(Spacer(1, 16))

        # Enhanced XAI Analysis Metrics
        if xai_data:
            content.append(Spacer(1, 15))
            content.append(
                Paragraph(
                    "🧠 <b>AI Explainability Metrics</b>", self.styles["Highlight"]
                )
            )
            content.append(Spacer(1, 10))

            analysis = xai_data.get("analysis", {})
            xai_metrics = []

            # Enhanced GradCAM metrics with visual indicators
            if "gradcam" in analysis:
                gradcam_stats = analysis["gradcam"].get("statistics", {})
                if gradcam_stats:
                    attention_mean = gradcam_stats.get("attentionMean", 0)
                    attention_std = gradcam_stats.get("attentionStd", 0)
                    attention_ratio = gradcam_stats.get("attentionRatio", 0)

                    # Color-coded attention metrics
                    focus_color = (
                        self.success_color
                        if attention_ratio > 2.0
                        else (
                            self.warning_color
                            if attention_ratio > 1.0
                            else self.error_color
                        )
                    )
                    focus_icon = (
                        "🎯"
                        if attention_ratio > 2.0
                        else ("⚡" if attention_ratio > 1.0 else "📡")
                    )

                    xai_metrics.extend(
                        [
                            [
                                "🔍 GradCAM Attention Focus:",
                                Paragraph(
                                    f"<b>{attention_mean:.4f}</b>",
                                    self.styles["CustomBody"],
                                ),
                            ],
                            [
                                "📊 GradCAM Attention Variability:",
                                Paragraph(
                                    f"<b>{attention_std:.4f}</b>",
                                    self.styles["CustomBody"],
                                ),
                            ],
                            [
                                "🎯 GradCAM Concentration Ratio:",
                                Paragraph(
                                    f"<font color='{focus_color}'><b>{attention_ratio:.2f}</b></font>",
                                    self.styles["CustomBody"],
                                ),
                            ],
                        ]
                    )

                    # Enhanced interpretation with visual indicators
                    if attention_ratio > 3.0:
                        focus_interpretation = f"🎯 <font color='{self.success_color}'><b>Highly focused attention</b></font>"
                    elif attention_ratio > 1.5:
                        focus_interpretation = f"⚡ <font color='{self.warning_color}'><b>Moderately focused attention</b></font>"
                    else:
                        focus_interpretation = f"📡 <font color='{self.error_color}'><b>Distributed attention</b></font>"

                    xai_metrics.append(
                        [
                            "🧠 GradCAM Focus Interpretation:",
                            Paragraph(focus_interpretation, self.styles["CustomBody"]),
                        ]
                    )

            # Enhanced Integrated Gradients metrics
            if "integratedGradients" in analysis:
                ig_stats = analysis["integratedGradients"].get("statistics", {})
                if ig_stats:
                    attr_mean = ig_stats.get("attributionMean", 0)
                    attr_std = ig_stats.get("attributionStd", 0)
                    attr_max = ig_stats.get("attributionMax", 0)
                    attr_min = ig_stats.get("attributionMin", 0)

                    # Color-coded attribution strength
                    attr_strength = abs(attr_mean)
                    attr_color = (
                        self.success_color
                        if attr_strength > 0.001
                        else (
                            self.warning_color
                            if attr_strength > 0.0001
                            else self.secondary_color
                        )
                    )

                    xai_metrics.extend(
                        [
                            [
                                "🔬 IG Attribution Mean:",
                                Paragraph(
                                    f"<font color='{attr_color}'><b>{attr_mean:.6f}</b></font>",
                                    self.styles["CustomBody"],
                                ),
                            ],
                            [
                                "📈 IG Attribution Std Dev:",
                                Paragraph(
                                    f"<b>{attr_std:.6f}</b>", self.styles["CustomBody"]
                                ),
                            ],
                            [
                                "📊 IG Attribution Range:",
                                Paragraph(
                                    f"<b>{attr_min:.6f}</b> to <b>{attr_max:.6f}</b>",
                                    self.styles["CustomBody"],
                                ),
                            ],
                        ]
                    )
            elif include_integrated_gradients:
                xai_metrics.append(
                    [
                        "🔬 Integrated Gradients:",
                        Paragraph(
                            f"<font color='{self.warning_color}'><b>Not performed (optional analysis)</b></font>",
                            self.styles["CustomBody"],
                        ),
                    ]
                )

            if xai_metrics:
                xai_table = Table(xai_metrics, colWidths=[2.8 * inch, 3.7 * inch])
                xai_table.setStyle(
                    TableStyle(
                        [
                            # Enhanced typography
                            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 11),
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            # Professional colors
                            ("TEXTCOLOR", (0, 0), (0, -1), self.primary_color),
                            ("TEXTCOLOR", (1, 0), (1, -1), self.secondary_color),
                            # Layout
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            # Beautiful alternating backgrounds
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                HexColor("#e0f2fe"),
                            ),  # Light blue
                            ("BACKGROUND", (0, 2), (-1, 2), HexColor("#e0f2fe")),
                            ("BACKGROUND", (0, 4), (-1, 4), HexColor("#e0f2fe")),
                            (
                                "BACKGROUND",
                                (0, 1),
                                (-1, 1),
                                HexColor("#f0f9ff"),
                            ),  # Very light blue
                            ("BACKGROUND", (0, 3), (-1, 3), HexColor("#f0f9ff")),
                            # Professional borders
                            ("GRID", (0, 0), (-1, -1), 1, self.border_color),
                            ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                            # Enhanced spacing
                            ("LEFTPADDING", (0, 0), (-1, -1), 12),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                            ("TOPPADDING", (0, 0), (-1, -1), 10),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                        ]
                    )
                )

                content.append(xai_table)
                content.append(Spacer(1, 20))

        return content

    def _build_detailed_findings(self, prediction_data: Dict[str, Any]) -> List:
        """Build detailed findings section."""
        content = []

        prediction = prediction_data.get("prediction", {})
        fibroid_areas = prediction.get("fibroidAreas", [])

        if not fibroid_areas:
            content.append(Paragraph("Detailed Findings", self.styles["SectionHeader"]))
            content.append(
                Paragraph(
                    "No fibroids were detected in the analyzed image.",
                    self.styles["CustomBody"],
                )
            )
            content.append(Spacer(1, 16))
            return content

        content.append(Paragraph("Detailed Findings", self.styles["SectionHeader"]))

        # Create findings table
        findings_data = [["Fibroid #", "Severity", "Area (mm²)", "Location"]]

        for i, fibroid in enumerate(fibroid_areas, 1):
            severity = fibroid.get("severity", "unknown").title()
            area = fibroid.get("area", 0)
            location = fibroid.get("location", {})
            loc_str = f"({location.get('x', 0):.1f}, {location.get('y', 0):.1f})"

            findings_data.append([str(i), severity, f"{area:.1f}", loc_str])

        table = Table(
            findings_data, colWidths=[1 * inch, 1.5 * inch, 1.5 * inch, 2 * inch]
        )
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 0), (-1, 0), self.primary_color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 1, self.primary_color),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f8fafc")]),
                ]
            )
        )

        content.append(table)
        content.append(Spacer(1, 16))

        return content

    def _build_xai_section(
        self, xai_data: Dict[str, Any], include_integrated_gradients: bool = True
    ) -> List:
        """Build XAI explanation section."""
        content = []

        content.append(Paragraph("AI Model Explanation", self.styles["SectionHeader"]))

        # Check what XAI methods were performed
        analysis = xai_data.get("analysis", {})
        has_gradcam = "gradcam" in analysis
        has_integrated_gradients = "integratedGradients" in analysis

        # Add methods performed section
        content.append(
            Paragraph("<b>Analysis Methods Performed:</b>", self.styles["Highlight"])
        )
        methods_performed = []

        if has_gradcam:
            methods_performed.append(
                "✓ GradCAM Analysis - Shows attention regions where the model focuses"
            )

        if has_integrated_gradients:
            methods_performed.append(
                "✓ Integrated Gradients Analysis - Provides pixel-level attribution analysis"
            )
        else:
            if include_integrated_gradients:
                methods_performed.append(
                    "• Integrated Gradients Analysis - Not performed (optional)"
                )
            else:
                methods_performed.append(
                    "• Integrated Gradients Analysis - Not performed for this report"
                )

        for method in methods_performed:
            content.append(Paragraph(method, self.styles["CustomBody"]))
        content.append(Spacer(1, 12))

        # Add GradCAM statistics if available
        if has_gradcam:
            gradcam_stats = analysis.get("gradcam", {}).get("statistics", {})
            if gradcam_stats:
                content.append(
                    Paragraph(
                        "<b>GradCAM Analysis Results:</b>", self.styles["Highlight"]
                    )
                )

                attention_mean = gradcam_stats.get("attentionMean", 0)
                attention_ratio = gradcam_stats.get("attentionRatio", 0)

                content.append(
                    Paragraph(
                        f"• Attention Focus Score: {attention_mean:.3f}",
                        self.styles["CustomBody"],
                    )
                )
                content.append(
                    Paragraph(
                        f"• Attention Concentration Ratio: {attention_ratio:.2f}",
                        self.styles["CustomBody"],
                    )
                )

                if attention_ratio > 2.0:
                    content.append(
                        Paragraph(
                            "• Model shows good focus on relevant regions",
                            self.styles["CustomBody"],
                        )
                    )
                elif attention_ratio < 1.0:
                    content.append(
                        Paragraph(
                            "• Model attention is distributed across the image",
                            self.styles["CustomBody"],
                        )
                    )
                else:
                    content.append(
                        Paragraph(
                            "• Model shows moderate focus on specific regions",
                            self.styles["CustomBody"],
                        )
                    )

                content.append(Spacer(1, 12))

        # Add general explanation if available
        explanation = analysis.get("explanation", {})

        if explanation.get("summary"):
            content.append(Paragraph("<b>Summary:</b>", self.styles["Highlight"]))
            content.append(Paragraph(explanation["summary"], self.styles["CustomBody"]))
            content.append(Spacer(1, 12))

        if explanation.get("keyFindings"):
            content.append(Paragraph("<b>Key Findings:</b>", self.styles["Highlight"]))
            for finding in explanation["keyFindings"]:
                content.append(Paragraph(f"• {finding}", self.styles["CustomBody"]))
            content.append(Spacer(1, 12))

        if explanation.get("confidence"):
            confidence = explanation["confidence"] * 100
            content.append(
                Paragraph(
                    f"<b>Explanation Confidence:</b> {confidence:.1f}%",
                    self.styles["Highlight"],
                )
            )

        # Add note about analysis completeness
        if not include_integrated_gradients:
            content.append(Spacer(1, 8))
            content.append(
                Paragraph(
                    "<b>Note:</b> This report was generated with GradCAM analysis only. "
                    "Integrated Gradients analysis can be performed separately for additional pixel-level insights.",
                    self.styles["CustomBody"],
                )
            )

        content.append(Spacer(1, 16))

        return content

    def _build_doctor_notes(self, doctor_notes: str) -> List:
        """Build beautiful doctor notes section with professional medical styling."""
        content = []

        # Professional section header
        header_table_data = [["👨‍⚕️ CLINICAL NOTES & PHYSICIAN OBSERVATIONS"]]
        header_table = Table(header_table_data, colWidths=[7.5 * inch])
        header_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 16),
                    ("TEXTCOLOR", (0, 0), (0, 0), self.primary_color),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("BACKGROUND", (0, 0), (0, 0), self.highlight_bg),
                    ("BOX", (0, 0), (0, 0), 1.5, self.primary_color),
                    ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                    ("TOPPADDING", (0, 0), (0, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (0, 0), 10),
                ]
            )
        )
        content.append(header_table)
        content.append(Spacer(1, 15))

        # Enhanced notes content with professional framing
        notes_table_data = [[doctor_notes]]
        notes_table = Table(notes_table_data, colWidths=[7.5 * inch])
        notes_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica"),
                    ("FONTSIZE", (0, 0), (0, 0), 11),
                    ("TEXTCOLOR", (0, 0), (0, 0), self.secondary_color),
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("VALIGN", (0, 0), (0, 0), "TOP"),
                    ("BACKGROUND", (0, 0), (0, 0), self.light_bg),
                    ("BOX", (0, 0), (0, 0), 1, self.border_color),
                    ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                    ("LEFTPADDING", (0, 0), (0, 0), 20),
                    ("RIGHTPADDING", (0, 0), (0, 0), 20),
                    ("TOPPADDING", (0, 0), (0, 0), 15),
                    ("BOTTOMPADDING", (0, 0), (0, 0), 15),
                ]
            )
        )

        content.append(notes_table)
        content.append(Spacer(1, 25))

        return content

    def _build_recommendations(
        self, prediction_data: Dict[str, Any], xai_data: Optional[Dict[str, Any]]
    ) -> List:
        """Build beautiful recommendations section with professional medical styling."""
        content = []

        # Professional section header
        header_table_data = [["💡 CLINICAL RECOMMENDATIONS & NEXT STEPS"]]
        header_table = Table(header_table_data, colWidths=[7.5 * inch])
        header_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 16),
                    ("TEXTCOLOR", (0, 0), (0, 0), self.primary_color),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("BACKGROUND", (0, 0), (0, 0), self.highlight_bg),
                    ("BOX", (0, 0), (0, 0), 1.5, self.primary_color),
                    ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                    ("TOPPADDING", (0, 0), (0, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (0, 0), 10),
                ]
            )
        )
        content.append(header_table)
        content.append(Spacer(1, 20))

        prediction = prediction_data.get("prediction", {})

        if prediction.get("fibroidDetected"):
            recommendations = [
                "🏥 Consult with a gynecologist for clinical correlation and comprehensive treatment planning.",
                "🔍 Consider MRI imaging for detailed characterization if clinically indicated.",
                "📊 Monitor symptoms and consider follow-up imaging if symptoms worsen or change.",
                "💊 Discuss treatment options based on symptoms, size, and location of detected fibroids.",
                "📋 Document findings in patient medical record for future reference.",
            ]

            # Add severity-specific recommendations
            fibroid_areas = prediction.get("fibroidAreas", [])
            if any(area.get("severity") == "severe" for area in fibroid_areas):
                recommendations.insert(
                    1,
                    "🚨 Large fibroids detected - consider urgent gynecological consultation.",
                )
        else:
            recommendations = [
                "✅ Continue routine gynecological care and regular check-ups.",
                "🔍 If symptoms persist, consider additional imaging or clinical evaluation.",
                "📅 Maintain regular follow-up as clinically appropriate.",
                "📝 Document negative findings for baseline comparison in future studies.",
            ]

        # Add XAI-based recommendations if available
        if xai_data:
            xai_recommendations = (
                xai_data.get("analysis", {})
                .get("explanation", {})
                .get("recommendations", [])
            )
            if xai_recommendations:
                recommendations.append("🤖 AI Analysis Recommendations:")
                recommendations.extend([f"   • {rec}" for rec in xai_recommendations])

        # Create recommendations table with professional styling
        rec_data = []
        for i, rec in enumerate(recommendations, 1):
            rec_data.append([f"{i}.", rec])

        rec_table = Table(rec_data, colWidths=[0.5 * inch, 7.0 * inch])
        rec_table.setStyle(
            TableStyle(
                [
                    # Typography
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    # Colors
                    ("TEXTCOLOR", (0, 0), (0, -1), self.primary_color),
                    ("TEXTCOLOR", (1, 0), (1, -1), self.secondary_color),
                    # Layout
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (1, 0), (1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    # Professional styling
                    ("BACKGROUND", (0, 0), (-1, -1), self.light_bg),
                    ("BOX", (0, 0), (-1, -1), 1, self.border_color),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, self.border_color),
                    ("ROUNDEDCORNERS", [5, 5, 5, 5]),
                    # Enhanced spacing
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        content.append(rec_table)
        content.append(Spacer(1, 25))

        return content

    def _build_footer(self) -> List:
        """Build beautiful professional medical report footer."""
        content = []

        # Professional separator
        content.append(Spacer(1, 30))
        content.append(HRFlowable(width="100%", thickness=2, color=self.primary_color))
        content.append(Spacer(1, 15))

        # Enhanced footer with medical disclaimer in professional table format
        footer_table_data = [
            ["⚠️ IMPORTANT MEDICAL DISCLAIMER"],
            [
                "This report is generated by an AI-powered medical imaging analysis system using advanced neural networks. "
                "The results should be interpreted by qualified medical professionals and used in conjunction with "
                "clinical findings and other diagnostic information. This analysis is not a substitute for "
                "professional medical diagnosis or treatment."
            ],
            [""],
            ["🏥 Medical Imaging Center - AI Analysis Division"],
            [
                f"📅 Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
            ],
            ["🤖 Powered by U-Net++ Neural Network with EfficientNet-B5 Encoder"],
            ["📧 For technical support: ai-support@medcenter.com"],
        ]

        footer_table = Table(footer_table_data, colWidths=[7.5 * inch])
        footer_table.setStyle(
            TableStyle(
                [
                    # Disclaimer header styling
                    ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, 0), 12),
                    ("TEXTCOLOR", (0, 0), (0, 0), self.error_color),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("BACKGROUND", (0, 0), (0, 0), HexColor("#fef2f2")),
                    # Disclaimer text styling
                    ("FONTNAME", (0, 1), (0, 1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (0, 1), 10),
                    ("TEXTCOLOR", (0, 1), (0, 1), self.secondary_color),
                    ("ALIGN", (0, 1), (0, 1), "LEFT"),
                    ("BACKGROUND", (0, 1), (0, 1), HexColor("#fef2f2")),
                    # Institution info styling
                    ("FONTNAME", (0, 3), (0, 3), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 3), (0, 3), 11),
                    ("TEXTCOLOR", (0, 3), (0, 3), self.primary_color),
                    ("ALIGN", (0, 3), (0, 3), "CENTER"),
                    ("BACKGROUND", (0, 3), (0, 3), self.light_bg),
                    # Technical details styling
                    ("FONTNAME", (0, 4), (0, 6), "Helvetica"),
                    ("FONTSIZE", (0, 4), (0, 6), 9),
                    ("TEXTCOLOR", (0, 4), (0, 6), self.secondary_color),
                    ("ALIGN", (0, 4), (0, 6), "CENTER"),
                    ("BACKGROUND", (0, 4), (0, 6), self.light_bg),
                    # Professional borders and styling
                    ("BOX", (0, 0), (0, -1), 1.5, self.primary_color),
                    ("INNERGRID", (0, 0), (0, -1), 0.5, self.border_color),
                    ("ROUNDEDCORNERS", [8, 8, 8, 8]),
                    # Enhanced padding
                    ("LEFTPADDING", (0, 0), (0, -1), 20),
                    ("RIGHTPADDING", (0, 0), (0, -1), 20),
                    ("TOPPADDING", (0, 0), (0, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (0, 0), 12),
                    ("TOPPADDING", (0, 1), (0, 1), 10),
                    ("BOTTOMPADDING", (0, 1), (0, 1), 15),
                    ("TOPPADDING", (0, 3), (0, 6), 8),
                    ("BOTTOMPADDING", (0, 3), (0, 6), 8),
                ]
            )
        )

        content.append(footer_table)
        content.append(Spacer(1, 20))

        # Final signature line
        signature_table_data = [
            [
                "🔬 Advanced AI Medical Imaging Analysis System",
                f"📋 Document ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ]
        ]

        signature_table = Table(
            signature_table_data, colWidths=[4.5 * inch, 3.0 * inch]
        )
        signature_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Oblique"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("TEXTCOLOR", (0, 0), (-1, -1), self.secondary_color),
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )

        content.append(signature_table)

        return content
