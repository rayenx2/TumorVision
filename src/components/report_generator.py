"""PDF report generator for kidney CT classification results."""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.components.gradcam import GradCAMResult
from src.components.uncertainty import UncertaintyResult
from src.entity.config_entity import ReportConfig
from src.utils.exception import PredictionError
from src.utils.logger import logger


@dataclass
class ReportInput:
    """All data required to generate a clinical report."""

    image_path: Path
    gradcam_result: GradCAMResult
    uncertainty_result: UncertaintyResult
    case_id: str
    patient_id: str | None = None


class ReportGenerator:
    """Generate professional PDF reports from prediction results."""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = self._setup_styles()

    def generate(self, report_input: ReportInput) -> Path:
        """Generate a PDF report from prediction results."""
        try:
            case_id = report_input.case_id
            timestamp = datetime.now()
            filename = f"case_{case_id}.pdf"
            output_path = self.output_dir / filename

            logger.info("Generating report for case %s", case_id)

            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                topMargin=2 * cm,
                bottomMargin=2 * cm,
                leftMargin=2 * cm,
                rightMargin=2 * cm,
                title=f"Kidney CT Analysis - Case {case_id}",
            )

            story = []

            self._build_header(story, case_id, timestamp)
            self._build_image_info(story, report_input)
            self._build_prediction_section(story, report_input)
            self._build_uncertainty_section(story, report_input.uncertainty_result)
            self._build_visual_evidence(story, report_input)
            self._build_probability_chart(story, report_input.uncertainty_result)
            self._build_disclaimer(story)

            doc.build(story, onFirstPage=self._page_footer, onLaterPages=self._page_footer)

            logger.info("Report saved to: %s", output_path)
            return output_path

        except Exception as e:
            raise PredictionError(e, sys)

    def _setup_styles(self):
        styles = getSampleStyleSheet()

        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontSize=20,
                textColor=colors.HexColor("#1a5490"),
                alignment=1,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportSubtitle",
                parent=styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#7f8c8d"),
                alignment=1,
                spaceAfter=20,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=styles["Heading2"],
                fontSize=13,
                textColor=colors.HexColor("#2c3e50"),
                spaceBefore=12,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="WarningBox",
                parent=styles["Normal"],
                fontSize=11,
                textColor=colors.HexColor("#c0392b"),
                backColor=colors.HexColor("#fadbd8"),
                borderColor=colors.HexColor("#c0392b"),
                borderWidth=1,
                borderPadding=10,
                spaceAfter=12,
            )
        )
        styles.add(
            ParagraphStyle(
                name="InfoBox",
                parent=styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#27ae60"),
                backColor=colors.HexColor("#d5f5e3"),
                borderPadding=8,
                spaceAfter=12,
            )
        )
        styles.add(
            ParagraphStyle(
                name="Disclaimer",
                parent=styles["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#7f8c8d"),
                alignment=4,
                spaceAfter=12,
            )
        )

        return styles

    def _build_header(self, story, case_id, timestamp):
        story.append(Paragraph(self.config.organization_name, self.styles["ReportTitle"]))
        story.append(Paragraph("Kidney CT Analysis Report", self.styles["ReportTitle"]))
        story.append(
            Paragraph(
                f"Case ID: {case_id}  |  Generated: {timestamp.strftime('%B %d, %Y at %H:%M')}",
                self.styles["ReportSubtitle"],
            )
        )

    def _build_image_info(self, story, report_input):
        story.append(Paragraph("Image Information", self.styles["SectionHeader"]))

        info_data = [
            ["Filename:", report_input.image_path.name],
            [
                "Image Size:",
                f"{self.config.image_size[0]} x {self.config.image_size[1]} pixels",
            ],
            ["Channels:", "RGB (3 channels)"],
            ["Patient ID:", report_input.patient_id or "Not provided"],
        ]

        table = Table(info_data, colWidths=[4 * cm, 12 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#2c3e50")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))

    def _build_prediction_section(self, story, report_input):
        gradcam = report_input.gradcam_result
        uncertainty = report_input.uncertainty_result

        story.append(Paragraph("AI Prediction", self.styles["SectionHeader"]))

        confidence_pct = uncertainty.confidence * 100
        prediction_text = (
            f"<b>Predicted Class:</b> {uncertainty.predicted_class}<br/>"
            f"<b>Mean Confidence (MC Dropout):</b> {confidence_pct:.2f}%<br/>"
            f"<b>Single-Shot Confidence:</b> {gradcam.confidence * 100:.2f}%"
        )

        if uncertainty.is_uncertain:
            story.append(Paragraph(prediction_text, self.styles["WarningBox"]))
            story.append(
                Paragraph(
                    "<b>HIGH UNCERTAINTY DETECTED</b><br/>"
                    "This prediction has high uncertainty or low confidence. "
                    "Radiologist review is strongly recommended before clinical decision.",
                    self.styles["WarningBox"],
                )
            )
        else:
            story.append(Paragraph(prediction_text, self.styles["InfoBox"]))

        story.append(Spacer(1, 12))

    def _build_uncertainty_section(self, story, uncertainty):
        story.append(Paragraph("Uncertainty Analysis", self.styles["SectionHeader"]))

        def warn(value, threshold, higher_is_worse=True):
            if higher_is_worse:
                return "ALERT" if value > threshold else "OK"
            return "ALERT" if value < threshold else "OK"

        metrics_data = [
            ["Metric", "Value", "Status"],
            [
                "Uncertainty Score (std)",
                f"{uncertainty.uncertainty_score:.4f}",
                warn(uncertainty.uncertainty_score, self.config.uncertainty_threshold),
            ],
            [
                "Mutual Information (epistemic)",
                f"{uncertainty.mutual_information:.4f}",
                warn(uncertainty.mutual_information, self.config.uncertainty_threshold),
            ],
            ["Predictive Entropy (total)", f"{uncertainty.predictive_entropy:.4f}", ""],
            [
                "Probability Margin",
                f"{uncertainty.probability_margin:.4f}",
                warn(uncertainty.probability_margin, 0.3, higher_is_worse=False),
            ],
            ["MC Iterations", str(uncertainty.iterations), ""],
        ]

        table = Table(metrics_data, colWidths=[7 * cm, 5 * cm, 2 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8f9fa")],
                    ),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))

    def _build_visual_evidence(self, story, report_input):
        story.append(Paragraph("Visual Evidence", self.styles["SectionHeader"]))

        gradcam = report_input.gradcam_result

        if gradcam.overlay_path is None or not Path(gradcam.overlay_path).exists():
            raise ValueError(
                "GradCAM overlay file not found. "
                "Generate Grad-CAM with output_path before creating report."
            )

        original_img = Image(str(report_input.image_path), width=7 * cm, height=7 * cm)
        overlay_img = Image(str(gradcam.overlay_path), width=7 * cm, height=7 * cm)

        image_table = Table(
            [
                [original_img, overlay_img],
                [
                    Paragraph("<i>Original CT Scan</i>", self.styles["Normal"]),
                    Paragraph("<i>Grad-CAM Heatmap Overlay</i>", self.styles["Normal"]),
                ],
            ],
            colWidths=[8 * cm, 8 * cm],
        )
        image_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ]
            )
        )
        story.append(image_table)
        story.append(Spacer(1, 12))

    def _build_probability_chart(self, story, uncertainty):
        story.append(Paragraph("Per-Class Probabilities", self.styles["SectionHeader"]))

        prob_data = [["Class", "Mean Probability", "Std Dev", "Visual"]]

        for class_name in self.config.class_names:
            mean = uncertainty.probabilities[class_name]
            std = uncertainty.probability_std[class_name]
            bar_length = int(mean * 20)
            bar = "#" * bar_length + "-" * (20 - bar_length)

            prob_data.append(
                [
                    class_name,
                    f"{mean:.4f} ({mean * 100:.1f}%)",
                    f"+/- {std:.4f}",
                    bar,
                ]
            )

        table = Table(prob_data, colWidths=[3 * cm, 4 * cm, 3 * cm, 6 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (3, 1), (3, -1), "Courier"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("FONTSIZE", (3, 1), (3, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8f9fa")],
                    ),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))

    def _build_disclaimer(self, story):
        story.append(Spacer(1, 20))
        story.append(Paragraph("Disclaimer", self.styles["SectionHeader"]))
        story.append(Paragraph(self.config.disclaimer, self.styles["Disclaimer"]))

    def _page_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#95a5a6"))
        canvas.drawCentredString(A4[0] / 2, 1 * cm, f"Page {doc.page}")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, "Generated by AI Decision-Support Tool")
        canvas.restoreState()
