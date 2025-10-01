"""Vision Agent for dental image analysis using YOLO with quality checks.

This agent:
- Validates image quality (blur, lighting, format)
- Runs YOLO detection on dental images
- Generates spatial insights using Gemini vision
- Provides actionable feedback on image quality
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.agents.specialized.base_agent import BaseAgent
from src.agents.state_models import AgentState, DetectionResult
from src.tools.yolo_tool import detect_issues, validate_image, InvalidImageError

logger = logging.getLogger(__name__)


class ImageQuality(BaseModel):
    """Image quality assessment result."""

    is_acceptable: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class VisionResult(BaseModel):
    """Complete vision analysis result."""

    detections: List[DetectionResult]
    spatial_insights: str
    annotated_image_path: str
    image_quality: ImageQuality
    confidence: float = Field(ge=0.0, le=1.0)


class VisionAgent(BaseAgent):
    """Agent for comprehensive dental image analysis."""

    def __init__(self):
        super().__init__(
            name="VisionAgent",
            max_retries=2,
            retry_delay=1.0,
            enable_circuit_breaker=True,
        )

    def _execute(self, state: AgentState, **kwargs) -> Dict[str, Any]:
        """Execute vision analysis pipeline."""
        image_path = state.image_path

        if not image_path:
            raise ValueError("No image path provided to VisionAgent")

        logger.info(f"VisionAgent: Processing image at {image_path}")

        # Step 1: Quality Assessment
        quality = self._assess_image_quality(image_path)
        logger.info(
            f"VisionAgent: Quality - Acceptable={quality.is_acceptable}, "
            f"Score={quality.quality_score:.2f}"
        )

        if not quality.is_acceptable:
            logger.warning(f"VisionAgent: Image quality issues: {quality.issues}")
            return {
                "success": False,
                "image_quality": quality,
                "detections": [],
                "spatial_insights": (
                    f"Image quality needs improvement. Issues: {', '.join(quality.issues)}. "
                    f"Recommendations: {', '.join(quality.recommendations)}"
                ),
                "confidence": quality.quality_score,
            }

        # Step 2: Validate image format/size
        try:
            validate_image(image_path)
        except InvalidImageError as e:
            logger.error(f"VisionAgent: Validation failed - {e}")
            return {
                "success": False,
                "image_quality": quality,
                "detections": [],
                "spatial_insights": f"Image validation failed: {str(e)}",
                "confidence": 0.0,
            }

        # Step 3: Run YOLO detection
        try:
            detections_json, annotated_path, spatial_insights = detect_issues(image_path)

            # Parse detections to structured format
            import json
            detections_list = json.loads(detections_json)

            detections = [
                DetectionResult(
                    class_name=d["class"],
                    confidence=d["confidence"],
                    bbox=d["bbox"],
                )
                for d in detections_list
            ]

            logger.info(f"VisionAgent: Detected {len(detections)} findings")
            for det in detections:
                logger.debug(
                    f"VisionAgent: - {det.class_name} (conf={det.confidence:.2f})"
                )

            # Calculate overall confidence
            avg_confidence = (
                sum(d.confidence for d in detections) / len(detections)
                if detections
                else 0.5
            )
            # Combine with image quality
            overall_confidence = (avg_confidence + quality.quality_score) / 2

            logger.info(f"VisionAgent: Spatial insights generated ({len(spatial_insights)} chars)")
            logger.debug(f"VisionAgent: Spatial excerpt: {spatial_insights[:150]}...")

            return {
                "success": True,
                "detections": detections,
                "spatial_insights": spatial_insights,
                "annotated_image_path": annotated_path,
                "image_quality": quality,
                "confidence": overall_confidence,
            }

        except Exception as e:
            logger.error(f"VisionAgent: YOLO detection failed - {e}")
            raise

    def _assess_image_quality(self, image_path: str) -> ImageQuality:
        """Assess dental image quality using computer vision metrics.

        Checks:
        - Blur (Laplacian variance)
        - Brightness (mean intensity)
        - Contrast (std dev)
        - Image size/resolution
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ImageQuality(
                    is_acceptable=False,
                    quality_score=0.0,
                    issues=["Cannot read image file"],
                    recommendations=["Check file format (JPG/PNG only)"],
                )

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]

            issues = []
            recommendations = []
            scores = []

            # 1. Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize
            scores.append(blur_score)

            if laplacian_var < 100:
                issues.append("Image is blurry")
                recommendations.append("Hold phone steady or use better lighting")
            elif laplacian_var < 200:
                issues.append("Image slightly out of focus")

            logger.debug(f"VisionAgent Quality: Blur variance={laplacian_var:.2f}")

            # 2. Brightness check
            mean_brightness = gray.mean()
            if mean_brightness < 50:
                brightness_score = mean_brightness / 50.0
                issues.append("Image too dark")
                recommendations.append("Increase lighting or use flash")
            elif mean_brightness > 200:
                brightness_score = (255 - mean_brightness) / 55.0
                issues.append("Image overexposed")
                recommendations.append("Reduce lighting or move away from light source")
            else:
                brightness_score = 1.0
            scores.append(brightness_score)

            logger.debug(f"VisionAgent Quality: Brightness={mean_brightness:.2f}")

            # 3. Contrast check
            contrast = gray.std()
            contrast_score = min(contrast / 60.0, 1.0)
            scores.append(contrast_score)

            if contrast < 30:
                issues.append("Low contrast")
                recommendations.append("Ensure good lighting on teeth area")

            logger.debug(f"VisionAgent Quality: Contrast std={contrast:.2f}")

            # 4. Resolution check
            min_dimension = min(height, width)
            if min_dimension < 100:
                resolution_score = 0.0
                issues.append("Resolution too low")
                recommendations.append("Image must be at least 100x100 pixels")
            elif min_dimension < 300:
                resolution_score = min_dimension / 300.0
                issues.append("Low resolution")
                recommendations.append("Use higher resolution or get closer")
            else:
                resolution_score = 1.0

            scores.append(resolution_score)

            # Overall quality score (weighted average)
            quality_score = (
                blur_score * 0.35 +
                brightness_score * 0.25 +
                contrast_score * 0.25 +
                resolution_score * 0.15
            )

            is_acceptable = quality_score >= 0.6 and len(issues) < 3

            logger.info(
                f"VisionAgent Quality: Overall={quality_score:.2f}, "
                f"Acceptable={is_acceptable}, Issues={len(issues)}"
            )

            return ImageQuality(
                is_acceptable=is_acceptable,
                quality_score=quality_score,
                issues=issues,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"VisionAgent: Quality assessment error - {e}")
            return ImageQuality(
                is_acceptable=True,  # Fail open - allow processing
                quality_score=0.5,
                issues=[],
                recommendations=[],
            )