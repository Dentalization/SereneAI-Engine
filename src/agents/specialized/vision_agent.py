"""Vision Agent for dental image analysis using YOLO with LLM-based quality checks.

This agent:
- Validates image quality using Gemini Vision (LLM-driven, NO hardcoded thresholds)
- Runs YOLO detection on dental images
- Generates spatial insights using Gemini vision
- Provides actionable feedback on image quality
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.specialized.base_agent import BaseAgent
from src.agents.state_models import AgentState, DetectionResult
from src.tools.yolo_tool import detect_issues, validate_image, InvalidImageError
from src.utils.llm import get_gemini_chat

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
        super().__init__(name="VisionAgent")

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
        """Assess dental image quality using Gemini Vision API (LLM-driven).

        NO hardcoded thresholds - the LLM determines quality.
        """
        QUALITY_ASSESSMENT_PROMPT = """Assess this dental/oral image for quality and usability in AI analysis.

Evaluate the following aspects:
1. **Sharpness/Focus**: Is the image clear or blurry?
2. **Lighting**: Is lighting adequate (not too dark/bright)?
3. **Composition**: Are teeth/oral cavity visible and centered?
4. **Resolution**: Is detail sufficient for analysis?
5. **Obstructions**: Any fingers, objects blocking view?

Provide assessment in JSON format:
{{
  "is_acceptable": boolean,  // Can this image be used for dental AI analysis?
  "quality_score": 0.0-1.0,  // Overall quality rating
  "issues": ["list of specific problems found"],
  "recommendations": ["actionable suggestions to improve"]
}}

Be strict: Only mark as acceptable if image is CLEARLY usable for dental diagnosis.
Respond with ONLY valid JSON."""

        try:
            # Read image and encode as base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine MIME type
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
            }.get(ext, "image/jpeg")

            # Create vision LLM client
            llm = get_gemini_chat(
                model="gemini-2.0-flash-exp",  # Use vision-capable model
                temperature=0.0,
            )

            # Invoke with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": QUALITY_ASSESSMENT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{image_data}",
                    },
                ]
            )

            response = llm.invoke(
                [message],
                config={"response_format": {"type": "json_object"}},
            )

            # Parse LLM response
            import json
            import re

            raw = response.content.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw

            result = json.loads(json_str)

            quality = ImageQuality(
                is_acceptable=result.get("is_acceptable", False),
                quality_score=result.get("quality_score", 0.0),
                issues=result.get("issues", []),
                recommendations=result.get("recommendations", []),
            )

            logger.info(
                f"VisionAgent Quality (LLM): Acceptable={quality.is_acceptable}, "
                f"Score={quality.quality_score:.2f}, Issues={len(quality.issues)}"
            )

            return quality

        except json.JSONDecodeError as e:
            logger.error(f"VisionAgent: Failed to parse quality assessment JSON - {e}")
            from src.utils.exceptions import VisionError
            raise VisionError(
                message=f"Failed to parse image quality assessment: {str(e)}",
                user_action="The system could not assess image quality. Please try a different image."
            ) from e

        except Exception as e:
            logger.error(f"VisionAgent: Quality assessment error - {e}")
            from src.utils.exceptions import VisionError
            raise VisionError(
                message=f"Image quality assessment failed: {str(e)}",
                user_action="Could not analyze image quality. Please ensure the image is a valid JPG/PNG file."
            ) from e