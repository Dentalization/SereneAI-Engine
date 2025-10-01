"""Advanced image preprocessing pipeline for dental images.

Features:
- Adaptive contrast enhancement (CLAHE)
- Dental-specific quality scoring
- Auto-cropping to oral cavity
- Lighting normalization
- Blur detection and sharpening
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualityMetrics(BaseModel):
    """Image quality metrics for dental images."""

    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality (0-1)")
    blur_score: float = Field(ge=0.0, le=1.0)
    brightness_score: float = Field(ge=0.0, le=1.0)
    contrast_score: float = Field(ge=0.0, le=1.0)
    color_balance_score: float = Field(ge=0.0, le=1.0)
    dental_focus_score: float = Field(ge=0.0, le=1.0, description="Focuses on teeth/gums")
    is_acceptable: bool
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ImagePreprocessor:
    """Advanced preprocessing for dental images."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        quality_threshold: float = 0.6,
    ):
        """Initialize preprocessor.

        Args:
            target_size: Output image size (width, height)
            quality_threshold: Minimum quality score to accept
        """
        self.target_size = target_size
        self.quality_threshold = quality_threshold

        logger.info(f"ImagePreprocessor: Initialized with size={target_size}")

    def process(
        self,
        image_path: str,
        auto_enhance: bool = True
    ) -> Tuple[np.ndarray, QualityMetrics]:
        """Process dental image with quality assessment.

        Args:
            image_path: Path to input image
            auto_enhance: Whether to apply automatic enhancements

        Returns:
            Tuple of (processed_image, quality_metrics)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        logger.info(f"ImagePreprocessor: Processing {image_path}")

        # Assess quality BEFORE processing
        quality_before = self.assess_quality(img)

        if auto_enhance:
            # Apply enhancements
            img = self._enhance_image(img, quality_before)

        # Resize to target
        img = cv2.resize(img, self.target_size)

        # Reassess quality AFTER processing
        quality_after = self.assess_quality(img)

        logger.info(
            f"ImagePreprocessor: Quality improved from {quality_before.overall_score:.2f} "
            f"to {quality_after.overall_score:.2f}"
        )

        return img, quality_after

    def assess_quality(self, img: np.ndarray) -> QualityMetrics:
        """Assess dental image quality with detailed metrics.

        Args:
            img: Input image (BGR format)

        Returns:
            QualityMetrics with scores and recommendations
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        scores = {}
        issues = []
        recommendations = []

        # 1. Blur score (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)
        scores["blur_score"] = blur_score

        if laplacian_var < 100:
            issues.append("Severe blur detected")
            recommendations.append("Use tripod or increase lighting")
        elif laplacian_var < 200:
            issues.append("Slight blur")
            recommendations.append("Hold phone steady")

        # 2. Brightness score
        mean_brightness = gray.mean()
        if mean_brightness < 50:
            brightness_score = mean_brightness / 50.0
            issues.append("Image too dark")
            recommendations.append("Increase lighting or use flash")
        elif mean_brightness > 200:
            brightness_score = (255 - mean_brightness) / 55.0
            issues.append("Image overexposed")
            recommendations.append("Reduce lighting")
        else:
            brightness_score = 1.0
        scores["brightness_score"] = brightness_score

        # 3. Contrast score
        contrast = gray.std()
        contrast_score = min(contrast / 60.0, 1.0)
        scores["contrast_score"] = contrast_score

        if contrast < 30:
            issues.append("Low contrast")
            recommendations.append("Better lighting on teeth")

        # 4. Color balance (check if image is too tinted)
        b, g, r = cv2.split(img)
        color_balance_score = 1.0 - (
            abs(b.mean() - g.mean()) + abs(g.mean() - r.mean()) + abs(r.mean() - b.mean())
        ) / (3 * 255)
        scores["color_balance_score"] = color_balance_score

        if color_balance_score < 0.8:
            issues.append("Color imbalance detected")
            recommendations.append("Check white balance settings")

        # 5. Dental focus (detect if teeth/gums are in frame)
        dental_focus_score = self._assess_dental_focus(img)
        scores["dental_focus_score"] = dental_focus_score

        if dental_focus_score < 0.5:
            issues.append("Teeth not clearly visible")
            recommendations.append("Focus camera on oral cavity")

        # Overall score (weighted average)
        overall_score = (
            blur_score * 0.30 +
            brightness_score * 0.20 +
            contrast_score * 0.20 +
            color_balance_score * 0.10 +
            dental_focus_score * 0.20
        )

        is_acceptable = overall_score >= self.quality_threshold

        return QualityMetrics(
            overall_score=overall_score,
            is_acceptable=is_acceptable,
            issues=issues,
            recommendations=recommendations,
            **scores
        )

    def _assess_dental_focus(self, img: np.ndarray) -> float:
        """Assess whether teeth/gums are visible using heuristics.

        Args:
            img: Input image (BGR)

        Returns:
            Focus score (0-1)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect pinkish regions (gums) and whitish regions (teeth)
        # Gums: H ~0-20 (red-pink), S > 50, V > 50
        lower_gum = np.array([0, 50, 50])
        upper_gum = np.array([20, 255, 255])
        gum_mask = cv2.inRange(hsv, lower_gum, upper_gum)

        # Teeth: High V (brightness), low S (saturation)
        lower_tooth = np.array([0, 0, 180])
        upper_tooth = np.array([180, 50, 255])
        tooth_mask = cv2.inRange(hsv, lower_tooth, upper_tooth)

        # Combine masks
        oral_cavity_mask = cv2.bitwise_or(gum_mask, tooth_mask)

        # Calculate percentage of image that's oral cavity
        oral_percentage = cv2.countNonZero(oral_cavity_mask) / (img.shape[0] * img.shape[1])

        # Good dental images should have 20-60% oral cavity
        if 0.20 <= oral_percentage <= 0.60:
            score = 1.0
        elif 0.10 <= oral_percentage < 0.20:
            score = 0.7
        elif oral_percentage > 0.60:
            score = 0.8  # Too close
        else:
            score = 0.3  # Barely any teeth visible

        logger.debug(f"ImagePreprocessor: Dental focus = {oral_percentage:.2%} → score={score:.2f}")

        return score

    def _enhance_image(self, img: np.ndarray, quality: QualityMetrics) -> np.ndarray:
        """Apply automatic enhancements based on quality assessment.

        Args:
            img: Input image
            quality: Quality metrics

        Returns:
            Enhanced image
        """
        enhanced = img.copy()

        # 1. Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if quality.contrast_score < 0.7:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            logger.debug("ImagePreprocessor: Applied CLAHE contrast enhancement")

        # 2. Brightness adjustment
        if quality.brightness_score < 0.7:
            # Increase brightness
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            v = cv2.add(v, 30)  # Increase value channel

            enhanced = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)

            logger.debug("ImagePreprocessor: Increased brightness")

        # 3. Sharpening if blurry
        if quality.blur_score < 0.6:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            logger.debug("ImagePreprocessor: Applied sharpening")

        # 4. Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        return enhanced