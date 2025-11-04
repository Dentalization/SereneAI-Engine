"""
YOLO detector wrapper for dental pathology detection.
Modernized implementation with proper error handling.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from src.config import get_settings


class YOLODetector:
    """
    YOLO-based dental pathology detector.

    Detects:
    - caries (cavities)
    - gingivitis (gum inflammation)
    - calculus (tartar)
    - hypodontia (missing teeth)
    - tooth_discoloration
    - ulcer (oral sores)

    Example:
        >>> detector = YOLODetector()
        >>> results = detector.detect("path/to/image.jpg")
        >>> for det in results:
        ...     print(f"{det['class']}: {det['confidence']:.2f}")
    """

    def __init__(self):
        """Initialize YOLO detector with model."""
        config = get_settings()
        self.config = config
        self.model = self._load_model()

    @lru_cache(maxsize=1)
    def _load_model(self):
        """
        Load YOLO model (cached).

        Returns:
            Loaded YOLO model instance
        """
        try:
            from ultralytics import YOLO

            model_path = Path(self.config.yolo_model_path)

            if model_path.exists():
                # Load custom dental model
                model = YOLO(str(model_path))
            else:
                # Fallback to YOLOv11 nano
                print(
                    f"Warning: Custom model not found at {model_path}. "
                    f"Using YOLOv11n as fallback."
                )
                model = YOLO("yolov11n.pt")

            # Set device
            model.to(self.config.yolo_device)

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(
        self,
        image_path: str,
        confidence_threshold: float | None = None,
    ) -> List[dict]:
        """
        Detect dental pathologies in image.

        Args:
            image_path: Path to dental image
            confidence_threshold: Minimum confidence (default: from config)

        Returns:
            List of detections with class, confidence, bbox

        Example:
            >>> detections = detector.detect("tooth.jpg", confidence_threshold=0.5)
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.yolo_confidence_threshold

        # Validate image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=confidence_threshold,
            device=self.config.yolo_device,
            verbose=False,
        )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                detections.append(
                    {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    }
                )

        return detections
