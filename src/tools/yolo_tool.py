"""YOLO-based detection and spatial analysis utilities.

Functionality unchanged; includes docstrings, typing, and small structure
improvements (e.g., lazy initialization and shared LLM utility).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Tuple

import cv2
import torch
from langchain_core.messages import HumanMessage
from ultralytics import YOLO

from src.config import load_config
from src.utils.llm import get_gemini_chat
from cachetools import LRUCache
import hashlib

config = load_config()
_model = None  # Lazily initialized YOLO model
_gemini_vision = None  # Lazily initialized Gemini client with vision support
_spatial_cache: LRUCache = LRUCache(maxsize=64)


def _get_gemini_vision():
    """Return a cached Gemini client configured for vision calls."""
    global _gemini_vision
    if _gemini_vision is None:
        _gemini_vision = get_gemini_chat(model="gemini-2.5-flash", temperature=0.3)
    return _gemini_vision


def load_yolo_model():
    """Load custom dental YOLO model with GPU if available."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _model = YOLO(config["model_path"])  # Use custom dental model
            _model.to(device)
            logging.info(f"Custom YOLO dental model loaded on {device}")
        except Exception as e:  # noqa: BLE001 - fallback to general model
            logging.error(f"Model load error: {str(e)}. Falling back to general model.")
            _model = YOLO("yolo11n.pt")  # Fallback if custom fails
            _model.to(device)
    return _model


class InvalidImageError(Exception):
    """Raised when an invalid image is provided."""


def preprocess_image(image_path: str):
    """Preprocess image: resize and enhance contrast."""
    img = cv2.imread(image_path)
    if img is None:
        raise InvalidImageError("Invalid image or not found.")
    img = cv2.resize(img, (640, 640))
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return img


def validate_image(image_path: str) -> bool:
    """Validate image quality and format."""
    if not os.path.exists(image_path):
        raise InvalidImageError("File not found.")
    if os.path.getsize(image_path) > config["max_file_size_mb"] * 1024 * 1024:
        raise InvalidImageError("File size >5MB.")
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 100 or img.shape[1] < 100:
        raise InvalidImageError("Low resolution.")
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise InvalidImageError("Format JPG/PNG only.")
    return True


def get_spatial_insights(image_path: str, detections: List[Dict[str, Any]]) -> str:
    """Use Gemini for enhanced spatial analysis on YOLO outputs via LangChain."""
    try:
        # Cache key: content hash of image + detections
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()
        key = (hashlib.sha1(img_bytes).hexdigest(), json.dumps(detections, sort_keys=True))
        if key in _spatial_cache:
            return _spatial_cache[key]

        # Read and encode image to base64
        image_data = base64.b64encode(img_bytes).decode("utf-8")

        # Format prompt with detections
        prompt = config["gemini_spatial_prompt"].format(
            detections=json.dumps(detections)
        )

        # Create message with image and text using LangChain format
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )

        # Invoke model with vision capabilities
        response = _get_gemini_vision().invoke([message])
        _spatial_cache[key] = response.content
        return response.content
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        logging.error(f"Gemini spatial error: {str(e)}")
        return "Spatial analysis unavailable. Focus on dental features only."


def detect_issues(image_path: str, threshold: float | None = None) -> Tuple[str, str, str]:
    """Detect dental issues with custom YOLO and Gemini spatial.

    Returns:
        tuple: (detections_json, annotated_image_path, spatial_insights)
    """
    try:
        validate_image(image_path)
        img = preprocess_image(image_path)
        model = load_yolo_model()
        # Inference mode for faster eval
        with torch.inference_mode():
            results = model(img)[0]
        detections: List[Dict[str, Any]] = []
        for result in results.boxes:
            conf = result.conf.item()
            if conf >= (threshold or config["confidence_threshold"]):
                cls_idx = int(result.cls.item())
                cls = model.names[cls_idx]
                box = result.xyxy.tolist()[0]
                detections.append({"class": cls, "confidence": conf, "bbox": box})
        if not detections:
            logging.warning("No dental detections; possible misconfiguration.")
        spatial_insights = get_spatial_insights(image_path, detections)
        annotated_img = results.plot()
        unique_id = uuid.uuid4().hex
        output_path = f"annotated_{unique_id}.jpg"
        cv2.imwrite(output_path, annotated_img)
        logging.info(f"Annotated image saved as {output_path}")
        return json.dumps(detections), output_path, spatial_insights
    except Exception as e:
        logging.error(f"YOLO error: {str(e)}")
        raise
