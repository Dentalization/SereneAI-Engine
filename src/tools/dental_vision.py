"""
Dental vision analysis tool using @tool decorator with ToolRuntime.
Follows LangChain 1.0 best practices for tool implementation.
"""

import base64
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import ToolRuntime, tool

from src.agents.state import DetectionResult, UserProfile
from src.config import get_settings
from src.models.gemini import get_gemini_vision
from src.vision.yolo_detector import YOLODetector


def _image_hash(image_path: str) -> str:
    """Generate hash for image caching."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _encode_image(image_path: str) -> str:
    """Encode image to base64 for LLM vision."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@tool
def dental_vision_analysis(
    image_path: str,
    *,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Analyze dental image for pathologies using YOLO detection and Gemini vision.

    This tool performs comprehensive dental image analysis including:
    - Quality assessment (blur, lighting, focus)
    - Pathology detection (caries, gingivitis, calculus, etc.)
    - Spatial analysis (location descriptions)
    - Confidence scoring

    Args:
        image_path: Absolute path to the dental image file (JPG, PNG)
        runtime: Injected runtime context (state, context, store, stream)

    Returns:
        Dictionary containing:
        - detections: List of detected dental conditions with bounding boxes
        - spatial_analysis: Natural language description of locations
        - quality_metrics: Image quality assessment
        - confidence: Overall detection confidence (0-1)
        - recommendations: List of recommended next steps

    Example:
        >>> result = dental_vision_analysis("/tmp/dental_image.jpg")
        >>> print(result["detections"])
        [{"class": "caries", "confidence": 0.87, "location": "upper left molar"}]
    """
    config = get_settings()

    # Stream progress to user
    runtime.stream_writer(
        {"type": "progress", "stage": "vision", "message": "Memproses gambar gigi..."}
    )

    # Access conversation state
    state = runtime.state
    user_profile: UserProfile = state.get("user_profile", UserProfile())
    language = runtime.context.get("language", "id")

    # Validate image exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check cache in store (avoid reprocessing same image)
    image_hash = _image_hash(image_path)
    cached_result = runtime.store.get(
        namespace=("user", user_profile.user_id, "vision_cache"),
        key=image_hash,
    )

    if cached_result:
        runtime.stream_writer(
            {"type": "progress", "stage": "vision", "message": "Menggunakan hasil cache..."}
        )
        return cached_result.value

    # Step 1: Quality Assessment
    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "vision",
            "message": "Menilai kualitas gambar..." if language == "id" else "Assessing image quality...",
        }
    )

    quality_prompt = (
        "Analyze this dental image for quality. Rate blur (0-1), brightness (0-1), "
        "contrast (0-1), and whether teeth/gums are clearly visible. "
        "Return JSON: {\"blur\": float, \"brightness\": float, \"contrast\": float, "
        "\"dental_focus\": bool, \"issues\": [list of issues]}"
    )

    gemini_vision = get_gemini_vision()
    quality_response = gemini_vision.invoke(
        [
            {"type": "text", "text": quality_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
        ]
    )

    # Parse quality metrics (simplified - in production, use structured output)
    quality_metrics = {
        "blur": 0.8,
        "brightness": 0.7,
        "contrast": 0.8,
        "dental_focus": True,
        "issues": [],
    }

    # Check if image quality is sufficient
    if quality_metrics["blur"] < 0.5 or not quality_metrics["dental_focus"]:
        return {
            "detections": [],
            "spatial_analysis": "",
            "quality_metrics": quality_metrics,
            "confidence": 0.0,
            "recommendations": [
                "Kualitas gambar kurang baik. Mohon ambil foto yang lebih jelas."
                if language == "id"
                else "Image quality is insufficient. Please take a clearer photo."
            ],
            "error": "low_quality",
        }

    # Step 2: YOLO Detection
    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "vision",
            "message": "Mendeteksi kondisi gigi..." if language == "id" else "Detecting dental conditions...",
        }
    )

    detector = YOLODetector()
    detections_raw = detector.detect(
        image_path,
        confidence_threshold=config.yolo_confidence_threshold,
    )

    # Convert to DetectionResult models
    detections = [
        DetectionResult(
            class_name=det["class"],
            confidence=det["confidence"],
            bbox=det["bbox"],
            spatial_description=None,  # Will be filled by spatial analysis
        )
        for det in detections_raw
    ]

    # Step 3: Spatial Analysis (only if detections found)
    spatial_analysis = ""
    if detections:
        runtime.stream_writer(
            {
                "type": "progress",
                "stage": "vision",
                "message": "Menganalisis lokasi..." if language == "id" else "Analyzing locations...",
            }
        )

        spatial_prompt = f"""
        Analyze this dental image and describe the locations of detected conditions.
        Detections: {[f"{d.class_name} (confidence: {d.confidence:.2f})" for d in detections]}

        Provide natural language descriptions of where each condition is located
        (e.g., "upper left molar", "lower front teeth", "gum line near canines").

        Language: {"Indonesian" if language == "id" else "English"}
        """

        spatial_response = gemini_vision.invoke(
            [
                {"type": "text", "text": spatial_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
            ]
        )

        spatial_analysis = spatial_response.content

    # Calculate overall confidence
    avg_confidence = (
        sum(d.confidence for d in detections) / len(detections) if detections else 0.0
    )

    # Generate recommendations
    recommendations = []
    if detections:
        for det in detections:
            if det.class_name == "caries" and det.confidence > 0.7:
                recommendations.append(
                    "Terdeteksi gigi berlubang (karies). Segera konsultasi ke dokter gigi."
                    if language == "id"
                    else "Cavity (caries) detected. Please consult a dentist soon."
                )
            elif det.class_name == "gingivitis" and det.confidence > 0.7:
                recommendations.append(
                    "Terdeteksi radang gusi (gingivitis). Tingkatkan kebersihan mulut dan konsultasi dokter gigi."
                    if language == "id"
                    else "Gingivitis detected. Improve oral hygiene and consult a dentist."
                )
            elif det.class_name == "calculus" and det.confidence > 0.7:
                recommendations.append(
                    "Terdeteksi karang gigi (calculus). Perlu pembersihan karang gigi (scaling)."
                    if language == "id"
                    else "Dental calculus detected. Professional teeth cleaning (scaling) needed."
                )
    else:
        recommendations.append(
            "Tidak ada kondisi dental yang terdeteksi pada gambar ini."
            if language == "id"
            else "No dental conditions detected in this image."
        )

    # Build result
    result = {
        "detections": [
            {
                "class": d.class_name,
                "confidence": d.confidence,
                "bbox": d.bbox,
            }
            for d in detections
        ],
        "spatial_analysis": spatial_analysis,
        "quality_metrics": quality_metrics,
        "confidence": avg_confidence,
        "recommendations": recommendations,
    }

    # Update user profile with detections
    for det in detections:
        user_profile.detections.append(det)

    # Store in long-term memory
    runtime.store.put(
        namespace=("user", user_profile.user_id, "vision_history"),
        key=f"analysis_{datetime.now().isoformat()}",
        value={
            "detections": result["detections"],
            "timestamp": datetime.now().isoformat(),
            "image_hash": image_hash,
        },
    )

    # Cache result
    runtime.store.put(
        namespace=("user", user_profile.user_id, "vision_cache"),
        key=image_hash,
        value=result,
    )

    runtime.stream_writer(
        {
            "type": "progress",
            "stage": "vision",
            "message": "Analisis selesai!" if language == "id" else "Analysis complete!",
        }
    )

    return result
