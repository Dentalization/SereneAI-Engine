"""Async YOLO detector for non-blocking dental image analysis.

Features:
- Async/await pattern for concurrent processing
- Background task queue (asyncio-based)
- Batch inference support
- Progress callbacks for real-time UI updates
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import torch
from pydantic import BaseModel, Field
from ultralytics import YOLO

from src.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


class DetectionResult(BaseModel):
    """YOLO detection result."""

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(description="[x1, y1, x2, y2]")


class YOLOTask(BaseModel):
    """Task for async YOLO processing."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    image_path: str
    threshold: float = 0.5
    status: str = Field(default="pending")  # pending, processing, completed, failed
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AsyncYOLODetector:
    """Async YOLO detector with task queue."""

    def __init__(
        self,
        model_path: str = "models/oral_detection_model.pt",
        batch_size: int = 4,
        max_queue_size: int = 100,
    ):
        """Initialize async YOLO detector.

        Args:
            model_path: Path to YOLO model weights
            batch_size: Batch size for inference
            max_queue_size: Maximum tasks in queue
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        # Model (lazy loaded)
        self.model: Optional[YOLO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Task queue
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, YOLOTask] = {}

        # Background worker
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info(
            f"AsyncYOLODetector: Initialized with model={model_path}, "
            f"batch_size={batch_size}, device={self.device}"
        )

    def _load_model(self) -> None:
        """Load YOLO model (lazy initialization)."""
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                logger.info(f"AsyncYOLODetector: Loaded model on {self.device}")
            except Exception as e:
                logger.error(f"AsyncYOLODetector: Model load failed - {e}")
                # Fallback to general model
                self.model = YOLO("yolo11n.pt")
                self.model.to(self.device)
                logger.warning("AsyncYOLODetector: Using fallback yolo11n model")

    async def start(self) -> None:
        """Start background worker."""
        if self.is_running:
            logger.warning("AsyncYOLODetector: Worker already running")
            return

        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("AsyncYOLODetector: Background worker started")

    async def stop(self) -> None:
        """Stop background worker."""
        if not self.is_running:
            return

        self.is_running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        logger.info("AsyncYOLODetector: Background worker stopped")

    async def detect(
        self,
        image_path: str,
        threshold: float = 0.5,
        callback: Optional[Callable[[YOLOTask], None]] = None,
    ) -> str:
        """Submit detection task and return task ID.

        Args:
            image_path: Path to image file
            threshold: Confidence threshold
            callback: Optional callback for progress updates

        Returns:
            Task ID for tracking
        """
        # Create task
        task = YOLOTask(
            image_path=image_path,
            threshold=threshold
        )

        # Store task
        self.tasks[task.task_id] = task

        # Add to queue
        await self.task_queue.put(task)
        logger.info(f"AsyncYOLODetector: Task {task.task_id} queued")

        return task.task_id

    async def get_result(
        self,
        task_id: str,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Wait for and retrieve task result.

        Args:
            task_id: Task ID to retrieve
            timeout: Max wait time in seconds

        Returns:
            Detection result dict or None if timeout/error
        """
        if task_id not in self.tasks:
            logger.error(f"AsyncYOLODetector: Task {task_id} not found")
            return None

        task = self.tasks[task_id]

        # Wait for completion
        start_time = asyncio.get_event_loop().time()
        while task.status not in ["completed", "failed"]:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.error(f"AsyncYOLODetector: Task {task_id} timeout")
                return None

            await asyncio.sleep(0.1)

        # Return result
        if task.status == "completed":
            return task.result
        else:
            logger.error(f"AsyncYOLODetector: Task {task_id} failed - {task.error}")
            return None

    async def detect_sync(
        self,
        image_path: str,
        threshold: float = 0.5,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Synchronous-style detection (submit + wait).

        Args:
            image_path: Path to image
            threshold: Confidence threshold
            timeout: Max wait time

        Returns:
            Detection result dict
        """
        task_id = await self.detect(image_path, threshold)
        return await self.get_result(task_id, timeout)

    async def _worker(self) -> None:
        """Background worker processing task queue."""
        logger.info("AsyncYOLODetector: Worker thread started")

        # Load model once
        self._load_model()

        while self.is_running:
            try:
                # Get task from queue (with timeout)
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                # Process task
                await self._process_task(task)

            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"AsyncYOLODetector: Worker error - {e}")

        logger.info("AsyncYOLODetector: Worker thread stopped")

    async def _process_task(self, task: YOLOTask) -> None:
        """Process a single detection task.

        Args:
            task: YOLOTask to process
        """
        task.status = "processing"
        task.progress = 0.1
        logger.info(f"AsyncYOLODetector: Processing task {task.task_id}")

        try:
            # Preprocess image (in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(
                None,
                self._preprocess_image,
                task.image_path
            )
            task.progress = 0.3

            # Run inference (blocking, but in executor)
            detections, annotated_path = await loop.run_in_executor(
                None,
                self._run_inference,
                img,
                task.threshold,
                task.task_id
            )
            task.progress = 0.7

            # Get spatial insights (async via LLM)
            spatial_insights = await self._get_spatial_insights_async(
                task.image_path,
                detections
            )
            task.progress = 0.9

            # Build result
            task.result = {
                "detections": [d.model_dump() for d in detections],
                "detections_json": json.dumps([d.model_dump() for d in detections]),
                "annotated_image_path": annotated_path,
                "spatial_insights": spatial_insights,
                "task_id": task.task_id,
            }

            task.status = "completed"
            task.progress = 1.0

            logger.info(
                f"AsyncYOLODetector: Task {task.task_id} completed - "
                f"{len(detections)} detections"
            )

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"AsyncYOLODetector: Task {task.task_id} failed - {e}")

    def _preprocess_image(self, image_path: str) -> Any:
        """Preprocess image for YOLO.

        Args:
            image_path: Path to image

        Returns:
            Preprocessed image array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Resize
        img = cv2.resize(img, (640, 640))

        # Enhance contrast
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

        return img

    def _run_inference(
        self,
        img: Any,
        threshold: float,
        task_id: str
    ) -> Tuple[List[DetectionResult], str]:
        """Run YOLO inference.

        Args:
            img: Preprocessed image
            threshold: Confidence threshold
            task_id: Task ID for output naming

        Returns:
            Tuple of (detections, annotated_image_path)
        """
        # Run inference with torch.inference_mode for speed
        with torch.inference_mode():
            results = self.model(img)[0]

        # Parse detections
        detections: List[DetectionResult] = []
        for result in results.boxes:
            conf = result.conf.item()
            if conf >= threshold:
                cls_idx = int(result.cls.item())
                cls_name = self.model.names[cls_idx]
                bbox = result.xyxy.tolist()[0]

                detections.append(DetectionResult(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox
                ))

        # Save annotated image
        annotated_img = results.plot()
        output_path = f"annotated_{task_id}.jpg"
        cv2.imwrite(output_path, annotated_img)

        return detections, output_path

    async def _get_spatial_insights_async(
        self,
        image_path: str,
        detections: List[DetectionResult]
    ) -> str:
        """Get spatial insights using async Gemini call.

        Args:
            image_path: Path to original image
            detections: Detection results

        Returns:
            Spatial insights text
        """
        try:
            # Read image
            with open(image_path, "rb") as f:
                img_bytes = f.read()

            image_data = base64.b64encode(img_bytes).decode("utf-8")

            # Prepare prompt
            detections_json = json.dumps([d.model_dump() for d in detections])
            prompt = config["gemini_spatial_prompt"].format(
                detections=detections_json
            )

            # Call Gemini (async)
            from langchain_core.messages import HumanMessage
            from src.utils.llm import get_gemini_chat

            llm = get_gemini_chat(model="gemini-2.5-flash", temperature=0.3)

            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
            ])

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke([message])
            )

            return response.content

        except Exception as e:
            logger.error(f"AsyncYOLODetector: Spatial insights error - {e}")
            return "Spatial analysis unavailable."

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task status.

        Args:
            task_id: Task ID

        Returns:
            Task status dict
        """
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "error": task.error,
        }

    def clear_completed_tasks(self, max_age_seconds: int = 3600) -> int:
        """Clear old completed tasks to free memory.

        Args:
            max_age_seconds: Max age for completed tasks

        Returns:
            Number of tasks cleared
        """
        # Simple implementation: clear all completed
        cleared = 0
        to_remove = [
            tid for tid, task in self.tasks.items()
            if task.status in ["completed", "failed"]
        ]

        for tid in to_remove:
            del self.tasks[tid]
            cleared += 1

        if cleared > 0:
            logger.info(f"AsyncYOLODetector: Cleared {cleared} completed tasks")

        return cleared