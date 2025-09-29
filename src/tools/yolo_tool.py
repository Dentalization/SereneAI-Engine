# yolo_tool.py
import cv2
from ultralytics import YOLO
import torch
import json
import os
import uuid  # Added for unique filenames
import base64
from src.config import load_config
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

config = load_config()
model = None

# Initialize LangChain ChatGoogleGenerativeAI with vision support
gemini_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=config['gemini_api_key'],
    temperature=0.3
)

def load_yolo_model():
    """Load custom dental YOLO model with GPU if available."""
    global model
    if model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = YOLO(config['model_path'])  # Use custom dental model
            model.to(device)
            logging.info(f"Custom YOLO dental model loaded on {device}")
        except Exception as e:
            logging.error(f"Model load error: {str(e)}. Falling back to general model.")
            model = YOLO('yolo11n.pt')  # Fallback if custom fails
            model.to(device)
    return model

class InvalidImageError(Exception):
    pass

def preprocess_image(image_path):
    """Preprocess image: resize and enhance contrast."""
    img = cv2.imread(image_path)
    if img is None:
        raise InvalidImageError("Invalid image or not found.")
    img = cv2.resize(img, (640, 640))
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return img

def validate_image(image_path):
    """Validate image quality and format."""
    if not os.path.exists(image_path):
        raise InvalidImageError("File not found.")
    if os.path.getsize(image_path) > config['max_file_size_mb'] * 1024 * 1024:
        raise InvalidImageError("File size >5MB.")
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 100 or img.shape[1] < 100:
        raise InvalidImageError("Low resolution.")
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise InvalidImageError("Format JPG/PNG only.")
    return True

def get_spatial_insights(image_path, detections):
    """Use Gemini for enhanced spatial analysis on YOLO outputs via LangChain."""
    try:
        # Read and encode image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Format prompt with detections
        prompt = config['gemini_spatial_prompt'].format(detections=json.dumps(detections))
        
        # Create message with image and text using LangChain format
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        )
        
        # Invoke model with vision capabilities
        response = gemini_model.invoke([message])
        return response.content
    except Exception as e:
        logging.error(f"Gemini spatial error: {str(e)}")
        return "Spatial analysis unavailable. Focus on dental features only."

def detect_issues(image_path, threshold=None):
    """Detect dental issues with custom YOLO and Gemini spatial."""
    try:
        validate_image(image_path)
        img = preprocess_image(image_path)
        model = load_yolo_model()
        results = model(img)[0]
        detections = []
        for result in results.boxes:
            conf = result.conf.item()
            if conf >= (threshold or config['confidence_threshold']):
                cls_idx = int(result.cls.item())
                cls = model.names[cls_idx]
                box = result.xyxy.tolist()[0]
                detections.append({'class': cls, 'confidence': conf, 'bbox': box})
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