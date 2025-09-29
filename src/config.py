# config.py
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

def load_config():
    """Load configuration settings."""
    load_dotenv()
    return {
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'cohere_api_key': os.getenv('COHERE_API_KEY'),
        'model_path': 'models/oral_detection_model.pt',  # Custom dental model
        'docs_path': 'docs/',
        'db_path': 'dental_chatbot.db',
        'confidence_threshold': 0.5,
        'max_file_size_mb': 5,
        'cache_size': 100,
        'classes': ['calculus', 'caries', 'gingivitis', 'hypodontia', 'tooth_discoloration', 'ulcer'],
        'class_colors': {
            'calculus': (255, 0, 0),
            'caries': (0, 255, 0),
            'gingivitis': (0, 0, 255),
            'hypodontia': (255, 255, 0),
            'tooth_discoloration': (255, 0, 255),
            'ulcer': (0, 255, 255)
        },
        'gemini_spatial_prompt': "Analyze dental image with detections {detections}. Describe spatial relations (e.g., upper/lower jaw, left/right side, specific tooth position like second molar), severity, relations to gums/tongue/other teeth. Focus on dental features; ignore non-dental objects. Provide precise marking insights."
    }

def setup_logging():
    """Setup logging with rotation and verbose level."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Env var: DEBUG/INFO/WARNING
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    logging.info(f"Logging setup with level: {log_level}")
