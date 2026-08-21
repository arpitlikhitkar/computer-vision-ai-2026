"""
Centralized Configuration Settings for Phase 5 — Household Face Recognition
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATABASE_DIR = os.path.join(OUTPUTS_DIR, "database")
ENROLLED_FACES_DIR = os.path.join(OUTPUTS_DIR, "enrolled_faces")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(ENROLLED_FACES_DIR, exist_ok=True)

# Pretrained Model Paths
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Database Path
DATABASE_PATH = os.path.join(DATABASE_DIR, "household_ai.db")

# Detection Thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.50
FACE_CONFIDENCE_THRESHOLD = 0.60

# Face Quality Thresholds
MIN_FACE_WIDTH = 40
MIN_FACE_HEIGHT = 40
BLUR_THRESHOLD = 60.0        # Laplacian Variance threshold
BRIGHTNESS_MIN = 35.0        # Minimum mean intensity
BRIGHTNESS_MAX = 225.0       # Maximum mean intensity

# Face Recognition Thresholds (SFace Cosine Similarity)
RECOGNITION_THRESHOLD = 0.65       # Similarity cutoff for KNOWN vs UNKNOWN
RECOGNITION_CONFIRMATION_COUNT = 3  # Frames required to confirm identity
RECOGNITION_INTERVAL_FRAMES = 2    # Frame sampling interval for CPU efficiency

# Enrollment Settings
ENROLLMENT_SAMPLE_COUNT = 10
MAX_EMBEDDINGS_PER_PERSON = 20

# System Settings
CAMERA_INDEX = 0
CAMERA_ID = "WEBCAM-HOUSEHOLD-01"
DEVICE = "cpu"  # Automatic fallback: 'cuda' if available else 'cpu'
DEBUG_MODE = True
