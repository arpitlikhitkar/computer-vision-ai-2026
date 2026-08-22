"""
Central Config & Settings Manager with JSON persistence for Household AI Desktop Application
Includes Phase 6 Feature Flags, Multi-Class Thresholds & Audio Alarm Toggle
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
PEOPLE_DIR = os.path.join(DATA_DIR, "people")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown")
EVENTS_DIR = os.path.join(DATA_DIR, "events")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

CONFIG_FILE_PATH = os.path.join(DATA_DIR, "app_settings.json")


FEATURE_FLAGS = {
    'multiclass_detection': True,   # Phase 6.1
    'multiclass_tracking': True,    # Phase 6.2
    'enhanced_ui': True,            # Phase 6.3
    'model_management': True,       # Phase 6.4
    'pose_estimation': False,       # Phase 6.5
    'hand_keypoints': False,        # Phase 6.6
    'relationship_engine': False,   # Phase 6.7
    'temporal_consistency': False,  # Phase 6.8
    'event_system': False,          # Phase 6.9
}


class AppConfig:
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        self.MODELS_DIR = MODELS_DIR
        self.DATA_DIR = DATA_DIR
        self.DATABASE_DIR = DATABASE_DIR
        self.PEOPLE_DIR = PEOPLE_DIR
        self.UNKNOWN_DIR = UNKNOWN_DIR
        self.EVENTS_DIR = EVENTS_DIR
        self.RECORDINGS_DIR = RECORDINGS_DIR

        self.FEATURE_FLAGS = FEATURE_FLAGS.copy()

        self.yolo_model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
        self.yolo_pose_model_path = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
        self.yunet_model_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
        self.sface_model_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
        self.database_path = os.path.join(DATABASE_DIR, "household_ai_pyside.db")

        self.camera_index = 0
        self.camera_width = 640
        self.camera_height = 480

        self.person_conf_threshold = 0.50
        self.multiclass_conf_threshold = 0.50
        self.nms_iou_threshold = 0.45
        self.face_conf_threshold = 0.60
        self.recognition_threshold = 0.65
        self.min_face_width = 40
        self.min_face_height = 40
        self.blur_threshold = 60.0
        self.enrollment_sample_count = 10
        self.track_buffer_frames = 30

        self.enable_audio_alarm = True  # Laptop Speaker Siren Alert Toggle

        self.load_from_json()

    def load_from_json(self):
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r") as f:
                    data = json.load(f)
                    self.camera_index = data.get("camera_index", self.camera_index)
                    self.person_conf_threshold = data.get("person_conf_threshold", self.person_conf_threshold)
                    self.multiclass_conf_threshold = data.get("multiclass_conf_threshold", self.multiclass_conf_threshold)
                    self.face_conf_threshold = data.get("face_conf_threshold", self.face_conf_threshold)
                    self.recognition_threshold = data.get("recognition_threshold", self.recognition_threshold)
                    self.enable_audio_alarm = data.get("enable_audio_alarm", self.enable_audio_alarm)
                    if "feature_flags" in data:
                        self.FEATURE_FLAGS.update(data["feature_flags"])
            except Exception as e:
                print(f"[CONFIG] Error loading settings json: {e}")

    def save_to_json(self):
        data = {
            "camera_index": self.camera_index,
            "person_conf_threshold": self.person_conf_threshold,
            "multiclass_conf_threshold": self.multiclass_conf_threshold,
            "face_conf_threshold": self.face_conf_threshold,
            "recognition_threshold": self.recognition_threshold,
            "enable_audio_alarm": self.enable_audio_alarm,
            "feature_flags": self.FEATURE_FLAGS
        }
        try:
            with open(CONFIG_FILE_PATH, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[CONFIG] Error saving settings json: {e}")


config = AppConfig()
