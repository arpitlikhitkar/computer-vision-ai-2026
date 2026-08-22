"""
AI Model Registry & Dependency Manager (Phase 6.4)

Tracks model load status, memory/VRAM footprint, dependency chains, and system capability matrix.
Enforces dependency rules:
- Pose Estimator requires YOLOv8n
- Relationship Engine requires YOLOv8n + Pose Estimator
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import torch
from app.config.settings import config


@dataclass
class AIModel:
    model_id: str
    name: str
    status: str       # 'LOADED', 'UNLOADED', 'ERROR'
    type: str         # 'Detection', 'Embedding', 'Keypoints', 'Logic'
    size_mb: float
    vram_gb: float
    dependencies: List[str]


class ModelRegistry:
    """
    Central AI Model Management & Dependency Engine.
    """
    def __init__(self):
        self.models: Dict[str, AIModel] = {
            "yolov8n": AIModel(
                model_id="yolov8n",
                name="YOLOv8n (Multi-Class)",
                status="LOADED",
                type="Detection",
                size_mb=6.2,
                vram_gb=0.8,
                dependencies=[]
            ),
            "yunet": AIModel(
                model_id="yunet",
                name="YuNet Face Detector",
                status="LOADED",
                type="Detection",
                size_mb=0.4,
                vram_gb=0.2,
                dependencies=[]
            ),
            "sface": AIModel(
                model_id="sface",
                name="SFace Face Embedder",
                status="LOADED",
                type="Embedding",
                size_mb=37.0,
                vram_gb=0.5,
                dependencies=["yunet"]
            ),
            "osnet": AIModel(
                model_id="osnet",
                name="OSNet Body Re-ID Embedder",
                status="LOADED",
                type="Embedding",
                size_mb=5.4,
                vram_gb=0.4,
                dependencies=["yolov8n"]
            ),
            "yolo_pose": AIModel(
                model_id="yolo_pose",
                name="YOLOv8n-Pose Estimator",
                status="UNLOADED",
                type="Keypoints",
                size_mb=6.5,
                vram_gb=0.9,
                dependencies=["yolov8n"]
            ),
            "relationship_engine": AIModel(
                model_id="relationship_engine",
                name="Spatial Relationship Engine",
                status="UNLOADED",
                type="Logic",
                size_mb=0.0,
                vram_gb=0.1,
                dependencies=["yolov8n", "yolo_pose"]
            )
        }

    def can_load(self, model_id: str) -> tuple[bool, str]:
        if model_id not in self.models:
            return False, f"Unknown model ID: {model_id}"

        model = self.models[model_id]
        for dep in model.dependencies:
            if self.models.get(dep, AIModel("", "", "UNLOADED", "", 0, 0, [])).status != "LOADED":
                dep_name = self.models[dep].name if dep in self.models else dep
                return False, f"Cannot load '{model.name}': Required dependency '{dep_name}' is NOT loaded."

        return True, "OK"

    def can_unload(self, model_id: str) -> tuple[bool, str]:
        if model_id not in self.models:
            return False, f"Unknown model ID: {model_id}"

        model = self.models[model_id]
        # Reverse dependency check: check if any active loaded model depends on this model
        for m_id, m_obj in self.models.items():
            if m_obj.status == "LOADED" and model_id in m_obj.dependencies:
                return False, f"Cannot unload '{model.name}': Active model '{m_obj.name}' depends on it!"

        return True, "OK"

    def set_model_status(self, model_id: str, status: str):
        if model_id in self.models:
            self.models[model_id].status = status

    def get_capability_matrix(self) -> List[Dict[str, str]]:
        capabilities = [
            {
                "feature": "Person Detection",
                "required": "YOLOv8n",
                "ready": "YES" if self.models["yolov8n"].status == "LOADED" else "NO"
            },
            {
                "feature": "Multi-Object Detection",
                "required": "YOLOv8n",
                "ready": "YES" if self.models["yolov8n"].status == "LOADED" else "NO"
            },
            {
                "feature": "Face Recognition",
                "required": "YuNet + SFace",
                "ready": "YES" if (self.models["yunet"].status == "LOADED" and self.models["sface"].status == "LOADED") else "NO"
            },
            {
                "feature": "Person Re-ID",
                "required": "OSNet",
                "ready": "YES" if self.models["osnet"].status == "LOADED" else "NO"
            },
            {
                "feature": "Pose Estimation",
                "required": "YOLOv8n-Pose",
                "ready": "YES" if self.models["yolo_pose"].status == "LOADED" else "NO"
            },
            {
                "feature": "Relationship Analysis",
                "required": "YOLOv8n + Pose + Relationship Engine",
                "ready": "YES" if (self.models["yolov8n"].status == "LOADED" and self.models["yolo_pose"].status == "LOADED" and self.models["relationship_engine"].status == "LOADED") else "NO"
            }
        ]
        return capabilities
