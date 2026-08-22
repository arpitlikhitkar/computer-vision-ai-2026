"""
Multi-Class YOLOv8 Detection Module (Phase 6.1)

Removes hardcoded person-only restriction and detects all 80 COCO classes.
Returns categorized detections: PERSON, ANIMAL, OBJECT, VEHICLE.
"""

import numpy as np
import torch
from ultralytics import YOLO

from app.config.settings import config
from app.ai.category_mapper import categorize_class


class MultiClassYOLO:
    """
    YOLOv8 Multi-Class Detector for all 80 COCO classes.
    """
    def __init__(self, model_path=None, device=None):
        path = model_path or config.yolo_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(path)
        self.names = self.model.names

    def detect(self, frame_bgr, conf_threshold=None):
        """
        Detects all 80 COCO classes in frame.
        Returns list of detection dicts:
        [
            {
                'class': 'cell phone',
                'confidence': 0.88,
                'bbox': [x1, y1, x2, y2],
                'track_id': None,
                'category': 'OBJECT'
            }, ...
        ]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        conf = conf_threshold if conf_threshold is not None else config.multiclass_conf_threshold

        results = self.model.predict(
            source=frame_bgr,
            device=self.device,
            verbose=False,
            conf=conf,
            imgsz=320
        )

        detections = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for xyxy, c_score, cls_id in zip(xyxys, confs, cls_ids):
            class_name = self.names.get(cls_id, f"class_{cls_id}")
            category = categorize_class(class_name)
            x1, y1, x2, y2 = map(int, xyxy)

            detections.append({
                'class': class_name,
                'confidence': float(c_score),
                'bbox': [x1, y1, x2, y2],
                'track_id': None,
                'category': category
            })

        return detections
