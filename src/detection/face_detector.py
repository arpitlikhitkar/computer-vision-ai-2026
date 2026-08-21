"""
Face Detector Module using OpenCV YuNet (FaceDetectorYN)

Provides ultra-fast face detection + 5-point landmark extraction inside image crops.
Landmarks returned:
1. Right Eye (x, y)
2. Left Eye (x, y)
3. Nose Tip (x, y)
4. Right Mouth Corner (x, y)
5. Left Mouth Corner (x, y)
"""

import os
import cv2
import numpy as np
from src.config import settings


class YuNetFaceDetector:
    """
    OpenCV YuNet ONNX Face Detector Wrapper.
    """
    def __init__(self, model_path=None, score_threshold=None, nms_threshold=0.3):
        self.model_path = model_path or settings.YUNET_MODEL_PATH
        self.score_threshold = score_threshold or settings.FACE_CONFIDENCE_THRESHOLD
        self.nms_threshold = nms_threshold

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[ERROR] YuNet model file not found: {self.model_path}")

        # Initialize detector with default dummy size (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )
        self.current_input_size = (320, 320)

    def detect_faces(self, img_bgr):
        """
        Detects faces in BGR image array.
        Returns list of face dicts:
        [
          {
            "bbox": [x, y, w, h],
            "confidence": float,
            "landmarks": [[re_x, re_y], [le_x, le_y], [n_x, n_y], [rm_x, rm_y], [lm_x, lm_y]],
            "raw": numpy_array (15 elements)
          }, ...
        ]
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        h, w = img_bgr.shape[:2]
        if (w, h) != self.current_input_size:
            self.detector.setInputSize((w, h))
            self.current_input_size = (w, h)

        # YuNet detect returns: (status, faces)
        # faces format per row (15 elements): [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, score]
        status, faces = self.detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
            score = float(face[14])
            landmarks = [
                [float(face[4]), float(face[5])],    # Right Eye
                [float(face[6]), float(face[7])],    # Left Eye
                [float(face[8]), float(face[9])],    # Nose Tip
                [float(face[10]), float(face[11])],  # Right Mouth Corner
                [float(face[12]), float(face[13])]   # Left Mouth Corner
            ]
            results.append({
                "bbox": bbox,
                "confidence": score,
                "landmarks": landmarks,
                "raw": face
            })

        return results


def associate_face_to_person_crop(person_crop, face_detector):
    """
    Detects faces inside a person crop and selects the largest/most prominent face.
    Returns (best_face_dict, relative_face_bbox) or (None, None).
    """
    if person_crop is None or person_crop.size == 0:
        return None

    faces = face_detector.detect_faces(person_crop)
    if not faces:
        return None

    # Sort faces by bounding box area (w * h) to pick largest face in person crop
    faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
    return faces[0]
