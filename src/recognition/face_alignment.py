"""
Face Alignment Module (Phase 5)

Performs 5-landmark affine transformation to align face eyes horizontally
and produce a standardized 112x112 aligned face chip for SFace feature extraction.
"""

import cv2
import numpy as np


class SFaceAligner:
    """
    OpenCV SFace Face Alignment Engine.
    Uses 5 facial landmarks (right eye, left eye, nose tip, right mouth, left mouth)
    to perform 5-point affine alignment.
    """
    def __init__(self, recognizer_instance=None):
        self.recognizer = recognizer_instance

    def align_face(self, img_bgr, face_dict):
        """
        Aligns a face image using 5-point landmark affine transformation.
        Input: img_bgr (BGR image array), face_dict (from YuNet detector with 'raw' array).
        Returns: aligned_face_chip (112x112 BGR numpy array) or None.
        """
        if img_bgr is None or face_dict is None or "raw" not in face_dict:
            return None

        if self.recognizer is None:
            # Create transient FaceRecognizerSF for alignCrop
            from src.config import settings
            self.recognizer = cv2.FaceRecognizerSF.create(settings.SFACE_MODEL_PATH, "")

        raw_face = face_dict["raw"]
        aligned_chip = self.recognizer.alignCrop(img_bgr, raw_face)
        return aligned_chip
