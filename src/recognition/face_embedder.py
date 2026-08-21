"""
Face Embedder Module (Phase 5)

Extracts 128-dimensional L2-normalized feature embeddings from aligned face chips (112x112)
using OpenCV SFace (FaceRecognizerSF).
"""

import os
import cv2
import numpy as np
from src.config import settings


class SFaceEmbedder:
    """
    OpenCV SFace 128-dimensional Face Feature Embedder.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path or settings.SFACE_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[ERROR] SFace model file not found: {self.model_path}")

        self.recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

    def extract_embedding(self, aligned_face_chip):
        """
        Extracts 128-d L2-normalized feature embedding vector from aligned 112x112 face chip.
        Returns: 1D float32 numpy array of shape (128,).
        """
        if aligned_face_chip is None or aligned_face_chip.size == 0:
            raise ValueError("[ERROR] Invalid or empty aligned face chip passed to extract_embedding.")

        # SFace feature extraction outputs (1, 128) float32 array
        feat = self.recognizer.feature(aligned_face_chip)
        embedding = feat.flatten().astype(np.float32)

        # L2 Normalization (ensures unit vector length = 1.0)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        return embedding
