"""
OSNet Body / Person Re-ID Feature Extractor Module (Phase 5.5)

Extracts 512-dimensional L2-normalized body appearance embeddings from full-person BGR crops
using pretrained OSNet (osnet_x0_25).
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torchreid


class OSNetBodyEmbedder:
    """
    OSNet 512-d Person Re-ID Feature Embedder.
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build OSNet architecture
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1000,
            loss="softmax",
            pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)

        # Standard Re-ID Image Preprocessing Transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_embedding(self, person_crop_bgr):
        """
        Extracts 512-d L2-normalized body feature vector from full-person BGR crop.
        Returns: 1D float32 numpy array of shape (512,).
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            raise ValueError("[ERROR] Invalid or empty person crop array passed to OSNetBodyEmbedder.")

        h, w = person_crop_bgr.shape[:2]
        if h < 20 or w < 10:
            raise ValueError(f"[ERROR] Person crop too small ({w}x{h}).")

        # Convert OpenCV BGR to PIL Image (RGB)
        rgb_crop = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_crop)

        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor_img)
            features = features.cpu().numpy().flatten().astype(np.float32)

        # L2 Normalization
        norm = float(np.linalg.norm(features))
        if norm > 0:
            features = features / norm

        return features
