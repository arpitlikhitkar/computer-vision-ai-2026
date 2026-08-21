"""
Face Quality Assessment Module (Phase 5)

Evaluates face image suitability before face recognition:
1. Minimum Face Dimension Check (Width >= 40, Height >= 40)
2. Image Sharpness / Blur Check (Variance of Laplacian)
3. Brightness / Lighting Check (Mean Intensity)
4. Contrast Check (Standard Deviation)

NOTE: Face Quality Score evaluates image usability (e.g. blur/lighting),
NOT identity matching confidence.
"""

import cv2
import numpy as np
from src.config import settings


def evaluate_face_quality(face_crop_bgr):
    """
    Evaluates image quality of a cropped face BGR array.
    Returns tuple: (is_good: bool, quality_score: float, reason: str)
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return False, 0.0, "Empty face crop array"

    h, w = face_crop_bgr.shape[:2]

    # 1. Size Check
    if w < settings.MIN_FACE_WIDTH or h < settings.MIN_FACE_HEIGHT:
        return False, 0.0, f"Face too small ({w}x{h} < {settings.MIN_FACE_WIDTH}x{settings.MIN_FACE_HEIGHT})"

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Blur Check (Variance of Laplacian)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < settings.BLUR_THRESHOLD:
        return False, laplacian_var, f"Face image too blurry (Variance {laplacian_var:.1f} < {settings.BLUR_THRESHOLD})"

    # 3. Brightness Check
    mean_intensity = float(np.mean(gray))
    if mean_intensity < settings.BRIGHTNESS_MIN:
        return False, mean_intensity, f"Face too dark (Mean intensity {mean_intensity:.1f} < {settings.BRIGHTNESS_MIN})"
    if mean_intensity > settings.BRIGHTNESS_MAX:
        return False, mean_intensity, f"Face overexposed (Mean intensity {mean_intensity:.1f} > {settings.BRIGHTNESS_MAX})"

    # 4. Contrast Check
    std_contrast = float(np.std(gray))
    if std_contrast < 15.0:
        return False, std_contrast, f"Low contrast image (Std {std_contrast:.1f} < 15.0)"

    # Compute composite quality score (0.0 to 100.0)
    score = float(min(100.0, (laplacian_var / 2.0) + (std_contrast * 0.5)))
    return True, score, "Face Quality GOOD"
