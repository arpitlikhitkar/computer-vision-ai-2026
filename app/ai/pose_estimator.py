"""
Robust Multi-Cue Pose Classifier & Keypoint Estimator Module

Combines:
- 5 YuNet Facial Landmarks (Approximated Yaw / Pitch)
- 17 YOLOv8-Pose Body Keypoints (Ears, Shoulders, Wrists, Hips)
- Face Visibility & Body Confidence
- Multi-Cue Fusion to differentiate:
  FRONTAL, LEFT_PROFILE_30, LEFT_PROFILE_60, LEFT_PROFILE_FULL,
  RIGHT_PROFILE_30, RIGHT_PROFILE_60, RIGHT_PROFILE_FULL,
  LOOK_UP, LOOK_DOWN, REAR, PARTIAL_OCCLUDED, UNKNOWN_POSE
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config.settings import config


COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_CONNECTIONS = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle")
]


def classify_detailed_pose(landmarks_yunet=None, pose_keypoints=None, face_detected=True, face_quality_good=True):
    """
    Multi-Cue Pose Classifier combining Face Landmarks, Body Keypoints, and Detection Quality.
    Returns:
        pose_label: str (e.g. FRONTAL, LEFT_PROFILE_30°, REAR, etc.)
        yaw_deg: float
        is_rear_view: bool
    """
    kpts = pose_keypoints or {}
    l_sh = kpts.get("left_shoulder")
    r_sh = kpts.get("right_shoulder")
    l_ear = kpts.get("left_ear")
    r_ear = kpts.get("right_ear")
    nose_kpt = kpts.get("nose")
    l_eye_kpt = kpts.get("left_eye")
    r_eye_kpt = kpts.get("right_eye")

    # 1. Evaluate REAR View using Multi-Cue Body Topology (NOT just faces == [])
    # Conditions for REAR:
    # a) Strong shoulder detections
    # b) Absence of frontal facial features (nose/eyes confidence < 0.25 in pose model)
    # c) Shoulder inversion OR Ear visibility geometry
    has_strong_shoulders = (l_sh is not None and r_sh is not None and
                            l_sh.get('confidence', 0) > 0.40 and r_sh.get('confidence', 0) > 0.40)
    no_facial_features_in_body = (nose_kpt is None and l_eye_kpt is None and r_eye_kpt is None)

    if has_strong_shoulders and no_facial_features_in_body and not face_detected:
        # Verified REAR view via body keypoint topology
        return "REAR", 180.0, True

    # Check for inverted shoulders (left shoulder positioned to the right of right shoulder in 2D projection)
    if has_strong_shoulders and l_sh['x'] > (r_sh['x'] + 20):
        return "REAR", 180.0, True

    # If no face is detected but body topology doesn't clearly confirm REAR (e.g. occlusion, blur, distance)
    if not face_detected or not landmarks_yunet or len(landmarks_yunet) < 3:
        if not face_quality_good and face_detected:
            return "PARTIAL_OCCLUDED", 0.0, False
        if has_strong_shoulders and no_facial_features_in_body:
            return "REAR", 180.0, True
        return "UNKNOWN_POSE", 0.0, False

    # 2. Extract 2D Landmark Coordinates from YuNet (5-Point)
    re_x, re_y = landmarks_yunet[0]
    le_x, le_y = landmarks_yunet[1]
    n_x, n_y = landmarks_yunet[2]

    dist_r = float(np.sqrt((n_x - re_x) ** 2 + (n_y - re_y) ** 2))
    dist_l = float(np.sqrt((n_x - le_x) ** 2 + (n_y - le_y) ** 2))
    eye_dist = float(np.sqrt((le_x - re_x) ** 2 + (le_y - re_y) ** 2))

    if eye_dist <= 1.0:
        return "FRONTAL", 0.0, False

    # Yaw Ratio: (dist_r - dist_l) / eye_dist
    yaw_ratio = (dist_r - dist_l) / eye_dist
    yaw_deg = float(np.clip(yaw_ratio * 75.0, -90.0, 90.0))

    # Pitch Difference: (nose_y - midpoint_eyes_y) / eye_dist
    eye_mid_y = (re_y + le_y) / 2.0
    pitch_diff = (n_y - eye_mid_y) / eye_dist

    if pitch_diff < -0.15:
        return "LOOK_UP", yaw_deg, False
    elif pitch_diff > 0.45:
        return "LOOK_DOWN", yaw_deg, False

    # Granular Profile Angles
    if yaw_ratio > 0.45:
        return "LEFT_PROFILE_FULL", yaw_deg, False
    elif yaw_ratio > 0.30:
        return "LEFT_PROFILE_60°", yaw_deg, False
    elif yaw_ratio > 0.14:
        return "LEFT_PROFILE_30°", yaw_deg, False
    elif yaw_ratio < -0.45:
        return "RIGHT_PROFILE_FULL", yaw_deg, False
    elif yaw_ratio < -0.30:
        return "RIGHT_PROFILE_60°", yaw_deg, False
    elif yaw_ratio < -0.14:
        return "RIGHT_PROFILE_30°", yaw_deg, False
    else:
        return "FRONTAL", yaw_deg, False


class PoseEstimator:
    """
    YOLOv8-Pose Keypoint Estimator.
    """
    def __init__(self, model_path=None, device=None):
        path = model_path or config.yolo_pose_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(path)

    def estimate_pose_for_person(self, frame_bgr, person_bbox):
        default_kpts = {k_name: None for k_name in COCO_KEYPOINTS}

        if frame_bgr is None or frame_bgr.size == 0 or not person_bbox:
            return default_kpts

        x1, y1, x2, y2 = person_bbox
        h, w = frame_bgr.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return default_kpts

        crop = frame_bgr[y1:y2, x1:x2]
        results = self.model.predict(source=crop, device=self.device, verbose=False, conf=0.15, imgsz=224)

        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return default_kpts

        kpts_data = results[0].keypoints.data[0].cpu().numpy()

        keypoints_abs = default_kpts.copy()
        for idx, k_name in enumerate(COCO_KEYPOINTS):
            if idx < len(kpts_data):
                rel_x, rel_y, conf_val = kpts_data[idx]
                if conf_val >= 0.25:
                    abs_x = int(x1 + rel_x)
                    abs_y = int(y1 + rel_y)
                    keypoints_abs[k_name] = {
                        'x': abs_x,
                        'y': abs_y,
                        'confidence': float(conf_val)
                    }

        return keypoints_abs

    @staticmethod
    def draw_skeleton(frame, keypoints_abs, color=(0, 255, 255)):
        if not keypoints_abs:
            return

        for p1_name, p2_name in SKELETON_CONNECTIONS:
            kp1 = keypoints_abs.get(p1_name)
            kp2 = keypoints_abs.get(p2_name)
            if kp1 and kp2:
                cv2.line(frame, (kp1['x'], kp1['y']), (kp2['x'], kp2['y']), color, 2, cv2.LINE_AA)

        for k_name, kp in keypoints_abs.items():
            if kp:
                radius = 6 if "wrist" in k_name else 3
                cv2.circle(frame, (kp['x'], kp['y']), radius, color, -1, cv2.LINE_AA)
