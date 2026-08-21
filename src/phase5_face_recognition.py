"""
Phase 5 — Household Face Recognition Main Pipeline

FULL PIPELINE:
Camera -> OpenCV -> YOLO Person Detection -> ByteTrack ->
YuNet Face Detection -> Quality Check -> 5-Landmark Affine Alignment ->
SFace 128-d Embedding -> SQLite Memory Search -> Candidate Threshold ->
Temporal State Machine -> Green (Known) / Red (Unknown) UI Overlay

Usage:
python src/phase5_face_recognition.py
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import settings
from src.detection.face_detector import YuNetFaceDetector
from src.recognition.face_quality import evaluate_face_quality
from src.recognition.face_alignment import SFaceAligner
from src.recognition.face_embedder import SFaceEmbedder
from src.recognition.face_matcher import FaceMatcher
from src.storage.database import initialize_database
from src.storage.embedding_repository import EmbeddingRepository
from src.storage.log_repository import LogRepository
from src.ui.overlay import render_face_recognition_overlay, render_system_header


def run_phase5_face_recognition():
    print("==================================================")
    print(" PHASE 5 — HOUSEHOLD FACE RECOGNITION SYSTEM")
    print("==================================================")

    # Initialize Database Schema
    initialize_database()
    embedding_repo = EmbeddingRepository()
    log_repo = LogRepository()

    # Load Enrolled Members Dictionary
    enrolled_persons = embedding_repo.get_all_active_enrolled_dictionary()
    print(f"[INFO] Loaded {len(enrolled_persons)} active enrolled household members from database.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU Mode"

    print(f"[INFO] Hardware Device:      {device_label}")
    print(f"[INFO] Person Model:         {settings.YOLO_MODEL_PATH}")
    print(f"[INFO] Face Detector:        YuNet (2023mar ONNX)")
    print(f"[INFO] Face Recognizer:      SFace (2021dec ONNX 128-d)")
    print(f"[INFO] Recognition Threshold: {settings.RECOGNITION_THRESHOLD}")
    print("==================================================")

    # Load Models
    try:
        yolo_model = YOLO(settings.YOLO_MODEL_PATH)
        face_detector = YuNetFaceDetector()
        face_embedder = SFaceEmbedder()
        face_aligner = SFaceAligner(face_embedder.recognizer)
        face_matcher = FaceMatcher(threshold=settings.RECOGNITION_THRESHOLD)
    except Exception as e:
        print(f"[ERROR] Failed initializing AI models: {e}")
        return False

    # Open Camera
    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(settings.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {settings.CAMERA_INDEX}.")
        return False

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Hardware Warmup
    print("[INFO] Warming up camera sensor...")
    for _ in range(15):
        cap.read()

    window_name = "Computer Vision AI - Household Face Recognition (Phase 5)"
    print("[SUCCESS] Phase 5 Face Recognition Pipeline Operational!")
    print("[INFO] Press 'Q', 'ESC', or click [X] button to exit.")

    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    # Track State Dictionary: track_id -> state dict
    track_states = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            display_frame = frame.copy()
            frame_h, frame_w = display_frame.shape[:2]
            current_time = time.time()

            # Run YOLO Person Detection + ByteTrack
            results = yolo_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # Person class
                device=device,
                verbose=False,
                conf=settings.PERSON_CONFIDENCE_THRESHOLD,
                imgsz=320
            )

            result = results[0]
            active_track_ids = []
            known_count = 0
            unknown_count = 0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                has_id = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_id else [None] * len(boxes)
                confidences = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for xyxy, conf, track_id in zip(xyxys, confidences, track_ids):
                    if track_id is None:
                        continue

                    active_track_ids.append(track_id)

                    # Initialize state machine for new Track ID
                    if track_id not in track_states:
                        track_states[track_id] = {
                            "state": "PENDING",       # NO_FACE, FACE_DETECTED, QUALITY_LOW, PENDING, KNOWN, UNKNOWN
                            "person_uuid": None,
                            "display_id": None,
                            "display_name": "UNKNOWN",
                            "similarity": 0.0,
                            "quality_msg": "",
                            "face_bbox": None,
                            "obs_count": 0,
                            "known_obs": 0,
                            "unknown_obs": 0,
                            "last_eval_time": 0.0
                        }

                    state = track_states[track_id]
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1_c = max(0, min(x1, frame_w - 1))
                    y1_c = max(0, min(y1, frame_h - 1))
                    x2_c = max(0, min(x2, frame_w))
                    y2_c = max(0, min(y2, frame_h))

                    person_crop = frame[y1_c:y2_c, x1_c:x2_c]

                    # Run Face Recognition Evaluation periodically per track
                    if (
                        person_crop.size > 0 and
                        (current_time - state["last_eval_time"]) >= 0.4
                    ):
                        state["last_eval_time"] = current_time

                        # 1. Face Detection inside Person Crop
                        faces = face_detector.detect_faces(person_crop)

                        if not faces:
                            if state["state"] == "PENDING":
                                state["state"] = "NO_FACE"
                                state["quality_msg"] = "Face not visible"
                        else:
                            # Select largest face
                            faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
                            best_face = faces[0]
                            state["face_bbox"] = best_face["bbox"]
                            fx, fy, fw, fh = best_face["bbox"]

                            face_crop = person_crop[fy:fy + fh, fx:fx + fw]

                            # 2. Face Quality Check
                            is_good, q_score, q_reason = evaluate_face_quality(face_crop)

                            if not is_good:
                                if state["state"] != "KNOWN":
                                    state["state"] = "QUALITY_LOW"
                                    state["quality_msg"] = q_reason
                            else:
                                # 3. 5-Point Landmark Affine Alignment
                                aligned_chip = face_aligner.align_face(person_crop, best_face)

                                if aligned_chip is not None:
                                    # 4. SFace 128-d Feature Embedding
                                    query_emb = face_embedder.extract_embedding(aligned_chip)

                                    # 5. Database Candidate Matching
                                    match_res = face_matcher.match_against_enrolled(query_emb, enrolled_persons)

                                    state["similarity"] = match_res["similarity"]

                                    # 6. Temporal State Machine Stabilization
                                    state["obs_count"] += 1

                                    if match_res["matched"]:
                                        state["known_obs"] += 1
                                        state["unknown_obs"] = 0
                                        if state["known_obs"] >= settings.RECOGNITION_CONFIRMATION_COUNT:
                                            state["state"] = "KNOWN"
                                            state["person_uuid"] = match_res["person_uuid"]
                                            state["display_id"] = match_res["display_id"]
                                            state["display_name"] = match_res["display_name"]

                                            # Log audit event
                                            log_repo.log_recognition_event(
                                                track_id=track_id,
                                                person_uuid=match_res["person_uuid"],
                                                recognition_result="KNOWN",
                                                similarity_score=match_res["similarity"]
                                            )
                                    else:
                                        state["unknown_obs"] += 1
                                        if state["unknown_obs"] >= 4:
                                            state["state"] = "UNKNOWN"
                                            state["display_name"] = "UNKNOWN"

                                            log_repo.log_recognition_event(
                                                track_id=track_id,
                                                person_uuid=None,
                                                recognition_result="UNKNOWN",
                                                similarity_score=match_res["similarity"]
                                            )

                    if state["state"] == "KNOWN":
                        known_count += 1
                    elif state["state"] == "UNKNOWN":
                        unknown_count += 1

                    # Render Visual Green/Red Bounding Box Overlay
                    render_face_recognition_overlay(
                        display_frame=display_frame,
                        track_id=track_id,
                        person_bbox=(x1, y1, x2, y2),
                        face_bbox=state.get("face_bbox"),
                        state_dict=state,
                        device_label=device,
                        fps=fps,
                        active_tracks_count=len(active_track_ids),
                        known_count=known_count,
                        unknown_count=unknown_count
                    )

            # Calculate Pipeline FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Render System Performance Header
            render_system_header(
                display_frame,
                device_label,
                fps,
                len(active_track_ids),
                known_count,
                unknown_count
            )

            cv2.imshow(window_name, display_frame)

            # Check Exit Conditions
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window [X] closed by user.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27, 32):
                print("[INFO] Exit key pressed. Exiting Phase 5 demo...")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Runtime exception: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed cleanly.")

    return True

if __name__ == "__main__":
    run_phase5_face_recognition()
