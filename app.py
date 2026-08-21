"""
Household AI Web Application & Desktop Software Server (Phase 5)

Runs local Flask Web Server providing:
- MJPEG Live Video Stream with Green/Red Face Bounding Boxes
- Interactive Web Member Enrollment API
- Household Members Management Dashboard API (Active/Inactive, Delete)
- Real-Time Recognition Audit Logs API
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

from src.config import settings
from src.detection.face_detector import YuNetFaceDetector
from src.recognition.face_quality import evaluate_face_quality
from src.recognition.face_alignment import SFaceAligner
from src.recognition.face_embedder import SFaceEmbedder
from src.recognition.face_matcher import FaceMatcher
from src.storage.database import initialize_database
from src.storage.person_repository import PersonRepository
from src.storage.embedding_repository import EmbeddingRepository
from src.storage.log_repository import LogRepository
from src.ui.overlay import render_face_recognition_overlay, render_system_header

app = Flask(__name__)

# Global AI Engine State
camera = None
yolo_model = None
face_detector = None
face_embedder = None
face_aligner = None
face_matcher = None
person_repo = None
embedding_repo = None
log_repo = None

enrolled_cache = {}
track_states = {}


def init_app_engine():
    """Initializes SQLite database and PyTorch/OpenCV AI models."""
    global yolo_model, face_detector, face_embedder, face_aligner, face_matcher
    global person_repo, embedding_repo, log_repo, enrolled_cache

    initialize_database()
    person_repo = PersonRepository()
    embedding_repo = EmbeddingRepository()
    log_repo = LogRepository()

    enrolled_cache = embedding_repo.get_all_active_enrolled_dictionary()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SERVER] Initializing AI models on device: {device}...")

    yolo_model = YOLO(settings.YOLO_MODEL_PATH)
    face_detector = YuNetFaceDetector()
    face_embedder = SFaceEmbedder()
    face_aligner = SFaceAligner(face_embedder.recognizer)
    face_matcher = FaceMatcher(threshold=settings.RECOGNITION_THRESHOLD)

    print("[SERVER] All AI Engine models loaded successfully!")


def get_camera_stream():
    """Lazy camera stream getter."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(settings.CAMERA_INDEX)

        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warmup
        for _ in range(10):
            camera.read()

    return camera


def generate_mjpeg_frames():
    """MJPEG Video Stream Generator."""
    global enrolled_cache, track_states

    cam = get_camera_stream()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1
        display_frame = frame.copy()
        frame_h, frame_w = display_frame.shape[:2]
        current_time = time.time()

        # Run YOLO + ByteTrack
        results = yolo_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
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

                if track_id not in track_states:
                    track_states[track_id] = {
                        "state": "PENDING",
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

                # Face Recognition Evaluation
                if (
                    person_crop.size > 0 and
                    (current_time - state["last_eval_time"]) >= 0.3
                ):
                    state["last_eval_time"] = current_time

                    faces = face_detector.detect_faces(person_crop)
                    if not faces:
                        if state["state"] == "PENDING":
                            state["state"] = "NO_FACE"
                            state["quality_msg"] = "Face not visible"
                    else:
                        faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
                        best_face = faces[0]
                        state["face_bbox"] = best_face["bbox"]

                        fx, fy, fw, fh = best_face["bbox"]
                        face_crop = person_crop[fy:fy + fh, fx:fx + fw]

                        is_good, q_score, q_reason = evaluate_face_quality(face_crop)
                        if not is_good:
                            if state["state"] != "KNOWN":
                                state["state"] = "QUALITY_LOW"
                                state["quality_msg"] = q_reason
                        else:
                            aligned_chip = face_aligner.align_face(person_crop, best_face)
                            if aligned_chip is not None:
                                query_emb = face_embedder.extract_embedding(aligned_chip)

                                # Refresh enrolled cache
                                enrolled_cache = embedding_repo.get_all_active_enrolled_dictionary()
                                match_res = face_matcher.match_against_enrolled(query_emb, enrolled_cache)

                                state["similarity"] = match_res["similarity"]
                                state["obs_count"] += 1

                                if match_res["matched"]:
                                    state["known_obs"] += 1
                                    state["unknown_obs"] = 0
                                    if state["known_obs"] >= 2:
                                        state["state"] = "KNOWN"
                                        state["person_uuid"] = match_res["person_uuid"]
                                        state["display_id"] = match_res["display_id"]
                                        state["display_name"] = match_res["display_name"]

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

        # Pipeline FPS calculation
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = time.time()

        render_system_header(
            display_frame,
            device,
            fps,
            len(active_track_ids),
            known_count,
            unknown_count
        )

        ret_jpg, jpeg_buf = cv2.imencode('.jpg', display_frame)
        if not ret_jpg:
            continue

        frame_bytes = jpeg_buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- Web Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_mjpeg_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/members', methods=['GET'])
def get_members():
    persons = person_repo.get_all_persons()
    result = []
    for p in persons:
        vecs = embedding_repo.get_embeddings_for_person(p["person_uuid"])
        result.append({
            "person_uuid": p["person_uuid"],
            "display_id": p["display_id"],
            "display_name": p["display_name"],
            "status": p["status"],
            "sample_count": len(vecs),
            "created_at": p["created_at"]
        })
    return jsonify({"members": result})


@app.route('/api/enroll', methods=['POST'])
def enroll_member_api():
    global enrolled_cache
    data = request.get_json() or {}
    name = data.get("name", "").strip()

    if not name:
        return jsonify({"success": False, "message": "Member name is required"}), 400

    cam = get_camera_stream()
    collected_embeddings = []
    collected_quality_scores = []
    last_cap_time = 0.0

    print(f"[API ENROLL] Starting face sample capture for '{name}'...")

    for _ in range(300):  # Maximum loop attempts (~10-15 seconds)
        if len(collected_embeddings) >= settings.ENROLLMENT_SAMPLE_COUNT:
            break

        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        current_time = time.time()
        if (current_time - last_cap_time) >= 0.5:
            faces = face_detector.detect_faces(frame)
            if len(faces) == 1:
                face = faces[0]
                fx, fy, fw, fh = face["bbox"]
                face_crop = frame[fy:fy + fh, fx:fx + fw]

                is_good, score, reason = evaluate_face_quality(face_crop)
                if is_good:
                    aligned_chip = face_aligner.align_face(frame, face)
                    if aligned_chip is not None:
                        emb = face_embedder.extract_embedding(aligned_chip)
                        collected_embeddings.append(emb)
                        collected_quality_scores.append(score)
                        last_cap_time = current_time
                        print(f"[API ENROLL] Sample {len(collected_embeddings)}/10 captured.")

        time.sleep(0.02)

    if len(collected_embeddings) >= settings.ENROLLMENT_SAMPLE_COUNT:
        person = person_repo.add_person(name)
        for emb, q_score in zip(collected_embeddings, collected_quality_scores):
            embedding_repo.add_embedding(person["person_uuid"], emb, quality_score=q_score)

        enrolled_cache = embedding_repo.get_all_active_enrolled_dictionary()

        return jsonify({
            "success": True,
            "display_name": name,
            "display_id": person["display_id"],
            "person_uuid": person["person_uuid"]
        })
    else:
        return jsonify({
            "success": False,
            "message": f"Captured only {len(collected_embeddings)}/10 samples. Please look directly at the camera in good lighting."
        }), 400


@app.route('/api/members/toggle', methods=['POST'])
def toggle_member_api():
    global enrolled_cache
    data = request.get_json() or {}
    uuid_val = data.get("uuid")
    new_status = data.get("status")

    if uuid_val and new_status in ("ACTIVE", "INACTIVE"):
        person_repo.update_person_status(uuid_val, new_status)
        enrolled_cache = embedding_repo.get_all_active_enrolled_dictionary()
        return jsonify({"success": True})
    return jsonify({"success": False}), 400


@app.route('/api/members/delete', methods=['POST'])
def delete_member_api():
    global enrolled_cache
    data = request.get_json() or {}
    uuid_val = data.get("uuid")

    if uuid_val:
        person_repo.delete_person(uuid_val)
        enrolled_cache = embedding_repo.get_all_active_enrolled_dictionary()
        return jsonify({"success": True})
    return jsonify({"success": False}), 400


@app.route('/api/logs', methods=['GET'])
def get_logs_api():
    logs = log_repo.get_recent_logs(20)
    result = []
    for l in logs:
        person_name = None
        if l["person_uuid"]:
            p = person_repo.get_person_by_uuid(l["person_uuid"])
            if p:
                person_name = p["display_name"]

        result.append({
            "id": l["id"],
            "track_id": l["track_id"],
            "person_name": person_name,
            "recognition_result": l["recognition_result"],
            "similarity_score": l["similarity_score"],
            "timestamp": l["timestamp"]
        })
    return jsonify({"logs": result})


if __name__ == "__main__":
    init_app_engine()
    print("==================================================")
    print(" HOUSEHOLD AI SOFTWARE WEB APP RUNNING!")
    print(" Open URL in Browser:  http://127.0.0.1:5000")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
