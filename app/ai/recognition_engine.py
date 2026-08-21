"""
AI Recognition Engine for PySide6 Application
Updated with 60-second Rolling Video Recording Buffer (-30s pre-event + +30s post-event)
"""

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config.settings import config
from src.detection.face_detector import YuNetFaceDetector
from src.recognition.face_quality import evaluate_face_quality
from src.recognition.face_alignment import SFaceAligner
from src.recognition.face_embedder import SFaceEmbedder
from app.ai.body_reid_embedder import OSNetBodyEmbedder
from app.ai.fusion_engine import MultiModalFusionEngine
from app.services.video_recorder import CircularFrameBuffer, EventVideoRecorderWorker
from app.database.embedding_repository import EmbeddingRepository
from app.database.event_repository import EventRepository
from app.database.unknown_repository import UnknownRepository


class RecognitionEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU Mode"

        self.yolo = YOLO(config.yolo_model_path)
        self.face_detector = YuNetFaceDetector()
        self.face_embedder = SFaceEmbedder()
        self.face_aligner = SFaceAligner(self.face_embedder.recognizer)
        self.body_embedder = OSNetBodyEmbedder(device=self.device)

        self.fusion_engine = MultiModalFusionEngine(threshold=config.recognition_threshold)

        self.embedding_repo = EmbeddingRepository()
        self.event_repo = EventRepository()
        self.unknown_repo = UnknownRepository()

        self.enrolled_cache = self.embedding_repo.get_all_active_enrolled_dictionary()
        self.track_states = {}

        # 30-Second Pre-Event Ring Buffer (-30s)
        self.ring_buffer = CircularFrameBuffer(max_seconds=30, fps=15)
        self.active_recorder_workers = []
        self.last_unknown_snapshot_time = 0.0

    def refresh_enrolled_cache(self):
        self.enrolled_cache = self.embedding_repo.get_all_active_enrolled_dictionary()
        self.fusion_engine.threshold = config.recognition_threshold

    def process_frame(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, 0, 0, 0

        # Push frame to 30-second Pre-Event Ring Buffer
        self.ring_buffer.append(frame_bgr)

        # Feed frame to active post-event recorder workers
        for worker in list(self.active_recorder_workers):
            if worker.isRunning():
                worker.add_post_event_frame(frame_bgr)
            else:
                self.active_recorder_workers.remove(worker)

        display_frame = frame_bgr.copy()
        frame_h, frame_w = display_frame.shape[:2]
        current_time = time.time()

        # Run YOLO + ByteTrack
        results = self.yolo.track(
            source=frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            device=self.device,
            verbose=False,
            conf=config.person_conf_threshold,
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

                if track_id not in self.track_states:
                    self.track_states[track_id] = {
                        "state": "PENDING",
                        "person_uuid": None,
                        "display_id": None,
                        "display_name": "UNKNOWN",
                        "face_score": 0.0,
                        "body_score": 0.0,
                        "final_score": 0.0,
                        "quality_msg": "",
                        "face_bbox": None,
                        "obs_count": 0,
                        "known_obs": 0,
                        "unknown_obs": 0,
                        "recorded_event": False,
                        "last_eval_time": 0.0
                    }

                state = self.track_states[track_id]
                x1, y1, x2, y2 = map(int, xyxy)
                x1_c = max(0, min(x1, frame_w - 1))
                y1_c = max(0, min(y1, frame_h - 1))
                x2_c = max(0, min(x2, frame_w))
                y2_c = max(0, min(y2, frame_h))

                person_crop = frame_bgr[y1_c:y2_c, x1_c:x2_c]

                # Process AI evaluation periodically per track
                if (
                    person_crop.size > 0 and
                    (current_time - state["last_eval_time"]) >= 0.35
                ):
                    state["last_eval_time"] = current_time

                    # Extract OSNet 512-d Body Re-ID Embedding
                    query_body_emb = None
                    if person_crop.shape[0] >= 40 and person_crop.shape[1] >= 20:
                        try:
                            query_body_emb = self.body_embedder.extract_embedding(person_crop)
                        except Exception:
                            query_body_emb = None

                    # Extract SFace 128-d Face Embedding if face present
                    query_face_emb = None
                    faces = self.face_detector.detect_faces(person_crop)

                    if faces:
                        faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
                        best_face = faces[0]
                        state["face_bbox"] = best_face["bbox"]

                        fx, fy, fw, fh = best_face["bbox"]
                        face_crop = person_crop[fy:fy + fh, fx:fx + fw]

                        is_good, q_score, q_reason = evaluate_face_quality(face_crop)
                        if is_good:
                            aligned_chip = self.face_aligner.align_face(person_crop, best_face)
                            if aligned_chip is not None:
                                query_face_emb = self.face_embedder.extract_embedding(aligned_chip)

                    # Run Multi-Modal Score Fusion
                    self.enrolled_cache = self.embedding_repo.get_all_active_enrolled_dictionary()
                    match_res = self.fusion_engine.match_multi_modal(
                        query_face_emb=query_face_emb,
                        query_body_emb=query_body_emb,
                        enrolled_dict=self.enrolled_cache
                    )

                    state["face_score"] = match_res["face_score"]
                    state["body_score"] = match_res["body_score"]
                    state["final_score"] = match_res["final_score"]
                    state["obs_count"] += 1

                    if match_res["matched"]:
                        state["known_obs"] += 1
                        state["unknown_obs"] = 0
                        if state["known_obs"] >= 2:
                            state["state"] = "KNOWN"
                            state["person_uuid"] = match_res["person_uuid"]
                            state["display_id"] = match_res["display_id"]
                            state["display_name"] = match_res["display_name"]

                            self.event_repo.add_event(
                                track_id=track_id,
                                person_uuid=match_res["person_uuid"],
                                recognition_result="KNOWN",
                                similarity_score=match_res["final_score"]
                            )
                    else:
                        state["unknown_obs"] += 1
                        if state["unknown_obs"] >= 4:
                            state["state"] = "UNKNOWN"
                            state["display_name"] = "UNKNOWN"

                            self.event_repo.add_event(
                                track_id=track_id,
                                person_uuid=None,
                                recognition_result="UNKNOWN",
                                similarity_score=match_res["final_score"]
                            )

                            # TRIGGER 60s VIDEO RECORDING (-30s pre-event + +30s post-event)
                            if not state["recorded_event"] and (current_time - self.last_unknown_snapshot_time) >= 10.0:
                                state["recorded_event"] = True
                                self.last_unknown_snapshot_time = current_time

                                pre_frames = self.ring_buffer.get_pre_event_snapshot()
                                rec_worker = EventVideoRecorderWorker(
                                    pre_event_frames=pre_frames,
                                    track_id=track_id,
                                    fps=15
                                )
                                rec_worker.start()
                                self.active_recorder_workers.append(rec_worker)

                # Format UI Bounding Box Label
                if state["state"] == "KNOWN":
                    known_count += 1
                    box_color = (0, 255, 127)  # GREEN
                    line1 = f"{state['display_name']} ({state['display_id']}) | Track: {track_id}"
                    line2 = f"Face: {state['face_score']*100:.0f}% | Body: {state['body_score']*100:.0f}% | Final: {state['final_score']*100:.0f}%"
                elif state["state"] == "UNKNOWN":
                    unknown_count += 1
                    box_color = (0, 0, 255)    # RED
                    line1 = f"UNKNOWN | Track: {track_id}"
                    line2 = f"Face: {state['face_score']*100:.0f}% | Body: {state['body_score']*100:.0f}% | Final: {state['final_score']*100:.0f}%"
                else:
                    box_color = (0, 215, 255)  # YELLOW
                    line1 = f"EVALUATING | Track: {track_id}"
                    line2 = "Analyzing multi-modal identity..."

                # Draw Bounding Box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw Header Label Box
                (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                max_w = max(tw1, tw2) + 10
                total_h = th1 + th2 + 10
                lbl_y1 = max(0, y1 - total_h)

                cv2.rectangle(display_frame, (x1, lbl_y1), (x1 + max_w, lbl_y1 + total_h), box_color, -1)
                cv2.putText(display_frame, line1, (x1 + 5, lbl_y1 + th1 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(display_frame, line2, (x1 + 5, lbl_y1 + th1 + th2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        return display_frame, len(active_track_ids), known_count, unknown_count
