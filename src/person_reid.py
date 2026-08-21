"""
Person Re-Identification (Re-ID) Script (Phase 4 - Multi-Person Global Association)

MULTI-PERSON RE-ID DATA ASSOCIATION PIPELINE:
1. Multi-Track Detection: For EVERY active Track ID, extract bounding box & OSNet crop.
2. Feature Extraction: Generate L2-normalized composite embeddings (OSNet + HSV Appearance).
3. Independent Candidate Evaluation: Compare each active track against ALL existing Person Records.
4. Debug Similarity Table: Print full candidate similarity matrix to console.
5. Global Conflict Resolution: Use Hungarian Algorithm (linear_sum_assignment) for 1-to-1 matching.
6. Identity Assignment: Guarantee no two simultaneous active tracks share the same Person Record.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
import numpy as np
from scipy.optimize import linear_sum_assignment

# Configurable Parameters
REID_SIMILARITY_THRESHOLD = 0.75  # Optimal cutoff for Re-ID match association
REID_CONFIRMATION_COUNT = 8        # Frames required to evaluate before declaring NEW person
MAX_EMBEDDINGS_PER_PERSON = 20     # Max embeddings stored per person record
SAVE_INTERVAL_SECONDS = 0.3        # Faster observation sampling interval (300ms)
MIN_CROP_WIDTH = 40                # Minimum bounding box width
MIN_CROP_HEIGHT = 80               # Minimum bounding box height
CONFIDENCE_THRESHOLD = 0.50        # Minimum detection confidence
CAMERA_ID = "WEBCAM-01"            # Camera source identifier
DEBUG_MODE = True                  # Print full similarity matrix in console


class OSNetReIDExtractor:
    """
    Dedicated Person Re-ID Feature Extractor using Torchreid OSNet + HSV Color Distribution.
    Produces L2-normalized feature embeddings.
    """
    def __init__(self, device="cpu"):
        import torchreid.reid.utils as utils
        self.extractor = utils.FeatureExtractor(
            model_name="osnet_x0_25",
            device=device,
            verbose=False
        )

    def extract_from_bgr(self, crop_bgr):
        """Extracts L2-normalized feature embedding from BGR image array."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Invalid or empty image array passed to extract_from_bgr.")

        import cv2
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. OSNet Deep Feature Embedding (512-d)
        features = self.extractor(img_rgb)
        feat = features.cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        norm_feat = feat / norm if norm > 0 else feat

        # 2. HSV Appearance Distribution (32-d)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        color_hist = np.concatenate([h_hist, s_hist]).flatten()
        color_norm = np.linalg.norm(color_hist)
        norm_color = color_hist / color_norm if color_norm > 0 else color_hist

        # Weighted combination: 70% Deep Features + 30% Color Appearance
        combined = np.concatenate([norm_feat * 0.7, norm_color * 0.3])
        final_norm = np.linalg.norm(combined)
        return combined / final_norm if final_norm > 0 else combined


class PersonRecordReID:
    """
    Manages individual person folder, crops, embeddings, and metadata.
    """
    def __init__(self, person_id, track_id, output_base_dir):
        self.person_id = person_id
        self.initial_track_id = track_id
        self.person_dir = os.path.join(output_base_dir, person_id)
        self.crops_dir = os.path.join(self.person_dir, "crops")
        self.embeddings_dir = os.path.join(self.person_dir, "embeddings")
        self.metadata_path = os.path.join(self.person_dir, "metadata.json")

        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        now_str = datetime.now(timezone.utc).isoformat()
        self.first_seen = now_str
        self.last_seen = now_str
        self.last_saved_time = 0.0
        self.crop_count = 0
        self.embedding_count = 0
        self.embeddings = []  # List of 1D numpy arrays

        self.save_metadata()

    def save_metadata(self):
        """Updates metadata.json for this person record."""
        data = {
            "person_id": self.person_id,
            "initial_track_id": self.initial_track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "crop_count": self.crop_count,
            "embedding_count": self.embedding_count,
            "camera_id": CAMERA_ID
        }
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Metadata update error for {self.person_id}: {e}")

    def add_observation(self, crop_img, embedding):
        """Saves a crop image and feature embedding."""
        import cv2
        self.crop_count += 1
        self.embedding_count += 1

        crop_name = f"crop_{self.crop_count:04d}.jpg"
        cv2.imwrite(os.path.join(self.crops_dir, crop_name), crop_img)

        emb_name = f"emb_{self.embedding_count:04d}.npy"
        np.save(os.path.join(self.embeddings_dir, emb_name), embedding)

        self.embeddings.append(embedding)
        if len(self.embeddings) > MAX_EMBEDDINGS_PER_PERSON:
            self.embeddings.pop(0)

        now_str = datetime.now(timezone.utc).isoformat()
        self.last_seen = now_str
        self.last_saved_time = time.time()
        self.save_metadata()

    def compute_similarity(self, query_embedding):
        """Computes top-k average Cosine Similarity against stored embeddings."""
        if not self.embeddings:
            return 0.0
        sims = [float(np.dot(query_embedding, stored_emb)) for stored_emb in self.embeddings]
        sims.sort(reverse=True)
        top_k = sims[:min(3, len(sims))]
        return float(np.mean(top_k))


def perform_global_data_association(active_track_ids, current_embeddings, person_records, track_states):
    """
    Performs 1-to-1 Global Data Association (Hungarian Algorithm)
    to resolve identity conflicts when multiple tracks are active simultaneously.
    """
    if not active_track_ids or not current_embeddings:
        return

    num_tracks = len(active_track_ids)
    num_records = len(person_records)

    # If no existing Person Records, all unconfirmed tracks increment eval count and register as NEW
    if num_records == 0:
        if DEBUG_MODE:
            print(f"[RE-ID MATRIX] Initializing first Person Record. Active Tracks: {active_track_ids}")
        for track_id in active_track_ids:
            state = track_states[track_id]
            if not state["confirmed"]:
                state["eval_count"] += 1
                if state["eval_count"] >= REID_CONFIRMATION_COUNT:
                    state["pending_new"] = True
        return

    # Build Similarity Matrix S[num_tracks][num_records]
    sim_matrix = np.zeros((num_tracks, num_records))
    for t_idx, track_id in enumerate(active_track_ids):
        emb = current_embeddings[t_idx]
        if emb is not None:
            for r_idx, record in enumerate(person_records):
                sim_matrix[t_idx, r_idx] = record.compute_similarity(emb)

    # Print Debug Matching Table to console
    if DEBUG_MODE:
        print("\n==================================================")
        print(" RE-ID CANDIDATE MATCHING MATRIX (DEBUG)")
        print("==================================================")
        for t_idx, track_id in enumerate(active_track_ids):
            print(f"Track {track_id}:")
            for r_idx, record in enumerate(person_records):
                sim_score = sim_matrix[t_idx, r_idx]
                print(f"  {record.person_id} = {sim_score:.2f}")
        print("--------------------------------------------------")

    # Global Hungarian Assignment (Cost Matrix = -Similarity)
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Process Global 1-to-1 Assignments
    assigned_tracks = set()
    assigned_records = set()

    for t_idx, r_idx in zip(row_ind, col_ind):
        track_id = active_track_ids[t_idx]
        record = person_records[r_idx]
        sim_score = sim_matrix[t_idx, r_idx]
        state = track_states[track_id]

        state["last_similarity"] = sim_score

        if sim_score >= REID_SIMILARITY_THRESHOLD:
            assigned_tracks.add(track_id)
            assigned_records.add(record.person_id)
            target_pid = record.person_id
            state["candidate_pid"] = target_pid

            if not state["confirmed"]:
                state["eval_count"] += 1
                state["pending_matches"][target_pid] = state["pending_matches"].get(target_pid, 0) + 1

                # Fast 2-frame confirmation for existing record match
                if state["pending_matches"][target_pid] >= 2:
                    state["person_record"] = record
                    state["confirmed"] = True
                    state["match_status"] = f"MATCH ({sim_score * 100:.0f}%)"
                    print(f"[GLOBAL MATCH] Track {track_id} assigned to {target_pid} (Sim: {sim_score:.2f})")
        else:
            state["candidate_pid"] = None

    # Handle Unassigned Tracks (Create NEW Person Records only after 8 unassigned observations)
    for t_idx, track_id in enumerate(active_track_ids):
        state = track_states[track_id]
        if not state["confirmed"] and track_id not in assigned_tracks:
            state["eval_count"] += 1
            if state["eval_count"] >= REID_CONFIRMATION_COUNT:
                # Registered as New Person Record
                state["pending_new"] = True


def run_person_reid(
    model_name="models/yolov8n.pt",
    camera_index=0,
    output_dir="outputs/persons",
    target_class=0
):
    """
    Runs real-time Person Detection + ByteTrack + Global Re-ID Data Association.
    """
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU Mode"

    print("==================================================")
    print(" PERSON RE-ID PIPELINE (Global Data Association)")
    print("==================================================")
    print(f"[INFO] Device:               {device_label}")
    print(f"[INFO] Detection Model:      {model_name}")
    print(f"[INFO] Re-ID Architecture:   OSNet (osnet_x0_25)")
    print(f"[INFO] Data Association:     Hungarian 1-to-1 Global Matching")
    print(f"[INFO] Similarity Threshold:  {REID_SIMILARITY_THRESHOLD}")
    print(f"[INFO] Confirmation Count:   {REID_CONFIRMATION_COUNT} observations")
    print("==================================================")

    try:
        model = YOLO(model_name)
        extractor = OSNetReIDExtractor(device=device)
    except Exception as e:
        print(f"[ERROR] Failed initializing models: {e}")
        return False

    # Open Camera Stream using DirectShow backend to prevent MSMF hardware lock
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        return False

    # Configure hardware MJPEG video stream & 640x480 resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Sensor warm-up: read 15 frames to allow auto-exposure to adjust to room lighting
    print("[INFO] Warming up camera hardware sensor...")
    for _ in range(15):
        cap.read()

    os.makedirs(output_dir, exist_ok=True)

    person_record_counter = 0
    person_records = []  # List of PersonRecordReID instances

    # Map: track_id -> state dict
    track_states = {}

    window_name = "Computer Vision AI - Multi-Person Re-ID (Global Matching)"
    print("[SUCCESS] Multi-Person Re-ID pipeline operational!")
    print("[INFO] Press 'Q' or 'q' to exit.")

    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0
    empty_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                empty_frame_count += 1
                if empty_frame_count <= 10:
                    time.sleep(0.03)
                    continue  # Retry intermittent camera frame drop
                else:
                    print("[WARNING] Exceeded maximum empty frame retries. Closing stream.")
                    break

            # Reset empty frame counter on valid frame
            empty_frame_count = 0

            # Create clean copy for UI rendering so PyTorch transforms don't mutate display buffer
            display_frame = frame.copy()
            frame_h, frame_w = display_frame.shape[:2]
            current_time = time.time()

            frame_count += 1

            # Run YOLO + ByteTrack
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[target_class],
                device=device,
                verbose=False,
                conf=0.35,
                imgsz=320  # Fast CPU inference size
            )

            result = results[0]
            active_track_ids = []
            current_embeddings = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                has_id = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_id else [None] * len(boxes)
                confidences = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                # Step 1: Collect crops & extract embeddings for all active tracks
                for xyxy, conf, track_id in zip(xyxys, confidences, track_ids):
                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = map(int, xyxy)
                    x1_c = max(0, min(x1, frame_w - 1))
                    y1_c = max(0, min(y1, frame_h - 1))
                    x2_c = max(0, min(x2, frame_w))
                    y2_c = max(0, min(y2, frame_h))

                    crop_w = x2_c - x1_c
                    crop_h = y2_c - y1_c

                    if track_id not in track_states:
                        track_states[track_id] = {
                            "person_record": None,
                            "confirmed": False,
                            "pending_matches": {},
                            "pending_new": False,
                            "eval_count": 0,
                            "last_eval_time": 0.0,
                            "last_similarity": 0.0,
                            "candidate_pid": None,
                            "match_status": "EVALUATING"
                        }

                    state = track_states[track_id]
                    emb = None

                    # Crop & Feature Extraction
                    if (
                        crop_w >= MIN_CROP_WIDTH and
                        crop_h >= MIN_CROP_HEIGHT and
                        conf >= CONFIDENCE_THRESHOLD and
                        (current_time - state["last_eval_time"]) >= SAVE_INTERVAL_SECONDS
                    ):
                        crop_bgr = frame[y1_c:y2_c, x1_c:x2_c]
                        if crop_bgr.size > 0:
                            state["last_eval_time"] = current_time
                            emb = extractor.extract_from_bgr(crop_bgr)
                            state["last_crop_bgr"] = crop_bgr
                            state["last_emb"] = emb

                    active_track_ids.append(track_id)
                    current_embeddings.append(state.get("last_emb", None))

                # Step 2: Perform 1-to-1 Global Data Association (Hungarian Algorithm)
                eval_embeddings = [e for e in current_embeddings if e is not None]
                eval_tracks = [t for t, e in zip(active_track_ids, current_embeddings) if e is not None]

                if eval_tracks:
                    perform_global_data_association(eval_tracks, eval_embeddings, person_records, track_states)

                # Step 3: Finalize records and render UI overlay for each track
                for xyxy, conf, track_id in zip(xyxys, confidences, track_ids):
                    if track_id is None:
                        continue

                    state = track_states[track_id]
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Create New Person Record if confirmed new
                    if state.get("pending_new", False) and not state["confirmed"]:
                        person_record_counter += 1
                        new_pid = f"PERSON-{person_record_counter:04d}"
                        new_record = PersonRecordReID(new_pid, track_id, output_dir)
                        person_records.append(new_record)

                        state["person_record"] = new_record
                        state["confirmed"] = True
                        state["match_status"] = "NEW PERSON"
                        state["pending_new"] = False
                        print(f"[RE-ID NEW RECORD] Created {new_pid} for Track {track_id}")

                    # If confirmed, append observation crop & embedding whenever a new sample is ready
                    if state["confirmed"] and state["person_record"] is not None:
                        if state.get("last_crop_bgr", None) is not None and state.get("last_emb", None) is not None:
                            state["person_record"].add_observation(state["last_crop_bgr"], state["last_emb"])
                            state["last_crop_bgr"] = None
                            state["last_emb"] = None

                    # Render UI Overlay
                    if state["person_record"] is not None:
                        pid_label = state["person_record"].person_id
                    else:
                        cand = state["candidate_pid"] if state["candidate_pid"] else "Evaluating"
                        pid_label = f"Cand: {cand}"

                    status_str = state["match_status"]
                    sim_str = f"Sim: {state['last_similarity'] * 100:.0f}%" if state["last_similarity"] > 0 else "Sim: N/A"

                    if "MATCH" in status_str:
                        box_color = (0, 255, 127)  # Emerald Green
                    elif "NEW" in status_str:
                        box_color = (0, 215, 255)  # Gold
                    else:
                        box_color = (0, 165, 255)  # Orange

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                    line1 = f"{pid_label} | Track: {track_id}"
                    line2 = f"Re-ID: {status_str} | {sim_str}"

                    (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    max_w = max(tw1, tw2) + 10
                    total_h = th1 + th2 + 12

                    lbl_y1 = max(0, y1 - total_h)
                    cv2.rectangle(display_frame, (x1, lbl_y1), (x1 + max_w, lbl_y1 + total_h), box_color, -1)

                    cv2.putText(display_frame, line1, (x1 + 5, lbl_y1 + th1 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, line2, (x1 + 5, lbl_y1 + th1 + th2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            header = f"Device: {device.upper()} | FPS: {fps:.1f} | Active Tracks: {len(active_track_ids)} | Records: {len(person_records)}"
            cv2.rectangle(display_frame, (10, 10), (660, 45), (30, 30, 30), -1)
            cv2.putText(display_frame, header, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            # Check if user clicked window [X] close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window [X] closed by user.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27, 32):  # 'q', 'Q', ESC, Spacebar
                print("[INFO] Exit key pressed. Exiting Multi-Person Re-ID demo...")
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
    run_person_reid()
