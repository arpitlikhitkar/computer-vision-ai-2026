"""
Real-Time Person Crop & Individual Track Recording (Phase 3)

CONCEPTUAL PIPELINE:
Camera Stream -> YOLO Detection -> ByteTrack -> Track ID -> Person Crop -> Person Record (PERSON-XXXX) -> Local Storage

CONCEPTS & DISTINCTIONS:
- Track ID: Assigned by ByteTrack temporarily during continuous visibility.
- Person ID: Session record folder identifier (e.g., PERSON-0001, PERSON-0002).
- Frame Sampling: Saving crops at timestamp intervals (e.g., 1.0 second) rather than every video frame.
- Metadata: JSON record tracking first_seen, last_seen, crop_count, and track_id.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone

# Configurable Parameters
SAVE_INTERVAL_SECONDS = 1.0   # Save at most 1 crop per second per person
CONFIDENCE_THRESHOLD = 0.50   # Minimum confidence score to record crop
MIN_BOX_WIDTH = 40            # Minimum pixel width of crop
MIN_BOX_HEIGHT = 80           # Minimum pixel height of crop
CAMERA_ID = "WEBCAM-01"       # Identifier for video source

class PersonRecord:
    """
    Manages individual person folder, crop files, and metadata JSON.
    """
    def __init__(self, person_id, track_id, output_base_dir):
        self.person_id = person_id
        self.track_id = track_id
        self.person_dir = os.path.join(output_base_dir, person_id)
        self.crops_dir = os.path.join(self.person_dir, "crops")
        self.metadata_path = os.path.join(self.person_dir, "metadata.json")
        
        # Ensure directories exist
        os.makedirs(self.crops_dir, exist_ok=True)
        
        now_str = datetime.now(timezone.utc).isoformat()
        self.first_seen = now_str
        self.last_seen = now_str
        self.last_saved_time = 0.0
        self.crop_count = 0
        
        # Write initial metadata file
        self.save_metadata()

    def save_metadata(self):
        """Writes or updates metadata.json for this person."""
        data = {
            "person_id": self.person_id,
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "crop_count": self.crop_count,
            "camera_id": CAMERA_ID
        }
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to update metadata for {self.person_id}: {e}")

    def add_crop(self, crop_img):
        """Saves a cropped image and updates metadata."""
        import cv2
        self.crop_count += 1
        crop_filename = f"crop_{self.crop_count:04d}.jpg"
        crop_filepath = os.path.join(self.crops_dir, crop_filename)
        
        # Save image crop
        cv2.imwrite(crop_filepath, crop_img)
        
        # Update last seen and saved timestamps
        now_str = datetime.now(timezone.utc).isoformat()
        self.last_seen = now_str
        self.last_saved_time = time.time()
        
        # Persist updated metadata
        self.save_metadata()


def run_person_recording(
    model_name="models/yolov8n.pt",
    camera_index=0,
    output_dir="outputs/persons",
    target_class=0
):
    """
    Runs real-time person tracking and saves individual crop records for every tracked person.
    """
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing required library: {e}")
        print("Please activate your virtual environment (.venv) and install dependencies.")
        return False

    # Determine Compute Device
    if torch.cuda.is_available():
        device = "cuda"
        device_label = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU Mode"

    print("==================================================")
    print(" Computer Vision AI - Phase 3 Person Recording")
    print("==================================================")
    print(f"[INFO] Device: {device_label}")
    print(f"[INFO] Loading Model: {model_name}")
    print(f"[INFO] Output Directory: {output_dir}")
    print(f"[INFO] Save Interval: {SAVE_INTERVAL_SECONDS} sec/crop")
    print("==================================================")

    # Load YOLO Model
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model '{model_name}': {e}")
        return False

    # Open Camera Stream using DirectShow backend to prevent MSMF hardware lock
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        return False

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(15):
        cap.read()

    # Clean Output Directory for Current Recording Session
    os.makedirs(output_dir, exist_ok=True)

    # State tracking data structures
    person_counter = 0
    # Map: track_id -> PersonRecord object
    tracked_persons = {}

    window_name = "Computer Vision AI - Person Crop & Track Recording"
    print("[SUCCESS] Camera feed active. Recording started!")
    print("[INFO] Press 'Q' or 'q' in the camera window to exit.")

    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from webcam stream.")
                break

            frame_count += 1
            frame_h, frame_w = frame.shape[:2]

            # Run YOLO Tracking with ByteTrack
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[target_class],
                device=device,
                verbose=False,
                conf=0.35
            )

            result = results[0]
            current_active_count = 0
            current_time = time.time()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                has_id = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_id else [None] * len(boxes)
                confidences = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for xyxy, conf, track_id in zip(xyxys, confidences, track_ids):
                    if track_id is None:
                        continue  # Skip unassigned tracks until ByteTrack confirms ID

                    current_active_count += 1

                    # Register new Person ID for newly discovered Track ID
                    if track_id not in tracked_persons:
                        person_counter += 1
                        person_id = f"PERSON-{person_counter:04d}"
                        tracked_persons[track_id] = PersonRecord(person_id, track_id, output_dir)
                        print(f"[NEW RECORD] Assigned {person_id} to Track ID {track_id}")

                    person_record = tracked_persons[track_id]
                    person_record.last_seen = datetime.now(timezone.utc).isoformat()

                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Clamp coordinates within image frame boundaries
                    x1_clamped = max(0, min(x1, frame_w - 1))
                    y1_clamped = max(0, min(y1, frame_h - 1))
                    x2_clamped = max(0, min(x2, frame_w))
                    y2_clamped = max(0, min(y2, frame_h))

                    crop_w = x2_clamped - x1_clamped
                    crop_h = y2_clamped - y1_clamped

                    # Quality Filter & Timestamp Sampling
                    if (
                        crop_w >= MIN_BOX_WIDTH and
                        crop_h >= MIN_BOX_HEIGHT and
                        conf >= CONFIDENCE_THRESHOLD and
                        (current_time - person_record.last_saved_time) >= SAVE_INTERVAL_SECONDS
                    ):
                        crop_img = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                        if crop_img.size > 0:
                            person_record.add_crop(crop_img)

                    # UI Drawing (Bounding Box + Labels)
                    box_color = (0, 215, 255)  # Gold/Cyan color for recording mode
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    header_text = f"{person_record.person_id} | Track: {track_id}"
                    sub_text = f"Conf: {conf * 100:.0f}% | Crops: {person_record.crop_count}"

                    # Text Background Container
                    (tw1, th1), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    max_w = max(tw1, tw2) + 10
                    total_h = th1 + th2 + 12

                    lbl_y1 = max(0, y1 - total_h)
                    cv2.rectangle(frame, (x1, lbl_y1), (x1 + max_w, lbl_y1 + total_h), box_color, -1)

                    # Render Header & Sub-text
                    cv2.putText(
                        frame,
                        header_text,
                        (x1 + 5, lbl_y1 + th1 + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        frame,
                        sub_text,
                        (x1 + 5, lbl_y1 + th1 + th2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )

            # Calculate FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Display Status Bar Header
            status_text = (
                f"Device: {device.upper()} | FPS: {fps:.1f} | "
                f"Active Tracks: {current_active_count} | Total Persons: {len(tracked_persons)}"
            )
            cv2.rectangle(frame, (10, 10), (620, 45), (30, 30, 30), -1)
            cv2.putText(
                frame,
                status_text,
                (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Display Live Frame
            cv2.imshow(window_name, frame)

            # Press Q to exit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[INFO] 'Q' key pressed. Stopping recording session...")
                break

    except KeyboardInterrupt:
        print("[INFO] Session interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Runtime exception: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera stream closed and windows destroyed cleanly.")

    return True

if __name__ == "__main__":
    run_person_recording()
