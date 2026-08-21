"""
Real-Time Person Tracking Script for Computer Vision & AI Setup (Phase 2)

CONCEPTUAL FLOW:
Webcam Frame -> Pretrained YOLO -> Person Detections -> Tracker (ByteTrack) -> Track IDs -> Bounding Boxes -> Live Display

DISTINCTIONS & CONCEPTS:
- Object Detection: "Is there a person in this frame?" (Independent on every single frame)
- Object Tracking: "Is this detected person the same object from previous frames?" (Links frames over time)
- Track ID: A temporary unique integer identifier assigned by the tracker to a detected object while in view.
- Permanent Identity / Person Re-ID: Recognizing that a person returning after leaving is the same real-world identity.
  (NOT implemented in Phase 2; Track IDs are temporary per video session / continuous visibility).

TRACKER SELECTION:
- We use ByteTrack (tracker="bytetrack.yaml"), which is lightweight, efficient, and ideally suited for real-time tracking on webcam streams.
"""

import sys
import time

def run_person_tracking(model_name="models/yolov8n.pt", camera_index=0, target_class=0):
    """
    Runs real-time YOLO object tracking on live camera feed, filtered for Persons.
    
    Parameters:
    - model_name: Path to YOLO weights (default 'models/yolov8n.pt')
    - camera_index: Webcam device index (default 0)
    - target_class: COCO class ID to filter and track (0 is 'person')
    """
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing required library: {e}")
        print("Please activate your virtual environment (.venv) and install dependencies.")
        return False

    # Determine Computing Device (GPU if available, CPU otherwise)
    # CPU Mode: Python -> PyTorch -> CPU -> YOLO -> ByteTrack
    # GPU Mode: Python -> PyTorch -> CUDA -> NVIDIA GPU -> YOLO -> ByteTrack
    if torch.cuda.is_available():
        device = "cuda"
        device_label = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU Mode"

    print("==================================================")
    print(" Computer Vision AI - Phase 2 Person Tracking")
    print("==================================================")
    print(f"[INFO] Device: {device_label}")
    print(f"[INFO] Loading YOLO Model: {model_name}")
    print(f"[INFO] Selected Tracker: ByteTrack (bytetrack.yaml)")
    print("==================================================")

    # Load Pretrained YOLO Model
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model '{model_name}': {e}")
        return False

    # Open Camera Stream
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"--------------------------------------------------")
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        print("--------------------------------------------------")
        print("Possible Reasons:")
        print(" 1. Camera in use by another app (Zoom, Teams, Web Browser).")
        print(" 2. Windows camera permissions restricted.")
        print(" 3. Hardware privacy shutter closed.")
        print("--------------------------------------------------")
        return False

    window_name = "Computer Vision AI - Real-Time Person Tracking"
    print("[SUCCESS] Webcam initialized successfully!")
    print("[INFO] Press 'Q' or 'q' in the window to exit cleanly.")

    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to capture frame from webcam stream.")
                break

            frame_count += 1

            # Run YOLO Tracking
            # - source=frame: current webcam image
            # - persist=True: informs tracker that frames form a continuous video stream
            # - tracker="bytetrack.yaml": uses ByteTrack algorithm for multi-object tracking
            # - classes=[target_class]: filters inference to Person class only
            # - conf=0.35: detection confidence threshold
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
            tracked_persons_count = 0

            # Process detections and track IDs
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                # Check if tracker IDs are assigned by ByteTrack
                has_id = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_id else [None] * len(boxes)
                confidences = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for xyxy, conf, track_id in zip(xyxys, confidences, track_ids):
                    tracked_persons_count += 1
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Format label string
                    # Show temporary Track ID if available, otherwise show 'Detecting...'
                    if track_id is not None:
                        id_str = f"Person ID: {track_id}"
                    else:
                        id_str = "Person ID: Pending"

                    conf_str = f"{conf * 100:.0f}%"
                    label = f"{id_str} | {conf_str}"

                    # Bounding Box Color (Emerald Green for Person Tracking)
                    box_color = (0, 255, 127)

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Background label container
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_bg_y1 = max(y1 - th - 10, 0)
                    cv2.rectangle(frame, (x1, label_bg_y1), (x1 + tw + 10, label_bg_y1 + th + 10), box_color, -1)
                    
                    # Draw label text (Black text on Emerald Green background)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, label_bg_y1 + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

            # Calculate FPS
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Status Overlay Header
            overlay = f"Device: {device.upper()} | FPS: {fps:.1f} | Active Tracks: {tracked_persons_count}"
            cv2.rectangle(frame, (10, 10), (560, 45), (30, 30, 30), -1)
            cv2.putText(
                frame,
                overlay,
                (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Display output frame
            cv2.imshow(window_name, frame)

            # Press Q to exit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[INFO] 'Q' pressed. Stopping person tracking stream...")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] An unexpected runtime error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released and windows closed cleanly.")

    return True

if __name__ == "__main__":
    run_person_tracking()
