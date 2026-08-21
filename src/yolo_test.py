"""
YOLO Webcam Person Detection Script for Computer Vision & AI Setup

This script runs live object/person detection using a free pretrained YOLO model.

DEVICE SELECTION ARCHITECTURE:
- CPU Mode: Python -> PyTorch -> CPU -> YOLO Model (Inference)
- GPU Mode: Python -> PyTorch -> CUDA -> NVIDIA GPU -> YOLO Model (Inference)

Current Mode: Automatically selected based on PyTorch torch.cuda.is_available()
"""

import sys
import time

def run_yolo_detection(model_name="models/yolov8n.pt", camera_index=0, target_class=0):
    """
    Runs YOLO object detection on live camera stream.
    
    Parameters:
    - model_name: Pretrained YOLO weight file (default 'models/yolov8n.pt')
    - camera_index: Webcam device index (default 0)
    - target_class: COCO class index to highlight (0 is 'person')
    """
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing required library: {e}")
        print("Ensure you have activated .venv and installed dependencies.")
        return False

    # Determine Compute Device
    # PyTorch automatically routes tensor computations to GPU if CUDA is present,
    # otherwise defaults to CPU.
    if torch.cuda.is_available():
        device = "cuda"
        device_label = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_label = "CPU Mode"

    print("==================================================")
    print(f"[INFO] Initializing YOLO Model: {model_name}")
    print(f"[INFO] Compute Device: {device_label}")
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
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        return False

    window_name = "Computer Vision AI - Pretrained YOLO Person Detection"
    print(f"[SUCCESS] Camera stream started.")
    print("[INFO] Press 'Q' or 'q' in the display window to exit.")

    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from camera stream.")
                break

            frame_count += 1

            # Run YOLO Inference on the frame using selected device
            # verbose=False suppresses frame-by-frame console logging
            results = model.predict(source=frame, device=device, verbose=False, conf=0.35)
            
            # Extract detection results for frame 0
            result = results[0]
            
            person_count = 0

            # Draw Custom Bounding Boxes for Person Detections (COCO Class 0 = Person)
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id == target_class:
                    person_count += 1
                    # Extract bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Draw vibrant box for detected person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)

                    # Label text: Person + confidence percentage
                    label = f"Person {conf * 100:.1f}%"
                    
                    # Background box for text clarity
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 127), -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 3, y1 - 4),
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
            overlay_text = f"Device: {device.upper()} | FPS: {fps:.1f} | Persons Detected: {person_count}"
            cv2.rectangle(frame, (10, 10), (550, 45), (30, 30, 30), -1)
            cv2.putText(
                frame,
                overlay_text,
                (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Display updated frame
            cv2.imshow(window_name, frame)

            # Press Q to exit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[INFO] Exiting YOLO detection...")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Stream closed and resources cleaned up.")

    return True

if __name__ == "__main__":
    run_yolo_detection()
