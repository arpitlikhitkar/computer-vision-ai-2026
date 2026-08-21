"""
Camera Test Script for Computer Vision & AI Setup

This script tests webcam access using OpenCV.
- Opens default camera (index 0).
- Displays live video stream.
- Exits on 'Q' keypress.
- Handles diagnostic errors gracefully.
"""

import sys
import time

def test_camera(camera_index=0):
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV is not installed in the current environment.")
        print("Please activate your virtual environment (.venv) and install dependencies.")
        return False

    print(f"[INFO] Attempting to open camera index {camera_index}...")
    
    # Initialize video capture device
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("--------------------------------------------------")
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        print("--------------------------------------------------")
        print("Possible Reasons & Troubleshooting:")
        print(" 1. Camera is currently in use by another application (Zoom, Teams, Browser, etc.).")
        print(" 2. Windows Privacy Settings blocking camera access to apps.")
        print("    -> Go to Settings -> Privacy & Security -> Camera -> Enable 'Let apps access your camera'.")
        print(" 3. Laptop hardware camera switch or cover is toggled off / disabled.")
        print(" 4. Incorrect camera index (try camera index 1 or 2 if multiple video devices exist).")
        print(" 5. Missing or outdated camera device drivers.")
        print("--------------------------------------------------")
        return False

    window_name = "Computer Vision AI - Camera Test"
    print(f"[SUCCESS] Camera opened successfully!")
    print("[INFO] Press 'Q' or 'q' in the camera window to exit.")
    
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to capture frame from webcam stream.")
                break
            
            # Calculate simple FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Overlay FPS counter on image
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "Press 'Q' to Exit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Display frame
            cv2.imshow(window_name, frame)

            # Check key press (Wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[INFO] 'Q' pressed. Exiting camera feed...")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Closing...")
    finally:
        # Proper resource cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows destroyed clean.")

    return True

if __name__ == "__main__":
    test_camera(camera_index=0)
