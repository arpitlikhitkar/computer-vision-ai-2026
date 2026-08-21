"""
Interactive Household Member Enrollment Script (Phase 5)

Captures 10 high-quality face samples per household member using webcam,
aligns faces using 5 facial landmarks, generates 128-d embeddings,
and stores them in SQLite database (household_ai.db).

Usage:
python src/enrollment/enroll_person.py
"""

import sys
import time
import cv2
import numpy as np
from src.config import settings
from src.enrollment.enrollment_manager import EnrollmentManager


PROMPT_INSTRUCTIONS = [
    "Look straight at the camera",
    "Turn head slightly to the left",
    "Turn head slightly to the right",
    "Tilt head slightly up",
    "Smile or change expression",
    "Look straight at the camera",
    "Move slightly closer",
    "Turn head slightly left",
    "Turn head slightly right",
    "Final sample: look straight"
]


def run_interactive_enrollment():
    print("==================================================")
    print(" HOUSEHOLD MEMBER ENROLLMENT SYSTEM (PHASE 5)")
    print("==================================================")

    display_name = input("Enter Member's Name (e.g. Rahul): ").strip()
    if not display_name:
        print("[ERROR] Name cannot be empty.")
        return False

    print(f"\n[INFO] Starting camera to capture {settings.ENROLLMENT_SAMPLE_COUNT} face samples for '{display_name}'...")
    print("[INFO] Follow on-screen instructions. Press 'Q' to abort.")

    manager = EnrollmentManager()

    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(settings.CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {settings.CAMERA_INDEX}.")
        return False

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warmup
    for _ in range(15):
        cap.read()

    window_name = f"Enrollment - {display_name}"
    collected_embeddings = []
    collected_quality_scores = []
    collected_chips = []

    last_capture_time = 0.0
    capture_interval = 1.0  # Seconds between sample captures

    try:
        while len(collected_embeddings) < settings.ENROLLMENT_SAMPLE_COUNT:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            display_frame = frame.copy()
            current_time = time.time()
            sample_idx = len(collected_embeddings)
            prompt = PROMPT_INSTRUCTIONS[sample_idx] if sample_idx < len(PROMPT_INSTRUCTIONS) else "Look at camera"

            # Run validation & extraction
            valid, chip, emb, msg = manager.validate_and_extract_sample(frame)

            # Draw Header Bar
            header_str = f"Enrolling: {display_name} | Progress: {sample_idx}/{settings.ENROLLMENT_SAMPLE_COUNT}"
            cv2.rectangle(display_frame, (10, 10), (630, 75), (30, 30, 30), -1)
            cv2.putText(display_frame, header_str, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Instruction: {prompt}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw Status Bar at Bottom
            status_color = (0, 255, 127) if valid else (0, 165, 255)
            cv2.rectangle(display_frame, (10, 440), (630, 470), (20, 20, 20), -1)
            cv2.putText(display_frame, f"Status: {msg}", (20, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

            # Capture sample if valid and interval passed
            if valid and (current_time - last_capture_time) >= capture_interval:
                last_capture_time = current_time
                collected_embeddings.append(emb)
                collected_quality_scores.append(90.0)
                if chip is not None:
                    collected_chips.append(chip)
                print(f"[SAMPLE CAPTURED {len(collected_embeddings)}/{settings.ENROLLMENT_SAMPLE_COUNT}] {msg}")

            cv2.imshow(window_name, display_frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Enrollment window closed by user.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] Enrollment aborted by user.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(collected_embeddings) >= settings.ENROLLMENT_SAMPLE_COUNT:
        person = manager.enroll_new_member(display_name, collected_embeddings, collected_quality_scores)
        print("\n==================================================")
        print(f" ENROLLMENT SUCCESSFUL FOR {display_name.upper()}")
        print(f" Display ID:  {person['display_id']}")
        print(f" UUID:        {person['person_uuid']}")
        print(f" Samples:     {len(collected_embeddings)} embeddings saved to database")
        print("==================================================")
        return True
    else:
        print("\n[WARNING] Enrollment incomplete. No database records created.")
        return False

if __name__ == "__main__":
    run_interactive_enrollment()
