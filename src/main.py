"""
Main Entry Point for computer-vision-ai

Allows running environment check, camera test, or YOLO webcam person detection.
"""

import sys
import argparse
from src.check_environment import check_env
from src.camera_test import test_camera
from src.yolo_test import run_yolo_detection
from src.person_tracking import run_person_tracking
from src.person_recording import run_person_recording
from src.person_reid import run_person_reid

def main():
    parser = argparse.ArgumentParser(description="Computer Vision & AI Setup Runner")
    parser.add_argument(
        "--mode",
        choices=["check", "camera", "yolo", "tracking", "recording", "reid"],
        default="check",
        help="Mode to run: 'check', 'camera', 'yolo', 'tracking', 'recording', 'reid' (person re-identification)"
    )
    args = parser.parse_args()

    if args.mode == "check":
        check_env()
    elif args.mode == "camera":
        test_camera()
    elif args.mode == "yolo":
        run_yolo_detection()
    elif args.mode == "tracking":
        run_person_tracking()
    elif args.mode == "recording":
        run_person_recording()
    elif args.mode == "reid":
        run_person_reid()

if __name__ == "__main__":
    main()
