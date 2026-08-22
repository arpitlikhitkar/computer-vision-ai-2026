"""
Camera Worker Thread Service for PySide6 UI
Updated with non-blocking stop() to eliminate GUI deadlocks and button freezes.
"""

import time
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from app.config.settings import config
from app.ai.recognition_engine import RecognitionEngine


class CameraWorker(QThread):
    # Signals emitted to Qt UI thread
    frame_processed = Signal(QImage, int, int, int, float)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.camera_index = config.camera_index
        self.engine = None

    def run(self):
        self.running = True
        self.status_changed.emit("Initializing AI Models...")

        try:
            self.engine = RecognitionEngine()
            self.status_changed.emit("RUNNING")
        except Exception as e:
            self.status_changed.emit(f"ERROR: {e}")
            self.running = False
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.status_changed.emit("CAMERA DISCONNECTED")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

        # Warmup
        for _ in range(5):
            if not self.running:
                break
            cap.read()

        fps_start = time.time()
        frame_cnt = 0
        fps = 0.0

        while self.running:
            ret, frame = cap.read()
            if not self.running:
                break

            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame_cnt += 1
            processed_frame, active_tracks, known_count, unknown_count = self.engine.process_frame(frame)

            if not self.running:
                break

            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_cnt / elapsed
                frame_cnt = 0
                fps_start = time.time()

            # Convert BGR frame to QImage for Qt UI rendering
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            self.frame_processed.emit(qt_img, active_tracks, known_count, unknown_count, fps)
            time.sleep(0.01)

        cap.release()
        self.status_changed.emit("STOPPED")

    def stop(self):
        """Non-blocking thread stop. Never blocks main GUI event loop."""
        self.running = False
