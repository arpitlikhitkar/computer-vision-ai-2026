"""
Live Camera Feed Screen for PySide6 Application
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage


class LiveCameraPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Control Header
        ctrl_layout = QHBoxLayout()

        self.btn_toggle = QPushButton("⏹ Stop Camera Feed")
        self.btn_toggle.setStyleSheet("background-color: #ef4444; color: white; padding: 10px 20px; font-weight: bold;")

        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")

        self.lbl_tracks = QLabel("Active Tracks: 0")
        self.lbl_tracks.setStyleSheet("font-size: 14px; font-weight: bold; color: #f59e0b;")

        ctrl_layout.addWidget(self.btn_toggle)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(self.lbl_fps)
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(self.lbl_tracks)
        ctrl_layout.addStretch()

        layout.addLayout(ctrl_layout)

        # Video Display Container Frame
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border: 2px solid #334155;
                border-radius: 12px;
            }
        """)
        vbox_video = QVBoxLayout(self.video_frame)
        vbox_video.setContentsMargins(0, 0, 0, 0)

        self.lbl_video = QLabel("Initializing Video Feed...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_video.setStyleSheet("color: #94a3b8; font-size: 16px; font-weight: bold;")
        vbox_video.addWidget(self.lbl_video)

        layout.addWidget(self.video_frame, 1)

    def update_frame(self, qt_image: QImage, active_tracks: int, known_cnt: int, unknown_cnt: int, fps: float):
        pixmap = QPixmap.fromImage(qt_image)
        target_size = self.lbl_video.size()
        if target_size.width() > 10 and target_size.height() > 10:
            scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_video.setPixmap(scaled)

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_tracks.setText(f"Active Tracks: {active_tracks} (Known: {known_cnt} | Unknown: {unknown_cnt})")
