"""
Live Camera Feed Screen for PySide6 Application
Supports instant Black Screen on Camera Stop, Fast Transformation & Responsive Click Signals
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
from app.services.alarm_service import alarm_service


class LiveCameraPage(QWidget):
    camera_toggle_requested = Signal(bool)  # True = Start, False = Stop

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_active = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Control Header
        ctrl_layout = QHBoxLayout()

        self.btn_toggle = QPushButton("⏹ Stop Camera Feed")
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                padding: 10px 22px;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        self.btn_toggle.clicked.connect(self.toggle_camera_feed)

        # Top Header Alarm Mute Button
        self.btn_mute_alarm = QPushButton("🔕 Stop Siren Alarm")
        self.btn_mute_alarm.setCursor(Qt.PointingHandCursor)
        self.btn_mute_alarm.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                padding: 10px 18px;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
        """)
        self.btn_mute_alarm.clicked.connect(self.stop_alarm_siren)

        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")

        self.lbl_tracks = QLabel("Active Tracks: 0 (Known: 0 | Unknown: 0)")
        self.lbl_tracks.setStyleSheet("font-size: 14px; font-weight: bold; color: #f59e0b;")

        ctrl_layout.addWidget(self.btn_toggle)
        ctrl_layout.addWidget(self.btn_mute_alarm)
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
        self.lbl_video.setStyleSheet("color: #94a3b8; font-size: 18px; font-weight: bold; background-color: #000000;")
        vbox_video.addWidget(self.lbl_video)

        layout.addWidget(self.video_frame, 1)

    def toggle_camera_feed(self):
        self.camera_active = not self.camera_active
        if not self.camera_active:
            self.btn_toggle.setText("▶ Start Camera Feed")
            self.btn_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    padding: 10px 22px;
                    font-weight: bold;
                    font-size: 13px;
                    border: none;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
            self.clear_video_feed()
            self.camera_toggle_requested.emit(False)
        else:
            self.btn_toggle.setText("⏹ Stop Camera Feed")
            self.btn_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    padding: 10px 22px;
                    font-weight: bold;
                    font-size: 13px;
                    border: none;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
            """)
            self.lbl_video.setText("Resuming Video Feed...")
            self.camera_toggle_requested.emit(True)

    def stop_alarm_siren(self):
        alarm_service.stop_alarm()

    def clear_video_feed(self):
        self.lbl_video.clear()
        self.lbl_video.setText("📷 Camera Feed Stopped")
        self.lbl_video.setStyleSheet("color: #64748b; font-size: 18px; font-weight: bold; background-color: #000000;")
        self.lbl_fps.setText("FPS: 0.0")
        self.lbl_tracks.setText("Active Tracks: 0 (Known: 0 | Unknown: 0)")

    def update_frame(self, qt_image: QImage, active_tracks: int, known_cnt: int, unknown_cnt: int, fps: float):
        if not self.camera_active:
            return

        pixmap = QPixmap.fromImage(qt_image)
        target_size = self.lbl_video.size()
        if target_size.width() > 10 and target_size.height() > 10:
            # FastTransformation for zero CPU lag on main GUI thread
            scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.lbl_video.setPixmap(scaled)

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_tracks.setText(f"Active Tracks: {active_tracks} (Known: {known_cnt} | Unknown: {unknown_cnt})")
