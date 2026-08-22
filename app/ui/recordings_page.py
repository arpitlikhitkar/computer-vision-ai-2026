"""
Recordings Module Page for PySide6 Application
Lists and plays event video clips with Human-Readable Date/Time & Exact Video Duration in Seconds!
"""

import os
import datetime
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from app.config.settings import config
from app.database.unknown_repository import UnknownRepository


class RecordingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.unknown_repo = UnknownRepository()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        top_layout = QHBoxLayout()
        title = QLabel("📹 Event Video Recordings (-30s Pre + +30s Post Clips)")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        self.btn_refresh = QPushButton("🔄 Refresh Recordings")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.setObjectName("secondaryBtn")
        self.btn_refresh.clicked.connect(self.load_recordings)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_refresh)

        layout.addLayout(top_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "File Name", "Event Type", "Recorded Date & Time", "Video Duration & Size", "Playback & Actions"
        ])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)

        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 175)
        self.table.setColumnWidth(3, 165)
        self.table.setColumnWidth(4, 280)

        self.table.verticalHeader().setDefaultSectionSize(48)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.table)

        self.load_recordings()

    def load_recordings(self):
        recordings_dir = os.path.join(config.DATA_DIR, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)

        files = [f for f in os.listdir(recordings_dir) if f.endswith(".mp4") or f.endswith(".avi")]
        files.sort(reverse=True)

        self.table.setRowCount(0)

        for row_idx, fname in enumerate(files):
            self.table.insertRow(row_idx)
            full_path = os.path.join(recordings_dir, fname)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)

            # 1. Format Human-Readable File Modification Date & Time
            mtime = os.path.getmtime(full_path)
            dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            # 2. Extract Exact Video Duration in Seconds
            duration_sec = 60.0
            try:
                cap = cv2.VideoCapture(full_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0 and frame_count > 0:
                        duration_sec = frame_count / fps
                cap.release()
            except Exception:
                duration_sec = 60.0

            item_name = QTableWidgetItem(fname)
            item_type = QTableWidgetItem("🔴 UNKNOWN EVENT")
            item_date = QTableWidgetItem(dt_str)
            item_size = QTableWidgetItem(f"⏱️ {duration_sec:.1f}s ({size_mb:.2f} MB)")

            self.table.setItem(row_idx, 0, item_name)
            self.table.setItem(row_idx, 1, item_type)
            self.table.setItem(row_idx, 2, item_date)
            self.table.setItem(row_idx, 3, item_size)

            # Cell Action Widget
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(6, 4, 6, 4)
            actions_layout.setSpacing(8)

            btn_play = QPushButton(f"▶ Play {duration_sec:.0f}s Clip")
            btn_play.setCursor(Qt.PointingHandCursor)
            btn_play.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 5px 12px;
                    font-size: 12px;
                    font-weight: bold;
                    min-height: 28px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
            btn_play.clicked.connect(lambda _, p=full_path: self.play_video(p))

            btn_folder = QPushButton("📁 Open Folder")
            btn_folder.setCursor(Qt.PointingHandCursor)
            btn_folder.setStyleSheet("""
                QPushButton {
                    background-color: #38bdf8;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 5px 12px;
                    font-size: 12px;
                    font-weight: bold;
                    min-height: 28px;
                }
                QPushButton:hover {
                    background-color: #0284c7;
                }
            """)
            btn_folder.clicked.connect(lambda _, p=recordings_dir: self.open_folder(p))

            actions_layout.addWidget(btn_play)
            actions_layout.addWidget(btn_folder)

            self.table.setCellWidget(row_idx, 4, actions_widget)

    def play_video(self, video_path):
        if os.path.exists(video_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
        else:
            QMessageBox.warning(self, "File Not Found", f"Video clip file not found: {video_path}")

    def open_folder(self, folder_path):
        if os.path.exists(folder_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        else:
            QMessageBox.warning(self, "Folder Not Found", f"Folder path not found: {folder_path}")
