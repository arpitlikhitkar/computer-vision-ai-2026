"""
Recordings Module Page for PySide6 Application
Lists and plays 60-second event video clips (-30s pre-event + +30s post-event)
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QMessageBox
)
from PySide6.QtCore import QUrl
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
        title = QLabel("📹 Event Video Recordings (-30s Pre + +30s Post)")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        btn_refresh = QPushButton("🔄 Refresh Recordings")
        btn_refresh.setObjectName("secondaryBtn")
        btn_refresh.clicked.connect(self.load_recordings)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(btn_refresh)

        layout.addLayout(top_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "File Name", "Event Type", "Date / Time", "File Size", "Playback"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
            mtime_str = os.path.pathsep.join([]) # transient

            item_name = QTableWidgetItem(fname)
            item_type = QTableWidgetItem("🔴 UNKNOWN EVENT (60s)")
            item_date = QTableWidgetItem(fname.split("_")[-1].replace(".mp4", ""))
            item_size = QTableWidgetItem(f"{size_mb:.2f} MB")

            self.table.setItem(row_idx, 0, item_name)
            self.table.setItem(row_idx, 1, item_type)
            self.table.setItem(row_idx, 2, item_date)
            self.table.setItem(row_idx, 3, item_size)

            btn_play = QPushButton("▶ Play 60s Video")
            btn_play.setStyleSheet("background-color: #10b981; color: white; font-weight: bold;")
            btn_play.clicked.connect(lambda _, p=full_path: self.play_video(p))

            self.table.setCellWidget(row_idx, 4, btn_play)

    def play_video(self, video_path):
        if os.path.exists(video_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
        else:
            QMessageBox.warning(self, "File Not Found", f"Video clip file not found: {video_path}")
