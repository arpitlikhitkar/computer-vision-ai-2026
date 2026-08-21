"""
Unknown People Detections Management Page for PySide6 Application
Displays unique folder per Unknown Person + Video Clip Playback + Open Folder Button
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QGridLayout, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices

from app.database.unknown_repository import UnknownRepository


class UnknownPage(QWidget):
    enroll_requested = Signal(str)

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
        title = QLabel("❓ Unknown Detections & Unique Person Folders")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        btn_refresh = QPushButton("🔄 Refresh List")
        btn_refresh.setObjectName("secondaryBtn")
        btn_refresh.clicked.connect(self.load_unknowns)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(btn_refresh)

        layout.addLayout(top_layout)

        # Scroll Area for Unknown Cards
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: transparent; border: none;")

        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setSpacing(15)

        self.scroll.setWidget(self.cards_container)
        layout.addWidget(self.scroll, 1)

        self.load_unknowns()

    def load_unknowns(self):
        for i in reversed(range(self.cards_layout.count())):
            widget = self.cards_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        unknowns = self.unknown_repo.get_all_unknowns(status_filter="PENDING")

        if not unknowns:
            lbl_empty = QLabel("No pending unknown detections.")
            lbl_empty.setStyleSheet("color: #94a3b8; font-size: 14px;")
            self.cards_layout.addWidget(lbl_empty, 0, 0)
            return

        col_max = 3
        for idx, u in enumerate(unknowns):
            row = idx // col_max
            col = idx % col_max
            card = self.create_unknown_card(u)
            self.cards_layout.addWidget(card, row, col)

    def create_unknown_card(self, u_data):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #ef4444;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        vbox = QVBoxLayout(card)

        # Image preview
        lbl_img = QLabel()
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("background-color: #0f172a; border-radius: 6px; min-height: 140px;")

        snap_path = u_data["snapshot_path"]
        if os.path.exists(snap_path):
            lbl_img.setPixmap(QPixmap(snap_path).scaled(200, 140, Qt.KeepAspectRatio))
        else:
            lbl_img.setText("Snapshot N/A")

        vbox.addWidget(lbl_img)

        # Determine unique folder path
        folder_dir = os.path.dirname(snap_path)
        folder_name = os.path.basename(folder_dir)

        has_video = u_data.get("video_clip_path") and os.path.exists(u_data["video_clip_path"])
        video_badge = "🎥 60s Video Clip Ready" if has_video else "📷 Snapshot Only"

        lbl_info = QLabel(
            f"<b>Unknown Track #{u_data['track_id']}</b> (ID: #{u_data['id']})<br>"
            f"📁 Folder: <code>{folder_name}</code><br>"
            f"Date: {u_data['created_at'][:10]} | Time: {u_data['created_at'][11:19]}<br>"
            f"Best Sim: {u_data['best_similarity']*100:.0f}%<br>"
            f"<span style='color: #10b981; font-weight: bold;'>{video_badge}</span>"
        )
        lbl_info.setStyleSheet("color: #f8fafc; font-size: 12px;")
        vbox.addWidget(lbl_info)

        # Media Action Buttons
        media_btn_layout = QHBoxLayout()
        if has_video:
            btn_play = QPushButton("▶ Play 60s Video")
            btn_play.setStyleSheet("background-color: #10b981; color: white; font-weight: bold;")
            btn_play.clicked.connect(lambda _, vp=u_data["video_clip_path"]: self.play_video_clip(vp))
            media_btn_layout.addWidget(btn_play)

        btn_folder = QPushButton("📁 Open Folder")
        btn_folder.setStyleSheet("background-color: #38bdf8; color: white; font-weight: bold;")
        btn_folder.clicked.connect(lambda _, fd=folder_dir: self.open_person_folder(fd))
        media_btn_layout.addWidget(btn_folder)

        vbox.addLayout(media_btn_layout)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_enroll = QPushButton("✨ Enroll Person")
        btn_enroll.clicked.connect(lambda _, uid=u_data["id"]: self.enroll_unknown(uid))

        btn_ignore = QPushButton("Ignore")
        btn_ignore.setObjectName("secondaryBtn")
        btn_ignore.clicked.connect(lambda _, uid=u_data["id"]: self.ignore_unknown(uid))

        btn_layout.addWidget(btn_enroll)
        btn_layout.addWidget(btn_ignore)

        vbox.addLayout(btn_layout)
        return card

    def play_video_clip(self, video_path):
        if os.path.exists(video_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))
        else:
            QMessageBox.warning(self, "File Not Found", f"Video clip file not found: {video_path}")

    def open_person_folder(self, folder_dir):
        if os.path.exists(folder_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_dir))
        else:
            QMessageBox.warning(self, "Folder Not Found", f"Folder not found: {folder_dir}")

    def enroll_unknown(self, unknown_id):
        name, ok = QInputDialog.getText(
            self, "Enroll Unknown Person",
            "Enter full name for this person to start enrollment wizard:"
        )
        if ok and name.strip():
            self.unknown_repo.update_status(unknown_id, "ENROLLED")
            self.enroll_requested.emit(name.strip())
            self.load_unknowns()

    def ignore_unknown(self, unknown_id):
        self.unknown_repo.update_status(unknown_id, "IGNORED")
        self.load_unknowns()
