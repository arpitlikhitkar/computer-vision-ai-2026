"""
Dashboard Page for PySide6 Application
"""

import torch
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt
from app.database.person_repository import PersonRepository
from app.database.event_repository import EventRepository
from app.database.unknown_repository import UnknownRepository


class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.person_repo = PersonRepository()
        self.event_repo = EventRepository()
        self.unknown_repo = UnknownRepository()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("System Dashboard & Performance Overview")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        # Grid of Status Cards
        grid = QGridLayout()
        grid.setSpacing(15)

        # 1. Camera Status Card
        self.card_camera = self.create_card("Camera Status", "CONNECTED", "#10b981")
        grid.addWidget(self.card_camera, 0, 0)

        # 2. AI Engine Card
        self.card_ai = self.create_card("AI Engine", "RUNNING", "#10b981")
        grid.addWidget(self.card_ai, 0, 1)

        # 3. Hardware Device Card
        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU Mode"
        self.card_device = self.create_card("Device", device_str, "#818cf8")
        grid.addWidget(self.card_device, 0, 2)

        # 4. FPS Indicator
        self.card_fps = self.create_card("Pipeline FPS", "15.0 FPS", "#38bdf8")
        grid.addWidget(self.card_fps, 1, 0)

        # 5. Active Tracks
        self.card_tracks = self.create_card("Active Tracks", "0", "#f59e0b")
        grid.addWidget(self.card_tracks, 1, 1)

        # 6. Registered Members
        self.card_members = self.create_card("Registered Members", "0", "#10b981")
        grid.addWidget(self.card_members, 1, 2)

        # 7. Today's Events
        self.card_events = self.create_card("Today's Events", "0", "#a855f7")
        grid.addWidget(self.card_events, 2, 0)

        # 8. Pending Unknown Detections
        self.card_unknowns = self.create_card("Pending Unknowns", "0", "#ef4444")
        grid.addWidget(self.card_unknowns, 2, 1)

        layout.addLayout(grid)
        layout.addStretch()

        self.refresh_stats()

    def create_card(self, title_str, value_str, accent_color):
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet("""
            QFrame#card {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 16px;
            }
        """)

        vbox = QVBoxLayout(card)
        lbl_title = QLabel(title_str)
        lbl_title.setStyleSheet("font-size: 13px; color: #94a3b8; font-weight: 500;")

        lbl_val = QLabel(value_str)
        lbl_val.setObjectName("valueLabel")
        lbl_val.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {accent_color}; margin-top: 8px;")

        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        return card

    def update_live_metrics(self, active_tracks, known_cnt, unknown_cnt, fps):
        self.update_card_val(self.card_fps, f"{fps:.1f} FPS")
        self.update_card_val(self.card_tracks, str(active_tracks))

    def update_card_val(self, card, val_str):
        val_lbl = card.findChild(QLabel, "valueLabel")
        if val_lbl:
            val_lbl.setText(val_str)

    def refresh_stats(self):
        members = self.person_repo.get_all_persons()
        todays_events = self.event_repo.get_todays_count()
        unknowns = self.unknown_repo.get_all_unknowns(status_filter="PENDING")

        self.update_card_val(self.card_members, str(len(members)))
        self.update_card_val(self.card_events, str(todays_events))
        self.update_card_val(self.card_unknowns, str(len(unknowns)))
