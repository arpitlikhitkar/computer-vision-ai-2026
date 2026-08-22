"""
Events Timeline Page for PySide6 Application (Phase 6.9)
Displays structured event timeline with filters, priority color-coding & image previews
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QGridLayout, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices

from app.database.event_repository import EventRepository


class EventsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.event_repo = EventRepository()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        top_layout = QHBoxLayout()
        title = QLabel("📜 Structured AI Events & Multi-Entity Timeline")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["ALL", "PERSON_DETECTED", "HOLDING", "ANIMAL_DETECTED", "ALERT_SUSPICIOUS"])
        self.combo_filter.currentTextChanged.connect(self.load_events)

        btn_refresh = QPushButton("🔄 Refresh Timeline")
        btn_refresh.setObjectName("secondaryBtn")
        btn_refresh.clicked.connect(self.load_events)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(QLabel("Filter Type:"))
        top_layout.addWidget(self.combo_filter)
        top_layout.addWidget(btn_refresh)

        layout.addLayout(top_layout)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: transparent; border: none;")

        self.container = QWidget()
        self.vbox_events = QVBoxLayout(self.container)
        self.vbox_events.setSpacing(10)

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

        self.load_events()

    def load_events(self):
        for i in reversed(range(self.vbox_events.count())):
            w = self.vbox_events.itemAt(i).widget()
            if w:
                w.setParent(None)

        type_filter = self.combo_filter.currentText()
        events = self.event_repo.get_events(type_filter=type_filter, limit=50)

        if not events:
            lbl_empty = QLabel("No events recorded for selected filter.")
            lbl_empty.setStyleSheet("color: #94a3b8; font-size: 14px;")
            self.vbox_events.addWidget(lbl_empty)
            return

        for evt in events:
            card = self.create_event_card(evt)
            self.vbox_events.addWidget(card)

    def create_event_card(self, evt):
        card = QFrame()

        priority = evt.get("priority", "INFO")
        border_color = "#10b981"  # Green
        if priority == "WARNING":
            border_color = "#eab308"  # Yellow
        elif priority == "ALERT":
            border_color = "#ef4444"  # Red

        card.setStyleSheet(f"""
            QFrame {{
                background-color: #1e293b;
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)

        hbox = QHBoxLayout(card)

        # Info
        s_name = evt.get("subject_name") or evt.get("subject_track_id") or "Entity"
        rel_str = f" ➔ {evt['relationship']} ➔ {evt['object_class']}" if evt.get("relationship") else ""

        lbl_info = QLabel(
            f"<b>[{evt['priority']}] {evt['event_type']}</b>{rel_str}<br>"
            f"Subject: <b>{s_name}</b> | Conf: {evt['confidence']*100:.0f}%<br>"
            f"<span style='color: #94a3b8; font-size: 11px;'>Timestamp: {evt['timestamp'][:19]}</span>"
        )
        lbl_info.setStyleSheet("color: #f8fafc; font-size: 13px;")

        hbox.addWidget(lbl_info, 1)

        if evt.get("image_path") and os.path.exists(evt["image_path"]):
            btn_img = QPushButton("📷 View Frame")
            btn_img.clicked.connect(lambda _, p=evt["image_path"]: QDesktopServices.openUrl(QUrl.fromLocalFile(p)))
            hbox.addWidget(btn_img)

        return card
