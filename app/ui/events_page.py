"""
Recognition Audit Events Page for PySide6 Application
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView
)
from PySide6.QtCore import Qt
from app.database.event_repository import EventRepository
from app.database.person_repository import PersonRepository


class EventsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.event_repo = EventRepository()
        self.person_repo = PersonRepository()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        top_layout = QHBoxLayout()
        title = QLabel("📜 Historical Recognition Audit Events")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        btn_refresh = QPushButton("🔄 Refresh Events")
        btn_refresh.setObjectName("secondaryBtn")
        btn_refresh.clicked.connect(self.load_events)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(btn_refresh)

        layout.addLayout(top_layout)

        # Table Widget
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Event ID", "Timestamp", "Track ID", "Recognition Result", "Similarity"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.table)

        self.load_events()

    def load_events(self):
        events = self.event_repo.get_recent_events(limit=50)
        self.table.setRowCount(0)

        for row_idx, e in enumerate(events):
            self.table.insertRow(row_idx)

            person_name = "UNKNOWN"
            if e["person_uuid"]:
                p = self.person_repo.get_person_by_uuid(e["person_uuid"])
                if p:
                    person_name = p["display_name"]

            res_str = f"🟢 KNOWN ({person_name})" if e["recognition_result"] == "KNOWN" else "🔴 UNKNOWN"

            self.table.setItem(row_idx, 0, QTableWidgetItem(str(e["id"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(e["timestamp"][:19]))
            self.table.setItem(row_idx, 2, QTableWidgetItem(f"Track {e['track_id']}"))
            self.table.setItem(row_idx, 3, QTableWidgetItem(res_str))
            self.table.setItem(row_idx, 4, QTableWidgetItem(f"{e['similarity_score']*100:.1f}%"))
