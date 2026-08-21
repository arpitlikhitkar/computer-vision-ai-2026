"""
People Management Page for PySide6 Application (Phase 5.5)
Displays Multi-View Face & Body Re-ID sample counts, profile management & profile merging!
Fixed Actions column width, row height, and button text visibility.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QMessageBox, QHeaderView, QInputDialog
)
from PySide6.QtCore import Qt
from app.database.person_repository import PersonRepository
from app.database.embedding_repository import EmbeddingRepository


class PeoplePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.person_repo = PersonRepository()
        self.embedding_repo = EmbeddingRepository()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        top_layout = QHBoxLayout()
        title = QLabel("👥 Enrolled Household Members (Multi-Modal Identity Profiles)")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search members by name or ID...")
        self.search_input.textChanged.connect(self.filter_members)

        btn_merge = QPushButton("🧩 Merge Duplicate Profiles")
        btn_merge.setStyleSheet("background-color: #818cf8; color: white; font-weight: bold; padding: 6px 12px;")
        btn_merge.clicked.connect(self.merge_duplicate_dialog)

        self.btn_refresh = QPushButton("🔄 Refresh List")
        self.btn_refresh.setObjectName("secondaryBtn")
        self.btn_refresh.clicked.connect(self.load_members)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(self.search_input)
        top_layout.addWidget(btn_merge)
        top_layout.addWidget(self.btn_refresh)

        layout.addLayout(top_layout)

        # Table Widget
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Person ID", "Name", "Status", "Face Samples", "Body Re-ID Samples", "Created At", "Actions"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Interactive)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        header.setSectionResizeMode(6, QHeaderView.Interactive)

        self.table.setColumnWidth(0, 110)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 130)
        self.table.setColumnWidth(4, 150)
        self.table.setColumnWidth(5, 110)
        self.table.setColumnWidth(6, 200)

        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.table)

        self.load_members()

    def load_members(self):
        persons = self.person_repo.get_all_persons()
        self.all_persons = persons
        self.render_table(persons)

    def filter_members(self, query):
        q = query.lower().strip()
        filtered = [
            p for p in self.all_persons
            if q in p["display_name"].lower() or q in p["display_id"].lower()
        ]
        self.render_table(filtered)

    def render_table(self, persons):
        self.table.setRowCount(0)

        for row_idx, p in enumerate(persons):
            self.table.insertRow(row_idx)

            face_objs = self.embedding_repo.get_face_embeddings_for_person(p["person_uuid"])
            body_objs = self.embedding_repo.get_body_reid_embeddings_for_person(p["person_uuid"])

            item_id = QTableWidgetItem(p["display_id"])
            item_name = QTableWidgetItem(p["display_name"])
            item_status = QTableWidgetItem("🟢 ACTIVE" if p["status"] == "ACTIVE" else "🔴 INACTIVE")
            item_face = QTableWidgetItem(f"{len(face_objs)} Face Samples")
            item_body = QTableWidgetItem(f"{len(body_objs)} Body Re-ID Samples")
            item_date = QTableWidgetItem(p["created_at"][:10])

            self.table.setItem(row_idx, 0, item_id)
            self.table.setItem(row_idx, 1, item_name)
            self.table.setItem(row_idx, 2, item_status)
            self.table.setItem(row_idx, 3, item_face)
            self.table.setItem(row_idx, 4, item_body)
            self.table.setItem(row_idx, 5, item_date)

            # Actions Widget with Clear Buttons & Icons
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 4, 4, 4)
            actions_layout.setSpacing(6)

            btn_toggle = QPushButton("🔄 Toggle")
            btn_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #334155;
                    color: #f8fafc;
                    border: 1px solid #475569;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    min-height: 26px;
                }
                QPushButton:hover {
                    background-color: #475569;
                }
            """)
            btn_toggle.clicked.connect(lambda _, u=p["person_uuid"], s=p["status"]: self.toggle_status(u, s))

            btn_delete = QPushButton("🗑️ Delete")
            btn_delete.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                    font-weight: bold;
                    min-height: 26px;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
            """)
            btn_delete.clicked.connect(lambda _, u=p["person_uuid"], n=p["display_name"]: self.delete_person(u, n))

            actions_layout.addWidget(btn_toggle)
            actions_layout.addWidget(btn_delete)

            self.table.setCellWidget(row_idx, 6, actions_widget)

    def merge_duplicate_dialog(self):
        persons = self.person_repo.get_all_persons()
        if len(persons) < 2:
            QMessageBox.information(self, "Merge Profiles", "At least 2 registered persons are required to merge duplicate profiles.")
            return

        person_labels = [f"{p['display_id']} - {p['display_name']}" for p in persons]

        source_item, ok1 = QInputDialog.getItem(
            self, "Select Duplicate Source Profile to Merge FROM",
            "Choose duplicate person profile (will be merged & removed):",
            person_labels, 0, False
        )
        if not ok1 or not source_item:
            return

        source_idx = person_labels.index(source_item)
        source_p = persons[source_idx]

        target_labels = [l for i, l in enumerate(person_labels) if i != source_idx]
        target_item, ok2 = QInputDialog.getItem(
            self, "Select Target Main Profile to Merge INTO",
            f"Merge all embeddings from '{source_item}' INTO main profile:",
            target_labels, 0, False
        )
        if not ok2 or not target_item:
            return

        target_idx_orig = person_labels.index(target_item)
        target_p = persons[target_idx_orig]

        reply = QMessageBox.question(
            self, "Confirm Profile Merge",
            f"Are you sure you want to merge duplicate profile '{source_p['display_id']} ({source_p['display_name']})' INTO '{target_p['display_id']} ({target_p['display_name']})'?\n\n"
            f"All face and body embeddings will be transferred to '{target_p['display_name']}', and '{source_p['display_id']}' will be removed.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.person_repo.merge_persons(source_p["person_uuid"], target_p["person_uuid"])
            QMessageBox.information(self, "Profiles Merged", f"Successfully merged '{source_p['display_name']}' into '{target_p['display_name']}'!")
            self.load_members()

    def toggle_status(self, uuid_val, current_status):
        new_status = "INACTIVE" if current_status == "ACTIVE" else "ACTIVE"
        self.person_repo.update_person_status(uuid_val, new_status)
        self.load_members()

    def delete_person(self, uuid_val, name):
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete member '{name}' permanently?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.person_repo.delete_person(uuid_val)
            self.load_members()
