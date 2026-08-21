"""
Application General Settings Page for PySide6 Application
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QPushButton, QMessageBox
from app.config.settings import config


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("⚙️ Application Settings & Information")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        card = QFrame()
        card.setObjectName("card")
        vbox = QVBoxLayout(card)

        info_lbl = QLabel(
            "<b>Household AI System — Educational Prototype</b><br><br>"
            "<b>Storage Mode</b>: 100% Local (SQLite & File System)<br>"
            "<b>Database Path</b>: <code>data/database/household_ai_pyside.db</code><br>"
            "<b>Version</b>: 5.0.0 Desktop Edition<br><br>"
            "<i>No external internet connections or cloud uploads are used. All face embeddings and data remain strictly local on your device.</i>"
        )
        info_lbl.setStyleSheet("color: #94a3b8; font-size: 13px;")
        vbox.addWidget(info_lbl)

        layout.addWidget(card)
        layout.addStretch()
