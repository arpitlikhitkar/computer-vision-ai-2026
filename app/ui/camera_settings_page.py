"""
Camera Settings Page for PySide6 Application
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QSpinBox, QComboBox, QPushButton, QMessageBox
)
from app.config.settings import config


class CameraSettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("📷 Camera Configuration Settings")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(15)

        self.spin_cam_idx = QSpinBox()
        self.spin_cam_idx.setRange(0, 5)
        self.spin_cam_idx.setValue(config.camera_index)

        self.combo_res = QComboBox()
        self.combo_res.addItems(["640 x 480 (Default)", "1280 x 720 (HD)"])

        form.addRow("Camera Index:", self.spin_cam_idx)
        form.addRow("Resolution:", self.combo_res)

        layout.addLayout(form)

        btn_save = QPushButton("💾 Save Camera Settings")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)

        layout.addStretch()

    def save_settings(self):
        config.camera_index = self.spin_cam_idx.value()
        config.save_to_json()
        QMessageBox.information(self, "Saved", "Camera settings saved successfully!")
