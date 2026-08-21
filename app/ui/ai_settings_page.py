"""
AI Model Threshold Settings Page for PySide6 Application
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QDoubleSpinBox, QPushButton, QMessageBox
)
from app.config.settings import config


class AISettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("🤖 AI Thresholds & Model Parameters")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(15)

        self.spin_yolo_conf = QDoubleSpinBox()
        self.spin_yolo_conf.setRange(0.1, 0.9)
        self.spin_yolo_conf.setSingleStep(0.05)
        self.spin_yolo_conf.setValue(config.person_conf_threshold)

        self.spin_face_conf = QDoubleSpinBox()
        self.spin_face_conf.setRange(0.1, 0.9)
        self.spin_face_conf.setSingleStep(0.05)
        self.spin_face_conf.setValue(config.face_conf_threshold)

        self.spin_rec_thresh = QDoubleSpinBox()
        self.spin_rec_thresh.setRange(0.4, 0.9)
        self.spin_rec_thresh.setSingleStep(0.05)
        self.spin_rec_thresh.setValue(config.recognition_threshold)

        form.addRow("YOLO Person Confidence Threshold:", self.spin_yolo_conf)
        form.addRow("YuNet Face Confidence Threshold:", self.spin_face_conf)
        form.addRow("SFace Recognition Similarity Threshold:", self.spin_rec_thresh)

        layout.addLayout(form)

        btn_save = QPushButton("💾 Save AI Settings")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)

        layout.addStretch()

    def save_settings(self):
        config.person_conf_threshold = self.spin_yolo_conf.value()
        config.face_conf_threshold = self.spin_face_conf.value()
        config.recognition_threshold = self.spin_rec_thresh.value()
        config.save_to_json()
        QMessageBox.information(self, "Saved", "AI Threshold Settings saved successfully!")
