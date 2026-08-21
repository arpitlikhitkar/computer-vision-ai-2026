"""
Model Management Page for PySide6 Application
"""

import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
from app.config.settings import config


class ModelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("🧠 Pretrained AI Models Management")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Model Name", "Task", "Architecture", "File Size", "Status"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        models_data = [
            ("yolov8n.pt", "Person Detection & Tracking", "YOLOv8 Nano (Ultralytics)", "6.2 MB", config.yolo_model_path),
            ("face_detection_yunet_2023mar.onnx", "Face Detection & 5-Landmarks", "OpenCV YuNet ONNX", "230 KB", config.yunet_model_path),
            ("face_recognition_sface_2021dec.onnx", "128-d Face Feature Embedding", "OpenCV SFace ONNX", "1.2 MB", config.sface_model_path)
        ]

        table.setRowCount(len(models_data))
        for row, (name, task, arch, size, path) in enumerate(models_data):
            exists = os.path.exists(path)
            status_str = "🟢 LOADED" if exists else "🔴 MISSING"

            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(task))
            table.setItem(row, 2, QTableWidgetItem(arch))
            table.setItem(row, 3, QTableWidgetItem(size))
            table.setItem(row, 4, QTableWidgetItem(status_str))

        layout.addWidget(table)
        layout.addStretch()
