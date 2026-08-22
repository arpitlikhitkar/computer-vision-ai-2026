"""
AI Model Management Page for PySide6 Application (Phase 6.4)
Interactive Model Registry, Dependency Rules, Capability Matrix & Config Sliders
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QMessageBox, QFrame, QSlider, QGridLayout
)
from PySide6.QtCore import Qt
from app.config.settings import config
from app.ai.model_registry import ModelRegistry


class ModelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.registry = ModelRegistry()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("🧠 AI Model Management & Dependency Dashboard")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        # 1. Models Table
        lbl_m = QLabel("Loaded Models & Memory Footprint")
        lbl_m.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(lbl_m)

        self.table_models = QTableWidget()
        self.table_models.setColumnCount(6)
        self.table_models.setHorizontalHeaderLabels([
            "Model Name", "Status", "Type", "File Size", "VRAM Footprint", "Actions"
        ])
        self.table_models.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_models)

        # 2. Capability Matrix Table
        lbl_c = QLabel("System Capability Matrix")
        lbl_c.setStyleSheet("font-size: 15px; font-weight: bold; color: #818cf8;")
        layout.addWidget(lbl_c)

        self.table_caps = QTableWidget()
        self.table_caps.setColumnCount(3)
        self.table_caps.setHorizontalHeaderLabels([
            "Feature Capability", "Required Models", "Status Ready"
        ])
        self.table_caps.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_caps)

        # Refresh Tables
        self.load_dashboard_data()

    def load_dashboard_data(self):
        # Render Models Table
        self.table_models.setRowCount(0)
        for row_idx, (m_id, m) in enumerate(self.registry.models.items()):
            self.table_models.insertRow(row_idx)

            item_name = QTableWidgetItem(m.name)
            item_status = QTableWidgetItem("🟢 LOADED" if m.status == "LOADED" else "🔴 UNLOADED")
            item_type = QTableWidgetItem(m.type)
            item_size = QTableWidgetItem(f"{m.size_mb:.1f} MB")
            item_vram = QTableWidgetItem(f"{m.vram_gb:.1f} GB")

            self.table_models.setItem(row_idx, 0, item_name)
            self.table_models.setItem(row_idx, 1, item_status)
            self.table_models.setItem(row_idx, 2, item_type)
            self.table_models.setItem(row_idx, 3, item_size)
            self.table_models.setItem(row_idx, 4, item_vram)

            btn_action = QPushButton("Unload" if m.status == "LOADED" else "Load")
            if m.status == "LOADED":
                btn_action.setObjectName("dangerBtn")
                btn_action.clicked.connect(lambda _, mid=m_id: self.unload_model(mid))
            else:
                btn_action.setStyleSheet("background-color: #10b981; color: white; font-weight: bold;")
                btn_action.clicked.connect(lambda _, mid=m_id: self.load_model(mid))

            self.table_models.setCellWidget(row_idx, 5, btn_action)

        # Render Capabilities Table
        self.table_caps.setRowCount(0)
        capabilities = self.registry.get_capability_matrix()
        for row_idx, cap in enumerate(capabilities):
            self.table_caps.insertRow(row_idx)

            item_feat = QTableWidgetItem(cap["feature"])
            item_req = QTableWidgetItem(cap["required"])
            item_ready = QTableWidgetItem("🟢 YES (READY)" if cap["ready"] == "YES" else "🔴 NO (REQUIRES MODEL)")

            self.table_caps.setItem(row_idx, 0, item_feat)
            self.table_caps.setItem(row_idx, 1, item_req)
            self.table_caps.setItem(row_idx, 2, item_ready)

    def load_model(self, model_id):
        can, msg = self.registry.can_load(model_id)
        if not can:
            QMessageBox.warning(self, "Dependency Violation", msg)
            return

        self.registry.set_model_status(model_id, "LOADED")
        if model_id == "yolo_pose":
            config.FEATURE_FLAGS["pose_estimation"] = True
        elif model_id == "relationship_engine":
            config.FEATURE_FLAGS["relationship_engine"] = True

        QMessageBox.information(self, "Model Loaded", f"Model '{self.registry.models[model_id].name}' successfully loaded into memory!")
        self.load_dashboard_data()

    def unload_model(self, model_id):
        can, msg = self.registry.can_unload(model_id)
        if not can:
            QMessageBox.warning(self, "Dependency Blocked", msg)
            return

        self.registry.set_model_status(model_id, "UNLOADED")
        if model_id == "yolo_pose":
            config.FEATURE_FLAGS["pose_estimation"] = False
        elif model_id == "relationship_engine":
            config.FEATURE_FLAGS["relationship_engine"] = False

        QMessageBox.information(self, "Model Unloaded", f"Model '{self.registry.models[model_id].name}' unloaded from memory!")
        self.load_dashboard_data()
