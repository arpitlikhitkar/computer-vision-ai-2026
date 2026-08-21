"""
Main Window for PySide6 Desktop Application
Includes Top Header Bar + Left Sidebar Navigation + QStackedWidget Pages
"""

import torch
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QButtonGroup, QFrame
)
from PySide6.QtCore import Qt

from app.ui.styles import DARK_THEME_QSS
from app.ui.dashboard_page import DashboardPage
from app.ui.live_camera_page import LiveCameraPage
from app.ui.people_page import PeoplePage
from app.ui.enroll_page import EnrollPage
from app.ui.unknown_page import UnknownPage
from app.ui.events_page import EventsPage
from app.ui.recordings_page import RecordingsPage
from app.ui.camera_settings_page import CameraSettingsPage
from app.ui.ai_settings_page import AISettingsPage
from app.ui.model_page import ModelPage
from app.ui.settings_page import SettingsPage
from app.services.camera_service import CameraWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Household AI — Security & Face Recognition Desktop Software")
        self.resize(1280, 800)
        self.setStyleSheet(DARK_THEME_QSS)

        self.camera_worker = None
        self.init_ui()
        self.start_camera_worker()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Header Bar
        top_header = QFrame()
        top_header.setObjectName("topHeader")
        top_layout = QHBoxLayout(top_header)
        top_layout.setContentsMargins(20, 10, 20, 10)

        lbl_logo = QLabel("🛡️ Household AI — Security Suite")
        lbl_logo.setObjectName("headerTitle")

        device_str = "NVIDIA CUDA (GPU)" if torch.cuda.is_available() else "CPU Mode"
        self.lbl_device_status = QLabel(f"Device: {device_str} | Status: INITIALIZING")
        self.lbl_device_status.setStyleSheet("color: #10b981; font-weight: bold;")

        top_layout.addWidget(lbl_logo)
        top_layout.addStretch()
        top_layout.addWidget(self.lbl_device_status)

        root_layout.addWidget(top_header)

        # 2. Main Content Splitter (Sidebar + Pages Stack)
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar Navigation Panel
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        vbox_sidebar = QVBoxLayout(sidebar)
        vbox_sidebar.setContentsMargins(10, 15, 10, 15)
        vbox_sidebar.setSpacing(8)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        nav_items = [
            ("📊 Dashboard", 0),
            ("🎥 Live Camera", 1),
            ("👤 People", 2),
            ("✨ Enroll Wizard", 3),
            ("❓ Unknown", 4),
            ("📜 Events", 5),
            ("📹 Recordings", 6),
            ("📷 Camera Settings", 7),
            ("🤖 AI Settings", 8),
            ("🧠 Models", 9),
            ("⚙️ Settings", 10)
        ]

        for text, page_idx in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            if page_idx == 0:
                btn.setChecked(True)

            self.btn_group.addButton(btn, page_idx)
            vbox_sidebar.addWidget(btn)

        vbox_sidebar.addStretch()
        body_layout.addWidget(sidebar)

        # QStackedWidget Pages
        self.stacked_widget = QStackedWidget()

        self.page_dashboard = DashboardPage()
        self.page_live = LiveCameraPage()
        self.page_people = PeoplePage()
        self.page_enroll = EnrollPage()
        self.page_unknown = UnknownPage()
        self.page_events = EventsPage()
        self.page_recordings = RecordingsPage()
        self.page_cam_settings = CameraSettingsPage()
        self.page_ai_settings = AISettingsPage()
        self.page_models = ModelPage()
        self.page_settings = SettingsPage()

        self.stacked_widget.addWidget(self.page_dashboard)     # Index 0
        self.stacked_widget.addWidget(self.page_live)          # Index 1
        self.stacked_widget.addWidget(self.page_people)        # Index 2
        self.stacked_widget.addWidget(self.page_enroll)        # Index 3
        self.stacked_widget.addWidget(self.page_unknown)       # Index 4
        self.stacked_widget.addWidget(self.page_events)        # Index 5
        self.stacked_widget.addWidget(self.page_recordings)    # Index 6
        self.stacked_widget.addWidget(self.page_cam_settings)  # Index 7
        self.stacked_widget.addWidget(self.page_ai_settings)   # Index 8
        self.stacked_widget.addWidget(self.page_models)        # Index 9
        self.stacked_widget.addWidget(self.page_settings)      # Index 10

        body_layout.addWidget(self.stacked_widget, 1)
        root_layout.addLayout(body_layout, 1)

        # Connect Sidebar Buttons to Stacked Widget
        self.btn_group.idClicked.connect(self.switch_page)

        # Connect Unknown Page Enroll Request Signal to Enroll Wizard
        self.page_unknown.enroll_requested.connect(self.trigger_enroll_from_unknown)

        # Connect Live Camera Toggle Button
        self.page_live.btn_toggle.clicked.connect(self.toggle_camera_worker)

    def switch_page(self, page_idx):
        self.stacked_widget.setCurrentIndex(page_idx)
        if page_idx == 0:
            self.page_dashboard.refresh_stats()
        elif page_idx == 2:
            self.page_people.load_members()
        elif page_idx == 4:
            self.page_unknown.load_unknowns()
        elif page_idx == 5:
            self.page_events.load_events()

    def trigger_enroll_from_unknown(self, name):
        self.page_enroll.set_prefill_name(name)
        # Check Enroll Wizard button
        for btn in self.btn_group.buttons():
            if self.btn_group.id(btn) == 3:
                btn.setChecked(True)
                break
        self.stacked_widget.setCurrentIndex(3)

    def start_camera_worker(self):
        if self.camera_worker is None or not self.camera_worker.isRunning():
            self.camera_worker = CameraWorker()
            self.camera_worker.frame_processed.connect(self.on_frame_processed)
            self.camera_worker.status_changed.connect(self.on_worker_status_changed)
            self.camera_worker.start()

    def toggle_camera_worker(self):
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.page_live.btn_toggle.setText("▶ Start Camera Feed")
            self.page_live.btn_toggle.setStyleSheet("background-color: #10b981; color: white; padding: 10px 20px; font-weight: bold;")
        else:
            self.start_camera_worker()
            self.page_live.btn_toggle.setText("⏹ Stop Camera Feed")
            self.page_live.btn_toggle.setStyleSheet("background-color: #ef4444; color: white; padding: 10px 20px; font-weight: bold;")

    def on_frame_processed(self, qt_img, active_tracks, known_cnt, unknown_cnt, fps):
        self.page_live.update_frame(qt_img, active_tracks, known_cnt, unknown_cnt, fps)
        self.page_dashboard.update_live_metrics(active_tracks, known_cnt, unknown_cnt, fps)

    def on_worker_status_changed(self, status_msg):
        device_str = "NVIDIA CUDA (GPU)" if torch.cuda.is_available() else "CPU Mode"
        self.lbl_device_status.setText(f"Device: {device_str} | Status: {status_msg}")

    def closeEvent(self, event):
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
        event.accept()
