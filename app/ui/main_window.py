"""
Main Window for PySide6 Desktop Application
Includes Top Header Bar + Left Sidebar Navigation + QStackedWidget Pages
Updated with Safe closeEvent & Multi-Modal Identity Protections
"""

import torch
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QButtonGroup, QFrame, QMessageBox
)
from PySide6.QtCore import Qt

from app.config.settings import config
from app.services.alarm_service import alarm_service
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
        self.lbl_device_status.setObjectName("statusLabel")

        # Top Header Alarm Stop & Mute Toggle Button
        self.btn_alarm_toggle = QPushButton()
        self.btn_alarm_toggle.setCursor(Qt.PointingHandCursor)
        self.update_alarm_btn_style()
        self.btn_alarm_toggle.clicked.connect(self.toggle_alarm_mute)

        top_layout.addWidget(lbl_logo)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_alarm_toggle)
        top_layout.addWidget(self.lbl_device_status)

        root_layout.addWidget(top_header)

        # 2. Main Content Area (Sidebar + Stacked Pages)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar Navigation
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(8)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        nav_items = [
            ("📊 Dashboard", 0),
            ("📹 Live Camera", 1),
            ("👥 People", 2),
            ("✨ Enroll Wizard", 3),
            ("❓ Unknown", 4),
            ("📜 Events", 5),
            ("🎥 Recordings", 6),
            ("📷 Camera Settings", 7),
            ("🤖 AI Settings", 8),
            ("🧠 Models", 9),
            ("⚙️ Settings", 10),
        ]

        for text, page_idx in nav_items:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("navBtn")
            self.btn_group.addButton(btn, page_idx)
            sidebar_layout.addWidget(btn)

            if page_idx == 0:
                btn.setChecked(True)

        sidebar_layout.addStretch()
        content_layout.addWidget(sidebar, 0)

        # QStackedWidget Pages
        self.stacked_widget = QStackedWidget()

        self.page_dashboard = DashboardPage()
        self.page_live_camera = LiveCameraPage()
        self.page_people = PeoplePage()
        self.page_enroll = EnrollPage()
        self.page_unknown = UnknownPage()
        self.page_events = EventsPage()
        self.page_recordings = RecordingsPage()
        self.page_camera_settings = CameraSettingsPage()
        self.page_ai_settings = AISettingsPage()
        self.page_models = ModelPage()
        self.page_settings = SettingsPage()

        self.stacked_widget.addWidget(self.page_dashboard)
        self.stacked_widget.addWidget(self.page_live_camera)
        self.stacked_widget.addWidget(self.page_people)
        self.stacked_widget.addWidget(self.page_enroll)
        self.stacked_widget.addWidget(self.page_unknown)
        self.stacked_widget.addWidget(self.page_events)
        self.stacked_widget.addWidget(self.page_recordings)
        self.stacked_widget.addWidget(self.page_camera_settings)
        self.stacked_widget.addWidget(self.page_ai_settings)
        self.stacked_widget.addWidget(self.page_models)
        self.stacked_widget.addWidget(self.page_settings)

        content_layout.addWidget(self.stacked_widget, 1)
        root_layout.addLayout(content_layout, 1)

        self.btn_group.idClicked.connect(self.switch_page)
        self.page_unknown.enroll_requested.connect(self.start_enrollment_for)
        self.page_live_camera.camera_toggle_requested.connect(self.handle_camera_toggle)

    def handle_camera_toggle(self, active: bool):
        if not active:
            if self.camera_worker and self.camera_worker.isRunning():
                self.camera_worker.stop()
                self.lbl_device_status.setText("Device: CPU Mode | Status: STOPPED")
        else:
            if not self.camera_worker or not self.camera_worker.isRunning():
                self.start_camera_worker()

    def update_alarm_btn_style(self):
        if config.enable_audio_alarm:
            self.btn_alarm_toggle.setText("🔔 Alarm Siren Active")
            self.btn_alarm_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            self.btn_alarm_toggle.setText("🔕 Alarm Muted")
            self.btn_alarm_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
            """)

    def toggle_alarm_mute(self):
        if hasattr(alarm_service, 'toggle_mute'):
            new_active = alarm_service.toggle_mute()
            self.update_alarm_btn_style()
            if new_active:
                QMessageBox.information(self, "Alarm Siren Sound", "🔔 Security Siren Sound ENABLED for Unknown Detections!")
            else:
                QMessageBox.information(self, "Alarm Siren Sound", "🔕 Security Siren Sound MUTED! All active alarms stopped.")

    def start_camera_worker(self):
        self.camera_worker = CameraWorker(parent=self)
        self.camera_worker.frame_processed.connect(self.page_live_camera.update_frame)
        self.camera_worker.status_changed.connect(self.update_status_bar)
        self.camera_worker.start()

    def update_status_bar(self, status_text):
        device_str = "NVIDIA CUDA (GPU)" if torch.cuda.is_available() else "CPU Mode"
        self.lbl_device_status.setText(f"Device: {device_str} | Status: {status_text}")

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

    def start_enrollment_for(self, person_name):
        self.page_enroll.input_name.setText(person_name)
        self.stacked_widget.setCurrentIndex(3)
        self.btn_group.button(3).setChecked(True)

    def closeEvent(self, event):
        if self.camera_worker:
            self.camera_worker.stop()
        if hasattr(alarm_service, 'stop_alarm'):
            try:
                alarm_service.stop_alarm()
            except Exception:
                pass
        event.accept()
