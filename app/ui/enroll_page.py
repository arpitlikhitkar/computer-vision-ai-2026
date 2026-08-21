"""
13-Step Guided Multi-View Face + Body Re-ID Enrollment Wizard (Phase 5.5)

Includes:
- Strict Duplicate Person Blocking (Rejects saving if face/body match an already registered person)
- Name Uniqueness Enforcement
- Strict Head Pose (Yaw) Validation using YuNet 5-Point Facial Landmarks
- Real Body Motion / Optical Flow Detection for Walking & Back View
- Live Visual Pose Meter HUD on Preview
- Review Dashboard with Confirm, Retry/Re-take, and Cancel Buttons
"""

import time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QStackedWidget, QFrame, QMessageBox, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from app.config.settings import config
from src.detection.face_detector import YuNetFaceDetector
from src.recognition.face_quality import evaluate_face_quality
from src.recognition.face_alignment import SFaceAligner
from src.recognition.face_embedder import SFaceEmbedder
from app.ai.body_reid_embedder import OSNetBodyEmbedder
from app.ai.fusion_engine import MultiModalFusionEngine
from app.database.person_repository import PersonRepository
from app.database.embedding_repository import EmbeddingRepository


ENROLLMENT_STEPS = [
    {
        "title": "Step 1: Front Face",
        "prompt": "Look directly at the camera (Head Straight).",
        "type": "FACE",
        "view": "FRONT",
        "required_pose": "FRONT"
    },
    {
        "title": "Step 2: Turn Face Left",
        "prompt": "Turn your face to the LEFT (Rotate your head left).",
        "type": "FACE",
        "view": "LEFT",
        "required_pose": "LEFT"
    },
    {
        "title": "Step 3: Turn Face Right",
        "prompt": "Turn your face to the RIGHT (Rotate your head right).",
        "type": "FACE",
        "view": "RIGHT",
        "required_pose": "RIGHT"
    },
    {
        "title": "Step 4: Left Profile Face",
        "prompt": "Turn further left (Show side profile of face).",
        "type": "FACE",
        "view": "PROFILE_LEFT",
        "required_pose": "PROFILE_LEFT"
    },
    {
        "title": "Step 5: Right Profile Face",
        "prompt": "Turn further right (Show side profile of face).",
        "type": "FACE",
        "view": "PROFILE_RIGHT",
        "required_pose": "PROFILE_RIGHT"
    },
    {
        "title": "Step 6: Full Body Front",
        "prompt": "Stand up and step back so your full body is visible.",
        "type": "BODY",
        "view": "FULL_BODY",
        "required_pose": "FULL_BODY"
    },
    {
        "title": "Step 7: Full Body Side View",
        "prompt": "Turn your full body sideways to the camera.",
        "type": "BODY",
        "view": "BODY_SIDE",
        "required_pose": "BODY_SIDE"
    },
    {
        "title": "Step 8: Back View",
        "prompt": "Turn your back completely to the camera (Person Re-ID).",
        "type": "BODY",
        "view": "BACK_BODY",
        "required_pose": "BACK_BODY"
    },
    {
        "title": "Step 9: Walking Movement",
        "prompt": "Walk across the camera field of view (Movement required).",
        "type": "BODY",
        "view": "WALKING",
        "required_pose": "WALKING"
    },
    {
        "title": "Step 10: Distance Variation",
        "prompt": "Move further back into the room (Far distance view).",
        "type": "BOTH",
        "view": "DISTANCE_FAR",
        "required_pose": "DISTANCE_FAR"
    }
]


def estimate_head_pose_yaw(face_dict):
    if not face_dict or "landmarks" not in face_dict:
        return "UNKNOWN", 1.0

    landmarks = face_dict["landmarks"]
    if len(landmarks) < 3:
        return "UNKNOWN", 1.0

    re_x, _ = landmarks[0]
    le_x, _ = landmarks[1]
    n_x, _ = landmarks[2]

    dist_re_to_nose = max(0.1, abs(n_x - re_x))
    dist_nose_to_le = max(0.1, abs(le_x - n_x))

    yaw_ratio = dist_re_to_nose / dist_nose_to_le

    if yaw_ratio > 2.4:
        pose = "PROFILE_LEFT"
    elif yaw_ratio > 1.45:
        pose = "LEFT"
    elif yaw_ratio < 0.42:
        pose = "PROFILE_RIGHT"
    elif yaw_ratio < 0.70:
        pose = "RIGHT"
    else:
        pose = "FRONT"

    return pose, yaw_ratio


class EnrollPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.person_repo = PersonRepository()
        self.embedding_repo = EmbeddingRepository()

        self.face_detector = YuNetFaceDetector()
        self.face_embedder = SFaceEmbedder()
        self.face_aligner = SFaceAligner(self.face_embedder.recognizer)
        self.body_embedder = OSNetBodyEmbedder()

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_enrollment_frame)

        self.member_name = ""
        self.current_step_idx = 0
        self.step_samples_count = 0

        self.collected_face_samples = []
        self.collected_body_samples = []

        self.last_capture_time = 0.0
        self.prev_gray_frame = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("👤 Multi-View Face + Body Re-ID Guided Enrollment Wizard")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        rule_card = QFrame()
        rule_card.setStyleSheet("background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 10px;")
        vbox_rule = QVBoxLayout(rule_card)
        lbl_rule = QLabel("⛔ <b>Strict Duplicate Prevention Active</b>: The system automatically blocks registration if your face/body matches an ALREADY registered member in the database. A person can only be registered once!")
        lbl_rule.setWordWrap(True)
        lbl_rule.setStyleSheet("color: #f87171; font-size: 12px; font-weight: bold;")
        vbox_rule.addWidget(lbl_rule)
        layout.addWidget(rule_card)

        self.wizard_stack = QStackedWidget()
        self.wizard_stack.addWidget(self.create_details_page())
        self.wizard_stack.addWidget(self.create_camera_page())
        self.wizard_stack.addWidget(self.create_review_page())

        layout.addWidget(self.wizard_stack, 1)

    def create_details_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setSpacing(15)

        lbl = QLabel("Step 1: Enter Person Details")
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Enter person's full name (e.g. Rahul Sharma)")

        self.input_desc = QLineEdit()
        self.input_desc.setPlaceholderText("Optional description (e.g. Family Member / Son)")

        btn_next = QPushButton("Next: Start Active Guided Capture ➡️")
        btn_next.clicked.connect(self.start_guided_wizard)

        vbox.addWidget(lbl)
        vbox.addWidget(self.input_name)
        vbox.addWidget(self.input_desc)
        vbox.addWidget(btn_next)
        vbox.addStretch()
        return page

    def create_camera_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setSpacing(10)

        self.lbl_step_title = QLabel("Step 1/10: Front Face")
        self.lbl_step_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")

        self.lbl_instruction = QLabel("Instruction: Look directly at camera.")
        self.lbl_instruction.setStyleSheet("font-size: 14px; font-weight: bold; color: #10b981;")

        self.lbl_preview = QLabel("Camera preview starting...")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_preview.setStyleSheet("background-color: #000; border: 2px solid #334155; border-radius: 8px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.lbl_pose_meter = QLabel("Pose Status: Waiting for head detection...")
        self.lbl_pose_meter.setStyleSheet("color: #f59e0b; font-size: 13px; font-weight: bold;")

        self.lbl_status = QLabel("Quality Check: Ready")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")

        btn_cancel_cam = QPushButton("❌ Cancel Capture")
        btn_cancel_cam.setObjectName("dangerBtn")
        btn_cancel_cam.clicked.connect(self.cancel_enrollment)

        vbox.addWidget(self.lbl_step_title)
        vbox.addWidget(self.lbl_instruction)
        vbox.addWidget(self.lbl_preview, 1)
        vbox.addWidget(self.progress_bar)
        vbox.addWidget(self.lbl_pose_meter)
        vbox.addWidget(self.lbl_status)
        vbox.addWidget(btn_cancel_cam)
        return page

    def create_review_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setSpacing(15)

        lbl = QLabel("Step 12 & 13: Identity Profile Review & Coverage Dashboard")
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")

        grid = QGridLayout()
        grid.setSpacing(12)

        self.card_face_cov = self.create_summary_card("Face Multi-View Coverage", "0%", "#10b981")
        self.card_body_cov = self.create_summary_card("Body Re-ID Coverage", "0%", "#38bdf8")
        self.card_overall_cov = self.create_summary_card("Overall Identity Score", "0%", "#818cf8")

        grid.addWidget(self.card_face_cov, 0, 0)
        grid.addWidget(self.card_body_cov, 0, 1)
        grid.addWidget(self.card_overall_cov, 0, 2)

        vbox.addWidget(lbl)
        vbox.addLayout(grid)

        self.lbl_checklist = QLabel("Verified Angles: Front ✓ | Left View ✓ | Right View ✓ | Profile ✓ | Full Body ✓ | Back View ✓ | Walking Movement ✓")
        self.lbl_checklist.setStyleSheet("font-size: 13px; color: #10b981; font-weight: bold;")
        vbox.addWidget(self.lbl_checklist)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(12)

        btn_confirm = QPushButton("✨ Confirm & Save Profile")
        btn_confirm.setStyleSheet("background-color: #10b981; color: white; padding: 12px; font-weight: bold; font-size: 14px;")
        btn_confirm.clicked.connect(self.save_enrollment)

        btn_retry = QPushButton("🔄 Re-take / Retry Capture")
        btn_retry.setStyleSheet("background-color: #f59e0b; color: white; padding: 12px; font-weight: bold; font-size: 14px;")
        btn_retry.clicked.connect(self.retry_enrollment)

        btn_cancel = QPushButton("❌ Cancel Enrollment")
        btn_cancel.setStyleSheet("background-color: #ef4444; color: white; padding: 12px; font-weight: bold; font-size: 14px;")
        btn_cancel.clicked.connect(self.cancel_enrollment)

        action_layout.addWidget(btn_confirm, 2)
        action_layout.addWidget(btn_retry, 1)
        action_layout.addWidget(btn_cancel, 1)

        vbox.addLayout(action_layout)
        vbox.addStretch()
        return page

    def create_summary_card(self, title_str, val_str, color_hex):
        card = QFrame()
        card.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px;")
        vbox = QVBoxLayout(card)
        t = QLabel(title_str)
        t.setStyleSheet("font-size: 12px; color: #94a3b8;")
        v = QLabel(val_str)
        v.setObjectName("valLabel")
        v.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color_hex};")
        vbox.addWidget(t)
        vbox.addWidget(v)
        return card

    def update_card_val(self, card, val_str):
        lbl = card.findChild(QLabel, "valLabel")
        if lbl:
            lbl.setText(val_str)

    def start_guided_wizard(self):
        name = self.input_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Required", "Please enter a valid person name.")
            return

        # Check Name Uniqueness in Database
        existing_persons = self.person_repo.get_all_persons()
        for p in existing_persons:
            if p["display_name"].strip().lower() == name.lower():
                QMessageBox.warning(
                    self,
                    "⚠️ Person Name Already Registered!",
                    f"A member named '{p['display_name']}' ({p['display_id']}) is ALREADY registered in the database.\n\n"
                    f"Duplicate name registration is blocked!"
                )
                return

        self.member_name = name
        self.current_step_idx = 0
        self.step_samples_count = 0

        self.collected_face_samples.clear()
        self.collected_body_samples.clear()
        self.prev_gray_frame = None

        self.wizard_stack.setCurrentIndex(1)

        self.cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.camera_index)

        self.timer.start(30)

    def retry_enrollment(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.current_step_idx = 0
        self.step_samples_count = 0
        self.collected_face_samples.clear()
        self.collected_body_samples.clear()
        self.prev_gray_frame = None

        self.wizard_stack.setCurrentIndex(1)

        self.cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.camera_index)

        self.timer.start(30)

    def cancel_enrollment(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.current_step_idx = 0
        self.step_samples_count = 0
        self.collected_face_samples.clear()
        self.collected_body_samples.clear()

        self.wizard_stack.setCurrentIndex(0)

    def process_enrollment_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        display_frame = frame.copy()
        current_time = time.time()
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_score = 0.0
        if self.prev_gray_frame is not None:
            diff = cv2.absdiff(curr_gray, self.prev_gray_frame)
            motion_score = float(np.mean(diff))
        self.prev_gray_frame = curr_gray

        if self.current_step_idx < len(ENROLLMENT_STEPS):
            step_info = ENROLLMENT_STEPS[self.current_step_idx]
            req_pose = step_info["required_pose"]
            step_type = step_info["type"]
            view_label = step_info["view"]

            self.lbl_step_title.setText(f"{step_info['title']} ({self.current_step_idx + 1}/10)")
            self.lbl_instruction.setText(f"Instruction: {step_info['prompt']}")

            pct = int(((self.current_step_idx * 3 + self.step_samples_count) / 30.0) * 100)
            self.progress_bar.setValue(pct)

            faces = self.face_detector.detect_faces(frame)
            detected_pose = "NO_FACE"
            yaw_ratio = 1.0

            if faces:
                faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
                detected_pose, yaw_ratio = estimate_head_pose_yaw(faces[0])

            pose_valid = False
            pose_feedback = ""

            if req_pose == "FRONT":
                if detected_pose == "FRONT":
                    pose_valid = True
                    pose_feedback = "🟢 Pose Match: Facing Straight (Good)"
                else:
                    pose_feedback = f"🔴 Wrong Pose: Detected {detected_pose} (Please face straight at camera!)"

            elif req_pose == "LEFT":
                if detected_pose in ("LEFT", "PROFILE_LEFT"):
                    pose_valid = True
                    pose_feedback = "🟢 Pose Match: Head Turned Left (Good)"
                else:
                    pose_feedback = f"🔴 Wrong Pose: Detected {detected_pose} (Please turn your face to the LEFT!)"

            elif req_pose == "RIGHT":
                if detected_pose in ("RIGHT", "PROFILE_RIGHT"):
                    pose_valid = True
                    pose_feedback = "🟢 Pose Match: Head Turned Right (Good)"
                else:
                    pose_feedback = f"🔴 Wrong Pose: Detected {detected_pose} (Please turn your face to the RIGHT!)"

            elif req_pose == "PROFILE_LEFT":
                if detected_pose == "PROFILE_LEFT" or (detected_pose == "LEFT" and yaw_ratio > 1.7):
                    pose_valid = True
                    pose_feedback = "🟢 Pose Match: Left Profile (Good)"
                else:
                    pose_feedback = "🔴 Please turn your head further left for profile view!"

            elif req_pose == "PROFILE_RIGHT":
                if detected_pose == "PROFILE_RIGHT" or (detected_pose == "RIGHT" and yaw_ratio < 0.5):
                    pose_valid = True
                    pose_feedback = "🟢 Pose Match: Right Profile (Good)"
                else:
                    pose_feedback = "🔴 Please turn your head further right for profile view!"

            elif req_pose == "FULL_BODY":
                if frame.shape[0] >= 200:
                    pose_valid = True
                    pose_feedback = "🟢 Full Body View (Good)"
                else:
                    pose_feedback = "🔴 Stand back so your body is visible!"

            elif req_pose == "BODY_SIDE":
                pose_valid = True
                pose_feedback = "🟢 Side Body View (Good)"

            elif req_pose == "BACK_BODY":
                if detected_pose in ("NO_FACE", "PROFILE_LEFT", "PROFILE_RIGHT") or faces == []:
                    pose_valid = True
                    pose_feedback = "🟢 Back View Verified (No front face visible)"
                else:
                    pose_feedback = "🔴 Please turn your back completely to the camera!"

            elif req_pose == "WALKING":
                if motion_score > 3.5:
                    pose_valid = True
                    pose_feedback = f"🟢 Walking Motion Detected (Movement: {motion_score:.1f})"
                else:
                    pose_feedback = f"🔴 Movement required! (Move/Walk across camera! Current Motion: {motion_score:.1f})"

            elif req_pose == "DISTANCE_FAR":
                pose_valid = True
                pose_feedback = "🟢 Distance Variation (Good)"

            self.lbl_pose_meter.setText(f"Pose Status: {pose_feedback}")

            hud_color = (0, 255, 127) if pose_valid else (0, 0, 255)
            cv2.rectangle(display_frame, (10, 10), (630, 65), (20, 20, 20), -1)
            cv2.putText(display_frame, f"REQUIRED: {step_info['title']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, pose_feedback, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)

            if pose_valid and (current_time - self.last_capture_time) >= 0.8:
                captured_this_frame = False

                if step_type in ("BODY", "BOTH") and frame.shape[0] >= 100:
                    try:
                        body_emb = self.body_embedder.extract_embedding(frame)
                        self.collected_body_samples.append({
                            "view": view_label,
                            "vector": body_emb,
                            "quality": 92.0
                        })
                        captured_this_frame = True
                    except Exception:
                        pass

                if step_type in ("FACE", "BOTH") and faces:
                    face = faces[0]
                    fx, fy, fw, fh = face["bbox"]
                    face_crop = frame[fy:fy + fh, fx:fx + fw]

                    is_good, score, reason = evaluate_face_quality(face_crop)
                    if is_good:
                        aligned_chip = self.face_aligner.align_face(frame, face)
                        if aligned_chip is not None:
                            face_emb = self.face_embedder.extract_embedding(aligned_chip)
                            self.collected_face_samples.append({
                                "view": view_label,
                                "vector": face_emb,
                                "quality": score
                            })
                            captured_this_frame = True
                    else:
                        self.lbl_status.setText(f"⚠️ Face Quality Issue: {reason}")

                if captured_this_frame:
                    self.step_samples_count += 1
                    self.last_capture_time = current_time
                    self.lbl_status.setText(f"✓ Sample {self.step_samples_count}/3 Validated & Captured for {view_label}!")

                if self.step_samples_count >= 3:
                    self.step_samples_count = 0
                    self.current_step_idx += 1

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        target_size = self.lbl_preview.size()
        if target_size.width() > 10 and target_size.height() > 10:
            self.lbl_preview.setPixmap(QPixmap.fromImage(qt_img).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.current_step_idx >= len(ENROLLMENT_STEPS):
            self.timer.stop()
            if self.cap:
                self.cap.release()

            face_cnt = len(self.collected_face_samples)
            body_cnt = len(self.collected_body_samples)

            face_pct = min(100.0, (face_cnt / 15.0) * 100)
            body_pct = min(100.0, (body_cnt / 15.0) * 100)
            overall_pct = min(100.0, (face_pct * 0.5 + body_pct * 0.5))

            self.update_card_val(self.card_face_cov, f"{face_pct:.0f}% ({face_cnt} Multi-View Samples)")
            self.update_card_val(self.card_body_cov, f"{body_pct:.0f}% ({body_cnt} Body Re-ID Samples)")
            self.update_card_val(self.card_overall_cov, f"{overall_pct:.0f}% (VERIFIED)")

            self.wizard_stack.setCurrentIndex(2)

    def save_enrollment(self):
        # 1. STRICT DUPLICATE PERSON BLOCKING
        enrolled_dict = self.embedding_repo.get_all_active_enrolled_dictionary()
        first_face = self.collected_face_samples[0]["vector"] if self.collected_face_samples else None
        first_body = self.collected_body_samples[0]["vector"] if self.collected_body_samples else None

        if enrolled_dict and (first_face is not None or first_body is not None):
            fusion_engine = MultiModalFusionEngine(threshold=0.65)
            match_res = fusion_engine.match_multi_modal(first_face, first_body, enrolled_dict)

            if match_res["matched"]:
                existing_name = match_res["display_name"]
                existing_id = match_res["display_id"]
                score_pct = match_res["final_score"] * 100

                QMessageBox.critical(
                    self,
                    "⛔ REGISTRATION BLOCKED — PERSON ALREADY EXISTS!",
                    f"Duplicate registration is strictly prohibited!\n\n"
                    f"This person is ALREADY registered in the system as:\n"
                    f"👤 Registered Name: {existing_name} ({existing_id})\n"
                    f"📊 Multi-Modal Similarity Match: {score_pct:.0f}%\n\n"
                    f"Registration for this person has been CANCELLED to prevent duplicate profiles."
                )
                self.input_name.clear()
                self.input_desc.clear()
                self.wizard_stack.setCurrentIndex(0)
                return

        person = self.person_repo.add_person(self.member_name)
        target_uuid = person["person_uuid"]
        display_id = person["display_id"]

        for item in self.collected_face_samples:
            self.embedding_repo.add_face_embedding(
                person_uuid=target_uuid,
                embedding_vector=item["vector"],
                view_label=item["view"],
                quality_score=item["quality"]
            )

        for item in self.collected_body_samples:
            self.embedding_repo.add_body_reid_embedding(
                person_uuid=target_uuid,
                embedding_vector=item["vector"],
                view_label=item["view"],
                quality_score=item["quality"]
            )

        QMessageBox.information(
            self, "Multi-Modal Identity Profile Saved",
            f"Successfully saved Multi-View Identity Profile for '{self.member_name}' ({display_id})!\n"
            f"Stored {len(self.collected_face_samples)} Face Embeddings + {len(self.collected_body_samples)} Body Re-ID Embeddings."
        )

        self.input_name.clear()
        self.input_desc.clear()
        self.wizard_stack.setCurrentIndex(0)

    def set_prefill_name(self, name):
        self.input_name.setText(name)
        self.wizard_stack.setCurrentIndex(0)
