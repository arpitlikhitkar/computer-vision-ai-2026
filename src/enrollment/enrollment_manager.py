"""
Enrollment Manager Module (Phase 5)

Manages Household Member Enrollment workflow:
- Validates single face detection per sample
- Enforces quality checks (size, blur, lighting)
- Aligns face and generates 128-d embeddings
- Saves validated embeddings & metadata to SQLite database
"""

import time
import cv2
import numpy as np
from src.config import settings
from src.detection.face_detector import YuNetFaceDetector
from src.recognition.face_quality import evaluate_face_quality
from src.recognition.face_alignment import SFaceAligner
from src.recognition.face_embedder import SFaceEmbedder
from src.storage.database import initialize_database
from src.storage.person_repository import PersonRepository
from src.storage.embedding_repository import EmbeddingRepository


class EnrollmentManager:
    def __init__(self, db_path=None):
        self.db_path = db_path
        initialize_database(self.db_path)
        self.person_repo = PersonRepository(self.db_path)
        self.embedding_repo = EmbeddingRepository(self.db_path)

        self.face_detector = YuNetFaceDetector()
        self.face_embedder = SFaceEmbedder()
        self.face_aligner = SFaceAligner(self.face_embedder.recognizer)

    def validate_and_extract_sample(self, frame_bgr):
        """
        Validates frame for enrollment:
        - Exactly 1 face detected
        - Quality check passes
        - Aligns face chip and generates 128-d embedding
        Returns tuple: (success: bool, face_chip: np.ndarray, embedding: np.ndarray, message: str)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, None, None, "Empty camera frame"

        faces = self.face_detector.detect_faces(frame_bgr)
        if len(faces) == 0:
            return False, None, None, "No face detected. Look directly at camera."
        if len(faces) > 1:
            return False, None, None, "Multiple faces detected. Only one person during enrollment."

        face = faces[0]
        bbox = face["bbox"]
        x, y, w, h = bbox

        x_c = max(0, min(x, frame_bgr.shape[1] - 1))
        y_c = max(0, min(y, frame_bgr.shape[0] - 1))
        w_c = min(w, frame_bgr.shape[1] - x_c)
        h_c = min(h, frame_bgr.shape[0] - y_c)

        face_crop = frame_bgr[y_c:y_c + h_c, x_c:x_c + w_c]

        # Quality Check
        is_good, score, reason = evaluate_face_quality(face_crop)
        if not is_good:
            return False, None, None, f"Quality Low: {reason}"

        # 5-Landmark Affine Alignment
        aligned_chip = self.face_aligner.align_face(frame_bgr, face)
        if aligned_chip is None:
            return False, None, None, "Face alignment failed."

        # Extract 128-d Embedding
        embedding = self.face_embedder.extract_embedding(aligned_chip)
        return True, aligned_chip, embedding, f"Good Sample Captured (Quality: {score:.0f})"

    def enroll_new_member(self, display_name, valid_embeddings_list, quality_scores_list=None):
        """
        Creates person record and saves list of 128-d embeddings to SQLite database.
        """
        if not valid_embeddings_list:
            raise ValueError("[ERROR] Cannot enroll person without valid embeddings.")

        person = self.person_repo.add_person(display_name)
        person_uuid = person["person_uuid"]

        if quality_scores_list is None:
            quality_scores_list = [100.0] * len(valid_embeddings_list)

        for emb, q_score in zip(valid_embeddings_list, quality_scores_list):
            self.embedding_repo.add_embedding(person_uuid, emb, quality_score=q_score)

        print(f"[SUCCESS] Enrolled {display_name} ({person['display_id']}) with {len(valid_embeddings_list)} face samples!")
        return person
