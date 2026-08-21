"""
Automated Integration Test Suite for Phase 5.5 Multi-Modal Face + Body Re-ID Engine
"""

import os
import unittest
import numpy as np

from app.database.database import init_database
from app.database.person_repository import PersonRepository
from app.database.embedding_repository import EmbeddingRepository
from app.ai.fusion_engine import MultiModalFusionEngine


class TestPhase55MultiModal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "data/database/test_phase55_household.db"
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

        init_database(cls.db_path)
        cls.person_repo = PersonRepository(cls.db_path)
        cls.embedding_repo = EmbeddingRepository(cls.db_path)

        # Synthetic 128-d Face Embedding for Rahul
        cls.face_rahul = np.zeros(128, dtype=np.float32)
        cls.face_rahul[0:5] = 1.0
        cls.face_rahul /= np.linalg.norm(cls.face_rahul)

        # Synthetic 512-d Body Re-ID Embedding for Rahul
        cls.body_rahul = np.zeros(512, dtype=np.float32)
        cls.body_rahul[10:20] = 1.0
        cls.body_rahul /= np.linalg.norm(cls.body_rahul)

        # Synthetic 128-d Face Embedding for Amit
        cls.face_amit = np.zeros(128, dtype=np.float32)
        cls.face_amit[20:25] = 1.0
        cls.face_amit /= np.linalg.norm(cls.face_amit)

        # Synthetic 512-d Body Re-ID Embedding for Amit
        cls.body_amit = np.zeros(512, dtype=np.float32)
        cls.body_amit[30:40] = 1.0
        cls.body_amit /= np.linalg.norm(cls.body_amit)

        # Enroll Rahul
        cls.person_rahul = cls.person_repo.add_person("Rahul")
        cls.embedding_repo.add_face_embedding(cls.person_rahul["person_uuid"], cls.face_rahul, view_label="FRONT")
        cls.embedding_repo.add_body_reid_embedding(cls.person_rahul["person_uuid"], cls.body_rahul, view_label="FRONT_BODY")

        # Enroll Amit
        cls.person_amit = cls.person_repo.add_person("Amit")
        cls.embedding_repo.add_face_embedding(cls.person_amit["person_uuid"], cls.face_amit, view_label="FRONT")
        cls.embedding_repo.add_body_reid_embedding(cls.person_amit["person_uuid"], cls.body_amit, view_label="FRONT_BODY")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def setUp(self):
        self.fusion_engine = MultiModalFusionEngine(threshold=0.65)
        self.enrolled = self.embedding_repo.get_all_active_enrolled_dictionary()

    def test_01_database_multimodal_embeddings(self):
        print("\n[TEST 1] Verifying Multi-Modal Database Storage...")
        f_vecs = self.embedding_repo.get_face_embeddings_for_person(self.person_rahul["person_uuid"])
        b_vecs = self.embedding_repo.get_body_reid_embeddings_for_person(self.person_rahul["person_uuid"])

        self.assertEqual(len(f_vecs), 1)
        self.assertEqual(len(b_vecs), 1)
        self.assertEqual(f_vecs[0]["vector"].shape, (128,))
        self.assertEqual(b_vecs[0]["vector"].shape, (512,))
        print(" -> SFace 128-d Face & OSNet 512-d Body embeddings verified in SQLite database!")

    def test_02_face_plus_body_known_match(self):
        print("\n[TEST 2] Verifying Face + Body Known Match (Rahul)...")
        res = self.fusion_engine.match_multi_modal(
            query_face_emb=self.face_rahul,
            query_body_emb=self.body_rahul,
            enrolled_dict=self.enrolled
        )

        self.assertTrue(res["matched"])
        self.assertEqual(res["display_name"], "Rahul")
        self.assertEqual(res["modality_used"], "FACE+BODY")
        self.assertAlmostEqual(res["final_score"], 1.0, places=4)
        print(f" -> Rahul Matched! Modality: {res['modality_used']} | Final Score: {res['final_score']*100:.1f}%")

    def test_03_body_only_fallback_when_face_turned(self):
        print("\n[TEST 3] Verifying Body Re-ID Fallback when Face is Turned / Unavailable...")
        # Face is None (turned away), Body is Rahul's body
        res = self.fusion_engine.match_multi_modal(
            query_face_emb=None,
            query_body_emb=self.body_rahul,
            enrolled_dict=self.enrolled
        )

        self.assertTrue(res["matched"])
        self.assertEqual(res["display_name"], "Rahul")
        self.assertEqual(res["modality_used"], "BODY_REID_ONLY")
        self.assertAlmostEqual(res["final_score"], 1.0, places=4)
        print(f" -> Body Re-ID Fallback Successful! Person recognized as {res['display_name']} using body appearance only!")

    def test_04_unknown_person_rejection(self):
        print("\n[TEST 4] Verifying Unknown Person Rejection...")
        stranger_face = np.zeros(128, dtype=np.float32)
        stranger_face[100:105] = 1.0
        stranger_face /= np.linalg.norm(stranger_face)

        stranger_body = np.zeros(512, dtype=np.float32)
        stranger_body[200:210] = 1.0
        stranger_body /= np.linalg.norm(stranger_body)

        res = self.fusion_engine.match_multi_modal(
            query_face_emb=stranger_face,
            query_body_emb=stranger_body,
            enrolled_dict=self.enrolled
        )

        self.assertFalse(res["matched"])
        self.assertEqual(res["display_name"], "UNKNOWN")
        self.assertLess(res["final_score"], 0.65)
        print(f" -> Unknown person correctly rejected! Final Score: {res['final_score']*100:.1f}% (< 65%)")


if __name__ == "__main__":
    unittest.main()
