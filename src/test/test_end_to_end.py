"""
End-to-End System Test for Phase 5 Household Face Recognition

Tests:
1. Database Schema & Tables Initialization
2. Member Enrollment (Rahul & Amit)
3. Embedding Generation & Storage in SQLite
4. Known Match Verification (Rahul -> Green Match)
5. Known Match Verification (Amit -> Green Match)
6. Unknown Person Verification (Non-enrolled -> Red UNKNOWN)
7. Member Deactivation (Deactivate Amit -> Returns UNKNOWN)
8. Multi-Person Simultaneous Matching (Rahul + Amit + Unknown)
9. Historical Recognition Audit Logging
"""

import os
import unittest
import numpy as np

from src.storage.database import initialize_database
from src.storage.person_repository import PersonRepository
from src.storage.embedding_repository import EmbeddingRepository
from src.storage.log_repository import LogRepository
from src.recognition.face_matcher import FaceMatcher


class TestPhase5EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = "outputs/database/test_household_ai.db"
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

        initialize_database(cls.db_path)
        cls.person_repo = PersonRepository(cls.db_path)
        cls.embedding_repo = EmbeddingRepository(cls.db_path)
        cls.log_repo = LogRepository(cls.db_path)

        # Create synthetic 128-d normalized embeddings for Rahul
        cls.emb_rahul_base = np.zeros(128, dtype=np.float32)
        cls.emb_rahul_base[0:10] = [0.8, 0.2, 0.1, 0.0, 0.5, 0.1, 0.3, 0.2, 0.1, 0.1]
        cls.emb_rahul_base /= np.linalg.norm(cls.emb_rahul_base)

        # Add slight variations for Rahul's 10 enrollment samples
        cls.rahul_samples = []
        for i in range(10):
            noise = np.random.normal(0, 0.02, 128).astype(np.float32)
            sample = cls.emb_rahul_base + noise
            sample /= np.linalg.norm(sample)
            cls.rahul_samples.append(sample)

        # Create synthetic 128-d normalized embeddings for Amit
        cls.emb_amit_base = np.zeros(128, dtype=np.float32)
        cls.emb_amit_base[20:30] = [0.1, 0.9, 0.2, 0.4, 0.1, 0.7, 0.1, 0.3, 0.1, 0.2]
        cls.emb_amit_base /= np.linalg.norm(cls.emb_amit_base)

        cls.amit_samples = []
        for i in range(10):
            noise = np.random.normal(0, 0.02, 128).astype(np.float32)
            sample = cls.emb_amit_base + noise
            sample /= np.linalg.norm(sample)
            cls.amit_samples.append(sample)

        # Enroll Rahul
        cls.person_rahul = cls.person_repo.add_person("Rahul")
        for s in cls.rahul_samples:
            cls.embedding_repo.add_embedding(cls.person_rahul["person_uuid"], s, quality_score=95.0)

        # Enroll Amit
        cls.person_amit = cls.person_repo.add_person("Amit")
        for s in cls.amit_samples:
            cls.embedding_repo.add_embedding(cls.person_amit["person_uuid"], s, quality_score=92.0)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def setUp(self):
        self.matcher = FaceMatcher(threshold=0.65)
        self.enrolled = self.embedding_repo.get_all_active_enrolled_dictionary()

    def test_01_database_enrollment(self):
        print("\n[TEST 1] Verifying Database Enrollment Records...")
        persons = self.person_repo.get_all_persons()
        self.assertEqual(len(persons), 2)
        self.assertEqual(persons[0]["display_name"], "Rahul")
        self.assertEqual(persons[1]["display_name"], "Amit")

        rahul_embs = self.embedding_repo.get_embeddings_for_person(self.person_rahul["person_uuid"])
        self.assertEqual(len(rahul_embs), 10)
        print(" -> Rahul and Amit enrolled successfully in SQLite database!")

    def test_02_match_rahul_known(self):
        print("\n[TEST 2] Verifying Known Person Match (Rahul -> Green)...")
        query_rahul = self.emb_rahul_base + np.random.normal(0, 0.01, 128).astype(np.float32)
        query_rahul /= np.linalg.norm(query_rahul)

        res = self.matcher.match_against_enrolled(query_rahul, self.enrolled)
        self.assertTrue(res["matched"])
        self.assertEqual(res["display_name"], "Rahul")
        self.assertGreaterEqual(res["similarity"], 0.85)
        print(f" -> Rahul matched successfully! Similarity: {res['similarity']*100:.1f}%")

    def test_03_match_amit_known(self):
        print("\n[TEST 3] Verifying Known Person Match (Amit -> Green)...")
        query_amit = self.emb_amit_base + np.random.normal(0, 0.01, 128).astype(np.float32)
        query_amit /= np.linalg.norm(query_amit)

        res = self.matcher.match_against_enrolled(query_amit, self.enrolled)
        self.assertTrue(res["matched"])
        self.assertEqual(res["display_name"], "Amit")
        self.assertGreaterEqual(res["similarity"], 0.85)
        print(f" -> Amit matched successfully! Similarity: {res['similarity']*100:.1f}%")

    def test_04_match_unknown_person(self):
        print("\n[TEST 4] Verifying Unknown Person Match (Non-enrolled -> Red)...")
        query_unknown = np.zeros(128, dtype=np.float32)
        query_unknown[50:60] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        query_unknown /= np.linalg.norm(query_unknown)

        res = self.matcher.match_against_enrolled(query_unknown, self.enrolled)
        self.assertFalse(res["matched"])
        self.assertEqual(res["display_name"], "UNKNOWN")
        self.assertLess(res["similarity"], 0.65)
        print(f" -> Unknown person correctly rejected! Best Similarity: {res['similarity']*100:.1f}% (< 65%)")

    def test_05_member_deactivation(self):
        print("\n[TEST 5] Verifying Member Deactivation...")
        # Deactivate Amit
        self.person_repo.update_person_status(self.person_amit["person_uuid"], "INACTIVE")
        enrolled_active = self.embedding_repo.get_all_active_enrolled_dictionary()

        query_amit = self.emb_amit_base
        res = self.matcher.match_against_enrolled(query_amit, enrolled_active)

        self.assertFalse(res["matched"])
        self.assertEqual(res["display_name"], "UNKNOWN")
        print(" -> Inactive member Amit correctly ignored during recognition!")

        # Re-activate Amit
        self.person_repo.update_person_status(self.person_amit["person_uuid"], "ACTIVE")

    def test_06_multi_person_simultaneous_matching(self):
        print("\n[TEST 6] Verifying Multi-Person Simultaneous Frame Matching...")
        query_rahul = self.emb_rahul_base
        query_amit = self.emb_amit_base
        query_stranger = np.zeros(128, dtype=np.float32)
        query_stranger[80:90] = 1.0
        query_stranger /= np.linalg.norm(query_stranger)

        # Track 1: Rahul
        res1 = self.matcher.match_against_enrolled(query_rahul, self.enrolled)
        # Track 2: Amit
        res2 = self.matcher.match_against_enrolled(query_amit, self.enrolled)
        # Track 3: Stranger
        res3 = self.matcher.match_against_enrolled(query_stranger, self.enrolled)

        self.assertEqual(res1["display_name"], "Rahul")
        self.assertEqual(res2["display_name"], "Amit")
        self.assertEqual(res3["display_name"], "UNKNOWN")

        print(" -> Multi-person frame matching verified!")
        print(f"    Track 1: {res1['display_name']} ({res1['similarity']*100:.1f}%)")
        print(f"    Track 2: {res2['display_name']} ({res2['similarity']*100:.1f}%)")
        print(f"    Track 3: {res3['display_name']} ({res3['similarity']*100:.1f}%)")

    def test_07_audit_logging(self):
        print("\n[TEST 7] Verifying Recognition Audit Logging...")
        self.log_repo.log_recognition_event(1, self.person_rahul["person_uuid"], "KNOWN", 0.95)
        self.log_repo.log_recognition_event(2, self.person_amit["person_uuid"], "KNOWN", 0.92)
        self.log_repo.log_recognition_event(3, None, "UNKNOWN", 0.45)

        logs = self.log_repo.get_recent_logs(3)
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]["recognition_result"], "UNKNOWN")
        self.assertEqual(logs[1]["recognition_result"], "KNOWN")
        print(" -> Audit logging verified in SQLite database!")


if __name__ == "__main__":
    unittest.main()
