"""
Automated Test for Candidate Thresholding and Known/Unknown Matcher (Phase 5 Test 6 & 7)
"""

import unittest
import numpy as np
from src.recognition.face_matcher import FaceMatcher


class TestFaceMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = FaceMatcher(threshold=0.70)

        self.vec_rahul = np.ones(128, dtype=np.float32) / np.sqrt(128)
        self.vec_amit = -np.ones(128, dtype=np.float32) / np.sqrt(128)

        self.enrolled = {
            "uuid-1": {
                "display_id": "PERSON-0001",
                "display_name": "Rahul",
                "status": "ACTIVE",
                "embeddings": [self.vec_rahul]
            },
            "uuid-2": {
                "display_id": "PERSON-0002",
                "display_name": "Amit",
                "status": "ACTIVE",
                "embeddings": [self.vec_amit]
            }
        }

    def test_known_person_match(self):
        query = np.ones(128, dtype=np.float32) / np.sqrt(128)
        res = self.matcher.match_against_enrolled(query, self.enrolled)

        self.assertTrue(res["matched"])
        self.assertEqual(res["display_name"], "Rahul")
        self.assertAlmostEqual(res["similarity"], 1.0, places=4)

    def test_unknown_person_match(self):
        query = np.zeros(128, dtype=np.float32)
        query[0] = 1.0  # Orthogonal to both

        res = self.matcher.match_against_enrolled(query, self.enrolled)

        self.assertFalse(res["matched"])
        self.assertEqual(res["display_name"], "UNKNOWN")
        self.assertLess(res["similarity"], 0.70)


if __name__ == "__main__":
    unittest.main()
