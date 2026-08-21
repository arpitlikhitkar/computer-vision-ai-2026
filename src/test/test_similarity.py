"""
Automated Test for Cosine Similarity Calculation (Phase 5 Test 3 & 4)
"""

import unittest
import numpy as np
from src.recognition.face_matcher import compute_cosine_similarity


class TestSimilarity(unittest.TestCase):
    def test_self_similarity(self):
        vec = np.ones(128, dtype=np.float32) / np.sqrt(128)
        sim = compute_cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_orthogonal_similarity(self):
        vec_a = np.zeros(128, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(128, dtype=np.float32)
        vec_b[1] = 1.0

        sim = compute_cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(sim, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
