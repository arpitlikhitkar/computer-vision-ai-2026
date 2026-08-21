"""
Automated Test for Face Quality Evaluator (Phase 5 Test 5)
"""

import unittest
import numpy as np
from src.recognition.face_quality import evaluate_face_quality


class TestFaceQuality(unittest.TestCase):
    def test_empty_crop(self):
        is_good, score, reason = evaluate_face_quality(None)
        self.assertFalse(is_good)

    def test_too_small_crop(self):
        small_crop = np.zeros((20, 20, 3), dtype=np.uint8)
        is_good, score, reason = evaluate_face_quality(small_crop)
        self.assertFalse(is_good)
        self.assertIn("too small", reason)


if __name__ == "__main__":
    unittest.main()
