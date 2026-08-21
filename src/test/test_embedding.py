"""
Automated Test for SFace Face Embedding Extractor (Phase 5 Test 1 & 2)
"""

import unittest
import numpy as np
from src.recognition.face_embedder import SFaceEmbedder


class TestFaceEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedder = SFaceEmbedder()

    def test_embedding_shape_and_norm(self):
        dummy_chip = np.zeros((112, 112, 3), dtype=np.uint8)
        dummy_chip[20:90, 20:90] = [180, 180, 200]

        emb = self.embedder.extract_embedding(dummy_chip)

        self.assertEqual(emb.shape, (128,))
        self.assertEqual(emb.dtype, np.float32)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
