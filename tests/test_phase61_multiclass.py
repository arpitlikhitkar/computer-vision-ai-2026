"""
Automated Unit & Integration Test Suite for Phase 6.1 Multi-Class Detection
"""

import unittest
import numpy as np
from app.ai.category_mapper import categorize_class, COCO_CATEGORIES
from app.ai.yolo_detector import MultiClassYOLO


class TestPhase61MultiClass(unittest.TestCase):
    def test_01_coco_category_mapping(self):
        print("\n[TEST 6.1.1] Testing COCO 80-Class Categorization...")
        self.assertEqual(categorize_class("person"), "PERSON")
        self.assertEqual(categorize_class("dog"), "ANIMAL")
        self.assertEqual(categorize_class("cat"), "ANIMAL")
        self.assertEqual(categorize_class("cell phone"), "OBJECT")
        self.assertEqual(categorize_class("bottle"), "OBJECT")
        self.assertEqual(categorize_class("car"), "VEHICLE")
        self.assertEqual(categorize_class("bicycle"), "VEHICLE")
        print(" -> All COCO categories mapped correctly!")

    def test_02_empty_frame_handling(self):
        print("\n[TEST 6.1.2] Testing Empty Frame Handling...")
        detector = MultiClassYOLO()
        res_empty = detector.detect(None)
        self.assertEqual(res_empty, [])

        dummy_zero = np.zeros((0, 0, 3), dtype=np.uint8)
        res_zero = detector.detect(dummy_zero)
        self.assertEqual(res_zero, [])
        print(" -> Empty frame handled gracefully without crash!")

    def test_03_detection_dict_format(self):
        print("\n[TEST 6.1.3] Testing Detection Output Dict Schema...")
        detector = MultiClassYOLO()
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame)
        self.assertIsInstance(dets, list)

        for d in dets:
            self.assertIn('class', d)
            self.assertIn('confidence', d)
            self.assertIn('bbox', d)
            self.assertIn('track_id', d)
            self.assertIn('category', d)
            self.assertIn(d['category'], ('PERSON', 'ANIMAL', 'OBJECT', 'VEHICLE'))
        print(f" -> Output format verified with all 5 required fields!")


if __name__ == "__main__":
    unittest.main()
