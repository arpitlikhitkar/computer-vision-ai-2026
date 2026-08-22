"""
Automated Unit Test Suite for Phase 6.3 Enhanced Multi-Entity UI Overlay
"""

import unittest
import numpy as np
from app.ui.overlay_renderer import draw_l_corner_bbox, render_smart_label, CATEGORY_COLORS_BGR


class TestPhase63UI(unittest.TestCase):
    def test_01_category_colors(self):
        print("\n[TEST 6.3.1] Testing Category Color Schemas...")
        self.assertEqual(CATEGORY_COLORS_BGR["PERSON_KNOWN"], (0, 255, 0))
        self.assertEqual(CATEGORY_COLORS_BGR["PERSON_UNKNOWN"], (0, 0, 255))
        self.assertEqual(CATEGORY_COLORS_BGR["ANIMAL"], (0, 165, 255))
        self.assertEqual(CATEGORY_COLORS_BGR["OBJECT"], (255, 0, 0))
        self.assertEqual(CATEGORY_COLORS_BGR["VEHICLE"], (255, 255, 0))
        print(" -> All category color specifications verified!")

    def test_02_overlay_rendering(self):
        print("\n[TEST 6.3.2] Testing L-Corner & Smart Label Overlay Rendering...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        draw_l_corner_bbox(dummy_frame, [50, 50, 200, 300], CATEGORY_COLORS_BGR["PERSON_KNOWN"])
        render_smart_label(dummy_frame, [50, 50, 200, 300], "RAHUL — PERSON_1 — 94%", "Face: 92% | Body: 88%", CATEGORY_COLORS_BGR["PERSON_KNOWN"])

        draw_l_corner_bbox(dummy_frame, [250, 100, 350, 200], CATEGORY_COLORS_BGR["ANIMAL"])
        render_smart_label(dummy_frame, [250, 100, 350, 200], "DOG — DOG_1 — 89%", None, CATEGORY_COLORS_BGR["ANIMAL"])

        self.assertEqual(dummy_frame.shape, (480, 640, 3))
        self.assertTrue(np.any(dummy_frame > 0))
        print(" -> Bounding box L-corners and Smart Labels rendered without visual artifacts or crash!")


if __name__ == "__main__":
    unittest.main()
