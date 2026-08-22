"""
Automated Unit Test Suite for Phase 6.7 & 6.8 Spatial Relationship Engine & Temporal Anti-False-Positive Filtering
"""

import unittest
import numpy as np
from app.ai.relationship_engine import RelationshipEngine


class TestPhase67Relationships(unittest.TestCase):
    def test_01_holding_detection_and_temporal_filtering(self):
        print("\n[TEST 6.7.1 & 6.8.1] Testing HOLDING Spatial Relationship & 5-Frame Temporal Confirmation...")
        engine = RelationshipEngine(holding_dist_px=50)

        person_pose = [{
            'track_id': 'PERSON_1',
            'bbox': [100, 100, 300, 400],
            'keypoints': {
                'right_wrist': {'x': 200, 'y': 250, 'confidence': 0.90}
            }
        }]

        phone_obj = [{
            'track_id': 'CELL_PHONE_1',
            'class': 'cell phone',
            'bbox': [190, 240, 220, 270]  # center at (205, 255) -> dist to wrist ~7px (<50px)
        }]

        # Frame 1 to 4: Candidate exists, but confirmed list is EMPTY (temporal check prevents false positive!)
        for frame_idx in range(1, 5):
            rels = engine.analyze_scene(person_pose, phone_obj, [])
            self.assertEqual(rels, [], f"Frame {frame_idx} should NOT trigger event before 5 frames!")
            print(f" -> Frame {frame_idx}/5: Correctly withheld candidate (Temporal Anti-False-Positive active).")

        # Frame 5: Confirmed!
        rels_5 = engine.analyze_scene(person_pose, phone_obj, [])
        self.assertEqual(len(rels_5), 1)
        self.assertEqual(rels_5[0]["type"], "HOLDING")
        self.assertEqual(rels_5[0]["subject"], "PERSON_1")
        self.assertEqual(rels_5[0]["object"], "CELL_PHONE_1")
        print(" -> Frame 5/5: HOLDING relationship event CONFIRMED after 5 consecutive frames!")

        # Visual rendering test
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.draw_relationships(dummy_frame, rels_5)
        self.assertTrue(np.any(dummy_frame > 0))
        print(" -> Magenta dashed relationship line & label rendered cleanly!")


if __name__ == "__main__":
    unittest.main()
