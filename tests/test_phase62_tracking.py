"""
Automated Unit Test Suite for Phase 6.2 Multi-Class Object Tracking
"""

import unittest
from app.ai.multiclass_tracker import MultiClassTracker


class TestPhase62Tracking(unittest.TestCase):
    def test_01_track_id_formatting_and_non_collision(self):
        print("\n[TEST 6.2.1] Testing Category Track ID Formatting & Non-Collision...")
        tracker = MultiClassTracker()

        sample_dets = [
            {'class': 'person', 'category': 'PERSON', 'raw_id': 1},
            {'class': 'dog', 'category': 'ANIMAL', 'raw_id': 1},
            {'class': 'dog', 'category': 'ANIMAL', 'raw_id': 2},
            {'class': 'cell phone', 'category': 'OBJECT', 'raw_id': 1},
            {'class': 'car', 'category': 'VEHICLE', 'raw_id': 1}
        ]

        tracked = tracker.process_tracks(sample_dets)

        track_ids = [t['track_id'] for t in tracked]
        print(f" -> Formatted Track IDs: {track_ids}")

        self.assertIn('PERSON_1', track_ids)
        self.assertIn('DOG_1', track_ids)
        self.assertIn('DOG_2', track_ids)
        self.assertIn('CELL_PHONE_1', track_ids)
        self.assertIn('CAR_1', track_ids)

        # Confirm no collisions across categories
        self.assertEqual(len(track_ids), len(set(track_ids)))
        print(" -> All Category Track IDs are unique with zero collisions!")


if __name__ == "__main__":
    unittest.main()
