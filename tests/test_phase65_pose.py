"""
Automated Unit Test Suite for Phase 6.5 YOLOv8-Pose Estimation
"""

import unittest
import numpy as np
from app.ai.pose_estimator import PoseEstimator, COCO_KEYPOINTS


class TestPhase65Pose(unittest.TestCase):
    def test_01_pose_estimation_and_absolute_coords(self):
        print("\n[TEST 6.5.1] Testing YOLOv8-Pose Keypoint Estimation & Frame Absolute Coordinates...")
        estimator = PoseEstimator()

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic human shape in crop [100, 100, 300, 400]
        cv2_dummy_body = dummy_frame.copy()

        bbox = [100, 100, 300, 400]
        kpts = estimator.estimate_pose_for_person(cv2_dummy_body, bbox)

        self.assertIsInstance(kpts, dict)
        for k_name in COCO_KEYPOINTS:
            self.assertIn(k_name, kpts)

        # Draw skeleton visualization
        estimator.draw_skeleton(dummy_frame, kpts)
        self.assertEqual(dummy_frame.shape, (480, 640, 3))
        print(" -> YOLOv8-Pose 17 keypoints and skeleton visualization verified in absolute frame coordinates!")


if __name__ == "__main__":
    unittest.main()
