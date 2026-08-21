"""
Automated Test for Pose Yaw Estimation, Motion Detection & AI Validation
"""

import unittest
from app.ui.enroll_page import estimate_head_pose_yaw


class TestPoseAndValidation(unittest.TestCase):
    def test_front_head_pose(self):
        # Nose centered between eyes
        face_dict = {
            "landmarks": [
                [100.0, 100.0],  # Right Eye
                [200.0, 100.0],  # Left Eye
                [150.0, 140.0]   # Nose Tip (Centered at 150)
            ]
        }
        pose, ratio = estimate_head_pose_yaw(face_dict)
        self.assertEqual(pose, "FRONT")
        print(f" -> Front Pose Verified: {pose} (yaw_ratio={ratio:.2f})")

    def test_left_head_pose(self):
        # Nose turned to left eye (nose_x = 185)
        face_dict = {
            "landmarks": [
                [100.0, 100.0],  # Right Eye
                [200.0, 100.0],  # Left Eye
                [185.0, 140.0]   # Nose Tip shifted right
            ]
        }
        pose, ratio = estimate_head_pose_yaw(face_dict)
        self.assertIn(pose, ("LEFT", "PROFILE_LEFT"))
        print(f" -> Left Turned Pose Verified: {pose} (yaw_ratio={ratio:.2f})")

    def test_right_head_pose(self):
        # Nose turned to right eye (nose_x = 115)
        face_dict = {
            "landmarks": [
                [100.0, 100.0],  # Right Eye
                [200.0, 100.0],  # Left Eye
                [115.0, 140.0]   # Nose Tip shifted left
            ]
        }
        pose, ratio = estimate_head_pose_yaw(face_dict)
        self.assertIn(pose, ("RIGHT", "PROFILE_RIGHT"))
        print(f" -> Right Turned Pose Verified: {pose} (yaw_ratio={ratio:.2f})")


if __name__ == "__main__":
    unittest.main()
