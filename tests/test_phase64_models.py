"""
Automated Unit Test Suite for Phase 6.4 AI Model Management & Dependency Engine
"""

import unittest
from app.ai.model_registry import ModelRegistry


class TestPhase64Models(unittest.TestCase):
    def test_01_dependency_rules_and_blocks(self):
        print("\n[TEST 6.4.1] Testing Model Load & Unload Dependency Rules...")
        reg = ModelRegistry()

        # Pose Estimator requires YOLOv8n (YOLOv8n is LOADED)
        can_pose, _ = reg.can_load("yolo_pose")
        self.assertTrue(can_pose)

        # Unload OSNet first (which depends on YOLOv8n)
        reg.set_model_status("osnet", "UNLOADED")

        # Now unload YOLOv8n (since no active loaded model depends on it)
        can_unload_yolo, _ = reg.can_unload("yolov8n")
        self.assertTrue(can_unload_yolo)

        # Re-load YOLOv8n and load Pose Estimator
        reg.set_model_status("yolov8n", "LOADED")
        reg.set_model_status("yolo_pose", "LOADED")

        # Now try to unload YOLOv8n while Pose Estimator is LOADED (Should be BLOCKED!)
        can_unload_yolo_blocked, msg = reg.can_unload("yolov8n")
        self.assertFalse(can_unload_yolo_blocked)
        print(f" -> Reverse dependency block verified: {msg}")

    def test_02_capability_matrix(self):
        print("\n[TEST 6.4.2] Testing System Capability Matrix...")
        reg = ModelRegistry()
        caps = reg.get_capability_matrix()
        self.assertIsInstance(caps, list)
        self.assertTrue(len(caps) >= 6)
        print(" -> Capability Matrix generated cleanly!")


if __name__ == "__main__":
    unittest.main()
