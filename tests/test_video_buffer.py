"""
Automated Test for Pre & Post Event Video Buffer Recorder
"""

import os
import unittest
import numpy as np
from app.services.video_recorder import CircularFrameBuffer, EventVideoRecorderWorker


class TestVideoBufferRecorder(unittest.TestCase):
    def test_circular_frame_buffer_capacity(self):
        buffer = CircularFrameBuffer(max_seconds=2, fps=10)  # maxlen = 20
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        for _ in range(50):
            buffer.append(dummy_frame)

        snapshot = buffer.get_pre_event_snapshot()
        self.assertEqual(len(snapshot), 20)
        print(" -> Circular Pre-Event Buffer capacity (20 frames) verified!")


if __name__ == "__main__":
    unittest.main()
