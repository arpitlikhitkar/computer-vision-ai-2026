"""
Pre-Event & Post-Event Rolling Video Buffer Recorder (Phase 5.5)

Maintains in-memory circular ring buffer of last 30 seconds (-30s pre-event)
and records next 30 seconds (+30s post-event) to generate complete 60-second MP4 video clips.
Saves every unknown person into a dedicated unique folder: data/unknown/UNKNOWN_TRACK_{track_id}/
"""

import os
import time
import shutil
import collections
import cv2
from PySide6.QtCore import QThread, Signal
from app.config.settings import config
from app.database.unknown_repository import UnknownRepository


class CircularFrameBuffer:
    """Thread-safe circular frame buffer holding last N seconds of frames."""
    def __init__(self, max_seconds=30, fps=15):
        self.maxlen = max_seconds * fps
        self.buffer = collections.deque(maxlen=self.maxlen)

    def append(self, frame_bgr):
        if frame_bgr is not None:
            self.buffer.append(frame_bgr.copy())

    def get_pre_event_snapshot(self):
        return list(self.buffer)


class EventVideoRecorderWorker(QThread):
    """Background QThread worker writing 60s MP4 video file (-30s pre + +30s post)."""
    recording_finished = Signal(str)

    def __init__(self, pre_event_frames, track_id, fps=15, parent=None):
        super().__init__(parent)
        self.pre_event_frames = pre_event_frames
        self.track_id = track_id
        self.fps = fps
        self.post_event_frames = []
        self.recording_post = True
        self.unknown_repo = UnknownRepository()

    def add_post_event_frame(self, frame_bgr):
        if self.recording_post and frame_bgr is not None:
            self.post_event_frames.append(frame_bgr.copy())
            if len(self.post_event_frames) >= (30 * self.fps):
                self.recording_post = False

    def run(self):
        timestamp_int = int(time.time())
        
        # Unique Folder per Unknown Person/Track ID: data/unknown/UNKNOWN_TRACK_{track_id}/
        unknown_person_dir = os.path.join(config.UNKNOWN_DIR, f"UNKNOWN_TRACK_{self.track_id:04d}")
        os.makedirs(unknown_person_dir, exist_ok=True)

        video_filename = f"video_60s_event_{timestamp_int}.mp4"
        video_path = os.path.join(unknown_person_dir, video_filename)

        # Global recordings copy
        recordings_dir = os.path.join(config.DATA_DIR, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        global_rec_path = os.path.join(recordings_dir, f"unknown_t{self.track_id}_{timestamp_int}.mp4")

        wait_start = time.time()
        while self.recording_post and (time.time() - wait_start) < 35.0:
            time.sleep(0.1)

        all_frames = self.pre_event_frames + self.post_event_frames
        if not all_frames:
            return

        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, float(self.fps), (w, h))

        for f in all_frames:
            writer.write(f)

        writer.release()
        print(f"[VIDEO RECORDER] Saved 60s Unknown Event Clip to unique folder: {video_path}")

        # Copy clip to global recordings directory
        try:
            shutil.copy2(video_path, global_rec_path)
        except Exception:
            pass

        # Save snapshot image into unique folder
        snap_path = os.path.join(unknown_person_dir, f"snapshot_t{self.track_id}_{timestamp_int}.jpg")
        cv2.imwrite(snap_path, all_frames[min(len(self.pre_event_frames), len(all_frames)-1)])

        self.unknown_repo.add_unknown_detection(
            snapshot_path=snap_path,
            track_id=self.track_id,
            best_similarity=0.50,
            video_clip_path=video_path
        )

        self.recording_finished.emit(video_path)
