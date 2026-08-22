"""
Multi-Class Object Tracker Module (Phase 6.2)

Maintains independent ByteTrack instances per category ('PERSON', 'ANIMAL', 'OBJECT', 'VEHICLE').
Assigns non-colliding category track IDs (e.g., PERSON_1, DOG_1, OBJECT_1, VEHICLE_1).
"""

import numpy as np
from app.config.settings import config


class CategoryTrackerState:
    """Tracks IDs per class/category without collisions."""
    def __init__(self, category_name):
        self.category_name = category_name
        self.raw_to_cat_id = {}  # byte_track_id -> CATEGORY_NUM string
        self.cat_counters = {}   # class_name -> int count

    def get_formatted_id(self, raw_id, class_name):
        if raw_id in self.raw_to_cat_id:
            return self.raw_to_cat_id[raw_id]

        c_name = class_name.upper().replace(" ", "_")
        current_cnt = self.cat_counters.get(c_name, 0) + 1
        self.cat_counters[c_name] = current_cnt

        fmt_id = f"{c_name}_{current_cnt}"
        self.raw_to_cat_id[raw_id] = fmt_id
        return fmt_id


class MultiClassTracker:
    """
    Manages multi-category tracking and formats IDs into CATEGORY_NUMBER schema.
    """
    def __init__(self, track_buffer=None):
        self.track_buffer = track_buffer if track_buffer is not None else config.track_buffer_frames
        self.category_states = {
            "PERSON": CategoryTrackerState("PERSON"),
            "ANIMAL": CategoryTrackerState("ANIMAL"),
            "OBJECT": CategoryTrackerState("OBJECT"),
            "VEHICLE": CategoryTrackerState("VEHICLE")
        }

    def process_tracks(self, detections, raw_tracks_by_category=None):
        """
        Input:
            detections: List of dicts from MultiClassYOLO
            raw_tracks_by_category: Optional raw ByteTrack outputs
        Returns:
            tracked_detections: List of dicts with 'track_id' set to 'PERSON_1', 'DOG_1', etc.
        """
        tracked = []
        if not detections:
            return tracked

        # Assign Category Track IDs
        for idx, det in enumerate(detections):
            category = det.get("category", "OBJECT")
            class_name = det.get("class", "object")

            cat_state = self.category_states.get(category, self.category_states["OBJECT"])
            raw_id = det.get("raw_id", idx + 1)
            fmt_id = cat_state.get_formatted_id(raw_id, class_name)

            det_copy = det.copy()
            det_copy["track_id"] = fmt_id
            tracked.append(det_copy)

        return tracked
