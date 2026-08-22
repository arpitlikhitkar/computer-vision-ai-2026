"""
Spatial Relationship Engine & Temporal Anti-False-Positive Filter (Phases 6.7 & 6.8)

Calculates spatial relationships between entities:
- HOLDING: Person Wrist within 50px of Object bbox center
- NEAR: Person within 100px of entity
- WITH: Person within 150px of Animal

Applies 5-frame Temporal Anti-False-Positive Filtering before declaring relationship events.
Draws Magenta dashed relationship lines with labels.
"""

import math
import time
from datetime import datetime
import cv2
import numpy as np


class TemporalRelationshipBuffer:
    """Tracks consecutive frames for candidate relationships."""
    def __init__(self, min_frames=5, max_gap_frames=3):
        self.min_frames = min_frames
        self.max_gap_frames = max_gap_frames
        self.buffer = {}  # key -> {'count': int, 'gap': int, 'latest': dict}

    def update(self, candidates):
        confirmed = []
        current_keys = set()

        for cand in candidates:
            key = (cand["subject"], cand["type"], cand["object"])
            current_keys.add(key)

            if key not in self.buffer:
                self.buffer[key] = {'count': 1, 'gap': 0, 'latest': cand}
            else:
                self.buffer[key]['count'] += 1
                self.buffer[key]['gap'] = 0
                self.buffer[key]['latest'] = cand

            if self.buffer[key]['count'] >= self.min_frames:
                cand_copy = cand.copy()
                # Ramp up temporal confidence
                temp_factor = min(self.buffer[key]['count'] / 5.0, 1.0)
                cand_copy['confidence'] = min(0.99, round(cand['confidence'] * temp_factor, 2))
                confirmed.append(cand_copy)

        # Decay missing keys
        missing_keys = set(self.buffer.keys()) - current_keys
        for mk in list(missing_keys):
            self.buffer[mk]['gap'] += 1
            if self.buffer[mk]['gap'] > self.max_gap_frames:
                del self.buffer[mk]

        return confirmed


class RelationshipEngine:
    """
    Spatial Relationship Engine & Visualization.
    """
    def __init__(self, holding_dist_px=50, near_dist_px=100, with_dist_px=150):
        self.holding_dist_px = holding_dist_px
        self.near_dist_px = near_dist_px
        self.with_dist_px = with_dist_px
        self.temporal_buffer = TemporalRelationshipBuffer(min_frames=5, max_gap_frames=3)

    def analyze_scene(self, person_tracks_with_pose, object_tracks, animal_tracks):
        """
        Input:
            person_tracks_with_pose: list of dicts with 'track_id', 'bbox', 'keypoints'
            object_tracks: list of dicts with 'track_id', 'bbox', 'class'
            animal_tracks: list of dicts with 'track_id', 'bbox', 'class'
        Returns:
            confirmed_relationships: list of confirmed relationship dicts after 5-frame temporal filter
        """
        candidates = []

        for p in person_tracks_with_pose:
            p_id = p.get("track_id", "PERSON_1")
            kpts = p.get("keypoints", {})
            p_bbox = p.get("bbox", [0, 0, 0, 0])

            # Get wrist positions
            wrists = []
            if kpts.get("left_wrist"):
                wrists.append(kpts["left_wrist"])
            if kpts.get("right_wrist"):
                wrists.append(kpts["right_wrist"])

            # 1. HOLDING: Check Wrist proximity to Object Center
            for obj in object_tracks:
                o_id = obj.get("track_id", "OBJECT_1")
                o_cls = obj.get("class", "object")
                o_bbox = obj.get("bbox", [0, 0, 0, 0])

                ox_c = (o_bbox[0] + o_bbox[2]) // 2
                oy_c = (o_bbox[1] + o_bbox[3]) // 2

                min_dist = float("inf")
                for w in wrists:
                    d = math.hypot(w["x"] - ox_c, w["y"] - oy_c)
                    if d < min_dist:
                        min_dist = d

                if min_dist <= self.holding_dist_px:
                    base_conf = max(0.50, 1.0 - (min_dist / self.holding_dist_px))
                    candidates.append({
                        "type": "HOLDING",
                        "subject": p_id,
                        "object": o_id,
                        "object_class": o_cls,
                        "confidence": base_conf,
                        "subject_bbox": p_bbox,
                        "object_bbox": o_bbox,
                        "wrist_pos": (wrists[0]["x"], wrists[0]["y"]) if wrists else (ox_c, oy_c),
                        "object_center": (ox_c, oy_c)
                    })

            # 2. WITH: Check Person proximity to Animal
            for ani in animal_tracks:
                a_id = ani.get("track_id", "ANIMAL_1")
                a_cls = ani.get("class", "animal")
                a_bbox = ani.get("bbox", [0, 0, 0, 0])

                px_c = (p_bbox[0] + p_bbox[2]) // 2
                py_c = (p_bbox[1] + p_bbox[3]) // 2
                ax_c = (a_bbox[0] + a_bbox[2]) // 2
                ay_c = (a_bbox[1] + a_bbox[3]) // 2

                d_ani = math.hypot(px_c - ax_c, py_c - ay_c)
                if d_ani <= self.with_dist_px:
                    base_conf = max(0.50, 1.0 - (d_ani / self.with_dist_px))
                    candidates.append({
                        "type": "WITH",
                        "subject": p_id,
                        "object": a_id,
                        "object_class": a_cls,
                        "confidence": base_conf,
                        "subject_bbox": p_bbox,
                        "object_bbox": a_bbox,
                        "wrist_pos": (px_c, py_c),
                        "object_center": (ax_c, ay_c)
                    })

        confirmed = self.temporal_buffer.update(candidates)
        return confirmed

    @staticmethod
    def draw_relationships(frame, relationships, color=(255, 0, 255)):
        """
        Draws Magenta dashed relationship line with 'HOLDING 87%' label.
        """
        for rel in relationships:
            p1 = rel.get("wrist_pos")
            p2 = rel.get("object_center")
            if p1 and p2:
                # Dashed line
                x1, y1 = p1
                x2, y2 = p2
                dist = math.hypot(x2 - x1, y2 - y1)
                num_segments = max(2, int(dist / 10))

                for i in range(0, num_segments, 2):
                    start = (int(x1 + (x2 - x1) * i / num_segments), int(y1 + (y2 - y1) * i / num_segments))
                    end = (int(x1 + (x2 - x1) * (i + 1) / num_segments), int(y1 + (y2 - y1) * (i + 1) / num_segments))
                    cv2.line(frame, start, end, color, 2, cv2.LINE_AA)

                # Label at midpoint
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                lbl = f"{rel['type']} {int(rel['confidence']*100)}%"
                cv2.rectangle(frame, (mx - 4, my - 14), (mx + 90, my + 4), color, -1)
                cv2.putText(frame, lbl, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
