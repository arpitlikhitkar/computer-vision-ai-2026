"""
Professional Multi-Entity Overlay Renderer (Phase 6.3)

Renders:
- Category Color Schemes (Green, Red, Orange, Blue, Cyan, Magenta, Yellow)
- L-Corner Bracket Bounding Boxes with Semi-Transparent Fill
- Smart Non-Overlapping Labels
- Status Indicators (🟢 Known, 🔴 Unknown, 🟡 Tracking)
"""

import cv2
import numpy as np


CATEGORY_COLORS_BGR = {
    "PERSON_KNOWN": (0, 255, 0),     # Green
    "PERSON_UNKNOWN": (0, 0, 255),   # Red
    "ANIMAL": (0, 165, 255),         # Orange
    "OBJECT": (255, 0, 0),           # Blue
    "VEHICLE": (255, 255, 0),        # Cyan
    "RELATIONSHIP": (255, 0, 255),   # Magenta
    "POSE": (0, 255, 255)            # Yellow
}


def draw_l_corner_bbox(frame, bbox, color, thickness=2, corner_length=15, alpha=0.10):
    """
    Draws professional bounding box with L-corner brackets and semi-transparent fill.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return

    # 1. Semi-transparent fill inside box
    if alpha > 0.0:
        sub = frame[y1:y2, x1:x2]
        color_rect = np.full(sub.shape, color, dtype=np.uint8)
        cv2.addWeighted(color_rect, alpha, sub, 1.0 - alpha, 0, dst=sub)

    # 2. Main rectangle border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # 3. L-Corner Brackets
    cl = min(corner_length, (x2 - x1) // 3, (y2 - y1) // 3)
    if cl > 2:
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + cl, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + cl), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - cl, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + cl), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + cl, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - cl), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - cl, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - cl), color, thickness)


def render_smart_label(frame, bbox, line1_str, line2_str=None, color=(0, 255, 0), y_offset=0):
    """
    Renders clean, readable smart label box above/below bounding box with collision offset.
    """
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame.shape[:2]

    (tw1, th1), _ = cv2.getTextSize(line1_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tw2, th2 = 0, 0
    if line2_str:
        (tw2, th2), _ = cv2.getTextSize(line2_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

    max_w = max(tw1, tw2) + 12
    total_h = th1 + (th2 + 6 if line2_str else 0) + 10

    lbl_x1 = max(0, min(x1, frame_w - max_w))
    lbl_y1 = max(0, y1 - total_h - y_offset)

    if lbl_y1 < 0:
        lbl_y1 = min(y2 + y_offset, frame_h - total_h)

    # Background Box
    cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x1 + max_w, lbl_y1 + total_h), color, -1)
    cv2.putText(frame, line1_str, (lbl_x1 + 6, lbl_y1 + th1 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if line2_str:
        cv2.putText(frame, line2_str, (lbl_x1 + 6, lbl_y1 + th1 + th2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
