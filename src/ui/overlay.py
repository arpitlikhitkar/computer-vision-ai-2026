"""
UI Overlay Module (Phase 5)

Renders Green (Known), Red (Unknown), and Orange (Quality Low) bounding boxes
and system headers on video stream frames.
"""

import cv2


def render_face_recognition_overlay(
    display_frame,
    track_id,
    person_bbox,
    face_bbox,
    state_dict,
    device_label="CPU Mode",
    fps=0.0,
    active_tracks_count=0,
    known_count=0,
    unknown_count=0
):
    """
    Renders Green/Red/Orange bounding boxes and metadata labels for a tracked person.
    """
    if display_frame is None or display_frame.size == 0 or person_bbox is None:
        return

    x1, y1, x2, y2 = map(int, person_bbox)

    rec_state = state_dict.get("state", "PENDING")
    disp_name = state_dict.get("display_name", "UNKNOWN")
    disp_id = state_dict.get("display_id", "")
    similarity = state_dict.get("similarity", 0.0)
    quality_msg = state_dict.get("quality_msg", "")

    # Color Scheme Rules:
    # KNOWN    -> Bright Green (0, 255, 127)
    # UNKNOWN  -> Crimson Red (0, 0, 255)
    # QUALITY_LOW / PENDING -> Orange (0, 165, 255)
    if rec_state == "KNOWN":
        box_color = (0, 255, 127)  # Bright Emerald Green
        status_label = f"{disp_name} ({disp_id})" if disp_id else disp_name
        sim_label = f"Match: {similarity * 100:.0f}%"
    elif rec_state == "UNKNOWN":
        box_color = (0, 0, 255)    # Bright Crimson Red
        status_label = "UNKNOWN"
        sim_label = f"Similarity: {similarity * 100:.0f}%"
    elif rec_state == "QUALITY_LOW":
        box_color = (0, 165, 255)  # Orange
        status_label = "FACE QUALITY LOW"
        sim_label = quality_msg if quality_msg else "Adjust Position"
    else:
        box_color = (0, 215, 255)  # Gold
        status_label = "EVALUATING"
        sim_label = f"Sim: {similarity * 100:.0f}%" if similarity > 0 else "Analyzing..."

    # Draw Bounding Box around Person
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

    # Draw Face Bounding Box if detected inside person crop
    if face_bbox is not None:
        fx, fy, fw, fh = map(int, face_bbox)
        # Convert face bbox relative to person bbox
        abs_fx = x1 + fx
        abs_fy = y1 + fy
        cv2.rectangle(display_frame, (abs_fx, abs_fy), (abs_fx + fw, abs_fy + fh), (255, 255, 255), 1)

    # Draw Labels
    line1 = f"{status_label} | Track: {track_id}"
    line2 = f"{sim_label}"

    (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    max_w = max(tw1, tw2) + 12
    total_h = th1 + th2 + 12

    lbl_y1 = max(0, y1 - total_h)
    cv2.rectangle(display_frame, (x1, lbl_y1), (x1 + max_w, lbl_y1 + total_h), box_color, -1)

    cv2.putText(display_frame, line1, (x1 + 5, lbl_y1 + th1 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(display_frame, line2, (x1 + 5, lbl_y1 + th1 + th2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def render_system_header(display_frame, device_label, fps, active_tracks, known_count, unknown_count):
    """
    Renders top system performance bar.
    """
    header_text = f"Device: {device_label.upper()} | FPS: {fps:.1f} | Active Tracks: {active_tracks} | Known: {known_count} | Unknown: {unknown_count}"
    cv2.rectangle(display_frame, (10, 10), (660, 45), (30, 30, 30), -1)
    cv2.putText(display_frame, header_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
