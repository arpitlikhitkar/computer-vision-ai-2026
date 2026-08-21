"""
Unknown Person Detection Repository
Updated with video_clip_path column support
"""

from datetime import datetime, timezone
from app.database.database import get_connection


class UnknownRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def add_unknown_detection(self, snapshot_path, track_id, best_similarity, camera_id="CAM-01", video_clip_path=None):
        now_str = datetime.now(timezone.utc).isoformat()
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO unknown_detections (snapshot_path, video_clip_path, track_id, best_similarity, camera_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?);
            """,
            (snapshot_path, video_clip_path, int(track_id), float(best_similarity), str(camera_id), now_str)
        )
        conn.commit()
        conn.close()

    def get_all_unknowns(self, status_filter="PENDING"):
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("SELECT * FROM unknown_detections WHERE status = ? ORDER BY id DESC;", (status_filter,))
        else:
            cursor.execute("SELECT * FROM unknown_detections ORDER BY id DESC;")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_status(self, unknown_id, new_status):
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE unknown_detections SET status = ? WHERE id = ?;", (new_status, unknown_id))
        conn.commit()
        conn.close()
        return True
