"""
Recognition Event Audit Log Repository
"""

from datetime import datetime, timezone
from app.database.database import get_connection


class EventRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def add_event(self, track_id, person_uuid, recognition_result, similarity_score):
        now_str = datetime.now(timezone.utc).isoformat()
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recognition_events (track_id, person_uuid, recognition_result, similarity_score, timestamp)
            VALUES (?, ?, ?, ?, ?);
            """,
            (int(track_id), person_uuid, str(recognition_result), float(similarity_score), now_str)
        )
        conn.commit()
        conn.close()

    def get_recent_events(self, limit=50):
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recognition_events ORDER BY id DESC LIMIT ?;", (int(limit),))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_todays_count(self):
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM recognition_events WHERE timestamp LIKE ?;", (f"{today_prefix}%",))
        count = cursor.fetchone()[0]
        conn.close()
        return count
