"""
Recognition Log Repository Module (Phase 5)

Manages historical recognition audit logs in SQLite database.
"""

from datetime import datetime, timezone
from src.storage.database import get_db_connection


class LogRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def log_recognition_event(self, track_id, person_uuid, recognition_result, similarity_score):
        """
        Logs a face recognition event (KNOWN, UNKNOWN, LOW_QUALITY) to DB.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recognition_logs (track_id, person_uuid, recognition_result, similarity_score, timestamp)
            VALUES (?, ?, ?, ?, ?);
            """,
            (int(track_id), person_uuid, str(recognition_result), float(similarity_score), now_str)
        )
        conn.commit()
        conn.close()

    def get_recent_logs(self, limit=20):
        """
        Returns recent recognition audit logs.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM recognition_logs ORDER BY id DESC LIMIT ?;",
            (int(limit),)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
