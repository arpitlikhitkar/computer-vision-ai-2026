"""
Person Repository Module (Phase 5)

Handles CRUD operations for Household Persons in SQLite database.
"""

import uuid
from datetime import datetime, timezone
from src.storage.database import get_db_connection


class PersonRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def add_person(self, display_name, display_id=None):
        """
        Creates a new person record with UUID and auto-generated display_id (e.g. PERSON-0001).
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        person_uuid = str(uuid.uuid4())
        now_str = datetime.now(timezone.utc).isoformat()

        if display_id is None:
            cursor.execute("SELECT COUNT(*) FROM persons;")
            count = cursor.fetchone()[0] + 1
            display_id = f"PERSON-{count:04d}"

        cursor.execute(
            """
            INSERT INTO persons (person_uuid, display_name, display_id, status, created_at, updated_at)
            VALUES (?, ?, ?, 'ACTIVE', ?, ?);
            """,
            (person_uuid, display_name, display_id, now_str, now_str)
        )
        conn.commit()
        conn.close()

        return {
            "person_uuid": person_uuid,
            "display_name": display_name,
            "display_id": display_id,
            "status": "ACTIVE",
            "created_at": now_str,
            "updated_at": now_str
        }

    def get_person_by_uuid(self, person_uuid):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE person_uuid = ?;", (person_uuid,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None
        return dict(row)

    def get_all_persons(self, status_filter=None):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("SELECT * FROM persons WHERE status = ? ORDER BY id ASC;", (status_filter,))
        else:
            cursor.execute("SELECT * FROM persons ORDER BY id ASC;")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_person_status(self, person_uuid, new_status):
        """Updates status between 'ACTIVE' and 'INACTIVE'."""
        if new_status not in ("ACTIVE", "INACTIVE"):
            raise ValueError("Status must be 'ACTIVE' or 'INACTIVE'")

        now_str = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE persons SET status = ?, updated_at = ? WHERE person_uuid = ?;",
            (new_status, now_str, person_uuid)
        )
        conn.commit()
        conn.close()
        return True

    def delete_person(self, person_uuid):
        """Hard deletes a person record and associated embeddings."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM persons WHERE person_uuid = ?;", (person_uuid,))
        conn.commit()
        conn.close()
        return True
