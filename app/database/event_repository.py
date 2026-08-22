"""
Event Repository & Multi-Entity Storage Engine (Phase 6.9 & Backwards Compatibility)

Creates and manages SQLite tables:
- entities
- relationships
- events (with 30-day auto-cleanup trigger)
- 5-second event deduplication
- Fully backwards compatible signature for add_event()
"""

import os
import sqlite3
import uuid
from datetime import datetime
from app.config.settings import config


class EventRepository:
    """
    SQLite Repository for Entities, Spatial Relationships, and Multi-Class Events.
    """
    def __init__(self, db_path=None):
        self.db_path = db_path or config.database_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Tracked Entities Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    bbox_x1 INTEGER, bbox_y1 INTEGER,
                    bbox_x2 INTEGER, bbox_y2 INTEGER,
                    is_active INTEGER DEFAULT 1,
                    metadata TEXT
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_track_id ON entities(track_id);")

            # 2. Spatial Relationships Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    relationship_type TEXT NOT NULL,
                    subject_track_id TEXT NOT NULL,
                    object_track_id TEXT NOT NULL,
                    object_class TEXT,
                    confidence REAL,
                    is_active INTEGER DEFAULT 1,
                    ended_at TEXT
                );
            """)

            # 3. Events Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    priority TEXT DEFAULT 'INFO',
                    subject_track_id TEXT,
                    subject_name TEXT,
                    object_track_id TEXT,
                    object_class TEXT,
                    relationship TEXT,
                    confidence REAL,
                    bbox_x1 INTEGER, bbox_y1 INTEGER,
                    bbox_x2 INTEGER, bbox_y2 INTEGER,
                    frame_number INTEGER,
                    image_path TEXT,
                    metadata TEXT,
                    is_read INTEGER DEFAULT 0
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")

            # Auto-cleanup Trigger (30 days)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS cleanup_old_events
                AFTER INSERT ON events
                BEGIN
                    DELETE FROM events WHERE timestamp < datetime('now', '-30 days');
                END;
            """)
            conn.commit()

    def add_event(self, event_type="INFO", subject_track_id=None, subject_name=None,
                  object_track_id=None, object_class=None, relationship=None,
                  confidence=1.0, priority="INFO", bbox=None, image_path=None, **kwargs):
        """
        Adds structured event with 5-second window deduplication.
        Backwards compatible with legacy kwargs (track_id, person_uuid, recognition_result, similarity_score).
        """
        # Handle legacy positional / keyword arguments
        if "track_id" in kwargs and not subject_track_id:
            subject_track_id = str(kwargs["track_id"])
        if "recognition_result" in kwargs:
            event_type = str(kwargs["recognition_result"])
        if "similarity_score" in kwargs:
            confidence = float(kwargs["similarity_score"])
        if "person_uuid" in kwargs and kwargs["person_uuid"]:
            subject_name = str(kwargs["person_uuid"])

        now_str = datetime.now().isoformat()
        b_x1, b_y1, b_x2, b_y2 = bbox if bbox else (0, 0, 0, 0)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Deduplication: Check if same event occurred within last 5 seconds
            cursor.execute("""
                SELECT id FROM events 
                WHERE event_type = ? AND subject_track_id = ? AND relationship IS ?
                AND timestamp > datetime('now', '-5 seconds')
                LIMIT 1;
            """, (event_type, subject_track_id, relationship))
            existing = cursor.fetchone()
            if existing:
                return existing[0]

            evt_uuid = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO events (
                    event_id, timestamp, event_type, priority, subject_track_id,
                    subject_name, object_track_id, object_class, relationship,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evt_uuid, now_str, event_type, priority, subject_track_id,
                subject_name, object_track_id, object_class, relationship,
                confidence, b_x1, b_y1, b_x2, b_y2, image_path
            ))
            conn.commit()
            return cursor.lastrowid

    def get_events(self, type_filter="ALL", limit=50):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if type_filter == "ALL":
                cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?;", (limit,))
            else:
                cursor.execute("SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?;", (type_filter, limit))

            return [dict(r) for r in cursor.fetchall()]

    def get_todays_count(self):
        """Returns integer count of events recorded today."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events WHERE DATE(timestamp) = DATE('now');")
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_recent_events(self, limit=10):
        """Returns recent events for dashboard widget."""
        return self.get_events(type_filter="ALL", limit=limit)
