"""
SQLite Database Engine for PySide6 Household AI Application
Updated with Auto-Migration Checks for Phase 5.5 Multi-View & Video Clip Columns
"""

import os
import sqlite3
from app.config.settings import config


def get_connection(db_path=None):
    path = db_path or config.database_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database(db_path=None):
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Enrolled Persons Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_uuid TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        display_id TEXT UNIQUE NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # 2. Multi-View Face Embeddings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_uuid TEXT NOT NULL,
        sample_type TEXT NOT NULL DEFAULT 'FACE',
        view_label TEXT NOT NULL DEFAULT 'FRONT',
        quality_score REAL NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_uuid) REFERENCES persons (person_uuid) ON DELETE CASCADE
    );
    """)

    # Auto-migration for face_embeddings table
    cursor.execute("PRAGMA table_info(face_embeddings);")
    face_cols = [r["name"] for r in cursor.fetchall()]
    if "sample_type" not in face_cols:
        cursor.execute("ALTER TABLE face_embeddings ADD COLUMN sample_type TEXT DEFAULT 'FACE';")
    if "view_label" not in face_cols:
        cursor.execute("ALTER TABLE face_embeddings ADD COLUMN view_label TEXT DEFAULT 'FRONT';")

    # 3. Multi-View Person Body Re-ID Embeddings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS person_reid_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_uuid TEXT NOT NULL,
        sample_type TEXT NOT NULL DEFAULT 'BODY',
        view_label TEXT NOT NULL DEFAULT 'FRONT_BODY',
        quality_score REAL NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_uuid) REFERENCES persons (person_uuid) ON DELETE CASCADE
    );
    """)

    # 4. Enrollment Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS enrollment_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_uuid TEXT NOT NULL,
        face_sample_count INTEGER NOT NULL,
        body_sample_count INTEGER NOT NULL,
        face_coverage_pct REAL NOT NULL,
        body_coverage_pct REAL NOT NULL,
        overall_coverage_pct REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_uuid) REFERENCES persons (person_uuid) ON DELETE CASCADE
    );
    """)

    # 5. Unknown Detections Table (with video_clip_path)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS unknown_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_path TEXT NOT NULL,
        video_clip_path TEXT,
        track_id INTEGER NOT NULL,
        best_similarity REAL NOT NULL,
        camera_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        created_at TEXT NOT NULL
    );
    """)

    # Auto-migration for unknown_detections table
    cursor.execute("PRAGMA table_info(unknown_detections);")
    cols = [r["name"] for r in cursor.fetchall()]
    if "video_clip_path" not in cols:
        cursor.execute("ALTER TABLE unknown_detections ADD COLUMN video_clip_path TEXT;")

    # 6. Recognition Event Audit Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recognition_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        person_uuid TEXT,
        recognition_result TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        video_clip_path TEXT,
        timestamp TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()
