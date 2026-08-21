"""
SQLite Database Initializer and Connection Manager (Phase 5)

Manages local SQLite database (household_ai.db) schema and thread connections.
"""

import os
import sqlite3
from src.config import settings


def get_db_connection(db_path=None):
    """
    Returns an active SQLite database connection with row factory enabled.
    """
    path = db_path or settings.DATABASE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database(db_path=None):
    """
    Creates normalized database tables: persons, face_embeddings, recognition_logs.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Persons Table
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

    # 2. Face Embeddings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_uuid TEXT NOT NULL,
        embedding BLOB NOT NULL,
        quality_score REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_uuid) REFERENCES persons (person_uuid) ON DELETE CASCADE
    );
    """)

    # 3. Recognition Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recognition_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        person_uuid TEXT,
        recognition_result TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        timestamp TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()
