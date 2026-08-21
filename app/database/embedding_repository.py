"""
Embedding Repository for Multi-View Face (128-d) and Person Body Re-ID (512-d) Vectors
Updated with safe column selection for backwards compatibility.
"""

from datetime import datetime, timezone
import numpy as np
from app.database.database import get_connection
from app.database.person_repository import PersonRepository


class EmbeddingRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def add_face_embedding(self, person_uuid, embedding_vector, view_label="FRONT", quality_score=100.0):
        """Saves 128-d SFace embedding vector with view label."""
        vector_float32 = np.array(embedding_vector, dtype=np.float32)
        blob_bytes = vector_float32.tobytes()
        now_str = datetime.now(timezone.utc).isoformat()

        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO face_embeddings (person_uuid, sample_type, view_label, quality_score, embedding, created_at)
            VALUES (?, 'FACE', ?, ?, ?, ?);
            """,
            (person_uuid, str(view_label), float(quality_score), blob_bytes, now_str)
        )
        conn.commit()
        conn.close()
        return True

    def add_body_reid_embedding(self, person_uuid, embedding_vector, view_label="FRONT_BODY", quality_score=100.0):
        """Saves 512-d OSNet body Re-ID embedding vector with view label."""
        vector_float32 = np.array(embedding_vector, dtype=np.float32)
        blob_bytes = vector_float32.tobytes()
        now_str = datetime.now(timezone.utc).isoformat()

        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO person_reid_embeddings (person_uuid, sample_type, view_label, quality_score, embedding, created_at)
            VALUES (?, 'BODY', ?, ?, ?, ?);
            """,
            (person_uuid, str(view_label), float(quality_score), blob_bytes, now_str)
        )
        conn.commit()
        conn.close()
        return True

    def get_face_embeddings_for_person(self, person_uuid):
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(face_embeddings);")
        cols = [r["name"] for r in cursor.fetchall()]

        if "view_label" in cols:
            cursor.execute("SELECT embedding, view_label, quality_score FROM face_embeddings WHERE person_uuid = ?;", (person_uuid,))
            rows = cursor.fetchall()
            conn.close()
            return [{
                "vector": np.frombuffer(r["embedding"], dtype=np.float32),
                "view": r["view_label"],
                "quality": r["quality_score"]
            } for r in rows]
        else:
            cursor.execute("SELECT embedding, quality_score FROM face_embeddings WHERE person_uuid = ?;", (person_uuid,))
            rows = cursor.fetchall()
            conn.close()
            return [{
                "vector": np.frombuffer(r["embedding"], dtype=np.float32),
                "view": "FRONT",
                "quality": r["quality_score"]
            } for r in rows]

    def get_body_reid_embeddings_for_person(self, person_uuid):
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(person_reid_embeddings);")
        cols = [r["name"] for r in cursor.fetchall()]

        if not cols:
            conn.close()
            return []

        cursor.execute("SELECT embedding, view_label, quality_score FROM person_reid_embeddings WHERE person_uuid = ?;", (person_uuid,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            "vector": np.frombuffer(r["embedding"], dtype=np.float32),
            "view": r["view_label"],
            "quality": r["quality_score"]
        } for r in rows]

    def get_all_active_enrolled_dictionary(self):
        person_repo = PersonRepository(self.db_path)
        active_persons = person_repo.get_all_persons(status_filter="ACTIVE")

        result = {}
        for p in active_persons:
            uuid_key = p["person_uuid"]
            face_objs = self.get_face_embeddings_for_person(uuid_key)
            body_objs = self.get_body_reid_embeddings_for_person(uuid_key)

            result[uuid_key] = {
                "display_id": p["display_id"],
                "display_name": p["display_name"],
                "status": p["status"],
                "face_embeddings": [f["vector"] for f in face_objs],
                "body_embeddings": [b["vector"] for b in body_objs],
                "face_views": set([f["view"] for f in face_objs]),
                "body_views": set([b["view"] for b in body_objs])
            }

        return result
