"""
Embedding Repository Module (Phase 5)

Manages binary blob serialization and retrieval of 128-d face embedding vectors.
"""

from datetime import datetime, timezone
import numpy as np
from src.storage.database import get_db_connection
from src.storage.person_repository import PersonRepository


class EmbeddingRepository:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def add_embedding(self, person_uuid, embedding_vector, quality_score=100.0):
        """
        Serializes float32 numpy array vector to binary BLOB and saves in DB.
        """
        if embedding_vector is None or len(embedding_vector) == 0:
            raise ValueError("[ERROR] Empty embedding vector passed to add_embedding.")

        vector_float32 = np.array(embedding_vector, dtype=np.float32)
        blob_bytes = vector_float32.tobytes()
        now_str = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO face_embeddings (person_uuid, embedding, quality_score, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (person_uuid, blob_bytes, float(quality_score), now_str)
        )
        conn.commit()
        conn.close()
        return True

    def get_embeddings_for_person(self, person_uuid):
        """
        Retrieves and deserializes all binary embedding BLOBs for a person into list of float32 vectors.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT embedding, quality_score FROM face_embeddings WHERE person_uuid = ?;",
            (person_uuid,)
        )
        rows = cursor.fetchall()
        conn.close()

        vectors = []
        for r in rows:
            blob = r["embedding"]
            vec = np.frombuffer(blob, dtype=np.float32)
            vectors.append(vec)

        return vectors

    def get_all_active_enrolled_dictionary(self):
        """
        Returns structured dictionary of all ACTIVE household members and their reference vectors
        for real-time memory matching:
        {
          person_uuid: {
            "display_id": "PERSON-0001",
            "display_name": "Rahul",
            "status": "ACTIVE",
            "embeddings": [vec1, vec2, ...]
          }, ...
        }
        """
        person_repo = PersonRepository(self.db_path)
        active_persons = person_repo.get_all_persons(status_filter="ACTIVE")

        result = {}
        for p in active_persons:
            uuid_key = p["person_uuid"]
            vecs = self.get_embeddings_for_person(uuid_key)
            result[uuid_key] = {
                "display_id": p["display_id"],
                "display_name": p["display_name"],
                "status": p["status"],
                "embeddings": vecs
            }

        return result
