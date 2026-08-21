"""
Face Matcher Module (Phase 5)

Computes Cosine Similarity between face embeddings and matches query embeddings
against enrolled household member database records.
"""

import numpy as np
from src.config import settings


def compute_cosine_similarity(emb1, emb2):
    """
    Computes Cosine Similarity between two L2-normalized vectors (dot product).
    Returns float value in range [-1.0, 1.0].
    """
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2))


class FaceMatcher:
    """
    Matches query face embeddings against enrolled household members.
    """
    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else settings.RECOGNITION_THRESHOLD

    def match_against_enrolled(self, query_embedding, enrolled_persons_dict):
        """
        Compares query embedding against all enrolled household members.
        Input:
            query_embedding: 128-d float32 array
            enrolled_persons_dict: {
                person_uuid: {
                    "display_id": "PERSON-0001",
                    "display_name": "Rahul",
                    "status": "ACTIVE",
                    "embeddings": [emb1, emb2, ...]
                }, ...
            }
        Returns:
            best_match_dict: {
                "matched": bool,
                "person_uuid": str or None,
                "display_id": str or None,
                "display_name": str or None,
                "similarity": float,
                "candidate_scores": {display_name: similarity, ...}
            }
        """
        if query_embedding is None or not enrolled_persons_dict:
            return {
                "matched": False,
                "person_uuid": None,
                "display_id": None,
                "display_name": "UNKNOWN",
                "similarity": 0.0,
                "candidate_scores": {}
            }

        candidate_scores = {}
        best_uuid = None
        best_display_id = None
        best_display_name = None
        highest_similarity = 0.0

        for uuid, person_data in enrolled_persons_dict.items():
            if person_data.get("status") != "ACTIVE":
                continue  # Skip inactive household members

            embeddings = person_data.get("embeddings", [])
            if not embeddings:
                continue

            # Compute Top-3 average Cosine Similarity for stability
            sims = [compute_cosine_similarity(query_embedding, ref_emb) for ref_emb in embeddings]
            sims.sort(reverse=True)
            top_k = sims[:min(3, len(sims))]
            avg_sim = float(np.mean(top_k))

            disp_name = person_data.get("display_name", "UNKNOWN")
            candidate_scores[disp_name] = avg_sim

            if avg_sim > highest_similarity:
                highest_similarity = avg_sim
                best_uuid = uuid
                best_display_id = person_data.get("display_id")
                best_display_name = disp_name

        is_matched = highest_similarity >= self.threshold

        return {
            "matched": is_matched,
            "person_uuid": best_uuid if is_matched else None,
            "display_id": best_display_id if is_matched else None,
            "display_name": best_display_name if is_matched else "UNKNOWN",
            "similarity": highest_similarity,
            "candidate_scores": candidate_scores
        }
