"""
Multi-Modal Score Fusion Engine (Phase 5.5)

Combines SFace 128-d Face Cosine Similarity and OSNet 512-d Body Re-ID Cosine Similarity.
Provides automatic fallback to Person Re-ID (body_score) when face is turned, occluded, or covered by hand!
"""

import numpy as np
from app.config.settings import config
from src.recognition.face_matcher import compute_cosine_similarity


class MultiModalFusionEngine:
    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else config.recognition_threshold
        self.body_fallback_threshold = 0.45  # Lower threshold for Body Re-ID fallback during face occlusion

    def match_multi_modal(self, query_face_emb, query_body_emb, enrolled_dict):
        if not enrolled_dict:
            return {
                "matched": False,
                "person_uuid": None,
                "display_id": None,
                "display_name": "UNKNOWN",
                "face_score": 0.0,
                "body_score": 0.0,
                "final_score": 0.0,
                "modality_used": "NONE"
            }

        best_uuid = None
        best_display_id = None
        best_display_name = None
        highest_final_score = 0.0
        best_face_score = 0.0
        best_body_score = 0.0
        best_modality = "NONE"

        for uuid, person_data in enrolled_dict.items():
            if person_data.get("status") != "ACTIVE":
                continue

            face_vecs = person_data.get("face_embeddings", [])
            body_vecs = person_data.get("body_embeddings", [])

            # 1. Compute Face Score (Top-3 Avg)
            face_score = 0.0
            if query_face_emb is not None and len(face_vecs) > 0:
                f_sims = [compute_cosine_similarity(query_face_emb, ref) for ref in face_vecs]
                f_sims.sort(reverse=True)
                face_score = float(np.mean(f_sims[:min(3, len(f_sims))]))

            # 2. Compute Body Re-ID Score (Top-3 Avg)
            body_score = 0.0
            if query_body_emb is not None and len(body_vecs) > 0:
                b_sims = [compute_cosine_similarity(query_body_emb, ref) for ref in body_vecs]
                b_sims.sort(reverse=True)
                body_score = float(np.mean(b_sims[:min(3, len(b_sims))]))

            # 3. Dynamic Weighted Score Fusion
            if query_face_emb is not None and query_body_emb is not None and len(face_vecs) > 0 and len(body_vecs) > 0:
                final_score = 0.65 * face_score + 0.35 * body_score
                modality = "FACE+BODY"
                req_threshold = self.threshold
            elif (query_face_emb is None or len(face_vecs) == 0) and query_body_emb is not None and len(body_vecs) > 0:
                final_score = body_score  # Face covered/turned -> Person Re-ID Fallback!
                modality = "BODY_REID_ONLY"
                req_threshold = self.body_fallback_threshold  # 0.45 for body fallback
            elif query_face_emb is not None and len(face_vecs) > 0:
                final_score = face_score  # Body unavailable -> Face Only
                modality = "FACE_ONLY"
                req_threshold = self.threshold
            else:
                final_score = 0.0
                modality = "NONE"
                req_threshold = self.threshold

            if final_score > highest_final_score:
                highest_final_score = final_score
                best_face_score = face_score
                best_body_score = body_score
                best_modality = modality
                best_uuid = uuid
                best_display_id = person_data.get("display_id")
                best_display_name = person_data.get("display_name", "UNKNOWN")
                best_req_threshold = req_threshold

        is_matched = False
        if best_uuid is not None:
            is_matched = highest_final_score >= best_req_threshold

        return {
            "matched": is_matched,
            "person_uuid": best_uuid if is_matched else None,
            "display_id": best_display_id if is_matched else None,
            "display_name": best_display_name if is_matched else "UNKNOWN",
            "face_score": best_face_score,
            "body_score": best_body_score,
            "final_score": highest_final_score,
            "modality_used": best_modality
        }
