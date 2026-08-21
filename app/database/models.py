"""
Data Models for Household AI Application
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PersonModel:
    id: int
    person_uuid: str
    display_name: str
    display_id: str
    status: str
    created_at: str
    updated_at: str


@dataclass
class UnknownDetectionModel:
    id: int
    snapshot_path: str
    track_id: int
    best_similarity: float
    camera_id: str
    status: str
    created_at: str


@dataclass
class RecognitionEventModel:
    id: int
    track_id: int
    person_uuid: Optional[str]
    recognition_result: str
    similarity_score: float
    timestamp: str
