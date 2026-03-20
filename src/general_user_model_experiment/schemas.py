from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class EventRecord(BaseModel):
    """Canonical event record for computer-use telemetry."""

    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    timestamp: datetime
    app: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    target: str = Field(default="")
    value: str = Field(default="")
    duration_sec: float = Field(default=0.0, ge=0.0)

    @field_validator("app", "action", "target", "value", mode="before")
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        return (value or "").strip()


class TrainResponse(BaseModel):
    users: int
    events: int
    clusters: int
    proposition_count: int
    top_features: list[str]


class UserProfileResponse(BaseModel):
    user_id: str
    cluster: int
    event_count: float
    session_count: float
    app_entropy: float
    action_entropy: float
    switch_rate: float
    dominant_app: str
    dominant_action: str
    active_hour_mean: float
    active_hour_std: float
    mean_session_duration: float
    anomaly_score: float


class SimilarUser(BaseModel):
    user_id: str
    similarity: float


class NextActionRequest(BaseModel):
    app: str
    action: str
    hour_of_day: int = Field(..., ge=0, le=23)
    duration_sec: float = Field(default=0.0, ge=0.0)


class NextActionResponse(BaseModel):
    predicted_action: str
    confidence: float


class PropositionRecord(BaseModel):
    proposition_id: str
    user_id: str
    key: str
    statement: str
    category: str
    confidence: float
    support_count: int
    evidence_count: int
    first_seen: str
    last_updated: str
    status: str
    version: int
    contradiction_count: int
    group: Optional[str] = None
    supersedes: list[str] = Field(default_factory=list)
    evidence_observations: list[str] = Field(default_factory=list)
    relevance_score: Optional[float] = None


class SuggestionRecord(BaseModel):
    suggestion_id: str
    user_id: str
    title: str
    rationale: str
    suggestion_type: str
    expected_benefit: float
    interruption_cost: float
    confidence: float
    urgency: float
    priority_score: float
    source_propositions: list[str]


class TrainMode(BaseModel):
    mode: Literal["sample", "upload"]
    sample_users: Optional[int] = Field(default=10, ge=2, le=200)
    sample_events_per_user: Optional[int] = Field(default=120, ge=20, le=2000)
