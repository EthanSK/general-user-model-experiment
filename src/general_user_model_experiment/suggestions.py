from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import numpy as np


@dataclass
class Suggestion:
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

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "user_id": self.user_id,
            "title": self.title,
            "rationale": self.rationale,
            "suggestion_type": self.suggestion_type,
            "expected_benefit": float(self.expected_benefit),
            "interruption_cost": float(self.interruption_cost),
            "confidence": float(self.confidence),
            "urgency": float(self.urgency),
            "priority_score": float(self.priority_score),
            "source_propositions": list(self.source_propositions),
        }


class SuggestionEngine:
    """GUMBO-inspired suggestion scoring with benefit/cost interruption tradeoff."""

    @staticmethod
    def _score(benefit: float, interruption_cost: float, confidence: float, urgency: float) -> float:
        benefit = float(np.clip(benefit, 0.0, 1.0))
        interruption_cost = float(np.clip(interruption_cost, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        urgency = float(np.clip(urgency, 0.0, 1.0))

        score = (0.52 * benefit * confidence) + (0.28 * urgency) + (0.20 * (1.0 - interruption_cost))
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _prop_confidence(propositions: list[dict], prefix: str, fallback: float = 0.6) -> tuple[float, list[str]]:
        matched = [p for p in propositions if str(p.get("key", "")).startswith(prefix)]
        if not matched:
            return fallback, []
        conf = max(float(p.get("confidence", fallback)) for p in matched)
        ids = [str(p.get("proposition_id")) for p in matched if p.get("proposition_id")]
        return conf, ids

    @staticmethod
    def _extract_active_hours(propositions: list[dict]) -> Optional[tuple[int, int]]:
        for p in propositions:
            key = str(p.get("key", ""))
            if not key.startswith("active_hours:"):
                continue
            window = key.split(":", maxsplit=1)[1]
            if "-" not in window:
                continue
            start_s, end_s = window.split("-", maxsplit=1)
            try:
                return int(start_s), int(end_s)
            except ValueError:
                return None
        return None

    def generate(
        self,
        user_id: str,
        profile: dict,
        propositions: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        out: list[Suggestion] = []

        switch_rate = float(profile.get("switch_rate", 0.0) or 0.0)
        anomaly_score = float(profile.get("anomaly_score", 0.0) or 0.0)
        dominant_app = str(profile.get("dominant_app", "")).lower()
        dominant_action = str(profile.get("dominant_action", "")).lower()

        focus_conf, focus_sources = self._prop_confidence(propositions, "focus:", fallback=0.55)
        collab_conf, collab_sources = self._prop_confidence(propositions, "collaboration:", fallback=0.55)
        workflow_conf, workflow_sources = self._prop_confidence(propositions, "workflow_transition:", fallback=0.5)

        if switch_rate >= 0.36:
            benefit = float(np.clip(0.42 + switch_rate, 0.0, 0.95))
            interruption_cost = float(np.clip(0.16 + (switch_rate * 0.22), 0.0, 0.50))
            urgency = float(np.clip(0.34 + (anomaly_score * 0.35), 0.0, 0.95))
            score = self._score(benefit, interruption_cost, focus_conf, urgency)
            out.append(
                Suggestion(
                    suggestion_id=str(uuid4()),
                    user_id=user_id,
                    title="Offer a 25-minute focus block",
                    rationale=(
                        "Frequent app switching suggests fragmented attention. "
                        "A short focus block with muted non-critical notifications can reduce context switching."
                    ),
                    suggestion_type="focus",
                    expected_benefit=benefit,
                    interruption_cost=interruption_cost,
                    confidence=focus_conf,
                    urgency=urgency,
                    priority_score=score,
                    source_propositions=focus_sources,
                )
            )

        if dominant_app in {"terminal", "vscode"} or dominant_action in {"run_command", "commit", "type"}:
            benefit = 0.72
            interruption_cost = 0.26
            urgency = 0.44
            score = self._score(benefit, interruption_cost, workflow_conf, urgency)
            out.append(
                Suggestion(
                    suggestion_id=str(uuid4()),
                    user_id=user_id,
                    title="Suggest reusable automation for repeated workflow",
                    rationale=(
                        "Observed tool usage indicates repetitive build/run/edit loops. "
                        "Auto-generating a script/snippet could save repeated effort."
                    ),
                    suggestion_type="automation",
                    expected_benefit=benefit,
                    interruption_cost=interruption_cost,
                    confidence=workflow_conf,
                    urgency=urgency,
                    priority_score=score,
                    source_propositions=workflow_sources,
                )
            )

        has_high_collab = any(str(p.get("key", "")).startswith("collaboration:high") for p in propositions)
        if has_high_collab or dominant_app == "slack":
            benefit = 0.68
            interruption_cost = 0.18
            urgency = 0.57
            score = self._score(benefit, interruption_cost, collab_conf, urgency)
            out.append(
                Suggestion(
                    suggestion_id=str(uuid4()),
                    user_id=user_id,
                    title="Bundle communication notifications",
                    rationale=(
                        "Collaboration-heavy activity can cause frequent interruptions. "
                        "Batching notifications into digest windows helps preserve deep work."
                    ),
                    suggestion_type="notification",
                    expected_benefit=benefit,
                    interruption_cost=interruption_cost,
                    confidence=collab_conf,
                    urgency=urgency,
                    priority_score=score,
                    source_propositions=collab_sources,
                )
            )

        hours_window = self._extract_active_hours(propositions)
        if hours_window is not None:
            start_hour, end_hour = hours_window
            late_window = (start_hour >= 21 or end_hour <= 5)
            if late_window:
                benefit = 0.46
                interruption_cost = 0.22
                urgency = 0.41
                confidence, sources = self._prop_confidence(propositions, "active_hours:", fallback=0.5)
                score = self._score(benefit, interruption_cost, confidence, urgency)
                out.append(
                    Suggestion(
                        suggestion_id=str(uuid4()),
                        user_id=user_id,
                        title="Offer end-of-day summary instead of real-time nudges",
                        rationale=(
                            "Activity appears concentrated late in the day. "
                            "Deferring non-urgent prompts to a summary can lower interruption burden."
                        ),
                        suggestion_type="timing",
                        expected_benefit=benefit,
                        interruption_cost=interruption_cost,
                        confidence=confidence,
                        urgency=urgency,
                        priority_score=score,
                        source_propositions=sources,
                    )
                )

        if anomaly_score >= 0.55:
            benefit = float(np.clip(0.48 + anomaly_score * 0.4, 0.0, 0.95))
            interruption_cost = 0.30
            urgency = float(np.clip(0.52 + anomaly_score * 0.35, 0.0, 0.98))
            score = self._score(benefit, interruption_cost, 0.67, urgency)
            out.append(
                Suggestion(
                    suggestion_id=str(uuid4()),
                    user_id=user_id,
                    title="Check for workflow change and refresh model context",
                    rationale=(
                        "Behavior is currently atypical for this user profile. "
                        "A lightweight check-in can prevent stale assumptions and improve personalization."
                    ),
                    suggestion_type="adaptation",
                    expected_benefit=benefit,
                    interruption_cost=interruption_cost,
                    confidence=0.67,
                    urgency=urgency,
                    priority_score=score,
                    source_propositions=[],
                )
            )

        # Always provide at least one low-interruption recommendation.
        if not out:
            score = self._score(0.45, 0.08, 0.52, 0.24)
            out.append(
                Suggestion(
                    suggestion_id=str(uuid4()),
                    user_id=user_id,
                    title="Capture a daily preference snapshot",
                    rationale=(
                        "There is limited high-confidence evidence so far. "
                        "A low-friction daily snapshot helps the model calibrate user preferences over time."
                    ),
                    suggestion_type="calibration",
                    expected_benefit=0.45,
                    interruption_cost=0.08,
                    confidence=0.52,
                    urgency=0.24,
                    priority_score=score,
                    source_propositions=[],
                )
            )

        out.sort(key=lambda item: item.priority_score, reverse=True)
        return [item.to_dict() for item in out[:top_k]]
