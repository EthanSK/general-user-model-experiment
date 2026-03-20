from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Observation:
    """Canonical observation produced from computer-use telemetry."""

    observation_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    app: str
    action: str
    target: str = ""
    value: str = ""
    duration_sec: float = 0.0
    source: str = "event"

    def to_text(self) -> str:
        parts = [self.app, self.action]
        if self.target:
            parts.append(self.target)
        if self.value:
            parts.append(self.value)
        return " ".join(parts)


@dataclass
class PropositionCandidate:
    key: str
    statement: str
    category: str
    confidence: float
    support_count: int
    group: Optional[str] = None
    evidence_observations: list[str] = field(default_factory=list)


@dataclass
class Proposition:
    """Confidence-weighted proposition about a user's behavior/preferences."""

    proposition_id: str
    user_id: str
    key: str
    statement: str
    category: str
    confidence: float
    support_count: int
    evidence_count: int
    first_seen: datetime
    last_updated: datetime
    status: str = "active"
    version: int = 1
    contradiction_count: int = 0
    group: Optional[str] = None
    supersedes: list[str] = field(default_factory=list)
    evidence_observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "proposition_id": self.proposition_id,
            "user_id": self.user_id,
            "key": self.key,
            "statement": self.statement,
            "category": self.category,
            "confidence": float(self.confidence),
            "support_count": int(self.support_count),
            "evidence_count": int(self.evidence_count),
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "status": self.status,
            "version": int(self.version),
            "contradiction_count": int(self.contradiction_count),
            "group": self.group,
            "supersedes": list(self.supersedes),
            "evidence_observations": list(self.evidence_observations),
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_key(key: str) -> str:
    return key.lower().strip().replace(" ", "_")


def _confidence(support_ratio: float, support_count: int) -> float:
    support_ratio = float(np.clip(support_ratio, 0.0, 1.0))
    volume = 1.0 - float(np.exp(-support_count / 28.0))
    conf = 0.58 * support_ratio + 0.42 * volume
    return float(np.clip(conf, 0.05, 0.99))


def _merge_ids(existing: list[str], incoming: list[str], limit: int = 40) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in [*incoming, *existing]:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


class PropositionMemory:
    """GUM-inspired proposition memory with retrieval + revision mechanics."""

    def __init__(self, confidence_floor: float = 0.22):
        self.confidence_floor = confidence_floor
        self._observations: dict[str, Observation] = {}
        self._propositions: dict[str, Proposition] = {}
        self._by_user_key: dict[tuple[str, str], str] = {}
        self._by_user_group: dict[tuple[str, str], str] = {}

    def ingest_events(self, events: pd.DataFrame) -> dict:
        if events.empty:
            raise ValueError("Cannot ingest empty events dataframe")

        required = {"user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"}
        missing = sorted(required - set(events.columns))
        if missing:
            raise ValueError(f"Missing required columns for proposition inference: {missing}")

        normalized = events.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
        normalized = normalized.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        observations: list[Observation] = []
        for row in normalized.to_dict(orient="records"):
            obs = Observation(
                observation_id=str(uuid4()),
                user_id=str(row["user_id"]),
                session_id=str(row["session_id"]),
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                app=str(row["app"]),
                action=str(row["action"]),
                target=str(row.get("target", "") or ""),
                value=str(row.get("value", "") or ""),
                duration_sec=float(row.get("duration_sec", 0.0) or 0.0),
            )
            observations.append(obs)
            self._observations[obs.observation_id] = obs

        by_user_obs: dict[str, list[str]] = {}
        for obs in observations:
            by_user_obs.setdefault(obs.user_id, []).append(obs.observation_id)

        updates = 0
        for user_id, user_df in normalized.groupby("user_id"):
            obs_ids = by_user_obs.get(str(user_id), [])
            candidates = self._infer_candidates(str(user_id), user_df.reset_index(drop=True), obs_ids)
            for candidate in candidates:
                if candidate.confidence < self.confidence_floor:
                    continue
                self._upsert_candidate(str(user_id), candidate)
                updates += 1

        return {
            "observations_ingested": len(observations),
            "propositions_updated": updates,
            "active_propositions": len([p for p in self._propositions.values() if p.status == "active"]),
        }

    def _infer_candidates(
        self,
        user_id: str,
        user_df: pd.DataFrame,
        evidence_observations: list[str],
    ) -> list[PropositionCandidate]:
        n_events = len(user_df)
        if n_events == 0:
            return []

        app_counts = user_df["app"].astype(str).value_counts()
        action_counts = user_df["action"].astype(str).value_counts()

        top_app = str(app_counts.idxmax())
        top_app_count = int(app_counts.iloc[0])
        top_app_share = top_app_count / n_events

        top_action = str(action_counts.idxmax())
        top_action_count = int(action_counts.iloc[0])
        top_action_share = top_action_count / n_events

        candidates: list[PropositionCandidate] = [
            PropositionCandidate(
                key=f"dominant_app:{_normalize_key(top_app)}",
                group="dominant_app",
                category="preference",
                statement=(
                    f"{user_id} primarily uses {top_app} "
                    f"({top_app_share:.0%} of observed events)."
                ),
                confidence=_confidence(top_app_share, top_app_count),
                support_count=top_app_count,
                evidence_observations=evidence_observations,
            ),
            PropositionCandidate(
                key=f"dominant_action:{_normalize_key(top_action)}",
                group="dominant_action",
                category="behavior",
                statement=(
                    f"{user_id} most often performs '{top_action}' actions "
                    f"({top_action_share:.0%} of observed events)."
                ),
                confidence=_confidence(top_action_share, top_action_count),
                support_count=top_action_count,
                evidence_observations=evidence_observations,
            ),
        ]

        # Working-hours proposition
        hours = user_df["timestamp"].dt.hour
        mean_hour = float(hours.mean())
        std_hour = float(hours.std(ddof=0) if len(hours) > 1 else 0.0)
        start_hour = int((mean_hour - max(1.0, std_hour)) % 24)
        end_hour = int((mean_hour + max(1.0, std_hour)) % 24)
        concentration = 1.0 / (1.0 + std_hour)
        candidates.append(
            PropositionCandidate(
                key=f"active_hours:{start_hour:02d}-{end_hour:02d}",
                group="active_hours",
                category="rhythm",
                statement=(
                    f"{user_id} is typically active around {start_hour:02d}:00–{end_hour:02d}:00 "
                    "based on observed interaction timestamps."
                ),
                confidence=_confidence(concentration, n_events),
                support_count=n_events,
                evidence_observations=evidence_observations,
            )
        )

        # Focus / context switching proposition
        switches = int((user_df["app"].shift(1) != user_df["app"]).sum() - 1)
        switches = max(switches, 0)
        switch_rate = switches / max(n_events - 1, 1)
        if switch_rate >= 0.42:
            focus_key = "focus:high_context_switching"
            focus_statement = (
                f"{user_id} frequently switches applications (switch rate {switch_rate:.2f}), "
                "suggesting fragmented focus periods."
            )
            focus_ratio = switch_rate
        elif switch_rate <= 0.22:
            focus_key = "focus:sustained_sessions"
            focus_statement = (
                f"{user_id} tends to stay in one app for longer stretches "
                f"(switch rate {switch_rate:.2f})."
            )
            focus_ratio = 1.0 - switch_rate
        else:
            focus_key = "focus:mixed"
            focus_statement = (
                f"{user_id} shows mixed focus behavior with moderate switching "
                f"(switch rate {switch_rate:.2f})."
            )
            focus_ratio = 0.45

        candidates.append(
            PropositionCandidate(
                key=focus_key,
                group="focus_pattern",
                category="focus",
                statement="".join(focus_statement),
                confidence=_confidence(focus_ratio, n_events),
                support_count=switches,
                evidence_observations=evidence_observations,
            )
        )

        # Transition proposition (workflow pattern)
        transitions = (
            user_df["app"].shift(1).astype(str)
            + " -> "
            + user_df["app"].astype(str)
        ).dropna()
        transitions = transitions[~transitions.str.startswith("nan")]
        if not transitions.empty:
            top_transition = transitions.value_counts().idxmax()
            transition_count = int(transitions.value_counts().max())
            transition_share = transition_count / len(transitions)
            if transition_count >= 3 and transition_share >= 0.10:
                tkey = _normalize_key(top_transition.replace(" -> ", "_to_"))
                candidates.append(
                    PropositionCandidate(
                        key=f"workflow_transition:{tkey}",
                        category="workflow",
                        statement=(
                            f"{user_id} repeatedly transitions from {top_transition} "
                            f"({transition_share:.0%} of app transitions)."
                        ),
                        confidence=_confidence(transition_share, transition_count),
                        support_count=transition_count,
                        evidence_observations=evidence_observations,
                    )
                )

        # Collaboration style proposition
        collab_actions = {"message", "review", "ack_alert", "comment", "share"}
        collab_share = float(user_df["action"].astype(str).isin(collab_actions).mean())
        if collab_share >= 0.24:
            collab_key = "collaboration:high"
            collab_statement = (
                f"{user_id} exhibits high collaboration activity "
                f"({collab_share:.0%} collaboration-related actions)."
            )
            collab_ratio = collab_share
        elif collab_share <= 0.10:
            collab_key = "collaboration:low"
            collab_statement = (
                f"{user_id} spends comparatively little time in collaboration actions "
                f"({collab_share:.0%})."
            )
            collab_ratio = 1.0 - collab_share
        else:
            collab_key = "collaboration:moderate"
            collab_statement = (
                f"{user_id} shows moderate collaboration activity "
                f"({collab_share:.0%})."
            )
            collab_ratio = 0.45

        candidates.append(
            PropositionCandidate(
                key=collab_key,
                group="collaboration_style",
                category="collaboration",
                statement=collab_statement,
                confidence=_confidence(collab_ratio, n_events),
                support_count=max(1, int(collab_share * n_events)),
                evidence_observations=evidence_observations,
            )
        )

        return candidates

    def _upsert_candidate(self, user_id: str, candidate: PropositionCandidate) -> Proposition:
        now = _utc_now()
        key = (user_id, candidate.key)
        existing_id = self._by_user_key.get(key)

        if existing_id is None:
            prop = Proposition(
                proposition_id=str(uuid4()),
                user_id=user_id,
                key=candidate.key,
                group=candidate.group,
                statement=candidate.statement,
                category=candidate.category,
                confidence=float(np.clip(candidate.confidence, 0.01, 0.99)),
                support_count=max(1, int(candidate.support_count)),
                evidence_count=len(candidate.evidence_observations),
                first_seen=now,
                last_updated=now,
                evidence_observations=list(candidate.evidence_observations[:40]),
            )
            self._propositions[prop.proposition_id] = prop
            self._by_user_key[key] = prop.proposition_id
        else:
            prop = self._propositions[existing_id]
            prop.statement = candidate.statement
            prop.category = candidate.category
            prop.version += 1
            prop.last_updated = now
            prop.status = "active"
            prop.group = candidate.group
            prop.support_count = int(round(0.72 * prop.support_count + 0.28 * max(1, candidate.support_count)))
            prop.evidence_count += len(candidate.evidence_observations)
            prop.confidence = float(
                np.clip(0.66 * prop.confidence + 0.34 * candidate.confidence, 0.01, 0.99)
            )
            prop.evidence_observations = _merge_ids(prop.evidence_observations, candidate.evidence_observations)

        # Exclusive-group revision logic
        if candidate.group:
            group_key = (user_id, candidate.group)
            previous_id = self._by_user_group.get(group_key)
            if previous_id and previous_id != prop.proposition_id:
                previous = self._propositions[previous_id]
                previous.status = "revised"
                previous.last_updated = now
                previous.contradiction_count += 1
                if previous.proposition_id not in prop.supersedes:
                    prop.supersedes.append(previous.proposition_id)
            self._by_user_group[group_key] = prop.proposition_id

        return prop

    def list_propositions(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = "active",
        min_confidence: float = 0.0,
        limit: int = 200,
    ) -> list[dict]:
        propositions = list(self._propositions.values())
        if user_id is not None:
            propositions = [p for p in propositions if p.user_id == user_id]
        if status is not None:
            propositions = [p for p in propositions if p.status == status]
        propositions = [p for p in propositions if p.confidence >= min_confidence]

        propositions.sort(key=lambda p: (p.confidence, p.last_updated), reverse=True)
        return [p.to_dict() for p in propositions[:limit]]

    def query(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        candidates = self.list_propositions(
            user_id=user_id,
            status="active",
            min_confidence=min_confidence,
            limit=2000,
        )
        if not candidates:
            return []

        if not query.strip():
            top = sorted(candidates, key=lambda p: p["confidence"], reverse=True)[:limit]
            for item in top:
                item["relevance_score"] = float(item["confidence"])
            return top

        docs = [
            f"{item['statement']} {item['category']} {item['key'].replace(':', ' ')}"
            for item in candidates
        ]

        lexical = np.zeros(len(docs), dtype=float)
        try:
            vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vect.fit_transform([*docs, query])
            lexical = cosine_similarity(matrix[:-1], matrix[-1]).reshape(-1)
        except ValueError:
            lexical = np.zeros(len(docs), dtype=float)

        now = _utc_now()
        scored: list[tuple[float, dict]] = []
        for idx, item in enumerate(candidates):
            last = datetime.fromisoformat(item["last_updated"])
            age_hours = max(0.0, (now - last).total_seconds() / 3600.0)
            recency = float(np.exp(-age_hours / (24.0 * 14.0)))
            score = 0.72 * float(lexical[idx]) + 0.20 * float(item["confidence"]) + 0.08 * recency
            out = dict(item)
            out["relevance_score"] = float(score)
            scored.append((score, out))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def proposition_count(self, status: Optional[str] = "active") -> int:
        if status is None:
            return len(self._propositions)
        return len([p for p in self._propositions.values() if p.status == status])

    def observation_count(self) -> int:
        return len(self._observations)
