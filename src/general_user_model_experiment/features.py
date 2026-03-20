from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FeatureArtifacts:
    user_features: pd.DataFrame
    top_apps: list[str]
    top_actions: list[str]


def _entropy(values: Iterable[int | float]) -> float:
    arr = np.array(list(values), dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _top_categories(series: pd.Series, k: int = 8) -> list[str]:
    counts = series.value_counts()
    return list(counts.head(k).index.astype(str))


def build_user_features(events: pd.DataFrame, top_k: int = 8) -> FeatureArtifacts:
    """Build aggregate features that describe user behavior from event streams."""
    if events.empty:
        raise ValueError("Cannot build features from empty events dataframe")

    events = events.sort_values("timestamp").copy()
    events["hour"] = events["timestamp"].dt.hour

    top_apps = _top_categories(events["app"], top_k)
    top_actions = _top_categories(events["action"], top_k)

    rows: list[dict] = []
    for user_id, user_df in events.groupby("user_id"):
        user_df = user_df.sort_values("timestamp").copy()
        app_counts = user_df["app"].value_counts()
        action_counts = user_df["action"].value_counts()

        app_switches = (user_df["app"].shift(1) != user_df["app"]).sum() - 1
        app_switches = max(app_switches, 0)

        session_lengths = user_df.groupby("session_id")["duration_sec"].sum()

        row = {
            "user_id": user_id,
            "event_count": float(len(user_df)),
            "session_count": float(user_df["session_id"].nunique()),
            "avg_duration": float(user_df["duration_sec"].mean()),
            "median_duration": float(user_df["duration_sec"].median()),
            "app_entropy": _entropy(app_counts.values),
            "action_entropy": _entropy(action_counts.values),
            "switch_rate": float(app_switches / max(len(user_df) - 1, 1)),
            "mean_session_duration": float(session_lengths.mean()),
            "max_session_duration": float(session_lengths.max()),
            "active_hour_mean": float(user_df["hour"].mean()),
            "active_hour_std": float(user_df["hour"].std(ddof=0) if len(user_df) > 1 else 0.0),
            "dominant_app": str(app_counts.idxmax()),
            "dominant_action": str(action_counts.idxmax()),
        }

        for app in top_apps:
            row[f"app_share::{app}"] = float((user_df["app"] == app).mean())

        for action in top_actions:
            row[f"action_share::{action}"] = float((user_df["action"] == action).mean())

        rows.append(row)

    feat_df = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
    return FeatureArtifacts(user_features=feat_df, top_apps=top_apps, top_actions=top_actions)


def build_next_action_training_set(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised data from adjacent event transitions."""
    if len(events) < 3:
        raise ValueError("Need at least 3 events to build transition dataset")

    events = events.sort_values(["user_id", "session_id", "timestamp"]).copy()
    events["next_action"] = events.groupby(["user_id", "session_id"])["action"].shift(-1)
    events = events.dropna(subset=["next_action"]).reset_index(drop=True)

    if events.empty:
        raise ValueError("No transitions available to train next-action predictor")

    X = pd.DataFrame(
        {
            "app": events["app"].astype(str),
            "action": events["action"].astype(str),
            "hour": events["timestamp"].dt.hour.astype(int),
            "duration_sec": events["duration_sec"].astype(float),
        }
    )
    y = events["next_action"].astype(str)
    return X, y
