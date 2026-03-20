from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


@dataclass
class Archetype:
    name: str
    apps: list[str]
    app_probs: list[float]
    actions: list[str]
    action_probs: list[float]


ARCHETYPES = [
    Archetype(
        name="builder",
        apps=["VSCode", "Terminal", "Chrome", "Slack"],
        app_probs=[0.38, 0.32, 0.18, 0.12],
        actions=["type", "run_command", "search", "message", "commit"],
        action_probs=[0.42, 0.24, 0.16, 0.10, 0.08],
    ),
    Archetype(
        name="analyst",
        apps=["Chrome", "Notion", "Sheets", "Slack"],
        app_probs=[0.30, 0.24, 0.30, 0.16],
        actions=["search", "read", "edit", "message", "export"],
        action_probs=[0.28, 0.22, 0.26, 0.14, 0.10],
    ),
    Archetype(
        name="operator",
        apps=["Terminal", "Grafana", "Chrome", "Slack"],
        app_probs=[0.35, 0.27, 0.18, 0.20],
        actions=["run_command", "monitor", "search", "message", "ack_alert"],
        action_probs=[0.30, 0.22, 0.18, 0.20, 0.10],
    ),
    Archetype(
        name="creator",
        apps=["Figma", "Chrome", "Notion", "Slack"],
        app_probs=[0.36, 0.26, 0.22, 0.16],
        actions=["design", "search", "edit", "message", "review"],
        action_probs=[0.30, 0.20, 0.25, 0.15, 0.10],
    ),
]


def _pick_archetypes(rng: np.random.Generator, n_users: int) -> list[Archetype]:
    probs = np.array([0.30, 0.25, 0.20, 0.25])
    idxs = rng.choice(len(ARCHETYPES), size=n_users, p=probs)
    return [ARCHETYPES[i] for i in idxs]


def generate_synthetic_events(
    n_users: int = 12,
    events_per_user: int = 120,
    start: datetime | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic computer-use telemetry for demo + testing."""
    if n_users < 2:
        raise ValueError("n_users must be >= 2")
    if events_per_user < 20:
        raise ValueError("events_per_user must be >= 20")

    rng = np.random.default_rng(random_state)
    start = start or datetime.now(timezone.utc) - timedelta(days=5)

    users = [f"user_{i:03d}" for i in range(1, n_users + 1)]
    user_archetypes = _pick_archetypes(rng, n_users)

    rows: list[dict] = []

    for uid, archetype in zip(users, user_archetypes):
        cursor = start + timedelta(hours=float(rng.integers(0, 24)))
        events_left = events_per_user
        session_no = 1

        while events_left > 0:
            session_len = int(min(events_left, rng.integers(16, 38)))
            session_id = f"{uid}-s{session_no:03d}"
            session_no += 1

            for _ in range(session_len):
                app = str(rng.choice(archetype.apps, p=archetype.app_probs))
                action = str(rng.choice(archetype.actions, p=archetype.action_probs))
                duration = float(rng.gamma(shape=2.4, scale=18.0))

                target = ""
                value = ""
                if action in {"search", "read"}:
                    target = str(rng.choice(["docs", "github", "stack-overflow", "calendar", "email"]))
                elif action in {"run_command", "monitor", "ack_alert"}:
                    target = str(rng.choice(["deploy", "logs", "alerts", "tests"]))
                elif action in {"edit", "design", "type", "commit"}:
                    target = str(rng.choice(["project-alpha", "spec", "ui", "notebook"]))

                rows.append(
                    {
                        "user_id": uid,
                        "session_id": session_id,
                        "timestamp": cursor.isoformat(),
                        "app": app,
                        "action": action,
                        "target": target,
                        "value": value,
                        "duration_sec": round(duration, 3),
                        "archetype": archetype.name,
                    }
                )

                # 5-120 seconds between atomic events
                cursor += timedelta(seconds=float(rng.integers(5, 120)))

            events_left -= session_len
            # Break between sessions (20 minutes to 8 hours)
            cursor += timedelta(minutes=float(rng.integers(20, 480)))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
