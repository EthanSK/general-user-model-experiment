from __future__ import annotations

from general_user_model_experiment.dataio import validate_event_frame
from general_user_model_experiment.simulation import generate_synthetic_events


def test_validate_event_frame_accepts_synthetic_core_columns() -> None:
    events = generate_synthetic_events(n_users=3, events_per_user=25, random_state=7)
    core = events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]

    validated = validate_event_frame(core)

    assert len(validated) == len(core)
    assert list(validated.columns) == [
        "user_id",
        "session_id",
        "timestamp",
        "app",
        "action",
        "target",
        "value",
        "duration_sec",
    ]
    assert validated["timestamp"].is_monotonic_increasing
