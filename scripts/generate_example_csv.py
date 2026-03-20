#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from general_user_model_experiment.simulation import generate_synthetic_events


def main() -> None:
    out_dir = Path("examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = generate_synthetic_events(n_users=6, events_per_user=40, random_state=123)
    core = events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]

    out_path = out_dir / "sample_events.csv"
    core.to_csv(out_path, index=False)
    print(f"wrote {len(core)} rows to {out_path}")


if __name__ == "__main__":
    main()
