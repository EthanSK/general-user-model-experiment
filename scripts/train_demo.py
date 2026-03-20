#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from general_user_model_experiment.model import GeneralUserModel
from general_user_model_experiment.simulation import generate_synthetic_events


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = generate_synthetic_events(n_users=14, events_per_user=130)
    core = events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]

    model = GeneralUserModel(n_clusters=4)
    model.fit(core)

    model_path = out_dir / "general_user_model.joblib"
    model.save(model_path)

    profiles = model.get_user_profiles()
    profiles.to_csv(out_dir / "user_profiles.csv", index=False)

    summary = model.summary()
    user_id = str(profiles.iloc[0]["user_id"])
    top_props = model.query_propositions(query="focus", user_id=user_id, limit=3)
    suggestions = model.suggest_for_user(user_id=user_id, top_k=3)

    print("Training complete")
    print(
        f"users={summary.users}, events={summary.events}, clusters={summary.clusters}, "
        f"active_propositions={summary.proposition_count}"
    )
    print(f"top_features={summary.top_features}")
    print(f"saved_model={model_path}")
    print(f"example_user={user_id}")
    print(f"example_propositions={len(top_props)}")
    print(f"example_suggestions={len(suggestions)}")


if __name__ == "__main__":
    main()
