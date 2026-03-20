from __future__ import annotations

from general_user_model_experiment.model import GeneralUserModel
from general_user_model_experiment.simulation import generate_synthetic_events


def test_general_user_model_end_to_end() -> None:
    events = generate_synthetic_events(n_users=7, events_per_user=55, random_state=17)
    core = events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]

    model = GeneralUserModel(n_clusters=3)
    model.fit(core)

    summary = model.summary()
    assert summary.users == 7
    assert summary.events == len(core)
    assert summary.proposition_count > 0

    user_id = str(core["user_id"].iloc[0])
    profile = model.get_user_profile(user_id)
    assert profile["user_id"] == user_id

    propositions = model.query_propositions("workflow", user_id=user_id, limit=8)
    assert isinstance(propositions, list)

    suggestions = model.suggest_for_user(user_id, top_k=5)
    assert len(suggestions) >= 1

    first = core.iloc[0]
    pred = model.predict_next_action(
        app=str(first["app"]),
        action=str(first["action"]),
        hour_of_day=int(first["timestamp"].hour),
        duration_sec=float(first["duration_sec"]),
    )
    assert 0.0 <= pred["confidence"] <= 1.0
    assert isinstance(pred["predicted_action"], str)
