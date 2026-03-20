from __future__ import annotations

from general_user_model_experiment.propositions import PropositionMemory
from general_user_model_experiment.simulation import generate_synthetic_events


def test_proposition_memory_ingest_and_query() -> None:
    events = generate_synthetic_events(n_users=5, events_per_user=40, random_state=11)
    core = events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]

    memory = PropositionMemory()
    stats = memory.ingest_events(core)

    assert stats["observations_ingested"] == len(core)
    assert stats["active_propositions"] > 0

    user_id = str(core["user_id"].iloc[0])
    props = memory.list_propositions(user_id=user_id)
    assert len(props) > 0

    retrieved = memory.query("focus", user_id=user_id, limit=5)
    assert isinstance(retrieved, list)
    assert len(retrieved) <= 5
