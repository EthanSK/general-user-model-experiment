from __future__ import annotations

from fastapi.testclient import TestClient

from general_user_model_experiment.api import app


client = TestClient(app)


def test_api_train_query_and_suggestions() -> None:
    train = client.post("/train/sample", params={"users": 6, "events_per_user": 45, "clusters": 3})
    assert train.status_code == 200
    train_json = train.json()
    assert train_json["proposition_count"] > 0

    profiles = client.get("/profiles")
    assert profiles.status_code == 200
    profile_rows = profiles.json()
    assert len(profile_rows) == 6

    user_id = profile_rows[0]["user_id"]

    props = client.get("/propositions", params={"user_id": user_id, "min_confidence": 0.1})
    assert props.status_code == 200
    assert len(props.json()) > 0

    retrieved = client.get("/propositions/query", params={"q": "focus", "user_id": user_id})
    assert retrieved.status_code == 200

    suggestions = client.get(f"/suggestions/{user_id}")
    assert suggestions.status_code == 200
    assert len(suggestions.json()) >= 1

    context = client.get(f"/context/{user_id}", params={"q": "workflow"})
    assert context.status_code == 200
    payload = context.json()
    assert "profile" in payload and "propositions" in payload and "suggestions" in payload
