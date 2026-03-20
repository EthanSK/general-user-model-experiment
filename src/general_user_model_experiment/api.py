from __future__ import annotations

from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .dataio import records_to_frame, validate_event_frame
from .model import GeneralUserModel
from .schemas import (
    NextActionRequest,
    NextActionResponse,
    PropositionRecord,
    SimilarUser,
    SuggestionRecord,
    TrainResponse,
    UserProfileResponse,
)
from .simulation import generate_synthetic_events

app = FastAPI(title="General User Model Experiment API", version="0.2.0")

_model: GeneralUserModel | None = None
_data: pd.DataFrame | None = None


def _ensure_model() -> GeneralUserModel:
    global _model
    if _model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train/sample or /train/upload first.")
    return _model


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "trained": _model is not None,
        "events_loaded": int(len(_data)) if _data is not None else 0,
    }


@app.post("/train/sample", response_model=TrainResponse)
def train_sample(users: int = 12, events_per_user: int = 120, clusters: int = 4) -> TrainResponse:
    global _model, _data

    events = generate_synthetic_events(n_users=users, events_per_user=events_per_user)
    core = events[["user_id", "session_id", "timestamp", "app", "action", "target", "value", "duration_sec"]]
    _data = core.copy()

    _model = GeneralUserModel(n_clusters=clusters)
    _model.fit(_data)
    summary = _model.summary()
    return TrainResponse(**summary.__dict__)


@app.post("/train/upload", response_model=TrainResponse)
async def train_upload(file: UploadFile = File(...), clusters: int = 4) -> TrainResponse:
    global _model, _data

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    body = (await file.read()).decode("utf-8")
    frame = pd.read_csv(StringIO(body))
    try:
        validated = validate_event_frame(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid event file: {exc}") from exc

    _data = validated
    _model = GeneralUserModel(n_clusters=clusters)
    _model.fit(_data)
    summary = _model.summary()
    return TrainResponse(**summary.__dict__)


@app.post("/train/records", response_model=TrainResponse)
def train_records(records: list[dict], clusters: int = 4) -> TrainResponse:
    global _model, _data

    try:
        _data = records_to_frame(records)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid records: {exc}") from exc

    _model = GeneralUserModel(n_clusters=clusters)
    _model.fit(_data)
    summary = _model.summary()
    return TrainResponse(**summary.__dict__)


@app.get("/profiles")
def list_profiles() -> list[dict]:
    model = _ensure_model()
    profiles = model.get_user_profiles().sort_values("user_id")
    return profiles.to_dict(orient="records")


@app.get("/profiles/{user_id}", response_model=UserProfileResponse)
def profile(user_id: str) -> UserProfileResponse:
    model = _ensure_model()
    try:
        data = model.get_user_profile(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return UserProfileResponse(**data)


@app.get("/profiles/{user_id}/similar", response_model=list[SimilarUser])
def similar(user_id: str, top_k: int = 5) -> list[SimilarUser]:
    model = _ensure_model()
    try:
        data = model.similar_users(user_id, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [SimilarUser(**item) for item in data]


@app.post("/predict/next-action", response_model=NextActionResponse)
def predict_next_action(request: NextActionRequest) -> NextActionResponse:
    model = _ensure_model()
    out = model.predict_next_action(
        app=request.app,
        action=request.action,
        hour_of_day=request.hour_of_day,
        duration_sec=request.duration_sec,
    )
    return NextActionResponse(**out)


@app.get("/propositions", response_model=list[PropositionRecord])
def propositions(
    user_id: str | None = None,
    status: str | None = Query(default="active"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=200, ge=1, le=2000),
) -> list[PropositionRecord]:
    model = _ensure_model()
    rows = model.list_propositions(
        user_id=user_id,
        status=status,
        min_confidence=min_confidence,
        limit=limit,
    )
    return [PropositionRecord(**row) for row in rows]


@app.get("/propositions/query", response_model=list[PropositionRecord])
def query_propositions(
    q: str = Query(..., description="Free-text query over proposition memory"),
    user_id: str | None = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=10, ge=1, le=100),
) -> list[PropositionRecord]:
    model = _ensure_model()
    rows = model.query_propositions(
        query=q,
        user_id=user_id,
        limit=limit,
        min_confidence=min_confidence,
    )
    return [PropositionRecord(**row) for row in rows]


@app.get("/suggestions/{user_id}", response_model=list[SuggestionRecord])
def suggestions(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
    min_confidence: float = Query(default=0.2, ge=0.0, le=1.0),
) -> list[SuggestionRecord]:
    model = _ensure_model()
    try:
        rows = model.suggest_for_user(user_id=user_id, top_k=top_k, min_confidence=min_confidence)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [SuggestionRecord(**row) for row in rows]


@app.get("/context/{user_id}")
def context_bundle(
    user_id: str,
    q: str = Query(default=""),
    proposition_limit: int = Query(default=8, ge=1, le=50),
    suggestion_limit: int = Query(default=5, ge=1, le=20),
) -> dict:
    model = _ensure_model()

    try:
        profile_data = model.get_user_profile(user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if q.strip():
        proposition_data = model.query_propositions(
            query=q,
            user_id=user_id,
            limit=proposition_limit,
            min_confidence=0.1,
        )
    else:
        proposition_data = model.list_propositions(
            user_id=user_id,
            min_confidence=0.1,
            limit=proposition_limit,
        )

    suggestion_data = model.suggest_for_user(user_id=user_id, top_k=suggestion_limit)

    return {
        "profile": profile_data,
        "propositions": proposition_data,
        "suggestions": suggestion_data,
    }
