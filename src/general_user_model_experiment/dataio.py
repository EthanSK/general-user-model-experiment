from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import EventRecord

REQUIRED_COLUMNS = [
    "user_id",
    "session_id",
    "timestamp",
    "app",
    "action",
    "target",
    "value",
    "duration_sec",
]


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    casted = df.copy()
    casted["timestamp"] = pd.to_datetime(casted["timestamp"], utc=True)
    casted["duration_sec"] = pd.to_numeric(casted["duration_sec"], errors="coerce").fillna(0.0)

    for col in ["user_id", "session_id", "app", "action", "target", "value"]:
        casted[col] = casted[col].astype(str).fillna("").str.strip()

    return casted


def validate_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    casted = _coerce_types(df[REQUIRED_COLUMNS])

    # Lightweight row-level validation via pydantic
    records = [EventRecord(**row).model_dump() for row in casted.to_dict(orient="records")]
    validated = pd.DataFrame.from_records(records)
    validated["timestamp"] = pd.to_datetime(validated["timestamp"], utc=True)
    return validated.sort_values("timestamp").reset_index(drop=True)


def load_events_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return validate_event_frame(df)


def records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(list(records))
    return validate_event_frame(df)
