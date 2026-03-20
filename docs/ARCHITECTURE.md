# Architecture

## Design goal

Build an open, inspectable implementation of a **general user model** from computer-use telemetry, following the core ideas of the GUM paper while remaining runnable on commodity local setups.

## System overview

```text
Telemetry events (CSV or synthetic)
    ↓
Validation + normalization
    ↓
Feature extraction --------------------→ behavioral profile model
    ↓                                    (embeddings, clustering,
Observation stream                       anomaly, next-action)
    ↓
Proposition inference (confidence-weighted)
    ↓
Revision engine (supersede stale/conflicting propositions)
    ↓
Retrieval layer (lexical + confidence + recency)
    ↓
Suggestion engine (benefit/cost/urgency scoring)
```

## Core modules

- `dataio.py`
  - Validates and normalizes event data.
- `features.py`
  - Aggregates per-user behavioral features.
- `propositions.py`
  - Maintains proposition memory with revision and retrieval.
- `model.py`
  - Orchestrates profile model + proposition memory + suggestions.
- `suggestions.py`
  - Scores proactive actions inspired by GUMBO-style tradeoffs.
- `api.py`
  - FastAPI service around training/inference/query.
- `app/streamlit_app.py`
  - Interactive demo UI.

## Proposition lifecycle

1. **Observation ingestion** from telemetry rows.
2. **Inference** to proposition candidates (dominant app/action, focus pattern, rhythm, transitions, collaboration style).
3. **Confidence calibration** from support ratio + evidence volume.
4. **Revision**: mutually-exclusive proposition groups (`dominant_app`, `focus_pattern`, etc.) supersede prior active proposition when evidence shifts.
5. **Retrieval**: query-time ranking combines lexical similarity, confidence, and recency.

## Behavioral model

The statistical profile model computes:

- aggregate user features
- low-dimensional embeddings (PCA)
- user clusters (KMeans)
- anomaly scores (Isolation Forest)
- next-action prediction from transitions (logistic regression fallback to majority class)

## Proactive suggestion scoring

Each candidate suggestion includes:

- expected benefit
- interruption cost
- confidence
- urgency

Priority score favors high-value, high-confidence, low-interruption suggestions.

## Privacy + deployment posture

This repo is designed for local-first experimentation:

- no mandatory external model APIs
- inspectable proposition memory
- deterministic synthetic data for reproducibility

Use caution with real personal telemetry and apply appropriate consent/security controls.
