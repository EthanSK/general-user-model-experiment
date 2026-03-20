# API Reference (v0.2)

Base: `http://127.0.0.1:8000`

## Health

### `GET /health`

Returns training status and loaded event count.

## Training

### `POST /train/sample`

Train on synthetic data.

Query params:

- `users` (default `12`)
- `events_per_user` (default `120`)
- `clusters` (default `4`)

### `POST /train/upload`

Upload CSV telemetry file and train.

Form-data:

- `file`: CSV
- `clusters`: int (optional)

### `POST /train/records`

Train from JSON records.

Body: `list[EventRecord]`

## User profiles

### `GET /profiles`

All profile rows.

### `GET /profiles/{user_id}`

Single profile summary.

### `GET /profiles/{user_id}/similar`

Nearest users in embedding space.

Query params:

- `top_k` (default `5`)

## Predictions

### `POST /predict/next-action`

Body:

```json
{
  "app": "VSCode",
  "action": "type",
  "hour_of_day": 14,
  "duration_sec": 55
}
```

## Proposition memory

### `GET /propositions`

List propositions.

Query params:

- `user_id` (optional)
- `status` (default `active`; optional `null` for all)
- `min_confidence` (default `0.0`)
- `limit` (default `200`)

### `GET /propositions/query`

Text retrieval over proposition memory.

Query params:

- `q` (required)
- `user_id` (optional)
- `min_confidence` (default `0.0`)
- `limit` (default `10`)

## Suggestions

### `GET /suggestions/{user_id}`

Returns ranked proactive suggestions.

Query params:

- `top_k` (default `5`)
- `min_confidence` (default `0.2`)

## Context bundle

### `GET /context/{user_id}`

Returns:

- profile
- propositions (query-filtered if `q` provided)
- suggestions

Query params:

- `q` (optional)
- `proposition_limit` (default `8`)
- `suggestion_limit` (default `5`)
