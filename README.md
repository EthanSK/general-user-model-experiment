# general-user-model-experiment

Open-source experiment for building a **general user model from computer-use telemetry**, inspired by:

- **Paper:** *Creating General User Models from Computer Use* (Omar Shaikh et al.)
  - https://arxiv.org/abs/2505.10831
- **Project site:** https://generalusermodels.github.io/
- **Upstream repo:** https://github.com/generalusermodels/gum (MIT)

> This repository is **inspired by** the GUM architecture and paper. It is **not a code clone** of the upstream implementation.

---

## What this repo implements

### Source-derived architecture (from paper/repo concepts)

- Observation-driven user modeling from raw computer-use events.
- Confidence-weighted proposition memory.
- Proposition retrieval for context (query interface).
- Continuous proposition revision as new evidence arrives.
- Proactive-assistant style suggestion scoring based on value vs interruption cost.

### New extensions in this repo

- Event-first, reproducible experimental pipeline (CSV/synthetic telemetry).
- Statistical user profiling with embeddings, clustering, anomaly scoring.
- Hybrid model: proposition memory + profile features.
- FastAPI backend with train/query/context endpoints.
- Streamlit UI for training, user exploration, proposition search, and suggestion review.
- Local testing/evaluation scaffold for open-source iteration.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### 2) Train a demo model

```bash
python scripts/train_demo.py
```

This generates:

- `artifacts/general_user_model.joblib`
- `artifacts/user_profiles.csv`

### 3) Run API

```bash
uvicorn general_user_model_experiment.api:app --reload
```

Docs:

- Swagger UI: http://127.0.0.1:8000/docs

### 4) Run UI

```bash
streamlit run app/streamlit_app.py
```

---

## Data format (CSV)

Required columns:

- `user_id`
- `session_id`
- `timestamp` (ISO-8601 recommended)
- `app`
- `action`
- `target`
- `value`
- `duration_sec`

---

## API highlights

- `POST /train/sample` — train on synthetic events
- `POST /train/upload` — train on uploaded CSV
- `GET /profiles` — all user profiles
- `GET /profiles/{user_id}` — single user profile
- `GET /profiles/{user_id}/similar` — nearest users in embedding space
- `GET /propositions` — list proposition memory
- `GET /propositions/query?q=...` — retrieve proposition context
- `GET /suggestions/{user_id}` — proactive suggestions
- `GET /context/{user_id}?q=...` — combined profile + propositions + suggestions

---

## Repository layout

```text
app/
  streamlit_app.py          # UI demo

docs/
  ARCHITECTURE.md
  API.md

paper/
  SOURCE_DERIVATION.md      # explicit source-vs-extension mapping

scripts/
  train_demo.py

src/general_user_model_experiment/
  api.py
  dataio.py
  evaluation.py
  features.py
  model.py
  propositions.py
  schemas.py
  simulation.py
  suggestions.py

tests/
  test_api.py
  test_dataio.py
  test_model.py
  test_propositions.py
```

---

## License + attribution

This project is licensed under MIT (see `LICENSE`).

Acknowledgment:

- Shaikh, O., Sapkota, S., Rizvi, S., Horvitz, E., Park, J. S., Yang, D., & Bernstein, M. S. (2025).
  *Creating General User Models from Computer Use.* arXiv:2505.10831.

Upstream GUM repository is MIT-licensed. This project re-implements ideas in an independent codebase and cites the original paper/repo.

---

## Citation

```bibtex
@misc{shaikh2025creatinggeneralusermodels,
  title={Creating General User Models from Computer Use},
  author={Omar Shaikh and Shardul Sapkota and Shan Rizvi and Eric Horvitz and Joon Sung Park and Diyi Yang and Michael S. Bernstein},
  year={2025},
  eprint={2505.10831},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2505.10831}
}
```
