# general-user-model-experiment — PLAN

## User request (verbatim)
> In a long running sub agent, make a general user model from computer use, like that gumbo paper that's already there, make an open source version of it, spend forever doing it, and publish it to my GitHub called the general u user model experiment. That should be the name. Kebab case. Make a UI for it and everything. Yeah. That's what's

## Clarification (verbatim)
> Make sure you use the creating general user models from computers paper by Omar Shaikh, etcetera. Do some research first. They have a GitHub repo. They have a GitHub repo apparently. Just go based on that as a starting point. Anything else? Any anything else you wanna tell this story? Make no mistakes. Shit.

## Canonical source anchors (required)
- Paper: **Creating General User Models from Computer Use** (Omar Shaikh et al.)
  - arXiv: https://arxiv.org/abs/2505.10831
  - paper page: https://oshaikh.com/papers/gums
- Project site: https://generalusermodels.github.io/
- Source repo: https://github.com/generalusermodels/gum
  - docs: https://generalusermodels.github.io/gum/

## Workspace discovery + assumption
- I searched the workspace for an existing local “gumbo paper” artifact (`gumbo`, `general user model`, `computer use`, related keywords).
- **Result:** no exact local paper/repo artifact was found in this workspace.
- **Assumption recorded (explicit):** proceed from the canonical online references above (paper + official GUM repo/docs) as the authoritative starting point.

## Research notes (what comes from paper/repo vs extension)
### Directly inspired by the GUM paper/repo
- Core abstraction: **Observations → Propositions (+ confidence) → Retrieval → Revision**.
- Natural-language proposition memory with confidence/uncertainty.
- Query interface over proposition memory for downstream apps.
- Mixed-initiative style suggestion scoring (benefit/cost vs interruption value).
- Local-first/privacy-aware framing (no mandatory external cloud dependency in this derivative experiment).

### This repo’s own extension work (not a clone)
- Structured event-first experimental pipeline for open reproducibility.
- Behavioral embedding + clustering model for user archetype mapping.
- Hybrid architecture combining proposition memory with statistical profile features.
- FastAPI service + Streamlit UI demo for practical experimentation.
- Synthetic dataset generator and tests for easy local validation.
- Explicit architecture/spec docs for open-source contributors.

## License constraints check
- Upstream `generalusermodels/gum` is MIT licensed.
- Implication: derivative/inspired open-source work is allowed with attribution.
- Implementation strategy here: **do not copy source code/assets**; re-implement architecture ideas and cite upstream paper/repo.

## Build plan
- [x] Initialize dedicated repo in `/Users/ethansk/.openclaw/workspace/files/general-user-model-experiment`.
- [x] Research canonical paper/repo and lock architecture direction to GUM references.
- [x] Implement core data model (observations, propositions, confidence, revision, retrieval).
- [x] Implement behavioral model extension (features/embeddings/clustering/next-action).
- [x] Build API (training, profiles, propositions, query, suggestions).
- [x] Build polished Streamlit UI (training/upload, user explorer, proposition memory, suggestions).
- [x] Add docs: README, architecture notes, spec paper, acknowledgment/citations.
- [x] Add tests and runnable scripts.
- [x] Run validation (tests + basic smoke run).
- [x] Commit incrementally.
- [x] Attempt GitHub repo create/push as `general-user-model-experiment`.

## Progress log
- 2026-03-20 16:05 — task started in target directory.
- 2026-03-20 16:10 — initial Python package scaffold created (simulation/model/api skeleton).
- 2026-03-20 16:13 — canonical references fetched/reviewed (arXiv abstract + HTML, official paper page, official repo, docs, license).
- 2026-03-20 16:13 — architecture corrected to be explicitly GUM-paper-grounded before continuing implementation.
- 2026-03-20 16:20 — implemented proposition-memory core (`propositions.py`): observation ingestion, proposition inference, confidence calibration, retrieval, revision/supersession.
- 2026-03-20 16:22 — implemented proactive suggestion engine (`suggestions.py`) with benefit/cost/urgency scoring.
- 2026-03-20 16:25 — upgraded `GeneralUserModel` to hybrid architecture (profile model + proposition memory + suggestions), plus robust small-dataset handling and save/load support.
- 2026-03-20 16:27 — expanded FastAPI surface with proposition and suggestion endpoints (`/propositions`, `/propositions/query`, `/suggestions/{user_id}`, `/context/{user_id}`).
- 2026-03-20 16:32 — built Streamlit UI (`app/streamlit_app.py`) covering training, cluster map, user explorer, proposition search, and suggestions tab.
- 2026-03-20 16:35 — authored open-source docs (`README.md`, `docs/ARCHITECTURE.md`, `docs/API.md`, `paper/SOURCE_DERIVATION.md`) and MIT license.
- 2026-03-20 16:38 — added test suite (`tests/test_*.py`) and helper scripts (`train_demo.py`, `generate_example_csv.py`) plus sample CSV.
- 2026-03-20 16:43 — created local `.venv` (Python 3.13), installed package + deps, and validated with `pytest` (4 passed) and `scripts/train_demo.py` smoke run.
- 2026-03-20 16:45 — committed incrementally in three commits (`feat(core)`, `feat(ui-docs)`, `test(scaffold)`).
- 2026-03-20 16:46 — created and pushed GitHub repo: `https://github.com/EthanSK/general-user-model-experiment`.
