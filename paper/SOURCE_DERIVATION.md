# Source Derivation and Extension Notes

This document explicitly distinguishes what is adapted from the GUM paper/repo and what is newly introduced in this repository.

## Canonical sources

- Paper: *Creating General User Models from Computer Use* (Shaikh et al., 2025)
  - https://arxiv.org/abs/2505.10831
- Project site: https://generalusermodels.github.io/
- Upstream repository: https://github.com/generalusermodels/gum (MIT)

## Source-derived concepts used here

- Build user models from broad computer-use observations.
- Represent inferred user knowledge/preferences as confidence-weighted propositions.
- Retrieve propositions as contextual memory.
- Revise proposition memory over time as new evidence arrives.
- Score proactive assistant actions using benefit/cost framing.

## New implementation choices in this repo

- Structured CSV/synthetic telemetry pipeline as first-class input (instead of full observer stack).
- Heuristic proposition inference engine for local reproducibility.
- Explicit proposition revision groups and supersession tracking.
- Hybrid model combining proposition memory with statistical profile embeddings/clusters.
- FastAPI + Streamlit reference stack for open-source demo usability.
- Test-first scaffold for local iteration and CI adoption.

## Non-goals / differences from upstream

- No direct code reuse from upstream GUM internals.
- No requirement for heavyweight multimodal runtime stack.
- Not a drop-in replacement for the upstream package.

## License handling

- Upstream GUM is MIT.
- This repository is MIT and includes attribution/citation.
- No upstream source code copied into this codebase.
