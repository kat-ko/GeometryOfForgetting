# Geometry of Forgetting

Geometric signatures of sequential learning. We train small feedforward
networks on a stream of tasks and measure manifold capacity and its GLUE
decomposition at every task boundary.

The scientific output is a *measurement*, not a continual-learning method.
Do not add regularizers, replay, or CL algorithms unless a spec file asks
for them.

This repository is the standalone checkout of the former `continual_geometry`
tree.

## Setup

Python ≥ 3.10. From the repo root:

```bash
uv sync --all-extras
# or
pip install -e ".[dev,figures,vendored,notebook]"
```

Agent / contributor rules: [`docs/AGENTS.md`](docs/AGENTS.md) (also at
[`AGENTS.md`](AGENTS.md)). Math spec: [`docs/00-math-spec.md`](docs/00-math-spec.md).

## Layout

| Path | What it is |
|------|------------|
| `src/` | Models, training, manifolds, GLUE / capacity pipeline |
| `scripts/` | Grids, audits, figure generation |
| `tests/` | Validation suite |
| `docs/` | Specs, experimental design, write-up notes |
| `paper/` | TeX source |
| `figures/` | Paper figures (png/pdf/json) |
| `results/` | Summary JSON and `LOG.md` |
| `notebooks/` | Analysis notebook (reads stored results) |
| `third_party/` | Vendored capacity estimators |

## What git tracks vs what stays local

**Tracked:** source, tests, docs, TeX, figures, `uv.lock`, and the summary
artifacts at `results/*.json` plus `results/LOG.md`.

**Local only (gitignored):**

- `.venv/`
- per-arm campaign directories `results/<name>/` (thousands of JSON records;
  regenerable from the matching `scripts/run_*.py`). New campaign folders are
  ignored automatically — drop summaries as `results/<name>.json`.
- human-only reference PDFs in `papers/`
- LaTeX build junk under `paper/harness/`
