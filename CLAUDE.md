# GeometryOfForgetting

See AGENTS.md and docs/PROJECT.md for project context.

## Cluster
Runs on ITU `real`. Cluster rules are in ~/.claude/CLAUDE.md.
Job scripts live in `sbatch/`.

## Known constraints
- `notebooks/run_v3.py` is Windows-only (sleep inhibit + `gof-venv` kernel).
  Cluster jobs must call `run_grid` directly, or `nbconvert` after `uv sync`.
- Cache keys hash `src/` + `v3_lib.py`. Editing either invalidates `notebooks/_v3_cache/`.
- `notebooks/_v3_cache/` is untracked. Copy `grid.jsonl` (4.9 MB) manually to resume the 270 arms.
