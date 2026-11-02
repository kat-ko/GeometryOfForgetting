from pathlib import Path

p = Path("results/LOG.md")
text = p.read_text(encoding="utf-8")
if not text.endswith("\n"):
    text += "\n"
entry = """
---

## 2026-11-02 — v3 digest, line endings, BLAS thread count

**Laptop build the 270 arms ran under.** `.venv`, Python 3.11.9, NumPy 2.4.6,
scipy-openblas 0.3.31.188.0, `MAX_THREADS=24`, Windows. Checked today via
`numpy.show_config` in that environment. The October run did not print its
BLAS build; this is the environment still on disk, not a log line from the run.

**Thread count.** `parallel_map` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `MKL_NUM_THREADS` to 1 only after the parent has imported NumPy, and only
just before the pool starts. OpenBLAS reads the variable at library load.
Windows `ProcessPoolExecutor` spawns a new interpreter, so those workers
imported NumPy with the variable already set: the 270 arms were one BLAS
thread each. A forked worker inherits the parent's already-loaded library,
so the same code on Linux would run multi-threaded OpenBLAS inside every
worker. `v3_lib` now sets the four variables (those three plus
`NUMEXPR_NUM_THREADS`) before `import numpy`. The notebook setup cell does
the same before its own NumPy import. The cluster job script exports them
before `uv run`. `src/` is unchanged by that edit; `v3_lib.py` is not, so
the cache digest moves off `30a08ddd7c615f4c`. The existing
`notebooks/_v3_cache/grid.jsonl` will not hit. It stays on the laptop.

**Line endings.** No `.gitattributes` before this commit. `core.autocrlf=true`
in the system gitconfig. The index of `src/manifolds/streams.py` is LF (0 CR).
All 29 hashed `src/**/*.py` files were CRLF on disk, and `code_digest` hashes
file bytes. A Linux checkout of the same commit would hash different bytes.
Added `* text=auto eol=lf`. The digest to compare is the one computed after
the working tree is LF, on both machines, before any cache is trusted.
"""
if "v3 digest, line endings" in text:
    raise SystemExit("already")
p.write_text(text + entry, encoding="utf-8", newline="\n")
print("log appended")
