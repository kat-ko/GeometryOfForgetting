"""Resume must skip completed arms, and the default must not destroy them.

Written after the default destroyed the pilot. `--resume` was opt-in and deletion was the
default, so re-invoking `--pilot` to *test* resume removed all eight completed arms
before recomputing them. `results/phase1/` is gitignored, so there was no way back. On
the 4-hour grid the same keystroke would have cost the run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase1 as R  # noqa: E402
from src import pipeline as pl  # noqa: E402


@pytest.fixture
def arm(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path)
    (tmp_path / "smoke").mkdir(parents=True, exist_ok=True)
    return pl.Phase1Spec()


def _write(spec, **override):
    path = R._path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"spec": {**asdict(spec), **override}, "key": spec.key, "usable": True,
           "geometry": [], "attribution": [], "wall_seconds": 1.0,
           "forgetting": {"CFr": 0.0}}
    path.write_text(json.dumps(rec))
    return path


def test_a_completed_arm_is_skipped_not_recomputed(arm, monkeypatch):
    _write(arm)
    monkeypatch.setattr(pl, "run_arm", lambda s: pytest.fail("recomputed a stored arm"))
    out = R._job(arm)
    assert out["skipped"] is True


def test_an_arm_whose_spec_changed_is_recomputed(arm, monkeypatch):
    """A file at the right path is not enough; it must be the same experiment."""
    _write(arm, n_t=999)
    calls = []
    monkeypatch.setattr(pl, "run_arm", lambda s: calls.append(s) or {
        "usable": True, "geometry": [], "attribution": [], "wall_seconds": 1.0,
        "forgetting": {"CFr": 0.0}, "spec": asdict(s), "key": s.key})
    out = R._job(arm)
    assert out["skipped"] is False and len(calls) == 1


def test_tuple_vs_list_round_trip_does_not_force_a_recompute(arm, monkeypatch):
    """`module_list` is a tuple in Python and a list after JSON, so a naive == fails."""
    path = _write(arm)
    stored = json.loads(path.read_text())
    assert any(isinstance(v, list) for v in stored["spec"].values()), \
        "no sequence field left to catch the tuple/list asymmetry"
    monkeypatch.setattr(pl, "run_arm", lambda s: pytest.fail("recomputed on round-trip"))
    assert R._job(arm)["skipped"] is True


def _run_main(monkeypatch, tmp_path, argv, arm):
    """Drive `main()` through the run path, stubbing only the compute and the write."""
    monkeypatch.setattr(sys, "argv", ["run_phase1.py", *argv])
    monkeypatch.setattr(R, "arms", lambda **kw: [arm])
    monkeypatch.setattr(R, "pmap", lambda fn, jobs, **kw: [
        {"key": "k", "path": "", "skipped": True} for _ in jobs])
    monkeypatch.setattr(R, "summarize", lambda paths: {
        "n_arms": len(paths), "n_usable": len(paths), "max_identity_residual": 0.0,
        "attribution": {}, "rho_coverage": {}, "forgetting_CFr": {}})
    monkeypatch.setattr(R, "ROOT", tmp_path)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    R.main()


def test_the_default_invocation_does_not_delete_existing_arms(arm, monkeypatch, tmp_path):
    """The regression that cost the pilot, exercising the branch that did it."""
    path = _write(arm)
    _run_main(monkeypatch, tmp_path, [], arm)
    assert path.exists(), "a default invocation removed a completed arm"


def test_fresh_moves_to_trash_rather_than_deleting(arm, monkeypatch, tmp_path):
    """Discarding is explicit, and still recoverable."""
    path = _write(arm)
    _run_main(monkeypatch, tmp_path, ["--fresh"], arm)
    assert not path.exists(), "--fresh left the old arm in place"
    trashed = list(R.OUT.glob(".trash-*/*.json"))
    assert len(trashed) == 1, f"--fresh did not preserve the discarded arm: {trashed}"
    assert json.loads(trashed[0].read_text())["key"] == arm.key


def test_resume_flag_is_accepted_for_compatibility(arm, monkeypatch, tmp_path):
    path = _write(arm)
    _run_main(monkeypatch, tmp_path, ["--resume"], arm)
    assert path.exists()
