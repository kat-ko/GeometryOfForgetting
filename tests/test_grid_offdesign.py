"""An off-design arm in the registered grid directory must fail loudly, not be adopted.

`Phase1Spec.key` carries γ but not `N`, so an exploratory arm written to `results/phase1/`
does not necessarily collide with anything — it just joins the grid, and every figure and the
scope audit silently include it. These tests pin the refusal.
"""

from __future__ import annotations

import json

import pytest

from src.analysis import grid as G


def _arm(**over) -> dict:
    spec = {"gamma_0": 1.0, "a": 0.0, "condition": "S-HL", "seed": 0, "stream_id": 0,
            "N": G.REGISTERED_N}
    spec.update(over)
    return {"spec": spec, "key": "k", "usable": True, "geometry": [],
            "code": {"modules": {}}}


def _write(d, name, rec):
    (d / name).write_text(json.dumps(rec))


def test_on_design_arm_loads(tmp_path, monkeypatch):
    _write(tmp_path, "a.json", _arm())
    monkeypatch.setattr(G, "ARMS", tmp_path)
    assert len(G.load()) == 1


def test_unregistered_gamma_is_refused(tmp_path, monkeypatch):
    _write(tmp_path, "g30.json", _arm(gamma_0=30.0))
    monkeypatch.setattr(G, "ARMS", tmp_path)
    with pytest.raises(ValueError, match="off-design"):
        G.load()


def test_unregistered_width_is_refused(tmp_path, monkeypatch):
    """The `N` case is the nastier one: the filename would not even collide."""
    _write(tmp_path, "n600.json", _arm(N=600))
    monkeypatch.setattr(G, "ARMS", tmp_path)
    with pytest.raises(ValueError, match="off-design"):
        G.load()


def test_explicit_arms_dir_allows_off_design(tmp_path):
    """Loading an off-design directory on purpose is fine — that is the visible act."""
    _write(tmp_path, "g30.json", _arm(gamma_0=30.0))
    assert len(G.load(arms_dir=tmp_path)) == 1


def test_unusable_off_design_arm_still_refused(tmp_path, monkeypatch):
    """Refusal precedes the usability filter, or an unusable off-design arm would pass
    unnoticed and set a precedent for the next one."""
    _write(tmp_path, "g30.json", _arm(gamma_0=30.0, ) | {"usable": False})
    monkeypatch.setattr(G, "ARMS", tmp_path)
    with pytest.raises(ValueError, match="off-design"):
        G.load()


def test_every_registered_gamma_is_accepted(tmp_path, monkeypatch):
    for i, g in enumerate(G.REGISTERED_GAMMAS):
        _write(tmp_path, f"{i}.json", _arm(gamma_0=g))
    monkeypatch.setattr(G, "ARMS", tmp_path)
    assert len(G.load()) == len(G.REGISTERED_GAMMAS)
