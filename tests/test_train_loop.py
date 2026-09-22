"""Sequential training loop: it must learn, it must forget, and it must be paired."""

from __future__ import annotations

import numpy as np
import pytest

from src.manifolds import dichotomies, generator
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init
from src.train import TrainConfig, run_stream, train_task
from src.train.loop import flatten_task, manifold_accuracy

P, M, D_AMB, D_INT, N = 8, 20, 30, 3, 64


def _setup(seed=0, gamma=1.0, steps=2000, lr0=0.2, target_loss=0.1):
    streams = paired_init(seed)
    arr = generator.make_arrangement(P, D_AMB, D_INT, 1.0, M, streams["data"])
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=gamma, lr0=lr0) for m in MODULES}
    model = TwoModuleNet.init(cfg, streams["shape"])
    return model, arr, streams, TrainConfig(
        steps_per_task=steps, record_every=100, target_loss=target_loss
    )


def test_flatten_task_shapes_and_targets():
    _, arr, _, _ = _setup()
    y = np.array([1.0, 1, 1, 1, -1, -1, -1, -1])
    X, target = flatten_task(arr.points, y)
    assert X.shape == (P * M, D_AMB) and target.shape == (P * M,)
    assert np.all(target[:M] == 1.0) and np.all(target[-M:] == -1.0)


def test_flatten_task_multi_output_and_group():
    _, arr, _, _ = _setup()
    from src.train.loop import flatten_group
    y = np.array([1.0, 1, 1, 1, -1, -1, -1, -1])
    y2 = np.array([-1.0, -1, -1, -1, 1, 1, 1, 1])
    X, Y = flatten_task(arr.points, np.stack([y, y2], axis=1))
    assert X.shape == (P * M, D_AMB) and Y.shape == (P * M, 2)
    Xg, Yg = flatten_group([arr.points, arr.points], np.stack([y, y2], axis=0))
    assert Xg.shape == (2 * P * M, D_AMB) and Yg.shape == (2 * P * M, 2)


def test_worst_of_k_k1_matches_matched_loss_steps():
    """K=1 worst-of-K is the same halt as matched_loss (pre-step prediction)."""
    y_steps = []
    for stop in ("matched_loss", "worst_of_k"):
        model, arr, streams, _ = _setup()
        y = dichotomies.sample_balanced(P, streams["stream"])
        cfg = TrainConfig(steps_per_task=20_000, record_every=100,
                          target_loss=0.1, stopping=stop)
        rec = train_task(model, arr.points, y, cfg, streams["data"])
        y_steps.append(rec.steps_taken)
        assert rec.converged
    assert y_steps[0] == y_steps[1]


def test_single_task_is_learnable():
    model, arr, streams, tcfg = _setup()
    y = dichotomies.sample_balanced(P, streams["stream"])
    rec = train_task(model, arr.points, y, tcfg, streams["data"])
    assert rec.train_accuracy > 0.95 and rec.converged
    assert rec.loss_curve[-1] < rec.loss_curve[0]


def test_weights_move_and_more_so_when_rich():
    """γ is the richness knob: larger γ must move the hidden weights more."""
    changes = {}
    for gamma in (0.1, 10.0):
        model, arr, streams, tcfg = _setup(gamma=gamma)
        y = dichotomies.sample_balanced(P, streams["stream"])
        rec = train_task(model, arr.points, y, tcfg, streams["data"])
        changes[gamma] = rec.weight_change["A"]
    assert changes[10.0] > changes[0.1]


def test_sequential_training_forgets():
    """The premise of the project: later tasks degrade earlier ones.

    Every task must converge first — otherwise the drop measures under-training.
    """
    model, arr, streams, tcfg = _setup()
    ys = np.stack([dichotomies.sample_balanced(P, streams["stream"]) for _ in range(4)])
    tasks, boundaries = run_stream(model, [arr.points] * 4, ys, tcfg, streams["data"])

    assert len(tasks) == 4 and len(boundaries) == 4
    assert all(t.converged for t in tasks)
    assert boundaries[0].retained_accuracy == {}
    assert boundaries[-1].current_accuracy > 0.95
    assert boundaries[-1].retained_accuracy[0] < tasks[0].train_accuracy
    assert np.isfinite(boundaries[-1].mean_retained_accuracy)


def test_under_training_is_flagged_not_silent():
    """A budget tuned on task 0 leaves later tasks unlearned; the run must say so."""
    model, arr, streams, _ = _setup()
    tight = TrainConfig(steps_per_task=60, record_every=100, target_loss=1e-4)
    ys = np.stack([dichotomies.sample_balanced(P, streams["stream"]) for _ in range(4)])
    tasks, _ = run_stream(model, [arr.points] * 4, ys, tight, streams["data"])
    assert not all(t.converged for t in tasks)
    assert not all(t.usable_for_forgetting for t in tasks)


def test_convergence_stops_early_when_target_reached():
    model, arr, streams, tcfg = _setup(steps=20_000)
    y = dichotomies.sample_balanced(P, streams["stream"])
    rec = train_task(model, arr.points, y, tcfg, streams["data"])
    assert rec.converged and rec.steps_taken < 20_000
    assert rec.final_loss <= tcfg.target_loss


def test_probe_dichotomy_is_recorded_and_never_trained():
    model, arr, streams, tcfg = _setup(steps=200)
    ys = np.stack([dichotomies.sample_balanced(P, streams["stream"]) for _ in range(2)])
    probe = dichotomies.sample_balanced(P, streams["probe"])
    _, boundaries = run_stream(
        model, [arr.points] * 2, ys, tcfg, streams["data"], probe_y=probe
    )
    assert all(np.isfinite(b.probe_accuracy) for b in boundaries)
    # the probe never appears among the trained dichotomies
    assert not any(np.array_equal(probe, y) for y in ys)


def test_run_is_reproducible_from_seed():
    out = []
    for _ in range(2):
        model, arr, streams, tcfg = _setup(seed=3, steps=150)
        ys = np.stack([dichotomies.sample_balanced(P, streams["stream"]) for _ in range(2)])
        tasks, _ = run_stream(model, [arr.points] * 2, ys, tcfg, streams["data"])
        out.append([t.final_loss for t in tasks])
    assert out[0] == out[1]


def test_representation_layout_matches_glue_expectation():
    model, arr, _, _ = _setup()
    reps = model.manifold_representation(arr.points, module="A")
    assert len(reps) == P and reps[0].shape == (N, M)
    both = model.manifold_representation(arr.points)
    assert both[0].shape == (2 * N, M)


def test_accuracy_at_init_is_zero_not_chance():
    """`f = 0` at init, and `sign(0) = 0` never matches ±1 — documented behaviour."""
    model, arr, streams, _ = _setup()
    y = dichotomies.sample_balanced(P, streams["stream"])
    assert manifold_accuracy(model, arr.points, y) == pytest.approx(0.0)


def test_accuracy_saturates_at_step_one_so_stopping_must_use_loss():
    """Why `stopping = "matched_loss"` and not an accuracy criterion.

    From `u = 0`, one step gives `u ∝ Σ_b y_b h(x_b)` — the kernel readout — whose
    *sign* does not depend on the learning rate. Train accuracy therefore jumps to
    ~0.99 at step 1 identically for every γ, while the loss is still far from
    converged. Stopping on accuracy would halt before any feature learning.
    """
    accs, losses = [], []
    for gamma in (0.03, 1.0, 10.0):
        model, arr, streams, tcfg = _setup(gamma=gamma)
        y = dichotomies.sample_balanced(P, streams["stream"])
        X, target = flatten_task(arr.points, y)
        model.sgd_step(X, target)
        from src.train.loop import accuracy as acc_fn
        accs.append(acc_fn(model, X, target))
        losses.append(0.5 * np.mean((model.forward(X) - target) ** 2))
    assert min(accs) > 0.9
    assert max(accs) - min(accs) < 1e-12      # identical across gamma
    assert min(losses) > 0.05                 # nowhere near converged
