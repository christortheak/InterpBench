"""Sweep default grid + depth-fraction resolution (recalibrated 2026-07-14).

The default grid is defined once per engine (Swift:
``ExperimentManifest.SweepSpec`` init defaults; server:
``tasks.DEFAULT_SWEEP_LAYER_FRACTIONS`` / ``DEFAULT_SWEEP_ALPHAS``) and MUST
stay identical across engines. Fractions resolve against the model's layer
count at sweep time with the same truncating/clamp/dedup/sort rule as Swift's
``SweepSpec.resolvedLayers``. Explicit grids in a manifest's sweep spec are
never touched by a default recalibration.
"""

import csv
import os

import test_sweep_promote as harness

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks


def test_default_grid_matches_swift_declare_default():
    # Researcher decision 2026-07-14 (live testing): stronger alphas
    # routinely push models into wasteful incoherence, and the live optimum
    # sits late in the network (L28/α0.08 on gemma-3-4b, ≈0.82 depth, inside
    # this grid). Twin assertion lives in Swift's OptimizationLifecycleTests.
    assert tasks.DEFAULT_SWEEP_LAYER_FRACTIONS == (0.5, 0.7, 0.85)
    assert tasks.DEFAULT_SWEEP_ALPHAS == (0.05, 0.08, 0.1, 0.13)


def test_default_fractions_resolve_against_model_depth():
    # gemma-3-4b depth: 34 blocks. 0.85·34 = 28.9 → L28, the live optimum.
    assert tasks.resolve_sweep_layers(
        34, tasks.DEFAULT_SWEEP_LAYER_FRACTIONS) == [17, 23, 28]
    # A 40-block model.
    assert tasks.resolve_sweep_layers(
        40, tasks.DEFAULT_SWEEP_LAYER_FRACTIONS) == [20, 28, 34]


def test_fraction_resolution_clamps_and_dedups():
    # 1.0 would name layer_count — clamps to the last valid block; 0 stays
    # the first block; near-duplicates collapse to one cell; output sorted.
    assert tasks.resolve_sweep_layers(10, [0.0, 1.0, 0.5, 0.51]) == [0, 5, 9]
    # Out-of-range garbage still lands on a valid block.
    assert tasks.resolve_sweep_layers(4, [-0.5, 2.0]) == [0, 3]


def test_spec_without_grid_falls_back_to_defaults(tmp_path, monkeypatch):
    # A sweep spec that declares instruments but no grid sweeps the DEFAULT
    # grid: fractions (0.5, 0.7, 0.85) on a 4-layer bundle → layers [2, 3],
    # × 4 default alphas = 8 steered cells (+ 1 baseline row).
    root = str(tmp_path)
    harness._sweep_workspace(root, "swdef")
    d = es.load_raw("swdef", root)
    d["sweep"] = {"devPromptsFile": "prompts/dev/dev.jsonl",
                  "batteryFile": "prompts/batteries/b.jsonl",
                  "maxTokens": 16}
    es.save_raw(d, root)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": harness._fake_bundle()})
    monkeypatch.setattr(tasks, "generate", harness._fake_generate())

    run_dir = tasks.sweep("swdef", root, model_provider=harness._fake_model,
                          log=lambda *_: None)

    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steered = [r for r in rows if int(r["layer"]) >= 0]
    assert sorted({int(r["layer"]) for r in steered}) == [2, 3]
    assert sorted({float(r["alpha"]) for r in steered}) == sorted(
        tasks.DEFAULT_SWEEP_ALPHAS)
    assert len(steered) == 2 * len(tasks.DEFAULT_SWEEP_ALPHAS)
    # The recommended condition's cell comes from the default grid.
    d = es.load_raw("swdef", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    slot = cond["slots"][0]
    assert slot["layer"] in (2, 3)
    assert slot["alpha"] in tasks.DEFAULT_SWEEP_ALPHAS


def test_explicit_grid_overrides_defaults(tmp_path, monkeypatch):
    # A manifest that declares its own grid is untouched by the default
    # recalibration: the sweep runs exactly the declared cells (which sit
    # OUTSIDE the current default grid).
    root = str(tmp_path)
    harness._sweep_workspace(root, "swexp")
    d = es.load_raw("swexp", root)
    d["sweep"]["layerFractions"] = [0.35]
    d["sweep"]["alphas"] = [0.04, 0.12]
    es.save_raw(d, root)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": harness._fake_bundle()})
    monkeypatch.setattr(tasks, "generate", harness._fake_generate())

    run_dir = tasks.sweep("swexp", root, model_provider=harness._fake_model,
                          log=lambda *_: None)

    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steered = [r for r in rows if int(r["layer"]) >= 0]
    # 0.35 · 4 layers → layer 1 only; alphas exactly as declared.
    assert {int(r["layer"]) for r in steered} == {1}
    assert sorted(float(r["alpha"]) for r in steered) == [0.04, 0.12]
