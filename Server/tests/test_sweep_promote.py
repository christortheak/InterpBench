"""Sweep selection criterion as manifest data + headless Promote.

The screening→confirmation lifecycle edge: a spec'd sweep selects its
``<concept>-recommended`` cell under a DATA-declared criterion (objective /
constraints / controls), stamps full selection provenance, and ``promote``
mints an agent (variant artifact) carrying that birth certificate. No GPU —
the decision rule is a pure function, and the sweep integration test fakes
extraction + generation.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import model_variant
from steerlab_server.experiment import promote as promote_mod
from steerlab_server.experiment import recipe_identity
from steerlab_server.experiment import sweep_selection as sel
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import SUBSTRATE, ConceptVectors


# --- criterion resolution ----------------------------------------------------

def test_selection_defaults_match_legacy_hardcoded_rule():
    c = sel.resolve_selection(None)
    assert c.metric == "markerDensity"
    assert c.capability_tolerance == 0.15
    assert c.coherence_floor == 0.45
    assert c.matched_norm_random_margin is None


def test_selection_explicit_values_round_trip():
    c = sel.resolve_selection({
        "objective": {"metric": "markerDensity"},
        "constraints": {"capabilityTolerance": 0.2, "coherenceFloor": 0.5},
        "controls": {"matchedNormRandomMargin": 0.05}})
    assert (c.capability_tolerance, c.coherence_floor) == (0.2, 0.5)
    assert c.matched_norm_random_margin == 0.05
    d = c.to_dict()
    assert d["objective"]["metric"] == "markerDensity"
    assert d["controls"]["matchedNormRandomMargin"] == 0.05


def test_all_declared_metrics_now_resolve():
    # These were declared-ahead refusals until the instruments landed
    # (2026-07-08); resolve now accepts all three.
    for metric in ("markerDensity", "judgeScore", "logprobShift"):
        c = sel.resolve_selection({"objective": {"metric": metric}})
        assert c.metric == metric


def test_unknown_metric_is_an_error():
    with pytest.raises(ValueError, match="unknown selection metric"):
        sel.resolve_selection({"objective": {"metric": "vibes"}})


def test_criterion_range_validation_fails_at_resolve():
    # capabilityTolerance must be finite and in [0, 1].
    for bad in (1.5, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="capabilityTolerance"):
            sel.resolve_selection({"constraints": {"capabilityTolerance": bad}})
    # coherenceFloor must be finite and in [0, 1].
    for bad in (1.5, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="coherenceFloor"):
            sel.resolve_selection({"constraints": {"coherenceFloor": bad}})
    # matchedNormRandomMargin must be finite and >= 0.
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="matchedNormRandomMargin"):
            sel.resolve_selection({"controls": {"matchedNormRandomMargin": bad}})
    # Boundary values are legal.
    c = sel.resolve_selection({
        "constraints": {"capabilityTolerance": 0, "coherenceFloor": 1},
        "controls": {"matchedNormRandomMargin": 0}})
    assert (c.capability_tolerance, c.coherence_floor,
            c.matched_norm_random_margin) == (0.0, 1.0, 0.0)


# --- the pure decision rule ----------------------------------------------------

BASE = sel.BaselineCell(metric=0.02, distinct2=0.8, battery_accuracy=0.9)


def _cell(layer, alpha, metric, distinct2=0.8, acc=0.9):
    return sel.SweepCell(layer=layer, alpha=alpha, metric=metric,
                         distinct2=distinct2, battery_accuracy=acc)


def test_select_best_eligible_cell_by_objective():
    cells = [_cell(3, 0.1, 0.1), _cell(3, 0.2, 0.4), _cell(5, 0.1, 0.3)]
    best = sel.select_cell(cells, BASE, sel.SelectionCriterion())
    assert (best.layer, best.alpha) == (3, 0.2)


def test_constraints_filter_cells():
    cells = [
        _cell(3, 0.2, 0.9, acc=0.7),         # capability drop beyond tolerance
        _cell(3, 0.4, 0.8, distinct2=0.3),   # coherence below floor
        _cell(5, 0.1, 0.2),                  # the only eligible cell
    ]
    best = sel.select_cell(cells, BASE, sel.SelectionCriterion())
    assert (best.layer, best.alpha) == (5, 0.1)


def test_winner_must_exceed_baseline_metric():
    assert sel.select_cell([_cell(3, 0.1, 0.02)], BASE,
                           sel.SelectionCriterion()) is None


def test_nothing_eligible_returns_none():
    assert sel.select_cell([], BASE, sel.SelectionCriterion()) is None
    only_bad = [_cell(3, 0.1, 0.5, distinct2=0.1)]
    assert sel.select_cell(only_bad, BASE, sel.SelectionCriterion()) is None


def test_control_margin_pass_and_fail():
    assert sel.control_passes(0.4, 0.1, 0.2)
    assert not sel.control_passes(0.4, 0.35, 0.2)
    assert "matched-norm control" in sel.control_failure_message(0.4, 0.35, 0.2)


# --- provenance plumbing -------------------------------------------------------

#: A representative sweep refusal, in the shape `no_selection_reason` now
#: produces (E3): the reason, plus the numbers that were binding.
FAILED_SELECTION = ("no cell passed the capability/coherence gates "
                    "(tolerance 0.15, floor 0.45)")

SELECTION_BLOCK = {
    "sweepRun": "20260707T000000000-exp-x-sweep",
    "criterion": sel.SelectionCriterion().to_dict(),
    "devPromptsHash": "d" * 64,
    "devMaxTokens": 1024,
    "winningCell": {"layer": 3, "alpha": 0.4},
    "metrics": {"markerDensity": 0.31, "baselineDensity": 0.02,
                "distinct2": 0.61, "batteryAccuracy": 0.9,
                "baselineBatteryAccuracy": 0.95},
}


def test_add_condition_preserves_selection_provenance(tmp_path):
    root = str(tmp_path)
    es.create("x", model_id="org/m", root=root)
    es.add_condition("x", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    d = es.load_raw("x", root)
    assert d["conditions"][0]["selection"] == SELECTION_BLOCK
    # The parsed manifest still loads with the extra field present.
    m = Manifest.load("x", root)
    assert m.conditions[0].name == "fear-recommended"
    assert m.conditions[0].alpha_in_norm_units is True


# --- shared fixtures -------------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _concept_fixture(root, name="fear"):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"),
           '{"text": "I feel dread"}\n{"text": "terror grips me"}\n')
    _write(os.path.join(d, "negative.jsonl"),
           '{"text": "calm morning"}\n{"text": "a quiet walk"}\n')
    _write(os.path.join(d, "markers.json"), json.dumps({"words": ["dread"]}))


def _experiment_with_concept(root, name, concept="fear", revision="abc"):
    _concept_fixture(root, concept)
    es.create(name, model_id="org/m", revision=revision, root=root)
    es.attach(name, [concept], root=root)
    return Manifest.load(name, root).concepts[0].stimulus_set_hash


def _vector_artifact(root, *, concept="fear", stimulus_hash, run="20260707T000001000-extract",
                     substrate=SUBSTRATE, method="meanDifference", revision="abc",
                     reading_position="last token", neutral_projection="none",
                     residual_norm_source="extraction-stimuli", extras=None):
    """A FULL recipe sidecar (reading position + projection + norm source),
    matching what the extraction writers stamp — so the artifact can prove
    its recipe identity to promote. Pass ``None`` / ``extras`` to simulate
    legacy or divergent artifacts."""
    run_dir = os.path.join(root, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, f"{concept}.safetensors"), "wb") as handle:
        handle.write(b"weights")
    sidecar = {"modelID": "org/m", "concept": concept, "layerCount": 4,
               "hiddenSize": 2, "stimulusSetHash": stimulus_hash,
               "normsPerLayer": [1.0] * 4, "residualNormPerLayer": [1.0] * 4}
    for key, value in (("extractionMethod", method), ("revision", revision),
                       ("substrate", substrate),
                       ("readingPosition", reading_position),
                       ("neutralProjection", neutral_projection),
                       ("residualNormSource", residual_norm_source)):
        if value is not None:
            sidecar[key] = value
    sidecar.update(extras or {})
    _write(os.path.join(run_dir, f"{concept}.json"), json.dumps(sidecar))
    return os.path.join("runs", run, concept)


# --- sweep integration (faked extraction + generation) ---------------------------

@contextmanager
def _fake_model(model_id, revision):
    # The sweep persists its re-derived vectors (sidecars stamp
    # ``model.revision``), so the fake must carry the attribute.
    yield SimpleNamespace(revision=revision)


def _fake_bundle(stimulus_hash="h"):
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        # The canonical source token a neutral-corpus-less extraction records
        # — the persisted sidecar must satisfy promote's full recipe-identity
        # match.
        residual_norm_source="extraction-stimuli", stimulus_hash=stimulus_hash)


def _spec(selection=None):
    spec = {"layerFractions": [0.5], "alphas": [0.4],
            "devPromptsFile": "prompts/dev/dev.jsonl",
            "batteryFile": "prompts/batteries/b.jsonl", "maxTokens": 16}
    if selection is not None:
        spec["selection"] = selection
    return spec


def _sweep_workspace(root, name, selection=None, revision="abc"):
    _experiment_with_concept(root, name, revision=revision)
    dev_hash = _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
                      '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    d = es.load_raw(name, root)
    d["sweep"] = _spec(selection)
    es.save_raw(d, root)
    return dev_hash


def _fake_generate(steered=("dread filled the quiet town before dawn broke 2",),
                   plain="the town woke slowly to a bright morning 2"):
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False):
        return steered[0] if injections else plain
    return generate


def test_spec_sweep_selects_and_stamps_provenance(tmp_path, monkeypatch):
    root = str(tmp_path)
    dev_hash = _sweep_workspace(root, "sw")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    run_dir = tasks.sweep("sw", root, model_provider=_fake_model, log=lambda *_: None)

    d = es.load_raw("sw", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    assert cond["alphaInNormUnits"] is True
    assert cond["slots"] == [{"concept": "fear", "layer": 2, "alpha": 0.4}]
    block = cond["selection"]
    assert block["sweepRun"] == os.path.basename(run_dir)
    assert block["devPromptsHash"] == dev_hash
    assert block["winningCell"] == {"layer": 2, "alpha": 0.4}
    assert block["criterion"]["objective"]["metric"] == "markerDensity"
    assert block["criterion"]["constraints"]["capabilityTolerance"] == 0.15
    assert block["metrics"]["markerDensity"] > block["metrics"]["baselineDensity"]
    assert block["metrics"]["batteryAccuracy"] == 1.0
    # recommendations.json carries the same block; sweep.csv has the baseline row.
    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
        recs = json.load(h)
    assert recs["fear"] == block
    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as h:
        csv_text = h.read()
    assert "fear,-1,0," in csv_text


def test_spec_sweep_declared_criterion_overrides_defaults(tmp_path, monkeypatch):
    # A stricter declared coherence floor (0.7, in range) rejects a steered
    # text whose distinct-2 is 0.6 — a cell the DEFAULT floor (0.45) accepts —
    # proving the manifest data, not compiled code, decides.
    root = str(tmp_path)
    _sweep_workspace(root, "strict", selection={
        "constraints": {"coherenceFloor": 0.7}})
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate(
        steered=("dread filled the quiet town dread filled the quiet town 2",)))

    run_dir = tasks.sweep("strict", root, model_provider=_fake_model,
                          log=lambda *_: None)

    d = es.load_raw("strict", root)
    assert not any(c["name"] == "fear-recommended" for c in d.get("conditions", []))
    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
        recs = json.load(h)
    # The refusal now also names the BINDING numbers (E3), so match the
    # reason rather than a frozen sentence.
    assert "no cell passed the capability/coherence gates" in recs["fear"]


def test_spec_sweep_control_margin_can_refuse(tmp_path, monkeypatch):
    # The fake generator steers identically for concept and control cells, so
    # any positive margin must refuse the recommendation — and record why.
    root = str(tmp_path)
    _sweep_workspace(root, "ctl", selection={
        "controls": {"matchedNormRandomMargin": 0.1}})
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    run_dir = tasks.sweep("ctl", root, model_provider=_fake_model,
                          log=lambda *_: None)

    d = es.load_raw("ctl", root)
    assert not any(c["name"] == "fear-recommended" for c in d.get("conditions", []))
    with open(os.path.join(run_dir, "recommendations.json"), encoding="utf-8") as h:
        recs = json.load(h)
    assert "matched-norm control" in recs["fear"]


def test_spec_sweep_control_block_stamps_random_vector_algorithm(tmp_path, monkeypatch):
    # A PASSING control (margin 0; the fake generator steers identically for
    # concept and control cells, so best - control == 0) must stamp WHICH
    # random-control recipe generated the direction — the cross-engine
    # contract string; unstamped control blocks are legacy runs.
    root = str(tmp_path)
    _sweep_workspace(root, "ctlstamp", selection={
        "controls": {"matchedNormRandomMargin": 0.0}})
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    tasks.sweep("ctlstamp", root, model_provider=_fake_model, log=lambda *_: None)

    d = es.load_raw("ctlstamp", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    control = cond["selection"]["control"]
    assert control["type"] == "randomMatchedNorm"
    assert control["randomVectorAlgorithm"] == "gaussian-isotropic-v1"
    assert control["randomVectorAlgorithm"] == tasks.RANDOM_VECTOR_ALGORITHM


# --- live previews + per-generation cancellation ---------------------------------

def test_preview_line_collapses_whitespace_and_truncates_unicode_safely():
    assert tasks._preview_line("a\nb\r\n\tc") == "a b c"
    assert tasks._preview_line("short") == "short"
    exactly = "x" * 160
    assert tasks._preview_line(exactly) == exactly  # no ellipsis at the limit
    over = "é" * 200                                # multi-byte code points
    out = tasks._preview_line(over)
    assert out == "é" * 160 + "…"
    assert len(out) == 161
    spaced = ("word " * 64).strip()                 # cut lands on a space run
    cut = tasks._preview_line(spaced)
    assert cut.endswith("…") and not cut[:-1].endswith(" ")
    assert "\n" not in tasks._preview_line("line one\nline two")


def test_spec_sweep_cancel_observed_between_generations(tmp_path, monkeypatch):
    # The live bug (2026-07-09): a cancel between concept/layer boundaries let
    # one layer × all alphas × all dev prompts + battery run to completion.
    # Cancel latency must now be at most ONE generation, with the same
    # partial-output bookkeeping as a boundary cancel — and every dev
    # generation logs a truncated preview so decoherence is visible live.
    root = str(tmp_path)
    _sweep_workspace(root, "cx")
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n'
           '{"text": "Write about the sea."}\n')
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    inner = _fake_generate()
    calls = {"n": 0}

    def counting_generate(*args, **kwargs):
        calls["n"] += 1
        return inner(*args, **kwargs)

    monkeypatch.setattr(tasks, "generate", counting_generate)
    logs = []
    # Baseline = 2 dev + 1 battery generations; the cancel arrives after the
    # grid cell's FIRST dev generation (call 4) and must stop right there.
    run_dir = tasks.sweep("cx", root, model_provider=_fake_model,
                          should_cancel=lambda: calls["n"] >= 4,
                          log=logs.append)

    assert calls["n"] == 4  # nothing generated after the cancel was observable
    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as h:
        csv_text = h.read()
    assert "fear,-1,0," in csv_text        # completed baseline row kept
    assert "fear,2,0.4," not in csv_text   # the interrupted cell is dropped
    # A cancelled sweep is an explicit resumable PARTIAL, never "complete"
    # (review 2026-08-03 round 2, P1): recommendations.json is the sweep's
    # completion marker, so it must NOT exist; the cancel parks the
    # directory with the same state file a checkpoint writes.
    from steerlab_server.experiment import resume as resume_mod
    assert not os.path.exists(os.path.join(run_dir, "recommendations.json"))
    assert resume_mod.is_resumable(run_dir, "sweep")
    assert resume_mod.read_state(run_dir)["reason"] == "cancel"
    assert any("cancel observed" in line
               and "stopping after the current generation" in line
               for line in logs)
    assert any(line.startswith('baseline dev 1/2: "') for line in logs)
    assert any(line.startswith('L2 α0.4 dev 1/2: "') for line in logs)


def test_sweep_refuses_unpinned_judge_score_before_model_load(tmp_path):
    # judgeScore is implemented, but its config comes from MANIFEST pins:
    # with no rubric pin the sweep must still refuse before the model loads.
    root = str(tmp_path)
    _sweep_workspace(root, "jm", selection={"objective": {"metric": "judgeScore"}})

    def exploding_provider(model_id, revision):  # pragma: no cover - must not run
        raise AssertionError("model must not load when the criterion is invalid")

    with pytest.raises(ValueError, match="pinned judge rubric"):
        tasks.sweep("jm", root, model_provider=exploding_provider,
                    log=lambda *_: None)


def test_legacy_specless_sweep_keeps_old_behavior(tmp_path, monkeypatch):
    # No "sweep" key in the manifest: the legacy single-prompt grid runs and
    # nothing is appended to the manifest.
    root = str(tmp_path)
    _experiment_with_concept(root, "legacy")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    run_dir = tasks.sweep("legacy", root, model_provider=_fake_model,
                          log=lambda *_: None)

    assert os.path.exists(os.path.join(run_dir, "sweep.csv"))
    assert not os.path.exists(os.path.join(run_dir, "recommendations.json"))
    assert not es.load_raw("legacy", root).get("conditions")


def test_sweep_persists_vectors_and_promote_matches_them(tmp_path, monkeypatch):
    # The live bug (2026-07-08): a successful sweep derived vectors in memory
    # but never persisted them, so Create Agent (promote) refused with "no
    # extraction artifact … matches this experiment's recipe". The sweep run
    # itself must now carry recipe-matching sidecars — written by the SAME
    # persist helper extract/validate use — and promote must succeed with NO
    # separate extract run.
    root = str(tmp_path)
    _sweep_workspace(root, "swpro")
    stimulus_hash = Manifest.load("swpro", root).concepts[0].stimulus_set_hash
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle(stimulus_hash)})
    monkeypatch.setattr(tasks, "generate", _fake_generate())

    run_dir = tasks.sweep("swpro", root, model_provider=_fake_model,
                          log=lambda *_: None)

    # The sweep run directory carries the artifact + a full recipe sidecar.
    assert os.path.exists(os.path.join(run_dir, "fear.safetensors"))
    with open(os.path.join(run_dir, "fear.json"), encoding="utf-8") as h:
        sidecar = json.load(h)
    assert sidecar["stimulusSetHash"] == stimulus_hash
    assert sidecar["extractionMethod"] == "meanDifference"
    assert sidecar["revision"] == "abc"
    assert sidecar["substrate"] == SUBSTRATE
    # The persisted sidecar is stamped with the canonical full-recipe
    # identity — the normal promote match path going forward.
    manifest = Manifest.load("swpro", root)
    assert sidecar["recipeIdentityHash"] == recipe_identity.identity_hash(
        recipe_identity.required_identity(manifest, manifest.concepts[0]))

    # Promote finds it: the ONLY artifact in the workspace is the sweep's.
    out = promote_mod.promote("swpro", "fear", root=root, log=lambda *_: None)
    injection = out["variant"]["injections"][0]
    assert injection["vectorArtifactID"] == os.path.join(
        "runs", os.path.basename(run_dir), "fear")
    assert out["variant"]["promotion"]["promotedBy"] == "criterion"


RESOLVED_REVISION = "8f2a" + "0" * 36


@contextmanager
def _resolving_model(model_id, revision):
    # ``model_loader.load`` resolves a revision-less load to the HF cache's
    # concrete commit (``revision or cached_revision(model_id)``) and the
    # sidecar writer stamps THAT — the fake mirrors the real resolution.
    yield SimpleNamespace(revision=revision or RESOLVED_REVISION)


def test_sweep_on_unpinned_draft_pins_revision_and_promote_matches(
        tmp_path, monkeypatch):
    # THE 2026-07-14 live bug: localhost server, gemma-3-4b-it, experiment
    # created WITHOUT --revision. The sweep loaded the model (resolving the
    # concrete cached revision), persisted sidecars stamping that resolved
    # revision — but the manifest still carried null, so promote's required
    # identity NEVER matched the sweep's own artifacts ("2 candidate
    # artifact(s) carry a DIFFERENT recipe identity"). Every model-loading
    # task must now pin the resolved revision into a DRAFT manifest before
    # persisting artifacts, exactly like Swift ``loadContainer(pinning:)`` —
    # so the user's flow (sweep → Create Agent) succeeds with no extra
    # extract run.
    root = str(tmp_path)
    _sweep_workspace(root, "userflow", revision=None)
    assert Manifest.load("userflow", root).model_revision is None
    stimulus_hash = Manifest.load("userflow", root).concepts[0].stimulus_set_hash
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle(stimulus_hash)})
    monkeypatch.setattr(tasks, "generate", _fake_generate())
    logs = []

    run_dir = tasks.sweep("userflow", root, model_provider=_resolving_model,
                          log=logs.append)

    # The draft manifest now pins the revision the sweep actually ran, loudly.
    assert Manifest.load("userflow", root).model_revision == RESOLVED_REVISION
    assert any("pinned model revision" in line for line in logs)
    # The persisted sidecar and the manifest agree on the concrete revision.
    with open(os.path.join(run_dir, "fear.json"), encoding="utf-8") as h:
        assert json.load(h)["revision"] == RESOLVED_REVISION
    # Create Agent (the user's exact next step) succeeds.
    out = promote_mod.promote("userflow", "fear", root=root, log=lambda *_: None)
    assert out["variant"]["promotion"]["promotedBy"] == "criterion"
    assert out["variant"]["baseRevision"] == RESOLVED_REVISION
    assert out["variant"]["injections"][0]["vectorArtifactID"] == os.path.join(
        "runs", os.path.basename(run_dir), "fear")


def test_extract_task_pins_unpinned_draft_revision(tmp_path, monkeypatch):
    # The same auto-pin fires on extract (and validate/run — shared helper):
    # the first model-loading task records the revision that actually ran.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "expin", revision=None)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle(stimulus_hash)})
    logs = []
    tasks.extract("expin", root, model_provider=_resolving_model,
                  log=logs.append)
    assert Manifest.load("expin", root).model_revision == RESOLVED_REVISION
    assert any("pinned model revision" in line for line in logs)


def test_frozen_manifest_without_revision_is_never_pinned_but_warned(tmp_path,
                                                                     monkeypatch):
    # Frozen manifests are immutable — the auto-pin must not touch them, but
    # the run says loudly which revision it actually used (Swift parity).
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "frzpin", revision=None)
    d = es.load_raw("frzpin", root)
    d["status"] = "frozen"
    es.save_raw(d, root, freeze_transition=True)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle(stimulus_hash)})
    logs = []
    tasks.extract("frzpin", root, model_provider=_resolving_model,
                  log=logs.append)
    assert Manifest.load("frzpin", root).model_revision is None
    assert any("without a pinned model revision" in line for line in logs)


# --- promote -----------------------------------------------------------------------

def _plant_completed_sweep_run(root, sweep_run=SELECTION_BLOCK["sweepRun"],
                               concept="fear", winning=None):
    """The completed sweep run a `<concept>-recommended` condition names:
    promotion now verifies the run's completion marker and that its
    recommendation still matches the condition (review 2026-08-03 round 3,
    P2), so promotable fixtures must carry real evidence."""
    run_dir = os.path.join(root, "runs", sweep_run)
    os.makedirs(run_dir, exist_ok=True)
    entry = {**SELECTION_BLOCK,
             "winningCell": winning or {"layer": 3, "alpha": 0.4}}
    with open(os.path.join(run_dir, "recommendations.json"), "w",
              encoding="utf-8") as handle:
        json.dump({concept: entry}, handle)
    return run_dir


def _promotable(root, name="pr"):
    stimulus_hash = _experiment_with_concept(root, name)
    artifact_rel = _vector_artifact(root, stimulus_hash=stimulus_hash)
    es.add_condition(name, {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": {**SELECTION_BLOCK,
                      "winningCell": {"layer": 3, "alpha": 0.4}}}, root)
    _plant_completed_sweep_run(root)
    return artifact_rel


def test_promote_refuses_condition_whose_sweep_run_is_missing(tmp_path):
    """A `<concept>-recommended` condition is a PROJECTION of a sweep run,
    never evidence in itself (review 2026-08-03 round 3, P2)."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gm")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    es.add_condition("gm", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    with pytest.raises(promote_mod.PromoteError,
                       match="which is not in runs/"):
        promote_mod.promote("gm", "fear", root=root, log=lambda *_: None)


def test_promote_refuses_condition_whose_sweep_never_completed(tmp_path):
    """The save-before-marker crash window: the condition landed, the run
    exists, but recommendations.json (the completion marker) never did."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gi")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    es.add_condition("gi", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    os.makedirs(os.path.join(root, "runs", SELECTION_BLOCK["sweepRun"]),
                exist_ok=True)  # run directory, no completion marker
    with pytest.raises(promote_mod.PromoteError,
                       match="that sweep never completed"):
        promote_mod.promote("gi", "fear", root=root, log=lambda *_: None)


def test_promote_refuses_condition_stale_against_its_run(tmp_path):
    """The run completed, but its recommendation no longer says what the
    manifest condition says — the condition is stale."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gs")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    es.add_condition("gs", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root, winning={"layer": 5, "alpha": 0.1})
    with pytest.raises(promote_mod.PromoteError,
                       match="the manifest condition is stale"):
        promote_mod.promote("gs", "fear", root=root, log=lambda *_: None)


def test_promote_fails_closed_on_condition_without_sweep_run(tmp_path):
    """FAIL-CLOSED (round 4, P2): both engines' schemas stamp sweepRun, so
    a condition without one is hand-written or legacy — not criterion
    evidence. The explicit cell override (stamped manualOverride) remains
    the deliberate escape hatch."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gl")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    legacy = {k: v for k, v in SELECTION_BLOCK.items() if k != "sweepRun"}
    es.add_condition("gl", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": legacy}, root)
    with pytest.raises(promote_mod.PromoteError,
                       match="carries no sweepRun stamp"):
        promote_mod.promote("gl", "fear", root=root, log=lambda *_: None)


def test_promote_refuses_sweep_run_that_is_not_a_plain_name(tmp_path):
    """Path-containment (round 4, P2): the manifest-provided run name joins
    onto runs/ — a traversal component must refuse, never resolve."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gt")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    es.add_condition("gt", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": {**SELECTION_BLOCK, "sweepRun": "../../etc"}}, root)
    with pytest.raises(promote_mod.PromoteError,
                       match="not a plain run name"):
        promote_mod.promote("gt", "fear", root=root, log=lambda *_: None)


def test_promote_run_name_containment_is_universal(tmp_path):
    """Round 5, P1: EVERY path joining a caller-provided run name onto
    runs/ validates through the shared helper — the explicit sweep_run
    argument and the pinned contract, not just the manifest condition."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gu")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    with pytest.raises(promote_mod.PromoteError,
                       match="not a plain run name"):
        promote_mod.promote("gu", "fear", root=root, log=lambda *_: None,
                            sweep_run="../../etc")
    with pytest.raises(promote_mod.PromoteError,
                       match="not a plain run name"):
        promote_mod.promote("gu", "fear", root=root, log=lambda *_: None,
                            pins=promote_mod.PromotionPins(
                                sweep_run="../../etc"))


def test_promote_refuses_recommendation_not_naming_its_own_run(tmp_path):
    """Self-identity (round 5, P2): the recommendation read from directory
    run-A must stamp run-A — an entry claiming another run means a copied
    or edited directory, and a certificate naming run B from bytes read
    under run A must never mint. Both the condition path and the explicit
    sweep_run path refuse."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gsr")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    copied = "20260707T999999999-exp-copied-sweep"
    es.add_condition("gsr", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": {**SELECTION_BLOCK, "sweepRun": copied}}, root)
    # Directory `copied` holds an entry still stamped for the ORIGINAL run.
    _plant_completed_sweep_run(root, sweep_run=copied)
    with pytest.raises(promote_mod.PromoteError,
                       match="does not name itself"):
        promote_mod.promote("gsr", "fear", root=root, log=lambda *_: None)
    with pytest.raises(promote_mod.PromoteError,
                       match="does not name itself"):
        promote_mod.promote("gsr", "fear", root=root, log=lambda *_: None,
                            sweep_run=copied)


def test_ambient_fallback_refuses_recommendation_not_naming_its_run(tmp_path):
    """Round 6, P1: the AMBIENT newest-run fallback — the criterion path
    when a frozen manifest carries no projected condition, and the
    manual-override gate — must apply the same self-identity check as the
    condition/explicit/pinned paths. Refusal is loud, never a silent skip
    to an older run."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gaf")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    # No projected condition (the frozen-manifest shape); the newest ambient
    # run's entry stamps ANOTHER run.
    _sweep_evidence_run(
        root, "gaf", _pinned_entry(),
        entry_stamped_run="20260101T000000000-exp-other-sweep")
    with pytest.raises(promote_mod.PromoteError,
                       match="does not name itself"):
        promote_mod.promote("gaf", "fear", root=root, log=lambda *_: None)
    # The manual-override gate consumes the same ambient evidence.
    with pytest.raises(promote_mod.PromoteError,
                       match="does not name itself"):
        promote_mod.promote("gaf", "fear", root=root, cell=(5, 0.2),
                            override_reason="deliberate deviation",
                            log=lambda *_: None)


def test_promote_certificate_copies_from_the_run_not_the_condition(tmp_path):
    """Round 4, P1: same cell, drifted provenance — a condition claiming a
    different criterion/metrics than its run must not certify those claims.
    The birth certificate copies the RUN's recommendation, and the drift is
    logged loudly."""
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "gc")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    tampered = {**SELECTION_BLOCK,
                "metrics": {"markerDensity": 999.0},
                "devPromptsHash": "e" * 64}
    es.add_condition("gc", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": tampered}, root)
    _plant_completed_sweep_run(root)
    messages: list = []
    out = promote_mod.promote("gc", "fear", root=root, log=messages.append)
    promo = out["variant"]["promotion"]
    assert promo["metrics"] == SELECTION_BLOCK["metrics"]
    assert promo["devPromptsHash"] == SELECTION_BLOCK["devPromptsHash"]
    assert any("differs from sweep run" in m for m in messages)


def test_promote_mints_agent_with_birth_certificate(tmp_path):
    root = str(tmp_path)
    artifact_rel = _promotable(root)
    out = promote_mod.promote("pr", "fear", root=root, log=lambda *_: None)

    variant = out["variant"]
    assert variant["name"] == "pr-fear-agent"
    assert variant["baseModelID"] == "org/m"
    assert variant["alphaInNormUnits"] is True
    assert variant["injections"] == [{
        "concept": "fear", "vectorArtifactID": artifact_rel,
        "layer": 3, "alpha": 0.4}]
    promo = variant["promotion"]
    assert promo["promotedBy"] == "criterion"
    assert "overrideReason" not in promo
    assert promo["experiment"] == "pr"
    assert promo["winningCell"] == {"layer": 3, "alpha": 0.4}
    assert promo["criterion"] == SELECTION_BLOCK["criterion"]
    assert promo["metrics"] == SELECTION_BLOCK["metrics"]
    assert promo["substrate"] == SUBSTRATE
    assert promo["experimentHash"] == Manifest.load("pr", root).content_hash()
    # The certificate records WHICH full recipe the matched artifact
    # satisfied.
    manifest = Manifest.load("pr", root)
    assert promo["recipeIdentityHash"] == recipe_identity.identity_hash(
        recipe_identity.required_identity(manifest, manifest.concepts[0]))
    # Round-trips through the variant decoder and shows up in the library.
    loaded = model_variant.ModelVariant.from_file(out["path"])
    assert loaded.promotion == promo
    assert any(v["name"] == "pr-fear-agent"
               for v in model_variant.list_variants(root))


def test_promote_cell_override_is_loud_and_stamped(tmp_path):
    root = str(tmp_path)
    _promotable(root, name="ov")
    messages = []
    out = promote_mod.promote("ov", "fear", cell=(5, 0.2), root=root,
                              override_reason="confirming the adjacent layer",
                              log=messages.append)
    promo = out["variant"]["promotion"]
    assert promo["promotedBy"] == "manualOverride"
    assert promo["overrideReason"] == "confirming the adjacent layer"
    assert promo["winningCell"] == {"layer": 5, "alpha": 0.2}
    # The declared criterion still travels (it says what was deviated from)…
    assert promo["criterion"] == SELECTION_BLOCK["criterion"]
    # …but the winning cell's metrics do NOT describe the overridden cell.
    assert "metrics" not in promo
    assert any("manualOverride" in m for m in messages)


def _plant_sweep_run(root, run, recommendations):
    """A sweep run directory (sweep.csv + recommendations.json) under runs/ —
    the manual-override path's evidence source."""
    run_dir = os.path.join(root, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "sweep.csv"),
           "concept,layer,alpha,markerDensity,distinct2,words,batteryAccuracy\n"
           "fear,-1,0,0.01,0.8,50,1.0\n")
    _write(os.path.join(run_dir, "recommendations.json"),
           json.dumps(recommendations))
    return run


def test_promote_cell_override_without_any_sweep_refuses(tmp_path):
    # Overriding with NO sweep at all is hand-creation wearing a promotion
    # badge — it must refuse, unlike overriding after a sweep ran.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "nosweep")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    with pytest.raises(promote_mod.PromoteError,
                       match="no sweep has run for 'fear'"):
        promote_mod.promote("nosweep", "fear", cell=(5, 0.2), root=root,
                            log=lambda *_: None)


def test_promote_cell_override_after_failed_sweep_stamps_outcome(tmp_path):
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "failsweep")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    run = _plant_sweep_run(
        root, "20260707T000010000-exp-failsweep-sweep",
        {"fear": FAILED_SELECTION})
    out = promote_mod.promote("failsweep", "fear", cell=(5, 0.2), root=root,
                              override_reason="confirming despite failed gates",
                              log=lambda *_: None)
    promo = out["variant"]["promotion"]
    assert promo["promotedBy"] == "manualOverride"
    assert promo["sweepRun"] == run
    assert promo["selectionOutcome"] == FAILED_SELECTION
    assert "criterion" not in promo
    assert "metrics" not in promo


def test_promote_cell_override_copies_provenance_from_sweep_run(tmp_path):
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "runprov")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    run = "20260707T000011000-exp-runprov-sweep"
    _plant_sweep_run(root, run, {"fear": {**SELECTION_BLOCK, "sweepRun": run}})
    # Same cell as the sweep's winner (3, 0.4): the winner's metrics travel.
    same = promote_mod.promote("runprov", "fear", cell=(3, 0.4), root=root,
                               log=lambda *_: None)
    promo = same["variant"]["promotion"]
    assert promo["promotedBy"] == "manualOverride"
    assert promo["sweepRun"] == run
    assert promo["criterion"] == SELECTION_BLOCK["criterion"]
    assert promo["devPromptsHash"] == SELECTION_BLOCK["devPromptsHash"]
    assert promo["metrics"] == SELECTION_BLOCK["metrics"]
    assert "selectionOutcome" not in promo
    # A DIFFERENT cell: criterion/context travel, metrics do not.
    other = promote_mod.promote("runprov", "fear", cell=(5, 0.2), root=root,
                                agent_name="runprov-other", log=lambda *_: None)
    promo2 = other["variant"]["promotion"]
    assert promo2["sweepRun"] == run
    assert promo2["criterion"] == SELECTION_BLOCK["criterion"]
    assert "metrics" not in promo2
    assert "selectionOutcome" not in promo2


def test_promote_criterion_falls_back_to_sweep_run_evidence(tmp_path):
    # A sweep on a FROZEN manifest cannot stamp `<concept>-recommended` — its
    # selection lives only in the run's recommendations.json. Criterion
    # promotion (no cell) must fall back to that evidence and promote it as
    # promotedBy="criterion" with the full provenance copied. (Simulated the
    # way it manifests: no stamped condition, run evidence present — the
    # mechanism is the missing condition, not the status.)
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "frozenrec")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    run = "20260707T000012000-exp-frozenrec-sweep"
    control = {"type": "randomMatchedNorm", "metricValue": 0.05, "margin": 0.0}
    _plant_sweep_run(root, run, {"fear": {**SELECTION_BLOCK, "sweepRun": run,
                                          "control": control}})
    out = promote_mod.promote("frozenrec", "fear", root=root, log=lambda *_: None)
    variant = out["variant"]
    # The sweep writes norm-unit cells: run-evidence defaults.
    assert variant["injections"][0]["layer"] == 3
    assert variant["injections"][0]["alpha"] == 0.4
    assert variant["bandWidth"] == 1
    assert variant["alphaInNormUnits"] is True
    promo = variant["promotion"]
    assert promo["promotedBy"] == "criterion"
    assert "overrideReason" not in promo
    assert promo["sweepRun"] == run
    assert promo["criterion"] == SELECTION_BLOCK["criterion"]
    assert promo["devPromptsHash"] == SELECTION_BLOCK["devPromptsHash"]
    assert promo["winningCell"] == {"layer": 3, "alpha": 0.4}
    assert promo["metrics"] == SELECTION_BLOCK["metrics"]
    assert promo["control"] == control
    assert "selectionOutcome" not in promo


def test_promote_criterion_refuses_on_failed_sweep_evidence(tmp_path):
    # A failure entry is a sweep that RAN and selected nothing: criterion
    # promotion must refuse with the sweep's own conclusion — only a loud
    # manual override can promote past it.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "failcrit")
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    _plant_sweep_run(root, "20260707T000013000-exp-failcrit-sweep",
                     {"fear": FAILED_SELECTION})
    with pytest.raises(promote_mod.PromoteError) as excinfo:
        promote_mod.promote("failcrit", "fear", root=root, log=lambda *_: None)
    message = str(excinfo.value)
    assert "the sweep selected no cell for 'fear'" in message
    assert FAILED_SELECTION in message
    assert "manual override" in message


def test_promote_criterion_prefers_manifest_condition_over_run_evidence(tmp_path):
    # When BOTH a stamped manifest condition and newer run evidence exist, the
    # manifest condition wins — it is the declared, saved state.
    root = str(tmp_path)
    _promotable(root, name="prec")
    newer_run = "20260708T000000000-exp-prec-sweep"
    _plant_sweep_run(root, newer_run, {"fear": {
        **SELECTION_BLOCK, "sweepRun": newer_run,
        "winningCell": {"layer": 2, "alpha": 0.1},
        "metrics": {"markerDensity": 0.99}}})
    out = promote_mod.promote("prec", "fear", root=root, log=lambda *_: None)
    promo = out["variant"]["promotion"]
    assert promo["sweepRun"] == SELECTION_BLOCK["sweepRun"]
    assert promo["sweepRun"] != newer_run
    assert promo["winningCell"] == {"layer": 3, "alpha": 0.4}
    assert promo["metrics"] == SELECTION_BLOCK["metrics"]
    assert out["variant"]["injections"][0]["layer"] == 3
    assert out["variant"]["injections"][0]["alpha"] == 0.4


def test_promote_requires_a_sweep_recommendation(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "norec")
    with pytest.raises(promote_mod.PromoteError, match="run 'experiment sweep' first"):
        promote_mod.promote("norec", "fear", root=root, log=lambda *_: None)


def test_promote_requires_a_matching_artifact(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "noart")
    es.add_condition("noart", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    # An artifact exists but its stimulus hash does not match the recipe.
    _vector_artifact(root, stimulus_hash="0" * 64)
    with pytest.raises(promote_mod.PromoteError, match="no extraction artifact"):
        promote_mod.promote("noart", "fear", root=root, log=lambda *_: None)


def test_promote_skips_foreign_substrate_artifacts(tmp_path):
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "foreign")
    es.add_condition("foreign", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    _vector_artifact(root, stimulus_hash=stimulus_hash, substrate="swift-mlx")
    with pytest.raises(promote_mod.PromoteError, match="no extraction artifact"):
        promote_mod.promote("foreign", "fear", root=root, log=lambda *_: None)


def test_promote_refuses_legacy_artifact_naming_unprovable_fields(tmp_path):
    # A legacy artifact (six-field sidecar: no reading position, projection,
    # or norm source) can no longer be silently matched: promote refuses,
    # NAMES every recipe field the artifact cannot prove, and says how to fix
    # it — never a fallback to the old partial-field match.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "legacyart")
    es.add_condition("legacyart", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    _vector_artifact(root, stimulus_hash=stimulus_hash, reading_position=None,
                     neutral_projection=None, residual_norm_source=None)
    with pytest.raises(promote_mod.PromoteError) as excinfo:
        promote_mod.promote("legacyart", "fear", root=root, log=lambda *_: None)
    message = str(excinfo.value)
    assert "cannot prove recipe fields" in message
    assert "[neutralProjection, readingPosition, residualNormSource]" in message
    assert "re-extract to stamp the full recipe" in message


def test_promote_refusal_names_the_differing_fields_per_candidate(tmp_path):
    # A near-miss candidate must be actionable: the refusal compares the
    # canonical identity components and NAMES every differing field with both
    # values — "revision (manifest: …, artifact: …)" — never just "a
    # DIFFERENT recipe identity".
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "fdiff", revision=None)
    es.add_condition("fdiff", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    # Candidate 1: the user's exact shape — manifest revision null, artifact
    # stamped with the resolved commit.
    one = _vector_artifact(root, stimulus_hash=stimulus_hash,
                           revision=RESOLVED_REVISION,
                           run="20260707T000005000-extract")
    # Candidate 2: right revision (null), divergent projection recipe.
    two = _vector_artifact(root, stimulus_hash=stimulus_hash, revision=None,
                           neutral_projection="legacy-pooled top-3 neutral PCs",
                           run="20260707T000006000-extract")
    with pytest.raises(promote_mod.PromoteError) as excinfo:
        promote_mod.promote("fdiff", "fear", root=root, log=lambda *_: None)
    message = str(excinfo.value)
    assert f"candidate '{one}'" in message
    assert "revision (manifest: null, artifact: 8f2a00000000…)" in message
    assert f"candidate '{two}'" in message
    assert "neutralProjection.mode (manifest: none, artifact: legacyPooled)" \
        in message
    assert "neutralProjection.count (manifest: null, artifact: 3)" in message
    # The old bare counter phrasing is gone.
    assert "candidate artifact(s) carry a DIFFERENT recipe identity" \
        not in message


def test_promote_refusal_diffs_a_stamped_but_mismatched_candidate(tmp_path):
    # Path (a) candidates (stamped recipeIdentityHash) still get a field
    # diff, computed from their recorded sidecar fields.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "fstamp")  # revision "abc"
    es.add_condition("fstamp", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    sidecar_components, _ = recipe_identity.candidate_identity({
        "modelID": "org/m", "concept": "fear", "stimulusSetHash": stimulus_hash,
        "revision": "def456789012345678", "extractionMethod": "meanDifference",
        "readingPosition": "last token", "neutralProjection": "none",
        "residualNormSource": "extraction-stimuli"})
    stamped = _vector_artifact(
        root, stimulus_hash=stimulus_hash, revision="def456789012345678",
        run="20260707T000007000-extract",
        extras={"recipeIdentityHash":
                recipe_identity.identity_hash(sidecar_components)})
    with pytest.raises(promote_mod.PromoteError) as excinfo:
        promote_mod.promote("fstamp", "fear", root=root, log=lambda *_: None)
    message = str(excinfo.value)
    assert f"candidate '{stamped}'" in message
    assert "revision (manifest: abc, artifact: def456789012…)" in message


def test_promote_never_lets_a_newer_different_recipe_win(tmp_path):
    # Newest-wins must never break a tie between DIFFERENT recipes: newer
    # artifacts whose reading position / projection / norm denominator differ
    # are excluded, and the older exact-identity match wins.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "tiebreak")
    es.add_condition("tiebreak", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    matching = _vector_artifact(root, stimulus_hash=stimulus_hash,
                                run="20260707T000001000-extract")
    _vector_artifact(root, stimulus_hash=stimulus_hash,
                     run="20260707T000002000-extract",
                     reading_position="mean from token 50")
    _vector_artifact(root, stimulus_hash=stimulus_hash,
                     run="20260707T000003000-extract",
                     neutral_projection="legacy-pooled top-3 neutral PCs")
    _vector_artifact(root, stimulus_hash=stimulus_hash,
                     run="20260707T000004000-extract",
                     residual_norm_source="neutral-corpus",
                     extras={"neutralCorpusHash": "0" * 64})
    out = promote_mod.promote("tiebreak", "fear", root=root, log=lambda *_: None)
    assert out["variant"]["injections"][0]["vectorArtifactID"] == matching


def test_promote_matches_stamped_recipe_identity_hash(tmp_path):
    # Path (a): an artifact stamped with the exact required
    # recipeIdentityHash matches on the stamp — the normal case for every
    # artifact the extraction writers produce going forward. A WRONG stamp is
    # a different identity, never a fallback candidate.
    root = str(tmp_path)
    stimulus_hash = _experiment_with_concept(root, "stamped")
    es.add_condition("stamped", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": SELECTION_BLOCK}, root)
    _plant_completed_sweep_run(root)
    manifest = Manifest.load("stamped", root)
    required_hash = recipe_identity.identity_hash(
        recipe_identity.required_identity(manifest, manifest.concepts[0]))
    # Deliberately sparse otherwise (legacy-shaped): the stamp alone carries
    # the proof.
    artifact_rel = _vector_artifact(
        root, stimulus_hash=stimulus_hash, reading_position=None,
        neutral_projection=None, residual_norm_source=None,
        extras={"recipeIdentityHash": required_hash})
    _vector_artifact(
        root, stimulus_hash=stimulus_hash, run="20260707T000009000-extract",
        reading_position=None, neutral_projection=None,
        residual_norm_source=None,
        extras={"recipeIdentityHash": "e" * 64})
    out = promote_mod.promote("stamped", "fear", root=root, log=lambda *_: None)
    assert out["variant"]["injections"][0]["vectorArtifactID"] == artifact_rel
    assert out["variant"]["promotion"]["recipeIdentityHash"] == required_hash


def test_promote_grand_mean_matches_on_the_full_population(tmp_path):
    # Grand-mean promotion requires the artifact to prove the FULL comparison
    # population (every member + stories hash): no stamp refuses naming the
    # field; a drifted member refuses as a different recipe; the exact pinned
    # population promotes — including across the Swift/server norm-source
    # label unification ("multi-concept-corpus" ≡ "extraction-stimuli").
    root = str(tmp_path)
    es.create("gm", model_id="org/m", revision="abc", root=root)
    population = {"fear": "f" * 64, "joy": "a" * 64, "anger": "b" * 64}
    d = es.load_raw("gm", root)
    d["concepts"] = [{"name": "fear", "stimulusSetHash": "f" * 64,
                      "options": {"method": "emotionGrandMean",
                                  "readingPosition": "last token"}}]
    d["grandMeanCorpus"] = {"concepts": ["fear", "joy", "anger"],
                            "hashes": population}
    d["conditions"] = [{
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.4}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": {**SELECTION_BLOCK,
                      "winningCell": {"layer": 3, "alpha": 0.4}}}]
    es.save_raw(d, root)
    _plant_completed_sweep_run(root)

    # No population stamp: unprovable, named.
    _vector_artifact(root, stimulus_hash="f" * 64, method="emotionGrandMean",
                     residual_norm_source="multi-concept-corpus")
    with pytest.raises(promote_mod.PromoteError,
                       match=r"\[grandMeanPopulation\]"):
        promote_mod.promote("gm", "fear", root=root, log=lambda *_: None)

    # A DIFFERENT population (one member's stories drifted) refuses.
    _vector_artifact(root, stimulus_hash="f" * 64, method="emotionGrandMean",
                     run="20260707T000002000-extract",
                     residual_norm_source="multi-concept-corpus",
                     extras={"grandMeanPopulation": {**population,
                                                     "joy": "c" * 64}})
    with pytest.raises(promote_mod.PromoteError, match="DIFFERENT"):
        promote_mod.promote("gm", "fear", root=root, log=lambda *_: None)

    # The full pinned population matches.
    artifact_rel = _vector_artifact(
        root, stimulus_hash="f" * 64, method="emotionGrandMean",
        run="20260707T000003000-extract",
        residual_norm_source="multi-concept-corpus",
        extras={"grandMeanPopulation": population})
    out = promote_mod.promote("gm", "fear", root=root, log=lambda *_: None)
    assert out["variant"]["injections"][0]["vectorArtifactID"] == artifact_rel


def test_promote_route(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _promotable(root, name="rt")

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    response = client.post("/api/experiment/rt/promote", json={"concept": "fear"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["variant"]["promotion"]["promotedBy"] == "criterion"

    missing = client.post("/api/experiment/rt/promote", json={})
    assert missing.status_code == 400

    norec = client.post("/api/experiment/rt/promote",
                        json={"concept": "unattached"})
    assert norec.status_code == 400


# --- the PINNED promotion contract (B2, 2026-07-26) --------------------------

def _sweep_evidence_run(root, experiment, entry, *, run=None,
                        epoch_hash=None, entry_stamped_run=None):
    """A sweep run directory the pinned contract can name: sweep.csv (the
    discovery marker), recommendations.json, and the manifest-epoch stamp."""
    run = run or f"20260726T000000000-exp-{experiment}-sweep"
    run_dir = os.path.join(root, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "sweep.csv"), "concept,layer,alpha\n")
    # The real sweep writer stamps its own basename; the self-identity gate
    # (round 5, P2) verifies it, so the fixture mirrors it —
    # `entry_stamped_run` forges an entry claiming ANOTHER run, the gate's
    # test lever.
    if isinstance(entry, dict):
        entry = {**entry, "sweepRun": entry_stamped_run or run}
    _write(os.path.join(run_dir, "recommendations.json"),
           json.dumps({"fear": entry}))
    if epoch_hash is None:
        epoch_hash = Manifest.load(experiment, root).content_hash()
    if epoch_hash is not False:
        _write(os.path.join(run_dir, "experiment-hash.txt"), epoch_hash)
    return run


def _pinned_entry(layer=3, alpha=0.4, run=None):
    return {**SELECTION_BLOCK,
            "sweepRun": run or "20260726T000000000-exp-pin-sweep",
            "winningCell": {"layer": layer, "alpha": alpha}}


def test_pinned_promotion_uses_only_the_named_run(tmp_path):
    """The ambient "newest sweep run" lookup must not be consulted at all."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry(layer=3, alpha=0.4))
    # A NEWER sweep with a different winner. Under the old recency rule this
    # would win; under the pinned contract it is never read.
    _sweep_evidence_run(
        root, "pin", _pinned_entry(layer=9, alpha=0.9),
        run="20260726T999999999-exp-pin-sweep")

    out = promote_mod.promote(
        "pin", "fear", root=root, log=lambda *_: None,
        pins=promote_mod.PromotionPins(sweep_run=named))

    promo = out["variant"]["promotion"]
    assert promo["winningCell"] == {"layer": 3, "alpha": 0.4}
    assert promo["promotedBy"] == "criterion"


def test_pinned_promotion_refuses_a_stale_expected_cell(tmp_path):
    """A pinned cell that disagrees with the sweep is a STALE PLAN — refused,
    never silently reinterpreted as a deliberate manual override."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry(layer=3, alpha=0.4))

    with pytest.raises(promote_mod.PromoteError, match="stale"):
        promote_mod.promote(
            "pin", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(
                sweep_run=named, winning_cell=(9, 0.9)))

    # The agreeing cell promotes normally.
    out = promote_mod.promote(
        "pin", "fear", root=root, log=lambda *_: None,
        pins=promote_mod.PromotionPins(
            sweep_run=named, winning_cell=(3, 0.4)))
    assert out["variant"]["promotion"]["winningCell"] == {"layer": 3, "alpha": 0.4}


def test_pinned_promotion_refuses_a_foreign_manifest_epoch(tmp_path):
    """A sweep run from a different epoch selected its cell under different
    settings — an agent minted from it would carry a false birth certificate."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(
        root, "pin", _pinned_entry(), epoch_hash="0" * 64)

    with pytest.raises(promote_mod.PromoteError, match="different manifest epoch"):
        promote_mod.promote(
            "pin", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(sweep_run=named))


def test_pinned_promotion_refuses_an_unstamped_run(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry(), epoch_hash=False)

    with pytest.raises(promote_mod.PromoteError, match="no experiment-hash stamp"):
        promote_mod.promote(
            "pin", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(sweep_run=named))


def test_pinned_promotion_refuses_a_drifted_expected_epoch(tmp_path):
    """The caller's own declared epoch is checked too: a plan made against an
    older manifest must not execute against the current one."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry())

    with pytest.raises(promote_mod.PromoteError, match="manifest changed"):
        promote_mod.promote(
            "pin", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(
                sweep_run=named, experiment_hash="0" * 64))


def test_pinned_artifact_is_selected_by_identity_not_recency(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    older = _vector_artifact(root, stimulus_hash=stimulus_hash,
                             run="20260707T000001000-extract")
    _vector_artifact(root, stimulus_hash=stimulus_hash,
                     run="20260726T000001000-extract")
    named = _sweep_evidence_run(root, "pin", _pinned_entry())

    # Unpinned: the newest interchangeable artifact wins (unchanged rule).
    unpinned = promote_mod.promote(
        "pin", "fear", root=root, log=lambda *_: None,
        pins=promote_mod.PromotionPins(sweep_run=named))
    assert "20260726T000001000-extract" in \
        unpinned["variant"]["injections"][0]["vectorArtifactID"]

    # Pinned: the NAMED artifact is used even though a newer one exists.
    pinned = promote_mod.promote(
        "pin", "fear", root=root, log=lambda *_: None, agent_name="older-agent",
        pins=promote_mod.PromotionPins(
            sweep_run=named, vector_artifact_id=older))
    assert pinned["variant"]["injections"][0]["vectorArtifactID"] == older


def test_pinned_artifact_that_does_not_match_the_recipe_refuses(tmp_path):
    """Pinning chooses WHICH artifact; it never waives WHETHER it matches."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    # A same-concept artifact extracted under a DIFFERENT method.
    foreign = _vector_artifact(root, stimulus_hash=stimulus_hash,
                               run="20260726T000002000-extract",
                               method="lat")
    named = _sweep_evidence_run(root, "pin", _pinned_entry())

    with pytest.raises(promote_mod.PromoteError,
                       match="no artifact at that path matches"):
        promote_mod.promote(
            "pin", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(
                sweep_run=named, vector_artifact_id=foreign))


def test_pinned_artifact_hash_mismatch_refuses(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry())

    with pytest.raises(promote_mod.PromoteError, match="re-extracted since"):
        promote_mod.promote(
            "pin", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(
                sweep_run=named, vector_artifact_hash="0" * 64))


def test_promotion_is_idempotent(tmp_path):
    """A retried stage must return the existing agent, not mint a competing
    duplicate that is then indistinguishable in every picker."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry())
    pins = promote_mod.PromotionPins(sweep_run=named)

    first = promote_mod.promote("pin", "fear", root=root,
                                log=lambda *_: None, pins=pins)
    second = promote_mod.promote("pin", "fear", root=root,
                                 log=lambda *_: None, pins=pins)

    assert second.get("idempotentReuse") is True
    assert second["path"] == first["path"]
    assert len(model_variant.list_variants(root)) == 1
    # The key is recorded on the artifact, and is stable across the retry.
    key = first["variant"]["promotion"]["promotionKey"]
    assert second["variant"]["promotion"]["promotionKey"] == key


def test_a_different_cell_is_a_different_promotion(tmp_path):
    """Idempotency must key on the REQUEST, not merely on the concept — an
    override to another cell is a genuinely different agent."""
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry())
    pins = promote_mod.PromotionPins(sweep_run=named)

    promote_mod.promote("pin", "fear", root=root, log=lambda *_: None,
                        pins=pins)
    other = promote_mod.promote(
        "pin", "fear", root=root, log=lambda *_: None, pins=pins,
        agent_name="override-agent", cell=(7, 0.7),
        override_reason="probing a stronger cell")

    assert other.get("idempotentReuse") is not True
    assert other["variant"]["promotion"]["promotedBy"] == "manualOverride"
    assert len(model_variant.list_variants(root)) == 2


def test_birth_certificate_records_which_bytes_were_injected(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "pin")
    stimulus_hash = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "pin", _pinned_entry())

    out = promote_mod.promote("pin", "fear", root=root, log=lambda *_: None,
                              pins=promote_mod.PromotionPins(sweep_run=named))
    # sha256 of the fixture's tensor bytes (b"weights").
    assert out["variant"]["promotion"]["vectorArtifactHash"] == \
        hashlib.sha256(b"weights").hexdigest()


def test_pinned_promote_route(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _experiment_with_concept(root, "rtp")
    stimulus_hash = Manifest.load("rtp", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "rtp", _pinned_entry())

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    ok = client.post("/api/experiment/rtp/promote",
                     json={"concept": "fear",
                           "pins": {"sweepRun": named,
                                    "winningCell": {"layer": 3, "alpha": 0.4}}})
    assert ok.status_code == 200, ok.text
    assert ok.json()["variant"]["promotion"]["promotionKey"]

    # Retrying the identical request is idempotent over the wire too.
    again = client.post("/api/experiment/rtp/promote",
                        json={"concept": "fear",
                              "pins": {"sweepRun": named,
                                       "winningCell": {"layer": 3, "alpha": 0.4}}})
    assert again.status_code == 200, again.text
    assert again.json()["idempotentReuse"] is True

    stale = client.post("/api/experiment/rtp/promote",
                        json={"concept": "fear",
                              "pins": {"sweepRun": named,
                                       "winningCell": {"layer": 9, "alpha": 0.9}}})
    assert stale.status_code == 400
    assert "stale" in stale.json()["detail"]

    malformed = client.post("/api/experiment/rtp/promote",
                            json={"concept": "fear", "pins": {}})
    assert malformed.status_code == 400


# --- E3: the refusal must say WHICH gate refused -----------------------------

def test_no_selection_reason_distinguishes_constraints_from_the_objective():
    """The old message always said "no cell passed the capability/coherence
    gates". That is one of two possible reasons and often the wrong one.

    Observed live 2026-07-26: a practicalwisdom sweep where all 36 cells sat
    inside both constraints and every objective value was NEGATIVE — the
    vector moved the objective the opposite way — yet the run recorded a gate
    failure, sending the researcher to loosen a tolerance that was never
    binding."""
    base = sel.BaselineCell(metric=0.0, distinct2=0.99, battery_accuracy=0.9)
    criterion = sel.resolve_selection({"objective": {"metric": "logprobShift"}})

    # All eligible, none beats baseline — the practicalwisdom shape.
    eligible_but_negative = [
        sel.SweepCell(layer=31, alpha=0.05, metric=-0.357,
                      distinct2=0.98, battery_accuracy=0.9),
        sel.SweepCell(layer=41, alpha=0.17, metric=-2.60,
                      distinct2=0.97, battery_accuracy=0.9)]
    assert sel.select_cell(eligible_but_negative, base, criterion) is None
    reason = sel.no_selection_reason(eligible_but_negative, base, criterion)
    assert "beat the baseline" in reason
    assert "OPPOSITE way" in reason
    assert "inside both constraints" in reason
    assert "capability/coherence gates" not in reason

    # Genuinely gate-blocked: the old message is correct here, and kept.
    gated = [sel.SweepCell(layer=31, alpha=0.05, metric=5.0,
                           distinct2=0.01, battery_accuracy=0.1)]
    gated_reason = sel.no_selection_reason(gated, base, criterion)
    assert "capability/coherence gates" in gated_reason

    # Mixed: some blocked, and the survivors still lose on the objective.
    mixed = eligible_but_negative + gated
    mixed_reason = sel.no_selection_reason(mixed, base, criterion)
    assert "beat the baseline" in mixed_reason
    assert "1 of 3 cells also failed" in mixed_reason


def test_no_selection_reason_handles_an_empty_grid():
    base = sel.BaselineCell(metric=0.0, distinct2=0.99, battery_accuracy=0.9)
    criterion = sel.resolve_selection(None)
    assert "measured no cells" in sel.no_selection_reason([], base, criterion)


def test_a_pinned_hash_with_an_unreadable_artifact_refuses(tmp_path, monkeypatch):
    """Comparing only when BOTH hashes exist failed open: an unreadable
    .safetensors let the promotion proceed and mint an agent whose claimed
    vector bytes were never verified — the one thing the pin exists to
    prevent."""
    root = str(tmp_path)
    _experiment_with_concept(root, "fo")
    stimulus_hash = Manifest.load("fo", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "fo", _pinned_entry())
    # The recipe still matches; only the tensor bytes are unreadable.
    monkeypatch.setattr(promote_mod, "_artifact_content_hash", lambda a: None)

    with pytest.raises(promote_mod.PromoteError, match="could not be read"):
        promote_mod.promote(
            "fo", "fear", root=root, log=lambda *_: None,
            pins=promote_mod.PromotionPins(
                sweep_run=named, vector_artifact_hash="0" * 64))

    # With no expected hash there is nothing to verify, so it still promotes.
    out = promote_mod.promote(
        "fo", "fear", root=root, log=lambda *_: None,
        pins=promote_mod.PromotionPins(sweep_run=named))
    assert out["variant"]["promotion"]["promotedBy"] == "criterion"


def test_the_promote_route_accepts_the_full_pin_set(tmp_path, monkeypatch):
    """The wire shape the Swift client now sends."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _experiment_with_concept(root, "wire")
    stimulus_hash = Manifest.load("wire", root).concepts[0].stimulus_set_hash
    artifact = _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "wire", _pinned_entry())

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    response = client.post(
        "/api/experiment/wire/promote",
        json={"concept": "fear",
              "pins": {"sweepRun": named,
                       "experimentHash": Manifest.load("wire", root).content_hash(),
                       "winningCell": {"layer": 3, "alpha": 0.4},
                       "vectorArtifactID": artifact,
                       "vectorArtifactHash": hashlib.sha256(b"weights").hexdigest()}})
    assert response.status_code == 200, response.text
    promotion = response.json()["variant"]["promotion"]
    assert promotion["promotedBy"] == "criterion"
    assert promotion["winningCell"] == {"layer": 3, "alpha": 0.4}
    # The coherence-evidence length travels into the certificate with the
    # distinct2 it contextualizes (c18 lesson): a certificate carrying a
    # coherence number without its measurement length re-opens the trap.
    assert promotion["devMaxTokens"] == 1024


def test_the_idempotency_key_binds_the_vector_bytes(tmp_path):
    """Excluding the bytes let a retry validate NEW bytes at the same path and
    then return an OLDER agent whose birth certificate names the old hash — an
    artifact asserting provenance it no longer has."""
    key_args = dict(
        experiment="e", experiment_hash="a" * 64, concept="fear",
        sweep_run="run-1", layer=41, alpha=0.1,
        vector_artifact_id="runs/x/fear", promoted_by="criterion",
        agent_name="agent")
    original = promote_mod.promotion_key(**key_args, vector_artifact_hash="a" * 64)
    reextracted = promote_mod.promotion_key(**key_args, vector_artifact_hash="b" * 64)
    assert original != reextracted
    # Byte-identical re-extraction stays idempotent.
    assert original == promote_mod.promotion_key(
        **key_args, vector_artifact_hash="a" * 64)


def test_changed_vector_bytes_mint_a_new_agent_rather_than_reusing(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "bytes")
    stimulus_hash = Manifest.load("bytes", root).concepts[0].stimulus_set_hash
    _vector_artifact(root, stimulus_hash=stimulus_hash)
    named = _sweep_evidence_run(root, "bytes", _pinned_entry())
    pins = promote_mod.PromotionPins(sweep_run=named)

    first = promote_mod.promote("bytes", "fear", root=root,
                                log=lambda *_: None, pins=pins)
    assert promote_mod.promote(
        "bytes", "fear", root=root, log=lambda *_: None,
        pins=pins).get("idempotentReuse") is True

    # Same path, different bytes: a DIFFERENT promotion, not a reuse of the
    # agent whose certificate names the old hash.
    import glob
    target = glob.glob(os.path.join(root, "runs", "*extract*", "fear.safetensors"))[0]
    with open(target, "wb") as handle:
        handle.write(b"different weights")
    again = promote_mod.promote("bytes", "fear", root=root,
                                log=lambda *_: None, pins=pins)
    assert again.get("idempotentReuse") is not True
    assert again["variant"]["promotion"]["vectorArtifactHash"] != \
        first["variant"]["promotion"]["vectorArtifactHash"]
