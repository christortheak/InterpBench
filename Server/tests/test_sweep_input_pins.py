"""Sweep-input hash pins (firewall closure, 2026-07-20): the sweep's dev
prompts and capability battery decide which cell wins, so they enter the same
drift firewall as markers — freeze pins them (cross-engine contract keys
``sweep.devPromptsHash`` + ``sweep.batteryHash``), verify() reports drift as a
VIOLATION, sweep start refuses to select on drifted inputs, and legacy
manifests without the keys stay byte-identical and verify clean. Second pass
(same day): freeze REFUSES — force included — when an operative sweep's input
file is missing (an absent pin would leave sweep start's legacy-unpinned
fallback open to later-created bytes); carried-inert sweeps neither pin nor
block. The ex-post provenance stamp (``selection.devPromptsHash``) is kept —
the start refusal is what makes pin and provenance agree."""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors

DEV_ROWS = '{"text": "Write about the town."}\n'
BATTERY_ROWS = '{"prompt": "What is 1+1?", "answer": "2"}\n'


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _concept(root, name="fear"):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    _write(os.path.join(d, "markers.json"), json.dumps({"words": ["dread"]}))


def _sweep_study(root, name, *, dev="prompts/dev/dev.jsonl",
                 battery="prompts/batteries/b.jsonl", declare_paths=True):
    """A draft concept study with a declared sweep block; returns the two
    input hashes."""
    _concept(root)
    dev_hash = _write(os.path.join(root, dev), DEV_ROWS)
    battery_hash = _write(os.path.join(root, battery), BATTERY_ROWS)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    d = es.load_raw(name, root)
    sweep = {"layerFractions": [0.5], "alphas": [0.4], "maxTokens": 16}
    if declare_paths:
        sweep["devPromptsFile"] = dev
        sweep["batteryFile"] = battery
    d["sweep"] = sweep
    es.save_raw(d, root)
    return dev_hash, battery_hash


def _choice_study(root, name, *, mapped=False):
    """A sweep study whose selection declares choice instruments; returns
    the per-file hashes keyed as the manifest declares them."""
    _sweep_study(root, name)
    if mapped:
        _concept(root, "hope")
        es.attach(name, ["hope"], root=root)
    d = es.load_raw(name, root)
    hashes = {}
    if mapped:
        files = {}
        for concept in ("fear", "hope"):
            rel = f"prompts/dev/{concept}-choices.jsonl"
            hashes[concept] = _write(
                os.path.join(root, rel),
                f'{{"id": "{concept}-1", "prompt": "p", '
                '"options": ["A", "B"]}\n')
            files[concept] = rel
        selection = {"objective": {"metric": "logprobShift",
                                   "choicePromptsFiles": files}}
    else:
        rel = "prompts/dev/choices.jsonl"
        hashes[None] = _write(
            os.path.join(root, rel),
            '{"id": "c-1", "prompt": "p", "options": ["A", "B"]}\n')
        selection = {"objective": {"metric": "logprobShift",
                                   "choicePromptsFile": rel}}
    d["sweep"]["selection"] = selection
    es.save_raw(d, root)
    return hashes


# --- freeze pins -------------------------------------------------------------

def test_freeze_pins_choice_instruments_and_verify_flags_drift(tmp_path):
    """Review 2026-08-02 (P1): the files that determine the WINNING CELL
    were the one sweep input not pinned at freeze — a choice file could
    change between freeze and execution with no manifest drift."""
    root = str(tmp_path)
    hashes = _choice_study(root, "cs")
    frozen = es.freeze("cs", force=True, root=root)
    objective = frozen["sweep"]["selection"]["objective"]
    assert objective["choicePromptsHash"] == hashes[None]
    assert Manifest.load("cs", root=root).verify(root) == []
    # Drift after freeze is a verify violation.
    _write(os.path.join(root, "prompts", "dev", "choices.jsonl"),
           '{"id": "c-1", "prompt": "EDITED", "options": ["A", "B"]}\n')
    assert any("sweep choice prompts" in v and "changed since pinning" in v
               for v in Manifest.load("cs", root=root).verify(root))


def test_freeze_pins_the_per_concept_choice_map(tmp_path):
    root = str(tmp_path)
    hashes = _choice_study(root, "cm", mapped=True)
    frozen = es.freeze("cm", force=True, root=root)
    objective = frozen["sweep"]["selection"]["objective"]
    assert objective["choicePromptsHashes"] == hashes
    assert Manifest.load("cm", root=root).verify(root) == []
    _write(os.path.join(root, "prompts", "dev", "hope-choices.jsonl"),
           '{"id": "hope-1", "prompt": "EDITED", "options": ["A", "B"]}\n')
    assert any("sweep choice prompts 'hope'" in v
               and "changed since pinning" in v
               for v in Manifest.load("cm", root=root).verify(root))


def test_freeze_refuses_when_a_choice_instrument_is_missing(tmp_path):
    root = str(tmp_path)
    _choice_study(root, "cx", mapped=True)
    os.remove(os.path.join(root, "prompts", "dev", "hope-choices.jsonl"))
    with pytest.raises(es.ExperimentStoreError,
                       match="sweep choice prompts 'hope'"):
        es.freeze("cx", force=True, root=root)


def test_inert_choice_fields_neither_pin_nor_block(tmp_path):
    """Review 2026-08-02 round 2 (P2): only logprobShift READS choice
    instruments, so a stale path carried under markerDensity must not
    block freezing (or packaging) over a file nothing reads."""
    root = str(tmp_path)
    _choice_study(root, "inert")
    d = es.load_raw("inert", root)
    d["sweep"]["selection"]["objective"]["metric"] = "markerDensity"
    es.save_raw(d, root)
    os.remove(os.path.join(root, "prompts", "dev", "choices.jsonl"))
    frozen = es.freeze("inert", force=True, root=root)
    objective = frozen["sweep"]["selection"]["objective"]
    assert "choicePromptsHash" not in objective
    assert Manifest.load("inert", root=root).verify(root) == []
    labels = [e.label for e in es.pinned_input_entries(frozen, root)]
    assert not any("choice prompts" in label for label in labels)


def test_freeze_pins_declared_sweep_inputs(tmp_path):
    root = str(tmp_path)
    dev_hash, battery_hash = _sweep_study(root, "s")
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["sweep"]["devPromptsHash"] == dev_hash
    assert frozen["sweep"]["batteryHash"] == battery_hash
    assert Manifest.load("s", root=root).verify(root) == []


def test_freeze_pins_default_paths_when_spec_omits_them(tmp_path):
    root = str(tmp_path)
    dev_hash, battery_hash = _sweep_study(
        root, "s", dev="prompts/dev/dev-prompts.jsonl",
        battery="prompts/batteries/basic.jsonl", declare_paths=False)
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["sweep"]["devPromptsHash"] == dev_hash
    assert frozen["sweep"]["batteryHash"] == battery_hash


def test_freeze_never_silently_repins(tmp_path):
    root = str(tmp_path)
    _sweep_study(root, "s")
    d = es.load_raw("s", root)
    d["sweep"]["devPromptsHash"] = "00" * 32  # pre-existing (drifted) pin
    es.save_raw(d, root)
    # Freeze must not repair the pin; the drift is a verify violation.
    with pytest.raises(es.ExperimentStoreError,
                       match="sweep dev prompts .* changed since pinning"):
        es.freeze("s", force=True, root=root)


def test_freeze_refuses_when_a_sweep_input_file_is_missing(tmp_path):
    # Firewall closure second pass (2026-07-20): a missing sweep input used
    # to pin NOTHING at freeze, and the file could then be created AFTER
    # freeze — sweep start's legacy-unpinned fallback would accept the
    # unpinned bytes. Freeze now refuses instead — and the refusal is
    # pin-surface integrity, NOT a force-skippable evidence gate (force=True
    # here must still refuse).
    root = str(tmp_path)
    _sweep_study(root, "s")
    os.remove(os.path.join(root, "prompts", "batteries", "b.jsonl"))
    with pytest.raises(
            es.ExperimentStoreError,
            match=r"sweep capability battery file "
                  r"'prompts/batteries/b\.jsonl' is missing"):
        es.freeze("s", force=True, root=root)
    # Still a draft, still unpinned — nothing was half-frozen.
    d = es.load_raw("s", root)
    assert d.get("status") == "draft"
    assert "batteryHash" not in d["sweep"]
    # The remedy is named in plain language.
    with pytest.raises(es.ExperimentStoreError,
                       match=r"remove/repoint the sweep's batteryFile"):
        es.freeze("s", force=True, root=root)
    # No sweep block at all → no keys anywhere, and no refusal.
    _concept(root, "calm")
    es.create("bare", model_id="org/m", revision="abc", root=root)
    es.attach("bare", ["calm"], root=root)
    bare = es.freeze("bare", force=True, root=root)
    assert "sweep" not in bare


def test_carried_inert_sweep_neither_pins_nor_blocks_freeze(tmp_path):
    # Operative-surface carve-out: a declared agentComparison study carries
    # its sweep INERT — a missing sweep input must neither refuse the freeze
    # nor gain a pin.
    root = str(tmp_path)
    _sweep_study(root, "s")
    d = es.load_raw("s", root)
    d["studyType"] = "agentComparison"
    es.save_raw(d, root)
    os.remove(os.path.join(root, "prompts", "batteries", "b.jsonl"))
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["status"] == "frozen"
    assert "devPromptsHash" not in frozen["sweep"]
    assert "batteryHash" not in frozen["sweep"]
    assert Manifest.load("s", root=root).verify(root) == []


# --- verify(): drift is a violation, legacy is clean --------------------------

def test_verify_flags_drift_and_missing_after_pinning(tmp_path):
    root = str(tmp_path)
    _sweep_study(root, "s")
    es.freeze("s", force=True, root=root)
    dev = os.path.join(root, "prompts", "dev", "dev.jsonl")
    _write(dev, '{"text": "EDITED"}\n')
    violations = Manifest.load("s", root=root).verify(root)
    assert any("sweep dev prompts 'prompts/dev/dev.jsonl' changed since "
               "pinning" in v for v in violations)
    os.remove(dev)
    violations = Manifest.load("s", root=root).verify(root)
    assert any("sweep dev prompts" in v and "file missing" in v
               for v in violations)
    battery = os.path.join(root, "prompts", "batteries", "b.jsonl")
    _write(battery, '{"prompt": "EDITED", "answer": "x"}\n')
    violations = Manifest.load("s", root=root).verify(root)
    assert any("sweep capability battery 'prompts/batteries/b.jsonl' "
               "changed since pinning" in v for v in violations)


def test_legacy_manifest_without_keys_verifies_clean_and_is_unchanged(tmp_path):
    root = str(tmp_path)
    _sweep_study(root, "legacy")
    d = es.load_raw("legacy", root)
    assert "devPromptsHash" not in d["sweep"]
    assert "batteryHash" not in d["sweep"]
    before = Manifest.from_dict(d).content_hash()
    assert Manifest.load("legacy", root=root).verify(root) == []
    # Drift in the (unpinned) files is invisible to verify — the legacy
    # state exactly; only freeze introduces the pins.
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "EDITED"}\n')
    assert Manifest.load("legacy", root=root).verify(root) == []
    assert Manifest.from_dict(es.load_raw("legacy", root)).content_hash() \
        == before


def test_carried_sweep_in_agent_comparison_is_inert(tmp_path):
    # Operative-surface rule: a declared agentComparison carries concept
    # machinery (sweep included) inert — a pinned-but-drifted sweep input
    # must not block its verification.
    root = str(tmp_path)
    _sweep_study(root, "s")
    d = es.load_raw("s", root)
    d["sweep"]["devPromptsHash"] = "00" * 32
    d["studyType"] = "agentComparison"
    d["variantConditions"] = [
        {"name": "v", "artifactPath": "x.json", "artifactHash": "aa",
         "artifact": {"baseModelID": "org/m"}}]
    es.save_raw(d, root)
    violations = Manifest.load("s", root=root).verify(root)
    assert not any("sweep dev prompts" in v for v in violations)


# --- sweep start refuses on drift, and provenance agrees ----------------------

@contextmanager
def _fake_model(model_id, revision):
    yield SimpleNamespace(revision=revision)


def _armed_sweep(root, name, monkeypatch):
    """Same fakes as test_sweep_objectives: no GPU, no network."""
    bundle = tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")

    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        return ("dread filled the quiet town before dawn broke 2"
                if injections else "the town woke slowly to a bright morning 2")

    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": bundle})
    monkeypatch.setattr(tasks, "generate", generate)
    return _fake_model


def test_sweep_start_refuses_drifted_pinned_dev_prompts(tmp_path, monkeypatch):
    root = str(tmp_path)
    _sweep_study(root, "s")
    d = es.load_raw("s", root)
    d["sweep"]["devPromptsHash"] = "00" * 32
    es.save_raw(d, root)
    fake_model = _armed_sweep(root, "s", monkeypatch)
    with pytest.raises(RuntimeError,
                       match="sweep dev prompts .* did not pin"):
        tasks.sweep("s", root, model_provider=fake_model,
                    log=lambda *_: None)


def test_sweep_start_refuses_drifted_pinned_battery(tmp_path, monkeypatch):
    root = str(tmp_path)
    _sweep_study(root, "s")
    d = es.load_raw("s", root)
    d["sweep"]["batteryHash"] = "00" * 32
    es.save_raw(d, root)
    fake_model = _armed_sweep(root, "s", monkeypatch)
    with pytest.raises(RuntimeError,
                       match="sweep capability battery .* did not pin"):
        tasks.sweep("s", root, model_provider=fake_model,
                    log=lambda *_: None)


def test_pinned_sweep_runs_and_provenance_agrees_with_the_pin(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    dev_hash, battery_hash = _sweep_study(root, "s")
    d = es.load_raw("s", root)
    d["sweep"]["devPromptsHash"] = dev_hash
    d["sweep"]["batteryHash"] = battery_hash
    es.save_raw(d, root)
    fake_model = _armed_sweep(root, "s", monkeypatch)
    run_dir = tasks.sweep("s", root, model_provider=fake_model,
                          log=lambda *_: None)
    d = es.load_raw("s", root)
    cond = next(c for c in d["conditions"] if c["name"] == "fear-recommended")
    # The ex-post provenance stamp and the manifest pin agree by
    # construction: the start refusal is the only other outcome.
    assert cond["selection"]["devPromptsHash"] == dev_hash
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as handle:
        assert json.load(handle)["fear"]["devPromptsHash"] == dev_hash
