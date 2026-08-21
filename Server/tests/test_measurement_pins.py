"""Measurement-side input pins (firewall closure, 2026-07-13): markers.json,
per-concept validation.jsonl, and the neutral-PC basis bytes enter the same
drift firewall as the stimuli — attach/freeze pin them, verify() reports
drift as a VIOLATION, legacy manifests pass with a non-blocking freeze
advisory. Cross-engine contract keys: concept pin ``validationHash``,
manifest ``markersHash``, condition ``neutralPCBasisHash``."""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import (
    Manifest, concept_validation_hash, markers_aggregate_hash)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _concept(root, name="fear", validation=None):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    if validation is not None:
        return _write(os.path.join(d, "validation.jsonl"), validation)
    return None


VALIDATION_ROW = '{"text": "the shadows closed in", "expresses": true}\n'


# --- attach pins validationHash ----------------------------------------------

def test_attach_pins_validation_hash_when_file_exists(tmp_path):
    root = str(tmp_path)
    digest = _concept(root, validation=VALIDATION_ROW)
    es.create("s", model_id="org/m", revision="abc", root=root)
    d = es.attach("s", ["fear"], root=root)
    assert d["concepts"][0]["validationHash"] == digest


def test_attach_pins_null_when_no_validation_file(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    d = es.attach("s", ["fear"], root=root)
    # The KEY is always present on new attaches — null means "pinned absent".
    assert "validationHash" in d["concepts"][0]
    assert d["concepts"][0]["validationHash"] is None


def test_attach_pins_grand_mean_validation_from_emotions_tree(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "prompts", "emotions", "fear", "stories.jsonl"),
           '{"concept": "fear", "text": "a long dread story"}\n')
    digest = _write(os.path.join(root, "prompts", "emotions", "fear",
                                 "validation.jsonl"), VALIDATION_ROW)
    es.create("gm", model_id="org/m", revision="abc", root=root)
    d = es.attach("gm", ["fear"], method="emotionGrandMean", root=root)
    assert d["concepts"][0]["validationHash"] == digest


# --- verify(): drift is a violation, legacy is clean ---------------------------

def test_verify_flags_validation_drift_and_disappearance(tmp_path):
    root = str(tmp_path)
    _concept(root, validation=VALIDATION_ROW)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)
    assert Manifest.load("s", root=root).verify(root) == []

    val = os.path.join(root, "prompts", "concepts", "fear", "validation.jsonl")
    with open(val, "a", encoding="utf-8") as handle:
        handle.write('{"text": "EDITED", "expresses": false}\n')
    violations = Manifest.load("s", root=root).verify(root)
    assert any("validation.jsonl changed since pinning" in v for v in violations)

    os.remove(val)
    violations = Manifest.load("s", root=root).verify(root)
    assert any("pinned validation.jsonl missing" in v for v in violations)


def test_verify_flags_validation_appearing_after_null_pin(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)  # pins validationHash: null
    assert Manifest.load("s", root=root).verify(root) == []
    _concept(root, validation=VALIDATION_ROW)
    violations = Manifest.load("s", root=root).verify(root)
    assert any("appeared after pinning" in v for v in violations)


def test_legacy_concept_pin_without_key_verifies_clean(tmp_path):
    root = str(tmp_path)
    _concept(root, validation=VALIDATION_ROW)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)
    # Simulate a pre-2026-07-13 attach: strip the pin key entirely.
    d = es.load_raw("s", root)
    del d["concepts"][0]["validationHash"]
    es.save_raw(d, root)
    assert Manifest.load("s", root=root).verify(root) == []
    # …but freeze surfaces the non-blocking advisory.
    advisories = es.freeze_advisories(es.load_raw("s", root), root)
    assert any("measurement-side inputs unpinned" in a for a in advisories)


# --- markersHash: pinned at freeze, drift is a violation ------------------------

MARKERS_FEAR = '{"words": ["dread", "terror"]}\n'
MARKERS_CALM = '{"words": ["serene"]}\n'


def test_markers_aggregate_hash_contract_fixture(tmp_path):
    """Cross-engine fixture: literal bytes → literal aggregate. Rule: for each
    distinct concept name sorted ascending, with an existing markers.json,
    the line "<name>\\t<sha256-hex>\\n"; aggregate = sha256(utf-8 concat)."""
    root = str(tmp_path)
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)
    _write(os.path.join(root, "prompts", "concepts", "calm", "markers.json"),
           MARKERS_CALM)
    assert markers_aggregate_hash(["fear", "calm", "fear"], root) == \
        "2bb18ec6b173f139962dc4c4eb982fe76ae35c559123703d852e642102ea4c94"
    # No markers anywhere → null pin.
    assert markers_aggregate_hash(["absent"], root) is None


def test_freeze_pins_markers_hash_and_verify_flags_drift(tmp_path):
    root = str(tmp_path)
    _concept(root)
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["markersHash"] == markers_aggregate_hash(["fear"], root)
    assert Manifest.load("s", root=root).verify(root) == []

    # Post-freeze markers drift → verification violation, like stimulus drift.
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           '{"words": ["EDITED"]}\n')
    violations = Manifest.load("s", root=root).verify(root)
    assert any("markers.json changed since pinning" in v for v in violations)


def test_freeze_pins_markers_null_and_flags_late_appearance(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)
    frozen = es.freeze("s", force=True, root=root)
    assert "markersHash" in frozen and frozen["markersHash"] is None
    assert Manifest.load("s", root=root).verify(root) == []
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)
    violations = Manifest.load("s", root=root).verify(root)
    assert any("markers.json appeared after pinning" in v for v in violations)


def test_legacy_manifest_without_markers_key_verifies_clean(tmp_path):
    root = str(tmp_path)
    _concept(root)
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)  # draft: no markersHash key yet
    assert "markersHash" not in es.load_raw("s", root)
    assert Manifest.load("s", root=root).verify(root) == []


# --- neutralPCBasisHash: written pins are now enforced ---------------------------

def _basis_manifest(root, *, path, digest):
    return Manifest.from_dict({
        "name": "nb", "modelID": "org/m",
        "concepts": [], "conditions": [
            {"name": "fear-a2",
             "slots": [{"concept": "fear", "layer": 3, "alpha": 2.0}],
             "neutralPCBasisPath": path, "neutralPCBasisHash": digest}],
        "variantConditions": [{"name": "v", "artifactPath": "",
                               "artifactHash": "x", "artifact": {}}]})


def test_neutral_pc_basis_hash_verified_against_file_bytes(tmp_path):
    root = str(tmp_path)
    basis_rel = "runs/neutral-pcs/b1/neutral-pc-basis.json"
    digest = _write(os.path.join(root, basis_rel),
                    {"kind": "neutralPCBasis", "componentsByLayer": {"3": [[1.0, 0.0]]}})
    ok = _basis_manifest(root, path=basis_rel, digest=digest)
    assert not [v for v in ok.verify(root) if "neutral-PC" in v]

    drifted = _basis_manifest(root, path=basis_rel, digest="00" * 32)
    assert any("neutral-PC basis" in v and "changed since pinning" in v
               for v in drifted.verify(root))

    missing = _basis_manifest(root, path="runs/neutral-pcs/gone.json",
                              digest=digest)
    assert any("neutral-PC basis" in v and "missing" in v
               for v in missing.verify(root))


def test_neutral_pc_basis_directory_path_resolves_sidecar(tmp_path):
    root = str(tmp_path)
    basis_dir = "runs/neutral-pcs/b2"
    digest = _write(os.path.join(root, basis_dir, "neutral-pc-basis.json"),
                    {"kind": "neutralPCBasis", "componentsByLayer": {}})
    ok = _basis_manifest(root, path=basis_dir, digest=digest)
    assert not [v for v in ok.verify(root) if "neutral-PC" in v]


def test_neutral_pc_basis_hash_without_path_is_incomplete_pin(tmp_path):
    manifest = _basis_manifest(str(tmp_path), path=None, digest="ab" * 32)
    assert any("neutralPCBasisHash without neutralPCBasisPath" in v
               for v in manifest.verify(str(tmp_path)))


def test_neutral_pc_basis_path_without_hash_is_legacy_clean(tmp_path):
    root = str(tmp_path)
    manifest = _basis_manifest(root, path="runs/neutral-pcs/gone.json",
                               digest=None)
    assert not [v for v in manifest.verify(root) if "neutral-PC" in v]


# --- frozen studies refuse to run on measurement drift ---------------------------

def test_frozen_study_fails_verification_on_markers_drift(tmp_path):
    """The run-time firewall (`_verify_or_warn`) hard-refuses a frozen study
    whose pinned markers drifted — the score-time input is no longer the
    frozen one."""
    from steerlab_server.experiment import tasks
    root = str(tmp_path)
    _concept(root)
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           MARKERS_FEAR)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["fear"], root=root)
    es.freeze("s", force=True, root=root)
    _write(os.path.join(root, "prompts", "concepts", "fear", "markers.json"),
           '{"words": ["EDITED"]}\n')
    with pytest.raises(RuntimeError, match="failed verification"):
        tasks._verify_or_warn(Manifest.load("s", root=root), root)


def test_concept_validation_hash_helper_paths(tmp_path):
    root = str(tmp_path)
    digest = _concept(root, validation=VALIDATION_ROW)
    assert concept_validation_hash("fear", paired=True, root=root) == digest
    # Dual-root lookup (2026-08-19): a set filed under the PAIRED home is
    # found by the grand-mean recipe too, via the fallback — it used to read
    # as absent, so a misfiled measurement-side input was silently unpinned.
    assert concept_validation_hash("fear", paired=False, root=root) == digest
    gm = _write(os.path.join(root, "prompts", "emotions", "fear",
                             "validation.jsonl"), VALIDATION_ROW + "\n")
    # …and the CANONICAL home always wins once it holds a file.
    assert gm != digest
    assert concept_validation_hash("fear", paired=False, root=root) == gm
    assert concept_validation_hash("fear", paired=True, root=root) == digest
