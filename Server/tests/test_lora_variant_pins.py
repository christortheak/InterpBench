"""LoRA training provenance on the freeze/verify surface (cluster-LoRA
readiness §0 amendments 1+2, contract §9).

An adapter enters a study as a VARIANT, so the manifest is where its training
provenance becomes evidence: freeze pins the dataset manifest + adapter
sidecar from the adapter's own v2 sidecar, verify() re-hashes both (drift is a
violation, exactly like stimulus drift), the existing ``variantValidity`` gate
refuses an evidence-grade adapter whose dataset never entered the pin surface,
and two non-blocking advisories name an exploratory adapter and a missing ex
ante matched control.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload, indent=1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


DATASET_MANIFEST_REL = "adapters/stance-lora-family-v1-manifest.json"
ADAPTER_DIR_REL = "runs/2026-08-12-lora/stance-lora-v1"
SIDECAR_REL = "runs/2026-08-12-lora/stance-lora-v1.json"


def _dataset_manifest(root, *, rows=8):
    return _write(os.path.join(root, DATASET_MANIFEST_REL),
                  {"bundleID": "stance-lora-family-v1",
                   "trainFiles": [{"path": "adapters/x/training/train.jsonl",
                                   "sha256": "ee" * 32, "rows": rows,
                                   "rowsRoot": "ff" * 32}]})


def _sidecar(root, *, schema_version=2, evidence_grade=True,
             manifest_hash="", bundle_id="stance-lora-family-v1"):
    os.makedirs(os.path.join(root, ADAPTER_DIR_REL), exist_ok=True)
    payload = {"adapterFormat": "hf-peft-lora", "baseModelID": "org/m"}
    if schema_version is not None:
        payload["schemaVersion"] = schema_version
    if schema_version and schema_version >= 2:
        payload["evidenceGrade"] = evidence_grade
        payload["trainingMode"] = "instruction_chat"
        payload["dataset"] = {"bundleID": bundle_id,
                              "manifestPath": DATASET_MANIFEST_REL,
                              "manifestHash": manifest_hash}
    return _write(os.path.join(root, SIDECAR_REL), payload)


def _variant_study(root, name="lora-study", *, adapters=True):
    artifact = {"name": "stance-lora", "baseModelID": "org/m",
                "injections": [], "temperature": 0.0,
                "adapters": ([{"name": "stance-lora",
                               "artifactPath": ADAPTER_DIR_REL,
                               "adapterDirectory": ADAPTER_DIR_REL,
                               "adapterHash": "ab" * 32}]
                             if adapters else [])}
    rel = "runs/model-variants/stance-lora/model-variant.json"
    digest = _write(os.path.join(root, rel), artifact)
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["variantConditions"] = [{"name": "stance-lora", "artifactPath": rel,
                               "artifactHash": digest, "artifact": artifact}]
    es.save_raw(d, root)
    return name


def _validate_evidence(root, name, *, conditions=("baseline", "stance-lora")):
    """Scope-matched validate evidence with per-condition battery results, so
    the batteryEvidence gate is not what refuses in these tests."""
    from steerlab_server.experiment import battery as battery_mod
    scope = Manifest.load(name, root=root).validation_scope_hash()
    battery_hash = battery_mod.live_hash(battery_mod.DEFAULT_BATTERY_FILE, root)
    rundir = os.path.join(root, "runs", f"v-exp-{name}-validate")
    os.makedirs(rundir, exist_ok=True)
    with open(os.path.join(rundir, "validation-evidence.json"), "w") as handle:
        json.dump({"schemaVersion": 1, "task": "validate",
                   "substrate": "python-hf-transformers",
                   "validationScopeHash": scope,
                   "batteryResults": [
                       {"condition": c, "batteryHash": battery_hash,
                        "total": 4, "correct": 4, "accuracy": 1.0}
                       for c in conditions]}, handle)
    with open(os.path.join(rundir, "validation-report.json"), "w") as handle:
        json.dump({"concepts": {}}, handle)
    return rundir


def _evidence_grade_study(root, name="lora-study", *, matched_control=None):
    """A study whose adapter sidecar is v2/evidence-grade and whose dataset
    manifest exists — the shape freeze is meant to pin from."""
    manifest_hash = _dataset_manifest(root)
    _sidecar(root, manifest_hash=manifest_hash)
    _variant_study(root, name)
    if matched_control is not None:
        d = es.load_raw(name, root)
        d["variantConditions"][0]["trainingProvenance"] = {
            "matchedControl": matched_control}
        es.save_raw(d, root)
    _validate_evidence(root, name)
    return name, manifest_hash


# --- freeze: auto-pin from the sidecar --------------------------------------

def test_freeze_pins_training_provenance_from_v2_sidecar(tmp_path):
    root = str(tmp_path)
    name, manifest_hash = _evidence_grade_study(
        root, matched_control={"variant": "stance-lora-control",
                               "kind": "shuffledAssistantPairing"})
    frozen = es.freeze(name, force=False, root=root)
    block = frozen["variantConditions"][0]["trainingProvenance"]
    assert block["datasetBundleID"] == "stance-lora-family-v1"
    assert block["datasetManifestPath"] == DATASET_MANIFEST_REL
    assert block["datasetManifestHash"] == manifest_hash
    assert block["evidenceGrade"] is True
    with open(os.path.join(root, SIDECAR_REL), "rb") as handle:
        assert block["adapterSidecarHash"] == \
            hashlib.sha256(handle.read()).hexdigest()
    # The ex ante declaration survives the auto-pin (never re-derived).
    assert block["matchedControl"]["kind"] == "shuffledAssistantPairing"
    # ...and the frozen hash COVERS the block: recomputing it after dropping
    # the pins must not reproduce the stamped freezeHash.
    stripped = json.loads(json.dumps(frozen))
    del stripped["variantConditions"][0]["trainingProvenance"]
    assert Manifest.from_dict(stripped).content_hash() != frozen["freezeHash"]
    assert Manifest.from_dict(frozen).content_hash() == frozen["freezeHash"]


def test_freeze_stamps_null_matched_control_when_undeclared(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(root)
    frozen = es.freeze(name, force=False, root=root)
    block = frozen["variantConditions"][0]["trainingProvenance"]
    assert "matchedControl" in block and block["matchedControl"] is None


def test_freeze_never_repins_a_declared_provenance_value(tmp_path):
    root = str(tmp_path)
    name, manifest_hash = _evidence_grade_study(root)
    d = es.load_raw(name, root)
    d["variantConditions"][0]["trainingProvenance"] = {
        "datasetManifestPath": DATASET_MANIFEST_REL,
        "datasetManifestHash": manifest_hash,
        "datasetBundleID": "declared-by-hand"}
    es.save_raw(d, root)
    frozen = es.freeze(name, force=False, root=root)
    block = frozen["variantConditions"][0]["trainingProvenance"]
    assert block["datasetBundleID"] == "declared-by-hand"


def test_legacy_v1_sidecar_freezes_unchanged(tmp_path):
    """Regression: an exploratory adapter gains NO new manifest keys."""
    root = str(tmp_path)
    _sidecar(root, schema_version=1)
    name = _variant_study(root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, force=False, root=root)
    assert "trainingProvenance" not in frozen["variantConditions"][0]


def test_no_sidecar_at_all_freezes_unchanged(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, force=False, root=root)
    assert "trainingProvenance" not in frozen["variantConditions"][0]


# --- verify(): drift is a violation -----------------------------------------

def test_dataset_manifest_drift_after_freeze_is_a_violation(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(root)
    es.freeze(name, force=False, root=root)
    _dataset_manifest(root, rows=9)  # one row appears after the freeze
    violations = Manifest.load(name, root=root).verify(root)
    assert any("dataset manifest changed since pinning" in v
               for v in violations), violations


def test_missing_dataset_manifest_is_a_violation(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(root)
    es.freeze(name, force=False, root=root)
    os.remove(os.path.join(root, DATASET_MANIFEST_REL))
    violations = Manifest.load(name, root=root).verify(root)
    assert any("file missing" in v and "dataset manifest" in v
               for v in violations), violations


def test_adapter_sidecar_drift_after_freeze_is_a_violation(tmp_path):
    root = str(tmp_path)
    name, manifest_hash = _evidence_grade_study(root)
    es.freeze(name, force=False, root=root)
    _sidecar(root, manifest_hash=manifest_hash, bundle_id="renamed-bundle")
    violations = Manifest.load(name, root=root).verify(root)
    assert any("adapter sidecar changed since pinning" in v
               for v in violations), violations


def test_missing_adapter_sidecar_is_a_violation(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(root)
    es.freeze(name, force=False, root=root)
    os.remove(os.path.join(root, SIDECAR_REL))
    violations = Manifest.load(name, root=root).verify(root)
    assert any("adapter sidecar: file missing" in v
               for v in violations), violations


def test_half_pinned_provenance_is_a_violation(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root)
    d = es.load_raw(name, root)
    d["variantConditions"][0]["trainingProvenance"] = {
        "datasetManifestPath": DATASET_MANIFEST_REL}
    es.save_raw(d, root)
    violations = Manifest.load(name, root=root).verify(root)
    assert any("half-pin certifies nothing" in v for v in violations), violations


# --- variantValidity gate ---------------------------------------------------

def test_evidence_grade_adapter_without_dataset_pin_fails_the_gate(tmp_path):
    """The sidecar claims evidence grade but its dataset block carries no
    manifest hash, so the auto-pin has nothing to stamp — the training data
    would stay outside the pin surface."""
    root = str(tmp_path)
    _dataset_manifest(root)
    _sidecar(root, manifest_hash="")
    # No manifestPath either: an incomplete provenance stamp.
    sidecar_path = os.path.join(root, SIDECAR_REL)
    with open(sidecar_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["dataset"] = {"bundleID": "stance-lora-family-v1"}
    _write(sidecar_path, payload)
    name = _variant_study(root)
    _validate_evidence(root, name)
    with pytest.raises(es.ExperimentStoreError,
                       match="trainingProvenance.datasetManifestHash"):
        es.freeze(name, force=False, root=root)
    # It is the variantValidity gate — force skips it loudly and stamps it.
    frozen = es.freeze(name, force=True, root=root)
    assert "variantValidity" in frozen["forcedGatesSkipped"]


def test_manifest_declared_evidence_grade_alone_arms_the_gate(tmp_path):
    """No readable sidecar (the artifact tree is remote), but the manifest
    itself says evidence-grade — the gate still applies."""
    root = str(tmp_path)
    name = _variant_study(root)
    d = es.load_raw(name, root)
    d["variantConditions"][0]["trainingProvenance"] = {"evidenceGrade": True}
    es.save_raw(d, root)
    _validate_evidence(root, name)
    with pytest.raises(es.ExperimentStoreError,
                       match="trainingProvenance.datasetManifestHash"):
        es.freeze(name, force=False, root=root)


def test_exploratory_adapter_does_not_fail_the_gate(tmp_path):
    root = str(tmp_path)
    _sidecar(root, schema_version=2, evidence_grade=False)
    name = _variant_study(root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, force=False, root=root)
    assert frozen["status"] == "frozen"


def test_evidence_grade_study_with_full_pins_freezes(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(
        root, matched_control={"variant": "stance-lora-control",
                               "kind": "shuffledAssistantPairing"})
    frozen = es.freeze(name, force=False, root=root)
    assert frozen["status"] == "frozen"
    assert not frozen.get("freezeForced")


# --- advisories (non-blocking) ----------------------------------------------

def test_exploratory_adapter_advisory(tmp_path):
    root = str(tmp_path)
    _sidecar(root, schema_version=1)
    name = _variant_study(root)
    advisories = es.freeze_advisories(es.load_raw(name, root), root)
    assert any("EXPLORATORY adapter" in a for a in advisories), advisories


def test_missing_matched_control_advisory(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(root)
    es.freeze(name, force=False, root=root)
    advisories = es.freeze_advisories(es.load_raw(name, root), root)
    assert any("no matchedControl declared" in a.lower()
               or "NO matchedControl declared" in a for a in advisories), advisories
    assert not any("EXPLORATORY adapter" in a for a in advisories), advisories


def test_declared_matched_control_silences_the_advisory(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(
        root, matched_control={"variant": "stance-lora-control",
                               "kind": "shuffledAssistantPairing"})
    es.freeze(name, force=False, root=root)
    advisories = es.freeze_advisories(es.load_raw(name, root), root)
    assert not any("matchedControl" in a for a in advisories), advisories


def test_adapterless_variant_gets_no_adapter_advisories(tmp_path):
    root = str(tmp_path)
    name = _variant_study(root, adapters=False)
    advisories = es.freeze_advisories(es.load_raw(name, root), root)
    assert not any("adapter" in a.lower() for a in advisories), advisories


# --- pin surface ------------------------------------------------------------

def test_dataset_manifest_joins_the_pinned_input_surface(tmp_path):
    root = str(tmp_path)
    name, _ = _evidence_grade_study(root)
    frozen = es.freeze(name, force=False, root=root)
    labels = {e.label: e.path
              for e in es.pinned_input_entries(frozen, root)}
    key = "variant 'stance-lora' training dataset manifest"
    assert key in labels
    assert labels[key].endswith(DATASET_MANIFEST_REL)
    # The no-git reproducibility floor snapshotted it too.
    snapshot = os.path.join(root, "experiments", name, "pinned",
                            DATASET_MANIFEST_REL)
    assert os.path.isfile(snapshot)
