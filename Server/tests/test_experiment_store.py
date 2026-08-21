"""Experiment authoring: create/attach/condition/duplicate/freeze + the
frozen-immutability guard and freeze gating."""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es


def _concept(root, name="french"):
    d = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "positive.jsonl"), "w").write('{"text": "bonjour"}\n')
    open(os.path.join(d, "negative.jsonl"), "w").write('{"text": "hello"}\n')


def test_create_attach_condition(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("My Study", model_id="org/m", root=root)
    d = es.attach("my-study", ["french"], method="lat", pool_from_token=50, root=root)
    assert d["concepts"][0]["name"] == "french"
    assert d["concepts"][0]["stimulusSetHash"]
    assert d["concepts"][0]["options"]["method"] == "lat"
    # readingPosition is the Swift Codable enum shape so the Mac app can read it
    assert d["concepts"][0]["options"]["readingPosition"] == {"meanFromToken": {"_0": 50}}
    d = es.add_condition("my-study", {"name": "fear-L5", "slots": [
        {"concept": "french", "layer": 5, "alpha": 2.0}], "bandWidth": 3}, root=root)
    assert d["conditions"][0]["name"] == "fear-L5"


def test_frozen_is_read_only(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s", model_id="org/m", revision="abc", root=root)
    es.attach("s", ["french"], root=root)  # verify needs a concept/variant
    # force-freeze (skips validate-evidence gate; verify still runs)
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["status"] == "frozen"
    assert frozen["frozenBy"] == "server" and frozen["freezeHash"]
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"temperature": 0.9}, root=root)


def test_freeze_requires_revision_without_force(tmp_path):
    root = str(tmp_path)
    es.create("s2", model_id="org/m", root=root)  # no revision
    with pytest.raises(es.ExperimentStoreError):
        es.freeze("s2", force=False, cached_revision=lambda m: None, root=root)


def test_freeze_requires_validate_run_when_concepts_present(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s3", model_id="org/m", revision="abc", root=root)
    es.attach("s3", ["french"], root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.freeze("s3", force=False, root=root)
    assert "validate" in str(exc.value)


def test_freeze_rejects_bare_scope_file_and_accepts_complete_evidence(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("s4", model_id="org/m", revision="abc", root=root)
    es.attach("s4", ["french"], root=root)
    from steerlab_server.experiment.manifest import Manifest
    scope = Manifest.load("s4", root=root).validation_scope_hash()

    # A run with only a bare scope file is NOT acceptable evidence.
    bare = os.path.join(root, "runs", "x-exp-s4-validate")
    os.makedirs(bare)
    open(os.path.join(bare, "validation-scope-hash.txt"), "w").write(scope)
    with pytest.raises(es.ExperimentStoreError):
        es.freeze("s4", force=False, root=root)

    # A complete validate run (evidence + report with content) IS acceptable.
    full = os.path.join(root, "runs", "y-exp-s4-validate")
    os.makedirs(full)
    json.dump({"schemaVersion": 1, "task": "validate", "validationScopeHash": scope},
              open(os.path.join(full, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"french": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(full, "validation-report.json"), "w"))
    frozen = es.freeze("s4", force=False, root=root)
    assert frozen["status"] == "frozen"


def test_resolve_paths(tmp_path):
    from steerlab_server.experiment import paths
    assert paths.resolve("/abs/path") == "/abs/path"
    assert paths.resolve("rel/x", root=str(tmp_path)) == os.path.join(str(tmp_path), "rel/x")
    assert paths.resolve("") == ""


def test_duplicate_makes_draft(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("orig", model_id="org/m", revision="abc", root=root)
    es.attach("orig", ["french"], root=root)
    es.freeze("orig", force=True, root=root)
    copy = es.duplicate("orig", "orig-v2", root=root)
    assert copy["status"] == "draft"
    assert "freezeHash" not in copy and "frozenBy" not in copy
    # editing the duplicate is allowed
    es.set_protocol("orig-v2", {"temperature": 0.5}, root=root)
