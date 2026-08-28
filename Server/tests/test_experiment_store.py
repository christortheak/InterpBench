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
        {"concept": "french", "layer": 5, "alpha": 2.0}], "bandWidth": 3,
        "alphaInNormUnits": False}, root=root)
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


def test_set_protocol_refuses_unknown_keys_at_the_store(tmp_path):
    """The store itself refuses, not just the client CLI: the HTTP authoring
    route hands request bodies straight to ``set_protocol``, so a drop here
    would keep that surface silent. Nothing is written on refusal — the
    valid keys in the same call must not land either."""
    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.set_protocol("s", {"temperature": 0.7, "notAField": 1}, root=root)
    assert "'notAField'" in str(exc.value)
    assert "temperature" in str(exc.value)  # the vocabulary is listed
    assert exc.value.repair_action
    d = es.load_raw("s", root)
    assert d["temperature"] == 0.0  # create's default, not 0.7
    assert "notAField" not in d
    # The contract keys the Mac spells as verbs land as protocol fields.
    d = es.set_protocol(
        "s", {"outcomeInstruments": ["sampledText"],
              "sweep": {"selection": {"objective": {"kind": "markerDensity"}}}},
        root=root)
    assert d["outcomeInstruments"] == ["sampledText"]
    assert d["sweep"]["selection"]["objective"]["kind"] == "markerDensity"
    # …with the instrument vocabulary enforced at declaration (Swift
    # `setOutcomeInstruments` twin) and sweep required to be an object.
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"outcomeInstruments": ["sampledTxt"]}, root=root)
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"sweep": "markerDensity"}, root=root)


def test_set_protocol_gates_the_sampling_protocol_fields(tmp_path):
    """The six generation-protocol fields carry declaration-time value gates
    (Swift twins: ``ExperimentStore.setSamplingProtocol`` /
    ``setExclusionRules``). Two loss classes: an out-of-vocabulary
    promptMode/seedPolicy is read by equality tests downstream and silently
    behaves as the default, and a non-numeric temperature/maxTokens/
    samplesPerItem BRICKS the manifest — ``Manifest.from_dict`` raises on the
    next load, so every later verb fails before it can repair. Nothing is
    written on refusal, including the valid keys of the same call."""
    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    # The stochastic replication arm that motivated the writers (25 samples ×
    # T=0.7 × 1024 tokens): authorable, and the run loop's own decoder reads
    # it back.
    d = es.set_protocol(
        "s", {"temperature": 0.7, "maxTokens": 1024, "samplesPerItem": 25,
              "seedPolicy": "derivedSHA256", "promptMode": "chatAssistant"},
        root=root)
    assert d["samplesPerItem"] == 25 and d["seedPolicy"] == "derivedSHA256"
    from steerlab_server.experiment.manifest import Manifest
    manifest = Manifest.from_dict(d)
    assert manifest.samples_per_item == 25
    assert manifest.seed_policy == "derivedSHA256"
    for bad in ({"temperature": "hot"}, {"temperature": -0.5},
                {"temperature": True},
                {"maxTokens": 0}, {"maxTokens": "lots"},
                {"promptMode": "freestyle"},
                {"samplesPerItem": 0}, {"samplesPerItem": "three"},
                {"seedPolicy": "diceRoll"},
                {"exclusionRules": [{"rule": "outOfRange"}]},
                {"exclusionRules": "outOfRange"}):
        with pytest.raises(es.ExperimentStoreError) as exc:
            es.set_protocol("s", dict(bad, taskDescription="x"), root=root)
        assert exc.value.repair_action, bad
    d = es.load_raw("s", root)
    assert "taskDescription" not in d  # the valid co-key never landed
    assert d["samplesPerItem"] == 25  # the earlier declaration survives
    # A JSON null clears like an absent key on decode, so None passes.
    es.set_protocol("s", {"seedPolicy": None}, root=root)
    assert "seedPolicy" not in es.load_raw("s", root)
    # Valid exclusion rules land verbatim; an unknown rule id refuses with
    # the engine's own wording and the Mac verb named in the repair.
    d = es.set_protocol(
        "s", {"exclusionRules": [
            {"rule": "unparseableEndpoint"},
            {"rule": "outOfRange", "min": 0, "max": 600}]}, root=root)
    assert [r["rule"] for r in d["exclusionRules"]] == [
        "unparseableEndpoint", "outOfRange"]
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.set_protocol("s", {"exclusionRules": [{"rule": "outOfRnge"}]},
                        root=root)
    assert "not recognized" in str(exc.value)
    assert "set-exclusions" in exc.value.repair_action


def test_an_explicit_json_null_clears_a_protocol_field(tmp_path):
    """Review round 10, finding 1. Every value gate reads
    ``fields.get(k) is not None``, so a null passes them all ungated — and the
    persistence loop used to WRITE it. ``Manifest.from_dict`` then raised
    ``TypeError`` on the next load and every later verb died before it could
    name the problem, verify included: a client verb that BRICKS the manifest.

    The site's own comment claimed "a JSON null clears like an absent key on
    decode"; the loop now makes that claim true by popping the key, which is
    also the symmetric affordance to the Swift writers' ``""`` clears."""
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    es.create("s", model_id="org/m", root=root)
    written = {"temperature": 0.7, "maxTokens": 1024, "samplesPerItem": 25,
               "seedPolicy": "derivedSHA256", "promptMode": "chatAssistant",
               "exclusionRules": [{"rule": "unparseableEndpoint"}],
               "outcomeInstruments": ["sampledText"]}
    es.set_protocol("s", dict(written), root=root)
    for key in written:
        # Write a value, null it, and the key is GONE — not present as null.
        assert key in es.load_raw("s", root), key
        d = es.set_protocol("s", {key: None}, root=root)
        assert key not in d, key
        assert key not in es.load_raw("s", root), key
        # …and the manifest still loads, which is the whole point: the
        # bricked state can no longer be produced.
        Manifest.from_dict(es.load_raw("s", root))

    # Nulling every field at once, on a fresh draft, is the same story.
    es.create("t", model_id="org/m", root=root)
    es.set_protocol("t", dict(written), root=root)
    d = es.set_protocol("t", {key: None for key in written}, root=root)
    assert not (set(written) & set(d))
    manifest = Manifest.from_dict(es.load_raw("t", root))
    assert manifest.temperature is None or manifest.temperature is not False
    # Nulling a key that was never there is a no-op, not a KeyError.
    es.set_protocol("t", {"temperature": None}, root=root)
    assert "temperature" not in es.load_raw("t", root)
    # An unknown key is still refused BEFORE any of this — a null does not
    # buy a way past the vocabulary.
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("t", {"notAField": None}, root=root)


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
