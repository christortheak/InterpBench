"""Confirmation-study perturbation machinery: a declared policy expands
MECHANICALLY into ordinary hashed conditions on a draft manifest — never
hand-picked post-hoc points. Pure-CPU: the expansion is a pure function, and
the store-backed tests exercise the authoring op's refusals against a tmp
workspace. The parity fixture pins the cross-engine contract shared with
Swift's ``ConfirmationStudyTests``.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import confirmation
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest


# --- fixtures ------------------------------------------------------------------

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


def _experiment_with_concept(root, name, concept="fear"):
    _concept_fixture(root, concept)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, [concept], root=root)


def _agent_fixture(root, *, name="fear-agent", concept="fear", layer=17,
                   alpha=0.75, adapters=(), injections=None, promoted=True,
                   run="20260707T000002000-variant"):
    """A variant artifact on disk, discoverable by ``list_variants`` (which
    keys on baseModelID + injections + promptMode)."""
    artifact = {
        "schemaVersion": 1, "name": name, "baseModelID": "org/m",
        "adapters": list(adapters),
        "injections": [{"concept": concept, "vectorArtifactID": "runs/x/fear",
                        "layer": layer, "alpha": alpha}]
        if injections is None else injections,
        "bandWidth": 1, "alphaInNormUnits": True,
        "promptMode": "chatAssistant", "qwenThinkingEnabled": False,
        "temperature": 0.0,
    }
    if promoted:
        artifact["promotion"] = {
            "experiment": "screen", "experimentHash": "h",
            "promotedAt": "2026-07-07T00:00:00Z", "promotedBy": "criterion",
            "winningCell": {"layer": layer, "alpha": alpha},
            "substrate": "python-hf-transformers", "appVersion": "test"}
    path = os.path.join(root, "runs", run, f"{name}.json")
    _write(path, json.dumps(artifact, indent=2, sort_keys=True))
    return os.path.join("runs", run, f"{name}.json")


POLICY = {
    "sourceAgent": {"name": "fear-agent", "artifactPath": "runs/x/a.json",
                    "artifactHash": "a" * 64, "promoted": True},
    "concept": "fear",
    "cell": {"layer": 17, "alpha": 0.75},
    "alphaDeltas": [0.25, 0.5],
    "includeMatchedNormControl": True,
    "declaredAt": "2026-07-07T00:00:00Z",
}


# --- cross-engine parity fixture -------------------------------------------------

def test_expansion_matches_the_cross_engine_contract():
    # THE shared contract, pinned identically in the Swift suite
    # (Tests/ExperimentKitTests/ConfirmationStudyTests.swift): same policy
    # inputs must produce the same condition names, layers, alphas, and
    # controlType values on both engines. Dyadic alphas (0.75 ± 0.25/0.5)
    # keep the arithmetic exact so the pinned expected list is representable.
    conditions = confirmation.expand_conditions(POLICY, band_width=1)
    expected = [
        ("fear-agent-anchor", 17, 0.75, None),
        ("fear-agent-minus-0.25", 17, 0.5, None),
        ("fear-agent-plus-0.25", 17, 1.0, None),
        ("fear-agent-minus-0.5", 17, 0.25, None),
        ("fear-agent-plus-0.5", 17, 1.25, None),
        ("fear-agent-control", 17, 0.75, "randomMatchedNorm"),
    ]
    assert [(c["name"], c["slots"][0]["layer"], c["slots"][0]["alpha"],
             c.get("controlType")) for c in conditions] == expected
    for c in conditions:
        assert c["slots"][0]["concept"] == "fear"
        assert c["alphaInNormUnits"] is True
        assert c["bandWidth"] == 1


def test_expansion_without_control_omits_the_control_condition():
    policy = {**POLICY, "alphaDeltas": [0.25], "includeMatchedNormControl": False}
    conditions = confirmation.expand_conditions(policy, band_width=3)
    assert [c["name"] for c in conditions] == [
        "fear-agent-anchor", "fear-agent-minus-0.25", "fear-agent-plus-0.25"]
    assert all(c["bandWidth"] == 3 for c in conditions)
    assert all("controlType" not in c for c in conditions)


def test_nonpositive_perturbed_alpha_refuses_the_whole_expansion():
    # 0.75 − 0.75 = 0: zero and negative alphas are refused, never clamped.
    policy = {**POLICY, "alphaDeltas": [0.25, 0.75]}
    with pytest.raises(confirmation.ConfirmationError, match="nonpositive"):
        confirmation.expand_conditions(policy, band_width=1)


def test_delta_formatting_is_minimal():
    assert confirmation._fmt(0.2) == "0.2"
    assert confirmation._fmt(0.25) == "0.25"
    assert confirmation._fmt(1.0) == "1"
    assert confirmation._fmt(0.5) == "0.5"


def test_generated_name_matcher_is_agent_scoped():
    for name in ("fear-agent-anchor", "fear-agent-control",
                 "fear-agent-minus-0.2", "fear-agent-plus-0.5"):
        assert confirmation.is_generated_name(name, "fear-agent")
    for name in ("baseline", "fear-recommended", "fear-agent", "other-anchor"):
        assert not confirmation.is_generated_name(name, "fear-agent")


# --- the authoring operation ------------------------------------------------------

def test_attach_expands_conditions_and_stamps_the_policy(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "cf")
    rel = _agent_fixture(root)

    d = confirmation.attach_perturbations(
        "cf", "fear-agent", deltas=(0.25,), root=root, log=lambda *_: None)

    policy = d["perturbationPolicy"]
    assert policy["sourceAgent"]["name"] == "fear-agent"
    assert policy["sourceAgent"]["promoted"] is True
    assert policy["sourceAgent"]["artifactPath"] == rel
    with open(os.path.join(root, rel), "rb") as handle:
        assert policy["sourceAgent"]["artifactHash"] == \
            hashlib.sha256(handle.read()).hexdigest()
    assert policy["concept"] == "fear"
    assert policy["cell"] == {"layer": 17, "alpha": 0.75}
    assert policy["alphaDeltas"] == [0.25]
    assert policy["includeMatchedNormControl"] is True
    assert [c["name"] for c in d["conditions"]] == [
        "fear-agent-anchor", "fear-agent-minus-0.25",
        "fear-agent-plus-0.25", "fear-agent-control"]
    assert d["conditions"][-1]["controlType"] == "randomMatchedNorm"
    # Persisted, and the parsed manifest tolerates + preserves the new key.
    saved = es.load_raw("cf", root)
    assert saved["perturbationPolicy"] == policy
    manifest = Manifest.load("cf", root)
    assert manifest.raw["perturbationPolicy"] == policy
    assert [c.name for c in manifest.conditions][-1] == "fear-agent-control"
    assert manifest.conditions[-1].control_type == "randomMatchedNorm"


def test_policy_participates_in_the_content_hash(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "ch")
    _agent_fixture(root)
    before = Manifest.load("ch", root).content_hash()
    confirmation.attach_perturbations(
        "ch", "fear-agent", root=root, log=lambda *_: None)
    after = Manifest.load("ch", root).content_hash()
    assert before != after


def test_rerun_replaces_previously_generated_conditions(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "rr")
    _agent_fixture(root)
    confirmation.attach_perturbations(
        "rr", "fear-agent", deltas=(0.25,), root=root, log=lambda *_: None)
    # A hand-authored condition must survive the replace.
    es.add_condition("rr", {
        "name": "hand-authored", "alphaInNormUnits": False,
        "slots": [{"concept": "fear", "layer": 3, "alpha": 0.1}]}, root)

    d = confirmation.attach_perturbations(
        "rr", "fear-agent", deltas=(0.5,), include_control=False,
        root=root, log=lambda *_: None)

    assert [c["name"] for c in d["conditions"]] == [
        "hand-authored", "fear-agent-anchor",
        "fear-agent-minus-0.5", "fear-agent-plus-0.5"]
    assert d["perturbationPolicy"]["alphaDeltas"] == [0.5]
    assert d["perturbationPolicy"]["includeMatchedNormControl"] is False


def test_frozen_manifests_refuse(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "fz")
    _agent_fixture(root)
    d = es.load_raw("fz", root)
    d["status"] = "frozen"
    es.save_raw(d, root, freeze_transition=True)
    with pytest.raises(confirmation.ConfirmationError,
                       match="is frozen — duplicate first"):
        confirmation.attach_perturbations(
            "fz", "fear-agent", root=root, log=lambda *_: None)


def test_adapter_bearing_agents_refuse(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "ad")
    _agent_fixture(root, name="lora-agent",
                   adapters=[{"name": "a", "artifactPath": "runs/x/a.json",
                              "adapterDirectory": "runs/x/a",
                              "adapterHash": "h"}])
    with pytest.raises(confirmation.ConfirmationError,
                       match="vector-only agents only"):
        confirmation.attach_perturbations(
            "ad", "lora-agent", root=root, log=lambda *_: None)


def test_zero_and_multi_injection_agents_refuse(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "mi")
    _agent_fixture(root, name="empty-agent", injections=[],
                   run="20260707T000003000-variant")
    _agent_fixture(root, name="multi-agent", injections=[
        {"concept": "fear", "vectorArtifactID": "x", "layer": 3, "alpha": 0.4},
        {"concept": "calm", "vectorArtifactID": "y", "layer": 5, "alpha": 0.2}],
        run="20260707T000004000-variant")
    with pytest.raises(confirmation.ConfirmationError, match="no injections"):
        confirmation.attach_perturbations(
            "mi", "empty-agent", root=root, log=lambda *_: None)
    with pytest.raises(confirmation.ConfirmationError,
                       match="multi-injection agents are not supported"):
        confirmation.attach_perturbations(
            "mi", "multi-agent", root=root, log=lambda *_: None)


def test_unattached_concept_refuses_without_silently_attaching(tmp_path):
    root = str(tmp_path)
    es.create("uc", model_id="org/m", revision="abc", root=root)
    _agent_fixture(root)
    with pytest.raises(confirmation.ConfirmationError,
                       match="attach concept 'fear' to 'uc' first"):
        confirmation.attach_perturbations(
            "uc", "fear-agent", root=root, log=lambda *_: None)
    assert not es.load_raw("uc", root).get("concepts")


def test_nonpositive_alpha_refuses_end_to_end_and_writes_nothing(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "np")
    _agent_fixture(root)
    with pytest.raises(confirmation.ConfirmationError, match="nonpositive"):
        confirmation.attach_perturbations(
            "np", "fear-agent", deltas=(0.75,), root=root, log=lambda *_: None)
    d = es.load_raw("np", root)
    assert "perturbationPolicy" not in d
    assert not d.get("conditions")


def test_explicit_path_resolution_records_promoted_false(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "pp")
    rel = _agent_fixture(root, name="hand-agent", promoted=False)
    d = confirmation.attach_perturbations(
        "pp", rel, root=root, log=lambda *_: None)
    policy = d["perturbationPolicy"]
    assert policy["sourceAgent"]["name"] == "hand-agent"
    assert policy["sourceAgent"]["promoted"] is False
    assert policy["sourceAgent"]["artifactPath"] == rel


def test_missing_agent_refuses(tmp_path):
    root = str(tmp_path)
    _experiment_with_concept(root, "na")
    with pytest.raises(confirmation.ConfirmationError, match="no variant named"):
        confirmation.attach_perturbations(
            "na", "ghost-agent", root=root, log=lambda *_: None)


# --- route ------------------------------------------------------------------------

def test_confirm_route(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    _experiment_with_concept(root, "rt")
    _agent_fixture(root)

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    response = client.post("/api/experiment/rt/confirm",
                           json={"agent": "fear-agent", "deltas": [0.25]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["perturbationPolicy"]["cell"] == {"layer": 17, "alpha": 0.75}
    assert [c["name"] for c in body["conditions"]] == [
        "fear-agent-anchor", "fear-agent-minus-0.25",
        "fear-agent-plus-0.25", "fear-agent-control"]

    missing = client.post("/api/experiment/rt/confirm", json={})
    assert missing.status_code == 400

    bad_deltas = client.post("/api/experiment/rt/confirm",
                             json={"agent": "fear-agent", "deltas": "nope"})
    assert bad_deltas.status_code == 400

    ghost = client.post("/api/experiment/rt/confirm",
                        json={"agent": "ghost-agent"})
    assert ghost.status_code == 400
