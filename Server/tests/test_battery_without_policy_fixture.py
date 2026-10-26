"""A validation-evidence file with a policy agent's static-part reading,
written by this engine.

An agent that uses an intervention policy cannot be read by the capability
battery as it runs, so its row is not applicable. When that agent also carries
static steering (vectors or adapters), validation reads the same agent with
its policies removed and records that reading INSIDE the not-applicable row,
under ``withoutPolicy`` — never as the condition's own score (ruling
2026-10-23).

This module produces the file the Mac side must read. It is producer-generated
on purpose (``test_battery_error_row_fixture.py`` uses the same idiom): the
bytes committed under ``Tests/Fixtures/cross-engine/`` come from the real
validation code, so the Mac test cannot pin its author's belief about what
this engine writes. The structural assertions always run against the fresh
document; the byte comparison waits for the file to be committed. To
regenerate, delete the fixture and re-run this module.

Swift twin: ``Tests/ExperimentKitTests/PolicyAgentFreezeTests.swift``
(``pythonStaticPartRowDecodesAndSatisfiesTheGate``).
"""

import json
import os

import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.model_variant as _owner_model_variant
import steerlab_server.experiment.validation_workflow as _owner_validation_workflow
import steerlab_server.experiment.vector_materialization as _owner_vector_materialization
from steerlab_server.experiment.manifest import Manifest
import golden_files
from test_intervention_policies import attached, setup

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(
    REPO, "Tests", "Fixtures", "cross-engine",
    "validation-evidence-without-policy.json")



def _agent(name, *, policies=None, injections=()):
    artifact = {"name": name, "baseModelID": "org/m", "adapters": [],
                "injections": list(injections), "temperature": 0.0,
                "promptMode": "chatAssistant"}
    if policies is not None:
        artifact["interventionPolicies"] = policies
    return {"name": name, "artifactPath": "", "artifactHash": "x",
            "artifact": artifact}


def test_validation_evidence_with_a_static_part_reading_is_current(tmp_path, monkeypatch):
    root = str(tmp_path)
    doc, _, _ = setup(tmp_path)
    policies = attached(doc, tmp_path)
    battery = os.path.join(root, "prompts", "batteries", "basic.jsonl")
    os.makedirs(os.path.dirname(battery))
    with open(battery, "w", encoding="utf-8") as handle:
        handle.write('{"prompt": "2+2?", "answer": "4"}\n'
                     '{"prompt": "Capital of France?", "answer": "Paris"}\n')
    injection = {"concept": "kindness", "vectorArtifactID": "runs/x/kindness",
                 "layer": 3, "alpha": 0.1}
    manifest = Manifest.from_dict({
        "name": "policy-with-steering", "modelID": "org/m",
        "variantConditions": [
            _agent("agent-plain"),
            _agent("agent-policy-only", policies=policies),
            _agent("agent-policy-and-vector", policies=policies,
                   injections=[injection])]})

    # The static part is the agent's own vector; building it needs the
    # vector's files, which this fixture does not carry. A stand-in returns
    # no hooks, and refuses a variant that still carries its policies, as
    # the real injection path does.
    def variant_injections(variant, **kwargs):
        assert not variant.intervention_policies
        return []

    monkeypatch.setattr(_owner_model_variant, "variant_injections", variant_injections)
    monkeypatch.setattr(_owner_generate, "generate",
                        lambda model, prompt, **kwargs:
                        "4" if "2+2" in prompt else "Rome")
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {})

    run_directory = _owner_validation_workflow._validate_impl(
        "policy-with-steering", manifest, object(), root, lambda *a: None)
    with open(os.path.join(run_directory, "validation-evidence.json"),
              encoding="utf-8") as handle:
        written = handle.read()
    evidence = json.loads(written)

    rows = {row["condition"]: row for row in evidence["batteryResults"]}
    assert list(rows) == ["baseline", "agent-plain", "agent-policy-only",
                          "agent-policy-and-vector"]
    battery_hash = rows["baseline"]["batteryHash"]
    # A policy agent with no static steering: the plain not-applicable row.
    assert rows["agent-policy-only"] == {
        "condition": "agent-policy-only", "batteryHash": battery_hash,
        "notApplicable": "interventionPolicy"}
    # A policy agent with a vector: the same row, plus the reading of the
    # vector alone in its own sub-object. The row itself has no score keys.
    mixed = rows["agent-policy-and-vector"]
    assert set(mixed) == {"condition", "batteryHash", "notApplicable",
                          "withoutPolicy"}
    assert mixed["withoutPolicy"]["correct"] == 1
    assert mixed["withoutPolicy"]["total"] == 2
    assert mixed["withoutPolicy"]["accuracy"] == 0.5
    assert "condition" not in mixed["withoutPolicy"]

    golden_files.check(FIXTURE, written, variable=golden_files.CROSS_ENGINE,
                       test_file="tests/test_battery_without_policy_fixture.py",
                       label="validation-evidence-without-policy.json")
