"""A validation-evidence file with one ERROR row, written by this engine.

When an agent cannot be loaded during validation, the battery records an
``error`` row for its condition, with no score keys, and carries on with the
other conditions. The Mac side used to reject the whole evidence file over
that one row, so a study with one broken agent appeared to have no validation
evidence at all.

This module produces the file the Mac side must read. It is producer-generated
on purpose (``test_portability_contracts.py`` explains the idiom): the bytes
committed under ``Tests/Fixtures/cross-engine/`` come from the real validation
code, so the Mac test cannot pin its author's belief about what this engine
writes. The structural assertions always run against the fresh document; the
byte comparison waits for the file to be committed. To regenerate, delete the
fixture and re-run this module.

Swift twin: ``Tests/ExperimentKitTests/BatteryEvidenceRowTests.swift``.
"""

import json
import os

import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.model_variant as _owner_model_variant
import steerlab_server.experiment.validation_workflow as _owner_validation_workflow
import steerlab_server.experiment.vector_materialization as _owner_vector_materialization
from steerlab_server.experiment.manifest import Manifest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(
    REPO, "Tests", "Fixtures", "cross-engine", "validation-evidence-error-row.json")

REGENERATE = ("stale fixture — delete it and re-run "
              "`python -m pytest tests/test_battery_error_row_fixture.py`, "
              "then commit")

#: What the engine says when an adapter agent cannot be loaded because the
#: adapter library is not installed. Raised here by a stand-in so the fixture
#: does not depend on what this test environment happens to have installed;
#: the wording is the engine's own (``model_variant.apply_adapter``).
LOAD_FAILURE = "variant adapters need peft: pip install -e .[lora]"


def _agent(name):
    return {"name": name, "artifactPath": "", "artifactHash": "x",
            "artifact": {"name": name, "baseModelID": "org/m", "adapters": [],
                         "injections": [], "temperature": 0.0,
                         "promptMode": "chatAssistant"}}


def test_validation_evidence_with_one_error_row_is_current(tmp_path, monkeypatch):
    root = str(tmp_path)
    battery = os.path.join(root, "prompts", "batteries", "basic.jsonl")
    os.makedirs(os.path.dirname(battery))
    with open(battery, "w", encoding="utf-8") as handle:
        handle.write('{"prompt": "2+2?", "answer": "4"}\n'
                     '{"prompt": "Capital of France?", "answer": "Paris"}\n')
    manifest = Manifest.from_dict({
        "name": "one-broken-agent", "modelID": "org/m",
        "variantConditions": [_agent("agent-ok"), _agent("agent-broken")]})

    real_apply = _owner_model_variant.apply_adapter

    def apply_adapter(model, variant, **kwargs):
        if variant.name == "agent-broken":
            raise RuntimeError(LOAD_FAILURE)
        return real_apply(model, variant, **kwargs)

    monkeypatch.setattr(_owner_model_variant, "apply_adapter", apply_adapter)
    monkeypatch.setattr(_owner_generate, "generate",
                        lambda model, prompt, **kwargs:
                        "4" if "2+2" in prompt else "Paris")
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {})

    run_directory = _owner_validation_workflow._validate_impl(
        "one-broken-agent", manifest, object(), root, lambda *a: None)
    with open(os.path.join(run_directory, "validation-evidence.json"),
              encoding="utf-8") as handle:
        written = handle.read()
    evidence = json.loads(written)

    # The shape the fixture promises the Mac side, asserted on the FRESH
    # document so a change here fails before any bytes are compared.
    rows = {row["condition"]: row for row in evidence["batteryResults"]}
    assert list(rows) == ["baseline", "agent-ok", "agent-broken"]
    assert rows["baseline"]["accuracy"] == 1.0
    assert rows["agent-ok"]["accuracy"] == 1.0
    # The error row: the condition, the battery it would have been scored on,
    # the engine's account of what went wrong — and NO score keys.
    assert rows["agent-broken"] == {
        "condition": "agent-broken",
        "batteryHash": rows["baseline"]["batteryHash"],
        "error": LOAD_FAILURE}
    assert evidence["task"] == "validate" and evidence["schemaVersion"] == 1

    if not os.path.exists(FIXTURE):
        with open(FIXTURE, "w", encoding="utf-8") as handle:
            handle.write(written)
        return
    with open(FIXTURE, encoding="utf-8") as handle:
        assert written == handle.read(), REGENERATE
