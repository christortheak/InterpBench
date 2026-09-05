"""Lifecycle decisions with supplied values, without a filesystem or model."""
from copy import deepcopy

import pytest

from steerlab_server.experiment import draft_protocol_policy, freeze_policy
from steerlab_server.experiment import manifest_mutation_policy as mutation
from steerlab_server.experiment.manifest_errors import ExperimentStoreError


def test_clearing_intent_does_not_override_frozen_immutability():
    existing = {"name": "study", "status": "draft", "conditions": [{"name": "arm"}]}
    proposed = {"name": "study", "status": "draft", "conditions": []}
    with pytest.raises(ExperimentStoreError) as failure:
        mutation.admit_save(proposed, existing)
    assert failure.value.gate == "armsCleared"
    mutation.admit_save(proposed, existing, clearing_arms=True)
    existing["status"] = "frozen"
    with pytest.raises(ExperimentStoreError) as failure:
        mutation.admit_save(proposed, existing, clearing_arms=True)
    assert failure.value.gate == "statusImmutable"


def test_draft_sync_distinguishes_omitted_and_explicitly_cleared_pins():
    old = {"modelRevision": "abc", "capabilityBatteryFile": "battery", "capabilityBatteryHash": "hash"}
    proposed = {"modelRevision": None}
    preserved = mutation.merge_server_pins(proposed, old)
    assert proposed["modelRevision"] is None
    assert proposed["capabilityBatteryHash"] == "hash"
    assert "modelRevision" not in preserved
    assert old["modelRevision"] == "abc"


def test_protocol_refusal_leaves_the_supplied_draft_unmodified():
    draft = {"name": "study", "modelID": "model", "temperature": 0.2}
    before = deepcopy(draft)
    with pytest.raises(ExperimentStoreError):
        draft_protocol_policy.apply_protocol("study", draft, {"temperature": 0.5, "samplesPerItem": 0})
    assert draft == before
    draft_protocol_policy.apply_protocol("study", draft, {"temperature": None})
    assert "temperature" not in draft


def test_panel_freeze_ignores_carried_model_output_evidence_but_keeps_judge_and_git_gates():
    draft = {"modelID": "model", "modelRevision": "abc", "studyKind": "multiAgent",
             "conditions": [{"name": "carried"}], "variantConditions": [{"name": "carried"}]}
    facts = freeze_policy.FreezeEvidence(jlens="lens", variant="variant", judge="judge",
                                         battery="battery", git="git")
    assert freeze_policy.evaluate("study", draft, facts) == [
        ("judgeValidity", "judge"), ("gitClean", "git")]


def test_validation_presence_does_not_erase_a_vacuous_probe():
    draft = {"modelID": "model", "modelRevision": "abc", "conditions": [{"name": "arm"}]}
    facts = freeze_policy.FreezeEvidence(validation_present=True, vacuous_validation="no probe")
    assert freeze_policy.evaluate("study", draft, facts) == [("validateEvidence", "no probe")]
    assert freeze_policy.evaluate("study", draft, freeze_policy.FreezeEvidence(validation_present=True)) == []


def test_force_skips_evidence_refusal_without_changing_failure_records_or_status_admission():
    failures = [("measurementPins", "first"), ("judgeValidity", "second"), ("measurementPins", "third")]
    before = list(failures)
    with pytest.raises(ExperimentStoreError) as failure:
        freeze_policy.admit_failures("study", failures, force=False)
    assert failure.value.gate == "measurementPins"
    assert str(failure.value) == "first"
    assert set(failure.value.gates) == {"measurementPins", "judgeValidity"}
    freeze_policy.admit_failures("study", failures, force=True)
    assert failures == before
    with pytest.raises(ExperimentStoreError) as failure:
        mutation.admit_freeze("study", {"status": "frozen"})
    assert failure.value.gate == "statusImmutable"
