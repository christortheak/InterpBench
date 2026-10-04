"""A study whose agent carries an intervention policy freezes without force.

The capability battery cannot run an intervention policy, so the
``batteryEvidence`` freeze gate could never be satisfied for such a condition
and ``--force`` — which stamps the whole study non-citable — was the only way
to freeze it. The gate now exempts exactly those conditions and nothing else:

* the frozen manifest carries ``capabilityBatteryNotApplied`` (a freeze stamp,
  outside the content hash and the canonical bytes) naming each exempted
  condition and the reason, and is NOT marked forced;
* baseline and every condition without a policy owe battery evidence exactly
  as before, and a not-applicable row never stands in for a score;
* the forced path and its two stamps are unchanged;
* validation writes an explicit not-applicable battery row for the condition
  instead of an error row, and a run's report records it too.

Swift twin: ``PolicyAgentFreezeTests``.
"""

import hashlib
import json
import os

import pytest

import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.validation_workflow as _owner_validation_workflow
from steerlab_server.client import design_identity
from steerlab_server.experiment import battery as battery_mod
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import freeze_policy, manifest_diff, run_reporting
from steerlab_server.experiment.manifest import Manifest
from test_intervention_policies import attached, setup

POLICY = "policy-agent"
PLAIN = "plain-agent"
SENTENCE = (
    "The capability battery was not applied to policy-agent, because its "
    "agent uses an intervention policy, which the battery cannot run. This "
    "study has no capability control for that agent.")
STAMP = [{"condition": POLICY, "reason": "interventionPolicy"}]


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload, indent=1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _condition(root, name, *, policies=None):
    """One variant condition with its agent pinned inline and on disk."""
    artifact = {"name": name, "baseModelID": "org/m", "adapters": [],
                "injections": [], "temperature": 0.0,
                "promptMode": "chatAssistant"}
    if policies is not None:
        artifact["interventionPolicies"] = policies
    rel = f"runs/model-variants/{name}/model-variant.json"
    digest = _write(os.path.join(root, rel), artifact)
    return {"name": name, "artifactPath": rel, "artifactHash": digest,
            "artifact": artifact}


def _study(tmp_path, name="policy-study", *, conditions=(POLICY,)):
    """A model-output study whose variant conditions are the named agents;
    ``policy-agent`` carries a real, published intervention policy."""
    root = str(tmp_path)
    doc, _, _ = setup(tmp_path)
    policies = attached(doc, tmp_path)
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["variantConditions"] = [
        _condition(root, c, policies=policies if c == POLICY else None)
        for c in conditions]
    es.save_raw(d, root)
    return root, name


def _scored(condition, battery_hash):
    return {"condition": condition, "batteryHash": battery_hash,
            "total": 4, "correct": 4, "accuracy": 1.0}


def _not_applicable(condition, battery_hash):
    return {"condition": condition, "batteryHash": battery_hash,
            "notApplicable": "interventionPolicy"}


def _validate_evidence(root, name, rows, stamp="v"):
    """Scope-matched validate evidence carrying exactly ``rows`` — each a
    callable taking the live battery hash."""
    scope = Manifest.load(name, root=root).validation_scope_hash()
    battery_hash = battery_mod.live_hash(battery_mod.DEFAULT_BATTERY_FILE, root)
    rundir = os.path.join(root, "runs", f"{stamp}-exp-{name}-validate")
    os.makedirs(rundir, exist_ok=True)
    with open(os.path.join(rundir, "validation-evidence.json"), "w") as handle:
        json.dump({"schemaVersion": 1, "task": "validate",
                   "substrate": "python-hf-transformers",
                   "validationScopeHash": scope,
                   "batteryResults": [row(battery_hash) for row in rows]},
                  handle)
    with open(os.path.join(rundir, "validation-report.json"), "w") as handle:
        json.dump({"concepts": {}}, handle)


def _baseline(h):
    return _scored("baseline", h)


# --- the gate and the stamp -------------------------------------------------

def test_policy_agent_study_freezes_without_force_and_carries_the_stamp(tmp_path):
    root, name = _study(tmp_path)
    _validate_evidence(root, name, [
        _baseline, lambda h: _not_applicable(POLICY, h)])

    frozen = es.freeze(name, force=False, root=root)

    assert frozen["status"] == "frozen"
    # Not forced: neither force stamp exists, so nothing marks it non-citable.
    assert "freezeForced" not in frozen
    assert "forcedGatesSkipped" not in frozen
    # The honest stamp: which condition, and why.
    assert frozen[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP
    # It is a freeze stamp like the others — outside the canonical bytes and
    # the content hash — so the frozen study verifies clean.
    canonical = os.path.join(root, "experiments", name, "freeze-canonical.json")
    with open(canonical, "rb") as handle:
        blob = handle.read()
    assert hashlib.sha256(blob).hexdigest() == frozen["freezeHash"]
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in json.loads(blob)
    assert Manifest.load(name, root=root).verify(root) == []
    # The stamp is on disk, not only in the returned dict.
    assert es.load_raw(name, root)[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP


def test_the_gate_does_not_need_a_row_for_the_exempt_condition(tmp_path):
    # Evidence that predates the not-applicable row (or came from an engine
    # that wrote an error row) still freezes: the exemption is decided by the
    # manifest, not by what the evidence happens to say.
    root, name = _study(tmp_path)
    _validate_evidence(root, name, [
        _baseline,
        lambda h: {"condition": POLICY, "batteryHash": h, "error": "skipped"}])
    frozen = es.freeze(name, force=False, root=root)
    assert frozen[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP


def test_freeze_says_why_in_the_advisories_and_the_settings_summary(tmp_path, capsys):
    root, name = _study(tmp_path)
    _validate_evidence(root, name, [_baseline])
    # A draft already says it, before the one-way step.
    assert SENTENCE in es.freeze_advisories(es.load_raw(name, root), root)

    frozen = es.freeze(name, force=False, root=root)

    assert SENTENCE in es.freeze_advisories(frozen, root)
    assert f"freeze '{name}' advisory: {SENTENCE}" in capsys.readouterr().err
    summary = os.path.join(root, "experiments", name, "preregistration.md")
    text = open(summary, encoding="utf-8").read()
    assert "## Capability battery" in text
    assert f"- {SENTENCE}" in text


def test_a_plain_agent_beside_a_policy_agent_still_needs_battery_evidence(tmp_path):
    root, name = _study(tmp_path, conditions=(POLICY, PLAIN))
    _validate_evidence(root, name, [
        _baseline, lambda h: _not_applicable(POLICY, h)])
    with pytest.raises(es.ExperimentStoreError) as refused:
        es.freeze(name, force=False, root=root)
    assert refused.value.gate == "batteryEvidence"
    # The refusal names the condition that owes evidence, and only that one.
    assert PLAIN in str(refused.value)
    assert POLICY not in str(refused.value)

    # With the plain agent scored, the study freezes and the stamp names only
    # the policy agent.
    _validate_evidence(root, name, [
        _baseline, lambda h: _not_applicable(POLICY, h),
        lambda h: _scored(PLAIN, h)], stamp="w")
    frozen = es.freeze(name, force=False, root=root)
    assert frozen[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP
    assert "freezeForced" not in frozen


def test_a_not_applicable_row_never_stands_in_for_a_score(tmp_path):
    # Validate evidence is matched by scope, which does not cover conditions:
    # a row written while a condition's agent carried a policy must not
    # satisfy the gate for an agent the battery CAN run.
    root, name = _study(tmp_path, conditions=(PLAIN,))
    _validate_evidence(root, name, [
        _baseline, lambda h: _not_applicable(PLAIN, h)])
    with pytest.raises(es.ExperimentStoreError) as refused:
        es.freeze(name, force=False, root=root)
    assert refused.value.gate == "batteryEvidence"
    assert PLAIN in str(refused.value)


def test_baseline_evidence_is_still_required(tmp_path):
    # The exemption is for the policy condition only. With no validate run at
    # all the study is refused exactly as any variant study is.
    root, name = _study(tmp_path)
    with pytest.raises(es.ExperimentStoreError) as refused:
        es.freeze(name, force=False, root=root)
    assert set(refused.value.gates) == {"validateEvidence", "batteryEvidence"}
    # A validate run that scored nothing for baseline is refused too.
    _validate_evidence(root, name, [lambda h: _not_applicable(POLICY, h)])
    with pytest.raises(es.ExperimentStoreError) as refused:
        es.freeze(name, force=False, root=root)
    assert refused.value.gate == "batteryEvidence"
    assert "baseline" in str(refused.value)


def test_battery_drift_is_still_checked_for_the_conditions_that_owe_evidence(tmp_path):
    root, name = _study(tmp_path)
    battery_hash = _write(
        os.path.join(root, "prompts", "batteries", "basic.jsonl"),
        '{"prompt": "2+2?", "answer": "4"}\n')
    d = es.load_raw(name, root)
    d["capabilityBatteryFile"] = "prompts/batteries/basic.jsonl"
    d["capabilityBatteryHash"] = battery_hash
    es.save_raw(d, root)
    # Evidence scored against a DIFFERENT battery than the pin.
    _validate_evidence(root, name, [
        lambda h: _scored("baseline", "00" * 32),
        lambda h: _not_applicable(POLICY, "00" * 32)])
    with pytest.raises(es.ExperimentStoreError, match="battery drifted") as refused:
        es.freeze(name, force=False, root=root)
    assert refused.value.gate == "batteryEvidence"
    assert POLICY not in str(refused.value)
    # Evidence from the pinned battery freezes.
    _validate_evidence(root, name, [
        lambda h: _scored("baseline", battery_hash),
        lambda h: _not_applicable(POLICY, battery_hash)], stamp="w")
    assert es.freeze(name, force=False, root=root)["status"] == "frozen"


# --- the forced path is unchanged -------------------------------------------

def test_forced_freeze_of_a_plain_variant_study_is_unchanged(tmp_path):
    root, name = _study(tmp_path, conditions=(PLAIN,))
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["freezeForced"] is True
    assert frozen["forcedGatesSkipped"] == ["validateEvidence", "batteryEvidence"]
    # No policy agent, so no stamp: this study's frozen bytes are what they
    # always were.
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in frozen
    assert "forced freeze — gates skipped: validateEvidence, batteryEvidence " \
        "— non-citable" in es.freeze_advisories(frozen, root)


def test_force_keeps_its_meaning_on_a_policy_agent_study(tmp_path):
    # Force still skips, and still stamps, the gates that would have failed —
    # here the missing validate run and baseline's missing battery evidence.
    # The not-applied stamp is written beside the force stamps, not instead.
    root, name = _study(tmp_path)
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["freezeForced"] is True
    assert frozen["forcedGatesSkipped"] == ["validateEvidence", "batteryEvidence"]
    assert frozen[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP


# --- additive schema ---------------------------------------------------------

def test_the_stamp_is_additive_and_outside_every_identity(tmp_path):
    root, name = _study(tmp_path)
    _validate_evidence(root, name, [_baseline])
    draft = es.load_raw(name, root)
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in draft
    draft_hash = Manifest.from_dict(draft).content_hash()

    frozen = es.freeze(name, force=False, root=root)

    # A manifest without the key loads; one with it loads and hashes the same.
    reloaded = Manifest.load(name, root=root)
    assert reloaded.raw[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP
    assert reloaded.content_hash() == frozen["freezeHash"]
    without = {k: v for k, v in frozen.items()
               if k != freeze_policy.BATTERY_NOT_APPLIED_KEY}
    assert Manifest.from_dict(without).content_hash() == frozen["freezeHash"]
    # (the draft hash differs only by what freeze itself pins)
    assert isinstance(draft_hash, str)
    # The diff and the portable design identity ignore it, like every stamp.
    assert manifest_diff.flattened(frozen) == manifest_diff.flattened(without)
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY in design_identity.LIFECYCLE
    # Duplicate starts a fresh draft with no stamp.
    copy = es.duplicate(name, "policy-study-v2", root=root)
    assert copy["status"] == "draft"
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in copy


def test_a_panel_study_carrying_a_policy_agent_gets_no_stamp(tmp_path):
    # The battery gate is not asked of a multi-agent study at all, so there
    # is nothing to exempt and nothing to stamp.
    root, name = _study(tmp_path)
    d = es.load_raw(name, root)
    d["studyKind"] = "multiAgent"
    assert freeze_policy.battery_not_applied(d) == []
    d["studyKind"] = "modelOutput"
    assert freeze_policy.battery_not_applied(d) == STAMP


def test_forward_referenced_and_plain_conditions_are_not_exempt():
    plain = {"name": "a", "artifact": {"injections": []}}
    forward = {"name": "b", "fromPromotion": {"concept": "c"},
               "artifact": {"interventionPolicies": [{}]}}
    empty = {"name": "c", "artifact": {"interventionPolicies": []}}
    for vc in (plain, forward, empty, {"name": "d"}, None):
        assert freeze_policy.battery_exemption_reason(vc) is None
    assert freeze_policy.battery_exemption_reason(
        {"name": "e", "artifact": {"interventionPolicies": [{}]}}
    ) == "interventionPolicy"


# --- the evidence rows --------------------------------------------------------

def test_validation_writes_a_not_applicable_row_for_a_policy_agent(tmp_path, monkeypatch):
    root, name = _study(tmp_path, conditions=(POLICY, PLAIN))
    battery_hash = _write(
        os.path.join(root, "prompts", "batteries", "basic.jsonl"),
        '{"prompt": "2+2?", "answer": "4"}\n')
    d = es.load_raw(name, root)
    d["capabilityBatteryFile"] = "prompts/batteries/basic.jsonl"
    d["capabilityBatteryHash"] = battery_hash
    manifest = Manifest.from_dict(d)
    monkeypatch.setattr(_owner_generate, "generate", lambda *a, **k: "4")
    logged = []

    rows = _owner_validation_workflow._battery_results(
        manifest, object(), root, logged.append)

    by_condition = {r["condition"]: r for r in rows}
    # An explicit not-applicable row: no score keys, and not an error.
    assert by_condition[POLICY] == {
        "condition": POLICY, "batteryHash": battery_hash,
        "notApplicable": "interventionPolicy"}
    # Baseline and the plain agent are scored exactly as before.
    assert by_condition["baseline"]["accuracy"] == 1.0
    assert by_condition[PLAIN]["accuracy"] == 1.0
    assert f"battery: {SENTENCE}" in logged

    # The row the validation wrote is the row the gate reads.
    scope = manifest.validation_scope_hash()
    rundir = os.path.join(root, "runs", f"v-exp-{name}-validate")
    os.makedirs(rundir)
    with open(os.path.join(rundir, "validation-evidence.json"), "w") as handle:
        json.dump({"schemaVersion": 1, "task": "validate",
                   "substrate": "python-hf-transformers",
                   "validationScopeHash": scope, "batteryResults": rows}, handle)
    with open(os.path.join(rundir, "validation-report.json"), "w") as handle:
        json.dump({"concepts": {}}, handle)
    es.save_raw(d, root)
    frozen = es.freeze(name, force=False, root=root)
    assert frozen[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP


def test_the_run_report_records_the_condition_the_battery_did_not_score(tmp_path):
    root, name = _study(tmp_path, conditions=(POLICY, PLAIN))
    manifest = Manifest.load(name, root=root)
    scored = {"baseline": {"accuracy": 1.0, "itemCount": 2, "batteryHash": "h"},
              PLAIN: {"accuracy": 0.5, "itemCount": 2, "batteryHash": "h"}}

    def report(directory, battery):
        path = os.path.join(root, "runs", directory)
        os.makedirs(path)
        run_reporting.write_report(name, manifest, [], path, battery=battery)
        with open(os.path.join(path, "report.json"), encoding="utf-8") as handle:
            return json.load(handle)

    with_battery = report("with-battery", scored)
    assert with_battery[freeze_policy.BATTERY_NOT_APPLIED_KEY] == STAMP
    # The scored blocks keep their shape, and the policy agent gets no
    # scoreless block a reader could mistake for a measured zero.
    assert with_battery["conditions"][PLAIN]["capabilityBattery"] == scored[PLAIN]
    assert "capabilityBattery" not in with_battery["conditions"].get(POLICY, {})
    # No battery ran, so there is nothing to say about what it skipped.
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in report("no-battery", None)


def test_a_run_report_for_a_study_without_a_policy_agent_is_unchanged(tmp_path):
    root, name = _study(tmp_path, conditions=(PLAIN,))
    manifest = Manifest.load(name, root=root)
    path = os.path.join(root, "runs", "plain")
    os.makedirs(path)
    run_reporting.write_report(
        name, manifest, [], path,
        battery={"baseline": {"accuracy": 1.0, "itemCount": 2, "batteryHash": "h"}})
    with open(os.path.join(path, "report.json"), encoding="utf-8") as handle:
        assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in json.load(handle)
