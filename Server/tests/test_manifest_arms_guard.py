"""open-issues §8 — a manifest must not lose its whole measured surface in one
silent write.

The reported symptom (`s4x-kz-s1-noreasons`: a workspace manifest with
`concepts: []` + `conditions: []` beside a run snapshot carrying 16 of each)
turned out NOT to be a rewrite — that workspace file was born a shell and never
touched again. But the census of writers found two paths on this engine that
CAN produce exactly that shape over a populated draft, and the reproductions
below are those paths. The guard is the belt-and-suspenders layer: the
transition is refused at the single save chokepoint unless the caller declares
it.

Swift twin: `Tests/ExperimentKitTests/ManifestArmsGuardTests.swift`.
"""

import json
import os

import pytest

from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment import lifecycle_gates


def _concept(root, name="fair"):
    d = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "positive.jsonl"), "w", encoding="utf-8").write('{"text":"fair"}\n')
    open(os.path.join(d, "negative.jsonl"), "w", encoding="utf-8").write('{"text":"unfair"}\n')


def _armed_study(root, name="armed-study"):
    """A draft with both a concept and a condition — the state §8's manifest
    was in on the engine that ran it."""
    _concept(root)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fair"], root=root)
    es.add_condition(name, {"name": "fair-L5", "slots": [
        {"concept": "fair", "layer": 5, "alpha": 1.0}]}, root=root)
    return es.load_raw(name, root)


def _shell_of(document):
    """The same manifest with its arms gone — a stale/skeleton document."""
    shell = dict(document)
    shell["concepts"] = []
    shell["conditions"] = []
    return shell


# ---------------------------------------------------------------- reproduction


def test_server_draft_sync_cannot_strip_every_arm(tmp_path):
    """REPRODUCTION 1 — `PUT /api/experiment/{name}/manifest`.

    `_merge_server_pins` preserves the revision pin, the battery pin, and
    `*-recommended` conditions; it never preserved `concepts`, and a
    hand-declared condition is not `*-recommended`. So a push of a document
    that simply never saw the arms silently installed a shell over them. Before
    the guard this call returned a happy `{"status": "draft"}`."""
    root = str(tmp_path)
    armed = _armed_study(root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.replace_draft_manifest("armed-study", _shell_of(armed), root=root)
    assert exc.value.gate == lifecycle_gates.ARMS_CLEARED
    assert "steerlab-cli" in exc.value.repair_action
    # The disk copy is intact — the refusal happened before any write.
    after = es.load_raw("armed-study", root)
    assert len(after["concepts"]) == 1
    assert len(after["conditions"]) == 1


def test_bundle_import_cannot_stomp_a_draft_that_gained_arms(tmp_path):
    """REPRODUCTION 2 — `import_bundle(..., allow_overwrite=True)`, which is
    what `_execute_run_bundle_inner` passes.

    The frozen-manifest firewall right above this check answers False for a
    draft, so a bundle packaged from a skeleton could take a workspace draft
    that had since been attached to back down to both-empty, verbatim bytes, no
    diff, no warning."""
    source = str(tmp_path / "source")
    _concept(source)
    es.create("shell-study", model_id="org/m", revision="abc", root=source)
    es.add_condition("shell-study", {"name": "placeholder", "slots": []}, root=source)
    meta = bundles.package_experiment("shell-study", root=source)
    # Now strip the source manifest to a true shell and repackage it.
    shell = es.load_raw("shell-study", source)
    shell["conditions"] = []
    es.save_raw(shell, source, clearing_arms=True)
    meta = bundles.package_experiment("shell-study", root=source,
                                      output_path=str(tmp_path / "shell.tar.gz"))

    target = str(tmp_path / "target")
    bundles.import_bundle(meta["bundlePath"], target_root=target)
    # The target's copy is then attached to — the workspace moved on.
    _concept(target)
    es.attach("shell-study", ["fair"], root=target)
    es.add_condition("shell-study", {"name": "fair-L5", "slots": [
        {"concept": "fair", "layer": 5, "alpha": 1.0}]}, root=target)

    with pytest.raises(bundles.BundleError, match="no concepts and no conditions"):
        bundles.import_bundle(meta["bundlePath"], target_root=target,
                              allow_overwrite=True)
    after = es.load_raw("shell-study", target)
    assert len(after["concepts"]) == 1
    assert len(after["conditions"]) == 1


# ---------------------------------------------------------------------- guard


def test_the_guard_refuses_the_transition_at_the_save_chokepoint(tmp_path):
    root = str(tmp_path)
    armed = _armed_study(root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.save_raw(_shell_of(armed), root)
    assert exc.value.gate == lifecycle_gates.ARMS_CLEARED
    # The refusal names what is being lost, so a reader can tell a stale
    # document from an intended reset without opening the file.
    assert "1 concept(s)" in str(exc.value)
    assert "1 condition(s)" in str(exc.value)


def test_losing_only_the_concepts_is_not_the_guarded_transition(tmp_path):
    """The rule is BOTH-empty, deliberately. A study that drops its concepts
    but keeps its conditions still has a measured surface, and narrowing an
    arm set is ordinary authoring."""
    root = str(tmp_path)
    armed = _armed_study(root)
    partial = dict(armed)
    partial["concepts"] = []
    es.save_raw(partial, root)
    assert es.load_raw("armed-study", root)["conditions"]


def test_a_new_empty_manifest_is_allowed(tmp_path):
    """Creation is the legitimate both-empty write: there is no transition,
    because there is nothing on disk to lose."""
    root = str(tmp_path)
    d = es.create("fresh", model_id="org/m", root=root)
    assert d["concepts"] == [] and d["conditions"] == []
    # …and re-saving a still-empty draft (every setter before the first
    # attach: pin-prompts, set-instruments, set-protocol) stays allowed.
    d["temperature"] = 0.7
    es.save_raw(d, root)
    assert es.load_raw("fresh", root)["temperature"] == 0.7


def test_attach_before_any_condition_is_allowed(tmp_path):
    root = str(tmp_path)
    _concept(root)
    es.create("attaching", model_id="org/m", root=root)
    d = es.attach("attaching", ["fair"], root=root)
    assert d["concepts"] and d["conditions"] == []


def test_removing_the_last_condition_declares_its_intent(tmp_path):
    """THE intentional clear-all flow on this engine. It goes to both-empty and
    must keep working — it is a researcher deleting one arm they named."""
    root = str(tmp_path)
    es.create("one-arm", model_id="org/m", root=root)
    es.add_condition("one-arm", {"name": "solo", "slots": []}, root=root)
    d = es.remove_condition("one-arm", "solo", root=root)
    assert d["conditions"] == []
    assert es.load_raw("one-arm", root)["conditions"] == []


def test_the_explicit_flag_is_the_only_way_through(tmp_path):
    root = str(tmp_path)
    armed = _armed_study(root)
    es.save_raw(_shell_of(armed), root, clearing_arms=True)
    after = es.load_raw("armed-study", root)
    assert after["concepts"] == [] and after["conditions"] == []


def test_a_frozen_manifest_still_answers_status_immutable(tmp_path):
    """Frozen manifests are read-only for a different reason and must keep
    saying so — an agent switching on `statusImmutable` must not start seeing
    `armsCleared` instead."""
    root = str(tmp_path)
    _armed_study(root, "to-freeze")
    es.freeze("to-freeze", force=True, root=root)
    frozen = es.load_raw("to-freeze", root)
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.save_raw(_shell_of(frozen), root)
    assert exc.value.gate == lifecycle_gates.STATUS_IMMUTABLE


def test_the_freeze_transition_is_not_caught_by_the_guard(tmp_path):
    """`freeze_transition=True` skips the existing-copy read entirely; the
    guard must live inside that branch rather than beside it, or freezing a
    legitimately arm-less multi-agent study would start refusing."""
    root = str(tmp_path)
    es.create("mas", model_id="org/m", revision="abc", root=root)
    d = es.load_raw("mas", root)
    d["status"] = "frozen"
    es.save_raw(d, root, freeze_transition=True)
    assert es.load_raw("mas", root)["status"] == "frozen"


def test_the_gate_id_is_in_the_closed_vocabulary():
    assert lifecycle_gates.ARMS_CLEARED in lifecycle_gates.LIFECYCLE_GATE_IDS
    assert (set(lifecycle_gates.LIFECYCLE_GATE_IDS)
            & set(es.FORCED_GATE_IDS)) == set()


def test_variant_only_incoming_is_not_a_disarm(tmp_path):
    """The guard's first false positive (caught at landing): a save whose
    whole measured surface lives in variantConditions — the agentComparison
    shape — must not be refused over a draft that held concepts/conditions."""
    root = str(tmp_path)
    document = _armed_study(root, name="meta-variant")
    incoming = _shell_of(document)
    incoming["variantConditions"] = [{
        "name": "agent-a", "variantPath": "runs/model-variants/agent-a.json",
    }]
    es.save_raw(incoming, root)  # must NOT raise: the surface moved, not died
    assert es.load_raw("meta-variant", root)["variantConditions"]
