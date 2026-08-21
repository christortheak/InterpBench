"""The foreign-local-judge pin gate (external review round 2, finding 3).

A local judge naming a model OTHER than the study model must pin the exact
bytes that will judge — `revision` and `dtype`. Without it, targeted retry
compares two sessions' recorded judge identities to decide whether earlier
verdicts may be REUSED, and `null == null` passes while the two sessions
loaded different defaults.

The gate shipped in 10adf47d8 with no direct coverage on either engine
(external review round 4, finding 1) — existing fixtures were merely edited
to survive it, which proves nothing about whether it fires. These are the
tests that prove it: refusal per missing field, acceptance when pinned,
silence for study-model judges, and force provenance.

Swift twin: `Tests/ExperimentKitTests/ForeignLocalJudgePinTests.swift`.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _study(root, name="s", *, judges, judged_sweep=False):
    """A judged study whose ONLY freeze problem can be the judge pins.

    ``judged_sweep`` declares a judgeScore sweep — the condition under which
    a study-model judge cannot pin a separate identity (round 5, finding 1).
    An operative sweep pins its dev prompts + battery at freeze, so those
    files must exist for the freeze attempt to reach the judge gate.
    """
    concept = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    d = es.load_raw(name, root)
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = _write(
        os.path.join(root, "prompts", "rubrics", "r.md"),
        "Judge which response is better.")
    d["judges"] = judges
    if judged_sweep:
        d["sweep"] = {"selection": {"objective": {"metric": "judgeScore"}}}
        _write(os.path.join(root, "prompts", "dev", "dev-prompts.jsonl"),
               '{"id": "d1", "prompt": "Describe the cellar."}\n')
        _write(os.path.join(root, "prompts", "batteries", "basic.jsonl"),
               '{"id": "b1", "prompt": "2+2?", "answer": "4"}\n')
    es.save_raw(d, root)
    _validate_evidence(root, name)
    return name


def _validate_evidence(root, name):
    from steerlab_server.experiment.manifest import Manifest
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"v-exp-{name}-validate")
    os.makedirs(rundir, exist_ok=True)
    json.dump({"schemaVersion": 1, "task": "validate",
               "substrate": "python-hf-transformers",
               "validationScopeHash": scope},
              open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"fear": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, "validation-report.json"), "w"))


def _foreign(**overrides):
    judge = {"name": "judge-2", "kind": "local", "model": "other/judge-12b"}
    judge.update(overrides)
    return [{"name": "judge-1", "kind": "local"}, judge]


# --- the gate fires, per missing field -----------------------------------------


def test_freeze_refuses_a_foreign_local_judge_missing_revision(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign(dtype="bfloat16"))
    with pytest.raises(es.ExperimentStoreError,
                       match="pin the exact bytes") as excinfo:
        es.freeze(name, root=root)
    message = str(excinfo.value)
    # Only the field actually absent is named as missing (the remedy
    # sentence later mentions both, which is correct).
    assert ("'judge-2' (model 'other/judge-12b') is missing revision."
            in message)


def test_freeze_refuses_a_foreign_local_judge_missing_dtype(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign(revision="cafe01"))
    with pytest.raises(es.ExperimentStoreError,
                       match="pin the exact bytes") as excinfo:
        es.freeze(name, root=root)
    assert ("'judge-2' (model 'other/judge-12b') is missing dtype"
            in str(excinfo.value))


def test_freeze_names_both_fields_when_both_are_absent(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign())
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.freeze(name, root=root)
    assert "is missing revision and dtype" in str(excinfo.value)


def test_whitespace_is_not_a_pin(tmp_path):
    """A field present but blank must not satisfy the gate — the recorded
    identity would still be empty at retry time."""
    root = str(tmp_path)
    name = _study(root, judges=_foreign(revision="   ", dtype="\t"))
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.freeze(name, root=root)
    assert "is missing revision and dtype" in str(excinfo.value)


def test_every_offender_is_named_not_just_the_first(tmp_path):
    root = str(tmp_path)
    judges = _foreign(revision="cafe01", dtype="bfloat16")
    judges.append({"name": "judge-3", "kind": "local", "model": "third/judge"})
    name = _study(root, judges=judges)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.freeze(name, root=root)
    assert "judge-3" in str(excinfo.value)


# --- the gate stays silent where it should -------------------------------------


def test_fully_pinned_foreign_local_judge_freezes(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign(revision="cafe01", dtype="bfloat16"))
    frozen = es.freeze(name, root=root)
    assert frozen["status"] == "frozen"
    assert not frozen.get("freezeForced")


def test_study_model_local_judges_need_no_pins(tmp_path):
    """A local judge resolving to the STUDY model inherits the study's
    pinned revision, so there is nothing for it to pin — both the blank
    form and the explicitly-named form.

    Asserted on the rule itself for the two-study-model panel, because
    FREEZING that panel is refused by a different gate (both judges are the
    same deterministic judge); the freeze half uses a distinct panel.
    """
    root = str(tmp_path)
    both = _study(root, "both", judges=[
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "local", "model": "org/m"},
    ])
    assert es.unpinned_foreign_local_judge_problem(
        es.load_raw(both, root)) is None

    name = _study(root, judges=[
        {"name": "judge-1", "kind": "local", "model": "org/m"},
        {"name": "judge-2", "kind": "openrouter",
         "model": "anthropic/claude-opus-4", "provider": "Anthropic"},
    ])
    assert es.freeze(name, root=root)["status"] == "frozen"


def test_external_judges_are_untouched(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=[
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "openrouter",
         "model": "anthropic/claude-opus-4", "provider": "Anthropic"},
    ])
    assert es.unpinned_foreign_local_judge_problem(
        es.load_raw(name, root)) is None


# --- draft visibility and force provenance -------------------------------------


def test_a_draft_shows_the_problem_before_freeze_refuses(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign())
    advisories = es.freeze_advisories(es.load_raw(name, root), root)
    assert any("pin the exact bytes" in a for a in advisories)


def test_force_freeze_skips_it_loudly_and_stamps_judge_validity(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign())
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["freezeForced"] is True
    assert "judgeValidity" in frozen["forcedGatesSkipped"]


# --- study-model judges cannot pin a different identity (round 5, F1) ---------


def _study_model_judge(**overrides):
    judge = {"name": "judge-1", "kind": "local", "model": "org/m"}
    judge.update(overrides)
    return [judge,
            {"name": "judge-2", "kind": "openrouter",
             "model": "anthropic/claude-opus-4", "provider": "Anthropic"}]


def test_freeze_refuses_a_study_model_judge_pinning_another_revision(tmp_path):
    """The sweep judges with the HELD weights and never loads a second
    revision, so the pin would be silently ignored — while `evaluate` DOES
    honor it. One manifest, two identities, depending on the verb."""
    root = str(tmp_path)
    name = _study(root, judges=_study_model_judge(revision="beef02"),
                  judged_sweep=True)
    with pytest.raises(es.ExperimentStoreError,
                       match="cannot pin a different identity") as excinfo:
        es.freeze(name, root=root)
    message = str(excinfo.value)
    assert "'judge-1' pins revision 'beef02'" in message
    assert "is pinned at 'abc'" in message


def test_freeze_refuses_a_study_model_judge_pinning_another_dtype(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_study_model_judge(dtype="float32"),
                  judged_sweep=True)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.freeze(name, root=root)
    # The study pins no dtype, so ANY declared dtype is a claim the sweep
    # cannot honor.
    assert "pins none (the device decides)" in str(excinfo.value)


def test_a_blank_model_judge_is_covered_too(tmp_path):
    """Blank model and explicit study model both resolve to the study
    model, so both must be checked."""
    root = str(tmp_path)
    name = _study(root, judges=_study_model_judge(model=None, revision="beef03"),
                  judged_sweep=True)
    with pytest.raises(es.ExperimentStoreError,
                       match="cannot pin a different identity"):
        es.freeze(name, root=root)


def test_pins_that_agree_with_the_study_are_legal(tmp_path):
    """Redundant, not wrong — the gate is about DIVERGENCE."""
    root = str(tmp_path)
    name = _study(root, judges=_study_model_judge(revision="abc"),
                  judged_sweep=True)
    assert es.freeze(name, root=root)["status"] == "frozen"


def test_a_foreign_judge_is_not_affected_by_this_gate(tmp_path):
    root = str(tmp_path)
    d = {"modelID": "org/m", "modelRevision": "abc",
         "sweep": {"selection": {"objective": {"metric": "judgeScore"}}},
         "judges": [
             {"name": "j", "kind": "local", "model": "other/judge-12b",
              "revision": "cafe01", "dtype": "bfloat16"}]}
    assert es.study_model_judge_pin_conflict(d) is None


def test_evaluate_only_studies_may_judge_with_another_checkpoint(tmp_path):
    """NOT a blanket rule: evaluate genuinely LOADS a declared judge
    revision, so a different checkpoint of the study repo is a legitimate
    judge there. Only a judgeScore sweep cannot honor it."""
    root = str(tmp_path)
    name = _study(root, judges=_study_model_judge(revision="beef04"))
    assert es.study_model_judge_pin_conflict(es.load_raw(name, root)) is None
    assert es.freeze(name, root=root)["status"] == "frozen"


# --- a pin must name a commit, not a moving ref (round 5, finding 4) ---------


@pytest.mark.parametrize("moving", [
    "main", "master", "HEAD", "refs/pr/1", "v1.0", "latest", "release-2026",
])
def test_moving_references_are_not_pins(moving):
    """A branch is re-pointed by definition and a tag can be moved, so
    neither identifies the bytes a run used — the old gate only required
    non-emptiness, and the loader recorded the symbolic name it was handed
    rather than the commit it resolved to."""
    assert not es._is_commit_like(moving)
    d = {"modelID": "org/m", "modelRevision": moving}
    problem = es.symbolic_revision_problem(d)
    assert problem is not None
    assert f"the study model pins '{moving}'" in problem


@pytest.mark.parametrize("commit", [
    "abc123", "cafe01", "0" * 40, "deadbeef", "ABC123",
])
def test_commit_hashes_pass(commit):
    assert es._is_commit_like(commit)
    assert es.symbolic_revision_problem(
        {"modelID": "org/m", "modelRevision": commit}) is None


def test_judge_revisions_are_checked_too(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign(revision="main", dtype="bfloat16"))
    with pytest.raises(es.ExperimentStoreError,
                       match="moving reference") as excinfo:
        es.freeze(name, root=root)
    message = str(excinfo.value)
    assert "judge 'judge-2' pins 'main'" in message
    # The remedy points at the thing that now works (round 5, finding 5).
    assert "Resolve button" in message


def test_force_freeze_stamps_it_under_the_revision_gate(tmp_path):
    root = str(tmp_path)
    name = _study(root, judges=_foreign(revision="main", dtype="bfloat16"))
    frozen = es.freeze(name, force=True, root=root)
    assert "revision" in frozen["forcedGatesSkipped"]


def test_an_absent_revision_is_not_this_gates_business():
    """Absence is the OTHER gate's concern (unpinned foreign judge); this
    one only judges the shape of a revision that exists."""
    assert es.symbolic_revision_problem({"modelID": "org/m"}) is None
    assert es.symbolic_revision_problem(
        {"modelID": "org/m", "modelRevision": ""}) is None
