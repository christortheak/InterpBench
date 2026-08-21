"""Force-freeze is loud, stamped, and engine-aligned (2026-07-13): every gate
is evaluated even under --force, each failing gate prints a warning naming its
id, the frozen manifest stamps ``freezeForced`` + ``forcedGatesSkipped``
(closed cross-engine id vocabulary), the pinned/ snapshot and workspace
auto-commit now run under force too, and freeze_advisories marks a forced
freeze non-citable. The two stamp keys stay OUT of the freeze-canonical
payload and the content hash, like every other freeze stamp."""

import hashlib
import json
import os
import subprocess

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _concept(root, name="fear"):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')


def _study(root, name="s", *, revision="abc"):
    _concept(root)
    es.create(name, model_id="org/m", revision=revision, root=root)
    es.attach(name, ["fear"], root=root)
    return name


def _validate_evidence(root, name, stamp="v"):
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"{stamp}-exp-{name}-validate")
    os.makedirs(rundir)
    json.dump({"schemaVersion": 1, "task": "validate",
               "substrate": "python-hf-transformers",
               "validationScopeHash": scope},
              open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"fear": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, "validation-report.json"), "w"))


def test_gate_id_vocabulary_is_closed():
    # THE cross-engine gate vocabulary. This literal is duplicated on
    # purpose: adding, removing, or renaming a gate id must fail THIS test
    # until the Swift twin (FreezeGateVocabularyTests.matchesServerLiteral,
    # asserting against FreezeGate.vocabulary) is updated in the same change.
    # Order is part of the contract — it is the order forcedGatesSkipped is
    # stamped in, and the order a refusal reports its `gates` in.
    assert es.FORCED_GATE_IDS == (
        "revision", "validateEvidence", "batteryEvidence", "judgeValidity",
        "variantValidity", "gitClean", "measurementPins")


def test_force_freeze_stamps_skipped_gates_and_warns(tmp_path, capsys):
    root = str(tmp_path)
    # Missing revision, no validate evidence, and a judged evaluation with no
    # rubric/panel: three gates fail; force must name all three.
    name = _study(root, revision=None)
    d = es.load_raw(name, root)
    d["evaluation"] = {"kind": "pairedJudge", "judgeModel": "claude-a",
                       "judgePrompt": "inline"}
    es.save_raw(d, root)
    frozen = es.freeze(name, force=True, cached_revision=lambda m: None,
                       root=root)
    assert frozen["status"] == "frozen"
    assert frozen["freezeForced"] is True
    # Historical gate order: revision → variantValidity → judgeValidity →
    # validateEvidence → batteryEvidence → gitClean.
    assert frozen["forcedGatesSkipped"] == [
        "revision", "judgeValidity", "validateEvidence"]
    # One loud warning line per skipped gate, naming its id.
    err = capsys.readouterr().err
    for gate_id in frozen["forcedGatesSkipped"]:
        assert f"skipping failing gate '{gate_id}'" in err
    # Every stamped id belongs to the closed vocabulary.
    assert set(frozen["forcedGatesSkipped"]) <= set(es.FORCED_GATE_IDS)


def test_force_freeze_with_missing_revision_stamps_revision_gate(tmp_path):
    root = str(tmp_path)
    name = _study(root, revision=None)
    frozen = es.freeze(name, force=True, cached_revision=lambda m: None,
                       root=root)
    assert "revision" in frozen["forcedGatesSkipped"]
    assert "validateEvidence" in frozen["forcedGatesSkipped"]


def test_force_freeze_with_all_gates_passing_stamps_empty_list(tmp_path):
    root = str(tmp_path)
    name = _study(root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["freezeForced"] is True
    assert frozen["forcedGatesSkipped"] == []


def test_unforced_freeze_carries_no_force_stamps(tmp_path):
    root = str(tmp_path)
    name = _study(root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, force=False, root=root)
    assert "freezeForced" not in frozen
    assert "forcedGatesSkipped" not in frozen


def test_canonical_bytes_exclude_force_stamps_and_hash_matches(tmp_path):
    root = str(tmp_path)
    name = _study(root, revision=None)
    frozen = es.freeze(name, force=True, cached_revision=lambda m: None,
                       root=root)
    canonical = os.path.join(root, "experiments", name, "freeze-canonical.json")
    with open(canonical, "rb") as handle:
        blob = handle.read()
    # sha256(canonical bytes) is still the freezeHash…
    assert hashlib.sha256(blob).hexdigest() == frozen["freezeHash"]
    payload = json.loads(blob)
    # …and the force stamps are excluded like every volatile freeze stamp.
    assert "freezeForced" not in payload
    assert "forcedGatesSkipped" not in payload
    # verify() of the frozen manifest stays clean: content_hash() must also
    # exclude the force stamps or the server-frozen drift check would fire.
    assert Manifest.load(name, root=root).verify(root) == []


def test_force_freeze_now_snapshots_pinned_inputs(tmp_path):
    root = str(tmp_path)
    name = _study(root, revision=None)
    es.freeze(name, force=True, cached_revision=lambda m: None, root=root)
    pinned = os.path.join(root, "experiments", name, "pinned")
    copied = os.path.join(pinned, "prompts", "concepts", "fear", "positive.jsonl")
    assert os.path.isfile(copied)
    assert open(copied, encoding="utf-8").read() == '{"text": "I feel dread"}\n'


def _git(root, *args):
    subprocess.run(["git", "-C", root, *args], check=True, capture_output=True,
                   env={"HOME": root, "PATH": os.environ["PATH"]})


def _git_out(root, *args):
    out = subprocess.run(["git", "-C", root, *args], check=True,
                         capture_output=True, text=True,
                         env={"HOME": root, "PATH": os.environ["PATH"]})
    return out.stdout.strip()


def test_force_freeze_auto_commits_standalone_workspace(tmp_path):
    root = str(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@t")
    name = _study(root, revision=None)
    frozen = es.freeze(name, force=True, cached_revision=lambda m: None,
                       root=root)
    # Two-commit lifecycle now happens under force too (Swift alignment):
    # the stamped gitCommit is the pre-stamp commit containing the pins.
    assert _git_out(root, "log", "-1", "--format=%s") == f"freeze {name} (stamp)"
    assert _git_out(root, "log", "-1", "--format=%s", "HEAD~1") == f"freeze {name}"
    assert frozen["gitCommit"] == _git_out(root, "rev-parse", "HEAD~1")
    assert _git_out(root, "status", "--porcelain") == ""
    tracked = set(_git_out(root, "ls-tree", "-r", "--name-only",
                           "HEAD").splitlines())
    assert f"experiments/{name}/pinned/prompts/concepts/fear/positive.jsonl" \
        in tracked


def test_freeze_advisories_flag_forced_freeze_as_non_citable(tmp_path):
    root = str(tmp_path)
    name = _study(root, revision=None)
    frozen = es.freeze(name, force=True, cached_revision=lambda m: None,
                       root=root)
    advisories = es.freeze_advisories(frozen, root)
    (forced,) = [a for a in advisories if "non-citable" in a]
    assert forced.startswith("forced freeze — gates skipped: ")
    for gate_id in frozen["forcedGatesSkipped"]:
        assert gate_id in forced


def test_duplicate_drops_force_stamps(tmp_path):
    root = str(tmp_path)
    name = _study(root, revision=None)
    es.freeze(name, force=True, cached_revision=lambda m: None, root=root)
    copy = es.duplicate(name, "copy", root=root)
    assert "freezeForced" not in copy
    assert "forcedGatesSkipped" not in copy
    assert copy["status"] == "draft"


def test_non_force_refusal_message_is_first_failing_gate(tmp_path):
    root = str(tmp_path)
    name = _study(root, revision=None)
    # revision is the first gate in the historical order.
    with pytest.raises(es.ExperimentStoreError, match="model revision not pinned"):
        es.freeze(name, force=False, cached_revision=lambda m: None, root=root)


def test_gate_id_survives_refusal(tmp_path):
    """WP0 step 3: the id the refusal path computed and threw away.

    Before this, ``freeze`` raised ``gate_failures[0][1]`` — prose only — so
    the gate id existed solely in the ``forcedGatesSkipped`` stamp, and a
    researcher who forced learned MORE about what was wrong than one who did
    not (audit §2.4 divergence 4). The message is unchanged; the id and the
    full failure list ride the structured fields.
    """
    root = str(tmp_path)
    name = _study(root, revision=None)
    with pytest.raises(es.ExperimentStoreError) as excinfo:
        es.freeze(name, force=False, cached_revision=lambda m: None, root=root)
    exc = excinfo.value
    # str() renders exactly the message it always did — same prose, same
    # exit code, for every caller that only prints it.
    assert str(exc) == (
        f"cannot freeze '{name}': model revision not pinned and org/m not in "
        "the local HF cache — load it once or freeze --force")
    assert exc.gate == "revision"
    # Every failing gate, in FORCED_GATE_IDS order — the same list, in the
    # same order, that a forced freeze stamps for the same manifest.
    assert exc.gates == ("revision", "validateEvidence")
    assert set(exc.gates) <= set(es.FORCED_GATE_IDS)
    forced = es.freeze(name, force=True, cached_revision=lambda m: None,
                       root=root)
    assert list(exc.gates) == forced["forcedGatesSkipped"]


def test_gateless_store_errors_are_unchanged():
    """The class is caught broadly, so a plain one-argument raise — every
    authoring refusal outside the freeze gates — keeps working untouched."""
    exc = es.ExperimentStoreError("cannot attach: no such concept")
    assert str(exc) == "cannot attach: no such concept"
    assert exc.gate is None
    assert exc.gates == ()
