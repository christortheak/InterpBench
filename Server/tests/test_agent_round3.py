"""FR6-B round 3: the revision rule refuses only commits that differ, a
repair that replaces an arm and clears the refusal, promotion at the level
the sweep ran at on every path, the editor's record of what an agent was
edited from, single-layer ablations, a check for panel seats, and the one
budget predicate.

Python twin of the round-3 cases in ``StudyAgentCreationTests``.
"""
import json
import os
import re
import shlex
from pathlib import Path

import pytest

from steerlab_server.experiment import (agent_identity, command_vocabulary as vocabulary,
                                        experiment_store as store, prompt_render,
                                        study_admission)
from steerlab_server.experiment.manifest import Manifest

import test_agent_create as base
from test_agent_round2 import QWEN, REVISION, study_with_arm

MODEL = base.MODEL
OTHER = "b" * 40


@pytest.fixture(autouse=True)
def restore_cli_root(monkeypatch):
    if 'STEERLAB_ROOT' in os.environ:
        monkeypatch.setenv('STEERLAB_ROOT', os.environ['STEERLAB_ROOT'])
    else:
        monkeypatch.delenv('STEERLAB_ROOT', raising=False)


# --- R1: the revision rule ------------------------------------------------------------


@pytest.mark.parametrize("agent, study, verdict", [
    ("a" * 40, "a" * 40, "match"),
    ("A" * 40, "a" * 40, "match"),
    ("a" * 40, OTHER, "differ"),
    ("main", "a" * 40, "unchecked"),
    ("a" * 40, "main", "unchecked"),
    ("aaaaaaa", "a" * 40, "unchecked"),          # a shortened hash the commit begins with
    ("bbbbbbb", "a" * 40, "differ"),             # one the commit does not begin with
    ("aaaaaaa", "aaaaaaab", "unchecked"),
    ("v1.0", "v1.1", "unchecked"),
])
def test_revisions_differ_only_when_both_name_commits(agent, study, verdict):
    assert agent_identity.compare_revisions(agent, study) == verdict


@pytest.mark.parametrize("revision, refused, phrase", [
    ("main", False, "main is a branch or tag name"),
    ("aaaaaaa", False, "a shortened commit hash that the other begins with"),
    (None, False, "the agent records no model revision"),
    ("bbbbbbb", True, None),
    (OTHER, True, None),
])
def test_verify_refuses_differing_commits_and_notes_the_rest(tmp_path, revision, refused,
                                                             phrase):
    artifact = {} if revision is None else {"baseRevision": revision}
    manifest = study_with_arm(tmp_path, artifact)
    violations = [v for v in manifest.verify(str(tmp_path)) if "agent arm" in v]
    notes = agent_identity.manifest_arm_notes(manifest)
    if refused:
        assert any("was made for revision" in v for v in violations), violations
        assert notes == []
    else:
        assert violations == []
        assert len(notes) == 1 and phrase in notes[0]
        assert "did not check" in notes[0] and REVISION in notes[0]


def test_the_client_verify_carries_the_revision_note(tmp_path, capsys, monkeypatch):
    study_with_arm(tmp_path, {"baseRevision": "main"})
    monkeypatch.setattr(Manifest, "verify", lambda self, root=None: [])
    code, document = base.cli(tmp_path, capsys, "experiment", "verify", "study")
    assert code == 0, document
    notes = [a for a in document["advisories"] if a["code"] == "agentRevisionNotChecked"]
    assert len(notes) == 1 and "main is a branch or tag name" in notes[0]["detail"]


def _agent_file(root, name, **fields):
    document = {"schemaVersion": 1, "name": name, "baseModelID": MODEL,
                "alphaInNormUnits": True, "createdAt": "2026-11-08T00:00:00Z", **fields}
    path = Path(root) / "runs" / "model-variants" / name / "model-variant.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path.relative_to(root).as_posix(), base.sha(path)


def _manifest_digest(root, name="study"):
    return base.sha(Path(root) / "experiments" / name / "experiment.json")


def _attach(root, capsys, path, digest, *extra, study="study"):
    return base.cli(root, capsys, "experiment", "attach-agent", study, "--artifact", path,
                    "--artifact-sha256", digest, "--manifest-sha256",
                    _manifest_digest(root, study), *extra)


@pytest.mark.parametrize("revision, refused", [("main", False), ("aaaaaaa", False),
                                               ("bbbbbbb", True), (OTHER, True)])
def test_attach_refuses_only_differing_commits(tmp_path, capsys, revision, refused):
    store.create("study", model_id=MODEL, revision=REVISION, root=str(tmp_path))
    path, digest = _agent_file(tmp_path, "pinned", baseRevision=revision)
    code, document = _attach(tmp_path, capsys, path, digest)
    if refused:
        assert code == 65 and document["error"]["reason"].startswith(
            "agent 'pinned' was made for revision"), document
    else:
        assert code == 0, document


# --- R2: attach-agent --replace, and repairs that clear the refusal ------------------


def test_replace_puts_the_agent_in_the_arms_place(tmp_path, capsys):
    store.create("study", model_id=MODEL, revision=REVISION, root=str(tmp_path))
    for name in ("first", "second", "third"):
        assert _attach(tmp_path, capsys, *_agent_file(tmp_path, name))[0] == 0
    code, document = _attach(tmp_path, capsys, *_agent_file(tmp_path, "fixed"),
                             "--replace", "second")
    assert code == 0, document
    arms = [a["name"] for a in store.load_raw("study", str(tmp_path))["variantConditions"]]
    assert arms == ["first", "fixed", "third"]
    code, document = _attach(tmp_path, capsys, *_agent_file(tmp_path, "other"),
                             "--replace", "absent")
    assert code != 0 and "no agent arm named absent" in document["error"]["reason"]
    # A frozen study is not edited.
    raw = store.load_raw("study", str(tmp_path))
    raw["status"] = "frozen"
    store.save_raw(raw, str(tmp_path))
    code, document = _attach(tmp_path, capsys, *_agent_file(tmp_path, "late"),
                             "--replace", "first")
    assert code != 0 and document["error"]["code"] == "statusImmutable"


def _commands(repair: str) -> list[str]:
    """Each client command a repair names, as written."""
    return re.findall(r"steerlab (.+?)(?= \(or the app's agent editor\)|, then |; then |$)",
                      repair)


def _follow(root, capsys, repair: str, *, extra: list[str]) -> None:
    """Run a repair's commands as written, filling only its placeholders: the
    new agent's name, the flags the ellipsis leaves open, and the digests the
    earlier commands return."""
    made = None
    for command in _commands(repair):
        words = shlex.split(command.replace(" …", ""))
        if words[:2] == ["agent", "create"]:
            words = [w if w != "<new-name>" else "fixed" for w in words] + extra
            code, preview = base.cli(root, capsys, *words)
            assert code == 0, preview
            code, written = base.cli(root, capsys, *words, "--review-sha256",
                                     preview["result"]["reviewSHA256"])
            assert code == 0, written
            made = written["result"]
            continue
        if words[:2] == ["experiment", "attach-agent"]:
            study = words[2]
            fills = iter([made["artifactFileSHA256"], _manifest_digest(root, study)])
            words = [made["path"] if w == "<path>" else next(fills) if w == "<digest>" else w
                     for w in words]
        code, document = base.cli(root, capsys, *words)
        assert code == 0, (command, document)


def _broken_study(root, capsys, case: str) -> None:
    """A draft with one agent arm that breaks the rule ``case`` names."""
    model = QWEN if case == "instrument" else MODEL
    store.create("study", model_id=model, revision=REVISION, root=str(root))
    if case == "level":
        path, digest = _agent_file(root, "thinker", baseRevision=REVISION,
                                   reasoningEffort="medium", qwenThinkingEnabled=True)
    elif case == "revision":
        path, digest = _agent_file(root, "elsewhere", baseRevision=OTHER)
    else:
        path, digest = _agent_file(root, "reasoner", baseModelID=QWEN,
                                   baseRevision=REVISION, reasoningEffort="on",
                                   qwenThinkingEnabled=True)
    raw = store.load_raw("study", str(root))
    artifact = json.loads((Path(root) / path).read_text())
    raw["variantConditions"] = [{"name": artifact["name"], "artifactPath": path,
                                 "artifactHash": digest, "artifact": artifact}]
    if case == "instrument":
        raw["outcomeInstruments"] = ["answerTokenLogprob"]
    store.save_raw(raw, str(root))


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("case", ["level", "revision", "instrument"])
def test_each_repair_followed_as_written_clears_the_refusal(tmp_path, capsys, case, frozen):
    _broken_study(tmp_path, capsys, case)
    if frozen:
        raw = store.load_raw("study", str(tmp_path))
        raw["status"] = "frozen"
        store.save_raw(raw, str(tmp_path))
    with vocabulary.speaking_as(vocabulary.CLIENT):
        refusal = study_admission.agent_arm_refusal(Manifest.load("study", str(tmp_path)),
                                                    str(tmp_path))
    assert refusal is not None
    repair = refusal.repair_action
    assert ("experiment duplicate study study-v2" in repair) is frozen
    if case == "instrument":
        assert "set-protocol" in repair and "attach-agent" not in repair
    else:
        assert "--replace" in repair
    _follow(tmp_path, capsys, repair, extra=[])
    repaired = Manifest.load("study-v2" if frozen else "study", str(tmp_path))
    assert study_admission.agent_arm_refusal(repaired, str(tmp_path)) is None
    assert agent_identity.manifest_arm_problems(repaired, str(tmp_path)) == []


# --- R4: promotion at the level the sweep ran at, on every path ------------------------


def test_the_default_path_mints_at_the_sweeps_level_after_a_set_sampling(tmp_path):
    """The recommended condition's path: the study's level moved after the
    sweep; the agent carries the level the sweep's cells ran at."""
    import test_sweep_promote as sweep
    from steerlab_server.experiment import promote
    root = str(tmp_path)
    sweep._promotable(root)
    raw = store.load_raw("pr", root)
    raw.update(reasoningEffort="medium", reasoningMaxTokens=64)
    snapshot = Path(root) / "runs" / sweep.SELECTION_BLOCK["sweepRun"] / "experiment.json"
    snapshot.write_text(json.dumps(raw))     # the sweep ran at medium
    raw.update(reasoningEffort="high")       # then: set-sampling --reasoning-effort high
    store.save_raw(raw, root)
    minted = promote.promote("pr", "fear", root=root, log=lambda *_: None)
    assert minted["variant"]["reasoningEffort"] == "medium"
    # The run-evidence path (no stamped condition) reads its run's snapshot.
    raw["conditions"] = [c for c in raw["conditions"] if c["name"] != "fear-recommended"]
    store.save_raw(raw, root)
    run = "20260707T000099000-exp-pr-sweep"
    sweep._plant_sweep_run(root, run, {"fear": {
        **sweep.SELECTION_BLOCK, "sweepRun": run, "winningCell": {"layer": 3, "alpha": 0.4}}})
    (Path(root) / "runs" / run / "experiment.json").write_text(
        json.dumps({**raw, "reasoningEffort": "low"}))
    again = promote.promote("pr", "fear", root=root, log=lambda *_: None,
                            agent_name="from-the-run")
    assert again["variant"]["reasoningEffort"] == "low"


def test_the_reuse_refusal_repeats_every_flag(tmp_path):
    import test_sweep_promote as sweep
    from steerlab_server.experiment import promote
    root = str(tmp_path)
    sweep._experiment_with_concept(root, "pin")
    stimulus = Manifest.load("pin", root).concepts[0].stimulus_set_hash
    sweep._vector_artifact(root, stimulus_hash=stimulus)
    run = sweep._sweep_evidence_run(root, "pin", sweep._pinned_entry())
    snapshot = Path(root) / "runs" / run / "experiment.json"
    level = json.loads(snapshot.read_text())
    level.update(reasoningEffort="medium", reasoningMaxTokens=64)
    snapshot.write_text(json.dumps(level))
    pins = promote.PromotionPins(sweep_run=run, winning_cell=(5, 0.25))
    request = dict(root=root, log=lambda *_: None, pins=pins, cell=(5, 0.25),
                   override_reason="the prior study's setting")
    minted = promote.promote("pin", "fear", **request)
    stale = json.loads(Path(minted["path"]).read_text())
    stale.pop("reasoningEffort")
    stale["qwenThinkingEnabled"] = False
    Path(minted["path"]).write_text(json.dumps(stale))
    with pytest.raises(promote.PromoteError) as refused:
        promote.promote("pin", "fear", **request)
    repair = refused.value.repair_action
    command = repair[repair.index("steerlab-server"):]
    assert shlex.split(command) == [
        "steerlab-server", "experiment", "promote", "pin", "fear",
        "--agent-name", "pin-fear-agent-medium", "--cell", "5:0.25",
        "--reason", "the prior study's setting", "--sweep-run", run,
        "--expect-cell", "5:0.25"]


# --- R5: what the editor's record says --------------------------------------------------


def test_an_edited_agent_names_what_it_was_edited_from():
    record = {"schemaVersion": 1, "madeWith": agent_identity.MADE_WITH_AGENT_EDITOR,
              "reason": None, "changed": ["systemPrompt", "reasoningEffort"],
              "editedFrom": {"kind": "swept", "artifactFileSHA256": "c" * 64,
                             "experiment": "tone", "sweepRun": "sweep-1"}}
    artifact = {"name": "a", "baseModelID": MODEL, "handmade": record,
                "injections": [{"concept": "c", "vectorArtifactID": "v", "layer": 2,
                                "alpha": 0.1}]}
    summary = agent_identity.agent_origin(artifact)["summary"]
    assert summary == ("Made by hand in the app's agent editor. The researcher edited it "
                       "from an agent experiment promote made from study tone (sweep run "
                       "sweep-1) and changed its thinking level and persona. No reason "
                       "was stated.")
    assert "no sweep selected" not in summary
    record["editedFrom"] = {"kind": "handmade", "handmade": {"madeWith": "agent create"}}
    record["changed"] = ["injections", "bandWidth", "temperature"]
    assert ("edited it from an agent made by hand and changed its steering, layer band, "
            "and Playground temperature.") in agent_identity.agent_origin(artifact)["summary"]


# --- R6: an ablation's layers ---------------------------------------------------------


def test_an_ablation_at_one_layer_lists_that_layer(tmp_path, capsys):
    reference, digest = base.make_vector(tmp_path)
    vector = ["--vector", reference, "--vector-sha256", digest, "--strength", "1",
              "--mode", "ablate"]
    code, preview = base.cli(tmp_path, capsys, *base.base_args() + vector + ["--layer", "2"])
    assert code == 0, preview
    injection = preview["result"]["document"]["injections"][0]
    assert injection["layers"] == [2] and injection["layer"] == 2
    assert any("ablated at layer 2 only" in note for note in preview["result"]["advisories"])
    code, preview = base.cli(tmp_path, capsys, *base.base_args() + vector + ["--layer", "all"])
    assert code == 0, preview
    planned = preview["result"]
    assert "layers" not in planned["document"]["injections"][0]
    assert planned["document"]["handmade"]["inputs"]["vectors"][0]["layer"] == "all"
    assert any("ablated at every layer" in note for note in planned["advisories"])
    code, refused = base.cli(tmp_path, capsys, *base.base_args() + [
        "--vector", reference, "--vector-sha256", digest, "--strength", "0.1",
        "--layer", "all"])
    assert code == 64 and "--layer all applies only to an ablating vector" in \
        refused["error"]["reason"]


def test_the_arm_rules_check_an_ablations_layers(tmp_path):
    reference, _ = base.make_vector(tmp_path)
    manifest = study_with_arm(tmp_path, {"injections": [
        {"concept": "formality", "vectorArtifactID": reference, "layer": 2, "alpha": 1,
         "mode": "ablate", "layers": [2, 9]}]})
    assert any("the ablation of 'formality' lists layer 9" in v
               for v in manifest.verify(str(tmp_path)))


def test_a_band_layer_where_the_vector_is_zero_needs_no_norm(tmp_path, capsys):
    """The engine skips a layer where the vector is zero, so neither agent
    create nor the arm rules ask for its residual norm."""
    reference, digest = base.make_vector(tmp_path)
    sidecar = tmp_path / f"{reference}.json"
    data = json.loads(sidecar.read_text())
    data["residualNormPerLayer"] = [10.0]       # layer 1 has no norm; the vector is zero there
    sidecar.write_text(json.dumps(data))
    code, preview = base.cli(tmp_path, capsys, *base.base_args() + [
        "--vector", reference, "--vector-sha256", digest, "--layer", "0",
        "--strength", "0.1", "--band-width", "3"])
    assert code == 0, preview
    manifest = study_with_arm(tmp_path, {"bandWidth": 3, "injections": [
        {"concept": "formality", "vectorArtifactID": reference, "layer": 0, "alpha": 0.1}]})
    assert not any("records no residual norm" in v for v in manifest.verify(str(tmp_path)))


# --- nits: inputs' revisions, the reason, a file outside the workspace ---------------------


def test_an_adapter_on_a_branch_is_noted_not_looped(tmp_path, capsys):
    adapter, digest = base.make_adapter(tmp_path, revision="main")
    code, preview = base.cli(tmp_path, capsys, *base.base_args() + [
        "--adapter", adapter, "--adapter-sha256", digest])
    assert code == 0, preview
    assert any("records revision main" in note for note in preview["result"]["advisories"])
    other, other_digest = base.make_adapter(tmp_path / "w2", revision="bbbbbbb")
    code, refused = base.cli(tmp_path / "w2", capsys, *base.base_args() + [
        "--adapter", other, "--adapter-sha256", other_digest])
    assert code == 65
    assert "full 40-character commit it names" in refused["error"]["repairAction"]


def test_a_reason_naming_a_home_folder_is_kept_and_warned(tmp_path, capsys):
    reason = "copied from " + "/" + "Users" + "/someone/notes.txt"
    code, preview = base.cli(tmp_path, capsys, *base.base_args("plain") + ["--reason", reason])
    assert code == 0, preview
    assert preview["result"]["document"]["handmade"]["reason"] == reason
    assert any("kept word for word" in note for note in preview["result"]["advisories"])
    code, quiet = base.cli(tmp_path, capsys, *base.base_args("quiet") + ["--reason", "a study"])
    assert not any("kept word for word" in note for note in quiet["result"]["advisories"])


# --- R3 and the one budget predicate --------------------------------------------------


def test_a_cast_agent_meets_the_rules_but_the_answer_token_one():
    capabilities = prompt_render.capabilities_for(MODEL)
    thinking = {"name": "s", "baseModelID": MODEL, "reasoningEffort": "medium"}
    problems = agent_identity.cast_agent_problems(
        "seat 'judge'", thinking, model_id=MODEL, model_revision=REVISION,
        capabilities=capabilities, root=None)
    assert problems and problems[0].startswith("seat 'judge': ")
    legacy = {"name": "s", "baseModelID": QWEN, "qwenThinkingEnabled": True}
    assert agent_identity.cast_agent_problems(
        "seat 'judge'", legacy, model_id=QWEN, model_revision=REVISION,
        capabilities=prompt_render.capabilities_for(QWEN), root=None) == []


def test_both_budget_gates_ask_the_one_predicate(tmp_path, monkeypatch):
    from steerlab_server.experiment import draft_protocol_policy
    plain = study_with_arm(tmp_path, {}, model=QWEN, reasoningEffort="off",
                           reasoningMaxTokens=256)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in plain.verify(str(tmp_path))
    monkeypatch.setattr(agent_identity, "budget_permitted_without_effort", lambda m: True)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in plain.verify(str(tmp_path))
    raw = store.load_raw("study", str(tmp_path))
    draft_protocol_policy.apply_protocol("study", raw, {"reasoningMaxTokens": 128},
                                         capabilities=None)
    assert raw["reasoningMaxTokens"] == 128


def test_the_agent_origin_fixture_is_current():
    """The editor-record summaries both engines must word alike; a difference
    means the fixture is stale (regenerate it with
    scripts/regenerate-cross-engine-fixtures.py, agent_origin_summaries)."""
    path = (Path(__file__).resolve().parents[2] / "Tests" / "Fixtures" / "cross-engine"
            / "agent-origin-summaries.json")
    cases = json.loads(path.read_text())
    assert len(cases) >= 6
    for case in cases:
        assert agent_identity.agent_origin(case["artifact"])["summary"] == case["summary"]
