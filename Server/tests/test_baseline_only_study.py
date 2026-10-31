"""A study that measures the baseline agent alone (public issue #4).

A researcher replicating published studies on an unmodified model wants a
study whose only subject is the baseline agent: the pinned base model, with no
vectors or adapters, at the study's sampling settings. The sanctioned spelling
is a declared ``studyType: agentComparison`` with no concept, no injection
condition, no agent (variant) condition, and no SAE latent arm. Before this
file, such a draft passed ``pack apply`` and ``data check`` and then refused
at ``verify`` (and so at freeze, where ``--force`` does not reach) as the empty
study, and the only workaround ran every item twice under two identical arms.

What is pinned here:

1. The rule (``manifest.baseline_only_study``), and that ``verify`` keeps the
   empty-study refusal for undeclared and concept-study drafts.
2. The cross-engine fixture ``baseline-only-study.json`` is this engine's
   answer for every shape, and the settings summary's Statistics section.
3. The client's verify and freeze: clean, a ``baselineOnlyStudy`` advisory,
   no validation or battery evidence asked for, not forced, and no
   capability-battery stamp. A paired rubric, which has nothing to pair with
   one arm, is refused at verify under ``studyDeclaration`` with the
   per-response route as the repair; a rubric that codes each response
   verifies.
4. End to end on a fake model: the run executes one arm, a sharded run plans
   and merges the same records, submission sizing prices them, the analysis
   has no rows and says ``baselineOnlyStudy`` instead of ``emptyAnalysis``,
   and the export and results page describe it plainly.

Swift twin: ``Tests/ExperimentKitTests/BaselineOnlyStudyTests.swift``.
"""

import csv
import hashlib
import io
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import steerlab_server.experiment.generate as _owner_generate
from steerlab_server import cli, cli_envelope, client_cli
from steerlab_server.api import submissions
from steerlab_server.client import results_export
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import freeze_policy, sharding, tasks
from steerlab_server.experiment import manifest as m
from steerlab_server.experiment import verification_refusal as refusal

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                       "baseline-only-study.json")
REGENERATE = ("stale fixture — re-run baseline_only_study() in "
              "scripts/regenerate-cross-engine-fixtures.py and commit")
STUDY = "replication"
REVISION = "a" * 40
ITEMS = ('{"id": "p0", "prompt": "Decide the case."}\n'
         '{"id": "p1", "prompt": "Explain the ruling."}\n'
         '{"id": "p2", "prompt": "Summarize the facts."}\n')
SEEDS = [0, 1]


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(cli_envelope, "now", lambda: 1_000.0)


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _raw(**fields):
    d = {"name": "s", "modelID": "org/m", "studyKind": "modelOutput",
         "studyType": "agentComparison", "concepts": [], "conditions": [],
         "variantConditions": []}
    d.update(fields)
    return d


# =============================================================================
# 1. The rule
# =============================================================================


def test_the_declared_agent_comparison_with_nothing_attached_is_baseline_only():
    assert m.baseline_only_study(_raw())
    assert m.baseline_only_study(_raw(conditions=[{"name": "baseline", "slots": []}]))
    # Carried concepts are inert in an agent comparison: still the baseline.
    assert m.baseline_only_study(_raw(concepts=[{"name": "fear"}]))
    assert [c.name for c in m.effective_conditions(m.Manifest.from_dict(_raw()))] \
        == ["baseline"]


@pytest.mark.parametrize("fields", [
    {"studyType": None},
    {"studyType": "conceptStudy"},
    {"studyType": "confirmAgent"},
    {"studyKind": "multiAgent", "studyType": "multiAgent"},
    {"variantConditions": [{"name": "agent-1"}]},
    {"saeLatentConditions": [{"name": "latent-1"}]},
    {"conditions": [{"name": "fear-hi",
                     "slots": [{"concept": "fear", "layer": 12, "alpha": 1.0}]}]},
    # A condition named baseline that steers is an injection condition.
    {"conditions": [{"name": "baseline",
                     "slots": [{"concept": "fear", "layer": 12, "alpha": 1.0}]}]},
])
def test_any_other_shape_is_not_baseline_only(fields):
    assert not m.baseline_only_study(_raw(**fields))


def test_verify_accepts_it_and_keeps_the_empty_study_where_it_is_a_mistake(tmp_path):
    def empty(d):
        return refusal.EMPTY_MODEL_OUTPUT_VIOLATION in \
            m.Manifest.from_dict(d).verify(str(tmp_path))
    assert not empty(_raw())
    assert empty(_raw(studyType=None))
    assert empty(_raw(studyType="conceptStudy"))


def test_the_dead_end_advice_now_leads_somewhere():
    """The refusal of a concept study with no arms names the sanctioned
    spelling of a deliberate baseline-only study, and that spelling now
    verifies, freezes, and runs."""
    problem = m.no_measured_conditions_problem(
        _raw(studyType=None, concepts=[{"name": "fear"}]))
    assert "agentComparison" in problem
    assert "baseline agent alone" in problem
    assert "BASELINE only" in problem and "promote" in problem


def test_freeze_asks_for_no_validation_and_no_battery_evidence():
    """The battery checks whether steering broke the model, and a
    baseline-only study steers nothing: the battery gate is not asked of it,
    exactly as it is not asked of a study without agents, so nothing is
    exempted and nothing is stamped."""
    d = _raw(modelRevision=REVISION)
    assert not freeze_policy.needs_validation(d)
    assert freeze_policy.battery_not_applied(d) == []
    assert freeze_policy.evaluate("s", d, freeze_policy.FreezeEvidence()) == []
    # A canonical baseline condition, or a carried concept, is inert here and
    # asks for no validation either; the same shapes in a concept study do.
    for extra in ({"conditions": [{"name": "baseline", "slots": []}]},
                  {"concepts": [{"name": "fear"}]}):
        assert not freeze_policy.needs_validation({**d, **extra}), extra
        assert freeze_policy.needs_validation(
            {**d, **extra, "studyType": "conceptStudy"}), extra


# =============================================================================
# 2. The cross-engine fixture
# =============================================================================


def test_the_fixture_is_this_engines_answer_for_every_shape(tmp_path):
    fixture = _fixture()
    for relative, text in fixture["rubrics"].items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text, encoding="utf-8")
    assert fixture["pairedRubricViolation"] == \
        refusal.baseline_only_paired_rubric_violation(
            "prompts/rubrics/paired.md"), REGENERATE
    assert {case["label"] for case in fixture["cases"]
            if case["pairedRubricRefused"]} == {"declared-with-a-paired-rubric"}
    assert fixture["advisory"] == m.BASELINE_ONLY_STUDY_ADVISORY, REGENERATE
    assert fixture["summary"] == m.BASELINE_ONLY_SUMMARY, REGENERATE
    assert fixture["effectsSentence"] == results_export.BASELINE_ONLY_EFFECTS, \
        REGENERATE
    labels = {case["label"] for case in fixture["cases"]}
    assert {"declared-nothing-attached", "undeclared-nothing-attached",
            "declared-concept-study-nothing-attached",
            "declared-with-an-agent"} <= labels
    for case in fixture["cases"]:
        manifest = case["manifest"]
        parsed = m.Manifest.from_dict(json.loads(json.dumps(manifest)))
        assert m.baseline_only_study(manifest) is case["baselineOnly"], \
            f"{case['label']}: {REGENERATE}"
        violations = parsed.verify(str(tmp_path))
        assert (refusal.EMPTY_MODEL_OUTPUT_VIOLATION in violations) \
            is case["emptyStudy"], f"{case['label']}: {REGENERATE}"
        assert any(v.startswith(refusal.BASELINE_ONLY_PAIRED_RUBRIC_PREFIX)
                   for v in violations) is case["pairedRubricRefused"], \
            f"{case['label']}: {REGENERATE}"
        if case["executedConditions"] is not None:
            assert m.effective_condition_names(parsed) == \
                case["executedConditions"], f"{case['label']}: {REGENERATE}"


def test_the_fixture_holds_this_engines_statistics_sections(tmp_path):
    for key, entry in _fixture()["preregistrationStatistics"].items():
        manifest = entry["manifest"]
        directory = tmp_path / "experiments" / manifest["name"]
        directory.mkdir(parents=True)
        es._write_preregistration(json.loads(json.dumps(manifest)), str(tmp_path))
        text = (directory / es.PREREG_FILENAME).read_text(encoding="utf-8")
        start = text.index("## Statistics\n\n") + len("## Statistics\n\n")
        assert text[start:text.index("\n\n", start)] == \
            entry["statisticsSection"], f"{key}: {REGENERATE}"


# =============================================================================
# 3 and 4. The lifecycle, end to end
# =============================================================================


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    prompts = root / "prompts" / "tasks" / "items.jsonl"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(ITEMS, encoding="utf-8")
    es.create(STUDY, model_id="org/m", revision=REVISION, root=str(root))
    raw = es.load_raw(STUDY, str(root))
    raw.update({"studyType": "agentComparison",
                "taskPromptsFile": "prompts/tasks/items.jsonl",
                "taskPromptsHash": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                "seeds": SEEDS, "temperature": 0.7, "maxTokens": 16})
    es.save_raw(raw, str(root))
    return root


def _client(argv, capsys):
    capsys.readouterr()
    code = client_cli.main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def _codes(document):
    return [entry["code"] for entry in document.get("advisories") or []]


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or REVISION)


def _generate(model, prompt, *, injections=None, **_):
    assert not injections, "a baseline-only run steers nothing"
    return f"An answer to: {prompt}"


def _run(root, monkeypatch, **extra):
    monkeypatch.setattr(_owner_generate, "generate", _generate)
    return tasks.run(STUDY, None, str(root), model_provider=_fake_model,
                     log=lambda *_: None, **extra)


def _records(directory):
    with open(os.path.join(directory, "generations.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_client_verifies_and_freezes_it_cleanly(workspace, capsys):
    code, verified = _client(["experiment", "verify", STUDY], capsys)
    assert code == 0 and verified["state"] == "okWithAdvisories"
    assert verified["result"]["violations"] == []
    assert _codes(verified) == ["baselineOnlyStudy"]
    assert verified["advisories"][0]["detail"] == m.BASELINE_ONLY_STUDY_ADVISORY

    code, frozen = _client(["experiment", "freeze", STUDY], capsys)
    assert code == 0, frozen
    assert frozen["result"]["status"] == "frozen"
    assert frozen["result"]["forced"] is False
    assert _codes(frozen) == ["baselineOnlyStudy"]
    stored = es.load_raw(STUDY, str(workspace))
    assert "freezeForced" not in stored and "forcedGatesSkipped" not in stored
    assert freeze_policy.BATTERY_NOT_APPLIED_KEY not in stored
    # The settings summary says what the one arm is, and promises no paired
    # statistics it cannot compute.
    summary = (workspace / "experiments" / STUDY / es.PREREG_FILENAME).read_text(
        encoding="utf-8")
    assert m.BASELINE_ONLY_SUMMARY in summary
    assert "no paired difference to estimate or test" in summary
    assert "same-case baseline" not in summary


def test_an_undeclared_empty_draft_is_still_refused(workspace, capsys):
    raw = es.load_raw(STUDY, str(workspace))
    raw.pop("studyType")
    es.save_raw(raw, str(workspace))
    code, document = _client(["experiment", "verify", STUDY], capsys)
    assert code == 65
    assert document["error"]["code"] == "emptyStudy"


def _pin_rubric(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    raw = es.load_raw(STUDY, str(root))
    raw.update({"judges": [{"name": "coder", "kind": "claude"}],
                "judgeRubricFile": relative,
                "judgeRubricHash": hashlib.sha256(path.read_bytes()).hexdigest()})
    es.save_raw(raw, str(root))


def test_a_paired_rubric_is_refused_at_verify_with_the_per_response_route(
        workspace, capsys):
    """Paired judging compares a condition with the baseline. With one arm
    there is nothing to pair, so such a study used to freeze, run, and then
    refuse at evaluate. verify refuses it now, before any of that."""
    rubrics = _fixture()["rubrics"]
    _pin_rubric(workspace, "prompts/rubrics/paired.md",
                rubrics["prompts/rubrics/paired.md"])
    for verb in ("verify", "freeze"):
        code, document = _client(["experiment", verb, STUDY], capsys)
        assert code == 65, (verb, document)
        assert document["error"]["code"] == "studyDeclaration", verb
        repair = document["error"]["repairAction"]
        assert "perResponseCoding" in repair, verb
        assert f"steerlab experiment pin-rubric {STUDY} prompts/rubrics/" in repair
    code, verified = _client(["experiment", "verify", STUDY], capsys)
    assert verified["result"]["violations"] == [
        refusal.baseline_only_paired_rubric_violation("prompts/rubrics/paired.md")]
    assert es.load_raw(STUDY, str(workspace))["status"] == "draft"

    # The route, followed: a rubric that codes each response verifies.
    _pin_rubric(workspace, "prompts/rubrics/coded.md",
                rubrics["prompts/rubrics/coded.md"])
    code, verified = _client(["experiment", "verify", STUDY], capsys)
    assert code == 0, verified
    assert _codes(verified) == ["baselineOnlyStudy"]


def test_the_same_paired_rubric_is_ordinary_beside_an_agent(tmp_path):
    """The refusal is about one arm, not about paired rubrics."""
    rubric = tmp_path / "prompts" / "rubrics" / "paired.md"
    rubric.parent.mkdir(parents=True)
    rubric.write_text("Which response is better?\n", encoding="utf-8")
    judged = {"judges": [{"name": "coder", "kind": "claude"}],
              "judgeRubricFile": "prompts/rubrics/paired.md",
              "judgeRubricHash": hashlib.sha256(rubric.read_bytes()).hexdigest()}
    alone = m.Manifest.from_dict(_raw(**judged)).verify(str(tmp_path))
    assert [v for v in alone
            if v.startswith(refusal.BASELINE_ONLY_PAIRED_RUBRIC_PREFIX)]
    concept = m.Manifest.from_dict(
        _raw(studyType="conceptStudy", **judged)).verify(str(tmp_path))
    assert not [v for v in concept
                if v.startswith(refusal.BASELINE_ONLY_PAIRED_RUBRIC_PREFIX)]
    # Judging switched off: no evaluate will run, so nothing to refuse.
    off = m.Manifest.from_dict(
        _raw(evaluation={"kind": "none"}, **judged)).verify(str(tmp_path))
    assert not [v for v in off
                if v.startswith(refusal.BASELINE_ONLY_PAIRED_RUBRIC_PREFIX)]


def test_it_runs_one_arm_analyzes_and_exports(workspace, capsys, monkeypatch):
    assert _client(["experiment", "freeze", STUDY], capsys)[0] == 0
    manifest = m.Manifest.load(STUDY, str(workspace))

    # The run: one arm, every item under every seed, nothing steered.
    run_dir = _run(workspace, monkeypatch)
    records = _records(run_dir)
    assert {r["condition"] for r in records} == {"baseline"}
    assert len(records) == 3 * len(SEEDS)
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        assert list(json.load(handle)["conditions"]) == ["baseline"]

    # Cluster sizing prices exactly those records.
    assert submissions._planned_records(manifest, ITEMS) == len(records)

    # A sharded fan-out plans the same records and merges to the same run.
    shard_dirs = [_run(workspace, monkeypatch, shard=sharding.ShardSpec(k, 2))
                  for k in range(2)]
    merged = sharding.merge_shard_runs(STUDY, shard_dirs, root=str(workspace),
                                       shard_job_ids=["j0", "j1"])
    assert [(r["condition"], r["promptID"], r["seed"]) for r in _records(merged)] \
        == [(r["condition"], r["promptID"], r["seed"]) for r in records]

    # The analysis, through the engine's own verb: no rows, and said as what
    # it is rather than as an empty analysis.
    monkeypatch.setenv("STEERLAB_ROOT", str(workspace))
    capsys.readouterr()
    code = cli.main(["experiment", "analyze", STUDY, "--source", run_dir, "--json"])
    analyzed = json.loads(capsys.readouterr().out)
    assert code == 0, analyzed
    assert analyzed["result"]["effectSizeCount"] == 0
    assert "baselineOnlyStudy" in _codes(analyzed)
    assert "emptyAnalysis" not in _codes(analyzed)
    monkeypatch.delenv("STEERLAB_ROOT")

    # The export: the one arm described, no comparison claimed, no
    # statistics promised, and an effects table with its columns and no rows.
    code, exported = _client(["results", "export", STUDY, "--run",
                              "runs/" + os.path.basename(run_dir)], capsys)
    assert code == 0, exported
    directory = exported["result"]["exportDirectory"]
    with open(os.path.join(directory, "methods.md"), encoding="utf-8") as handle:
        methods = handle.read()
    assert "- **baseline**: no intervention (the only arm: this study measures " \
        "the baseline agent alone)." in methods
    assert "Outcomes compared between conditions: none. Baseline only" in methods
    assert "no estimate, interval, or test applies" in methods
    assert "The engine compared each condition with the baseline" not in methods
    assert "no analysis of this run was found" not in methods
    assert "steers nothing, so freeze asked for no battery evidence" in methods
    with open(os.path.join(directory, "effects.csv"), encoding="utf-8") as handle:
        assert list(csv.reader(io.StringIO(handle.read())))[1:] == []
    with open(os.path.join(directory, "codebook.md"), encoding="utf-8") as handle:
        assert "This table has no rows" in handle.read()
    assert not any(entry["what"] == "the bootstrap settings"
                   for entry in exported["result"]["notAvailable"])

    # The results page: the headline and the effects say baseline only and
    # point to the descriptive summary; nothing reads as a missing analysis.
    code, reported = _client(["results", "report", STUDY, "--run",
                              "runs/" + os.path.basename(run_dir)], capsys)
    assert code == 0, reported
    with open(reported["result"]["htmlPath"], encoding="utf-8") as handle:
        page = handle.read()
    assert "Baseline only: this study measures the baseline agent alone" in page
    assert "no analysis and no completed evaluation" not in page
    assert "The analysis stored no effect over all items" not in page
    assert "Every condition is compared with the baseline" not in page
    assert "What the run stored for each condition" in page
