"""Re-reading a run's stored choices with the current choice parser.

A choice is read once, when a run writes its records, and runs are immutable,
so a run read by an older parser keeps that parser's mistakes. These tests
hold the read-only check (``results check-choices``), the opt-in re-read
during analysis (``analyze --reread-choices``), and the advisory an ordinary
analysis carries, to the hand-written cross-engine fixture
``Tests/Fixtures/cross-engine/choice-reread.json``. The Mac engine reads the
same file (``ChoiceRereadTests.swift``) and must produce the same counts and
the same effect rows.
"""

import csv
import hashlib
import json
import os

import pytest

from steerlab_server import cli, cli_envelope, client_cli
from steerlab_server.client import diagnostic_commands, results_commands
from steerlab_server.experiment import (choice_reread, judicial,
                                        lifecycle_gates, tasks)
from steerlab_server.experiment.manifest import Manifest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "cross-engine")
RUN = "20261017T000000000-exp-reread-run"


def _fixture():
    with open(os.path.join(FIXTURES, "choice-reread.json"), encoding="utf-8") as handle:
        return json.load(handle)


FIXTURE = _fixture()
STUDY = FIXTURE["study"]
EXPECTED = FIXTURE["expected"]


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _workspace(tmp_path, *, in_workspace=True, pinned=False, exclusion_rules=None,
               records=None):
    """A workspace holding the fixture's study and one completed run of it.
    Returns ``(root, run directory)``."""
    root = tmp_path / "ws"
    text = FIXTURE["taskPromptsText"]
    manifest = {
        "name": STUDY, "modelID": "test/model", "concepts": [],
        "taskPromptsFile": FIXTURE["taskPromptsFile"],
        "taskPromptsHash": _digest(text),
        "conditions": [{"name": "steered",
                        "slots": [{"concept": "c", "layer": 1, "alpha": 2.0}]}],
    }
    if exclusion_rules:
        manifest["exclusionRules"] = exclusion_rules
    experiment = root / "experiments" / STUDY
    experiment.mkdir(parents=True)
    (experiment / "experiment.json").write_text(json.dumps(manifest), encoding="utf-8")
    relative = FIXTURE["taskPromptsFile"]
    if in_workspace:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    if pinned:
        path = experiment / "pinned" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    run_dir = root / "runs" / RUN
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "experiment-hash.txt").write_text(
        Manifest.from_dict(manifest).content_hash() + "\n", encoding="utf-8")
    (run_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(record) for record in (records or FIXTURE["records"])) + "\n",
        encoding="utf-8")
    (run_dir / "report.json").write_text("{}", encoding="utf-8")
    return str(root), str(run_dir)


def _run_bytes(run_dir):
    found = {}
    for name in sorted(os.listdir(run_dir)):
        with open(os.path.join(run_dir, name), "rb") as handle:
            found[name] = hashlib.sha256(handle.read()).hexdigest()
    return found


def _pooled(out, endpoint, condition="steered"):
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["endpoint"] == endpoint and row["condition"] == condition
                and row["stratifyBy"] in ("", "pooled")]
    assert len(rows) == 1, rows
    return float(rows[0]["deltaMean"]), int(rows[0]["n"])


# --- the parser version -------------------------------------------------------


def test_the_parser_version_is_the_fixtures_and_is_bumped_with_the_rules():
    """The constant names the parsing rules. The parsing fixture states those
    rules case by case, so when its bytes change the rules have changed, and
    the version (here and in Judicial.choiceParserVersion) must move with
    them."""
    assert judicial.CHOICE_PARSER_VERSION == FIXTURE["parserVersion"]
    with open(os.path.join(FIXTURES, "choice-parsing.json"), "rb") as handle:
        current = hashlib.sha256(handle.read()).hexdigest()
    assert current == FIXTURE["parsingFixtureSHA256"], FIXTURE["parserVersionNote"]


# --- the re-read itself -------------------------------------------------------


def _options():
    from steerlab_server.experiment import task_inputs
    return {p["id"]: p["options"] for p in task_inputs.parse_prompts(FIXTURE["taskPromptsText"])
            if p.get("options")}


def test_the_fixture_records_reread_to_the_hand_written_counts():
    entries = list(enumerate(FIXTURE["records"], start=1))
    fresh, report = choice_reread.reread(entries, _options())
    assert report["total"] == EXPECTED["total"]
    assert report["conditions"] == EXPECTED["conditions"]
    assert report["examples"] == EXPECTED["examples"]
    # Only records that carry parsedChoice are read; the two without options
    # (lines 7 and 14) are not.
    assert sorted(fresh) == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]


def test_the_sentence_is_built_from_the_stamped_numbers():
    value = {"parserVersion": FIXTURE["parserVersion"], "total": EXPECTED["total"]}
    assert choice_reread.sentence(value) == EXPECTED["sentence"]


@pytest.mark.parametrize("stored,fresh,kind", [
    ("A", "A", None), (None, None, None), ("A", "B", "changedToOtherOption"),
    ("A", None, "changedToNoAnswer"), (None, "B", "gainedAnswer")])
def test_kinds_of_change(stored, fresh, kind):
    assert choice_reread.kind_of(stored, fresh) == kind


# --- finding the options ------------------------------------------------------


def test_check_reads_the_verified_workspace_file(tmp_path):
    root, run_dir = _workspace(tmp_path)
    report = choice_reread.check(run_dir, root, study=STUDY)
    assert report["parserVersion"] == FIXTURE["parserVersion"]
    assert report["taskPrompts"] == {
        "file": FIXTURE["taskPromptsFile"], "sha256": _digest(FIXTURE["taskPromptsText"]),
        "readFrom": FIXTURE["taskPromptsFile"], "location": "workspace"}
    assert (report["total"], report["conditions"], report["examples"]) == (
        EXPECTED["total"], EXPECTED["conditions"], EXPECTED["examples"])


def test_check_falls_back_to_a_matching_pinned_copy(tmp_path):
    root, run_dir = _workspace(tmp_path, in_workspace=False, pinned=True)
    report = choice_reread.check(run_dir, root, study=STUDY)
    assert report["taskPrompts"]["location"] == "pinned"
    assert report["taskPrompts"]["readFrom"] == (
        f"experiments/{STUDY}/pinned/{FIXTURE['taskPromptsFile']}")
    assert report["total"] == EXPECTED["total"]


def test_a_changed_file_is_never_used(tmp_path):
    """A workspace file and a pinned copy that both differ from the run's
    pinned hash are refused, with a repair that names the file and the
    pinned folder."""
    root, run_dir = _workspace(tmp_path, pinned=True)
    changed = FIXTURE["taskPromptsText"].replace('"target": "B"', '"target": "A"')
    for path in (os.path.join(root, FIXTURE["taskPromptsFile"]),
                 os.path.join(root, "experiments", STUDY, "pinned", FIXTURE["taskPromptsFile"])):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(changed)
    with pytest.raises(lifecycle_gates.LifecycleError) as refusal:
        choice_reread.check(run_dir, root, study=STUDY)
    assert refusal.value.gate == lifecycle_gates.PIN_DRIFT
    assert FIXTURE["taskPromptsFile"] in str(refusal.value)
    assert f"experiments/{STUDY}/pinned/" in refusal.value.repair_action


def test_a_run_without_pinned_prompts_is_refused(tmp_path):
    root, run_dir = _workspace(tmp_path)
    settings = json.loads(open(os.path.join(run_dir, "experiment.json"), encoding="utf-8").read())
    settings.pop("taskPromptsHash")
    with open(os.path.join(run_dir, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(settings, handle)
    with pytest.raises(lifecycle_gates.LifecycleError) as refusal:
        choice_reread.check(run_dir, root, study=STUDY)
    assert refusal.value.gate == lifecycle_gates.MISSING_PREREQUISITE


def test_records_from_another_prompts_file_are_refused(tmp_path):
    """A record whose item has no options in the verified file, or whose own
    taskPromptsHash names a different file, cannot be re-read: its options
    are unknown."""
    records = [dict(record) for record in FIXTURE["records"]]
    records[0]["promptID"] = "elsewhere"
    root, run_dir = _workspace(tmp_path, records=records)
    with pytest.raises(lifecycle_gates.LifecycleError) as refusal:
        choice_reread.check(run_dir, root, study=STUDY)
    assert refusal.value.gate == lifecycle_gates.PIN_DRIFT
    assert "'elsewhere'" in str(refusal.value)

    records = [dict(record) for record in FIXTURE["records"]]
    records[0]["taskPromptsHash"] = "0" * 64
    root2, run_dir2 = _workspace(tmp_path / "second", records=records)
    with pytest.raises(lifecycle_gates.LifecycleError):
        choice_reread.check(run_dir2, root2, study=STUDY)


# --- analyze, with and without --reread-choices --------------------------------


def test_an_ordinary_analysis_uses_the_stored_choices(tmp_path):
    root, run_dir = _workspace(tmp_path)
    before = _run_bytes(run_dir)
    out = tasks.analyze(STUDY, root=root, log=lambda *_: None)
    expected = EXPECTED["choiceRate"]["stored"]
    assert _pooled(out, "choiceRate") == pytest.approx((expected["estimate"], expected["n"]), abs=1e-5)
    assert not os.path.exists(os.path.join(out, choice_reread.STAMP_FILE))
    assert _run_bytes(run_dir) == before


def test_reread_choices_uses_the_current_reading_and_stamps_it(tmp_path):
    root, run_dir = _workspace(tmp_path)
    before = _run_bytes(run_dir)
    out = tasks.analyze(STUDY, root=root, log=lambda *_: None, reread_choices=True)
    expected = EXPECTED["choiceRate"]["reread"]
    assert _pooled(out, "choiceRate") == pytest.approx((expected["estimate"], expected["n"]), abs=1e-5)
    with open(os.path.join(out, choice_reread.STAMP_FILE), encoding="utf-8") as handle:
        stamp = json.load(handle)
    assert stamp["parserVersion"] == FIXTURE["parserVersion"]
    assert stamp["sourceRun"] == RUN
    assert stamp["runUnchanged"] is True
    assert (stamp["total"], stamp["conditions"], stamp["examples"]) == (
        EXPECTED["total"], EXPECTED["conditions"], EXPECTED["examples"])
    assert stamp["taskPrompts"]["sha256"] == _digest(FIXTURE["taskPromptsText"])
    assert stamp["note"] == EXPECTED["note"]
    # The run directory is byte for byte what it was.
    assert _run_bytes(run_dir) == before


def test_reread_choices_reaches_the_exclusions(tmp_path):
    """An unparseableEndpoint rule on parsedChoice reads the re-read value:
    stored, steered q4 (no answer) is excluded; re-read, q3 is excluded
    instead. Word-count differences per item (steered minus baseline): q1 5,
    q2 5, q3 5, q4 4, q5 0, q6 2, q7 -1. Without q4: 16 over 6 items; without
    q3: 15 over 6 items."""
    rules = [{"rule": "unparseableEndpoint", "endpoint": "parsedChoice"}]
    root, _run_dir = _workspace(tmp_path, exclusion_rules=rules)
    stored = tasks.analyze(STUDY, root=root, log=lambda *_: None)
    assert _pooled(stored, "wordCount") == pytest.approx((16 / 6, 6), abs=1e-5)
    reread = tasks.analyze(STUDY, root=root, log=lambda *_: None, reread_choices=True)
    assert _pooled(reread, "wordCount") == pytest.approx((15 / 6, 6), abs=1e-5)


def test_reread_choices_refuses_without_verified_options(tmp_path):
    root, _run_dir = _workspace(tmp_path, in_workspace=False)
    with pytest.raises(lifecycle_gates.LifecycleError) as refusal:
        tasks.analyze(STUDY, root=root, log=lambda *_: None, reread_choices=True)
    assert refusal.value.gate == lifecycle_gates.PIN_DRIFT


# --- the engine command line ---------------------------------------------------


def _engine(argv, capsys):
    code = cli.main(argv + ["--json"])
    return code, json.loads(capsys.readouterr().out)


def test_the_flag_is_declared_on_the_analyze_verb():
    spec = cli_envelope.spec_for("experiment", "analyze")
    assert "--reread-choices" in spec.boolean_flags


def test_an_ordinary_analysis_says_when_choices_read_differently(tmp_path, capsys):
    root, _run_dir = _workspace(tmp_path)
    code, document = _engine(["experiment", "analyze", STUDY, "--root", root], capsys)
    assert code == 0
    advice = [entry for entry in document["advisories"]
              if entry["code"] == choice_reread.ADVISORY_CODE]
    assert len(advice) == 1
    detail = advice[0]["detail"]
    assert detail.startswith("5 of 12 stored choices in run '" + RUN + "'")
    assert "results check-choices" in detail and "--reread-choices" in detail


def test_a_reread_analysis_carries_no_such_advisory(tmp_path, capsys):
    root, _run_dir = _workspace(tmp_path)
    code, document = _engine(
        ["experiment", "analyze", STUDY, "--root", root, "--reread-choices"], capsys)
    assert code == 0
    assert choice_reread.ADVISORY_CODE not in {entry["code"] for entry in document.get("advisories", [])}


def test_the_advisory_never_blocks_an_unverifiable_run(tmp_path, capsys):
    root, _run_dir = _workspace(tmp_path, in_workspace=False)
    code, document = _engine(["experiment", "analyze", STUDY, "--root", root], capsys)
    assert code == 0
    assert choice_reread.ADVISORY_CODE not in {entry["code"] for entry in document.get("advisories", [])}


# --- the client's results check-choices, and the bridge -------------------------


def test_the_client_check_writes_nothing_and_reports_the_counts(tmp_path, capsys, monkeypatch):
    root, run_dir = _workspace(tmp_path)
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    before = sorted(os.walk(root))
    code = client_cli.main(["--root", root, "results", "check-choices", STUDY,
                            "--run", "runs/" + RUN, "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 0
    assert document["verb"] == "results check-choices"
    result = document["result"]
    assert result["run"] == "runs/" + RUN
    assert (result["total"], result["conditions"], result["examples"]) == (
        EXPECTED["total"], EXPECTED["conditions"], EXPECTED["examples"])
    assert document["changed"] is False
    assert sorted(os.walk(root)) == before


def test_the_client_check_refuses_with_the_gate_and_repair(tmp_path, capsys, monkeypatch):
    root, _run_dir = _workspace(tmp_path, in_workspace=False)
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    code = client_cli.main(["--root", root, "results", "check-choices", STUDY, "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 65
    assert document["error"]["code"] == lifecycle_gates.PIN_DRIFT
    assert "pinned" in document["error"]["repairAction"]


def test_the_bridge_answers_the_same_report(tmp_path):
    root, _run_dir = _workspace(tmp_path)
    answer = diagnostic_commands.workspace_action(
        results_commands.CHECK_CHOICES_BRIDGE_ACTION,
        {"workspaceRoot": root, "study": STUDY, "client": "steerlab-cli"})
    assert answer["checked"] is True
    assert answer["total"] == EXPECTED["total"]


def test_the_bridge_returns_a_refusal_as_data(tmp_path):
    root, _run_dir = _workspace(tmp_path, in_workspace=False)
    answer = diagnostic_commands.workspace_action(
        results_commands.CHECK_CHOICES_BRIDGE_ACTION,
        {"workspaceRoot": root, "study": STUDY})
    assert answer["checked"] is False
    assert answer["refusal"]["code"] == lifecycle_gates.PIN_DRIFT
    assert answer["refusal"]["state"] == "refused"


# --- the readers show the stamp, as stored ---------------------------------------


def test_the_export_and_the_results_page_say_the_choices_were_reread(tmp_path):
    from steerlab_server.client import results_export
    from steerlab_server.client.reports import study_results
    root, _run_dir = _workspace(tmp_path)
    tasks.analyze(STUDY, root=root, log=lambda *_: None, reread_choices=True)
    result = results_export.export_results(root, STUDY, out=str(tmp_path / "export"))
    folder = result["exportDirectory"]
    methods = open(os.path.join(folder, "methods.md"), encoding="utf-8").read()
    codebook = open(os.path.join(folder, "codebook.md"), encoding="utf-8").read()
    page = open(os.path.join(folder, "report.html"), encoding="utf-8").read()
    assert "- Choices: " + EXPECTED["sentence"] in methods
    assert EXPECTED["sentence"] in codebook
    assert "parsed_choice and parsed_choice_failed hold the choices as the run stored them" in codebook
    assert EXPECTED["sentence"] in page
    manifest = json.load(open(os.path.join(folder, "manifest.json"), encoding="utf-8"))
    assert any(entry["path"].endswith("/" + choice_reread.STAMP_FILE) and entry["role"] == "analysis"
               for entry in manifest["sources"])
    # The page drawn by `results report` is the export's own page.
    assert EXPECTED["sentence"] in study_results.render(results_export.read_results(root, STUDY))


def test_an_ordinary_analysis_export_says_nothing_about_a_reread(tmp_path):
    from steerlab_server.client import results_export
    root, _run_dir = _workspace(tmp_path)
    tasks.analyze(STUDY, root=root, log=lambda *_: None)
    result = results_export.export_results(root, STUDY, out=str(tmp_path / "export"))
    for name in ("methods.md", "codebook.md", "report.html"):
        text = open(os.path.join(result["exportDirectory"], name), encoding="utf-8").read()
        assert "re-read" not in text, name


# --- the HTTP route the app's server Analyze uses --------------------------------


def _wait_for_job(client, job_id, timeout=30.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = next((j for j in client.get("/api/jobs").json()["jobs"] if j["id"] == job_id), None)
        if job and job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def test_the_route_rereads_on_request_and_notes_the_count_otherwise(tmp_path, monkeypatch):
    """rereadChoices reaches the real analysis through the app's route, and
    an ordinary analysis there logs the same advisory the CLI gives."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    root, _run_dir = _workspace(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    client = TestClient(app)

    job = _wait_for_job(client, client.post(f"/api/experiment/{STUDY}/analyze",
                                            json={"rereadChoices": True}).json()["jobId"])
    assert job["status"] == "succeeded", job
    reread_out = job["result"]["runDirectory"]
    assert os.path.exists(os.path.join(reread_out, choice_reread.STAMP_FILE))
    expected = EXPECTED["choiceRate"]["reread"]
    assert _pooled(reread_out, "choiceRate") == pytest.approx(
        (expected["estimate"], expected["n"]), abs=1e-5)

    job_id = client.post(f"/api/experiment/{STUDY}/analyze").json()["jobId"]
    job = _wait_for_job(client, job_id)
    assert job["status"] == "succeeded", job
    assert not os.path.exists(os.path.join(job["result"]["runDirectory"], choice_reread.STAMP_FILE))
    logs = json.dumps(client.get(f"/api/jobs/{job_id}").json())
    assert "5 of 12 stored choices in run '" + RUN + "'" in logs
    assert "Re-read choices" in logs


def test_the_route_refuses_reread_choices_on_any_other_verb():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    response = TestClient(app).post(f"/api/experiment/{STUDY}/evaluate", json={"rereadChoices": True})
    assert response.status_code == 400
    assert "analyze" in response.json()["detail"]
