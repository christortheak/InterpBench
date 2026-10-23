"""A run checkpointed and then resumed to completion reads complete
(2026-10-23).

Found live in a qualification run: a study run cancelled mid-run parked
(``run-status.json`` = ``checkpointed``, 15 records), was resumed, cancelled
again while its continuation was saving its place (leaving the previous
``resume-state.json`` and an empty ``resume-state.json.tmp``), and resumed
again. The last continuation finished: 36 responses, ``report.json``, every
record identical to an uninterrupted run. But the first checkpoint's status
and the empty temp file were still in the directory and came home with the
evidence, so ``is_partial`` (and the Mac's run classification, which reads
the same file first) barred a complete run from every evidence-grade gate.
The resume message also counted 15 kept records where the folder held 18.

Pinned here:

- the whole sequence through ``bundle execute`` ends with a ``completed``
  status (the park kept as ``resumedFrom``), no saved place, no temp file,
  ``is_partial`` False, and records equal to an uninterrupted run's;
- a run cancelled and never resumed stays partial;
- a failed status and ``FAILED.md`` are never settled, a status is settled
  only once the completion artifact exists, an uninterrupted run is given
  no status, and a torn one stays torn;
- sweeps and panels settle on completion the same way;
- the resume message counts the record lines a continuation keeps, not the
  saved place's stale count.

The Swift reader's half is ``RunStatusFileTests`` (``ResumedRun`` cases).
"""

import json
import os
from types import SimpleNamespace

import pytest

import steerlab_server.experiment.choice_scoring as _owner_choice_scoring
import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.model_resources as _owner_model_resources
import steerlab_server.experiment.panel_workflow as panel_workflow
import steerlab_server.experiment.vector_materialization as _owner_vector_materialization
from steerlab_server.experiment import bundles, multi_agent, resume, run_status, tasks

from test_manual_resubmit import (  # noqa: F401 - fixtures by import
    _dummy_script, _manager, fake_slurm)
from test_resume_after_cancel import _end_marker
from test_resume_checkpoint import (_deterministic_generate, _fake_model,
                                    _patch_study_fakes, _study_fixture)
from test_resume_from_records import (Killed, _panel_generate, _panel_model,
                                      _panel_workspace)
from test_sweep_checkpoint import _fake_score, _workspace as _sweep_workspace
from test_sweep_checkpoint import _fake_bundle as _sweep_bundle
from test_sweep_checkpoint import _fake_model as _sweep_model


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


def _status(run_dir):
    return run_status.read_status(run_dir)


def _park_status(run_dir, *, stage="run", items=3, reason="signal"):
    """The status ``bundle execute`` writes when a run parks."""
    parked = run_status.RunStatus(run_dir, stage=stage, experiment="demo",
                                  item_label="record")
    parked.item_count = items
    parked.checkpointed(reason=reason)


def _tmp_state(run_dir):
    return os.path.join(run_dir, resume.RESUME_STATE_FILENAME + ".tmp")


# --- 1. the incident, end to end through bundle execute ---------------------------

def _loads_without_a_model(monkeypatch):
    monkeypatch.setattr(
        _owner_model_resources, "_load_model",
        lambda manifest, dtype, device=None: SimpleNamespace(
            model_id=manifest.model_id, revision="abc"))


def test_a_run_parked_twice_then_completed_reads_complete(tmp_path,
                                                          monkeypatch):
    source = str(tmp_path / "source")
    prompts = _study_fixture(source, "parked")
    bundle_path = bundles.package_experiment("parked", root=source)["bundlePath"]
    _loads_without_a_model(monkeypatch)

    # The uninterrupted reference, through the same entry point.
    _patch_study_fakes(monkeypatch, _deterministic_generate())
    reference = bundles.execute_run_bundle(
        bundle_path, verb="run", target_root=str(tmp_path / "reference"),
        prompts_path=prompts, package_evidence_on_complete=False)
    reference_dir = reference["runDirectory"]
    reference_lines = _read(os.path.join(reference_dir,
                                         "generations.jsonl")).splitlines(True)
    assert len(reference_lines) == 8  # 2 conditions x 3 prompts + 2 choices
    assert _status(reference_dir) is None
    assert not run_status.is_partial(reference_dir)

    target = str(tmp_path / "target")
    record_path = str(tmp_path / "records" / "job1.json")
    run = dict(verb="run", target_root=target, prompts_path=prompts,
               package_evidence_on_complete=False, record_path=record_path)

    # First attempt: cancelled during the third response, which finishes;
    # the run parks and says so.
    flag = resume.CheckpointFlag()
    _patch_study_fakes(monkeypatch,
                       _deterministic_generate([0], arm_flag_at=3, flag=flag))
    with pytest.raises(resume.CheckpointRequested) as parked:
        bundles.execute_run_bundle(bundle_path, checkpoint=flag, **run)
    run_dir = parked.value.run_directory
    first_park = len(_read(os.path.join(run_dir,
                                        "generations.jsonl")).splitlines())
    assert 0 < first_park < 6
    status = _status(run_dir)
    assert status["status"] == "checkpointed"
    assert status["itemsWritten"] == first_park
    assert run_status.is_partial(run_dir)

    # Second attempt: it went on two more responses and was killed while
    # saving its place. What that leaves: the record lines past the saved
    # place, the PREVIOUS state, and an empty temp file.
    with open(os.path.join(run_dir, "generations.jsonl"), "ab") as handle:
        handle.writelines(reference_lines[first_park:first_park + 2])
    open(_tmp_state(run_dir), "w").close()
    assert resume.read_state(run_dir)["completedRecords"] == first_park

    # Third attempt: the requeued script follows its pointer and finishes.
    after = [0]
    _patch_study_fakes(monkeypatch, _deterministic_generate(after))
    result = bundles.execute_run_bundle(bundle_path, **run)
    assert result["runDirectory"] == run_dir
    # Nothing generated twice: only the sampled responses still missing.
    assert after[0] == sum(
        1 for line in reference_lines[first_park + 2:]
        if resume.record_kind(json.loads(line)) == resume.KIND_SAMPLED)

    assert _read(os.path.join(run_dir, "generations.jsonl")) \
        == b"".join(reference_lines)
    assert resume.is_complete(run_dir)
    assert not os.path.exists(resume.state_path(run_dir))
    assert not os.path.exists(_tmp_state(run_dir))
    status = _status(run_dir)
    assert status["status"] == "completed"
    assert status["evidenceComplete"] is True
    assert status["itemsWritten"] == 8
    assert "error" not in status and "errorType" not in status
    assert status["resumedFrom"]["status"] == "checkpointed"
    assert status["resumedFrom"]["itemsWritten"] == first_park
    assert status["resumedFrom"]["errorType"] == "CheckpointRequested"
    assert status["finishedAt"] >= status["resumedFrom"]["finishedAt"]
    # The shape the Swift reader's fixture (RunStatusFileTests.settledStatus)
    # copies.
    assert sorted(status) == [
        "evidenceComplete", "experiment", "finishedAt", "invalidResponses",
        "itemLabel", "itemsWritten", "resumedFrom", "schemaVersion", "stage",
        "startedAt", "status"]
    assert sorted(status["resumedFrom"]) == [
        "error", "errorType", "finishedAt", "itemsWritten", "status"]
    assert not run_status.is_partial(run_dir)
    record = json.loads(_read(record_path))
    assert record["status"] == "succeeded"


def test_a_run_cancelled_and_never_resumed_stays_partial(tmp_path,
                                                         monkeypatch):
    source = str(tmp_path / "source")
    prompts = _study_fixture(source, "stays")
    bundle_path = bundles.package_experiment("stays", root=source)["bundlePath"]
    _loads_without_a_model(monkeypatch)
    flag = resume.CheckpointFlag()
    _patch_study_fakes(monkeypatch,
                       _deterministic_generate([0], arm_flag_at=2, flag=flag))
    with pytest.raises(resume.CheckpointRequested) as parked:
        bundles.execute_run_bundle(
            bundle_path, verb="run", target_root=str(tmp_path / "target"),
            prompts_path=prompts, package_evidence_on_complete=False,
            record_path=str(tmp_path / "records" / "job1.json"),
            checkpoint=flag)
    run_dir = parked.value.run_directory
    assert _status(run_dir)["status"] == "checkpointed"
    assert run_status.is_partial(run_dir)
    assert resume.is_resumable(run_dir)


def test_a_job_stopped_after_its_report_settles_on_re_execution(
        tmp_path, monkeypatch):
    """A continuation stopped after writing report.json and before settling
    its status re-executes into the already-complete path, which finishes
    the bookkeeping instead of packaging the parked status."""
    source = str(tmp_path / "source")
    prompts = _study_fixture(source, "late")
    bundle_path = bundles.package_experiment("late", root=source)["bundlePath"]
    _loads_without_a_model(monkeypatch)
    _patch_study_fakes(monkeypatch, _deterministic_generate())
    run = dict(verb="run", target_root=str(tmp_path / "target"),
               prompts_path=prompts, package_evidence_on_complete=False,
               record_path=str(tmp_path / "records" / "job1.json"))
    run_dir = bundles.execute_run_bundle(bundle_path, **run)["runDirectory"]
    # The stop's leftovers: the parked status and the saved place.
    _park_status(run_dir, items=3)
    resume.write_state(run_dir, run_id=os.path.basename(run_dir), verb="run",
                       completed_records=3, reason="signal")
    assert run_status.is_partial(run_dir)

    monkeypatch.setattr(tasks, "run", lambda *a, **k: pytest.fail(
        "a complete run must not run again"))
    result = bundles.execute_run_bundle(bundle_path, **run)
    assert result["alreadyComplete"] is True
    assert _status(run_dir)["status"] == "completed"
    assert not run_status.is_partial(run_dir)
    assert resume.read_state(run_dir) is None


# --- 2. what is never settled ----------------------------------------------------

def _completed_dir(tmp_path, name="run"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "report.json").write_text("{}", encoding="utf-8")
    return str(run_dir)


@pytest.mark.parametrize("parked_status", ["checkpointed", "inProgress"])
def test_a_parked_status_is_settled_once_the_report_exists(tmp_path,
                                                           parked_status):
    run_dir = _completed_dir(tmp_path)
    status = run_status.RunStatus(run_dir, stage="run", item_label="record")
    status.item_count = 4
    status.status = parked_status
    status.write()

    assert run_status.settle_after_resumed_completion(
        run_dir, completion_file="report.json", items_written=9)
    settled = _status(run_dir)
    assert settled["status"] == "completed"
    assert settled["itemsWritten"] == 9
    assert settled["resumedFrom"]["status"] == parked_status
    assert settled["resumedFrom"]["itemsWritten"] == 4
    assert not run_status.is_partial(run_dir)
    # Settling twice changes nothing.
    assert not run_status.settle_after_resumed_completion(
        run_dir, completion_file="report.json", items_written=9)


def test_a_parked_status_without_the_report_is_left_parked(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _park_status(str(run_dir))
    assert not run_status.settle_after_resumed_completion(
        str(run_dir), completion_file="report.json")
    assert _status(str(run_dir))["status"] == "checkpointed"
    assert run_status.is_partial(str(run_dir))


def test_a_failed_run_stays_failed(tmp_path):
    run_dir = _completed_dir(tmp_path)
    status = run_status.RunStatus(run_dir, stage="run", item_label="record")
    status.fail(RuntimeError("the judge refused"))
    before = _read(os.path.join(run_dir, run_status.STATUS_FILENAME))

    assert not run_status.settle_after_resumed_completion(
        run_dir, completion_file="report.json")
    assert _read(os.path.join(run_dir, run_status.STATUS_FILENAME)) == before
    assert os.path.exists(os.path.join(run_dir,
                                       run_status.FAILURE_NOTE_FILENAME))
    assert run_status.is_partial(run_dir)


def test_a_parked_status_beside_a_failure_note_is_left_alone(tmp_path):
    run_dir = _completed_dir(tmp_path)
    _park_status(run_dir)
    with open(os.path.join(run_dir, run_status.FAILURE_NOTE_FILENAME), "w",
              encoding="utf-8") as handle:
        handle.write("# run FAILED\n")
    assert not run_status.settle_after_resumed_completion(
        run_dir, completion_file="report.json")
    assert _status(run_dir)["status"] == "checkpointed"


def test_an_uninterrupted_run_gets_no_status_and_a_torn_one_stays_torn(
        tmp_path):
    clean = _completed_dir(tmp_path, "clean")
    assert not run_status.settle_after_resumed_completion(
        clean, completion_file="report.json")
    assert _status(clean) is None

    torn = _completed_dir(tmp_path, "torn")
    with open(os.path.join(torn, run_status.STATUS_FILENAME), "w",
              encoding="utf-8") as handle:
        handle.write('{"status": "checkp')
    assert not run_status.settle_after_resumed_completion(
        torn, completion_file="report.json")
    assert _status(torn) is run_status.UNREADABLE
    assert run_status.is_partial(torn)


# --- 3. the temp file a killed state write leaves ----------------------------------

def test_completion_and_resume_remove_an_unfinished_state_write(tmp_path):
    run_dir = str(tmp_path)
    open(_tmp_state(run_dir), "w").close()
    resume.clear_state(run_dir)
    assert not os.path.exists(_tmp_state(run_dir))

    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"condition": "baseline", "promptIndex": 0,
                                 "promptID": "p0", "sampleIndex": 0}) + "\n")
    resume.write_state(run_dir, run_id="r", verb="run", completed_records=1,
                       reason="signal")
    open(_tmp_state(run_dir), "w").close()
    writer = resume.GenerationWriter(run_dir, resume=True)
    writer.close()
    assert not os.path.exists(_tmp_state(run_dir))
    assert resume.read_state(run_dir)["completedRecords"] == 1


# --- 4. sweeps and panels settle the same way ---------------------------------------

def test_a_parked_sweep_resumed_to_completion_reads_complete(tmp_path,
                                                             monkeypatch):
    root = str(tmp_path)
    _sweep_workspace(root, "sw")
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {"fear": _sweep_bundle()})
    monkeypatch.setattr(_owner_choice_scoring, "_score_choice", _fake_score)
    flag = resume.CheckpointFlag()
    calls: list = []

    def tripping_generate(model, prompt, *, injections=None, **kwargs):
        calls.append(prompt)
        if len(calls) == 4:
            flag.request()
        return "the town woke slowly 2"
    monkeypatch.setattr(_owner_generate, "generate", tripping_generate)
    with pytest.raises(resume.CheckpointRequested) as parked:
        tasks.sweep("sw", root, model_provider=_sweep_model, checkpoint=flag,
                    log=lambda *_: None)
    run_dir = parked.value.run_directory
    _park_status(run_dir, stage="sweep", items=parked.value.completed_records)

    monkeypatch.setattr(_owner_generate, "generate",
                        lambda model, prompt, **kwargs: "the town woke slowly 2")
    tasks.sweep("sw", root, model_provider=_sweep_model, run_directory=run_dir,
                log=lambda *_: None)
    assert resume.is_complete(run_dir, "sweep")
    status = _status(run_dir)
    assert status["status"] == "completed"
    assert status["resumedFrom"]["status"] == "checkpointed"
    assert not run_status.is_partial(run_dir)


def test_a_parked_panel_resumed_to_completion_reads_complete(tmp_path,
                                                             monkeypatch):
    root, manifest = _panel_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(multi_agent, "generate",
                        _panel_generate([0], kill_at=5))
    seen = {}
    with pytest.raises(Killed):
        panel_workflow.run_multi_agent_study(
            "panel-a", manifest, _panel_model(), str(root), log=lambda _: None,
            on_run_directory=lambda created: seen.setdefault("dir", created))
    run_dir = seen["dir"]
    _park_status(run_dir, items=4)
    assert run_status.is_partial(run_dir)

    monkeypatch.setattr(multi_agent, "generate", _panel_generate([0]))
    panel_workflow.run_multi_agent_study(
        "panel-a", manifest, _panel_model(), str(root), log=lambda _: None,
        run_directory=run_dir)
    assert resume.is_complete(run_dir)
    status = _status(run_dir)
    assert status["status"] == "completed"
    assert status["resumedFrom"]["status"] == "checkpointed"
    assert not run_status.is_partial(run_dir)


# --- 5. the resume message counts what the continuation keeps ----------------------

def test_the_resume_message_counts_the_record_lines_kept(tmp_path, fake_slurm,
                                                         monkeypatch):
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "counted")
    _patch_study_fakes(monkeypatch, _deterministic_generate())
    reference_dir = tasks.run("counted", prompts, root,
                              model_provider=_fake_model, log=lambda *_: None)
    reference_lines = _read(os.path.join(
        reference_dir, "generations.jsonl")).splitlines(True)

    mgr = _manager(tmp_path)
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9701",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "recordsDirectory": str(records)})
    pointer = resume.pointer_path_for_record(str(records / f"{job.id}.json"))
    flag = resume.CheckpointFlag()
    _patch_study_fakes(monkeypatch,
                       _deterministic_generate([0], arm_flag_at=2, flag=flag))
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("counted", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda created: resume.write_pointer(
                      pointer, created, verb="run", experiment="counted"))
    run_dir = resume.read_pointer(pointer)["runDirectory"]
    saved = resume.read_state(run_dir)["completedRecords"]
    # A continuation went on and was killed while saving its place.
    with open(os.path.join(run_dir, "generations.jsonl"), "ab") as handle:
        handle.writelines(reference_lines[saved:saved + 3])
    open(_tmp_state(run_dir), "w").close()
    kept = saved + 3
    assert resume.read_state(run_dir)["completedRecords"] == saved

    assert mgr.cancel(job.id) is True
    _end_marker(script, "9701")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9702")
    outcome = mgr.resubmit(job.id)
    assert outcome["completedRecords"] == kept
    assert f"The {kept} response records it had completed are kept" \
        in outcome["message"]
    assert mgr.get(job.id).result["resumedAfterCancel"]["completedRecords"] \
        == kept
