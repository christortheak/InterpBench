"""Partial-evidence retention (2026-07-24).

External review ``docs/CLUSTER-SHARDING-JUDGING-REVIEW-2026-07-23.md``: a
failed evaluate used to leave NOTHING on disk — the run directory was
created after the whole panel finished, so an invalid verdict on the last
pair destroyed every successful judgment before it. The researcher was left
with "winner None (expected 'A', 'B', or 'tie')" and an SSH session.

Contract under test: successful judgments survive a mid-panel failure,
malformed judge responses are preserved verbatim, the surviving directory
says plainly that it failed — and it is never mistakable for a result.
"""

import json
import time
import os
import tarfile

import pytest

from steerlab_server.experiment import bundles, paired_judge, run_status, tasks

from test_evaluate_deferred import _fixture
from test_resume_checkpoint import _bundle_fixture


def _evaluate_runs(root):
    runs = os.path.join(root, "runs")
    return sorted(e for e in os.listdir(runs) if e.endswith("-exp-ev-evaluate"))


def _only_evaluate_run(root):
    found = _evaluate_runs(root)
    assert len(found) == 1, found
    return os.path.join(root, "runs", found[0])


def test_failure_mid_panel_keeps_the_judgments_that_succeeded(
        tmp_path, monkeypatch):
    root, _run = _fixture(tmp_path, judges=[
        {"name": "j1", "kind": "claude"}, {"name": "j2", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = []

    def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
        calls.append(task_prompt)
        # Judge 1 judges both pairs; judge 2 dies on its first call — the
        # shape of the live incident (a panel that partly succeeded).
        if len(calls) > 2:
            raise RuntimeError("judge transport exploded")
        return {"winner": "tie", "confidence": 0.5}

    monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
    with pytest.raises(RuntimeError, match="exploded"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)

    out = _only_evaluate_run(root)
    # The two judgments judge 1 actually produced are ON DISK.
    rows = [json.loads(line) for line in
            open(os.path.join(out, "judgments.jsonl"), encoding="utf-8")]
    assert len(rows) == 2
    assert {r["judge"] for r in rows} == {"j1"}

    # ... and the directory says what happened, naming what did not run.
    status = run_status.read_status(out)
    assert status["status"] == "failed"
    assert status["evidenceComplete"] is False
    assert status["itemsWritten"] == 2
    assert status["completedUnits"] == ["j1"]
    assert status["pendingUnits"] == ["j2"]
    assert "exploded" in status["error"]
    assert status["errorType"] == "RuntimeError"

    # A failure record, never a result: no judge-report.json is written, and
    # the human-readable note leads with that distinction.
    assert not os.path.exists(os.path.join(out, "judge-report.json"))
    note = open(os.path.join(out, run_status.FAILURE_NOTE_FILENAME),
                encoding="utf-8").read()
    assert "failure record" in note
    assert "j2" in note
    assert run_status.is_partial(out)


def test_malformed_judge_responses_are_preserved_verbatim(
        tmp_path, monkeypatch):
    # The incident that motivated the review: the parser refused and the
    # raw text was discarded, so there was nothing to look at. A retry that
    # SUCCEEDS still leaves its failed attempt on disk — "the judge needed
    # two tries" is exactly what a quiet retry erases.
    root, _run = _fixture(tmp_path, judges=[{"name": "j1", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    attempts = []

    def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise paired_judge.JudgeResponseError(
                "unbalanced JSON in judge response",
                'Sure! Here is my verdict: {"winner": "A", "confid')
        return {"winner": "tie", "confidence": 0.5}

    monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)

    failures = [json.loads(line) for line in
                open(os.path.join(out, run_status.INVALID_RESPONSES_FILENAME),
                     encoding="utf-8")]
    assert len(failures) == 1
    assert failures[0]["rawResponse"].startswith("Sure! Here is my verdict:")
    assert failures[0]["error"] == "unbalanced JSON in judge response"
    assert failures[0]["attempt"] == 1
    # The evaluation itself SUCCEEDED — the retry worked — so this is a
    # complete result that also records the wobble.
    assert os.path.exists(os.path.join(out, "judge-report.json"))
    assert run_status.read_status(out)["invalidResponses"] == 1
    assert not run_status.is_partial(out)


def test_invalid_winner_refusal_keeps_the_raw_verdict(tmp_path, monkeypatch):
    # The literal "winner None" incident: twice-invalid refuses the phase
    # (unchanged — the parser is NOT weakened), but both raw attempts and
    # every judgment produced before it now survive.
    root, _run = _fixture(tmp_path, judges=[{"name": "j1", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = []

    def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
        calls.append(task_prompt)
        if len(calls) == 1:
            return {"winner": "tie", "confidence": 0.5}
        return {"winner": None, "brief_reason": "both were good"}

    monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
    # 1 noncompliant of 2 pairs is past the 25% cap, so the phase still
    # refuses — but the refusal is now the SYSTEMIC message, and the
    # noncompliant pair survives as a recorded row beside the good one
    # (2026-08-09).
    with pytest.raises(RuntimeError, match="systemic judge failure"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)

    out = _only_evaluate_run(root)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "judgments.jsonl"), encoding="utf-8")]
    assert len(rows) == 2
    assert sum(1 for r in rows if r.get("noncompliant")) == 1
    failures = [json.loads(line) for line in
                open(os.path.join(out, run_status.INVALID_RESPONSES_FILENAME),
                     encoding="utf-8")]
    assert [f["attempt"] for f in failures] == [1, 2]
    assert all(f["verdict"] == {"winner": None,
                                "brief_reason": "both were good"}
               for f in failures)
    assert run_status.is_partial(out)


def test_isolated_noncompliance_is_stamped_on_the_written_report(
        tmp_path, monkeypatch):
    # The completion side of the 2026-08-09 policy: a run that survives
    # isolated judge noncompliance must SAY so in judge-report.json. The
    # count used to reach only the in-memory report paired_judge returned —
    # the on-disk artifact claimed a complete column.
    root, run_dir = _fixture(tmp_path, judges=[
        {"name": "j1", "kind": "claude"}, {"name": "j2", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Two extra pairs: 1 noncompliant of 4 sits AT the 25% cap, not past it,
    # so the evaluation completes.
    with open(os.path.join(run_dir, "generations.jsonl"), "a",
              encoding="utf-8") as handle:
        for i in (2, 3):
            for condition, output in (("baseline", f"base {i}"),
                                      ("fear", f"steered {i}")):
                handle.write(json.dumps(
                    {"promptID": f"p{i}", "seed": 0, "condition": condition,
                     "prompt": f"Describe room {i}.",
                     "output": output}) + "\n")

    afraid_calls = []

    def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
        # Judge columns run sequentially and a noncompliant pair burns both
        # attempts back to back, so the first two calls on the "afraid" pair
        # are j1's column; j2 later judges the same pair cleanly.
        if "afraid" in (a, b):
            afraid_calls.append(1)
            if len(afraid_calls) <= 2:
                return {"winner": None, "brief_reason": "cannot decide"}
        return {"winner": "tie", "confidence": 0.5}

    monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)

    report = json.load(open(os.path.join(out, "judge-report.json"),
                            encoding="utf-8"))
    blocks = {b["name"]: b for b in report["judges"]}
    assert blocks["j1"]["noncompliantJudgments"] == 1
    # Nonzero-only, like salvagedVerdicts: the clean column carries no key.
    assert "noncompliantJudgments" not in blocks["j2"]
    # ... and the report-level panel total matches the per-judge sum.
    assert report["noncompliantJudgments"] == 1
    # A disclosed-incomplete column is still a completed result.
    assert not run_status.is_partial(out)


def test_completed_evaluate_is_marked_complete(tmp_path, monkeypatch):
    root, _run = _fixture(tmp_path, judges=[
        {"name": "j1", "kind": "claude"}, {"name": "j2", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        paired_judge, "judge_pair",
        lambda *a, **k: {"winner": "tie", "confidence": 0.5})
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)

    status = run_status.read_status(out)
    assert status["status"] == "completed"
    assert status["evidenceComplete"] is True
    assert status["pendingUnits"] == []
    assert status["itemsWritten"] == 4
    assert not run_status.is_partial(out)
    assert not os.path.exists(
        os.path.join(out, run_status.FAILURE_NOTE_FILENAME))
    assert not os.path.exists(
        os.path.join(out, run_status.INVALID_RESPONSES_FILENAME))


def _bundle_names(path):
    with tarfile.open(path, "r:gz") as tar:
        return set(tar.getnames())


def test_failed_bundle_execute_packages_what_survived(tmp_path, monkeypatch):
    # The gap that stranded data on the cluster: the failure path recorded
    # the error and re-raised, so nothing was ever packaged and the app's
    # import affordance (which keys on a returned bundle) had nothing to
    # offer.
    bundle_path, target, record_path = _bundle_fixture(tmp_path)
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    # Scheduler logs live beside the submission, not in the run directory.
    with open(os.path.join(os.path.dirname(record_path), "slurm-77.err"),
              "w", encoding="utf-8") as handle:
        handle.write("CUDA out of memory\n")

    def failing_run(name, prompts_path=None, root=None, dtype="auto",
                    device=None, *, checkpoint=None, run_directory=None,
                    on_run_directory=None, **kwargs):
        run_dir = os.path.join(root, "runs", "20260724T000000000-exp-bexec-run")
        os.makedirs(run_dir, exist_ok=True)
        on_run_directory(run_dir)
        with open(os.path.join(run_dir, "generations.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write('{"condition": "baseline", "promptID": "p0"}\n')
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(tasks, "run", failing_run)
    with pytest.raises(RuntimeError, match="out of memory"):
        bundles.execute_run_bundle(bundle_path, verb="run",
                                   target_root=target,
                                   record_path=record_path)

    record = json.load(open(record_path, encoding="utf-8"))
    assert record["status"] == "failed"
    result = record["result"]
    assert result["partialEvidence"] is True
    bundle = result["evidenceBundle"]
    # Marked partial in the NAME (obvious in a listing) while keeping the
    # suffix every existing scanner and importer matches on.
    assert bundle["bundlePath"].endswith(".partial.evidence-bundle.tar.gz")
    assert bundle["evidenceComplete"] is False
    assert bundle["failure"]["errorType"] == "RuntimeError"
    assert bundle["failure"]["verb"] == "run"
    assert "out of memory" in bundle["failure"]["traceback"]

    names = _bundle_names(bundle["bundlePath"])
    assert any(n.endswith("generations.jsonl") for n in names)
    assert "diagnostics/slurm-77.err" in names


def test_failure_before_any_directory_packages_nothing(tmp_path, monkeypatch):
    # Nothing was produced, so there is nothing to retrieve — which is a
    # different (and honest) outcome from losing it. No bundle, no claim.
    bundle_path, target, record_path = _bundle_fixture(tmp_path)

    def failing_run(name, prompts_path=None, root=None, dtype="auto",
                    device=None, *, checkpoint=None, run_directory=None,
                    on_run_directory=None, **kwargs):
        raise RuntimeError("manifest verification failed")

    monkeypatch.setattr(tasks, "run", failing_run)
    with pytest.raises(RuntimeError, match="verification"):
        bundles.execute_run_bundle(bundle_path, verb="run",
                                   target_root=target,
                                   record_path=record_path)
    result = json.load(open(record_path, encoding="utf-8"))["result"]
    assert "evidenceBundle" not in result
    assert "partialEvidence" not in result
    assert "verification" in result["error"]


def test_packaging_failure_never_masks_the_real_error(tmp_path, monkeypatch):
    bundle_path, target, record_path = _bundle_fixture(tmp_path)

    def failing_run(name, prompts_path=None, root=None, dtype="auto",
                    device=None, *, checkpoint=None, run_directory=None,
                    on_run_directory=None, **kwargs):
        run_dir = os.path.join(root, "runs", "20260724T000000000-exp-bexec-run")
        os.makedirs(run_dir, exist_ok=True)
        on_run_directory(run_dir)
        raise RuntimeError("the real problem")

    def exploding_package(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tasks, "run", failing_run)
    monkeypatch.setattr(bundles, "package_evidence", exploding_package)
    # The ORIGINAL error propagates — a researcher must not end up
    # debugging the packager instead of the run.
    with pytest.raises(RuntimeError, match="the real problem"):
        bundles.execute_run_bundle(bundle_path, verb="run",
                                   target_root=target,
                                   record_path=record_path)
    result = json.load(open(record_path, encoding="utf-8"))["result"]
    assert result["evidencePackagingError"] == "OSError: disk full"
    assert "evidenceBundle" not in result


def test_partial_bundle_of_a_failed_evaluate_carries_the_judgments(
        tmp_path, monkeypatch):
    # End to end: the evaluate retention fix and the packaging fix meet —
    # a mid-panel judge failure produces a bundle the Mac can import,
    # containing the judgments that succeeded and the raw bad response.
    root, _run = _fixture(tmp_path, judges=[{"name": "j1", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = []

    def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
        calls.append(task_prompt)
        if len(calls) == 1:
            return {"winner": "tie", "confidence": 0.5}
        raise paired_judge.JudgeResponseError("no JSON", "I refuse to judge.")

    monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
    # 1 of 2 pairs noncompliant exceeds the cap: the phase still fails,
    # under the systemic-failure message (2026-08-09).
    with pytest.raises(RuntimeError, match="systemic judge failure"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)

    out = _only_evaluate_run(root)
    meta = bundles.package_evidence(
        out, failure={"error": "judge refused", "errorType": "RuntimeError"})
    assert meta["evidenceComplete"] is False
    names = _bundle_names(meta["bundlePath"])
    assert any(n.endswith("judgments.jsonl") for n in names)
    assert any(n.endswith(run_status.INVALID_RESPONSES_FILENAME)
               for n in names)
    assert any(n.endswith(run_status.FAILURE_NOTE_FILENAME) for n in names)
    assert not any(n.endswith("judge-report.json") for n in names)


def test_partial_validate_run_is_never_freeze_evidence(tmp_path):
    # The other half of the retention principle: a failed run's data comes
    # HOME, but it must never satisfy an evidence-grade gate. A partial
    # validate run that happens to carry an evidence stamp (a stale one, a
    # hand-copied one) is still not evidence.
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    from test_evidence_tier import _validate_evidence

    root = str(tmp_path)
    es.create("s", model_id="org/m", revision="abc", root=root)
    _validate_evidence(root, "s")
    scope = Manifest.load("s", root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", "v-exp-s-validate")

    # Complete as far as the artifacts go — and accepted, today.
    assert es._matching_validate_evidence(scope, root) is not None

    # Now mark it for what it is. The artifacts did not change; the run's
    # own account of itself did, and that is enough to disqualify it.
    status = run_status.RunStatus(rundir, stage="validate", experiment="s")
    status.fail(RuntimeError("node evicted"))
    assert run_status.is_partial(rundir)
    assert es._matching_validate_evidence(scope, root) is None


class TestEveryVerbComesHome:
    """Retention is not evaluate-only (2026-07-24, round 2).

    `run` reports its directory via a callback and `evaluate` attaches its
    own to the exception, but `extract`, `validate`, and `sweep` name theirs
    nowhere — so a failure in those packaged nothing at all. Those are also
    the stages where the GPU hours live.
    """

    def _fail_verb(self, tmp_path, monkeypatch, verb, make_dir=True):
        bundle_path, target, record_path = _bundle_fixture(tmp_path)

        def failing(name, root=None, *a, **k):
            if make_dir:
                run_dir = os.path.join(
                    root, "runs", f"20260724T120000000-exp-bexec-{verb}")
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "notes.txt"), "w",
                          encoding="utf-8") as handle:
                    handle.write("partial work\n")
            raise RuntimeError(f"{verb} blew up")

        monkeypatch.setattr(tasks, verb, failing)
        with pytest.raises(RuntimeError, match="blew up"):
            bundles.execute_run_bundle(bundle_path, verb=verb,
                                       target_root=target,
                                       record_path=record_path)
        return target, json.load(open(record_path, encoding="utf-8"))["result"]

    @pytest.mark.parametrize("verb", ["extract", "validate", "sweep"])
    def test_unreported_directories_are_recovered_marked_and_packaged(
            self, tmp_path, monkeypatch, verb):
        target, result = self._fail_verb(tmp_path, monkeypatch, verb)
        assert result["partialEvidence"] is True
        bundle = result["evidenceBundle"]
        assert bundle["evidenceComplete"] is False

        run_dir = os.path.join(
            target, "runs", f"20260724T120000000-exp-bexec-{verb}")
        status = run_status.read_status(run_dir)
        assert status["stage"] == verb
        assert status["status"] == "failed"
        assert f"{verb} blew up" in status["error"]
        assert os.path.isfile(
            os.path.join(run_dir, run_status.FAILURE_NOTE_FILENAME))
        names = _bundle_names(bundle["bundlePath"])
        assert any(n.endswith("notes.txt") for n in names)

    def test_a_stage_that_created_nothing_packages_nothing(
            self, tmp_path, monkeypatch):
        # Honest: no directory means nothing was produced, which is a
        # different outcome from losing it.
        _target, result = self._fail_verb(
            tmp_path, monkeypatch, "sweep", make_dir=False)
        assert "evidenceBundle" not in result
        assert "partialEvidence" not in result

    def test_ambiguous_candidates_refuse_to_guess(self, tmp_path, monkeypatch):
        # Packaging the WRONG run is worse than packaging none. Two
        # candidate directories (a concurrent job in the same root) must
        # not resolve to a coin flip.
        bundle_path, target, record_path = _bundle_fixture(tmp_path)

        def failing(name, root=None, *a, **k):
            for stamp in ("20260724T120000000", "20260724T120000001"):
                os.makedirs(os.path.join(root, "runs",
                                         f"{stamp}-exp-bexec-sweep"),
                            exist_ok=True)
            raise RuntimeError("sweep blew up")

        monkeypatch.setattr(tasks, "sweep", failing)
        with pytest.raises(RuntimeError, match="blew up"):
            bundles.execute_run_bundle(bundle_path, verb="sweep",
                                       target_root=target,
                                       record_path=record_path)
        result = json.load(open(record_path, encoding="utf-8"))["result"]
        assert "evidenceBundle" not in result

    def test_run_marks_its_partial_generations(self, tmp_path, monkeypatch):
        bundle_path, target, record_path = _bundle_fixture(tmp_path)

        def failing_run(name, prompts_path=None, root=None, dtype="auto",
                        device=None, *, checkpoint=None, run_directory=None,
                        on_run_directory=None, **kwargs):
            run_dir = os.path.join(root, "runs",
                                   "20260724T130000000-exp-bexec-run")
            os.makedirs(run_dir, exist_ok=True)
            on_run_directory(run_dir)
            with open(os.path.join(run_dir, "generations.jsonl"), "w",
                      encoding="utf-8") as handle:
                handle.write('{"condition": "baseline", "promptID": "p0"}\n')
            raise RuntimeError("node evicted")

        monkeypatch.setattr(tasks, "run", failing_run)
        with pytest.raises(RuntimeError, match="evicted"):
            bundles.execute_run_bundle(bundle_path, verb="run",
                                       target_root=target,
                                       record_path=record_path)
        run_dir = os.path.join(target, "runs",
                               "20260724T130000000-exp-bexec-run")
        status = run_status.read_status(run_dir)
        assert status["stage"] == "run"
        assert status["status"] == "failed"
        # Counted in the stage's own units, so the note reads sensibly.
        assert status["itemLabel"] == "record"
        assert status["itemsWritten"] == 1
        assert run_status.is_partial(run_dir)

    def test_a_stages_own_status_is_never_overwritten(self, tmp_path,
                                                      monkeypatch):
        # evaluate names which judges finished and which did not; the
        # generic bundle-level marker must not flatten that into "failed".
        root, partial = None, None
        root, _run = _fixture(tmp_path, judges=[
            {"name": "j1", "kind": "claude"}, {"name": "j2", "kind": "claude"}])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        calls = []

        def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
            calls.append(1)
            if len(calls) > 2:
                raise RuntimeError("boom")
            return {"winner": "tie", "confidence": 0.5}

        monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
        with pytest.raises(RuntimeError):
            tasks.evaluate("ev", root=root, log=lambda *_: None)
        out = _only_evaluate_run(root)
        before = run_status.read_status(out)

        bundles._mark_partial_run(out, verb="evaluate", name="ev",
                                  exc=RuntimeError("generic"))
        assert run_status.read_status(out) == before
        assert run_status.read_status(out)["pendingUnits"] == ["j2"]


def test_checkpointed_runs_are_parked_not_failed(tmp_path, monkeypatch):
    # Checkpoint-on-signal is the reliability path WORKING. It is still
    # incomplete (no gate may take it), but calling a successful requeue a
    # failure would train the researcher to ignore the marker that matters.
    from steerlab_server.experiment import resume as resume_mod

    bundle_path, target, record_path = _bundle_fixture(tmp_path)

    def checkpointing_run(name, prompts_path=None, root=None, dtype="auto",
                          device=None, *, checkpoint=None, run_directory=None,
                          on_run_directory=None, **kwargs):
        run_dir = os.path.join(root, "runs", "20260724T140000000-exp-bexec-run")
        os.makedirs(run_dir, exist_ok=True)
        on_run_directory(run_dir)
        raise resume_mod.CheckpointRequested(run_dir, "run", 7)

    monkeypatch.setattr(tasks, "run", checkpointing_run)
    with pytest.raises(resume_mod.CheckpointRequested):
        bundles.execute_run_bundle(bundle_path, verb="run",
                                   target_root=target,
                                   record_path=record_path)
    run_dir = os.path.join(target, "runs", "20260724T140000000-exp-bexec-run")
    status = run_status.read_status(run_dir)
    assert status["status"] == "checkpointed"
    assert status["itemsWritten"] == 7
    # Incomplete, so still barred from evidence gates ...
    assert run_status.is_partial(run_dir)
    # ... but NOT dressed up as a failure.
    assert not os.path.exists(
        os.path.join(run_dir, run_status.FAILURE_NOTE_FILENAME))


class TestDirectJobRetention:
    """External review 2026-07-24, finding 2.

    `execute_run_bundle` packaged failed BUNDLED work, but the direct
    /api/experiment/{name}/{verb} route runs tasks IN-PROCESS inside a
    durable job — so a failed server-resident evaluate/sweep/run recorded
    the error and stranded its output in scratch. Retention now lives in
    the durable-job layer, which covers every job kind rather than one
    route.
    """

    def _manager(self, tmp_path):
        from steerlab_server.api.jobs import DurableJobStore, JobManager
        return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                          sweep_orphans=False)

    def _await(self, manager, job_id, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = manager.get(job_id)
            if job and job.status in ("succeeded", "failed", "cancelled"):
                return job
            time.sleep(0.02)
        raise AssertionError("job did not finish")

    def test_a_failed_in_process_job_packages_what_it_produced(
            self, tmp_path, monkeypatch):
        from steerlab_server.experiment import paths

        monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
        manager = self._manager(tmp_path)

        def failing(job):
            # Exactly how a direct verb behaves: create a run directory
            # through the normal helper, write some output, then die.
            run_dir = paths.make_unique_run_directory(
                "exp-s-sweep", str(tmp_path))
            with open(os.path.join(run_dir, "sweep.csv"), "w",
                      encoding="utf-8") as handle:
                handle.write("layer,alpha\n12,4.0\n")
            raise RuntimeError("CUDA out of memory")

        job = manager.submit("experiment:sweep", failing)
        finished = self._await(manager, job.id)

        assert finished.status == "failed"
        assert "out of memory" in finished.error
        # The data came home rather than staying in scratch.
        result = finished.result or {}
        assert result["partialEvidence"] is True
        bundle = result["evidenceBundle"]
        assert bundle["evidenceComplete"] is False
        assert bundle["failure"]["errorType"] == "RuntimeError"
        names = _bundle_names(bundle["bundlePath"])
        assert any(n.endswith("sweep.csv") for n in names)
        # ... and it is marked as the failure record it is.
        assert run_status.is_partial(result["runDirectory"])
        assert os.path.isfile(os.path.join(result["runDirectory"],
                                           run_status.FAILURE_NOTE_FILENAME))

    def test_a_job_that_produced_nothing_records_nothing(self, tmp_path):
        manager = self._manager(tmp_path)

        def failing(job):
            raise RuntimeError("refused before doing any work")

        finished = self._await(manager, manager.submit("experiment:run",
                                                       failing).id)
        assert finished.status == "failed"
        assert "evidenceBundle" not in (finished.result or {})

    def test_a_succeeding_job_is_never_marked_partial(self, tmp_path,
                                                      monkeypatch):
        from steerlab_server.experiment import paths

        monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
        manager = self._manager(tmp_path)

        def ok(job):
            run_dir = paths.make_unique_run_directory(
                "exp-s-run", str(tmp_path))
            return {"runDirectory": run_dir}

        finished = self._await(manager, manager.submit("experiment:run", ok).id)
        assert finished.status == "succeeded"
        assert "partialEvidence" not in (finished.result or {})

    def test_one_jobs_directory_never_leaks_into_another(self, tmp_path,
                                                         monkeypatch):
        # The context variable is scoped per job. Without that, a job that
        # produced nothing would inherit the previous job's directory and
        # package the WRONG run — worse than packaging none.
        from steerlab_server.experiment import paths

        monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
        manager = self._manager(tmp_path)

        def produces(job):
            paths.make_unique_run_directory("exp-s-run", str(tmp_path))
            raise RuntimeError("first job died")

        def produces_nothing(job):
            raise RuntimeError("second job died before any work")

        first = self._await(manager, manager.submit("experiment:run",
                                                    produces).id)
        assert (first.result or {}).get("partialEvidence") is True
        second = self._await(manager, manager.submit("experiment:run",
                                                     produces_nothing).id)
        assert "evidenceBundle" not in (second.result or {})


class TestPipelineFailurePackagesTheChain:
    """External review round 2, finding 4.

    `package_evidence` follows `pipeline.json` to every stage directory —
    but only when handed the chain ROOT. Given a stage directory it
    packages that stage alone. So a pipeline that dies in evaluate would
    come home with the evaluate partial and WITHOUT the generations, which
    are the expensive GPU output and the reason retention exists.

    The priority was backwards: an incidentally-observed stage directory
    (exception attribute, context variable) outranked the explicitly
    reported chain root.
    """

    def _pipeline_fixture(self, tmp_path, monkeypatch, *, fail_in):
        from steerlab_server.experiment import paths

        bundle_path, target, record_path = _bundle_fixture(tmp_path)

        def failing_pipeline(name, root=None, dtype="auto", device=None, *,
                             checkpoint=None, pipeline_run_directory=None,
                             on_pipeline_directory=None, **kwargs):
            chain = paths.make_unique_run_directory(
                f"exp-{name}-pipeline", root)
            on_pipeline_directory(chain)
            # An earlier stage that SUCCEEDED — the generations we must not
            # lose — plus the ledger that points at it.
            generations = paths.make_unique_run_directory(
                f"exp-{name}-run", root)
            with open(os.path.join(generations, "generations.jsonl"), "w",
                      encoding="utf-8") as handle:
                handle.write('{"condition": "baseline", "promptID": "p0"}\n')
            with open(os.path.join(chain, "pipeline.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"disposition": "failed",
                           "stageResults": {"run": {"runDirectory": generations}}},
                          handle)
            # ... then the stage that died, created LAST so it is what the
            # context variable observes.
            stage = paths.make_unique_run_directory(
                f"exp-{name}-{fail_in}", root)
            with open(os.path.join(stage, "partial.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write("some work\n")
            raise RuntimeError(f"{fail_in} stage exploded")

        monkeypatch.setattr(tasks, "pipeline", failing_pipeline)
        with pytest.raises(RuntimeError, match="exploded"):
            bundles.execute_run_bundle(bundle_path, verb="pipeline",
                                       target_root=target,
                                       record_path=record_path)
        return target, json.load(open(record_path, encoding="utf-8"))["result"]

    @pytest.mark.parametrize("fail_in", ["evaluate", "sweep"])
    def test_the_whole_chain_is_packaged_not_just_the_failing_stage(
            self, tmp_path, monkeypatch, fail_in):
        _target, result = self._pipeline_fixture(
            tmp_path, monkeypatch, fail_in=fail_in)
        bundle = result["evidenceBundle"]
        assert bundle["evidenceComplete"] is False
        names = _bundle_names(bundle["bundlePath"])
        # The generations from the stage that SUCCEEDED survive ...
        assert any(n.endswith("generations.jsonl") for n in names), names
        # ... alongside the ledger that explains the chain ...
        assert any(n.endswith("pipeline.json") for n in names), names
        # ... and the partial output of the stage that died.
        assert any(n.endswith("partial.txt") for n in names), names

    def test_both_the_chain_root_and_the_failing_stage_are_marked(
            self, tmp_path, monkeypatch):
        target, result = self._pipeline_fixture(
            tmp_path, monkeypatch, fail_in="evaluate")
        runs = os.path.join(target, "runs")
        chain = [d for d in os.listdir(runs) if d.endswith("-pipeline")][0]
        stage = [d for d in os.listdir(runs) if d.endswith("-evaluate")][0]

        # The ROOT must be marked, or an imported chain reads as
        # unannotated — which reads as citable.
        root_status = run_status.read_status(os.path.join(runs, chain))
        assert root_status["stage"] == "pipeline"
        assert root_status["status"] == "failed"
        # The failing STAGE says which stage it was, not "pipeline".
        stage_status = run_status.read_status(os.path.join(runs, stage))
        assert stage_status["stage"] == "evaluate"
        assert stage_status["status"] == "failed"

    def test_a_non_pipeline_verb_still_packages_its_own_directory(
            self, tmp_path, monkeypatch):
        # The rule is scoped: only a pipeline prefers the reported root.
        # A plain run must still package the run directory itself.
        from steerlab_server.experiment import paths

        bundle_path, target, record_path = _bundle_fixture(tmp_path)

        def failing_run(name, prompts_path=None, root=None, dtype="auto",
                        device=None, *, checkpoint=None, run_directory=None,
                        on_run_directory=None, **kwargs):
            run_dir = paths.make_unique_run_directory(f"exp-{name}-run", root)
            on_run_directory(run_dir)
            with open(os.path.join(run_dir, "generations.jsonl"), "w",
                      encoding="utf-8") as handle:
                handle.write('{"condition": "baseline"}\n')
            raise RuntimeError("run exploded")

        monkeypatch.setattr(tasks, "run", failing_run)
        with pytest.raises(RuntimeError, match="exploded"):
            bundles.execute_run_bundle(bundle_path, verb="run",
                                       target_root=target,
                                       record_path=record_path)
        result = json.load(open(record_path, encoding="utf-8"))["result"]
        names = _bundle_names(result["evidenceBundle"]["bundlePath"])
        assert any(n.endswith("generations.jsonl") for n in names)


class TestStatusFileIntegrity:
    """External review 2026-07-24, finding 4 — the fail-open.

    The status file was written in place (truncate, then write) and a torn
    result read as "no status", which read as "legacy", which read as
    citable. So the crash that made a run partial could also make it look
    complete: the worst possible direction for a marker whose whole job is
    to say "do not trust this".
    """

    def test_write_is_atomic_and_leaves_no_temp_files(self, tmp_path):
        status = run_status.RunStatus(str(tmp_path), stage="run")
        status.note_item()
        status.complete()
        names = sorted(os.listdir(tmp_path))
        assert names == [run_status.STATUS_FILENAME]
        # A reader never sees a half-written file: replace() is atomic, so
        # the content is always one complete document.
        assert run_status.read_status(str(tmp_path))["status"] == "completed"

    def test_torn_status_fails_closed_as_partial(self, tmp_path):
        with open(os.path.join(tmp_path, run_status.STATUS_FILENAME), "w",
                  encoding="utf-8") as handle:
            handle.write('{"stage": "run", "sta')  # killed mid-write
        assert run_status.read_status(str(tmp_path)) is run_status.UNREADABLE
        assert run_status.is_partial(str(tmp_path))
        assert not run_status.has_readable_status(str(tmp_path))

    def test_absent_and_torn_are_distinguishable(self, tmp_path):
        # The distinction the fail-open collapsed: no file is a LEGACY run
        # (trusted); a broken file is a SUSPECT one.
        assert run_status.read_status(str(tmp_path)) is None
        assert not run_status.is_partial(str(tmp_path))
        open(os.path.join(tmp_path, run_status.STATUS_FILENAME), "w").close()
        assert run_status.read_status(str(tmp_path)) is run_status.UNREADABLE
        assert run_status.is_partial(str(tmp_path))

    def test_a_json_non_object_is_also_unreadable(self, tmp_path):
        with open(os.path.join(tmp_path, run_status.STATUS_FILENAME), "w",
                  encoding="utf-8") as handle:
            handle.write('["not", "an", "object"]')
        assert run_status.read_status(str(tmp_path)) is run_status.UNREADABLE
        assert run_status.is_partial(str(tmp_path))

    def test_a_torn_status_is_replaced_not_preserved(self, tmp_path):
        # `_mark_partial_run` declines to overwrite a stage's OWN account —
        # but a torn file is not an account of anything, so replacing it
        # with a real failure record is strictly better.
        with open(os.path.join(tmp_path, run_status.STATUS_FILENAME), "w",
                  encoding="utf-8") as handle:
            handle.write("{tru")
        bundles._mark_partial_run(str(tmp_path), verb="run", name="s",
                                  exc=RuntimeError("node evicted"))
        status = run_status.read_status(str(tmp_path))
        assert isinstance(status, dict)
        assert status["status"] == "failed"
        assert "node evicted" in status["error"]

    def test_a_partial_validate_run_with_a_torn_status_is_not_evidence(
            self, tmp_path):
        # The fail-closed rule has to reach the freeze gate too.
        from steerlab_server.experiment import experiment_store as es
        from steerlab_server.experiment.manifest import Manifest

        from test_evidence_tier import _validate_evidence

        root = str(tmp_path)
        es.create("s", model_id="org/m", revision="abc", root=root)
        _validate_evidence(root, "s")
        scope = Manifest.load("s", root=root).validation_scope_hash()
        assert es._matching_validate_evidence(scope, root) is not None
        rundir = os.path.join(root, "runs", "v-exp-s-validate")
        with open(os.path.join(rundir, run_status.STATUS_FILENAME), "w",
                  encoding="utf-8") as handle:
            handle.write("{tru")
        assert es._matching_validate_evidence(scope, root) is None


def test_legacy_run_without_status_is_not_partial(tmp_path):
    # Legacy runs carry no status file. They are unannotated, NOT
    # incomplete — their own completion artifacts still govern, and
    # treating them as partial would retroactively invalidate real results.
    plain = str(tmp_path)
    assert run_status.read_status(plain) is None
    assert not run_status.is_partial(plain)


# ------------------------------------------------- healing after completion


def test_heal_after_completion_rewrites_the_failure_era_record(tmp_path):
    """Observed live (2026-08-11, memo-study campaigns c18/c19): a pipeline failed on a
    judge 402, was resumed to disposition:completed, and its directory still
    said status:failed with a FAILED.md — the ledger and the status record
    contradicted each other, and readers of the status saw a run that 'must
    not be cited'. Healing makes the record match reality while preserving
    the history."""
    from steerlab_server.experiment import run_status as rs

    run_dir = str(tmp_path / "pipe")
    os.makedirs(run_dir)
    status = rs.RunStatus(run_dir, stage="pipeline", experiment="s",
                          item_label="artifact")
    status.fail(RuntimeError("OpenRouter judge call failed: HTTP 402"))
    assert rs.is_partial(run_dir)
    assert os.path.exists(os.path.join(run_dir, rs.FAILURE_NOTE_FILENAME))

    assert rs.heal_after_completion(run_dir) is True
    healed = json.load(open(os.path.join(run_dir, rs.STATUS_FILENAME)))
    assert healed["status"] == "completed"
    assert healed["evidenceComplete"] is True
    assert healed["error"] is None and healed["errorType"] is None
    # History preserved, not erased.
    assert "HTTP 402" in healed["supersededError"]["error"]
    assert healed["supersededError"]["errorType"] == "RuntimeError"
    assert healed["healedAt"] > 0
    assert not os.path.exists(os.path.join(run_dir, rs.FAILURE_NOTE_FILENAME))
    assert not rs.is_partial(run_dir)


def test_heal_after_completion_is_a_noop_on_a_clean_directory(tmp_path):
    from steerlab_server.experiment import run_status as rs

    run_dir = str(tmp_path / "clean")
    os.makedirs(run_dir)
    # No status file, no note: nothing to heal, nothing invented.
    assert rs.heal_after_completion(run_dir) is False
    assert not os.path.exists(os.path.join(run_dir, rs.STATUS_FILENAME))

    # An already-completed record is left byte-identical.
    status = rs.RunStatus(run_dir, stage="evaluate")
    status.complete()
    before = open(os.path.join(run_dir, rs.STATUS_FILENAME), "rb").read()
    assert rs.heal_after_completion(run_dir) is False
    assert open(os.path.join(run_dir, rs.STATUS_FILENAME), "rb").read() == before
