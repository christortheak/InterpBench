"""`steerlab-server study submit --parallel N` (2026-08-07).

Before this, automatic multi-GPU fan-out was API-only (`parallelJobs` on
`POST /api/studies/submit-bundle`) and the CLI's `bundle execute --shard k/K`
produced partials that NOTHING merged. The flag routes a server-resident
`study submit` through the SAME `_submit_sharded_bundle` machinery the API
uses, so the parent/shard records it writes are the ones
`JobManager._reconcile_shard_parents` already knows how to merge.

What is pinned here:

1. CLI parsing — the flag reaches `submit_study`, its alias works, a
   non-integer/zero exits 64 rather than silently running one job.
2. Fan-out shape — one parent + K shard children carrying exactly the keys
   the reconciler reads (`shardChildren`, `shardMerge`, `parallelJobs`), and
   the merge path recognises the parent.
3. Refusal — a verb (or executor) that does not shard REFUSES here, unlike
   the app-facing bundle path which degrades with a note.
4. The cap.
"""

import json
import os

import pytest

from steerlab_server import cli
from steerlab_server.api import submissions

# The fake sbatch/squeue/sacct rig and the study fixture already used by the
# fan-out tests: this flag must fan out the same study the same way.
from test_sharding import _manager, _study_fixture
from test_sharding import fake_slurm  # noqa: F401  (fixture re-export)


# --- 1. CLI parsing ------------------------------------------------------------

class _Recorder:
    """Stands in for `submit_study`, capturing the kwargs the CLI passes."""

    def __init__(self, shard_job_ids=None):
        self.calls: list[dict] = []
        self._shard_job_ids = shard_job_ids

    def __call__(self, experiment, **kwargs):
        self.calls.append({"experiment": experiment, **kwargs})
        return submissions.StudySubmission(
            "parent1", experiment, kwargs.get("verb", "run"), "slurm", True,
            {}, None, None, [], "records", "submission",
            shard_job_ids=self._shard_job_ids)


class _StubJobs:
    """Stands in for the CLI's JobManager: the CLI reads the parent record
    back to detect an aborted fan-out; parsing tests have no store, so the
    lookup finds nothing."""

    def get(self, job_id):
        return None


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(submissions, "submit_study", rec)
    # The CLI constructs a JobManager eagerly; keep it off the real store.
    monkeypatch.setattr("steerlab_server.api.jobs.JobManager",
                        lambda *a, **k: _StubJobs())
    return rec


def test_parallel_defaults_to_one(recorder, capsys):
    assert cli.main(["study", "submit", "s", "--verb", "run"]) == 0
    assert recorder.calls[0]["parallel_jobs"] == 1


def test_parallel_flag_reaches_submit_study(recorder):
    assert cli.main(["study", "submit", "s", "--verb", "run",
                     "--parallel", "6"]) == 0
    assert recorder.calls[0]["parallel_jobs"] == 6


def test_parallel_jobs_alias_is_accepted(recorder):
    """The API spells it `parallelJobs` and §7.2 of the CLI reference
    recommended `--parallel-jobs`; a flag the CLI silently swallowed would
    run one job while the researcher believed they had fanned out."""
    assert cli.main(["study", "submit", "s", "--verb", "run",
                     "--parallel-jobs", "4"]) == 0
    assert recorder.calls[0]["parallel_jobs"] == 4


@pytest.mark.parametrize("value", ["abc", "2.5", ""])
def test_non_integer_parallel_exits_64(recorder, value, capsys):
    assert cli.main(["study", "submit", "s", "--verb", "run",
                     "--parallel", value]) == 64
    assert not recorder.calls
    assert "--parallel expects an integer" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3"])
def test_non_positive_parallel_exits_64(recorder, value, capsys):
    assert cli.main(["study", "submit", "s", "--verb", "run",
                     "--parallel", value]) == 64
    assert not recorder.calls
    assert "at least 1" in capsys.readouterr().err


def test_trailing_parallel_with_no_value_exits_64(recorder, capsys):
    """§7.5's positional fragility: `_flag` yields None at end of line, and a
    silent single job is the worst possible reading of `--parallel`."""
    assert cli.main(["study", "submit", "s", "--verb", "run",
                     "--parallel"]) == 64
    assert not recorder.calls
    assert "requires a number" in capsys.readouterr().err


def test_usage_line_documents_parallel_and_gres(capsys):
    assert cli.main(["study"]) == 64
    err = capsys.readouterr().err
    assert "--parallel N" in err
    # The GPU-type answer belongs where the researcher looks for it.
    assert "--gres" in err and "STEERLAB_SLURM_GRES" in err


def test_fanned_out_submission_names_the_merger(monkeypatch, capsys):
    """The merge runs only in a LIVE server's reconciler; a CLI fan-out with
    no server running would leave the partials sitting forever."""
    rec = _Recorder(shard_job_ids=["a", "b", "c"])
    monkeypatch.setattr(submissions, "submit_study", rec)
    monkeypatch.setattr("steerlab_server.api.jobs.JobManager",
                        lambda *a, **k: _StubJobs())
    assert cli.main(["study", "submit", "s", "--verb", "run",
                     "--parallel", "3"]) == 0
    err = capsys.readouterr().err
    assert "3 shard jobs" in err and "serve" in err


# --- 2. fan-out shape the reconciler recognises --------------------------------

def _submit(tmp_path, monkeypatch, *, name="parsub", verb="run", parallel=3,
            jobs=None, **kwargs):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    target = str(tmp_path / "target")
    os.makedirs(target, exist_ok=True)
    source = str(tmp_path / "source")
    _study_fixture(source, name)
    jobs = jobs or _manager(tmp_path)
    return jobs, submissions.submit_study(
        name, verb=verb, jobs=jobs, executor="slurm", root=source,
        target_root=target,
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=parallel, **kwargs)


def test_study_submit_creates_parent_and_shard_children(tmp_path, monkeypatch,
                                                        fake_slurm):
    jobs, submission = _submit(tmp_path, monkeypatch, parallel=3)

    assert submission.shard_job_ids and len(submission.shard_job_ids) == 3
    assert submission.to_dict()["shardJobIDs"] == submission.shard_job_ids

    parent = jobs.get(submission.job_id)
    rr = parent.requested_resources
    # Exactly the keys `_reconcile_shard_parents` / `_merge_shard_parent` read.
    assert rr["parallelJobs"] == 3
    assert rr["shardChildren"] == submission.shard_job_ids
    assert rr["shardMerge"]["experiment"] == "parsub"
    assert rr["shardMerge"]["verb"] == "run"
    # The merge re-reads the bundle this submission packaged — it must be a
    # durable file, not a temp the CLI process owned.
    assert os.path.isfile(rr["shardMerge"]["bundlePath"])
    assert parent.executor_job_id is None
    assert (parent.result or {})["shardJobs"] == submission.shard_job_ids

    assert len(fake_slurm.calls("sbatch")) == 3
    for index, child_id in enumerate(submission.shard_job_ids):
        child = jobs.get(child_id)
        assert child.status == "submitted"
        assert child.requested_resources["shardIndex"] == index
        assert child.requested_resources["shardCount"] == 3
        assert child.requested_resources["parentJob"] == parent.id
        script = open(child.requested_resources["scriptPath"],
                      encoding="utf-8").read()
        assert f"--shard {index}/3" in script
        assert "--no-evidence" in script   # the merge packages evidence, once


def test_the_merge_path_picks_up_a_cli_fanned_out_parent(tmp_path, monkeypatch,
                                                         fake_slurm):
    """The point of routing through `_submit_sharded_bundle`: the reconciler
    that merges app-submitted batches must claim these records too. Drive it
    with children that have not finished, so the tick classifies the parent
    without attempting a real merge."""
    jobs, submission = _submit(tmp_path, monkeypatch, name="parrec", parallel=2)
    for child_id in submission.shard_job_ids:
        child = jobs.get(child_id)
        child.status = "running"
        jobs.store.update(child)

    assert jobs._reconcile_shard_parents() >= 1
    parent = jobs.get(submission.job_id)
    assert parent.status == "running"


def test_dry_run_prepares_every_shard_without_sbatch(tmp_path, monkeypatch,
                                                     fake_slurm):
    jobs, submission = _submit(tmp_path, monkeypatch, name="pardry",
                               parallel=4, dry_run=True)
    assert len(submission.shard_job_ids) == 4
    assert fake_slurm.calls("sbatch") == []
    for child_id in submission.shard_job_ids:
        assert jobs.get(child_id).status == "prepared"


def test_failed_fanout_exits_1_with_the_schedulers_words(tmp_path, monkeypatch,
                                                         fake_slurm, capsys):
    """Live incident 2026-08-09: five QOSMaxSubmitJobPerUserLimit-refused
    submissions exited 0, because `_submit_sharded_bundle` reports the abort
    through the parent job record (the app's contract) and the CLI printed
    the returned no-shard submission as success. The CLI must read the
    record back: exit 1, name the parent job id, and put sbatch's own
    stderr on the terminal — never a JSON blob of nulls and silence."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    source = str(tmp_path / "source")
    _study_fixture(source, "parqos")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    target = str(tmp_path / "target")
    os.makedirs(target, exist_ok=True)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "1")   # shard 2 of 3 fails

    code = cli.main(["study", "submit", "parqos", "--verb", "run",
                     "--parallel", "3", "--target", target,
                     "--gres", "A100", "--walltime", "00:30:00"])
    captured = capsys.readouterr()
    assert code == 1

    # stdout still carries the submission JSON (the parent id is there for
    # scripts), and it shows the truthful no-shards shape.
    payload = json.loads(captured.out)
    assert payload["shardJobIDs"] is None
    parent_id = payload["jobId"]

    # No cryptic blockers: the parent job id and the scheduler's actual
    # stderr are on the terminal, not buried in a job record.
    assert "study submit failed" in captured.err
    assert parent_id in captured.err
    assert "shard 2 of 3 failed to submit" in captured.err
    assert "fake quota exceeded" in captured.err
    # The success-path "fanned out across N shard jobs" note must not print.
    assert "fanned out across" not in captured.err

    # The record the CLI read is the durable one the app renders.
    from steerlab_server.api.jobs import JobManager
    parent = JobManager().get(parent_id)
    assert parent is not None and parent.status == "failed"


# --- 3. refusals ---------------------------------------------------------------

@pytest.mark.parametrize("verb", ["validate", "sweep", "evaluate", "analyze",
                                  "extract", "verify"])
def test_non_sharding_verbs_refuse(tmp_path, monkeypatch, fake_slurm, verb):
    """Divergence from the app-facing bundle path, on purpose: a researcher
    typing `--parallel 8` on a verb that cannot shard must be told, not
    handed one job and a note buried in a job record."""
    with pytest.raises(ValueError) as excinfo:
        _submit(tmp_path, monkeypatch, name=f"parref{verb}", verb=verb,
                parallel=8)
    message = str(excinfo.value)
    assert "parallelJobs=8 refused" in message
    assert "does not shard" in message
    assert "run-first pipeline" in message


def test_local_executor_refuses_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    source = str(tmp_path / "source")
    _study_fixture(source, "parloc")
    with pytest.raises(ValueError, match="only Slurm submissions shard"):
        submissions.submit_study("parloc", verb="run", jobs=_manager(tmp_path),
                                 executor="local", root=source,
                                 parallel_jobs=2)


def test_refusal_leaves_no_submission_directory(tmp_path, monkeypatch,
                                                fake_slurm):
    """The fan-out resolves before anything is packaged, so a refused submit
    does not litter runs/ with an orphan submission directory and bundle."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    source = str(tmp_path / "source")
    _study_fixture(source, "parclean")
    runs_before = sorted(os.listdir(os.path.join(source, "runs"))) \
        if os.path.isdir(os.path.join(source, "runs")) else []
    with pytest.raises(ValueError):
        submissions.submit_study("parclean", verb="validate",
                                 jobs=_manager(tmp_path), executor="slurm",
                                 root=source, parallel_jobs=2)
    runs_after = sorted(os.listdir(os.path.join(source, "runs"))) \
        if os.path.isdir(os.path.join(source, "runs")) else []
    assert runs_after == runs_before


def test_parallel_one_is_the_historical_single_job_path(tmp_path, monkeypatch,
                                                        fake_slurm):
    jobs, submission = _submit(tmp_path, monkeypatch, name="parone", parallel=1)
    assert submission.shard_job_ids is None
    assert submission.slurm_job_id is not None
    parent = jobs.get(submission.job_id)
    assert "shardChildren" not in (parent.requested_resources or {})


def test_parallel_and_resume_from_are_mutually_exclusive(tmp_path, monkeypatch,
                                                         fake_slurm):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _submit(tmp_path, monkeypatch, name="parres", parallel=2,
                resume_from="somejob")


# --- 4. the cap ----------------------------------------------------------------

def test_cap_refuses_loudly(tmp_path, monkeypatch, fake_slurm):
    from steerlab_server.experiment.sharding import MAX_PARALLEL_JOBS

    assert MAX_PARALLEL_JOBS == 64
    with pytest.raises(ValueError, match="fan-out cap"):
        _submit(tmp_path, monkeypatch, name="parcap",
                parallel=MAX_PARALLEL_JOBS + 1)


def test_cap_boundary_is_accepted_by_the_resolver():
    from steerlab_server.experiment.sharding import MAX_PARALLEL_JOBS

    parallel, note, reason = submissions._resolve_parallel_jobs(
        MAX_PARALLEL_JOBS, verb="run", executor="slurm", pipeline_stages=None)
    assert parallel == MAX_PARALLEL_JOBS and note is None and reason is None


def test_cli_surfaces_the_cap_refusal_as_exit_1(tmp_path, monkeypatch, capsys,
                                                fake_slurm):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    source = str(tmp_path / "source")
    _study_fixture(source, "parcli")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    assert cli.main(["study", "submit", "parcli", "--verb", "run",
                     "--parallel", "200"]) == 1
    assert "fan-out cap" in capsys.readouterr().err


# --- 5. the resolver's three-tuple contract ------------------------------------

def test_resolver_reports_a_prefix_free_reason():
    """`submit_study` refuses in its own words, so the resolver hands back the
    reason without the `parallelJobs=N ignored:` framing the bundle path logs."""
    parallel, note, reason = submissions._resolve_parallel_jobs(
        4, verb="validate", executor="slurm", pipeline_stages=None)
    assert parallel == 1
    assert note == f"parallelJobs=4 ignored: {reason}"
    assert not reason.startswith("parallelJobs")


def test_run_first_pipeline_shards_and_others_do_not():
    shards, note, _ = submissions._resolve_parallel_jobs(
        3, verb="pipeline", executor="slurm",
        pipeline_stages=["run", "evaluate", "analyze"])
    assert shards == 3 and note is None

    single, note, reason = submissions._resolve_parallel_jobs(
        3, verb="pipeline", executor="slurm",
        pipeline_stages=["extract", "run"])
    assert single == 1
    assert "first stage is 'run'" in reason
