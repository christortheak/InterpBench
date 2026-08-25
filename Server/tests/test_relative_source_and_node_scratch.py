"""Two live-cluster defects, and the repairs for them.

**A — a relative ``--source`` broke, and the epoch guard blamed the run**
(ledger 2026-08-21, reproduced on jobs 47606365/47606373). ``study submit
<exp> --verb analyze --source runs/<dir>`` rendered the path verbatim into the
bundle command; the rendered sbatch ``cd``s into ``<submitdir>/slurm`` before
``srun``, so the child resolved ``runs/<dir>`` against the SLURM directory
instead of the ``--target`` root named on the same command line. The stamp read
then returned nothing, and the epoch guard reported "carries no
experiment-hash stamp … or pass allowUnverifiedEpoch" — for a run that was
correctly stamped, with a repair that coaches the operator into switching off a
scientific firewall to work around a path bug.

**B — node-scratch cleanup lived only in the study renderer** (ledger
2026-08-23). Anything not rendered by ``executors.render_slurm_script`` got
neither the gres request nor the EXIT trap, silently.
"""

import json
import os
import shutil
import stat
import subprocess

import pytest

from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.submissions import submit_study
from steerlab_server.experiment import (bundles, experiment_store as es,
                                        run_epoch)


def _concept(root, name="fair"):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "positive.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text":"fair"}\n')
    with open(os.path.join(directory, "negative.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"text":"unfair"}\n')


def _study(root, name="Submit Study"):
    _concept(root)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach("submit-study", ["fair"], root=root)


FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "fakebin")


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """The committed fake scheduler binaries on PATH, with a call log.

    A local copy of ``test_fake_scheduler``'s fixture (fixtures do not cross
    modules and neither lives in ``conftest``); it is deliberately the reduced
    form — this file only needs to READ what sbatch was called with.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode
                     | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    monkeypatch.setenv(
        "PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(tmp_path / "state.json"))
    for key in ("FAKE_SBATCH_FAIL", "FAKE_SLURM_JOB_ID",
                "STEERLAB_MAINTENANCE_CALENDAR", "STEERLAB_SLURM_SBATCH"):
        monkeypatch.delenv(key, raising=False)

    class Handle:
        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            if not path.exists():
                return []
            return path.read_text(encoding="utf-8").splitlines()

    return Handle()


# =============================================================================
# A1. The child resolves a relative source against the target root
# =============================================================================


def test_relative_source_resolves_against_the_target_root(tmp_path):
    target = str(tmp_path / "workspace")
    assert bundles.resolve_against_target("runs/r", target) == \
        os.path.join(target, "runs", "r")


def test_absolute_source_is_untouched(tmp_path):
    absolute = str(tmp_path / "elsewhere" / "runs" / "r")
    assert bundles.resolve_against_target(absolute, str(tmp_path)) == absolute


def test_absent_source_stays_absent(tmp_path):
    """"Not supplied" must keep meaning "not supplied" — resolving None into
    the target root would silently turn "analyze the newest run" into
    "analyze the workspace directory"."""
    assert bundles.resolve_against_target(None, str(tmp_path)) is None
    assert bundles.resolve_against_target("", str(tmp_path)) == ""


def test_relative_source_survives_a_child_whose_cwd_is_elsewhere(
        tmp_path, monkeypatch):
    """The live failure, in one test: the child runs with its cwd set to the
    slurm directory (what the rendered sbatch does), and a relative --source
    must still find the run under --target."""
    from steerlab_server.experiment import tasks

    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    elsewhere = tmp_path / "submit-dir" / "slurm"
    elsewhere.mkdir(parents=True)
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)

    seen = {}

    def fake_analyze(name, root=None, source_run=None, **kwargs):
        seen["source_run"] = source_run
        return os.path.join(root, "runs", "fake-exp-submit-study-analyze")

    monkeypatch.setattr(tasks, "analyze", fake_analyze)
    monkeypatch.chdir(elsewhere)
    bundles.execute_run_bundle(
        meta["bundlePath"], verb="analyze", target_root=target,
        source_path="runs/some-run", package_evidence_on_complete=False)

    resolved = seen["source_run"]
    assert os.path.isabs(resolved)
    assert resolved.startswith(os.path.realpath(target))
    assert str(elsewhere) not in resolved


def test_relative_prompts_already_resolve_against_the_root(tmp_path):
    """Audit verdict, pinned: ``--prompts`` was NOT vulnerable to the same
    route, and must stay that way. ``_load_prompts`` joins a relative prompts
    file to the root it is handed — which, on the bundle path, is ``--target``
    — so the child's working directory never enters it. The test exists
    because "already correct" is a claim that rots silently."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path / "root")
    _study(root)
    prompts = os.path.join(root, "prompts", "tasks", "t.jsonl")
    os.makedirs(os.path.dirname(prompts), exist_ok=True)
    with open(prompts, "w", encoding="utf-8") as handle:
        handle.write('{"id":"p1","text":"hello"}\n')
    manifest = Manifest.load("submit-study", root)

    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    cwd = os.getcwd()
    os.chdir(elsewhere)
    try:
        rows = tasks._load_prompts(manifest, "prompts/tasks/t.jsonl", root)
    finally:
        os.chdir(cwd)
    assert len(rows) == 1


def test_resume_from_is_a_run_id_not_a_path(tmp_path):
    """Audit verdict, pinned: ``--resume-from`` names a run ID and is joined
    to the target root's ``runs/`` itself, so there is no relative path for a
    working directory to capture. It REFUSES a separator outright."""
    from steerlab_server.experiment import tasks

    with pytest.raises(RuntimeError, match="invalid resume run id"):
        tasks._load_resumable_judgments(
            "s", "runs/some-run", str(tmp_path), {}, lambda *a: None)
    with pytest.raises(RuntimeError, match="invalid resume run id"):
        tasks._load_resumable_judgments(
            "s", "..", str(tmp_path), {}, lambda *a: None)


def test_the_baked_target_root_is_absolute(tmp_path, monkeypatch):
    """The ANCHOR itself: `--target` is what every relative artifact path is
    resolved against, so a relative root would move the whole problem up one
    level instead of solving it."""
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(str(root))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", "source")     # a RELATIVE root
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    submission = submit_study("submit-study", verb="verify", jobs=jobs,
                              executor="slurm", dry_run=True)
    target = submission.command[submission.command.index("--target") + 1]
    assert os.path.isabs(target)


# =============================================================================
# A2. The guard's two failure modes are two different refusals
# =============================================================================


def _stamped_run(tmp_path, stamp="aaaa"):
    directory = tmp_path / "20260821T000000000-exp-s-run"
    directory.mkdir()
    (directory / "experiment-hash.txt").write_text(stamp, encoding="utf-8")
    return str(directory)


def _unstamped_run(tmp_path):
    directory = tmp_path / "20260821T000000000-exp-s-run"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return str(directory)


def test_a_missing_source_directory_is_its_own_refusal(tmp_path):
    missing = str(tmp_path / "runs" / "not-here")
    refusal = run_epoch.unreadable_source_refusal("analyze", missing)
    assert refusal is not None
    assert "does not exist" in refusal
    # The RESOLVED absolute path, because the whole bug class is "the relative
    # path you typed landed somewhere you did not expect".
    assert os.path.abspath(missing) in refusal


def test_only_the_unstamped_refusal_mentions_the_epoch_escape(tmp_path):
    """The firewall escape belongs to exactly one of the two failures.

    A missing directory's repair is a path correction: forgiving a missing
    stamp cannot make an absent directory readable, and offering the escape
    there is what taught an operator to reach for it."""
    missing = str(tmp_path / "runs" / "not-here")
    missing_refusal, _, _ = run_epoch.epoch_refusal(
        "analyze", "s", "bbbb", missing, allow_unverified=False)
    unstamped_refusal, _, _ = run_epoch.epoch_refusal(
        "analyze", "s", "bbbb", _unstamped_run(tmp_path),
        allow_unverified=False)

    assert missing_refusal != unstamped_refusal
    assert "allowUnverifiedEpoch" not in missing_refusal
    assert "experiment-hash stamp" not in missing_refusal
    assert "allowUnverifiedEpoch" in unstamped_refusal
    assert "carries no experiment-hash stamp" in unstamped_refusal


def test_a_source_that_is_a_file_is_not_a_stamp_problem(tmp_path):
    path = tmp_path / "generations.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    refusal, _, _ = run_epoch.epoch_refusal(
        "analyze", "s", "bbbb", str(path), allow_unverified=False)
    assert "is not a directory" in refusal
    assert "allowUnverifiedEpoch" not in refusal


def test_allow_unverified_does_not_forgive_a_missing_directory(tmp_path):
    """The escape forgives a MISSING STAMP. A directory that is not there has
    no stamp to forgive, and admitting it would only move the failure to the
    first read of generations.jsonl."""
    missing = str(tmp_path / "runs" / "not-here")
    refusal, unverified, _ = run_epoch.epoch_refusal(
        "analyze", "s", "bbbb", missing, allow_unverified=True)
    assert refusal is not None
    assert unverified is False


def test_a_readable_stamped_run_still_passes(tmp_path):
    """The new probe must not become a gate of its own."""
    assert run_epoch.epoch_refusal(
        "analyze", "s", "aaaa", _stamped_run(tmp_path),
        allow_unverified=False) == (None, False, None)


def test_the_missing_source_repair_is_a_path_correction():
    repair = run_epoch.unreadable_source_repair("analyze", "s")
    assert "--source" in repair
    assert "allow-unverified-epoch" not in repair


def test_the_task_guard_raises_the_typed_missing_prerequisite(tmp_path,
                                                              monkeypatch):
    from steerlab_server.experiment import lifecycle_gates, tasks
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path / "root")
    _study(root)
    manifest = Manifest.load("submit-study", root)
    with pytest.raises(Exception) as excinfo:
        tasks._require_source_epoch(
            "analyze", "submit-study", manifest,
            str(tmp_path / "runs" / "gone"), allow_unverified_epoch=False)
    error = excinfo.value
    # Typed as a PATH problem, not as an epoch problem: an agent switching on
    # `manifestEpoch` must not treat a typo as evidence about provenance.
    assert getattr(error, "gate", None) == lifecycle_gates.MISSING_PREREQUISITE
    assert "allowUnverifiedEpoch" not in str(error)


# =============================================================================
# A3. Submit refuses a nonexistent source before it spends a queue slot
# =============================================================================


def test_submit_refuses_a_missing_source_and_names_the_resolved_path(
        tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    with pytest.raises(ValueError) as excinfo:
        submit_study("submit-study", verb="analyze", jobs=jobs,
                     executor="slurm", dry_run=True,
                     source_path="runs/never-existed")
    message = str(excinfo.value)
    assert "--source" in message
    assert os.path.join(root, "runs", "never-existed") in message
    # The refusal must leave nothing behind — it fires before a submission
    # directory or a packaged bundle exists.
    runs = os.path.join(str(tmp_path / "scratch"))
    assert not os.path.isdir(runs) or not os.listdir(runs)


def test_the_submit_refusal_is_typed_and_carries_a_path_repair(tmp_path,
                                                               monkeypatch):
    """`--json` must carry the sentence that fixes it. The ledger's operator
    got "exited 1 — see the diagnostics on stderr", which is how the useful
    words ended up reachable only as prose."""
    from steerlab_server import cli
    from steerlab_server.api.submissions import SubmissionRefusal

    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    with pytest.raises(SubmissionRefusal) as excinfo:
        submit_study("submit-study", verb="analyze", jobs=jobs,
                     executor="slurm", dry_run=True, source_path="runs/gone")
    assert excinfo.value.code == "submissionPath"
    assert "--source" in excinfo.value.repair_action
    # Still a ValueError, so the API's 400 handler is untouched.
    assert isinstance(excinfo.value, ValueError)

    # Human mode keeps exit 1 byte-for-byte; only the document changed.
    assert cli.main(["study", "submit", "submit-study", "--verb", "analyze",
                     "--source", "runs/gone", "--dry-run"]) == 1


def test_submit_accepts_a_relative_source_that_exists_under_the_target(
        tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    os.makedirs(os.path.join(root, "runs", "real-run"))
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    submission = submit_study(
        "submit-study", verb="analyze", jobs=jobs, executor="slurm",
        dry_run=True, source_path="runs/real-run")
    # The relative spelling is what gets baked in; the CHILD resolves it
    # against the same --target that is on the command line beside it.
    assert "runs/real-run" in submission.command
    assert "--target" in submission.command


# =============================================================================
# B1. --resume and --dependency reach the RENDERED script
# =============================================================================


def test_study_submit_resume_reaches_the_rendered_sbatch(tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    parked = os.path.join(root, "runs", "20260823T000000000-exp-x-run")
    os.makedirs(parked)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    submission = submit_study(
        "submit-study", verb="run", jobs=jobs, executor="slurm", dry_run=True,
        resume_directory=parked)

    assert "--resume" in submission.command
    assert parked in submission.command
    with open(submission.slurm_bundle["script_path"], encoding="utf-8") as h:
        script = h.read()
    # The POINT of routing a resume through `study submit`: the rendered
    # script is what carries the node-scratch trap. A hand-rolled sbatch does
    # not, which is the whole reason this flag exists.
    assert "--resume" in script and parked in script
    assert "trap cleanup_node_scratch EXIT" in script


def test_study_submit_refuses_a_missing_resume_directory(tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    with pytest.raises(ValueError, match="--resume"):
        submit_study("submit-study", verb="run", jobs=jobs, executor="slurm",
                     dry_run=True, resume_directory="runs/never-parked")


def test_bundle_execute_resume_outranks_the_pointer(tmp_path, monkeypatch):
    """An operator naming a parked directory is answering the question the
    pointer cannot: the pointer only remembers what THIS job record started,
    and a run parked by an EARLIER job has no pointer here at all."""
    from steerlab_server.experiment import tasks

    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    parked = os.path.join(target, "runs", "20260823T000000000-exp-p-run")
    os.makedirs(parked)

    seen = {}

    def fake_run(name, prompts, root, dtype, device, **kwargs):
        seen["run_directory"] = kwargs.get("run_directory")
        return parked

    monkeypatch.setattr(tasks, "run", fake_run)
    bundles.execute_run_bundle(
        meta["bundlePath"], verb="run", target_root=target,
        resume_directory="runs/20260823T000000000-exp-p-run",
        package_evidence_on_complete=False)
    assert seen["run_directory"] == os.path.join(
        os.path.realpath(target), "runs", "20260823T000000000-exp-p-run")


def test_bundle_execute_refuses_resume_on_a_verb_that_cannot_honour_it(
        tmp_path):
    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    with pytest.raises(bundles.BundleError, match="--resume applies"):
        bundles.execute_run_bundle(
            meta["bundlePath"], verb="verify", target_root=target,
            resume_directory=str(tmp_path), package_evidence_on_complete=False)


def test_dependency_shape_is_validated():
    from steerlab_server.api.executors import normalized_dependency

    assert normalized_dependency("afterok:12345") == "afterok:12345"
    assert normalized_dependency("afterany:1,afterok:2") == "afterany:1,afterok:2"
    assert normalized_dependency("afterok:12345+10") == "afterok:12345+10"
    assert normalized_dependency("singleton") == "singleton"
    assert normalized_dependency(" afterok:1?afterok:2 ") == "afterok:1?afterok:2"
    for bad in ("", "12345", "afterok", "afterok:", "aftersomething:1",
                "afterok:abc", "afterok:1 afterok:2", "afterok:1,,afterok:2"):
        with pytest.raises(ValueError):
            normalized_dependency(bad)


def test_dependency_reaches_sbatch_and_not_the_script(tmp_path, monkeypatch,
                                                      fake_slurm):
    """A COMMAND-LINE argument, deliberately not an ``#SBATCH`` header:
    auto-resubmit-on-checkpoint re-submits the same file verbatim, and a
    continuation must not re-wait on a dependency already satisfied."""
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    submission = submit_study(
        "submit-study", verb="verify", jobs=jobs, executor="slurm",
        dependency="afterok:47606443")

    assert any("--dependency=afterok:47606443" in call
               for call in fake_slurm.calls("sbatch"))
    with open(submission.slurm_bundle["script_path"], encoding="utf-8") as h:
        assert "--dependency" not in h.read()
    job = jobs.get(submission.job_id)
    assert job.result["dependency"] == "afterok:47606443"


def test_a_malformed_dependency_refuses_before_anything_is_submitted(
        tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    with pytest.raises(ValueError, match="dependency"):
        submit_study("submit-study", verb="verify", jobs=jobs,
                     executor="slurm", dry_run=True,
                     dependency="afterok:not-a-job")


# =============================================================================
# B2/B3. One definition of cleanup; the path comes from the site profile
# =============================================================================


def _cleanup_probe(stage_dir_value, environ=None, **extra_env):
    """Run the rendered cleanup function in a real bash with ``rm`` traced."""
    from steerlab_server.node_scratch import cleanup_lines

    body = "\n".join(cleanup_lines(environ=environ or {}))
    if "trap cleanup_node_scratch EXIT" not in body:
        return None            # no trap rendered at all
    body = body.split("trap cleanup_node_scratch EXIT")[0]
    probe = ("set -euo pipefail\nrm() { echo \"RM $*\"; }\n" + body
             + "\ncleanup_node_scratch\n")
    env = {"PATH": os.environ.get("PATH", ""), "SLURM_JOB_ID": "12345",
           "USER": "someuser", **extra_env}
    if stage_dir_value is not None:
        env["STEERLAB_NODE_STAGE_DIR"] = stage_dir_value
    result = subprocess.run(["bash", "-c", probe], capture_output=True,
                            text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_cleanup_removes_the_directory_the_site_template_names():
    """Not a hardcoded ``/lscratch/$SLURM_JOB_ID``: that is one site's
    spelling. A site whose template is ``$SLURM_TMPDIR`` or
    ``/local_scratch/$USER/$SLURM_JOB_ID`` used to get NO cleanup while the
    render looked perfectly correct."""
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID") == \
        "RM -rf -- /lscratch/12345"
    assert _cleanup_probe("/lscratch/${SLURM_JOB_ID}") == \
        "RM -rf -- /lscratch/12345"
    assert _cleanup_probe("$SLURM_TMPDIR", SLURM_TMPDIR="/tmp/job.12345") == \
        "RM -rf -- /tmp/job.12345"
    assert _cleanup_probe("/local_scratch/$USER/$SLURM_JOB_ID") == \
        "RM -rf -- /local_scratch/someuser/12345"


def test_cleanup_still_refuses_anything_not_job_scoped():
    """The guard IS the safety argument: a shared node cache is not this
    job's to delete."""
    assert _cleanup_probe("/lscratch/steerlab-shared-cache") == ""
    assert _cleanup_probe("/lscratch/cache") == ""
    # A template naming a variable this module does not authorise is not
    # provably job-scoped, so nobody cleans it up — never everybody.
    assert _cleanup_probe("/lscratch/$SOME_SITE_VARIABLE") == ""
    assert _cleanup_probe(None) == ""


def test_cleanup_removes_nothing_when_expansion_collapses_the_path():
    """A declared-but-unset job variable must not turn into ``rm -rf ''`` or,
    worse, ``rm -rf /``."""
    assert _cleanup_probe("$SLURM_TMPDIR") == ""
    # The one that would have hurt: `/models` is absolute and fully expanded,
    # so every downstream sanity check passes — and it is somebody else's
    # directory. A variable the template names and the node has not set means
    # the path is UNKNOWN, and an unknown path is never removed.
    assert _cleanup_probe("$SLURM_TMPDIR/models") == ""
    assert _cleanup_probe("/local_scratch/$USER/$SLURM_JOB_ID", USER="") == ""


# --------------------------------------------------------------------------
# Traversal: job-scoped is not the same as CONTAINED (external review
# 2026-08-24)
# --------------------------------------------------------------------------


def test_a_traversing_template_is_refused_even_though_it_is_job_scoped():
    """The reviewer's reproduction, exactly.

    ``/lscratch/$SLURM_JOB_ID/../../shared`` passes every guard that existed:
    it names ``$SLURM_JOB_ID``, so it is job-scoped by construction; it
    expands to an absolute path; nothing is left unexpanded. And it resolves
    to ``/shared`` — somebody else's directory, handed to ``rm -rf``.
    """
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID/../../shared") == ""
    assert _cleanup_probe("/lscratch/${SLURM_JOB_ID}/../../shared") == ""
    # The variable-leading spelling of the same escape: `$SLURM_TMPDIR/..` is
    # the job directory's PARENT, which Slurm did not allocate to this job.
    assert _cleanup_probe("$SLURM_TMPDIR/..",
                          SLURM_TMPDIR="/tmp/job.12345") == ""
    assert _cleanup_probe("$SLURM_TMPDIR/../sibling",
                          SLURM_TMPDIR="/tmp/job.12345") == ""
    # A `.` component is refused on the same rule rather than normalized: a
    # staging template never needs one, so it is a signal, not a spelling.
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID/./stage") == ""
    assert _cleanup_probe("./lscratch/$SLURM_JOB_ID") == ""


def test_a_variable_whose_value_traverses_is_refused_too():
    """The template can be spotless while the VALUE carries the escape — the
    check therefore runs on the expanded result as well as on the template."""
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID",
                          SLURM_JOB_ID="12345/../../etc") == ""
    assert _cleanup_probe("/local_scratch/$USER/$SLURM_JOB_ID",
                          USER="../../root") == ""


def test_the_removal_must_stay_beneath_the_template_s_own_anchor():
    """Containment, stated positively: a path is removed only when it IS the
    template's anchor directory or sits under it. The anchor is the literal
    text before the first variable — or, for a template that begins with one,
    that variable's own expansion, which is what keeps a bare
    ``$SLURM_TMPDIR`` removable while ``$SLURM_TMPDIR/..`` is not."""
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID/stage") == \
        "RM -rf -- /lscratch/12345/stage"
    assert _cleanup_probe("$SLURM_TMPDIR/work",
                          SLURM_TMPDIR="/tmp/job.12345") == \
        "RM -rf -- /tmp/job.12345/work"
    # An anchor of `/` would contain everything, which is not a containment
    # check at all.
    assert _cleanup_probe("/$SLURM_JOB_ID") == ""


def test_every_legitimate_template_still_cleans_up_byte_for_byte():
    """The containment rules must cost the sites that were already correct
    NOTHING: the same templates remove the same directories, character for
    character, as before this landed."""
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID") == \
        "RM -rf -- /lscratch/12345"
    assert _cleanup_probe("/lscratch/${SLURM_JOB_ID}") == \
        "RM -rf -- /lscratch/12345"
    assert _cleanup_probe("$SLURM_TMPDIR", SLURM_TMPDIR="/tmp/job.12345") == \
        "RM -rf -- /tmp/job.12345"
    assert _cleanup_probe("${SLURM_TMPDIR}", SLURM_TMPDIR="/tmp/job.12345") \
        == "RM -rf -- /tmp/job.12345"
    assert _cleanup_probe("/local_scratch/$USER/$SLURM_JOB_ID") == \
        "RM -rf -- /local_scratch/someuser/12345"
    assert _cleanup_probe("/lscratch/$SLURM_JOBID", SLURM_JOBID="99") == \
        "RM -rf -- /lscratch/99"


def test_a_site_whose_scheduler_purges_scratch_gets_no_trap():
    from steerlab_server.node_scratch import (SCHEDULER_PURGES_ENV,
                                              cleanup_lines)

    lines = cleanup_lines(environ={SCHEDULER_PURGES_ENV: "1"})
    text = "\n".join(lines)
    assert "trap cleanup_node_scratch EXIT" not in text
    assert "rm -rf" not in text
    assert "SCHEDULER" in text          # it says WHY, not just nothing


def test_the_purge_declaration_is_absent_by_default():
    """Absent field = today's behaviour, byte-for-byte."""
    from steerlab_server.node_scratch import cleanup_lines

    assert "trap cleanup_node_scratch EXIT" in "\n".join(
        cleanup_lines(environ={}))


def test_the_profile_field_renders_the_env_key_only_when_declared():
    from steerlab_server.api import site_environment
    from steerlab_server.api.site_profile import SiteStorage
    from steerlab_server.node_scratch import SCHEDULER_PURGES_ENV

    assert SiteStorage.decode({}).node_scratch_purged_by_scheduler is False
    declared = SiteStorage.decode({"nodeScratchPurgedByScheduler": True})
    assert declared.node_scratch_purged_by_scheduler is True
    assert hasattr(site_environment, "node_scratch")


def test_the_study_renderer_and_the_wrapper_share_one_definition(
        tmp_path, monkeypatch):
    """The whole point of extracting the block: "how this site cleans up" is
    ONE definition, not a rendered copy and an ad-hoc copy that drift."""
    from steerlab_server.api.executors import (SlurmExecutor, SlurmResources,
                                               render_slurm_script)
    from steerlab_server import node_scratch

    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    monkeypatch.setenv("STEERLAB_SLURM_SCRATCH_GRES", "lscratch:100")
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "bundle"), ["true"],
        resources=SlurmResources(walltime="01:00:00", use_srun=False))
    study_script = render_slurm_script(bundle)
    wrapper = node_scratch.render_wrapper()
    block = "\n".join(node_scratch.cleanup_lines())

    assert block in study_script
    assert block in wrapper


def test_the_wrapper_carries_the_gres_and_the_trap(tmp_path, monkeypatch):
    from steerlab_server import node_scratch

    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    monkeypatch.setenv("STEERLAB_SLURM_SCRATCH_GRES", "lscratch:100")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100,H100")
    monkeypatch.setenv("STEERLAB_SLURM_GRES", "A100")
    wrapper = node_scratch.render_wrapper()

    assert "#SBATCH --gres=gpu:A100:1,lscratch:100" in wrapper
    assert "trap cleanup_node_scratch EXIT" in wrapper
    # The stage template rides VERBATIM — single-quoted, so $SLURM_JOB_ID
    # expands on the node and never here.
    assert "export STEERLAB_NODE_STAGE_DIR='/node-scratch/$SLURM_JOB_ID'" \
        in wrapper
    # The payload is the only variable.
    assert wrapper.rstrip().endswith('"$@"')


def test_the_wrapper_is_valid_bash(tmp_path, monkeypatch):
    from steerlab_server import node_scratch

    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    path = tmp_path / "wrapper.sbatch"
    path.write_text(node_scratch.render_wrapper(), encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(path)],
                          capture_output=True).returncode == 0


def test_the_wrapper_renders_without_a_declared_gpu_vocabulary(monkeypatch):
    """Declare-or-refuse is right for a STUDY job and wrong for this
    artifact: a wrapper that cannot render because the site's GPU list is
    missing leaves the operator with no canonical way to request node scratch
    at all — the exact hole being closed."""
    from steerlab_server import node_scratch

    monkeypatch.setenv("STEERLAB_SLURM_GRES", "A100")
    monkeypatch.delenv("STEERLAB_SLURM_GPU_TYPES", raising=False)
    monkeypatch.setenv("STEERLAB_SLURM_SCRATCH_GRES", "lscratch:100")
    wrapper = node_scratch.render_wrapper()
    assert "#SBATCH --gres=lscratch:100" in wrapper


def test_writing_the_wrapper_is_idempotent(tmp_path, monkeypatch):
    from steerlab_server import node_scratch

    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    root = str(tmp_path / "meta")
    first = node_scratch.write_wrapper(metadata_root=root)
    assert first["written"] is True
    assert os.path.isfile(first["path"])
    assert first["path"].endswith(node_scratch.WRAPPER_NAME)
    second = node_scratch.write_wrapper(metadata_root=root)
    assert second["written"] is False


def test_the_wrapper_lands_beside_the_controller_job_script(tmp_path):
    from steerlab_server import controller_render, node_scratch

    root = str(tmp_path / "meta")
    assert os.path.dirname(node_scratch.wrapper_path(root)) == \
        os.path.dirname(controller_render.rendered_path(root))


def test_the_cli_renders_the_wrapper(tmp_path, monkeypatch, capsys):
    from steerlab_server import cli

    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    root = str(tmp_path / "meta")
    assert cli.main(["site", "node-scratch-wrapper",
                     "--metadata-root", root]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] is True
    assert payload["nodeStageDirTemplate"] == "/node-scratch/$SLURM_JOB_ID"
    with open(payload["path"], encoding="utf-8") as handle:
        assert "trap cleanup_node_scratch EXIT" in handle.read()
