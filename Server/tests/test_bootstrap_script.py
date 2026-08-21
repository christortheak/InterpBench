"""WS5 provisioning scripts: bootstrap.sh --dry-run emits a parseable JSON
report; the controller-job template refuses unsafe starts and documents the
bind decision + resubmit chain."""

import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
BOOTSTRAP = os.path.join(SCRIPTS, "bootstrap.sh")
BOOTSTRAP_JOB = os.path.join(SCRIPTS, "submit-bootstrap-job.sh")
CONTROLLER = os.path.join(SCRIPTS, "controller-job.sbatch.template")

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def test_bootstrap_dry_run_emits_parseable_json_report(tmp_path):
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run",
         "--prefix", str(tmp_path / "env"),
         "--repo", str(tmp_path / "repo"),
         "--workspace", str(tmp_path / "ws"),
         "--hf-cache", str(tmp_path / "hf"),
         "--env-file", str(tmp_path / "cluster.env"),
         "--account", "lab1", "--hello"],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["ok"] is True
    assert report["envFile"] == str(tmp_path / "cluster.env")
    assert report["prefix"] == str(tmp_path / "env")
    steps = report["steps"]
    assert set(steps) == {"condaDetect", "envCreate", "manifestCheck",
                          "torchInstall", "serverInstall", "jlensStage",
                          "envFile", "profileValidate", "helloJob"}
    # Dry run executes nothing: every requested step is only planned.
    assert steps["condaDetect"] == "planned"
    assert steps["helloJob"] == "planned"      # --hello was requested
    # No deployment-manifest.json in the tmp repo -> the packaged-payload
    # verification is honestly skipped (the dev-checkout push path).
    assert steps["manifestCheck"] == "skipped"
    assert not (tmp_path / "cluster.env").exists()


def test_bootstrap_dry_run_marks_skips(tmp_path):
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run", "--no-torch",
         "--env-file", str(tmp_path / "cluster.env")],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["torchInstall"] == "skipped"
    assert report["steps"]["helloJob"] == "skipped"   # no --hello
    # J-lens staging is opt-in: a routine bootstrap must not install the
    # git-pinned reference package (it floors transformers>=5.5) or pull ~4 GB
    # of lens tensors as a side effect.
    assert report["steps"]["jlensStage"] == "skipped"


def test_bootstrap_dry_run_plans_jlens_stage_when_requested(tmp_path):
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run", "--with-jlens",
         "--hf-cache", str(tmp_path / "hf"),
         "--env-file", str(tmp_path / "cluster.env")],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["jlensStage"] == "planned"
    # The plan names the acquisition verb and each model it will fetch. The
    # step DELEGATES rather than carrying its own downloader — pattern scoping,
    # landed-file verification, progress, cancel, and failure triage all live
    # in the verb, and a second copy here is how the two drift apart.
    assert "jlens acquire" in proc.stdout
    assert "acquire: google/gemma-3-27b-it" in proc.stdout
    assert "acquire: google/gemma-3-4b-it" in proc.stdout
    assert str(tmp_path / "hf") in proc.stdout
    assert not (tmp_path / "hf").exists()   # dry run downloads nothing


def test_bootstrap_jlens_step_delegates_instead_of_downloading_itself(tmp_path):
    """One implementation, two entry points (plan §11.0.1).

    Guards the specific regression of reintroducing a bootstrap-resident
    snapshot_download: it would duplicate the verb's scoping and verification,
    and the copy is what silently rots.
    """
    script = open(BOOTSTRAP, encoding="utf-8").read()
    assert "jlens acquire" in script
    assert "snapshot_download" not in script


def test_bootstrap_dry_run_plans_manifest_check_for_packaged_payloads(tmp_path):
    # A pushed PACKAGED payload carries deployment-manifest.json at the repo
    # root; the dry-run plan then includes the sha256 verification step.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "deployment-manifest.json").write_text("{}")
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run", "--repo", str(repo),
         "--env-file", str(tmp_path / "cluster.env")],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["manifestCheck"] == "planned"
    assert "verify" in proc.stdout


def test_bootstrap_rejects_unknown_flag():
    proc = subprocess.run([bash, BOOTSTRAP, "--frobnicate"],
                          text=True, capture_output=True, check=False)
    assert proc.returncode == 64
    assert "unknown flag" in proc.stderr


def test_bootstrap_syntax_ok():
    assert subprocess.run([bash, "-n", BOOTSTRAP]).returncode == 0
    assert subprocess.run([bash, "-n", BOOTSTRAP_JOB]).returncode == 0
    assert subprocess.run([bash, "-n", CONTROLLER]).returncode == 0


def test_bootstrap_job_dry_run_is_scheduler_free(tmp_path):
    """The app can review the exact plan from a login host without sbatch;
    only the confirmed real run crosses into a CPU allocation."""
    proc = subprocess.run(
        [bash, BOOTSTRAP_JOB,
         "--bootstrap-script", BOOTSTRAP,
         "--job-workspace", str(tmp_path / "ws"),
         "--job-partition", "batch",
         "--job-cpus", "4", "--job-memory", "16G",
         "--job-walltime", "02:00:00", "--dry-run", "--",
         "--repo", str(tmp_path / "repo"),
         "--workspace", str(tmp_path / "ws"),
         "--hf-cache", str(tmp_path / "hf")],
        text=True, capture_output=True, check=False,
        env={**os.environ, "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["envCreate"] == "planned"
    assert not (tmp_path / "ws").exists()


def test_bootstrap_job_runs_inside_cpu_allocation_and_streams_log(tmp_path):
    """--wait is the historical foreground follow, now opt-in."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_bootstrap = tmp_path / "bootstrap.sh"
    fake_bootstrap.write_text(
        "#!/bin/bash\n"
        "echo bootstrap-on-${SLURM_JOB_ID:-none}\n"
        "echo '{\"ok\":true,\"steps\":{\"condaDetect\":\"ok\"},"
        "\"envFile\":\"/home/me/e\",\"prefix\":\"/home/me/p\"}'\n",
        encoding="utf-8")
    fake_bootstrap.chmod(fake_bootstrap.stat().st_mode | stat.S_IXUSR)

    # The fake scheduler executes synchronously but honors the generated log
    # directive. That is enough to exercise script rendering, status handoff,
    # log forwarding, and final exit propagation without a Slurm install.
    fake_sbatch = bindir / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "job=\"${@: -1}\"\n"
        "log=$(sed -n 's/^#SBATCH --output=//p' \"$job\")\n"
        "SLURM_JOB_ID=4242 bash \"$job\" > \"$log\" 2>&1\n"
        "echo 4242\n",
        encoding="utf-8")
    fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)
    fake_squeue = bindir / "squeue"
    fake_squeue.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_squeue.chmod(fake_squeue.stat().st_mode | stat.S_IXUSR)

    workspace = tmp_path / "workspace"
    env = {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin",
           "HOME": str(tmp_path), "USER": "tester",
           "STEERLAB_BOOTSTRAP_POLL_SECONDS": "0"}
    proc = subprocess.run(
        [bash, BOOTSTRAP_JOB, "--wait",
         "--bootstrap-script", str(fake_bootstrap),
         "--job-workspace", str(workspace),
         "--job-partition", "batch", "--job-cpus", "4",
         "--job-memory", "16G", "--job-walltime", "02:00:00", "--"],
        text=True, capture_output=True, check=False, env=env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "STEERLAB_BOOTSTRAP_JOB_ID=4242" in proc.stdout
    assert "bootstrap-on-4242" in proc.stdout
    assert "completed successfully" in proc.stdout

    scripts = list(workspace.glob("steerlab-bootstrap.*.sbatch"))
    assert len(scripts) == 1
    rendered = scripts[0].read_text(encoding="utf-8")
    assert "#SBATCH --partition=batch" in rendered
    assert "#SBATCH --cpus-per-task=4" in rendered
    assert "#SBATCH --mem=16G" in rendered
    assert "#SBATCH --gres" not in rendered
    assert "#SBATCH --export=NONE" in rendered


def test_bootstrap_job_takes_the_setup_class_from_the_rendered_env_and_flags(tmp_path):
    """WP5 Step 9 (audit c19): the setup job's shape is SITE data.

    Two carriers, in this precedence: the app resolves `scheduler.setupJob` and
    passes flags (inside the reviewed plan hash), and a hand run on the login
    host picks the same facts out of the rendered env file it sourced. The
    script's own literals are only the floor under both. The class's own
    `--job-extra-sbatch` directives are emitted; the site-WIDE placement
    directives deliberately are not (audit §6.x item 1: a CPU-only job pinned
    to GPU node features queues forever)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue"):
        stub = bindir / name
        stub.write_text("#!/bin/bash\necho 4242\n", encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    fake_bootstrap = tmp_path / "bootstrap.sh"
    fake_bootstrap.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    def submit(workspace, *extra, **env_extra):
        env = {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin",
               "HOME": str(tmp_path), "USER": "tester", **env_extra}
        proc = subprocess.run(
            [bash, BOOTSTRAP_JOB, "--bootstrap-script", str(fake_bootstrap),
             "--job-workspace", str(workspace), "--job-partition", "batch",
             *extra, "--"],
            text=True, capture_output=True, check=False, env=env)
        return proc, workspace

    # 1. The rendered env file's setup keys, with no flags at all.
    proc, workspace = submit(
        tmp_path / "ws-env",
        STEERLAB_SETUP_CPUS="6", STEERLAB_SETUP_MEMORY="32G",
        STEERLAB_SETUP_WALLTIME="03:00:00",
        STEERLAB_SETUP_EXTRA_SBATCH="--hint=nomultithread --nice=50")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    script = next(workspace.glob("steerlab-bootstrap.*.sbatch")).read_text()
    assert "#SBATCH --cpus-per-task=6" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --time=03:00:00" in script
    assert "#SBATCH --hint=nomultithread" in script
    assert "#SBATCH --nice=50" in script

    # 2. Flags win over the environment, and REPLACE its directive list.
    proc, workspace = submit(
        tmp_path / "ws-flags", "--job-cpus", "2", "--job-memory", "8G",
        "--job-extra-sbatch", "--nice=100",
        STEERLAB_SETUP_CPUS="6", STEERLAB_SETUP_EXTRA_SBATCH="--hint=nomultithread")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    script = next(workspace.glob("steerlab-bootstrap.*.sbatch")).read_text()
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --mem=8G" in script
    assert "#SBATCH --nice=100" in script
    assert "nomultithread" not in script

    # 3. A directive that is not one is refused, not written into a job script.
    proc, _ = submit(tmp_path / "ws-bad", "--job-extra-sbatch", "; rm -rf /")
    assert proc.returncode == 64
    assert "invalid --job-extra-sbatch" in proc.stderr


def test_bootstrap_job_honors_queue_seam_and_does_not_mistake_query_error_for_exit(
        tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    query_marker = tmp_path / "query.seen"
    fake_bootstrap = tmp_path / "bootstrap.sh"
    fake_bootstrap.write_text(
        "#!/bin/bash\n"
        f"while [ ! -f {query_marker!s} ]; do sleep 0.01; done\n"
        "echo '{\"ok\":true,\"steps\":{},\"envFile\":null,\"prefix\":null}'\n",
        encoding="utf-8")
    fake_bootstrap.chmod(fake_bootstrap.stat().st_mode | stat.S_IXUSR)

    fake_sbatch = bindir / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "job=\"${@: -1}\"\n"
        "log=$(sed -n 's/^#SBATCH --output=//p' \"$job\")\n"
        "(SLURM_JOB_ID=4343 bash \"$job\" > \"$log\" 2>&1) &\n"
        "echo 4343\n",
        encoding="utf-8")
    fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)
    query_log = tmp_path / "query.log"
    queue_wrapper = bindir / "queue-wrapper"
    queue_wrapper.write_text(
        "#!/bin/bash\n"
        f"echo \"$*\" >> {query_log!s}\n"
        # Release the job only once the FOLLOW loop's own `-j` query has run
        # (the pre-submit adoption guard asks a different, by-name question),
        # so the wait loop is guaranteed to meet the failing query.
        f'case "$*" in *" -j "*) touch {query_marker!s} ;; esac\n'
        "echo temporarily-unavailable >&2\n"
        "exit 2\n",
        encoding="utf-8")
    queue_wrapper.chmod(queue_wrapper.stat().st_mode | stat.S_IXUSR)

    workspace = tmp_path / "workspace"
    env = {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin",
           "HOME": str(tmp_path), "USER": "tester",
           "STEERLAB_SLURM_SQUEUE": "queue-wrapper",
           "STEERLAB_BOOTSTRAP_POLL_SECONDS": "0.01",
           "STEERLAB_BOOTSTRAP_HEARTBEAT_POLLS": "2"}
    proc = subprocess.run(
        [bash, BOOTSTRAP_JOB, "--wait",
         "--bootstrap-script", str(fake_bootstrap),
         "--job-workspace", str(workspace),
         "--job-partition", "batch", "--job-cpus", "1",
         "--job-memory", "1G", "--job-walltime", "00:10:00", "--"],
        text=True, capture_output=True, check=False, env=env, timeout=5)

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "queue query failed; job 4343 may still be active" in proc.stderr
    assert "completed successfully" in proc.stdout
    assert "-h -j 4343 -o %T" in query_log.read_text(encoding="utf-8")


# --- submit/wait split + durable adoption (review finding 5) ---------------
#
# The defect: the helper submitted and then held the SSH session open for the
# whole bootstrap. A sleeping Mac or a dropped connection between sbatch and
# the end of the poll loop lost every record of a job that was really
# submitted, so a retry queued a SECOND bootstrap.


def _scriptbin(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _detached_slurm(tmp_path):
    """A fake scheduler that ACCEPTS the job and leaves it queued forever, plus
    a squeue whose answer is driven by env vars. This is the shape the defect
    lives in: the job exists, but nothing about it has finished."""
    bindir = tmp_path / "bin"
    fake_bootstrap = tmp_path / "bootstrap.sh"
    fake_bootstrap.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_bootstrap.chmod(fake_bootstrap.stat().st_mode | stat.S_IXUSR)
    _scriptbin(bindir, "sbatch",
               'echo "$@" >> "$FAKE_SBATCH_CALLS"\necho 5150\n')
    # `-n <name>` is the by-name backstop query; anything else is `-j <id>`.
    _scriptbin(bindir, "squeue",
               'case "$*" in *" -n "*) printf "%s" "${FAKE_BY_NAME:-}"'
               ' ; [ -n "${FAKE_BY_NAME:-}" ] && echo ; exit 0 ;; esac\n'
               '[ "${FAKE_SQUEUE_FAILS:-0}" = "1" ] && '
               '{ echo unavailable >&2; exit 2; }\n'
               'printf "%s" "${FAKE_STATE:-}"\n'
               '[ -n "${FAKE_STATE:-}" ] && echo\nexit 0\n')
    return bindir, fake_bootstrap


def _submit(tmp_path, bindir, fake_bootstrap, workspace, *extra, **env_extra):
    env = {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin",
           "HOME": str(tmp_path), "USER": "tester",
           "FAKE_SBATCH_CALLS": str(tmp_path / "sbatch.calls"),
           **env_extra}
    return subprocess.run(
        [bash, BOOTSTRAP_JOB, *extra,
         "--bootstrap-script", str(fake_bootstrap),
         "--job-workspace", str(workspace),
         "--job-partition", "batch", "--job-cpus", "1",
         "--job-memory", "1G", "--job-walltime", "00:10:00", "--"],
        text=True, capture_output=True, check=False, env=env, timeout=10)


def _status(tmp_path, bindir, workspace, *extra, **env_extra):
    env = {**os.environ, "PATH": f"{bindir}:/usr/bin:/bin",
           "HOME": str(tmp_path), "USER": "tester", **env_extra}
    return subprocess.run(
        [bash, BOOTSTRAP_JOB, "--status", "--job-workspace", str(workspace),
         *extra],
        text=True, capture_output=True, check=False, env=env, timeout=10)


def _verdict(proc):
    for line in proc.stdout.splitlines():
        if line.startswith("STEERLAB_BOOTSTRAP_STATUS="):
            return line.split("=", 1)[1].split(" ", 1)[0]
    return None


def _breadcrumb(workspace):
    text = (workspace / "steerlab-bootstrap.pending").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def test_bootstrap_job_submits_and_returns_without_holding_the_connection(
        tmp_path):
    """The default no longer waits: sbatch, print the durable handles, exit.

    The job id and its status file must be on stdout BEFORE anything can be
    interrupted — that is the whole recovery contract with the caller.
    """
    bindir, fake_bootstrap = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    proc = _submit(tmp_path, bindir, fake_bootstrap, workspace,
                   FAKE_STATE="PENDING")
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "STEERLAB_BOOTSTRAP_JOB_ID=5150" in proc.stdout
    assert "STEERLAB_BOOTSTRAP_STATUS_FILE=" in proc.stdout
    assert "submitted and detached" in proc.stdout
    # It did NOT follow the job.
    assert "completed successfully" not in proc.stdout
    assert "state PENDING" not in proc.stdout

    crumb = _breadcrumb(workspace)
    assert crumb["jobID"] == "5150"
    assert crumb["statusFile"].endswith(".status")
    assert crumb["log"].endswith(".log")
    # The breadcrumb points at the job the caller was told about.
    assert f"STEERLAB_BOOTSTRAP_STATUS_FILE={crumb['statusFile']}" in proc.stdout


def test_bootstrap_status_reads_the_queue_then_the_status_file(tmp_path):
    bindir, fake_bootstrap = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    _submit(tmp_path, bindir, fake_bootstrap, workspace, FAKE_STATE="PENDING")
    status_file = pathlib.Path(_breadcrumb(workspace)["statusFile"])

    # Queued, then running: read from the scheduler.
    assert _verdict(_status(tmp_path, bindir, workspace,
                            FAKE_STATE="PENDING")) == "pending"
    assert _verdict(_status(tmp_path, bindir, workspace,
                            FAKE_STATE="RUNNING")) == "running"
    # The job's own status file outranks the queue, in both directions.
    status_file.write_text("0\n", encoding="utf-8")
    assert _verdict(_status(tmp_path, bindir, workspace,
                            FAKE_STATE="RUNNING")) == "completed-ok"
    status_file.write_text("17\n", encoding="utf-8")
    assert _verdict(_status(tmp_path, bindir, workspace,
                            FAKE_STATE="")) == "completed-code-17"
    # Gone from the queue with no status record: the job died unrecorded.
    status_file.unlink()
    assert _verdict(_status(tmp_path, bindir, workspace,
                            FAKE_STATE="")) == "vanished"
    # A FAILED query is never proof of death — the repository's standing rule.
    assert _verdict(_status(tmp_path, bindir, workspace,
                            FAKE_SQUEUE_FAILS="1")) == "unknown"


def test_bootstrap_status_forwards_the_job_log_so_the_report_survives(tmp_path):
    """The caller used to parse bootstrap's JSON report out of the streamed
    log. With the wait removed, a terminal status must carry that log tail —
    or a detached bootstrap could never report which step failed."""
    bindir, fake_bootstrap = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    _submit(tmp_path, bindir, fake_bootstrap, workspace, FAKE_STATE="PENDING")
    crumb = _breadcrumb(workspace)
    pathlib.Path(crumb["log"]).write_text(
        "installing…\n"
        '{"ok":true,"steps":{"condaDetect":"ok"},"envFile":"/home/me/e",'
        '"prefix":"/home/me/p"}\n', encoding="utf-8")
    pathlib.Path(crumb["statusFile"]).write_text("0\n", encoding="utf-8")

    proc = _status(tmp_path, bindir, workspace, FAKE_STATE="")
    assert proc.returncode == 0, proc.stderr
    assert _verdict(proc) == "completed-ok"
    # The report is the LAST json line, exactly as the transcript parser wants.
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["prefix"] == "/home/me/p"


def test_bootstrap_job_adopts_an_in_flight_job_instead_of_resubmitting(
        tmp_path):
    """The heart of the fix: a retry after a dropped connection must find the
    job that is already queued, not queue a second one."""
    bindir, fake_bootstrap = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    _submit(tmp_path, bindir, fake_bootstrap, workspace, FAKE_STATE="PENDING")
    calls = (tmp_path / "sbatch.calls").read_text(encoding="utf-8")
    assert len(calls.splitlines()) == 1

    again = _submit(tmp_path, bindir, fake_bootstrap, workspace,
                    FAKE_STATE="RUNNING")
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert "STEERLAB_BOOTSTRAP_ADOPT=5150" in again.stdout
    assert "STEERLAB_BOOTSTRAP_JOB_ID=5150" in again.stdout
    assert _verdict(again) == "running"
    # Nothing was submitted the second time.
    assert (tmp_path / "sbatch.calls").read_text(encoding="utf-8") == calls

    # An unreadable queue adopts too: an unproven death never licenses a
    # resubmit (the same doctrine the controller layer already enforces).
    unreadable = _submit(tmp_path, bindir, fake_bootstrap, workspace,
                         FAKE_SQUEUE_FAILS="1")
    assert "STEERLAB_BOOTSTRAP_ADOPT=5150" in unreadable.stdout
    assert (tmp_path / "sbatch.calls").read_text(encoding="utf-8") == calls

    # --force-new is the operator's override.
    forced = _submit(tmp_path, bindir, fake_bootstrap, workspace, "--force-new",
                     FAKE_STATE="RUNNING")
    assert "STEERLAB_BOOTSTRAP_ADOPT" not in forced.stdout
    assert len((tmp_path / "sbatch.calls").read_text(
        encoding="utf-8").splitlines()) == 2


def test_bootstrap_job_resubmits_once_the_recorded_job_has_finished(tmp_path):
    bindir, fake_bootstrap = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    _submit(tmp_path, bindir, fake_bootstrap, workspace, FAKE_STATE="PENDING")
    pathlib.Path(_breadcrumb(workspace)["statusFile"]).write_text(
        "0\n", encoding="utf-8")
    again = _submit(tmp_path, bindir, fake_bootstrap, workspace, FAKE_STATE="")
    assert "STEERLAB_BOOTSTRAP_ADOPT" not in again.stdout
    assert len((tmp_path / "sbatch.calls").read_text(
        encoding="utf-8").splitlines()) == 2


def test_bootstrap_job_falls_back_to_the_queue_when_the_breadcrumb_is_gone(
        tmp_path):
    """A cleaned scratch directory loses the breadcrumb but not the job. Only
    then is the broader by-name query asked (it could match another
    workspace's bootstrap, which is why it is the backstop, not the rule)."""
    bindir, fake_bootstrap = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    _submit(tmp_path, bindir, fake_bootstrap, workspace, FAKE_STATE="PENDING")
    (workspace / "steerlab-bootstrap.pending").unlink()
    again = _submit(tmp_path, bindir, fake_bootstrap, workspace,
                    FAKE_BY_NAME="5150 RUNNING", FAKE_STATE="RUNNING")
    assert "STEERLAB_BOOTSTRAP_ADOPT=5150" in again.stdout
    assert len((tmp_path / "sbatch.calls").read_text(
        encoding="utf-8").splitlines()) == 1


def test_bootstrap_status_refuses_without_a_job_to_report_on(tmp_path):
    bindir, _ = _detached_slurm(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proc = _status(tmp_path, bindir, workspace)
    assert proc.returncode == 64
    assert "no job breadcrumb" in proc.stderr
    assert _verdict(_status(tmp_path, bindir, workspace, "--job-id", "77",
                            FAKE_STATE="PENDING")) == "pending"


def test_controller_template_refuses_without_token(tmp_path):
    env_file = tmp_path / "cluster.env"
    env_file.write_text(
        "export STEERLAB_SERVER_PROFILE=cluster\n"
        "export STEERLAB_EXECUTOR=slurm\n", encoding="utf-8")
    proc = subprocess.run(
        [bash, CONTROLLER], text=True, capture_output=True, check=False,
        env={**os.environ, "STEERLAB_ENV_FILE": str(env_file),
             "STEERLAB_AUTH_TOKEN": ""})
    assert proc.returncode == 64
    assert "refusing to bind 0.0.0.0 without STEERLAB_AUTH_TOKEN" in proc.stderr


def test_controller_template_refuses_non_cluster_profile(tmp_path):
    env_file = tmp_path / "cluster.env"
    env_file.write_text("export STEERLAB_SERVER_PROFILE=local\n",
                        encoding="utf-8")
    proc = subprocess.run(
        [bash, CONTROLLER], text=True, capture_output=True, check=False,
        env={**os.environ, "STEERLAB_ENV_FILE": str(env_file)})
    assert proc.returncode == 64
    assert "STEERLAB_SERVER_PROFILE" in proc.stderr


def test_controller_template_documents_the_contract():
    text = open(CONTROLLER, encoding="utf-8").read()
    # The bind decision is deliberate and must stay documented in-file.
    assert "THE BIND DECISION" in text
    assert "STEERLAB_BIND=0.0.0.0" in text
    assert "STEERLAB_AUTH_MODE=token" in text
    # Resubmit chain (rewritten 2026-08-18, open-issues §1): a pre-expiry USR1
    # trap that queues one successor, replacing the `--dependency=afterany`
    # toggle nothing ever set. The two anchors `_chain_library` slices on must
    # stay put, and the billed-allocation opt-out must stay documented.
    assert CHAIN_START in text
    assert f"\n{CHAIN_END}\n" in text
    assert "chain_successor()" in text
    assert "STEERLAB_CONTROLLER_RESUBMIT" in text
    assert "serverd.chain.json" in text
    assert "serverd.no-chain" in text
    # The node-discovery contract the app-side tunnel manager reads.
    assert "serverd.host" in text


def test_controller_template_app_render_composes_headers_and_leaves_no_placeholders(
        tmp_path):
    """Mirror the app-side render command (ClusterProvisioner
    .controllerRemoteCommand) as WP5 Step 9 rewrote it: resolve $WS from the
    bootstrap env file, mkdir the log dir, PRINT the composed #SBATCH block
    (from the shared header renderer — here the block that renderer produces
    for a site with a `batch` partition and a `lab1` account) under the
    shebang, append the workspace-side log directives, and pipe the template's
    BODY (`tail -n +2`, so the file gets exactly one shebang) through the two
    body placeholders that remain.

    The rendered file must carry no unresolved placeholder on any active line
    — the defect where @WORKSPACE@ survived to sbatch and Slurm silently
    dropped the logs — and every directive must precede the first executable
    line, or sbatch stops reading them."""
    ws = tmp_path / "ws"
    env_file = tmp_path / "cluster.env"
    env_file.write_text(f'export STEERLAB_ROOT="{ws}"\n', encoding="utf-8")
    rendered = tmp_path / "controller-job.sbatch"
    headers = " ".join(
        f"'{line}'" for line in (
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=steerlab-serverd",
            "#SBATCH --partition=batch",
            "#SBATCH --account=lab1",
            "#SBATCH --time=24:00:00",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --mem=16G",
        ))

    def render_command(envfile):
        return (
            f'WS="$( . {envfile} 2>/dev/null; printf \'%s\' "${{STEERLAB_ROOT:-}}" )" && '
            '[ -n "$WS" ] || { echo "controller-job: no workspace root" >&2; exit 64; }; '
            f'mkdir -p {tmp_path}/meta "$WS/runs" && '
            # Render provenance, computed in the REMOTE shell (open-issues §1
            # field report, 2026-08-20). Every substitution exits 0 even when
            # the fact is missing, or this `&&` chain would skip the render.
            f'STEERLAB_TPL_SHA="$( {{ sha256sum "{CONTROLLER}" 2>/dev/null '
            f'|| shasum -a 256 "{CONTROLLER}" 2>/dev/null '
            '|| echo unknown; } | awk \'{print $1}\' )" && '
            'STEERLAB_RENDERED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ || echo unknown)" && '
            'STEERLAB_SOURCE_REV="$( { cat /nonexistent/BUILD_COMMIT 2>/dev/null '
            '|| echo unknown; } | head -n 1 )" && '
            "{ printf '%s\\n' " + headers + " "
            '"#SBATCH --output=$WS/runs/steerlab-serverd.%j.out" '
            '"#SBATCH --error=$WS/runs/steerlab-serverd.%j.err"; '
            f'tail -n +2 "{CONTROLLER}" '
            "| sed -e 's|@PYTHON@|/home/me/envs/steerlab/bin/python|g' "
            "-e 's|@PORT@|8080|g'"
            ' -e "s|@TEMPLATE_SHA256@|$STEERLAB_TPL_SHA|g"'
            ' -e "s|@RENDERED_AT@|$STEERLAB_RENDERED_AT|g"'
            ' -e "s|@SOURCE_REVISION@|$STEERLAB_SOURCE_REV|g"; } '
            f'> "{rendered}" && cd "$WS" && pwd'
        )

    proc = subprocess.run([bash, "-c", render_command(env_file)], text=True,
                          capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert (ws / "runs").is_dir()   # created BEFORE sbatch, or Slurm drops logs
    # sbatch would run from $WS (scratch side), not the login shell's cwd.
    assert proc.stdout.strip() == os.path.realpath(ws)
    text = rendered.read_text(encoding="utf-8")
    lines = text.splitlines()
    active = [ln for ln in lines
              if ln.startswith("#SBATCH ") or not ln.lstrip().startswith("#")]
    leftovers = [ln for ln in active if re.search(r"@[A-Z0-9_]+@", ln)]
    assert not leftovers, leftovers
    # Exactly one shebang, and it leads.
    assert lines[0] == "#!/usr/bin/env bash"
    assert [ln for ln in lines if ln.startswith("#!")] == ["#!/usr/bin/env bash"]
    # Every directive is above the first line sbatch stops reading at.
    first_code = next(i for i, ln in enumerate(lines[1:], 1)
                      if ln.strip() and not ln.lstrip().startswith("#"))
    directives = [i for i, ln in enumerate(lines) if ln.startswith("#SBATCH ")]
    assert directives and max(directives) < first_code
    # The site's declared facts, composed — not constants living in the script.
    assert "#SBATCH --partition=batch" in text
    assert "#SBATCH --account=lab1" in text
    assert "#SBATCH --mem=16G" in text
    assert "#SBATCH --ntasks=1" in text
    assert f"#SBATCH --output={ws}/runs/steerlab-serverd.%j.out" in text
    assert f"#SBATCH --error={ws}/runs/steerlab-serverd.%j.err" in text
    # The render stamp names the TEMPLATE's own bytes, and the chain marker
    # the server reads at boot carries the same digest (§1 field report).
    template_sha = hashlib.sha256(open(CONTROLLER, "rb").read()).hexdigest()
    stamp = next(ln for ln in lines
                 if ln.startswith("# steerlab-render-stamp:"))
    assert f"sha256={template_sha}" in stamp
    assert "renderedAt=20" in stamp and "source=unknown" in stamp
    assert (f'export STEERLAB_CONTROLLER_CHAIN="template-{template_sha}"'
            in text)
    # The body still runs: the refusals survive rendering.
    assert "refusing to bind 0.0.0.0" in text
    # The server is a background child, not an `exec` — the walltime chain's
    # USR1 trap cannot fire in a shell that has replaced itself (§1).
    assert '"/home/me/envs/steerlab/bin/python" -m steerlab_server.cli serve' in text
    assert 'wait "$STEERLAB_SERVER_PID"' in text
    # FAIL CLOSED when neither the site profile nor the env file names a
    # workspace: never a quiet $HOME fallback (job logs must not land /home).
    proc = subprocess.run([bash, "-c", render_command(tmp_path / "missing.env")],
                          text=True, capture_output=True, check=False,
                          env={**os.environ, "STEERLAB_ROOT": ""})
    assert proc.returncode == 64
    assert "no workspace root" in proc.stderr


def test_bootstrap_hello_job_headers_and_workspace_paths(tmp_path):
    """The hello job obeys the required-header rule (--ntasks) and keeps its
    script, working dir, and logs workspace-side — never a relative --output
    that lands wherever bootstrap ran (usually /home)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_sbatch = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "fakebin", "sbatch")
    target = bindir / "sbatch"
    shutil.copy(fake_sbatch, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR)
    # A stub conda satisfies detection; the pre-created prefix python makes
    # envCreate take its idempotent "env exists — reusing" path, so neither
    # stub is ever really executed. The default --repo has no Server/ under
    # this HOME, so serverInstall fails LOUDLY but the run continues to the
    # hello step (only conda/envCreate failures abort).
    for stub in (bindir / "conda", tmp_path / "env" / "bin" / "python"):
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    ws = tmp_path / "ws"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("STEERLAB_") and not k.startswith("FAKE_")}
    # Inside an allocation (guard passes); PATH = stubs + system tools. The
    # fake sbatch logs its cwd — the submit must happen FROM the workspace
    # (--chdir moves the job's working dir, not the submit command's).
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    env.update(PATH=f"{bindir}:/usr/bin:/bin", SLURM_JOB_ID="123456",
               HOME=str(tmp_path), FAKE_SLURM_LOG=str(log_dir))
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--hello", "--no-torch",
         "--workspace", str(ws),
         "--env-file", str(tmp_path / "cluster.env"),
         "--prefix", str(tmp_path / "env")],
        text=True, capture_output=True, check=False, env=env)
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["helloJob"] == "ok", (proc.stdout, proc.stderr)
    script = (ws / "steerlab-hello.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in script
    assert f"#SBATCH --chdir={ws}" in script
    assert f"#SBATCH --output={ws}/%x.%j.out" in script
    assert "#SBATCH --output=%x.%j.out" not in script
    cwds = (log_dir / "sbatch-cwd.calls").read_text(encoding="utf-8").splitlines()
    assert cwds == [os.path.realpath(ws)]


def test_controller_template_carries_no_site_headers_of_its_own():
    """WP5 Step 9 (audit G6): the template is the controller's shell BODY.

    Its `#SBATCH` block used to be a fixed set with four @PLACEHOLDER@s, which
    meant partition, walltime, cores, memory and the account were the SCRIPT's
    facts — a site could declare `scheduler.controllerJob` and be ignored, and
    the required-header guarantee this test used to assert here was a guarantee
    about a constant. The block is now composed from the profile by the shared
    header renderer at render time, so the guarantee lives on the RENDERED file
    (asserted above); what must hold HERE is that no second, stale copy of
    those directives survives in the template to contradict it."""
    text = open(CONTROLLER, encoding="utf-8").read()
    directives = [ln for ln in text.splitlines() if ln.startswith("#SBATCH ")]
    assert not directives, directives
    # Only the body's own placeholders remain; the site-fact ones are gone.
    assert set(re.findall(r"@([A-Z]+)@", text)) <= {"PYTHON", "PORT", "PLACEHOLDER"}
    # …and the file says where the block now comes from, so a hand-runner is
    # not left with a script Slurm will reject for want of a partition.
    assert "RENDERED, NOT AUTHORED" in text
    assert "cluster preview --site <id> --job-class controller" in text
    # Job logs still belong /scratch-side (the workspace runs/ dir), never the
    # submit-time cwd on /home (don't park job data in /home).
    assert "#SBATCH --output=steerlab-serverd.%j.out" not in text
    assert "job data never lands in /home" in text


# --- the login/submit-node guard (the built-in fallback rule) --------------------------------

def _guardable_env(**extra):
    """An env with no Slurm allocation and a minimal PATH (no conda/mamba),
    so the guard decision — not this machine's toolchain — drives the test."""
    env = {k: v for k, v in os.environ.items()
           if k not in {"SLURM_JOB_ID"} and not k.startswith("STEERLAB_")}
    env.update(extra)
    return env


def test_bootstrap_refuses_outside_a_slurm_allocation(tmp_path):
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--env-file", str(tmp_path / "cluster.env"),
         "--prefix", str(tmp_path / "env")],
        text=True, capture_output=True, check=False, env=_guardable_env())
    assert proc.returncode == 64
    assert "refusing to run" in proc.stderr
    # The remedy is part of the contract. Since WP5 step 12 it is GENERIC —
    # the script names `salloc`/`srun --pty` and points at the site's own
    # published wrapper, instead of hardcoding one institution's verb (the
    # site-declared branch already prints
    # environment.interactiveAllocationCommand).
    assert "get an allocation with your site's command" in proc.stderr
    assert "salloc / srun --pty" in proc.stderr
    assert "--force-login" in proc.stderr
    assert not (tmp_path / "cluster.env").exists()


def test_bootstrap_force_login_skips_the_guard(tmp_path):
    # With --force-login (the xfer-node pip-rerun path) the guard is skipped;
    # with an empty PATH the run then fails at conda detection — proof it got
    # PAST the guard without touching this machine.
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--force-login",
         "--env-file", str(tmp_path / "cluster.env"),
         "--prefix", str(tmp_path / "env")],
        text=True, capture_output=True, check=False,
        env=_guardable_env(PATH=str(tmp_path)))
    assert proc.returncode == 1
    assert "refusing to run" not in proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["condaDetect"] == "failed"


def test_bootstrap_slurm_allocation_passes_the_guard(tmp_path):
    # Inside an allocation (SLURM_JOB_ID set, as interact/batch do) no flag is
    # needed; the empty PATH again stops the run at conda detection.
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--env-file", str(tmp_path / "cluster.env"),
         "--prefix", str(tmp_path / "env")],
        text=True, capture_output=True, check=False,
        env=_guardable_env(PATH=str(tmp_path), SLURM_JOB_ID="123456"))
    assert proc.returncode == 1
    assert "refusing to run" not in proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["steps"]["condaDetect"] == "failed"


def _policy_guard_run(tmp_path, hostname, env_text=None, **extra_env):
    """Run bootstrap.sh with a FAKE `hostname`, so the guard's decision is about
    the declared policy and not about the machine the suite runs on.

    PATH holds only that fakebin plus the handful of coreutils the guard itself
    needs — no conda, so a run that gets PAST the guard stops at conda
    detection and never touches this machine."""
    bindir = tmp_path / "guardbin"
    _scriptbin(bindir, "hostname", f'printf "%s\\n" {hostname}\n')
    for tool in ("sed", "tail", "grep", "id"):
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    flags = []
    if env_text is not None:
        source = tmp_path / "rendered.env"
        source.write_text(env_text, encoding="utf-8")
        flags = ["--env-file-from", str(source)]
    return subprocess.run(
        [bash, BOOTSTRAP, "--env-file", str(tmp_path / "cluster.env"),
         "--prefix", str(tmp_path / "env"), *flags],
        text=True, capture_output=True, check=False, timeout=60,
        env=_guardable_env(PATH=str(bindir), HOME=str(tmp_path), **extra_env))


def test_bootstrap_guard_honors_a_site_declared_hostname_pattern(tmp_path):
    """WP5 step 10 (audit c35/c36): the guard is the SITE's, not the script's. A site
    that calls its login nodes `login*` is honoured on a host whose name the script's
    `^ss-sub` would never match — and the refusal names policy.loginNodes rather
    than the script's built-in rule."""
    declared = ("export STEERLAB_LOGIN_NODE_PATTERNS='^login ^submit[0-9]+'\n"
                "export STEERLAB_LOGIN_NODE_ALLOW_COMPUTE=0\n"
                "export STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION=0\n")
    refused = _policy_guard_run(tmp_path / "a", "login3.example.edu", declared,
                                SLURM_JOB_ID="4242")
    assert refused.returncode == 64
    assert "refusing to run on 'login3.example.edu'" in refused.stderr
    assert "policy.loginNodes" in refused.stderr
    assert "BUILT-IN FALLBACK" not in refused.stderr
    assert not (tmp_path / "a" / "cluster.env").exists()

    # A compute host at the same site is not a login node, so it runs: past the
    # guard, and only conda detection stops it.
    allowed = _policy_guard_run(tmp_path / "b", "c4-13.example.edu", declared)
    assert allowed.returncode == 1
    assert "refusing to run" not in allowed.stderr
    assert _report(allowed)["steps"]["condaDetect"] == "failed"


def test_bootstrap_guard_never_refuses_when_the_site_declares_no_login_nodes(tmp_path):
    """The other half of the audit's step-10 gate: an EMPTY pattern list never
    refuses. The site says nothing is a login node and no allocation is needed,
    so the historical `^ss-sub` host — outside any allocation — runs. This is
    what makes the guard the site's to turn off, which is the whole of c35."""
    proc = _policy_guard_run(
        tmp_path, "ss-sub1.example.edu",
        "export STEERLAB_LOGIN_NODE_ALLOW_COMPUTE=1\n"
        "export STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION=0\n")
    assert proc.returncode == 1
    assert "refusing to run" not in proc.stderr
    assert "permitted by this site's" in proc.stdout
    assert _report(proc)["steps"]["condaDetect"] == "failed"


def test_bootstrap_guard_falls_back_to_its_built_in_rule_without_a_profile(tmp_path):
    """No rendered policy reached this run — a hand-run bootstrap.sh — so the
    script applies the rule it has always applied, and says so. Same shape as
    step 7's fallback heredoc: the constants were demoted to defaults, not
    deleted, and the message labels them as such."""
    proc = _policy_guard_run(tmp_path, "ss-sub2.example.edu", SLURM_JOB_ID="4242")
    assert proc.returncode == 64
    assert "refusing to run on 'ss-sub2.example.edu'" in proc.stderr
    assert "BUILT-IN FALLBACK" in proc.stderr
    # The historical RULE survives (`^ss-sub`, stated as the script's own);
    # the historical remedy is now generic (WP5 step 12), because a wrapper
    # verb is one institution's, and a site that has one declares it.
    assert "'^ss-sub'" in proc.stderr
    assert "get an allocation with your site's command" in proc.stderr
    for identifier in ("sapelo", "gacrc", "uga.edu"):
        assert identifier not in proc.stderr.lower(), identifier


def test_the_committed_v1_render_reproduces_the_historical_guard(tmp_path):
    """End to end, across the artifacts: the env file the RENDERER produces for
    a v1 (pre-schema-2) site carries the same guard the script used to hardcode.
    Materializing an existing site must not disarm it — which is why the v1
    default set states `^ss-sub` rather than leaving it to the script."""
    golden = (pathlib.Path(__file__).resolve().parent.parent.parent
              / "prompts" / "fixtures" / "cluster-site-profile"
              / "v1-maximal.env.golden.txt").read_text(encoding="utf-8")
    assert "export STEERLAB_LOGIN_NODE_PATTERNS='^ss-sub'" in golden
    refused = _policy_guard_run(tmp_path / "a", "ss-sub1.site.example", golden,
                                SLURM_JOB_ID="4242")
    assert refused.returncode == 64
    assert "policy.loginNodes" in refused.stderr
    # …and, as before schema 2, being outside an allocation refuses too.
    outside = _policy_guard_run(tmp_path / "b", "c4-13.example", golden)
    assert outside.returncode == 64
    assert "outside a Slurm allocation" in outside.stderr


def test_bootstrap_dry_run_works_anywhere(tmp_path):
    # --dry-run never reaches the guard — it must work on a login node.
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run",
         "--env-file", str(tmp_path / "cluster.env")],
        text=True, capture_output=True, check=False, env=_guardable_env())
    assert proc.returncode == 0
    assert "refusing to run" not in proc.stderr


# --- WP5 steps 6-7: materialization (--env-file-from) ----------------------
#
# Step 6 added the flag; step 7 made it the path the app and steerlab-cli take
# by default and demoted the script's own heredoc to the MANUAL, no-profile
# fallback. Every test above still runs the fallback unmodified, which is the
# no-regression proof: a hand-run bootstrap.sh converges exactly as it always
# did. What changed is that the profile-driven path no longer consults the
# heredoc's constants at all, so the site's declared facts (audit a1-a7) are
# what land.

#: A file shaped like `ClusterEnvironmentRenderer.renderEnvFile` output — the
#: comment header, `export` lines in all three quotings, and the token as a
#: `$(cat …)` PATH indirection rather than a value.
RENDERED_ENV = (
    "# SteerLab cluster environment — rendered from the site profile.\n"
    "# Site: Example Slurm (profile schema 2, neutral defaults)\n"
    "export STEERLAB_SERVER_PROFILE=cluster\n"
    'export STEERLAB_ROOT="$HOME/ws"\n'
    "export STEERLAB_NODE_STAGE_DIR='/lscratch/$SLURM_JOB_ID'\n"
    'export STEERLAB_AUTH_TOKEN="$(cat "$HOME/.steerlab-token")"\n'
)


def _materialized_run(tmp_path, *, source_text=RENDERED_ENV, write_source=True,
                      sha256=None, extra=(), pre_existing_env=None,
                      materialize=True):
    """Drive bootstrap.sh all the way to the env-file step with fake tooling.

    conda, the env prefix's python, and the repo are all stand-ins: the steps
    before the env file must merely SUCCEED, so that what the test observes is
    the env-file step and nothing else.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    prefix = tmp_path / "envprefix"
    repo = tmp_path / "repo"
    (repo / "Server").mkdir(parents=True, exist_ok=True)
    _scriptbin(prefix / "bin", "python", "exit 0\n")
    bindir = tmp_path / "fakebin"
    _scriptbin(bindir, "conda", "exit 0\n")

    source = tmp_path / "rendered.env"
    if write_source:
        source.write_text(source_text, encoding="utf-8")
    env_file = tmp_path / "cluster.env"
    if pre_existing_env is not None:
        env_file.write_text(pre_existing_env, encoding="utf-8")

    flags = ["--env-file-from", str(source)] if materialize else []
    if sha256 is not None:
        flags += ["--env-file-sha256", sha256]
    env = {k: v for k, v in os.environ.items() if not k.startswith("STEERLAB_")}
    env.update({"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(home),
                "USER": "tester", "SLURM_JOB_ID": "424242"})
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--no-torch",
         "--prefix", str(prefix), "--repo", str(repo),
         "--workspace", str(tmp_path / "ws"), "--hf-cache", str(tmp_path / "hf"),
         "--env-file", str(env_file), *flags, *extra],
        text=True, capture_output=True, check=False, env=env, timeout=60)
    return proc, env_file, source


def _report(proc):
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_env_file_from_installs_the_rendered_file_verbatim(tmp_path):
    """The whole point of materialization: what the renderer produced is what
    the cluster sources, byte for byte. A script that 'mostly' reproduced it
    would put the two engines' environments quietly out of step."""
    digest = hashlib.sha256(RENDERED_ENV.encode("utf-8")).hexdigest()
    proc, env_file, _ = _materialized_run(tmp_path, sha256=digest)
    assert proc.returncode == 0, proc.stderr
    report = _report(proc)
    assert report["steps"]["envFile"] == "ok"
    assert env_file.read_text(encoding="utf-8") == RENDERED_ENV
    assert "installed" in proc.stdout
    # None of the heredoc's own constants leaked in alongside it.
    assert "STEERLAB_SLURM_MEMORY" not in env_file.read_text(encoding="utf-8")


def test_env_file_from_refuses_a_digest_that_does_not_match_the_reviewed_plan(
        tmp_path):
    """The integrity half of the review gate. The digest in --env-file-sha256
    is inside the bootstrap plan hash the human approved, so a file that hashes
    to anything else is by definition not what was approved."""
    proc, env_file, _ = _materialized_run(tmp_path, sha256="00" * 32)
    assert proc.returncode == 1
    assert _report(proc)["steps"]["envFile"] == "failed"
    assert "does not match the reviewed plan" in proc.stderr
    # Nothing was installed: a refused verification leaves the site untouched.
    assert not env_file.exists()


def test_env_file_from_refuses_a_malformed_file(tmp_path):
    """A rendered env file is comments, blanks, and `export KEY=…`. Anything
    else is not a render — a truncated push, an error page, or something that
    would EXECUTE when the file is sourced."""
    proc, env_file, _ = _materialized_run(
        tmp_path,
        source_text=RENDERED_ENV + "rm -rf \"$HOME\"\n")
    assert proc.returncode == 1
    assert _report(proc)["steps"]["envFile"] == "failed"
    assert "not a rendered env file" in proc.stderr
    assert not env_file.exists()


def test_env_file_from_refuses_an_empty_or_missing_source(tmp_path):
    missing, env_file, _ = _materialized_run(tmp_path, write_source=False)
    assert missing.returncode == 1
    assert "does not exist" in missing.stderr
    assert not env_file.exists()

    empty, env_file2, _ = _materialized_run(
        tmp_path / "b", source_text="")
    assert empty.returncode == 1
    assert "is empty" in empty.stderr
    assert not env_file2.exists()


def test_env_file_from_keeps_the_no_clobber_rule(tmp_path):
    """--force means the same thing on both paths: an existing env file is a
    local edit until the caller says otherwise."""
    proc, env_file, _ = _materialized_run(
        tmp_path, pre_existing_env="export STEERLAB_HAND_EDITED=1\n")
    assert proc.returncode == 0, proc.stderr
    assert _report(proc)["steps"]["envFile"] == "skipped"
    assert env_file.read_text(encoding="utf-8") == "export STEERLAB_HAND_EDITED=1\n"

    forced, forced_file, _ = _materialized_run(
        tmp_path / "c", pre_existing_env="export STEERLAB_HAND_EDITED=1\n",
        extra=("--force",))
    assert forced.returncode == 0, forced.stderr
    assert forced_file.read_text(encoding="utf-8") == RENDERED_ENV


def test_env_file_sha256_without_a_source_is_a_usage_error():
    """A digest with nothing to hash would silently disable the integrity
    check, so it refuses instead of being ignored."""
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run", "--env-file-sha256", "00" * 32],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 64
    assert "requires --env-file-from" in proc.stderr


def test_dry_run_plans_the_install_and_keeps_the_report_contract(tmp_path):
    """The reviewable half: the plan says which file will be installed and
    which digest it must hash to — and the machine-readable report keeps
    exactly the shape every existing caller parses."""
    source = tmp_path / "rendered.env"
    source.write_text(RENDERED_ENV, encoding="utf-8")
    digest = hashlib.sha256(RENDERED_ENV.encode("utf-8")).hexdigest()
    proc = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run",
         "--env-file", str(tmp_path / "cluster.env"),
         "--env-file-from", str(source), "--env-file-sha256", digest],
        text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert f"install {tmp_path / 'cluster.env'} from {source}" in proc.stdout
    assert f"sha256 must equal {digest}" in proc.stdout
    # The plan must NOT advertise the heredoc's inputs — they are not consulted.
    assert "partition=gpu_p gres=" not in proc.stdout

    report = _report(proc)
    assert report["ok"] is True
    assert set(report["steps"]) == {"condaDetect", "envCreate", "manifestCheck",
                                    "torchInstall", "serverInstall", "jlensStage",
                                    "envFile", "profileValidate", "helloJob"}
    assert report["steps"]["envFile"] == "planned"
    assert report["envFile"] == str(tmp_path / "cluster.env")
    assert not (tmp_path / "cluster.env").exists()


# --- WP5 step 7: the constants are the FALLBACK, not the site's facts ------

#: Every value `bootstrap.sh`'s heredoc hardcodes that describes ONE site
#: rather than the product (audit a1-a7, c9, c13, c39, c44). A rendered
#: environment for some other site must contain none of them.
SCRIPT_FALLBACK_VALUES = (
    'export STEERLAB_SLURM_MEMORY="80G"',
    'export STEERLAB_SLURM_GPU_TYPES="L4,A100,H100"',
    'export STEERLAB_SLURM_GPU_VRAM="L4:24,A100:80,H100:80"',
    "export STEERLAB_SLURM_REQUEUE=1",
    "export STEERLAB_PURGE_DAYS=30",
    "export STEERLAB_PURGE_WARN_DAYS=20",
    "export STEERLAB_NODE_STAGE_DIR='/lscratch/$SLURM_JOB_ID'",
    'export STEERLAB_SLURM_PARTITION="gpu_p"',
    'export STEERLAB_SLURM_GRES="gpu:A100:1"',
)


def test_a_rendered_generic_site_installs_with_no_fallback_constant(tmp_path):
    """**The step-7 gate.** Render the committed generic (v2-neutral) profile
    with this engine's renderer, hand it to bootstrap.sh, and the file the
    cluster will source contains not one of the script's own site constants —
    the GPU vocabulary, the 80G job memory, the 30/20-day purge window, the
    node-staging template, the gpu_p partition. A generic site that declares
    none of them gets none of them, instead of silently inheriting one
    institution's shape through the heredoc."""
    from steerlab_server.api import site_environment
    from steerlab_server.api.site_profile import ClusterSiteProfile

    fixture = (pathlib.Path(__file__).resolve().parent.parent.parent
               / "prompts" / "fixtures" / "cluster-site-profile" / "v2-neutral.json")
    profile = ClusterSiteProfile.decode_json(fixture.read_text(encoding="utf-8"))
    rendered = site_environment.render_env_file(profile)
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    proc, env_file, _ = _materialized_run(
        tmp_path, source_text=rendered, sha256=digest)
    assert proc.returncode == 0, proc.stderr
    assert _report(proc)["steps"]["envFile"] == "ok"
    installed = env_file.read_text(encoding="utf-8")
    assert installed == rendered
    for constant in SCRIPT_FALLBACK_VALUES:
        assert constant not in installed, constant
    # …and the file still says what a cluster env file must say.
    assert "export STEERLAB_SERVER_PROFILE=cluster" in installed
    assert 'export STEERLAB_AUTH_TOKEN="$(cat "$HOME/.steerlab-token")"' in installed


def test_the_fallback_heredoc_keeps_its_values_but_says_they_are_fallbacks(tmp_path):
    """The other half of step 7: the constants were demoted, not deleted. A
    hand-run bootstrap.sh — no profile, no rendered file — still converges to
    exactly the environment it always wrote, so the manual path is unbroken.
    What changed is that the file, the console, and the plan all now say those
    values are this script's defaults rather than the site's declared facts, so
    nobody reads a fallback as a statement about the cluster."""
    proc, env_file, _ = _materialized_run(tmp_path, materialize=False)
    assert proc.returncode == 0, proc.stderr
    assert _report(proc)["steps"]["envFile"] == "ok"
    written = env_file.read_text(encoding="utf-8")
    for constant in SCRIPT_FALLBACK_VALUES:
        if "PARTITION" in constant or "GRES" in constant:
            continue  # supplied by flags in this run, not by the heredoc
        assert constant in written, constant
    assert 'export HF_HUB_OFFLINE=1' in written
    assert 'export STEERLAB_METADATA_ROOT="$HOME/.steerlab"' in written

    assert "FALLBACK" in written
    assert "--env-file-from" in written
    assert "BUILT-IN FALLBACK values" in proc.stdout

    plan = subprocess.run(
        [bash, BOOTSTRAP, "--dry-run", "--env-file", str(tmp_path / "planned.env")],
        text=True, capture_output=True, check=False)
    assert plan.returncode == 0, plan.stderr
    assert "BUILT-IN FALLBACK values" in plan.stdout
    # The plan-transcript contract the Swift consumers parse is untouched.
    assert _report(plan)["steps"]["envFile"] == "planned"


# --- WP5 step 11: the pre-stage free-space floor (audit c47) ------------------


def _stage_jlens_source() -> str:
    """The REAL `stage_jlens` body, lifted from the script by its own function
    header. The step is unreachable end-to-end without a conda install, so the
    function is exercised directly rather than re-implemented here — an
    anchored extraction fails loudly if the function is renamed, where a grep
    would quietly keep passing."""
    text = pathlib.Path(BOOTSTRAP).read_text(encoding="utf-8")
    start = text.index("stage_jlens() {")
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


def _run_stage_jlens(tmp_path, *, declared_gb=None, rendered=True,
                     free_floor_gb=None):
    """Drive `stage_jlens` with the surrounding script stubbed to the state it
    would be in at step 4b: a successful server install, a python that is a
    no-op, and either a pushed render or nothing at all."""
    cache = tmp_path / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    rendered_line = ""
    ambient = ""
    if declared_gb is not None and rendered:
        rendered_line = (f'  [ "$1" = STEERLAB_PRESTAGE_MIN_FREE_GB ] '
                         f'&& printf "%s" {declared_gb}\n')
    elif declared_gb is not None:
        ambient = f"export STEERLAB_PRESTAGE_MIN_FREE_GB={declared_gb}\n"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        f"{ambient}"
        f'HF_CACHE="{cache}"\n'
        'PYBIN=true\nREPO=/nowhere\n'
        'JLENS_MODELS="acme/one"\n'
        f'JLENS_MIN_FREE_KB={int((free_floor_gb or 8) * 1048576)}\n'
        'get_step() { printf ok; }\n'
        'set_step() { printf "SET %s %s\\n" "$1" "$2"; }\n'
        'fail_step() { printf "FAIL %s %s\\n" "$1" "$2"; }\n'
        'env_file_export_value() {\n'
        f"{rendered_line}"
        '  return 0\n'
        '}\n'
        + _stage_jlens_source()
        + "\nstage_jlens\n",
        encoding="utf-8")
    return subprocess.run([bash, str(harness)], text=True, capture_output=True,
                          check=False, timeout=60)


def test_prestage_floor_comes_from_the_render_when_the_site_declares_one(tmp_path):
    """Audit c47. The ~8 GiB floor is sized from SteerLab's own lens bytes, so
    it stays the script's fallback; what a site knows that the script cannot —
    a shared group quota, a thin node-local cache — arrives as
    STEERLAB_PRESTAGE_MIN_FREE_GB and wins. An absurd declared floor must
    refuse on a filesystem the built-in floor would have waved through."""
    proc = _run_stage_jlens(tmp_path / "a", declared_gb=10_000_000)
    assert "FAIL jlensStage" in proc.stdout
    assert "constraints.storage.prestageMinFreeGB" in proc.stdout
    assert "from the pushed render" in proc.stdout
    # A declared refusal speaks generically — it names the SITE's key, never
    # some other institution's filesystem.
    for identifier in ("sapelo", "gacrc", "uga.edu"):
        assert identifier not in proc.stdout.lower(), identifier

    # A floor the filesystem clears lets the step proceed to acquisition.
    ok = _run_stage_jlens(tmp_path / "b", declared_gb=1)
    assert "SET jlensStage ok" in ok.stdout


def test_prestage_floor_falls_back_to_the_scripts_own_and_says_so(tmp_path):
    """No render, no sourced site env — a hand run. The script applies the rule
    it has always applied, with the message it has always used — generic
    wording since WP5 step 12, which took the institutional names out of the
    shipped script while leaving the fallback rule itself untouched."""
    proc = _run_stage_jlens(tmp_path, free_floor_gb=10_000_000)
    assert "FAIL jlensStage" in proc.stdout
    assert "GROUP quota" in proc.stdout
    assert "constraints.storage.prestageMinFreeGB" not in proc.stdout
    for identifier in ("sapelo", "gacrc", "uga.edu"):
        assert identifier not in proc.stdout.lower(), identifier


def test_prestage_floor_reads_a_sourced_site_environment_too(tmp_path):
    """The second link of the same chain the login-node guard uses: a hand run
    with the installed site env already sourced still gets the site's floor."""
    proc = _run_stage_jlens(tmp_path, declared_gb=10_000_000, rendered=False)
    assert "FAIL jlensStage" in proc.stdout
    assert "from the sourced site environment" in proc.stdout


# --- WP5 step 12: the module list is site data (audit c26) ---------------------


def _module_plan(tmp_path, *, rendered=None, sourced=None, flag=None):
    """The dry-run plan's condaDetect line, which names the resolved module
    list and where it came from. Dry run so nothing loads anything."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    argv = [bash, BOOTSTRAP, "--dry-run", "--env-file", str(tmp_path / "cluster.env")]
    env = dict(os.environ)
    env.pop("STEERLAB_MODULES", None)
    if rendered is not None:
        source = tmp_path / "rendered.env"
        source.write_text(
            f'export STEERLAB_MODULES="{rendered}"\n', encoding="utf-8")
        argv += ["--env-file-from", str(source)]
    if sourced is not None:
        env["STEERLAB_MODULES"] = sourced
    if flag is not None:
        argv += ["--modules", flag]
    proc = subprocess.run(argv, text=True, capture_output=True, check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    return next(line for line in proc.stdout.splitlines() if "[condaDetect]" in line)


def test_module_list_resolves_from_the_site_before_the_scripts_fallback(tmp_path):
    """Audit c26. ``module load Miniforge3`` was one institution's way of
    providing conda, hardcoded for every site. It is now
    ``environment.modules``, resolved in the same four-step order as the
    login-node guard: an explicit flag, the pushed render, the sourced site
    environment, then the script's own fallback — which is labelled as one."""
    declared = _module_plan(tmp_path / "a", rendered="Python/3.12 CUDA/12.8")
    assert "module load Python/3.12 CUDA/12.8" in declared
    assert "the pushed render" in declared

    sourced = _module_plan(tmp_path / "b", sourced="SitePython")
    assert "module load SitePython" in sourced
    assert "the sourced site environment" in sourced

    # An explicit flag beats both, and an empty one means "load nothing" —
    # a site whose Python needs no module must be able to say so.
    flagged = _module_plan(tmp_path / "c", rendered="Ignored", flag="MyModule")
    assert "module load MyModule" in flagged and "Ignored" not in flagged
    assert _module_plan(tmp_path / "d", flag="") .endswith("no modules declared || use conda/mamba on PATH")

    fallback = _module_plan(tmp_path / "e")
    assert "module load Miniforge3" in fallback
    assert "(fallback)" in fallback


# --- controller pre-expiry successor chain (open-issues §1) -------------------
#
# The handler is bash inside the sbatch template, so "engine-pure" here means
# the real shell code with FAKE scheduler binaries under it: the same
# `tests/fakebin` doubles `test_fake_scheduler.py`'s `fake_slurm` fixture uses,
# driven by the same env protocol (FAKE_SLURM_LOG / FAKE_SBATCH_FAIL, plus the
# FAKE_SQUEUE_NAMED name-lookup this chain needs). No Slurm, no timing, no
# background server: the chain region is sliced out of the template and called
# synchronously, so every assertion is deterministic.

FAKEBIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")

#: The banner that opens the chain region and the trap line that closes it.
#: `test_controller_template_documents_the_contract` pins both, so renaming an
#: anchor fails loudly instead of silently slicing an empty library.
CHAIN_START = "# --- pre-expiry successor chain"
CHAIN_END = "trap chain_successor USR1"


def _chain_library(tmp_path):
    """The template's chain region, extracted verbatim into a sourceable file."""
    lines = open(CONTROLLER, encoding="utf-8").read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(CHAIN_START))
    end = next(i for i, line in enumerate(lines) if line.startswith(CHAIN_END))
    assert start < end
    path = tmp_path / "chain-region.sh"
    path.write_text("\n".join(lines[start:end]) + "\n", encoding="utf-8")
    return path


def _fakebin(tmp_path):
    """Copy the committed doubles and re-assert the exec bit (iCloud sync can
    drop it) — `fake_slurm`'s own preamble, without the pytest fixture."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run_chain(tmp_path, *, calls=1, metadata=None, job_id="47463858",
               job_name="steerlab-serverd", env=None):
    """Source the chain region and call ``chain_successor`` ``calls`` times."""
    metadata = metadata or (tmp_path / "meta")
    metadata.mkdir(exist_ok=True)
    log_dir = tmp_path / "calls"
    log_dir.mkdir(exist_ok=True)
    harness = tmp_path / "harness.sh"
    body = [f'. "{_chain_library(tmp_path)}"'] + ["chain_successor"] * calls
    harness.write_text("\n".join(body) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [bash, str(harness)], text=True, capture_output=True, check=False,
        env={
            **os.environ,
            "PATH": str(_fakebin(tmp_path)) + os.pathsep + os.environ.get("PATH", ""),
            "FAKE_SLURM_LOG": str(log_dir),
            "FAKE_SLURM_JOB_ID": "47999999",
            "STEERLAB_METADATA_ROOT": str(metadata),
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_NAME": job_name,
            **(env or {}),
        })
    submissions = []
    call_log = log_dir / "sbatch.calls"
    if call_log.exists():
        submissions = [line for line in call_log.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
    marker = metadata / "serverd.chain.json"
    return proc, submissions, marker


def test_controller_chain_submits_one_successor_and_stamps_lineage(tmp_path):
    """§1's whole point: a pre-expiry USR1 must produce a successor job, and
    the successor must be traceable to the job that bore it."""
    proc, submissions, marker = _run_chain(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert len(submissions) == 1, submissions
    # --export=NONE is deliberate: the running allocation's SLURM_* variables
    # outrank the successor's own #SBATCH directives.
    assert "--export=NONE" in submissions[0]
    assert submissions[0].endswith("controller-job.sbatch") or "harness.sh" in submissions[0]
    lineage = json.loads(marker.read_text(encoding="utf-8"))
    assert lineage["predecessor"] == "47463858"
    assert lineage["successor"] == "47999999"
    assert lineage["chainIndex"] == 1
    assert "successor queued: job 47999999" in proc.stdout


def test_controller_chain_is_idempotent_within_one_job(tmp_path):
    """Slurm can deliver the warning more than once, and the operator can send
    one by hand; the second call must be a no-op, not a second job."""
    proc, submissions, _ = _run_chain(tmp_path, calls=3)
    assert len(submissions) == 1, submissions
    assert proc.stdout.count("successor already submitted for job 47463858 — no-op") == 2


def test_controller_chain_no_ops_when_a_sibling_controller_is_already_queued(tmp_path):
    """A legacy `afterany` successor, or an operator-started controller, means
    a successor exists already — the guard asks squeue by job name."""
    proc, submissions, marker = _run_chain(
        tmp_path, env={"FAKE_SQUEUE_NAMED": "47463858\n48000001"})
    assert submissions == []
    assert not marker.exists()
    assert "already queued or running — no successor" in proc.stdout


def test_controller_chain_ignores_only_itself_in_the_squeue_answer(tmp_path):
    """Our own job is always in that answer (we are RUNNING); it must not read
    as "a successor already exists" or the chain would never fire at all."""
    _, submissions, _ = _run_chain(tmp_path, env={"FAKE_SQUEUE_NAMED": "47463858"})
    assert len(submissions) == 1


def test_controller_chain_respects_the_operator_veto_marker(tmp_path):
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "serverd.no-chain").write_text("", encoding="utf-8")
    proc, submissions, marker = _run_chain(tmp_path, metadata=metadata)
    assert submissions == []
    assert not marker.exists()
    assert "vetoed by" in proc.stdout


def test_controller_chain_respects_the_billed_allocation_toggle(tmp_path):
    proc, submissions, _ = _run_chain(
        tmp_path, env={"STEERLAB_CONTROLLER_RESUBMIT": "0"})
    assert submissions == []
    assert "chaining disabled" in proc.stdout


def test_controller_chain_defaults_to_on(tmp_path):
    """The historical toggle defaulted OFF and nothing ever set it — which is
    exactly why job 47463858 died with no successor. Absent env means ON."""
    proc, submissions, _ = _run_chain(tmp_path)
    assert len(submissions) == 1
    assert "chaining disabled" not in proc.stdout


def test_controller_chain_failure_is_loud_and_leaves_no_marker(tmp_path):
    """A site's queued-job cap is the expected failure. It must not look like
    success, and it must not poison the marker — the interim ops rule (cycle
    serverd by hand) is what applies next."""
    proc, submissions, marker = _run_chain(
        tmp_path, env={"FAKE_SBATCH_FAIL": "sbatch: error: QOSMaxSubmitJobPerUserLimit"})
    assert proc.returncode == 0, "a failed chain must not take the daemon down"
    assert not marker.exists()
    assert "WARNING: could not queue successor" in proc.stderr
    assert "QOSMaxSubmitJobPerUserLimit" in proc.stderr
    assert "steerlab-cli cluster controller start" in proc.stderr


def test_controller_chain_index_continues_the_lineage(tmp_path):
    """A generation that finds itself named as the previous marker's successor
    adopts that chain index, so the lineage counts generations rather than
    resetting to 1 every 24 h."""
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "serverd.chain.json").write_text(
        json.dumps({"predecessor": "47000000", "successor": "47463858",
                    "chainIndex": 3}), encoding="utf-8")
    proc, submissions, marker = _run_chain(tmp_path, metadata=metadata)
    assert len(submissions) == 1
    assert json.loads(marker.read_text(encoding="utf-8"))["chainIndex"] == 4
    assert "chained successor at chain index 3" in proc.stdout


def test_controller_template_traps_only_usr1_and_waits_on_the_server(tmp_path):
    """The two structural facts the chain depends on: TERM (what `scancel`
    sends) is NOT trapped, so a deliberate cancel ends the chain; and the
    server is a background child rather than an `exec`, because an exec'd
    process replaces the shell and no trap can fire."""
    text = open(CONTROLLER, encoding="utf-8").read()
    assert "\ntrap chain_successor USR1\n" in text
    assert "trap chain_successor USR1 TERM" not in text
    assert 'exec "@PYTHON@"' not in text
    assert '"@PYTHON@" -m steerlab_server.cli serve --port "$PORT" &' in text
    assert 'wait "$STEERLAB_SERVER_PID"' in text
    # The mechanism that never fired is gone, not merely disabled.
    assert "--dependency=afterany" not in text.split("HISTORY.")[1].split("NOW.")[1]


# --- node-local scratch gres (cluster-operator requirement, 2026-08-19) ------

def test_scratch_gres_flag_adds_the_line_to_the_synthesized_env(tmp_path):
    """``--scratch-gres`` is how a hand run states what the profile-driven path
    states as ``constraints.storage.nodeScratchGres``: node-local scratch
    requested as its own Slurm gres token."""
    proc, env_file, _ = _materialized_run(
        tmp_path, materialize=False, extra=["--scratch-gres", "lscratch:100"])
    assert proc.returncode == 0, proc.stderr
    installed = env_file.read_text(encoding="utf-8")
    assert 'export STEERLAB_SLURM_SCRATCH_GRES="lscratch:100"' in installed


def test_scratch_gres_is_absent_by_default(tmp_path):
    """Declare-or-omit: without the flag the synthesized file gains no line and
    nothing about an existing hand-bootstrapped site changes."""
    proc, env_file, _ = _materialized_run(tmp_path, materialize=False)
    assert proc.returncode == 0, proc.stderr
    assert "STEERLAB_SLURM_SCRATCH_GRES" not in env_file.read_text(encoding="utf-8")
