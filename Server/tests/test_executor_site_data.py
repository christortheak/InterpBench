"""Executor site-data generalization (TURNKEY-CLUSTER-PLAN WS1) + requeue.

The GPU-type vocabulary, per-type VRAM table, account, and requeue policy are
SITE data on ``SlurmResources`` (env-driven) —
the concrete-type-required RULE is unchanged. Plus the checkpoint-aware state
mapping and the child-status collection in the rendered sbatch script.
"""

import os
import shutil
import subprocess

import pytest

from steerlab_server.api.executors import (
    SchedulerCommands, SlurmExecutor, SlurmResources, map_slurm_state,
    render_slurm_script, scheduler_commands, scheduler_poll_commands)
from steerlab_server.experiment.resume import CHECKPOINT_EXIT_CODE


def _clear_site_env(monkeypatch):
    for name in ("STEERLAB_SLURM_GPU_TYPES", "STEERLAB_SLURM_GPU_VRAM",
                 "STEERLAB_SLURM_ACCOUNT", "STEERLAB_SLURM_REQUEUE",
                 "STEERLAB_SLURM_SACCT", "STEERLAB_SLURM_SQUEUE",
                 "STEERLAB_SLURM_SBATCH", "STEERLAB_SLURM_SCANCEL",
                 "STEERLAB_SLURM_QOS", "STEERLAB_SLURM_CONSTRAINT",
                 "STEERLAB_SLURM_RESERVATION", "STEERLAB_SLURM_EXTRA_SBATCH",
                 "STEERLAB_SLURM_REQUIRED_HEADERS",
                 "STEERLAB_SLURM_CPUS_PER_TASK", "STEERLAB_SLURM_SIGNAL_SECONDS",
                 "STEERLAB_SLURM_SIGNAL_TARGET", "STEERLAB_SLURM_EXPORT_MODE",
                 "STEERLAB_SLURM_SCRATCH_GRES"):
        monkeypatch.delenv(name, raising=False)


# --- GPU-type vocabulary -----------------------------------------------------------

def test_undeclared_vocabulary_refuses_rather_than_assuming_one(monkeypatch):
    """WP5 Step 8 (audit G4/a5): the built-in ``L4,A100,H100`` is GONE. Having
    one institution's inventory as the code default is how two "fallbacks" for
    the same cluster came to disagree (bootstrap.sh excluded P100, this file
    included it), and how a request could be validated against hardware the
    site does not have. With nothing declared there is nothing to validate
    against, so a typed gres is refused — loudly, naming the remedy."""
    _clear_site_env(monkeypatch)
    resources = SlurmResources.from_env()
    assert resources.gpu_types == []
    with pytest.raises(ValueError, match="declares no GPU vocabulary"):
        SlurmResources(gres="A100").normalized_gres()
    with pytest.raises(ValueError, match="STEERLAB_SLURM_GPU_TYPES"):
        SlurmResources(gres="gpu:A100:1").normalized_gres()
    # A job that asks for no GPU is unaffected: there is nothing to check.
    assert SlurmResources(gres=None).normalized_gres() is None


def test_a_directly_constructed_resource_inherits_the_declared_vocabulary(monkeypatch):
    """``/api/slurm/bundle``, the OptVec campaign and job rehydration build
    resources field-by-field rather than through ``from_env``; the site's
    vocabulary must reach them too, or they would each need their own copy of
    the site data (which is how the default became load-bearing)."""
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "V100,MI300X")
    assert SlurmResources(gres="V100").normalized_gres() == "gpu:V100:1"
    # An EXPLICIT vocabulary still wins over the site's (the per-request path).
    assert SlurmResources(
        gres="A100", gpu_types=["A100"]).normalized_gres() == "gpu:A100:1"


def test_site_vocabulary_overrides_via_env(monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "V100, MI300X")
    resources = SlurmResources.from_env()
    assert resources.gpu_types == ["V100", "MI300X"]
    resources.gres = "V100"
    assert resources.normalized_gres() == "gpu:V100:1"
    resources.gres = "gpu:MI300X:2"
    assert resources.normalized_gres() == "gpu:MI300X:2"
    # The legacy default names are NOT valid at this site; the message names the
    # site's vocabulary, not a hardcoded one.
    resources.gres = "A100"
    with pytest.raises(ValueError, match="V100, MI300X"):
        resources.normalized_gres()
    resources.gres = "gpu:A100:1"
    with pytest.raises(ValueError, match="V100, MI300X"):
        resources.normalized_gres()


def test_abstract_gpu_still_refused_under_any_vocabulary(monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "V100")
    resources = SlurmResources.from_env()
    resources.gres = "gpu:1"
    with pytest.raises(ValueError, match="concrete GPU type"):
        resources.normalized_gres()


# --- VRAM table ---------------------------------------------------------------------

def test_gpu_vram_parses_to_typed_table(monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_VRAM", "A100:80, H100:80,L4:24,P100:16")
    resources = SlurmResources.from_env()
    assert resources.gpu_vram_gb == {"A100": 80, "H100": 80, "L4": 24, "P100": 16}


def test_gpu_vram_defaults_empty_and_rejects_malformed(monkeypatch):
    _clear_site_env(monkeypatch)
    assert SlurmResources.from_env().gpu_vram_gb == {}
    monkeypatch.setenv("STEERLAB_SLURM_GPU_VRAM", "A100=80")
    with pytest.raises(ValueError, match="STEERLAB_SLURM_GPU_VRAM"):
        SlurmResources.from_env()
    monkeypatch.setenv("STEERLAB_SLURM_GPU_VRAM", "A100:eighty")
    with pytest.raises(ValueError, match="TYPE:GB"):
        SlurmResources.from_env()


# --- account + requeue ----------------------------------------------------------------

def _script(tmp_path, resources):
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "bundle"), ["python", "-c", "pass"], resources=resources)
    with open(bundle.script_path, encoding="utf-8") as handle:
        return handle.read()


def test_account_env_reaches_sbatch_directive(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_ACCOUNT", "example-lab")
    resources = SlurmResources.from_env()
    assert resources.account == "example-lab"
    assert "#SBATCH --account=example-lab" in _script(tmp_path, resources)


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False),
])
def test_requeue_env_truthiness(monkeypatch, value, expected):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_REQUEUE", value)
    assert SlurmResources.from_env().requeue is expected


def test_requeue_directive_rendered_only_when_set(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    with_requeue = _script(tmp_path / "a", SlurmResources(requeue=True))
    assert "#SBATCH --requeue" in with_requeue
    without = _script(tmp_path / "b", SlurmResources())
    assert "#SBATCH --requeue" not in without


def test_from_env_defaults_when_unset(monkeypatch):
    _clear_site_env(monkeypatch)
    resources = SlurmResources.from_env()
    assert resources.account is None
    assert resources.requeue is False


# --- required sbatch headers ----------------------------------------------------------

def test_ntasks_header_always_rendered(tmp_path, monkeypatch):
    # --ntasks is a required header at some sites; every SteerLab
    # job is a single task, so the constant 1 is always emitted.
    _clear_site_env(monkeypatch)
    assert "#SBATCH --ntasks=1" in _script(tmp_path, SlurmResources())


# --- env-export ordering (the --export=NONE silent-no-op bug) -------------------------

def test_env_exports_precede_the_module_reconstruction_block(tmp_path, monkeypatch):
    """Under ``#SBATCH --export=NONE`` the job starts with a bare environment;
    the module/conda reconstruction block reads STEERLAB_MODULES & friends, so
    the bundle.env exports MUST come first or activation silently no-ops."""
    _clear_site_env(monkeypatch)
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "bundle"), ["python", "-c", "pass"],
        env={"STEERLAB_MODULES": "CUDA/12.8.0",
             "STEERLAB_CONDA_ENV": "steerlab"},
        resources=SlurmResources())
    with open(bundle.script_path, encoding="utf-8") as handle:
        script = handle.read()
    export_at = script.index("export STEERLAB_MODULES=")
    module_block_at = script.index('if [ -n "${STEERLAB_MODULES:-}" ]; then')
    conda_export_at = script.index("export STEERLAB_CONDA_ENV=")
    conda_block_at = script.index('if [ -n "${STEERLAB_CONDA_ENV:-}" ]')
    assert export_at < module_block_at
    assert conda_export_at < conda_block_at


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_rendered_script_actually_loads_modules_under_bare_env(tmp_path, monkeypatch):
    """Real-bash proof of the ordering fix: run the rendered script with a
    bare environment (as --export=NONE would) and a fake ``module`` binary —
    the module load must actually fire with the bundle.env value."""
    _clear_site_env(monkeypatch)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    module_log = tmp_path / "module.calls"
    fake_module = bindir / "module"
    fake_module.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$@\" >> {module_log}\n", encoding="utf-8")
    fake_module.chmod(0o755)
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "bundle"), ["true"],
        env={"STEERLAB_MODULES": "CUDA/12.8.0 Miniforge3"},
        resources=SlurmResources(use_srun=False, signal_seconds=0))
    env = {k: v for k, v in os.environ.items() if not k.startswith("STEERLAB_")}
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(["bash", bundle.script_path],
                          text=True, capture_output=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    calls = module_log.read_text(encoding="utf-8").splitlines()
    assert calls == ["load CUDA/12.8.0", "load Miniforge3"]


# --- site-configurable poll commands ---------------------------------------------------

def test_poll_commands_default_and_env_override(monkeypatch):
    _clear_site_env(monkeypatch)
    assert scheduler_poll_commands() == ("sacct", "squeue")
    resources = SlurmResources.from_env()
    assert resources.sacct_command == "sacct"
    assert resources.squeue_command == "squeue"
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "sacct-site")
    monkeypatch.setenv("STEERLAB_SLURM_SQUEUE", "sq")
    assert scheduler_poll_commands() == ("sacct-site", "sq")
    resources = SlurmResources.from_env()
    assert resources.sacct_command == "sacct-site"
    assert resources.squeue_command == "sq"
    # Whitespace-only values fall back to the raw defaults.
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "  ")
    monkeypatch.setenv("STEERLAB_SLURM_SQUEUE", "")
    assert scheduler_poll_commands() == ("sacct", "squeue")


def test_all_four_scheduler_binaries_are_site_data(monkeypatch):
    """WP5 G5 / audit c8: ``sbatch`` and ``scancel`` were literals in the
    engine, so a site that wraps them could declare it in its profile and be
    ignored. All four now come from the same rendered keys."""
    _clear_site_env(monkeypatch)
    assert scheduler_commands() == SchedulerCommands()
    monkeypatch.setenv("STEERLAB_SLURM_SBATCH", "sbatch-wrap")
    monkeypatch.setenv("STEERLAB_SLURM_SCANCEL", "scancel-wrap")
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "sacct-wrap")
    monkeypatch.setenv("STEERLAB_SLURM_SQUEUE", "sq")
    commands = scheduler_commands()
    assert (commands.submit, commands.query, commands.accounting, commands.cancel) == (
        "sbatch-wrap", "sq", "sacct-wrap", "scancel-wrap")
    resources = SlurmResources.from_env()
    assert resources.sbatch_command == "sbatch-wrap"
    assert resources.scancel_command == "scancel-wrap"


# --- placement directives the site can finally state (audit c1-c5) --------------------

def test_placement_directives_reach_the_rendered_script(tmp_path, monkeypatch):
    """QOS, node features, reservation and site-wide extra directives were
    reachable only as per-request ``extraSbatch`` strings before WP5 Step 8 —
    a site could not state them at all. The keys are exactly the ones the
    site-profile renderer emits."""
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_QOS", "gpu_qos")
    monkeypatch.setenv("STEERLAB_SLURM_CONSTRAINT", "hasgpu&ib")
    monkeypatch.setenv("STEERLAB_SLURM_RESERVATION", "maint-window")
    monkeypatch.setenv("STEERLAB_SLURM_EXTRA_SBATCH", "--exclusive --nice=100")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100")
    resources = SlurmResources.from_env()
    assert resources.qos == "gpu_qos"
    assert resources.constraints == ["hasgpu", "ib"]
    assert resources.reservation == "maint-window"
    assert resources.extra_sbatch == ["--exclusive", "--nice=100"]
    script = _script(tmp_path, resources)
    assert "#SBATCH --qos=gpu_qos" in script
    assert "#SBATCH --constraint=hasgpu&ib" in script
    assert "#SBATCH --reservation=maint-window" in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --nice=100" in script


def test_cpus_signal_and_export_mode_are_site_data(tmp_path, monkeypatch):
    """audit c11, c12, c15. The signal TARGET travels with the lead: without
    it the engine would render ``USR1@N`` for a site whose own preview
    promises ``B:USR1@N``."""
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_CPUS_PER_TASK", "12")
    monkeypatch.setenv("STEERLAB_SLURM_SIGNAL_SECONDS", "300")
    monkeypatch.setenv("STEERLAB_SLURM_SIGNAL_TARGET", "batch-forward")
    monkeypatch.setenv("STEERLAB_SLURM_EXPORT_MODE", "all")
    resources = SlurmResources.from_env()
    assert resources.cpus_per_task == 12
    assert resources.export_none is False
    script = _script(tmp_path, resources)
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --signal=B:USR1@300" in script
    assert "#SBATCH --export=NONE" not in script
    # The SLURM_EXPORT_ENV=ALL fix stays regardless of the header: it is about
    # srun stripping the script's own environment, not about --export.
    assert "export SLURM_EXPORT_ENV=ALL" in script


def test_malformed_integer_site_data_fails_loudly(monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_CPUS_PER_TASK", "four")
    with pytest.raises(ValueError, match="STEERLAB_SLURM_CPUS_PER_TASK"):
        SlurmResources.from_env()


# --- declared required headers (audit c4) --------------------------------------------

def test_declared_required_header_without_a_value_refuses_to_render(
        tmp_path, monkeypatch):
    """A site whose sbatch rejects jobs without ``--mem`` should hear about it
    here, not from the scheduler after a queue wait. ``--ntasks`` is always
    satisfied: constant 1 is SteerLab's own shape, not site data."""
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_REQUIRED_HEADERS", "partition,mem,ntasks")
    resources = SlurmResources.from_env()
    assert resources.required_headers == ["partition", "mem", "ntasks"]
    with pytest.raises(ValueError, match="required sbatch headers"):
        _script(tmp_path / "missing", resources)
    resources.partition = "gpu_p"
    resources.memory = "80G"
    script = _script(tmp_path / "complete", resources)
    assert "#SBATCH --partition=gpu_p" in script
    assert "#SBATCH --mem=80G" in script
    assert "#SBATCH --ntasks=1" in script


def test_unknown_required_header_token_is_reported(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_REQUIRED_HEADERS", "partition,nodes")
    resources = SlurmResources.from_env()
    resources.partition = "gpu_p"
    with pytest.raises(ValueError, match="nodes \\(unknown"):
        _script(tmp_path, resources)


# --- the rendered script's signal/status plumbing -----------------------------------

def test_script_collects_child_status_and_exits_with_it(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    script = _script(tmp_path, SlurmResources(signal_seconds=600))
    # The trap only forwards — it must NOT reap the child, or the main wait
    # loop could never recover the real exit status (85 vs 128+sig).
    trap_block = script.split("trap checkpoint USR1 TERM")[0]
    assert 'kill -USR1 "${STEERLAB_CHILD_PID}"' in trap_block
    assert 'wait "${STEERLAB_CHILD_PID}" || true' not in trap_block
    # The main loop re-waits through signal interruptions and propagates the
    # child's own status as the job's exit code.
    assert 'STEERLAB_CHILD_STATUS=$?' in script
    assert 'while [ "${STEERLAB_CHILD_STATUS}" -ge 128 ]' in script
    assert 'exit "${STEERLAB_CHILD_STATUS}"' in script


# --- checkpoint-aware state mapping ---------------------------------------------------

def test_map_slurm_state_checkpoint_exit_cases():
    assert CHECKPOINT_EXIT_CODE == 85  # the fixed cross-wave contract
    assert map_slurm_state("FAILED", exit_code="85:0") == "checkpointed"
    assert map_slurm_state("FAILED", exit_code="85:15") == "checkpointed"
    assert map_slurm_state("FAILED", exit_code="1:0") == "failed"
    assert map_slurm_state("FAILED") == "failed"
    # Only a FAILED job carries the batch script's exit code; TIMEOUT stays a
    # failure even if sacct decorates it oddly.
    assert map_slurm_state("TIMEOUT", exit_code="85:0") == "failed"


def test_map_slurm_state_requeue_class_states_are_alive():
    assert map_slurm_state("REQUEUED") == "submitted"
    assert map_slurm_state("PREEMPTED") == "submitted"
    assert map_slurm_state("RESIZING") == "running"
    assert map_slurm_state("SUSPENDED") == "running"
    # Legacy mappings untouched.
    assert map_slurm_state("PENDING") == "submitted"
    assert map_slurm_state("CANCELLED by 1234") == "cancelled"
    assert map_slurm_state("NODE_FAIL") == "failed"
    assert map_slurm_state("") is None


# --- node-local scratch gres (cluster-operator requirement, 2026-08-19) ------------------

def test_scratch_gres_read_from_env_and_absent_by_default(monkeypatch):
    _clear_site_env(monkeypatch)
    assert SlurmResources.from_env().scratch_gres is None
    assert SlurmResources().normalized_scratch_gres() is None
    monkeypatch.setenv("STEERLAB_SLURM_SCRATCH_GRES", " lscratch:100 ")
    assert SlurmResources.from_env().scratch_gres == "lscratch:100"
    # An empty declaration is "absent", never an empty gres token.
    monkeypatch.setenv("STEERLAB_SLURM_SCRATCH_GRES", "   ")
    assert SlurmResources.from_env().scratch_gres is None


def test_scratch_gres_appends_to_the_gpu_gres_in_the_rendered_header(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100")
    script = _script(tmp_path, SlurmResources(gres="A100", scratch_gres="lscratch:100"))
    assert "#SBATCH --gres=gpu:A100:1,lscratch:100" in script


def test_scratch_gres_renders_alone_when_the_job_asks_for_no_gpu(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    script = _script(tmp_path, SlurmResources(scratch_gres="lscratch:100"))
    assert "#SBATCH --gres=lscratch:100" in script


def test_a_per_request_gpu_gres_override_keeps_the_sites_scratch(tmp_path, monkeypatch):
    """Why ``scratch_gres`` is a SEPARATE field: the GPU-session path replaces
    ``resources.gres`` from the request and would otherwise silently drop the
    site's node-local staging accounting with it."""
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100,H100")
    monkeypatch.setenv("STEERLAB_SLURM_SCRATCH_GRES", "lscratch:100")
    resources = SlurmResources.from_env()
    resources.gres = "H100"  # the per-request override
    assert "#SBATCH --gres=gpu:H100:1,lscratch:100" in _script(tmp_path, resources)


def test_scratch_gres_does_not_weaken_the_gpu_vocabulary_check(monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "H100")
    with pytest.raises(ValueError, match="concrete GPU type"):
        SlurmResources(
            gres="A100", gpu_types=["H100"],
            scratch_gres="lscratch:100").normalized_gres()


@pytest.mark.parametrize("token", [
    "lscratch:100 --mail-user=evil@example.com",   # a smuggled second directive
    "lscratch:100\nexport EVIL=1",                 # a second env line
    "lscratch:$(whoami)",                          # command substitution
    "lscratch:100;rm -rf /",
])
def test_malformed_scratch_gres_refuses_loudly(token):
    with pytest.raises(ValueError, match="not a safe Slurm token"):
        SlurmResources(scratch_gres=token).normalized_scratch_gres()


def test_undeclared_scratch_gres_leaves_the_header_byte_identical(tmp_path, monkeypatch):
    _clear_site_env(monkeypatch)
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100")
    script = _script(tmp_path, SlurmResources(gres="A100"))
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --gres=gpu:A100:1" in directives
    # No comma, no scratch token: a site that declared nothing asks for nothing.
    assert not [line for line in directives if "," in line or "lscratch" in line]
