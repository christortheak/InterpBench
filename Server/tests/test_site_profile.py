"""WP5 step 4 — the Python site-profile schema + renderer, in lockstep with
Swift (``docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md`` §2.0, §3, §5).

The load-bearing tests here are the RENDER-EQUALITY ones: both engines decode
the same committed profile JSON in ``prompts/fixtures/cluster-site-profile/``
and are pinned to the same committed rendered bytes. That byte equality is the
only real guarantee the two schemas have not drifted — a field renamed,
defaulted differently, or emitted in a different order on either side fails
here.

The goldens were produced BY THE SWIFT RENDERER (it landed first, and its
hand-transcribed compatibility proof against ``bootstrap.sh`` / ``executors.py``
is the authority): run the Swift twin with
``TEST_RUNNER_STEERLAB_WRITE_CLUSTER_GOLDENS=1`` to regenerate them, never
this engine.

Swift twin: ``Tests/ExperimentKitTests/ClusterEnvironmentRendererTests.swift``
(rendering) and ``ClusterSiteProfileTests.swift`` (the wire-name key list).
"""

import json
from pathlib import Path

import pytest

from steerlab_server.api import site_environment as env
from steerlab_server.api.site_profile import (
    CURRENT_SCHEMA_VERSION,
    ClusterSiteProfile,
    InterruptionPolicy,
    JobClassResources,
    JobDefaults,
    LoginNodePolicy,
    MaintenancePolicy,
    SchedulerCommands,
    SchedulerLimits,
    SiteEnvironment,
    SitePolicy,
    SiteProfileError,
    SiteStorage,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "fixtures" / "cluster-site-profile"
)

#: The committed cross-engine fixtures, same list as the Swift twin.
FIXTURES = ("v1-maximal", "v2-maximal", "v2-neutral", "example-slurm-site")


def load_fixture(name: str) -> ClusterSiteProfile:
    return ClusterSiteProfile.decode_json(
        (FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def golden(name: str) -> str:
    path = FIXTURE_DIR / name
    assert path.exists(), (
        f"missing committed golden {path} — the cluster renderer is byte-pinned "
        "cross-engine; restore the fixture, never delete it")
    return path.read_text(encoding="utf-8")


def header_document(profile: ClusterSiteProfile) -> str:
    """The ``#SBATCH`` blocks for every job class, in one file. The section
    marker is fixture format, not renderer output — both engines join the
    per-class line lists identically."""
    blocks = [
        "\n".join(
            [f"# --- {job_class} ---"]
            + env.render_scheduler_headers(profile, job_class))
        for job_class in env.JOB_CLASSES
    ]
    return "\n\n".join(blocks) + "\n"


def unresolved_document(profile: ClusterSiteProfile) -> str:
    """``key<TAB>detail``, one per line, in the renderer's declared order."""
    return "\n".join(
        f"{fact.key}\t{fact.detail}" for fact in env.unresolved_facts(profile)) + "\n"


# --- render equality (the point of this step) ---------------------------------


@pytest.mark.parametrize("fixture", FIXTURES)
def test_env_file_matches_the_committed_golden(fixture):
    produced = env.render_env_file(load_fixture(fixture))
    assert produced == golden(f"{fixture}.env.golden.txt"), (
        f"{fixture}.env.golden.txt drifted — this file is the cross-engine "
        "contract (the Swift renderer reads the same bytes). Change the "
        "renderer only deliberately on BOTH engines, then regenerate from Swift.")


@pytest.mark.parametrize("fixture", FIXTURES)
def test_scheduler_headers_match_the_committed_golden(fixture):
    assert header_document(load_fixture(fixture)) == golden(
        f"{fixture}.headers.golden.txt")


@pytest.mark.parametrize("fixture", FIXTURES)
def test_unresolved_facts_match_the_committed_golden(fixture):
    assert unresolved_document(load_fixture(fixture)) == golden(
        f"{fixture}.unresolved.golden.txt")


def test_fixtures_cover_both_default_sets():
    # A v1 file keeps its stamp through decode — that is what selects the legacy
    # default set. Without both branches the equality contract would only ever
    # cover one half of WP5 §2.0 rule 3.
    assert load_fixture("v1-maximal").schema_version == 1
    assert env.default_set(load_fixture("v1-maximal")) == env.LEGACY_V1
    assert env.default_set(load_fixture("v2-maximal")) == env.NEUTRAL_V2
    assert env.default_set(load_fixture("v2-neutral")) == env.NEUTRAL_V2


def test_rendering_is_deterministic_and_timestamp_free():
    for fixture in FIXTURES:
        profile = load_fixture(fixture)
        first = env.render_env_file(profile)
        for _ in range(8):
            assert env.render_env_file(profile) == first
        # Re-decoding the same document must render identically: storage roots
        # and the VRAM table are dicts, and a leaked iteration order would drift.
        assert env.render_env_file(load_fixture(fixture)) == first


def test_rendered_file_carries_no_secret_value():
    # The token is a PATH indirection, never a value — the preview surface shows
    # this text unedited.
    for fixture in FIXTURES:
        text = env.render_env_file(load_fixture(fixture))
        assignment = text.split("STEERLAB_AUTH_TOKEN=", 1)[1].split("\n", 1)[0]
        assert assignment.startswith('"$(cat ')


# --- the wire-name contract ---------------------------------------------------


def test_maximal_fixture_carries_every_declared_key():
    """Mirrors ``ClusterSiteProfileTests.maximalV2ProfileCarriesEveryDeclaredKey``.
    The expected lists are transcribed from that test: a field renamed on either
    engine, or added on one only, fails on both."""
    root = json.loads((FIXTURE_DIR / "v2-maximal.json").read_text(encoding="utf-8"))

    def keys(container, expected, label):
        assert set(container.keys()) == set(expected), f"key drift in {label}"

    keys(
        root,
        [
            "schemaVersion", "name", "notes", "transport", "topology", "scheduler",
            "constraints", "environment", "policy", "bootstrapPath",
        ], "root")

    constraints = root["constraints"]
    keys(
        constraints,
        [
            "computeEgress", "storageRoots", "purgeDays", "purgeWarnDays",
            "maintenanceSource", "storage",
        ], "constraints")
    keys(
        constraints["storage"],
        [
            "nodeStageDirTemplate", "nodeScratchGres", "hubOfflineMode",
            "metadataRequiresLocalFilesystem",
            "scanFileCap", "freeSpaceWarnGB", "freeSpaceFailGB", "calendarStaleDays",
            "quotaCommand", "scannedRoles", "prestageMinFreeGB",
        ], "constraints.storage")

    keys(
        root["environment"],
        [
            "moduleSystem", "moduleInitScript", "modules", "pythonProvider",
            "pythonVersion", "envPrefix", "condaProfileScript", "condaEnvName",
            "venvPath", "pythonExecutable", "torchIndexURL", "torchVariant",
            "serverExtras", "envFilePath", "tokenFilePath", "remoteRepoPath",
            "interactiveAllocationCommand", "transferHost", "sshControlPersist",
        ], "environment")

    policy = root["policy"]
    keys(
        policy,
        [
            "loginNodes", "maintenance", "transferMethod", "externalServiceEgress",
            "bindOverride", "authModeOverride",
        ], "policy")
    keys(
        policy["loginNodes"],
        ["hostnamePatterns", "allowCompute", "requireAllocation"], "policy.loginNodes")
    keys(
        policy["maintenance"],
        ["calendarPath", "sourceURL", "sourceNote"], "policy.maintenance")

    keys(root["scheduler"], ["kind", "slurm"], "scheduler")
    slurm = root["scheduler"]["slurm"]
    keys(
        slurm,
        [
            "partitions", "gpuTypes", "gpuVRAMGB", "defaultGres", "defaultPartition",
            "accountRequired", "account", "billedAllocations", "maxParallelGPUJobs",
            "gpus", "qos", "constraints", "reservation", "extraSbatch",
            "requiredHeaders", "commands", "jobDefaults", "interruption",
            "submitFromBundleDirectory", "jobNamePrefix", "limits",
            "accountingVisibilityGraceSeconds", "controllerJob", "setupJob",
            "gpuSession",
        ], "scheduler.slurm")
    keys(
        slurm["partitions"][0],
        ["name", "maxWalltimeHours", "allowedGPUTypes", "qos"], "partitions[]")
    keys(slurm["gpus"][0], ["name", "vramGB", "computeCapability"], "gpus[]")
    keys(slurm["commands"], ["submit", "query", "accounting", "cancel"], "commands")
    keys(slurm["jobDefaults"], ["memory", "walltime", "cpusPerTask"], "jobDefaults")
    keys(
        slurm["interruption"],
        [
            "requeue", "autoResubmit", "autoResubmitLimit", "signalSeconds",
            "signalTarget", "exportMode",
        ], "interruption")
    keys(
        slurm["limits"],
        ["maxSubmittedJobs", "maxRunningJobs", "maxParallelGPUJobs"], "limits")
    for job_class in ("controllerJob", "setupJob", "gpuSession"):
        keys(
            slurm[job_class],
            [
                "partition", "cpusPerTask", "memory", "walltime", "gres", "port",
                "idleMinutes", "extraSbatch",
            ], job_class)


def test_maximal_fixture_decodes_every_field():
    site = load_fixture("v2-maximal")
    assert site.schema_version == 2
    assert site.name == "Example HPC (v2 maximal)"
    assert site.topology == "daemonInJob"
    assert site.bootstrap_path == "Server/scripts/bootstrap.sh"
    assert site.transport.kind == "ssh"
    assert site.transport.host == "slurm.example.edu"
    assert site.transport.proxy_jump == "bastion.example.edu"
    assert site.transport.remote_port == 9090
    assert site.transport.vpn_expected is True

    slurm = site.slurm
    assert slurm is not None
    assert [p.name for p in slurm.partitions] == ["gpu_p", "batch"]
    assert slurm.partitions[0].allowed_gpu_types == ["A100", "H100"]
    assert slurm.partitions[0].qos == "gpu_qos"
    assert slurm.partitions[1].qos is None
    assert [(g.name, g.vram_gb, g.compute_capability) for g in slurm.gpus] == [
        ("A100", 80, "sm_80"), ("H100", 80, "sm_90")]
    assert slurm.commands == SchedulerCommands(
        submit="sbatch-wrap", query="sq", accounting="sacct-wrap", cancel="scancel-wrap")
    assert slurm.job_defaults == JobDefaults(
        memory="120G", walltime="12:00:00", cpus_per_task=8)
    assert slurm.interruption == InterruptionPolicy(
        requeue=True, auto_resubmit=True, auto_resubmit_limit=3, signal_seconds=300,
        signal_target="batch-forward", export_mode="all")
    assert slurm.limits == SchedulerLimits(
        max_submitted_jobs=20, max_running_jobs=8, max_parallel_gpu_jobs=15)
    assert slurm.submit_from_bundle_directory is False
    assert slurm.job_name_prefix == "steerlab-x"
    assert slurm.accounting_visibility_grace_seconds == 900
    assert slurm.required_headers == ["partition", "mem", "ntasks"]
    assert slurm.constraints == ["hasgpu", "ib"]
    assert slurm.extra_sbatch == ["--exclusive"]
    assert slurm.reservation == "maint-window"
    assert slurm.qos == "normal"
    assert slurm.gpu_session.gres == "gpu:H100:1"
    assert slurm.gpu_session.idle_minutes == 45
    assert slurm.controller_job.port == 9090
    # An explicit JSON null is "absent", matching Swift's decodeIfPresent.
    assert slurm.controller_job.gres is None

    storage = site.constraints.storage
    assert storage.node_stage_dir_template == "/tmp/$SLURM_JOB_ID"
    assert storage.node_scratch_gres == "lscratch:100"
    assert storage.hub_offline_mode == "online"
    assert storage.metadata_requires_local_filesystem is True
    assert storage.scanned_roles == ["workspace", "metadata", "hfCache", "archive"]
    assert storage.quota_command == "lfs quota -u $USER /scratch"
    assert (storage.scan_file_cap, storage.free_space_warn_gb, storage.free_space_fail_gb) == (
        25000, 10, 1)
    assert (storage.calendar_stale_days, storage.prestage_min_free_gb) == (30, 8)

    assert site.environment.module_system == "lmod"
    assert site.environment.python_provider == "venv"
    assert site.environment.modules == ["Miniforge3", "CUDA/12.8"]
    assert site.environment.server_extras == ["lora", "test"]
    assert site.environment.ssh_control_persist == "4h"
    assert site.environment.transfer_host == "xfer.example.edu"

    assert site.policy.login_nodes == LoginNodePolicy(
        hostname_patterns=["^login", "^submit"], allow_compute=False,
        require_allocation=True)
    assert site.policy.maintenance == MaintenancePolicy(
        calendar_path="~/.steerlab/maintenance.json",
        source_url="https://status.example.edu/feed.json",
        source_note="announced on the users list")
    assert site.policy.external_service_egress == "no"
    assert (site.policy.bind_override, site.policy.auth_mode_override) == (
        "127.0.0.1", "token")


# --- decode semantics ---------------------------------------------------------


def test_v1_fixture_migrates_in_memory_only():
    site = load_fixture("v1-maximal")
    # The stamp is PRESERVED, not upgraded — the renderer's default set reads it.
    assert site.schema_version == 1
    slurm = site.slurm
    assert slurm is not None
    assert slurm.gpu_types == ["A100", "H100", "L4"]
    assert slurm.gpu_vram_gb == {"A100": 80, "H100": 80, "L4": 24}
    # v1 → v2: the vocabulary appears under its v2 name, in declaration order.
    assert [(g.name, g.vram_gb, g.compute_capability) for g in slurm.gpus] == [
        ("A100", 80, None), ("H100", 80, None), ("L4", 24, None)]
    # …and the parallel-job cap reaches limits.
    assert slurm.limits.max_parallel_gpu_jobs == 15
    assert slurm.resolved_max_parallel_gpu_jobs == 15
    # Everything a v1 profile never stated is at its neutral default.
    assert slurm.commands == SchedulerCommands()
    assert slurm.job_defaults == JobDefaults()
    assert slurm.interruption == InterruptionPolicy()
    assert slurm.limits.max_submitted_jobs is None
    assert slurm.controller_job == JobClassResources()
    assert slurm.setup_job == JobClassResources()
    assert slurm.gpu_session == JobClassResources()
    assert site.environment == SiteEnvironment()
    assert site.policy == SitePolicy()
    assert site.constraints.storage == SiteStorage()


def test_absent_optionals_default_and_unknown_keys_are_ignored():
    site = ClusterSiteProfile.decode_json(
        '{"name": "Bare", "transport": {"kind": "ssh", "host": "h.test"},'
        ' "somethingANewerBuildAdded": {"x": 1}}')
    assert site.schema_version == 1  # absent stamp means v1
    assert site.notes == ""
    assert site.topology == "externalServer"
    assert site.scheduler.kind == "none"
    assert site.slurm is None
    assert site.transport.remote_port == 8080
    assert site.transport.vpn_expected is False
    assert site.environment == SiteEnvironment()
    assert site.policy == SitePolicy()
    assert site.constraints.storage == SiteStorage()
    assert site.metadata_root == "~/.steerlab"


def test_empty_schema2_blocks_decode_to_their_defaults():
    site = ClusterSiteProfile.decode_json(json.dumps({
        "schemaVersion": 2, "name": "Sparse",
        "transport": {"kind": "ssh", "host": "h"},
        "constraints": {"storage": {}},
        "environment": {}, "policy": {"loginNodes": {}, "maintenance": {}},
        "scheduler": {"kind": "slurm", "slurm": {
            "commands": {}, "jobDefaults": {}, "interruption": {}, "limits": {},
            "controllerJob": {}, "setupJob": {}, "gpuSession": {},
            "partitions": [{"name": "batch"}], "gpus": [{"name": "A100"}]}},
    }))
    slurm = site.slurm
    assert slurm is not None
    assert slurm.commands == SchedulerCommands(
        submit="sbatch", query="squeue", accounting="sacct", cancel="scancel")
    assert slurm.job_defaults == JobDefaults(memory=None, walltime=None, cpus_per_task=4)
    assert slurm.interruption == InterruptionPolicy()
    assert slurm.interruption.signal_seconds == 600
    assert slurm.interruption.export_mode == "none"
    assert slurm.limits == SchedulerLimits()
    assert slurm.accounting_visibility_grace_seconds == 600
    assert slurm.job_name_prefix == "steerlab"
    assert slurm.submit_from_bundle_directory is True
    assert site.environment.python_provider == "conda"
    assert site.environment.module_system == "none"
    assert site.environment.server_extras == ["all"]
    assert site.environment.env_file_path == "$HOME/steerlab-cluster.env"
    assert site.environment.token_file_path == "$HOME/.steerlab-token"
    assert site.environment.remote_repo_path == "~/steerlab"
    assert site.environment.ssh_control_persist == "8h"
    assert site.constraints.storage.hub_offline_mode == "auto"
    assert site.policy.external_service_egress == "unknown"
    assert site.policy.login_nodes.allow_compute is True  # empty list never refuses


def test_v2_only_keys_backfill_the_fields_todays_consumers_read():
    site = ClusterSiteProfile.decode_json(json.dumps({
        "schemaVersion": 2, "name": "New", "transport": {"kind": "ssh", "host": "h"},
        "scheduler": {"kind": "slurm", "slurm": {
            "gpus": [{"name": "H100", "vramGB": 80, "computeCapability": "sm_90"},
                     {"name": "L4"}],
            "limits": {"maxParallelGPUJobs": 6}}},
    }))
    slurm = site.slurm
    assert slurm is not None
    assert slurm.gpu_types == ["H100", "L4"]
    assert slurm.gpu_vram_gb == {"H100": 80}  # a type with no VRAM has no row
    assert slurm.max_parallel_gpu_jobs == 6
    assert slurm.resolved_max_parallel_gpu_jobs == 6
    assert [g.name for g in slurm.resolved_gpus] == ["H100", "L4"]


def test_refuses_a_newer_schema_version():
    with pytest.raises(SiteProfileError, match="newer than this build"):
        ClusterSiteProfile.decode_json(
            '{"schemaVersion": 99, "name": "X", "transport": {"kind": "ssh", "host": "h"}}')
    # …and accepts exactly the current one.
    assert ClusterSiteProfile.decode_json(
        '{"schemaVersion": %d, "name": "X", "transport": {"kind": "ssh", "host": "h"}}'
        % CURRENT_SCHEMA_VERSION).schema_version == CURRENT_SCHEMA_VERSION


def test_unknown_enum_raw_value_raises_rather_than_defaulting():
    for document in (
        '{"name": "X", "transport": {"kind": "ssh", "host": "h"}, "topology": "typo"}',
        '{"name": "X", "transport": {"kind": "ssh", "host": "h"},'
        ' "constraints": {"computeEgress": "maybe"}}',
        '{"name": "X", "transport": {"kind": "ssh", "host": "h"},'
        ' "environment": {"pythonProvider": "pyenv"}}',
        '{"name": "X", "transport": {"kind": "ssh", "host": "h"},'
        ' "constraints": {"storage": {"hubOfflineMode": "sometimes"}}}',
        '{"name": "X", "transport": {"kind": "carrier-pigeon"}}',
    ):
        with pytest.raises(SiteProfileError):
            ClusterSiteProfile.decode_json(document)


def test_required_and_mistyped_fields_fail_loudly():
    with pytest.raises(SiteProfileError, match="name is required"):
        ClusterSiteProfile.decode_json('{"transport": {"kind": "ssh", "host": "h"}}')
    with pytest.raises(SiteProfileError, match="transport is required"):
        ClusterSiteProfile.decode_json('{"name": "X"}')
    with pytest.raises(SiteProfileError, match="not a parseable base URL"):
        ClusterSiteProfile.decode_json(
            '{"name": "X", "transport": {"kind": "direct", "baseURL": "nonsense"}}')
    with pytest.raises(SiteProfileError, match="must be an integer"):
        ClusterSiteProfile.decode_json(
            '{"name": "X", "transport": {"kind": "ssh", "host": "h"},'
            ' "constraints": {"purgeDays": "thirty"}}')
    with pytest.raises(SiteProfileError, match="not valid JSON"):
        ClusterSiteProfile.decode_json("{")


# --- renderer semantics the goldens alone would not name ----------------------


def test_quoting_matches_each_values_shell_semantics():
    lines = {}
    for line in env.render_env_file(load_fixture("v2-maximal")).split("\n"):
        if line.startswith("export "):
            key, _, value = line[len("export "):].partition("=")
            lines[key] = value
    # Single-quoted: the placeholder is expanded on the compute node.
    assert lines["STEERLAB_NODE_STAGE_DIR"] == "'/tmp/$SLURM_JOB_ID'"
    # Single-quoted: regexes must not be glob-expanded by the shell.
    assert lines["STEERLAB_LOGIN_NODE_PATTERNS"] == "'^login ^submit'"
    # Double-quoted: $HOME must expand when the file is sourced.
    assert lines["STEERLAB_PREFIX"] == '"$HOME/envs/site"'
    # Bare: numbers and fixed vocabulary.
    assert lines["STEERLAB_PURGE_DAYS"] == "45"
    assert lines["STEERLAB_SERVER_PROFILE"] == "cluster"
    assert lines["STEERLAB_AUTH_TOKEN"] == '"$(cat "$HOME/.site-token")"'


def test_tilde_is_normalized_to_HOME():
    # A tilde inside double quotes is a literal tilde in POSIX shell.
    resolved = env.resolved_environment(load_fixture("v2-maximal"))
    assert resolved["STEERLAB_METADATA_ROOT"] == "$HOME/.steerlab-meta"
    assert resolved["STEERLAB_VENV"] == "$HOME/envs/site"
    assert resolved["STEERLAB_MAINTENANCE_CALENDAR"] == "$HOME/.steerlab/maintenance.json"
    assert env.shell_expanded("~") == "$HOME"
    assert env.shell_expanded("/already/absolute") == "/already/absolute"


def test_gpu_vocabulary_follows_declaration_order_and_refuses_to_invent():
    v1 = env.gpu_vocabulary(load_fixture("v1-maximal"))
    assert v1.types_value == "A100,H100,L4"
    assert v1.vram_value == "A100:80,H100:80,L4:24"
    # Declare-or-refuse: an undeclared v2 vocabulary stays empty rather than
    # inheriting one institution's inventory.
    assert env.gpu_vocabulary(load_fixture("v2-neutral")).is_empty


def test_control_characters_cannot_inject_a_line():
    site = load_fixture("v2-neutral")
    site.constraints.storage_roots["workspace"] = "/scratch/me\nexport EVIL=1"
    text = env.render_env_file(site)
    assert "EVIL=1\n" not in text
    resolved = env.resolved_environment(site)
    assert "EVIL" not in resolved
    assert resolved["STEERLAB_ROOT"] == "/scratch/meexport EVIL=1"


def test_the_login_node_guard_is_stated_by_every_scheduler_render():
    """WP5 step 10 (audit c35). ``bootstrap.sh`` READS this policy now, and
    reads "no keys at all" as "no rendered profile reached me" — so silence is
    not available to a site that means "no login-node rule". Hence: a v1 render
    states the script's own historical guard (materializing an existing site
    must not disarm it), and a v2 render states the schema's neutral one."""
    v1 = env.resolved_environment(load_fixture("v1-maximal"))
    assert v1["STEERLAB_LOGIN_NODE_PATTERNS"] == "^ss-sub"
    assert v1["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "0"
    assert v1["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "1"

    neutral = env.resolved_environment(load_fixture("v2-neutral"))
    assert "STEERLAB_LOGIN_NODE_PATTERNS" not in neutral  # empty list
    assert neutral["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "1"
    assert neutral["STEERLAB_LOGIN_NODE_REQUIRE_ALLOCATION"] == "0"

    # A declared policy wins in both directions.
    maximal = env.resolved_environment(load_fixture("v2-maximal"))
    assert maximal["STEERLAB_LOGIN_NODE_PATTERNS"] == "^login ^submit"
    assert maximal["STEERLAB_LOGIN_NODE_ALLOW_COMPUTE"] == "0"
    # …and audit c52: the judge preflight's egress reasoning is declare-or-omit.
    assert maximal["STEERLAB_EXTERNAL_SERVICE_EGRESS"] == "no"
    assert "STEERLAB_EXTERNAL_SERVICE_EGRESS" not in neutral
    assert "STEERLAB_EXTERNAL_SERVICE_EGRESS" not in v1


def test_schedulerless_site_renders_no_slurm_keys():
    site = ClusterSiteProfile.decode_json(json.dumps({
        "schemaVersion": 2, "name": "Workstation",
        "transport": {"kind": "direct", "baseURL": "http://127.0.0.1:8080"},
        "topology": "externalServer", "scheduler": {"kind": "none"},
        "constraints": {"computeEgress": "yes"},
    }))
    resolved = env.resolved_environment(site)
    assert resolved["STEERLAB_SERVER_PROFILE"] == "workstation"
    assert resolved["STEERLAB_EXECUTOR"] == "local"
    assert not [key for key in resolved if key.startswith("STEERLAB_SLURM_")]
    assert resolved["HF_HUB_OFFLINE"] == "0"  # egress yes + auto ⇒ hub reachable
    assert env.render_scheduler_headers(site, "study") == []
    # …and no login-node guard: its reader is bootstrap.sh, which provisions a
    # scheduler site. A workstation has no login node to make a statement about.
    assert not [key for key in resolved if key.startswith("STEERLAB_LOGIN_NODE_")]


def test_unknown_job_class_refuses():
    with pytest.raises(ValueError, match="unknown job class"):
        env.render_scheduler_headers(load_fixture("v2-maximal"), "postdoc")


def test_cpu_only_classes_do_not_inherit_site_wide_placement_directives():
    """WP5 §6.x handoff item 1, resolved at Step 6 and mirrored here.

    Site-wide ``--constraint`` / ``--reservation`` / ``extraSbatch`` describe
    where a site's COMPUTE runs. A CPU-only controller or setup job that
    inherits them asks a CPU partition for GPU node features, which Slurm
    queues forever rather than rejecting — the daemon never starts and there is
    no error to read. What a CPU-only class needs, it declares for itself.
    """
    profile = load_fixture("v2-maximal")
    site_wide = ["#SBATCH --constraint=hasgpu&ib",
                 "#SBATCH --reservation=maint-window",
                 "#SBATCH --exclusive"]
    for job_class in ("study", "gpuSession"):
        headers = env.render_scheduler_headers(profile, job_class)
        for directive in site_wide:
            assert directive in headers, f"{job_class} must still carry {directive}"
    for job_class in ("controller", "setup"):
        headers = env.render_scheduler_headers(profile, job_class)
        for directive in site_wide:
            assert directive not in headers, \
                f"{job_class} is CPU-only and must not inherit {directive}"
    # Nothing is lost: each class's own extraSbatch is still emitted.
    assert "#SBATCH --nice=100" in env.render_scheduler_headers(profile, "controller")
    assert "#SBATCH --hint=nomultithread" in env.render_scheduler_headers(profile, "setup")


# --- node-local scratch gres (cluster-operator requirement, 2026-08-19) ----------


def test_scratch_gres_rides_the_gres_carrying_classes_only():
    """Swift twin: ``nodeScratchGresRidesTheGresCarryingClassesOnly``. Node-local
    scratch is requested as a gres of its own on the classes that stage models;
    the controller and setup jobs stage nothing and must not ask for it."""
    site = load_fixture("example-slurm-site")
    assert env.resolved_scratch_gres(site) == "lscratch:100"
    for job_class in ("study", "gpuSession"):
        headers = env.render_scheduler_headers(site, job_class)
        assert "#SBATCH --gres=gpu:A100:1,lscratch:100" in headers
    for job_class in ("controller", "setup"):
        headers = env.render_scheduler_headers(site, job_class)
        assert not [line for line in headers if line.startswith("#SBATCH --gres=")]


def test_scratch_gres_renders_in_the_env_file_expanding_quoted():
    text = env.render_env_file(load_fixture("example-slurm-site"))
    assert 'export STEERLAB_SLURM_SCRATCH_GRES="lscratch:100"' in text


def test_scratch_gres_renders_alone_without_a_gpu_gres():
    """A site with scratch but no GPU gres asks for the scratch alone rather
    than dropping it for want of a GPU half."""
    site = load_fixture("v2-neutral")
    site.constraints.storage.node_scratch_gres = "lscratch:100"
    assert "#SBATCH --gres=lscratch:100" in env.render_scheduler_headers(site, "study")


def test_undeclared_scratch_gres_changes_nothing():
    """The declare-or-omit half (hard requirement): a site that never heard of
    node-local scratch renders exactly what it rendered before."""
    for fixture in ("v1-maximal", "v2-neutral"):
        site = load_fixture(fixture)
        assert env.resolved_scratch_gres(site) is None
        assert "STEERLAB_SLURM_SCRATCH_GRES" not in env.render_env_file(site)
        for job_class in env.JOB_CLASSES:
            for line in env.render_scheduler_headers(site, job_class):
                if line.startswith("#SBATCH --gres="):
                    assert "lscratch" not in line and "," not in line


def test_combined_gres_helper_matches_the_swift_twin():
    assert env.combined_gres("gpu:A100:1", "lscratch:100") == "gpu:A100:1,lscratch:100"
    assert env.combined_gres("gpu:A100:1", None) == "gpu:A100:1"
    assert env.combined_gres(None, "lscratch:100") == "lscratch:100"
    assert env.combined_gres(None, None) is None
