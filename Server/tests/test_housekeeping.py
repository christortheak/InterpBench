"""WS3 housekeeping: status shape, purge scan, HF-cache inventory, maintenance
calendar (validation, default path, executor pickup), snapshot persistence,
throughput folding, route privilege, and capability flags."""

import json
import os
import shutil
import time

import pytest

from steerlab_server.api import housekeeping as hk
from steerlab_server.api.executors import crosses_maintenance_window, first_crossing_window
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.profile import ServerProfile, capability_snapshot, validate_profile


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Isolated workspace + metadata + HF cache roots."""
    root = tmp_path / "workspace"
    (root / "runs").mkdir(parents=True)
    (root / "experiments").mkdir(parents=True)
    meta = tmp_path / "meta"
    meta.mkdir()
    hf = tmp_path / "hf"
    (hf / "hub").mkdir(parents=True)
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(meta))
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_PURGE_DAYS", raising=False)
    monkeypatch.delenv("STEERLAB_PURGE_WARN_DAYS", raising=False)
    monkeypatch.delenv("STEERLAB_HOUSEKEEPING_SCAN_CAP", raising=False)
    # WP5 step 11 widened what this module reads from the rendered site env;
    # the fixture's isolation has to widen with it, or a developer's own
    # sourced cluster env silently changes what these tests assert.
    for key in ("STEERLAB_HOUSEKEEPING_ROLES", "STEERLAB_FREE_SPACE_WARN_GB",
                "STEERLAB_FREE_SPACE_FAIL_GB", "STEERLAB_CALENDAR_STALE_DAYS",
                "STEERLAB_QUOTA_COMMAND", "STEERLAB_METADATA_REQUIRES_LOCAL_FS",
                "STEERLAB_ARCHIVE_ROOT", "STEERLAB_ASSET_ROOT",
                "STEERLAB_NODE_CACHE_ROOT"):
        monkeypatch.delenv(key, raising=False)
    return {"root": root, "meta": meta, "hf": hf}


def _age(path, days: float) -> None:
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


def _fabricate_hub_model(hf_root, org="acme", name="tiny", revision="abc123",
                         weight_bytes=4096, config=None):
    repo = hf_root / "hub" / f"models--{org}--{name}"
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    if config is not None:
        (snap / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snap


# --- status shape ---------------------------------------------------------------

def test_status_shape_all_top_level_keys_camel_case(site):
    body = hk.status()
    # "quota" joined the contract at WP5 step 11 (audit c46): the site's own
    # quota command's OUTPUT, beside the df numbers it corrects. It is null
    # here because this deployment declares no command — an honest "the site
    # never said", not a missing key.
    assert set(body) == {"generatedAt", "roots", "quota", "purgeRisk", "hfCache",
                         "maintenance", "evidence", "throughput"}
    assert body["quota"] is None
    assert {"workspace", "metadata", "hfCache"} <= set(body["roots"])
    for entry in body["roots"].values():
        assert set(entry) == {"path", "totalBytes", "freeBytes", "usedBytes",
                              "warning", "scope"}
        # df-level numbers are whole-filesystem, not quota — the scope marker
        # is what lets the UI say so instead of implying headroom.
        assert entry["scope"] == "filesystem"
    # No expensive scan has run: stale-but-honest nulls, not fabricated data.
    assert body["purgeRisk"] is None
    assert body["hfCache"] is None
    assert body["evidence"] == {"bundles": []}
    assert body["throughput"] == {"entries": []}
    assert body["maintenance"]["calendarPath"] is None
    assert body["maintenance"]["stale"] is True
    assert body["maintenance"]["next"] is None


# --- storage roles, thresholds, quota (WP5 step 11: audit c45, c46, c48) -----------

def test_scanned_roles_default_to_the_built_in_three(site):
    assert hk.scanned_roles() == ["workspace", "metadata", "hfCache"]


def test_declared_roles_bring_the_archive_tier_into_the_scan(site, tmp_path,
                                                             monkeypatch):
    """Audit c48/a1. The archive root has been a declarable profile field since
    before WP5 and NOTHING ever looked at it: the role set was a code constant.
    Declared, the tier a site keeps its cold artifacts on is finally reported —
    which is the point of declaring it."""
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("STEERLAB_ARCHIVE_ROOT", str(archive))
    assert "archive" not in hk.disk_roots()   # declared, but not scanned

    monkeypatch.setenv("STEERLAB_HOUSEKEEPING_ROLES", "workspace,archive")
    roots = hk.disk_roots()
    assert set(roots) == {"workspace", "archive"}
    assert roots["archive"]["path"] == os.path.realpath(str(archive))
    # A role the deployment has no path for drops out rather than erroring:
    # a typo must not take the housekeeping card down.
    monkeypatch.setenv("STEERLAB_HOUSEKEEPING_ROLES", "workspace,nonsense")
    assert set(hk.disk_roots()) == {"workspace"}


def test_free_space_thresholds_are_the_sites_not_the_engines(site, monkeypatch):
    """Audit c45. 10 GiB/1 GiB were code constants; on a site whose scratch tier
    is measured in petabytes they never fire, and on a thin node-local cache
    they fire constantly. A declared threshold above the real free space must
    produce the warning."""
    assert hk.disk_roots()["workspace"]["warning"] is None   # engine defaults
    free_gb = shutil.disk_usage(site["root"]).free / 1024**3
    monkeypatch.setenv("STEERLAB_FREE_SPACE_WARN_GB", str(int(free_gb) + 10))
    assert "low free space" in hk.disk_roots()["workspace"]["warning"]
    monkeypatch.setenv("STEERLAB_FREE_SPACE_FAIL_GB", str(int(free_gb) + 5))
    assert "critically low" in hk.disk_roots()["workspace"]["warning"]


def test_calendar_staleness_window_is_declarable(site, monkeypatch):
    hk.write_maintenance([{"start": "2026-08-01T06:00:00Z",
                           "end": "2026-08-01T18:00:00Z"}])
    path = hk.maintenance_calendar_path()
    _age(path, 10)
    assert hk.maintenance_status()["stale"] is False     # engine default: 30 d
    monkeypatch.setenv("STEERLAB_CALENDAR_STALE_DAYS", "7")
    assert hk.maintenance_status()["stale"] is True


def test_quota_report_displays_output_verbatim_and_never_parses_it(site,
                                                                   monkeypatch):
    """Audit c46. The df numbers are whole-filesystem and overstate the headroom
    on a quota'd tier; the site's own command is the number that bites. It is
    reported as TEXT — parsing it would mean assuming a format no two sites
    share, and a mis-parsed quota is worse than an unparsed one."""
    assert hk.quota_report() is None                    # no command declared
    monkeypatch.setenv("STEERLAB_QUOTA_COMMAND",
                       "printf 'Disk quotas for user x: 41%% of 500G\\n'")
    report = hk.quota_report()
    assert report["output"] == "Disk quotas for user x: 41% of 500G"
    assert report["exitCode"] == 0
    assert report["error"] is None
    # Nothing derived from the text: no parsed bytes, no percentage, no verdict.
    assert set(report) == {"command", "output", "exitCode", "error", "ranAt"}


def test_quota_report_survives_a_command_that_fails(site, monkeypatch):
    monkeypatch.setenv("STEERLAB_QUOTA_COMMAND",
                       "printf 'no such filesystem\\n' >&2; exit 3")
    report = hk.quota_report()
    assert report["exitCode"] == 3
    assert "exited 3" in report["error"]
    assert "no such filesystem" in report["output"]     # stderr is still shown


def test_status_never_runs_the_quota_command_but_refresh_does(site, monkeypatch):
    """The cheap read path stays cheap: a wedged `lfs quota` on a degraded
    filesystem must not hang every status poll. The tick runs it, the snapshot
    carries it, status() serves the snapshot's copy."""
    marker = site["meta"] / "quota-ran"
    monkeypatch.setenv("STEERLAB_QUOTA_COMMAND",
                       f"printf ok > {marker}; printf 'quota: ok\\n'")
    assert hk.status()["quota"] is None
    assert not marker.exists()

    hk.refresh()
    assert marker.exists()
    marker.unlink()
    served = hk.status()["quota"]
    assert served["output"] == "quota: ok"
    assert not marker.exists()      # served from the snapshot, not re-run


# --- purge scan -------------------------------------------------------------------

def test_purge_scan_counts_old_files_by_later_of_atime_mtime(site):
    runs = site["root"] / "runs"
    old = runs / "run-a" / "generations.jsonl"
    old.parent.mkdir()
    old.write_text("x" * 100, encoding="utf-8")
    _age(old, 25)
    fresh = runs / "run-b" / "generations.jsonl"
    fresh.parent.mkdir()
    fresh.write_text("y", encoding="utf-8")
    exp_old = site["root"] / "experiments" / "e1" / "experiment.json"
    exp_old.parent.mkdir()
    exp_old.write_text("{}", encoding="utf-8")
    _age(exp_old, 40)

    risk = hk.scan_purge_risk()
    assert risk["thresholdDays"] == 30
    assert risk["warnDays"] == 20
    assert risk["fileCount"] == 2
    assert risk["totalBytes"] == 102
    assert risk["warning"] is None
    worst_paths = [w["path"] for w in risk["worst"]]
    assert worst_paths[0].endswith("experiment.json")  # oldest first
    assert risk["worst"][0]["ageDays"] > 39
    assert all(str(fresh) != p for p in worst_paths)


def test_purge_scan_respects_env_thresholds_and_cap(site, monkeypatch):
    monkeypatch.setenv("STEERLAB_PURGE_DAYS", "10")
    monkeypatch.setenv("STEERLAB_PURGE_WARN_DAYS", "5")
    monkeypatch.setenv("STEERLAB_HOUSEKEEPING_SCAN_CAP", "2")
    runs = site["root"] / "runs"
    for index in range(3):
        path = runs / f"f{index}.jsonl"
        path.write_text("z", encoding="utf-8")
        _age(path, 7)
    risk = hk.scan_purge_risk()
    assert risk["thresholdDays"] == 10
    assert risk["warnDays"] == 5
    assert risk["fileCount"] == 2          # capped: a floor, not the truth
    assert "capped at 2 files" in risk["warning"]


def test_purge_scan_uses_run_root_when_set(site, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch-runs"
    scratch.mkdir()
    old = scratch / "old.bin"
    old.write_bytes(b"12345")
    _age(old, 60)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(scratch))
    risk = hk.scan_purge_risk()
    assert risk["fileCount"] == 1
    assert risk["worst"][0]["path"] == str(old)


# --- HF cache inventory ------------------------------------------------------------

def test_hf_cache_inventory_reads_hub_layout(site):
    _fabricate_hub_model(site["hf"], weight_bytes=10,
                         config={"num_hidden_layers": 2})
    inventory = hk.scan_hf_cache()
    assert inventory["root"] == os.path.realpath(str(site["hf"]))
    assert len(inventory["models"]) == 1
    entry = inventory["models"][0]
    assert entry["modelId"] == "acme/tiny"
    assert entry["revision"] == "abc123"
    assert entry["sizeBytes"] >= 10
    assert entry["lastUsedAt"] is not None


def test_hf_cache_inventory_none_without_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    assert hk.scan_hf_cache() is None


def test_model_snapshot_info_reports_weights_and_config(site):
    config = {"num_hidden_layers": 2, "num_key_value_heads": 2,
              "num_attention_heads": 4, "head_dim": 8, "torch_dtype": "bfloat16"}
    _fabricate_hub_model(site["hf"], weight_bytes=2048, config=config)
    info = hk.model_snapshot_info("acme/tiny")
    assert info["weightsBytes"] == 2048
    assert info["revision"] == "abc123"
    assert info["config"]["num_hidden_layers"] == 2
    assert hk.model_snapshot_info("acme/absent") is None


# --- maintenance calendar ------------------------------------------------------------

def test_write_maintenance_validates_and_persists_default_path(site):
    with pytest.raises(ValueError):
        hk.write_maintenance("not-a-list")
    with pytest.raises(ValueError):
        hk.write_maintenance([{"start": "garbage", "end": "2026-08-01T00:00:00Z"}])
    with pytest.raises(ValueError):
        hk.write_maintenance([{"start": "2026-08-02T00:00:00Z",
                               "end": "2026-08-01T00:00:00Z"}])
    with pytest.raises(ValueError):  # naive timestamps refused
        hk.write_maintenance([{"start": "2026-08-01T00:00:00",
                               "end": "2026-08-02T00:00:00"}])

    stored = hk.write_maintenance([
        {"start": "2026-08-01T06:00:00Z", "end": "2026-08-01T18:00:00Z",
         "label": "site quarterly"},
    ])
    expected_path = os.path.join(str(site["meta"]), "maintenance.json")
    assert stored["calendarPath"] == expected_path
    assert os.path.isfile(expected_path)
    assert stored["windows"][0]["label"] == "site quarterly"

    status = hk.maintenance_status()
    assert status["calendarPath"] == expected_path
    assert status["stale"] is False
    assert len(status["windows"]) == 1


def test_executor_picks_up_default_calendar_without_env(site):
    from datetime import datetime, timedelta, timezone
    start = datetime.now(timezone.utc) + timedelta(hours=1)
    end = start + timedelta(hours=4)
    hk.write_maintenance([{"start": start.isoformat(), "end": end.isoformat(),
                           "label": "drain"}])
    # env var unset: the executor resolves the same metadata-root default.
    assert crosses_maintenance_window("04:00:00", None) is True
    window = first_crossing_window("04:00:00", None)
    assert window["label"] == "drain"
    assert crosses_maintenance_window("00:30:00", None) is False


def test_maintenance_env_path_still_wins(site, tmp_path, monkeypatch):
    custom = tmp_path / "cal.json"
    monkeypatch.setenv("STEERLAB_MAINTENANCE_CALENDAR", str(custom))
    stored = hk.write_maintenance([])
    assert stored["calendarPath"] == str(custom)
    assert os.path.isfile(custom)


# --- snapshot persistence -----------------------------------------------------------

def test_refresh_persists_snapshot_and_status_serves_it_stale(site):
    old = site["root"] / "runs" / "r" / "old.jsonl"
    old.parent.mkdir()
    old.write_text("x", encoding="utf-8")
    _age(old, 45)
    _fabricate_hub_model(site["hf"])
    snapshot = hk.refresh()
    path = os.path.join(str(site["meta"]), hk.SNAPSHOT_FILENAME)
    assert os.path.isfile(path)
    body = hk.status()
    # Stale-but-honest: the cheap route serves the tick's expensive scans
    # verbatim, stamped with the SNAPSHOT's generatedAt.
    assert body["generatedAt"] == snapshot["generatedAt"]
    assert body["purgeRisk"]["fileCount"] == 1
    assert body["hfCache"]["models"][0]["modelId"] == "acme/tiny"


def test_refresh_lists_evidence_bundles_with_job_attribution(site, tmp_path):
    run_dir = site["root"] / "runs" / "20260712T000000-exp-x-run"
    run_dir.mkdir()
    bundle = run_dir / "20260712T000000-exp-x-run.evidence-bundle.tar.gz"
    bundle.write_bytes(b"tar")
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      sweep_orphans=False)
    jobs.record_external(
        "study-submit", status="succeeded", executor="slurm",
        result={"runDirectory": str(run_dir)})
    snapshot = hk.refresh(jobs=jobs)
    bundles = snapshot["evidence"]["bundles"]
    assert len(bundles) == 1
    assert bundles[0]["runId"] == "20260712T000000-exp-x-run"
    assert bundles[0]["sizeBytes"] == 3
    assert bundles[0]["jobId"] is not None


# --- throughput ----------------------------------------------------------------------

def _terminal_job(jobs, *, model="org/m", gres="gpu:L4:1", elapsed=3600.0,
                  count=100, status="succeeded"):
    return jobs.record_external(
        "study-submit", status=status, executor="slurm",
        requested_resources={"gres": gres, "modelID": model},
        result={"elapsedSeconds": elapsed, "recordCount": count})


def test_fold_throughput_cumulative_mean_and_idempotent(site, tmp_path):
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      sweep_orphans=False)
    _terminal_job(jobs, elapsed=3600, count=100)
    table = hk.fold_throughput(jobs)
    entry = table["entries"][0]
    assert entry["modelId"] == "org/m"
    assert entry["gpuType"] == "L4"
    assert entry["recordsPerHour"] == 100.0
    assert entry["samples"] == 1

    # Folding again must not double-count the same job.
    table = hk.fold_throughput(jobs)
    assert table["entries"][0]["samples"] == 1

    _terminal_job(jobs, elapsed=3600, count=200)
    table = hk.fold_throughput(jobs)
    assert table["entries"][0]["recordsPerHour"] == 150.0
    assert table["entries"][0]["samples"] == 2
    assert hk.throughput_lookup("org/m", "L4")["samples"] == 2
    assert hk.throughput_lookup("org/m", "A100") is None


def test_fold_throughput_skips_unknowable_records(site, tmp_path):
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      sweep_orphans=False)
    # No modelID stamp → skipped, never guessed.
    jobs.record_external("study-submit", status="succeeded", executor="slurm",
                         requested_resources={"gres": "gpu:L4:1"},
                         result={"elapsedSeconds": 3600, "recordCount": 50})
    # Non-terminal → skipped.
    jobs.record_external("study-submit", status="running", executor="slurm",
                         requested_resources={"gres": "gpu:L4:1", "modelID": "org/m"},
                         result={"elapsedSeconds": 3600, "recordCount": 50})
    # No counts → skipped.
    jobs.record_external("study-submit", status="succeeded", executor="slurm",
                         requested_resources={"gres": "gpu:L4:1", "modelID": "org/m"})
    assert hk.fold_throughput(jobs)["entries"] == []


def test_parse_gpu_type_variants():
    assert hk.parse_gpu_type("gpu:A100:1") == "A100"
    assert hk.parse_gpu_type("gpu:A100") == "A100"
    assert hk.parse_gpu_type("A100") == "A100"
    assert hk.parse_gpu_type(None) is None
    assert hk.parse_gpu_type("") is None


# --- routes: privilege + behavior ------------------------------------------------------

def test_housekeeping_route_privilege(site, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    client = TestClient(app)

    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    # status is read-only and unprivileged.
    assert client.get("/api/housekeeping/status").status_code == 200
    # refresh/maintenance are privileged: token required under the Slurm executor.
    assert client.post("/api/housekeeping/refresh").status_code == 503
    assert client.post("/api/housekeeping/maintenance",
                       json={"windows": []}).status_code == 503
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekret")
    headers = {"authorization": "Bearer sekret"}
    assert client.post("/api/housekeeping/refresh",
                       headers=headers).status_code == 200
    assert client.post("/api/housekeeping/maintenance", headers=headers,
                       json={"windows": []}).status_code == 200


def test_maintenance_route_validates_and_stores(site, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    client = TestClient(app)
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")

    bad = client.post("/api/housekeeping/maintenance", json={
        "windows": [{"start": "2026-08-02T00:00:00Z", "end": "2026-08-01T00:00:00Z"}]})
    assert bad.status_code == 400

    good = client.post("/api/housekeeping/maintenance", json={
        "windows": [{"start": "2026-08-01T06:00:00Z",
                     "end": "2026-08-01T18:00:00Z", "label": None}]})
    assert good.status_code == 200
    body = good.json()
    assert body["calendarPath"].endswith("maintenance.json")
    assert len(body["windows"]) == 1

    status = client.get("/api/housekeeping/status").json()
    assert len(status["maintenance"]["windows"]) == 1


# --- capability flags + profile warnings -------------------------------------------------

def test_purge_scan_says_whether_the_window_is_the_sites_or_the_engines(
        site, monkeypatch):
    """The 30/20 pair was transcribed from one institution's /scratch policy. A
    card that prints "30 days" without saying where it came from quotes a
    stranger's policy back at the researcher, so the scan reports its source."""
    assert hk.scan_purge_risk()["policySource"] == "default"
    monkeypatch.setenv("STEERLAB_PURGE_DAYS", "45")
    risk = hk.scan_purge_risk()
    assert risk["policySource"] == "site"
    assert risk["thresholdDays"] == 45


def test_capability_snapshot_advertises_housekeeping_and_preflight(site):
    cluster = capability_snapshot()["cluster"]
    assert cluster["housekeeping"] is True
    assert cluster["preflight"] is True


def test_capability_snapshot_reports_what_housekeeping_will_actually_look_at(
        site, monkeypatch):
    """A client cannot otherwise tell an archive tier the site never declared
    from one this scan simply does not walk, nor a missing quota line from a
    site that declared no command."""
    policy = capability_snapshot()["cluster"]["housekeepingPolicy"]
    assert policy == {"scannedRoles": ["workspace", "metadata", "hfCache"],
                      "quotaCommand": None, "purgeDays": None,
                      "metadataRequiresLocalFilesystem": False}

    monkeypatch.setenv("STEERLAB_HOUSEKEEPING_ROLES", "workspace,archive")
    monkeypatch.setenv("STEERLAB_QUOTA_COMMAND", "quota_report")
    monkeypatch.setenv("STEERLAB_PURGE_DAYS", "45")
    monkeypatch.setenv("STEERLAB_METADATA_REQUIRES_LOCAL_FS", "1")
    policy = capability_snapshot()["cluster"]["housekeepingPolicy"]
    assert policy == {"scannedRoles": ["workspace", "archive"],
                      "quotaCommand": "quota_report", "purgeDays": "45",
                      "metadataRequiresLocalFilesystem": True}


def test_validate_profile_probes_metadata_locking_only_when_declared(
        site, monkeypatch):
    """Audit c43. "SQLite job DB on /home, NOT Lustre (locking)" was a COMMENT
    in bootstrap.sh — a rule nothing enforced and a new site had no way to
    inherit. Declared, it becomes a real POSIX-record-lock probe of the mount."""
    assert "metadataLocking" not in {c["name"] for c in validate_profile()["checks"]}
    monkeypatch.setenv("STEERLAB_METADATA_REQUIRES_LOCAL_FS", "1")
    checks = {c["name"]: c["status"] for c in validate_profile()["checks"]}
    # tmp_path is a local filesystem, so the probe passes — the assertion that
    # matters is that a real acquire/release happened, not that it failed.
    assert checks["metadataLocking"] == "ok"


def test_validate_profile_names_the_missing_quota_command_on_a_cluster(
        site, monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(site["root"] / "runs"))
    checks = {c["name"]: c["status"] for c in validate_profile()["checks"]}
    assert checks["quotaCommand"] == "warn"
    monkeypatch.setenv("STEERLAB_QUOTA_COMMAND", "definitely-not-a-real-binary -u")
    checks = {c["name"]: c["status"] for c in validate_profile()["checks"]}
    assert checks["quotaCommand"] == "warn"     # declared, but not on PATH
    monkeypatch.setenv("STEERLAB_QUOTA_COMMAND", "printf ok")
    checks = {c["name"]: c["status"] for c in validate_profile()["checks"]}
    assert checks["quotaCommand"] == "ok"


def test_validate_profile_warns_on_unset_purge_days_for_cluster(site, monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(site["root"] / "runs"))
    report = validate_profile()
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names.get("purgePolicy") == "warn"
    monkeypatch.setenv("STEERLAB_PURGE_DAYS", "30")
    report = validate_profile()
    assert "purgePolicy" not in {c["name"] for c in report["checks"]}


def test_validate_profile_bind_batch_topology_downgrades_to_warn(site, monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(site["root"] / "runs"))
    monkeypatch.setenv("STEERLAB_BIND", "0.0.0.0")
    # Without token/batch: hard failure, as before.
    monkeypatch.setenv("STEERLAB_LAUNCH_TOPOLOGY", "tunnel")
    report = validate_profile()
    assert {c["name"]: c["status"] for c in report["checks"]}["bind"] == "fail"
    # Daemon-in-a-job with token auth: warning, not failure (documented in the
    # controller template — the tunnel reaches the node's external interface).
    monkeypatch.setenv("STEERLAB_LAUNCH_TOPOLOGY", "batch")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "token")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekret")
    report = validate_profile()
    assert {c["name"]: c["status"] for c in report["checks"]}["bind"] == "warn"


# --- CLI -----------------------------------------------------------------------------

def test_cli_housekeeping_status_and_maintenance_set(site, tmp_path, capsys):
    from steerlab_server import cli
    assert cli.main(["housekeeping", "status"]) == 0
    body = json.loads(capsys.readouterr().out)
    # Same contract as the API body — "quota" added at WP5 step 11 (audit c46).
    assert set(body) == {"generatedAt", "roots", "quota", "purgeRisk", "hfCache",
                         "maintenance", "evidence", "throughput"}

    windows_file = tmp_path / "windows.json"
    windows_file.write_text(json.dumps({"windows": [
        {"start": "2026-08-01T06:00:00Z", "end": "2026-08-01T18:00:00Z",
         "label": "quarterly"}]}), encoding="utf-8")
    assert cli.main(["housekeeping", "maintenance", "set",
                     "--file", str(windows_file)]) == 0
    stored = json.loads(capsys.readouterr().out)
    assert stored["windows"][0]["label"] == "quarterly"

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps([{"start": "nope", "end": "nope"}]),
                        encoding="utf-8")
    assert cli.main(["housekeeping", "maintenance", "set",
                     "--file", str(bad_file)]) == 1


def test_cli_housekeeping_status_refresh_scans(site, capsys):
    old = site["root"] / "runs" / "old.jsonl"
    old.write_text("x", encoding="utf-8")
    _age(old, 45)
    from steerlab_server import cli
    assert cli.main(["housekeeping", "status", "--refresh"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["purgeRisk"]["fileCount"] == 1
