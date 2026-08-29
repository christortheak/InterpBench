"""Housekeeping: quota, purge risk, HF-cache inventory, maintenance calendar,
evidence bundles, and per-(model, GPU) throughput (TURNKEY-CLUSTER-PLAN WS3).

Design: the expensive scans (purge risk over the runs/experiments trees, the
HF-cache inventory, the evidence-bundle walk) run in a low-frequency
housekeeping *tick* (piggybacked on the JobManager monitor thread, hourly) —
or on demand via ``refresh()`` — and the result is persisted to
``metadata_root/housekeeping.json`` so a restarted server serves
stale-but-honest data with its original ``generatedAt``. ``status()`` is the
cheap read path: it serves the persisted snapshot plus live ``df`` stats, the
(small) maintenance calendar, and the (small) throughput table.

Everything here is stdlib-only — no torch/transformers import tax — so the
CLI verbs and the preflight checks in ``submissions.py`` can use it freely.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone

from .profile import ServerProfile
from .safe_paths import StorageRoots

SNAPSHOT_FILENAME = "housekeeping.json"
THROUGHPUT_FILENAME = "throughput.json"
MAINTENANCE_FILENAME = "maintenance.json"
EVIDENCE_SUFFIX = ".evidence-bundle.tar.gz"   # bundles.package_evidence naming

# WP5 Step 11: every constant below is this ENGINE's fallback, reached only
# when the site's rendered env file says nothing. The site's own values arrive
# as the STEERLAB_* keys read beside each one (the profile's
# constraints.purgeDays / constraints.storage.*), so none of these is a claim
# about any institution. The purge pair is the one that used to be — it was
# transcribed from one institution's /scratch policy — so the scan reports which of the
# two it used rather than presenting a default as the site's policy.
PURGE_DAYS_DEFAULT = 30
PURGE_WARN_DAYS_DEFAULT = 20
SCAN_FILE_CAP_DEFAULT = 50_000
WORST_OFFENDER_COUNT = 10
CALENDAR_STALE_DAYS = 30       # a calendar untouched this long is suspect
LOW_FREE_WARN_GB_DEFAULT = 10
LOW_FREE_FAIL_GB_DEFAULT = 1
LOW_FREE_WARN_BYTES = LOW_FREE_WARN_GB_DEFAULT * 1024**3
LOW_FREE_FAIL_BYTES = LOW_FREE_FAIL_GB_DEFAULT * 1024**3
#: Roles the scan reports when the site declares no ``scannedRoles``.
DEFAULT_SCANNED_ROLES = ("workspace", "metadata", "hfCache")
#: Seconds the site's quota command may run before it is abandoned. It is a
#: diagnostic: a wedged ``lfs quota`` must never hold the housekeeping tick.
QUOTA_COMMAND_TIMEOUT_SECONDS = 20
#: Quota output is displayed verbatim, so it is bounded rather than trusted.
QUOTA_OUTPUT_CAP_CHARS = 8192
_FOLDED_JOB_CAP = 5000

# Serializes read-modify-write of throughput.json and snapshot writes between
# the hourly tick thread, the refresh route, and CLI callers in-process.
_IO_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_json(path: str) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str) -> str | None:
    raw = (os.environ.get(name) or "").strip()
    return raw or None


# --- roots (cheap, always live) -------------------------------------------------

def hf_cache_root() -> str:
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")


def hf_hub_dir(cache_root: str | None = None) -> str:
    """Hub directory holding ``models--org--name`` repos, resolved the way
    ``huggingface_hub`` resolves it: explicit ``cache_root`` argument →
    ``<cache_root>/hub``; else ``HF_HUB_CACHE`` verbatim (it names the hub
    dir directly); else ``$HF_HOME/hub``. Mirrors
    ``steering.model_loader.hf_hub_dir`` — duplicated (not imported) because
    this module stays stdlib-only while model_loader imports torch. Keep the
    two in sync: installs download through ``huggingface_hub``, so a scan
    resolving the cache differently reports a freshly installed model as
    absent (live cluster run 2026-07-17)."""
    if cache_root:
        return os.path.join(cache_root, "hub")
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return hub
    return os.path.join(hf_cache_root(), "hub")


def scanned_roles() -> list[str]:
    """Storage roles the scan reports (audit c48). ``STEERLAB_HOUSEKEEPING_ROLES``
    is the site's declaration (``constraints.storage.scannedRoles``); unset means
    the engine's built-in three, which is what the role set was FIXED at before
    WP5 Step 11 — an archive tier could be declared and was never looked at.
    Unknown role names are kept: they resolve to no path and drop out below,
    which is quieter than refusing a scan over a typo."""
    raw = _env_str("STEERLAB_HOUSEKEEPING_ROLES")
    if not raw:
        return list(DEFAULT_SCANNED_ROLES)
    roles = [part.strip() for part in raw.split(",")]
    return [role for role in roles if role] or list(DEFAULT_SCANNED_ROLES)


def quota_report(profile: ServerProfile | None = None) -> dict | None:
    """The site's quota command and its OUTPUT VERBATIM (audit c46), or None
    when the site declares none.

    Never parsed: no two sites' quota tools agree on a format, and a
    mis-parsed quota is worse than an unparsed one. This closes the honesty
    marker in ``disk_roots`` — df numbers are whole-filesystem and overstate
    the headroom on a quota'd tier, and this is the number that bites.

    Trust: the command is site data from the reviewed, hash-pinned env file
    (the same trust level as ``STEERLAB_PYTHON`` and ``STEERLAB_TRANSFER_METHOD``,
    which this server already executes), and it is deliberately run through the
    shell because sites write it with ``$USER`` and pipes. It runs only on the
    expensive housekeeping tick, never on the cheap ``status()`` read path.
    """
    profile = profile or ServerProfile.from_env()
    command = profile.quota_command
    if not command:
        return None
    import subprocess  # local: keeps the module's import cost at stdlib-cheap

    report: dict = {"command": command, "output": None, "exitCode": None,
                    "error": None, "ranAt": _now_iso()}
    try:
        completed = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=QUOTA_COMMAND_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        report["error"] = (f"quota command timed out after "
                           f"{QUOTA_COMMAND_TIMEOUT_SECONDS}s")
        return report
    except OSError as exc:
        report["error"] = f"quota command could not be run: {exc}"
        return report
    text = (completed.stdout or "") + (completed.stderr or "")
    if len(text) > QUOTA_OUTPUT_CAP_CHARS:
        text = text[:QUOTA_OUTPUT_CAP_CHARS] + "\n… (output truncated)"
    report["output"] = text.strip() or None
    report["exitCode"] = completed.returncode
    if completed.returncode != 0:
        report["error"] = f"quota command exited {completed.returncode}"
    return report


def disk_roots(profile: ServerProfile | None = None) -> dict:
    """Live ``df``-level stats per storage role. Which roles are reported is the
    site's declaration (``scanned_roles()``); the built-in set is the historical
    workspace / metadata / hfCache. Each role is included only when its path
    resolves to a directory, so a role a deployment does not use is absent
    rather than null."""
    profile = profile or ServerProfile.from_env()
    available = {
        "workspace": profile.root,
        "metadata": profile.metadata_root,
        "hfCache": hf_cache_root(),
        # Roles the profile has carried since WP5 Step 3 but that nothing ever
        # scanned (audit c40-c42, a1): a site whose runs land on scratch and
        # whose archive is a separate quota'd tier could not see either.
        "run": profile.run_root,
        "asset": profile.asset_root,
        "archive": profile.archive_root,
        "nodeCache": profile.node_cache_root,
    }
    warn_bytes = _env_int("STEERLAB_FREE_SPACE_WARN_GB",
                          LOW_FREE_WARN_GB_DEFAULT) * 1024**3
    fail_bytes = _env_int("STEERLAB_FREE_SPACE_FAIL_GB",
                          LOW_FREE_FAIL_GB_DEFAULT) * 1024**3
    out: dict = {}
    for role in scanned_roles():
        path = available.get(role)
        if not path or not os.path.isdir(path):
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        warning: str | None = None
        if usage.free < fail_bytes:
            warning = (f"critically low free space: "
                       f"{usage.free / 1024**3:.1f} GiB free")
        elif usage.free < warn_bytes:
            warning = f"low free space: {usage.free / 1024**3:.1f} GiB free"
        out[role] = {
            "path": os.path.realpath(path),
            "totalBytes": int(usage.total),
            "freeBytes": int(usage.free),
            "usedBytes": int(usage.used),
            "warning": warning,
            # HONESTY MARKER: these are df-level numbers for the WHOLE
            # filesystem, not the caller's allocation quota (Lustre/GPFS
            # quotas bite long before df does). The real quota, where a site
            # declares the command that reports it, is the sibling "quota"
            # block — displayed verbatim, never parsed into these numbers.
            "scope": "filesystem",
        }
    return out


# --- maintenance calendar --------------------------------------------------------

def maintenance_calendar_path(profile: ServerProfile | None = None) -> str:
    """The effective calendar path: ``STEERLAB_MAINTENANCE_CALENDAR`` when set,
    else ``metadata_root/maintenance.json`` (the same default the executor's
    submit-time refusal reads, so windows written via the API always bind)."""
    profile = profile or ServerProfile.from_env()
    return profile.maintenance_calendar_path or os.path.join(
        profile.metadata_root, MAINTENANCE_FILENAME)


def _parse_window(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    try:
        start = datetime.fromisoformat(str(item["start"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(item["end"]).replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError):
        return None
    if end <= start:
        return None
    label = item.get("label")
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "label": str(label) if label is not None else None,
    }


def maintenance_status(profile: ServerProfile | None = None) -> dict:
    path = maintenance_calendar_path(profile)
    exists = os.path.isfile(path)
    raw = _read_json(path) if exists else None
    items = raw.get("windows", []) if isinstance(raw, dict) else (raw or [])
    windows = [w for w in (_parse_window(item) for item in items) if w is not None]
    windows.sort(key=lambda w: w["start"])
    now = datetime.now(timezone.utc)
    upcoming = [w for w in windows
                if datetime.fromisoformat(w["end"]) > now]
    stale = True
    if exists:
        # How long a calendar may sit untouched before it is suspect is site
        # data (audit c45): a site that publishes maintenance quarterly and one
        # that publishes weekly disagree about what "stale" means.
        stale_days = _env_int("STEERLAB_CALENDAR_STALE_DAYS", CALENDAR_STALE_DAYS)
        try:
            age_days = (now.timestamp() - os.path.getmtime(path)) / 86400.0
            stale = age_days > stale_days
        except OSError:
            stale = True
    return {
        "calendarPath": path if exists else None,
        "windows": windows,
        "next": upcoming[0] if upcoming else None,
        "stale": stale,
    }


def write_maintenance(windows: object,
                      profile: ServerProfile | None = None) -> dict:
    """Validate and persist the maintenance calendar. Raises ``ValueError`` on
    malformed input (unparseable timestamps, end <= start) — a silently dropped
    window would let a doomed job queue."""
    if not isinstance(windows, list):
        raise ValueError("maintenance windows must be a list of "
                         "{start, end, label} objects")
    stored: list[dict] = []
    for index, item in enumerate(windows):
        if not isinstance(item, dict):
            raise ValueError(f"window[{index}] is not an object")
        try:
            start = datetime.fromisoformat(
                str(item["start"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(item["end"]).replace("Z", "+00:00"))
        except KeyError as exc:
            raise ValueError(f"window[{index}] is missing {exc.args[0]!r}")
        except (ValueError, TypeError):
            raise ValueError(
                f"window[{index}] start/end must be ISO-8601 timestamps")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError(
                f"window[{index}] timestamps must carry a timezone "
                "(e.g. 2026-08-01T06:00:00Z)")
        if end <= start:
            raise ValueError(f"window[{index}] end must be after start")
        label = item.get("label")
        stored.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": str(label) if label is not None else None,
        })
    stored.sort(key=lambda w: w["start"])
    path = maintenance_calendar_path(profile)
    _atomic_write_json(path, {
        "schemaVersion": 1,
        "updatedAt": _now_iso(),
        "windows": stored,
    })
    return {"calendarPath": path, "windows": stored}


# --- purge-risk scan (expensive; tick/refresh only) -------------------------------

def scan_purge_risk(profile: ServerProfile | None = None,
                    roots: StorageRoots | None = None) -> dict:
    """Files under the runs + experiments trees whose age exceeds the warn
    threshold. Age uses the LATER of atime/mtime — Lustre atime is unreliable
    (often mount-option-frozen), so mtime is the honest floor."""
    profile = profile or ServerProfile.from_env()
    roots = roots or StorageRoots.from_profile(profile)
    threshold_days = _env_int("STEERLAB_PURGE_DAYS", PURGE_DAYS_DEFAULT)
    warn_days = _env_int("STEERLAB_PURGE_WARN_DAYS", PURGE_WARN_DAYS_DEFAULT)
    # Whether those two numbers ARE this site's policy or merely this engine's
    # fallback: the constants were transcribed from one institution's /scratch
    # rule, so a card that shows "30 days" without saying which is quoting a
    # stranger's policy back at the researcher (WP5 Step 11).
    policy_source = "site" if _env_str("STEERLAB_PURGE_DAYS") else "default"
    cap = _env_int("STEERLAB_HOUSEKEEPING_SCAN_CAP", SCAN_FILE_CAP_DEFAULT)
    now = datetime.now(timezone.utc).timestamp()

    scanned = 0
    capped = False
    at_risk_count = 0
    at_risk_bytes = 0
    offenders: list[tuple[float, str, int]] = []
    trees = [roots.runs, os.path.join(roots.workspace, "experiments")]
    for tree in trees:
        if capped or not os.path.isdir(tree):
            continue
        for dirpath, _dirs, filenames in os.walk(tree):
            for name in filenames:
                if scanned >= cap:
                    capped = True
                    break
                scanned += 1
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path, follow_symlinks=False)
                except OSError:
                    continue
                age_days = (now - max(st.st_atime, st.st_mtime)) / 86400.0
                if age_days > warn_days:
                    at_risk_count += 1
                    at_risk_bytes += st.st_size
                    offenders.append((age_days, path, st.st_size))
            if capped:
                break
    offenders.sort(key=lambda item: item[0], reverse=True)
    worst = [{"path": path, "ageDays": round(age, 1), "bytes": size}
             for age, path, size in offenders[:WORST_OFFENDER_COUNT]]
    return {
        "scannedAt": _now_iso(),
        "thresholdDays": threshold_days,
        "warnDays": warn_days,
        "policySource": policy_source,
        "fileCount": at_risk_count,
        "totalBytes": at_risk_bytes,
        "worst": worst,
        "warning": (f"scan capped at {cap} files — counts are a floor"
                    if capped else None),
    }


# --- HF cache inventory (expensive-ish; tick/refresh + preflight) ------------------

def _decode_repo_dirname(name: str) -> str:
    encoded = name[len("models--"):]
    parts = encoded.split("--")
    return parts[0] + "/" + "--".join(parts[1:]) if len(parts) >= 2 else encoded


def _repo_dirname(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _refs_main(repo_dir: str) -> str | None:
    try:
        with open(os.path.join(repo_dir, "refs", "main"),
                  encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return None
    return value or None


def _snapshot_dirs(repo_dir: str) -> list[str]:
    snap_root = os.path.join(repo_dir, "snapshots")
    try:
        return sorted(
            os.path.join(snap_root, entry) for entry in os.listdir(snap_root)
            if os.path.isdir(os.path.join(snap_root, entry)))
    except OSError:
        return []


def _tree_bytes_and_mtime(directory: str) -> tuple[int, float | None]:
    """(total bytes, latest atime/mtime) over a tree, following symlinks for
    size (HF snapshots symlink into blobs/) but deduping by realpath so a
    blob shared by two snapshot entries counts once."""
    total = 0
    latest: float | None = None
    seen: set[str] = set()
    for dirpath, _dirs, filenames in os.walk(directory):
        for name in filenames:
            path = os.path.join(dirpath, name)
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            try:
                st = os.stat(real)
            except OSError:
                continue
            total += st.st_size
            used = max(st.st_atime, st.st_mtime)
            latest = used if latest is None else max(latest, used)
    return total, latest


def scan_hf_cache(cache_root: str | None = None) -> dict | None:
    """Inventory of the HF hub cache (``HF_HOME`` layout:
    ``hub/models--org--name/snapshots/<revision>``). None when no hub tree."""
    hub = hf_hub_dir(cache_root)
    # Displayed root: the cache root when derivable, else the hub dir itself
    # (an explicit HF_HUB_CACHE need not live under any HF_HOME).
    root = cache_root or (
        hf_cache_root() if not os.environ.get("HF_HUB_CACHE") else hub)
    try:
        entries = sorted(os.listdir(hub))
    except OSError:
        return None
    models: list[dict] = []
    for name in entries:
        if not name.startswith("models--"):
            continue
        repo_dir = os.path.join(hub, name)
        snapshots = _snapshot_dirs(repo_dir)
        if not snapshots and not os.path.isdir(os.path.join(repo_dir, "blobs")):
            continue
        revision = _refs_main(repo_dir) or (
            os.path.basename(snapshots[-1]) if snapshots else None)
        blobs = os.path.join(repo_dir, "blobs")
        if os.path.isdir(blobs):
            size_bytes, _ = _tree_bytes_and_mtime(blobs)
        else:
            size_bytes = sum(_tree_bytes_and_mtime(snap)[0] for snap in snapshots)
        last_used: float | None = None
        for snap in snapshots:
            _, used = _tree_bytes_and_mtime(snap)
            if used is not None:
                last_used = used if last_used is None else max(last_used, used)
        models.append({
            "modelId": _decode_repo_dirname(name),
            "revision": revision,
            "sizeBytes": int(size_bytes),
            "lastUsedAt": _iso(last_used) if last_used is not None else None,
        })
    return {"root": os.path.realpath(root), "models": models}


_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".gguf")


def model_snapshot_info(model_id: str, revision: str | None = None,
                        cache_root: str | None = None) -> dict | None:
    """Resolve one cached model's snapshot for the memory-fit preflight:
    real weight bytes on disk plus the parsed ``config.json``. None when the
    model is not in the cache. When the pinned revision's snapshot is absent
    but another snapshot exists, that snapshot is used and reported —
    preflight degrades honestly instead of pretending the cache is empty."""
    repo_dir = os.path.join(hf_hub_dir(cache_root), _repo_dirname(model_id))
    if not os.path.isdir(repo_dir):
        return None
    snapshots = _snapshot_dirs(repo_dir)
    if not snapshots:
        return None
    chosen: str | None = None
    if revision:
        candidate = os.path.join(repo_dir, "snapshots", revision)
        if os.path.isdir(candidate):
            chosen = candidate
    if chosen is None:
        main = _refs_main(repo_dir)
        if main:
            candidate = os.path.join(repo_dir, "snapshots", main)
            if os.path.isdir(candidate):
                chosen = candidate
    if chosen is None:
        chosen = snapshots[-1]
    weights = 0
    seen: set[str] = set()
    for dirpath, _dirs, filenames in os.walk(chosen):
        for name in filenames:
            if not name.endswith(_WEIGHT_EXTENSIONS):
                continue
            real = os.path.realpath(os.path.join(dirpath, name))
            if real in seen:
                continue
            seen.add(real)
            try:
                weights += os.path.getsize(real)
            except OSError:
                continue
    config = _read_json(os.path.join(chosen, "config.json"))
    return {
        "snapshotPath": chosen,
        "revision": os.path.basename(chosen),
        "revisionMatchesRequest": (revision is None
                                   or os.path.basename(chosen) == revision),
        "weightsBytes": int(weights),
        "config": config if isinstance(config, dict) else None,
    }


# --- evidence bundles ---------------------------------------------------------------

def scan_evidence_bundles(profile: ServerProfile | None = None,
                          jobs=None,
                          roots: StorageRoots | None = None) -> list[dict]:
    """Evidence bundles under the runs tree, with the owning job/run where the
    job store knows them (the reconciler folds ``evidenceBundle`` into results)."""
    profile = profile or ServerProfile.from_env()
    roots = roots or StorageRoots.from_profile(profile)
    by_bundle_path: dict[str, str] = {}
    by_run_dir: dict[str, str] = {}
    if jobs is not None:
        for job in jobs.list():
            result = job.result or {}
            evidence = result.get("evidenceBundle")
            if isinstance(evidence, dict) and evidence.get("bundlePath"):
                by_bundle_path[os.path.realpath(evidence["bundlePath"])] = job.id
            run_dir = result.get("runDirectory")
            if isinstance(run_dir, str) and run_dir:
                by_run_dir[os.path.realpath(run_dir)] = job.id
    out: list[dict] = []
    cap = _env_int("STEERLAB_HOUSEKEEPING_SCAN_CAP", SCAN_FILE_CAP_DEFAULT)
    scanned = 0
    if os.path.isdir(roots.runs):
        for dirpath, _dirs, filenames in os.walk(roots.runs):
            for name in filenames:
                scanned += 1
                if scanned > cap:
                    break
                if not name.endswith(EVIDENCE_SUFFIX):
                    continue
                path = os.path.join(dirpath, name)
                real = os.path.realpath(path)
                try:
                    st = os.stat(real)
                except OSError:
                    continue
                job_id = (by_bundle_path.get(real)
                          or by_run_dir.get(os.path.realpath(dirpath)))
                out.append({
                    "jobId": job_id,
                    "runId": name[:-len(EVIDENCE_SUFFIX)],
                    "path": real,
                    "sizeBytes": int(st.st_size),
                    "createdAt": _iso(st.st_mtime),
                })
            if scanned > cap:
                break
    out.sort(key=lambda item: item["createdAt"], reverse=True)
    return out


# --- throughput table ------------------------------------------------------------

def _throughput_path(profile: ServerProfile | None = None) -> str:
    profile = profile or ServerProfile.from_env()
    return os.path.join(profile.metadata_root, THROUGHPUT_FILENAME)


def read_throughput(profile: ServerProfile | None = None) -> dict:
    data = _read_json(_throughput_path(profile))
    if not isinstance(data, dict):
        return {"schemaVersion": 1, "entries": [], "foldedJobIds": []}
    entries = [e for e in data.get("entries", []) if isinstance(e, dict)]
    folded = [str(j) for j in data.get("foldedJobIds", [])]
    return {"schemaVersion": 1, "entries": entries, "foldedJobIds": folded}


def parse_gpu_type(gres: str | None) -> str | None:
    """``gpu:A100:1`` / ``gpu:A100`` / ``A100`` → ``A100``; None when absent."""
    if not gres or not str(gres).strip():
        return None
    value = str(gres).strip()
    if value.startswith("gpu:"):
        parts = value.split(":")
        return parts[1] or None if len(parts) > 1 else None
    return value


def throughput_lookup(model_id: str, gpu_type: str | None,
                      profile: ServerProfile | None = None,
                      instrument_family: str | None = None) -> dict | None:
    """The (model, GPU) throughput entry, optionally narrowed to ONE
    instrument family (open issues §7).

    ``instrument_family=None`` returns the GLOBAL entry — every folded job
    for that model+GPU regardless of family, which is what every historical
    caller means and what a table written before families existed holds.
    A family that has never been folded returns None, and the caller is
    expected to fall back to the global entry and SAY it fell back.
    """
    for entry in read_throughput(profile)["entries"]:
        if (entry.get("modelId") == model_id
                and entry.get("gpuType") == gpu_type
                and entry.get("instrumentFamily") == instrument_family):
            return entry
    return None


def fold_throughput(jobs, profile: ServerProfile | None = None) -> dict:
    """Fold terminal job records carrying ``elapsedSeconds`` + ``recordCount``
    into the per-(modelId, gpuType) records-per-hour table (cumulative mean).

    ``modelId`` comes from the submission-time stamp in the job's requested
    resources (read from the actual manifest/bundle) — when a record has no
    stamp it is SKIPPED, never guessed. ``gpuType`` parses from the requested
    gres; local-executor records fold under ``gpuType: null``.

    Since 2026-08-20 (open issues §7) a job ALSO folds into its instrument
    family's own entry when its preflight recorded one — the global entry
    keeps its historical meaning (everything), and the family entry is what
    the estimator prefers. The family entry is additive: an older reader
    that ignores ``instrumentFamily`` still finds the global entry first,
    because the sort puts the family-less entry ahead of its siblings.

    A token-bounded family entry also carries a ``tokensBasis`` — the
    ``maxTokens`` its first stamped sample generated under — and later
    stamped samples are normalized to that basis before entering the mean
    (records-per-hour is taken as inverse-linear in generated tokens).
    The global entry never carries a basis: it keeps its historical meaning.
    """
    from .jobs import TERMINAL  # local import: jobs.py imports us lazily
    from .instrument_family import stamped_family, stamped_max_tokens
    profile = profile or ServerProfile.from_env()
    with _IO_LOCK:
        table = read_throughput(profile)
        folded = set(table["foldedJobIds"])
        entries: list[dict] = table["entries"]
        changed = False
        for job in jobs.list():
            if job.id in folded or job.status not in TERMINAL:
                continue
            result = job.result or {}
            elapsed = result.get("elapsedSeconds")
            count = result.get("recordCount")
            model_id = (job.requested_resources or {}).get("modelID")
            if (not isinstance(elapsed, (int, float)) or elapsed <= 0
                    or not isinstance(count, int) or count <= 0
                    or not model_id):
                continue
            gpu_type = parse_gpu_type(
                (job.requested_resources or {}).get("gres"))
            rate = count / (elapsed / 3600.0)
            family = stamped_family(job.requested_resources)
            tokens = stamped_max_tokens(job.requested_resources)
            for bucket in (None, family) if family else (None,):
                entry = next((e for e in entries
                              if e.get("modelId") == model_id
                              and e.get("gpuType") == gpu_type
                              and e.get("instrumentFamily") == bucket), None)
                if entry is None:
                    minted = {
                        "modelId": model_id,
                        "gpuType": gpu_type,
                        "recordsPerHour": round(rate, 3),
                        "samples": 1,
                        "updatedAt": _now_iso(),
                    }
                    if bucket is not None:
                        minted["instrumentFamily"] = bucket
                        if tokens:
                            minted["tokensBasis"] = tokens
                    entries.append(minted)
                else:
                    samples = int(entry.get("samples", 0))
                    mean = float(entry.get("recordsPerHour", 0.0))
                    basis = entry.get("tokensBasis")
                    fold_rate = rate
                    if (bucket is not None and tokens
                            and isinstance(basis, (int, float)) and basis > 0):
                        # A rate measured at 2048 tokens is ~8× slower than the
                        # same hardware's 256-token rate; normalize each sample
                        # to the entry's basis or the mean means nothing. An
                        # entry minted before bases existed stays basisless and
                        # keeps folding raw rates — the estimator then says the
                        # token budget it is assuming instead of scaling.
                        fold_rate = rate * tokens / basis
                    entry["recordsPerHour"] = round(
                        (mean * samples + fold_rate) / (samples + 1), 3)
                    entry["samples"] = samples + 1
                    entry["updatedAt"] = _now_iso()
            folded.add(job.id)
            changed = True
        if changed:
            table["entries"] = sorted(
                entries, key=lambda e: (e.get("modelId") or "",
                                        e.get("gpuType") or "",
                                        e.get("instrumentFamily") or ""))
            table["foldedJobIds"] = list(folded)[-_FOLDED_JOB_CAP:]
            _atomic_write_json(_throughput_path(profile), table)
        return table


# --- snapshot + status -----------------------------------------------------------

def _snapshot_path(profile: ServerProfile | None = None) -> str:
    profile = profile or ServerProfile.from_env()
    return os.path.join(profile.metadata_root, SNAPSHOT_FILENAME)


def refresh(jobs=None, profile: ServerProfile | None = None) -> dict:
    """Full housekeeping rescan: fold throughput, run the expensive scans,
    persist the snapshot, and return the full status body."""
    profile = profile or ServerProfile.from_env()
    roots = StorageRoots.from_profile(profile)
    if jobs is not None:
        try:
            fold_throughput(jobs, profile)
        except Exception:  # noqa: BLE001 - a bad record must not kill the scan
            pass
    snapshot = {
        "generatedAt": _now_iso(),
        "roots": disk_roots(profile),
        # Expensive (a subprocess against a parallel filesystem), so it belongs
        # to the tick and is served from the snapshot by status(), like the
        # tree walks beside it.
        "quota": quota_report(profile),
        "purgeRisk": scan_purge_risk(profile, roots),
        "hfCache": scan_hf_cache(),
        "maintenance": maintenance_status(profile),
        "evidence": {"bundles": scan_evidence_bundles(profile, jobs, roots)},
        "throughput": {"entries": read_throughput(profile)["entries"]},
    }
    with _IO_LOCK:
        try:
            _atomic_write_json(_snapshot_path(profile), snapshot)
        except OSError:
            pass  # serving beats persisting; next tick retries
    return snapshot


def status(profile: ServerProfile | None = None) -> dict:
    """Cheap status: the last persisted snapshot's expensive scans (purge risk,
    HF cache, evidence, the site quota command's output — stale-but-honest,
    stamped with the snapshot's ``generatedAt``) overlaid with live df stats,
    the live maintenance calendar, and the live throughput table. Never scans
    trees and never runs the quota command."""
    profile = profile or ServerProfile.from_env()
    snapshot = _read_json(_snapshot_path(profile))
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    return {
        "generatedAt": snapshot.get("generatedAt") or _now_iso(),
        "roots": disk_roots(profile),
        "quota": snapshot.get("quota"),
        "purgeRisk": snapshot.get("purgeRisk"),
        "hfCache": snapshot.get("hfCache"),
        "maintenance": maintenance_status(profile),
        "evidence": snapshot.get("evidence") or {"bundles": []},
        "throughput": {"entries": read_throughput(profile)["entries"]},
    }
