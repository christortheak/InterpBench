"""Model install: the one verb that is ALLOWED to go online.

Cluster profiles run the server with ``HF_HUB_OFFLINE=1`` so study runs stay
hermetic (a run must never silently pull a different snapshot mid-study). But
``huggingface_hub`` freezes that flag into module constants at import time, so
the install verb cannot simply flip ``os.environ`` — it downloads in a CHILD
process whose environment forces ``HF_HUB_OFFLINE=0``. Division of labor:
install fetches online; everything else (loads, runs) resolves offline from
the shared cache, which doubles as the completeness verifier.

The child emits one JSON record per line (``{"log": …}`` progress, then a
final ``{"ok": …}`` outcome), so the parent can stream job logs and map
failures to actionable remedies (token missing vs license not accepted vs no
egress) instead of surfacing raw hub tracebacks. A watcher thread in the
child reports throttled byte progress (cache-side blob sizes vs the
estimate), and the parent polls the job's cooperative cancel flag —
cancellation terminates the child's process group instead of letting a 30 GB
download run to completion.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

CHILD_SOURCE = """
import fnmatch, json, os, sys, threading, time

model_id = sys.argv[1]
revision = sys.argv[2] or None
# Repo-relative glob scoping. Empty = whole repo (every model install). Present
# for artifact repos that hold many models in one tree: neuronpedia/jacobian-lens
# is ~57 GB across 36 models, of which one lens is a few GB, so an unscoped
# fetch there is not a slow success — it is a filled group quota.
allow_patterns = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

def emit(record):
    print(json.dumps(record), flush=True)

def watch_progress(estimate_bytes):
    # Byte progress without touching hub internals: sum the repo's blobs
    # (finished + *.incomplete) in the cache every interval. One repo's blob
    # dir stays small in file COUNT even when huge in bytes, so the walk is
    # cheap. The number is CACHE OCCUPANCY, not this job's transfer — say so:
    # for a resumed install "cache holds 60%" is exactly the honest figure,
    # and blobs shared with other revisions inflate transfer-framing.
    from huggingface_hub.constants import HF_HUB_CACHE
    repo_dir = os.path.join(
        HF_HUB_CACHE, "models--" + model_id.replace("/", "--"), "blobs")
    interval = float(os.environ.get("STEERLAB_INSTALL_PROGRESS_SECONDS", "15"))
    while True:
        time.sleep(max(interval, 0.01))
        done = 0
        try:
            for name in os.listdir(repo_dir):
                try:
                    done += os.path.getsize(os.path.join(repo_dir, name))
                except OSError:
                    continue
        except OSError:
            continue
        if done <= 0:
            continue
        if estimate_bytes:
            emit({"log": "cache holds %.1f of ~%.1f GB (%d%%)" % (
                done / 1e9, estimate_bytes / 1e9,
                min(100, round(100 * done / estimate_bytes)))})
        else:
            emit({"log": "cache holds %.1f GB" % (done / 1e9)})

try:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    estimate = 0
    try:  # best-effort size estimate before committing the disk
        info = HfApi().model_info(model_id, revision=revision,
                                  files_metadata=True)
        siblings = info.siblings or []
        if allow_patterns:
            # Estimate what we will ACTUALLY fetch. Summing every sibling under
            # a scoped download reports the whole repo — "cache holds 0.4 of
            # ~57 GB (1%)" for a download that is nearly done.
            siblings = [f for f in siblings
                        if any(fnmatch.fnmatch(f.rfilename, p) for p in allow_patterns)]
        estimate = sum(f.size or 0 for f in siblings)
        if estimate:
            emit({"log": "estimated download: %.1f GB%s" % (
                estimate / 1e9, " (scoped)" if allow_patterns else "")})
    except Exception:  # estimate only, never blocks
        pass
    threading.Thread(target=watch_progress, args=(estimate,),
                     daemon=True).start()
    # Only pass the kwarg when scoping is actually requested: every ordinary
    # model install goes through this call, and widening its signature for a
    # feature none of them use is blast radius for nothing.
    scope = {"allow_patterns": allow_patterns} if allow_patterns else {}
    path = snapshot_download(repo_id=model_id, revision=revision, **scope)
    try:
        # Atomic completion marker (engineer review 2026-07-18): the hub
        # writes refs/main BEFORE the snapshot files, so a size scan that
        # fingerprints refs alone can cache a partial download forever.
        # The marker lands only after snapshot_download returns.
        #
        # NOT written for a scoped fetch: the marker asserts "this repo is
        # fully downloaded", which a scoped fetch never makes true. Claiming it
        # would make a one-folder slice of a 36-model artifact repo look
        # complete, and a later fetch of a different folder would trust that.
        # The sizes TTL already covers partial trees.
        if not allow_patterns:
            repo_dir = os.path.join(
                HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
            marker = os.path.join(repo_dir, ".steerlab-install-complete")
            tmp_marker = marker + ".tmp"
            with open(tmp_marker, "w") as handle:
                handle.write(path)
            os.replace(tmp_marker, marker)
    except Exception:
        pass  # marker is best-effort; the sizes TTL is the backstop
    emit({"ok": True, "path": path})
except Exception as exc:
    emit({"ok": False, "kind": type(exc).__name__, "detail": str(exc)})
    sys.exit(1)
"""

_NETWORK_KINDS = {
    "ConnectionError", "ConnectTimeout", "ConnectTimeoutError", "NewConnectionError",
    "MaxRetryError", "ProxyError", "ReadTimeout", "SSLError", "Timeout",
}


def install_env(base: dict | None = None) -> dict:
    """The child's environment: the parent's (HF_HOME, token path, proxies)
    with offline mode forced OFF and progress bars silenced (tqdm would flood
    the job log at one line per chunk)."""
    env = dict(os.environ if base is None else base)
    env["HF_HUB_OFFLINE"] = "0"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return env


def hf_token_status(env: dict | None = None) -> tuple[bool, str]:
    """``(present, where)`` for the token the install child would use,
    resolved against the CHILD's effective environment (pass the same dict
    given to ``run_install``; defaults to this process's) — engineer review
    2026-07-18: diagnostics that inspect the parent env can lie about the
    child. Resolution mirrors huggingface_hub: HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN env, else ``HF_TOKEN_PATH``, else
    ``$HF_HOME/token`` — always reported as a REAL path (the literal
    ``$HF_HOME`` in an error message hid that the token file did not move
    with a relocated cache)."""
    lookup = os.environ if env is None else env
    if lookup.get("HF_TOKEN") or lookup.get("HUGGING_FACE_HUB_TOKEN"):
        return True, "the HF_TOKEN environment variable"
    path = lookup.get("HF_TOKEN_PATH")
    if not path:
        home = lookup.get("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface")
        path = os.path.join(home, "token")
    try:
        with open(path, encoding="utf-8") as handle:
            present = bool(handle.read().strip())
    except OSError:
        present = False
    return present, path


def friendly_failure(kind: str, detail: str, model_id: str,
                     env: dict | None = None) -> str:
    """Map a hub exception to the remedy the researcher can actually take.
    ``env`` is the child's effective environment when the caller has one."""
    lowered = f"{kind} {detail}".lower()
    if kind == "GatedRepoError" or "gated repo" in lowered or "403" in detail:
        token_present, where = hf_token_status(env)
        if not token_present:
            return (
                f"{model_id} is gated and NO Hugging Face token is installed "
                f"— expected one at {where}. If the HF cache was recently "
                "moved, the token file did not move with it: re-run the "
                "app's Install HF Token…, or copy the old cache's `token` "
                "file next to the new one, then retry")
        return (
            f"{model_id} is gated and the token has no access — visit "
            f"https://huggingface.co/{model_id} with the account that owns "
            f"the token at {where} and accept the license, then retry")
    if kind == "RepositoryNotFoundError" or "401" in detail:
        token_present, where = hf_token_status(env)
        auth_hint = (
            f"install a Hugging Face READ token (expected at {where}) first"
            if not token_present else
            f"the token at {where} may lack access to it")
        return (
            f"{model_id} was not found or needs authentication — check the "
            f"id's spelling; for gated/private repos, {auth_hint}")
    if kind in ("LocalEntryNotFoundError", "OfflineModeIsEnabled"):
        return (
            "the snapshot is incomplete locally and the installer could not go "
            "online even though installs force HF_HUB_OFFLINE=0 — " + detail)
    if kind in _NETWORK_KINDS or "connection" in lowered or "name resolution" in lowered:
        return (
            "cannot reach huggingface.co from this node — no compute egress? "
            "Stage the model from a transfer/xfer host into the shared HF "
            "cache instead. (" + detail + ")")
    if "hex hash value" in lowered:
        # hf_xet chokes on stale partial-download state (live 2026-07-18,
        # right after failed unauthenticated attempts left debris).
        # run_install already retried once over plain HTTP to get here.
        return (
            f"the Xet download backend failed parsing a hash while fetching "
            f"{model_id} — usually stale partial-download state. Clear it "
            "and retry: rm -rf $HF_HOME/xet, and delete *.incomplete files "
            "under the repo's blobs/ directory; HF_HUB_DISABLE_XET=1 in the "
            "site env forces plain-HTTP downloads permanently")
    return f"{kind}: {detail}"


def _terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM the child's whole process group (hub downloads spawn worker
    processes), escalating to SIGKILL if it lingers."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait()


def run_install(
    model_id: str,
    revision: str | None,
    log,
    *,
    cancelled=None,
    python: str | None = None,
    child_source: str | None = None,
    env: dict | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """`_run_install_once` plus ONE self-healing retry: when the Xet
    download backend fails parsing a hash (stale partial-download state —
    live 2026-07-18, right after unauthenticated attempts left debris), the
    retry runs with ``HF_HUB_DISABLE_XET=1`` (plain HTTP). A second failure
    surfaces the cleanup remedy via ``friendly_failure``."""
    try:
        return _run_install_once(
            model_id, revision, log, cancelled=cancelled, python=python,
            child_source=child_source, env=env, allow_patterns=allow_patterns)
    except RuntimeError as exc:
        message = str(exc).lower()
        # Both spellings: the raw hub detail ("hex hash value") and the
        # friendly_failure mapping of it ("xet download backend").
        if "hex hash value" not in message and "xet download backend" not in message:
            raise
        log("the Xet download backend failed parsing a hash (usually stale "
            "partial state) — retrying once over plain HTTP "
            "(HF_HUB_DISABLE_XET=1)")
        retry_env = dict(env if env is not None else install_env())
        retry_env["HF_HUB_DISABLE_XET"] = "1"
        return _run_install_once(
            model_id, revision, log, cancelled=cancelled, python=python,
            child_source=child_source, env=retry_env,
            allow_patterns=allow_patterns)


def _run_install_once(
    model_id: str,
    revision: str | None,
    log,
    *,
    cancelled=None,
    python: str | None = None,
    child_source: str | None = None,
    env: dict | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Run the online child to completion, streaming its progress into
    ``log``, and return the local snapshot path. Raises ``RuntimeError`` with
    an actionable message on any failure.

    ``cancelled`` is polled between output reads (pass ``lambda:
    job.cancelled``): a cooperative job-cancel flag alone would leave a 30 GB
    download running to completion, so observing it TERMINATES the child's
    process group before raising. The other keyword seams exist for tests
    (inject a fake child) — production callers pass none of them."""
    child_env = env if env is not None else install_env()
    token_present, where = hf_token_status(child_env)
    if not token_present:
        # Say it BEFORE the download attempt: public repos still install
        # unauthenticated, but gated ones (google/gemma-*) will fail — and
        # the researcher should learn about a missing token from this line,
        # not from a misleading access error minutes later.
        log(f"WARNING: no Hugging Face token found (expected at {where}) — "
            "gated repos will fail; use the app's Install HF Token… if this "
            "install needs one")
    argv = [python or sys.executable, "-c", child_source or CHILD_SOURCE,
            model_id, revision or "",
            json.dumps(allow_patterns) if allow_patterns else ""]
    # stderr merges into stdout: stray hub warnings become plain log lines and
    # a single stream cannot deadlock the pipe buffers. start_new_session
    # gives the child its own process group so cancellation can kill the whole
    # download tree, not just the top process.
    proc = subprocess.Popen(
        argv, env=env if env is not None else install_env(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True)
    assert proc.stdout is not None
    lines: queue.SimpleQueue = queue.SimpleQueue()

    def read_stdout(stream, sink):
        for raw in stream:
            sink.put(raw)
        sink.put(None)  # EOF marker

    reader = threading.Thread(
        target=read_stdout, args=(proc.stdout, lines), daemon=True)
    reader.start()

    outcome: dict | None = None
    eof = False
    while not eof:
        if cancelled is not None and cancelled():
            log("cancel observed — terminating the download")
            _terminate_group(proc)
            raise RuntimeError(
                f"install of {model_id} cancelled — partial blobs stay in the "
                "cache and a re-run resumes from them")
        try:
            raw = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if raw is None:
            eof = True
            continue
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            log(line)
            continue
        if not isinstance(record, dict):
            log(line)
        elif "ok" in record:
            outcome = record
        elif "log" in record:
            log(str(record["log"]))
        else:
            log(line)
    code = proc.wait()
    if outcome is None:
        raise RuntimeError(
            f"installer exited with code {code} before reporting a result — "
            "see the job log above")
    if outcome.get("ok"):
        path = str(outcome.get("path", ""))
        log(f"installed → {path}")
        return path
    raise RuntimeError(friendly_failure(
        str(outcome.get("kind", "UnknownError")),
        str(outcome.get("detail", "")), model_id,
        env=env if env is not None else install_env()))
