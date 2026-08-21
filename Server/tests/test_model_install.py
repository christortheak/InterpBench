"""Install-model verb: validation, MLX rejection, and the ONLINE-child
policy — installs force HF_HUB_OFFLINE=0 in a child process while the server
itself stays offline-hermetic, and hub failures map to actionable remedies
(token vs license vs egress)."""

import sys
import time

import pytest

from steerlab_server.api import model_install


# --- the online-child policy ------------------------------------------------

def test_install_env_forces_online_and_preserves_cache_settings():
    base = {"HF_HUB_OFFLINE": "1", "HF_HOME": "/work/lab/hf-cache",
            "https_proxy": "http://proxy:3128"}
    env = model_install.install_env(base)
    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert env["HF_HOME"] == "/work/lab/hf-cache"   # token + cache location survive
    assert env["https_proxy"] == "http://proxy:3128"
    assert base["HF_HUB_OFFLINE"] == "1"            # caller's dict untouched


def test_child_source_is_valid_python():
    compile(model_install.CHILD_SOURCE, "<child>", "exec")


# --- failure → remedy mapping -------------------------------------------------

def test_gated_failure_with_a_token_points_at_the_license_page(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    (tmp_path / "token").write_text("hf_secret", encoding="utf-8")
    message = model_install.friendly_failure(
        "GatedRepoError", "403 Client Error", "google/gemma-3-4b-it")
    assert "https://huggingface.co/google/gemma-3-4b-it" in message
    assert "accept the license" in message
    # Names the REAL token path, never a literal $HF_HOME.
    assert str(tmp_path / "token") in message


def test_gated_failure_without_a_token_names_the_missing_file(
        tmp_path, monkeypatch):
    # Live 2026-07-18: HF_HOME moved to /scratch but the token file stayed
    # at the old location — the error blamed the token's ACCESS when there
    # was no token at all, and the researcher (who did have access) was
    # sent to re-accept a license they had already accepted.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    message = model_install.friendly_failure(
        "GatedRepoError", "403 Client Error", "google/gemma-3-12b-it")
    assert "NO Hugging Face token is installed" in message
    assert str(tmp_path / "token") in message
    assert "recently moved" in message
    assert "Install HF Token" in message


def test_not_found_or_401_points_at_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    message = model_install.friendly_failure(
        "RepositoryNotFoundError", "401 Client Error", "google/gemma3-4b-it")
    assert "spelling" in message
    assert str(tmp_path / "token") in message


def test_token_status_honors_the_child_env_and_hf_token_path(tmp_path):
    # Engineer review 2026-07-18: diagnostics must inspect the CHILD's
    # effective environment, not the parent's — and honor HF_TOKEN_PATH.
    token = tmp_path / "elsewhere" / "tok"
    token.parent.mkdir()
    token.write_text("hf_secret", encoding="utf-8")
    present, where = model_install.hf_token_status(
        {"HF_TOKEN_PATH": str(token)})
    assert present and where == str(token)
    present, where = model_install.hf_token_status(
        {"HF_HOME": str(tmp_path)})
    assert not present and where == str(tmp_path / "token")
    assert model_install.hf_token_status({"HF_TOKEN": "x"}) == (
        True, "the HF_TOKEN environment variable")
    # friendly_failure threads the same env through.
    message = model_install.friendly_failure(
        "GatedRepoError", "403", "org/m", env={"HF_HOME": str(tmp_path)})
    assert "NO Hugging Face token is installed" in message
    assert str(tmp_path / "token") in message


def test_network_failure_points_at_egress_and_staging():
    message = model_install.friendly_failure(
        "ConnectionError", "Failed to establish a new connection", "Qwen/Qwen3-4B")
    assert "egress" in message
    assert "transfer/xfer host" in message


def test_unknown_kinds_pass_through_verbatim():
    assert model_install.friendly_failure("WeirdError", "boom", "a/b") == "WeirdError: boom"


# --- run_install through a real child process ---------------------------------

SUCCESS_CHILD = """
import json, sys
print(json.dumps({"log": "estimated download: 1.2 GB"}), flush=True)
print("stray non-json warning", flush=True)
print(json.dumps({"ok": True, "path": "/fake/cache/" + sys.argv[1]}), flush=True)
"""

GATED_CHILD = """
import json, sys
print(json.dumps({"ok": False, "kind": "GatedRepoError", "detail": "403"}), flush=True)
sys.exit(1)
"""

SILENT_CHILD = "import sys; sys.exit(3)"


def test_run_install_streams_logs_and_returns_the_path():
    lines = []
    path = model_install.run_install(
        "Qwen/Qwen3-4B", "abc", lines.append, child_source=SUCCESS_CHILD)
    assert path == "/fake/cache/Qwen/Qwen3-4B"
    assert "estimated download: 1.2 GB" in lines
    assert "stray non-json warning" in lines           # non-JSON output is logged, not fatal
    assert any("installed" in line for line in lines)


def test_run_install_maps_gated_failure(monkeypatch):
    # A token IS configured here, so the remedy is the license page (the
    # no-token variant is covered by the friendly_failure tests above).
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    with pytest.raises(RuntimeError, match="accept the license"):
        model_install.run_install(
            "google/gemma-3-4b-it", None, lambda _: None, child_source=GATED_CHILD)


XET_FLAKY_CHILD = """
import json, os, sys
if os.environ.get("HF_HUB_DISABLE_XET") == "1":
    print(json.dumps({"ok": True, "path": "/tmp/snap"}))
else:
    print(json.dumps({"ok": False, "kind": "RuntimeError",
                      "detail": "Task error: Unable to parse string as hex hash value"}))
    sys.exit(1)
"""


def test_run_install_retries_xet_hash_failures_over_plain_http():
    # Live 2026-07-18: hf_xet choked on stale partial-download state. The
    # installer self-heals: one retry with HF_HUB_DISABLE_XET=1 (plain
    # HTTP), announced in the log; only a second failure surfaces the
    # cleanup remedy.
    lines: list[str] = []
    path = model_install.run_install(
        "org/m", None, lines.append, child_source=XET_FLAKY_CHILD)
    assert path == "/tmp/snap"
    assert any("HF_HUB_DISABLE_XET=1" in line for line in lines)


XET_ALWAYS_BROKEN_CHILD = """
import json, sys
print(json.dumps({"ok": False, "kind": "RuntimeError",
                  "detail": "Task error: Unable to parse string as hex hash value"}))
sys.exit(1)
"""


def test_run_install_xet_failure_twice_surfaces_the_cleanup_remedy():
    with pytest.raises(RuntimeError, match=r"rm -rf .HF_HOME/xet"):
        model_install.run_install(
            "org/m", None, lambda _: None,
            child_source=XET_ALWAYS_BROKEN_CHILD)


def test_run_install_reports_a_child_that_died_without_a_result():
    with pytest.raises(RuntimeError, match="code 3 before reporting"):
        model_install.run_install(
            "a/b", None, lambda _: None, child_source=SILENT_CHILD)


def test_run_install_cancel_terminates_the_child_instead_of_waiting():
    # A child that would "download" forever: cancellation must kill it and
    # raise promptly, not block on its stdout to completion.
    ENDLESS_CHILD = "import time\nwhile True: time.sleep(0.05)\n"
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        model_install.run_install(
            "google/gemma-3-27b-it", None, lambda _: None,
            cancelled=lambda: True, child_source=ENDLESS_CHILD)
    assert time.monotonic() - started < 10  # SIGTERM path, not walltime


def test_run_install_without_cancel_flag_still_completes():
    path = model_install.run_install(
        "Qwen/Qwen3-4B", None, lambda _: None,
        cancelled=lambda: False, child_source=SUCCESS_CHILD)
    assert path == "/fake/cache/Qwen/Qwen3-4B"


def test_child_reports_throttled_progress_from_cache_blobs(tmp_path, monkeypatch):
    # Real CHILD_SOURCE, fake hub: a stub huggingface_hub package whose
    # snapshot_download writes blobs slowly enough for the watcher thread
    # (interval shrunk via env) to observe and report byte progress.
    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    cache = tmp_path / "hub-cache"
    blobs = cache / "models--org--tiny" / "blobs"
    blobs.mkdir(parents=True)
    (stub / "constants.py").write_text(
        f"HF_HUB_CACHE = {str(cache)!r}\n", encoding="utf-8")
    (stub / "__init__.py").write_text(
        """
import os, time
from . import constants

class HfApi:
    def model_info(self, *a, **k):
        raise RuntimeError("no estimate in this test")

def snapshot_download(repo_id, revision=None):
    blob_dir = os.path.join(
        constants.HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"), "blobs")
    with open(os.path.join(blob_dir, "weight.incomplete"), "wb") as fh:
        fh.write(b"x" * 4096)
    time.sleep(0.4)   # give the watcher a few ticks
    return blob_dir
""", encoding="utf-8")
    env = model_install.install_env()
    env["PYTHONPATH"] = str(tmp_path / "stub")
    env["STEERLAB_INSTALL_PROGRESS_SECONDS"] = "0.05"
    lines = []
    path = model_install.run_install("org/tiny", None, lines.append, env=env)
    assert path.endswith("blobs")
    # Labeled as cache occupancy, not this job's transfer: resumed installs
    # legitimately start high, and shared blobs would inflate a byte-moved
    # framing.
    assert any(line.startswith("cache holds ") for line in lines), lines


def test_run_install_child_env_defaults_to_the_online_policy():
    # The child sees HF_HUB_OFFLINE=0 even when the parent (this test) is
    # offline-hermetic — proven by echoing the child's actual environment.
    ECHO_ENV_CHILD = """
import json, os, sys
print(json.dumps({"ok": True,
                  "path": os.environ.get("HF_HUB_OFFLINE", "unset")}), flush=True)
"""
    import os
    old = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        seen = model_install.run_install(
            "a/b", None, lambda _: None, child_source=ECHO_ENV_CHILD)
    finally:
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old
    assert seen == "0"


# --- the HTTP route ------------------------------------------------------------

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from steerlab_server.api.app import app  # noqa: E402


def test_install_rejects_bad_ids():
    client = TestClient(app)
    assert client.post("/api/models/install",
                       json={"modelID": "../../etc/passwd"}).status_code == 400
    assert client.post("/api/models/install",
                       json={"modelID": "noslash"}).status_code == 400
    response = client.post("/api/models/install",
                           json={"modelID": "mlx-community/gemma-3-4b-it-4bit"})
    assert response.status_code == 400
    assert "family twin" in response.json()["detail"]


def test_install_submits_job_and_runs_the_online_child(monkeypatch):
    monkeypatch.setattr(model_install, "CHILD_SOURCE", SUCCESS_CHILD)
    client = TestClient(app)
    response = client.post("/api/models/install",
                           json={"modelID": "Qwen/Qwen3-4B", "revision": "abc"})
    assert response.status_code == 200
    job_id = response.json()["jobId"]
    job = None
    for _ in range(100):  # daemon-thread job; poll briefly
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert job is not None and job["status"] == "succeeded"
    assert job["result"]["path"] == "/fake/cache/Qwen/Qwen3-4B"
    assert job["result"]["modelID"] == "Qwen/Qwen3-4B"
