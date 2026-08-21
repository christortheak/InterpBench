"""Runtime workspace switching (POST /api/workspace/switch).

The serving root flows through ``paths.project_root()`` (the STEERLAB_ROOT
env, read per call), so the switch is an env flip — these tests prove the
flip actually re-scopes the artifact routes, that creation seeds the same
skeleton the app seeds, and that the containment / busy-job / privilege
refusals hold.
"""

import json
import os
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from steerlab_server.api.app import app, state

client = TestClient(app)


def _make_workspace(base, name, concept=None):
    root = base / name
    for sub in ("prompts", "experiments", "runs"):
        (root / sub).mkdir(parents=True)
    if concept:
        cdir = root / "prompts" / "concepts" / concept
        cdir.mkdir(parents=True)
        (cdir / "positive.jsonl").write_text('{"text": "yay"}\n', encoding="utf-8")
        (cdir / "negative.jsonl").write_text('{"text": "meh"}\n', encoding="utf-8")
    return root


def test_switch_repoints_subsequent_reads(tmp_path, monkeypatch):
    """The happy path: after a switch, catalog routes list the NEW tree's
    content — proven with two real workspaces holding different artifacts."""
    ws1 = _make_workspace(tmp_path, "ws1", concept="joy")
    ws2 = _make_workspace(tmp_path, "ws2", concept="sorrow")
    (ws2 / "experiments" / "demo").mkdir()
    (ws2 / "experiments" / "demo" / "experiment.json").write_text(
        json.dumps({"name": "demo", "modelID": "org/m", "status": "draft",
                    "conditions": []}), encoding="utf-8")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws1))

    names = [c["name"] for c in client.get("/api/concepts").json()["concepts"]]
    assert "joy" in names and "sorrow" not in names

    resp = client.post("/api/workspace/switch", json={"root": str(ws2)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["switched"] is True
    assert body["root"] == os.path.realpath(str(ws2))
    # The response carries the same payload /api/info gives.
    assert body["service"] == "steerlab-server"
    assert "capabilities" in body

    assert client.get("/api/info").json()["root"] == os.path.realpath(str(ws2))
    names = [c["name"] for c in client.get("/api/concepts").json()["concepts"]]
    assert "sorrow" in names and "joy" not in names
    experiments = client.get("/api/experiments").json()["experiments"]
    assert any(e["name"] == "demo" for e in experiments)


def test_switch_creates_and_seeds_a_new_workspace(tmp_path, monkeypatch):
    ws1 = _make_workspace(tmp_path, "ws1")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws1))
    target = tmp_path / "fresh-workspace"
    assert not target.exists()

    resp = client.post("/api/workspace/switch", json={"root": str(target)})
    assert resp.status_code == 200
    for sub in ("prompts", "experiments", "runs"):
        assert (target / sub).is_dir()
    # The marker app-created workspaces carry, so the new tree is never
    # mistaken for a source checkout.
    assert (target / "WORKSPACE.md").is_file()
    assert resp.json()["rootLooksLikeSourceCheckout"] is False


def test_switch_never_seeds_an_existing_directory(tmp_path, monkeypatch):
    """Pointing at an existing tree serves it AS-IS: no WORKSPACE.md is
    written (stamping a source checkout would defeat the pairing warning)."""
    ws1 = _make_workspace(tmp_path, "ws1")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws1))
    checkoutish = tmp_path / "checkout"
    checkoutish.mkdir()
    (checkoutish / "Package.swift").write_text("x", encoding="utf-8")

    resp = client.post("/api/workspace/switch", json={"root": str(checkoutish)})
    assert resp.status_code == 200
    assert not (checkoutish / "WORKSPACE.md").exists()
    assert resp.json()["rootLooksLikeSourceCheckout"] is True


def test_switch_requires_absolute_path_and_existing_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    assert client.post("/api/workspace/switch",
                       json={"root": "relative/path"}).status_code == 400
    assert client.post("/api/workspace/switch", json={}).status_code == 400
    deep = tmp_path / "no" / "such" / "parents" / "ws"
    resp = client.post("/api/workspace/switch", json={"root": str(deep)})
    assert resp.status_code == 400
    assert "parent" in resp.json()["detail"]
    afile = tmp_path / "afile"
    afile.write_text("x", encoding="utf-8")
    assert client.post("/api/workspace/switch",
                       json={"root": str(afile)}).status_code == 400


def test_parent_allowlist_contains_switches(tmp_path, monkeypatch):
    inside = _make_workspace(tmp_path / "allowed", "ws-in")
    outside = _make_workspace(tmp_path / "elsewhere", "ws-out")
    monkeypatch.setenv("STEERLAB_ROOT", str(inside))
    monkeypatch.setenv("STEERLAB_WORKSPACE_PARENT", str(tmp_path / "allowed"))

    refused = client.post("/api/workspace/switch", json={"root": str(outside)})
    assert refused.status_code == 403
    assert "STEERLAB_WORKSPACE_PARENT" in refused.json()["detail"]
    # Traversal inside the parent does not escape it (realpath containment).
    sneaky = str(tmp_path / "allowed" / ".." / "elsewhere" / "ws-out")
    assert client.post("/api/workspace/switch",
                       json={"root": sneaky}).status_code == 403

    allowed = client.post(
        "/api/workspace/switch", json={"root": str(inside.parent / "ws-two")})
    assert allowed.status_code == 200
    assert allowed.json()["root"] == os.path.realpath(
        str(inside.parent / "ws-two"))


def test_non_loopback_bind_refuses_without_parent(tmp_path, monkeypatch):
    """A non-loopback server without the parent allowlist refuses even with a
    valid token: on shared nodes a repoint is a write-anywhere primitive."""
    ws = _make_workspace(tmp_path, "ws")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws))
    monkeypatch.setenv("STEERLAB_BIND", "0.0.0.0")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    monkeypatch.delenv("STEERLAB_WORKSPACE_PARENT", raising=False)

    resp = client.post("/api/workspace/switch",
                       json={"root": str(tmp_path / "other")},
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 403
    assert "STEERLAB_WORKSPACE_PARENT" in resp.json()["detail"]

    # The parent allowlist re-enables switching on the same deployment.
    monkeypatch.setenv("STEERLAB_WORKSPACE_PARENT", str(tmp_path))
    resp = client.post("/api/workspace/switch",
                       json={"root": str(tmp_path / "other")},
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 200


def test_cluster_profile_refuses_without_parent(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path, "ws")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws))
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    monkeypatch.delenv("STEERLAB_WORKSPACE_PARENT", raising=False)
    resp = client.post("/api/workspace/switch",
                       json={"root": str(tmp_path / "other")},
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 403


def test_switch_is_privileged(monkeypatch, tmp_path):
    """Same gate as its _PRIVILEGED_PREFIXES peers: on a Slurm deployment the
    route needs a bearer token even at the default auth mode."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    resp = client.post("/api/workspace/switch", json={"root": str(tmp_path)})
    assert resp.status_code == 503
    # The read-only GET stays open.
    assert client.get("/api/workspace").status_code == 200


def test_switch_refused_while_jobs_are_running(tmp_path, monkeypatch):
    ws1 = _make_workspace(tmp_path, "ws1")
    ws2 = _make_workspace(tmp_path, "ws2")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws1))

    release = threading.Event()
    started = threading.Event()

    def work(job):
        started.set()
        release.wait(timeout=30)
        return {}

    job = state.jobs.submit("test:block", work)
    try:
        assert started.wait(timeout=10)
        resp = client.post("/api/workspace/switch", json={"root": str(ws2)})
        assert resp.status_code == 409
        assert job.id in resp.json()["detail"]
        assert "test:block" in resp.json()["detail"]
    finally:
        release.set()
    for _ in range(100):
        if state.jobs.get(job.id).status == "succeeded":
            break
        time.sleep(0.05)
    assert state.jobs.get(job.id).status == "succeeded"
    assert client.post("/api/workspace/switch",
                       json={"root": str(ws2)}).status_code == 200


def test_metadata_root_is_pinned_across_switches(tmp_path, monkeypatch):
    """When STEERLAB_METADATA_ROOT is unset its default derives from the
    root; the switch pins it to the pre-switch value so metadata (jobs DB,
    housekeeping) stays server-instance-scoped instead of splitting across
    workspaces."""
    from steerlab_server.api.profile import ServerProfile
    ws1 = _make_workspace(tmp_path, "ws1")
    ws2 = _make_workspace(tmp_path, "ws2")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws1))
    monkeypatch.delenv("STEERLAB_METADATA_ROOT", raising=False)
    before = ServerProfile.from_env().metadata_root

    assert client.post("/api/workspace/switch",
                       json={"root": str(ws2)}).status_code == 200
    assert ServerProfile.from_env().metadata_root == before
    assert os.environ["STEERLAB_METADATA_ROOT"] == before


def test_workspace_get_and_capability_flag(tmp_path, monkeypatch):
    ws = _make_workspace(tmp_path, "ws")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws))
    monkeypatch.delenv("STEERLAB_WORKSPACE_PARENT", raising=False)

    body = client.get("/api/workspace").json()
    assert body["root"] == os.path.realpath(str(ws))
    assert body["switchable"] is True
    assert body["parent"] is None

    caps = client.get("/api/capabilities").json()
    assert caps["workspace"]["switch"] is True
    assert caps["workspace"]["switchable"] is True

    monkeypatch.setenv("STEERLAB_WORKSPACE_PARENT", str(tmp_path))
    body = client.get("/api/workspace").json()
    assert body["parent"] == os.path.realpath(str(tmp_path))


# --------------------------------------------------------------------------
# The check-then-switch window (review finding, 2026-08-21)
# --------------------------------------------------------------------------
# The active-job refusal above is only as strong as its atomicity. Before
# ``api/workspace_lock`` the scan and the env mutation were two unsynchronized
# steps, so a submission arriving between them was neither refused nor
# contained: its submission directory and packaged bundle resolved against the
# OLD root and everything after the switch against the NEW one.
#
# Two layers are pinned: the ROUTE holds the exclusive side across scan AND
# mutation (deterministic, no threads — the lock state is read from inside the
# scan itself), and the LOCK delivers the exclusion those holds assume.

from steerlab_server.api import workspace_lock          # noqa: E402
from steerlab_server.api.jobs import JobManager          # noqa: E402


def test_the_active_job_scan_runs_inside_the_exclusive_section(
        tmp_path, monkeypatch):
    """The scan the refusal depends on must see a locked root.

    Read from INSIDE ``JobManager.list`` — the exact call the route makes —
    so this cannot pass by locking somewhere else in the handler.
    """
    ws1 = _make_workspace(tmp_path, "ws1")
    ws2 = _make_workspace(tmp_path, "ws2")
    monkeypatch.setenv("STEERLAB_ROOT", str(ws1))

    seen = []
    original = JobManager.list

    def listing(self):
        lock = workspace_lock.WORKSPACE_ROOT_LOCK
        seen.append({"exclusive": lock._writer,
                     "root": os.environ.get("STEERLAB_ROOT")})
        return original(self)

    monkeypatch.setattr(JobManager, "list", listing)
    assert client.post("/api/workspace/switch",
                       json={"root": str(ws2)}).status_code == 200

    assert seen, "the switch never scanned the job table"
    assert seen[0]["exclusive"] is True, (
        "the active-job scan ran without the exclusive workspace-root lock — "
        "a job submitted between the scan and the env flip would be neither "
        "refused nor contained")
    # …and the scan saw the OLD root: the mutation is after it, in the SAME
    # critical section.
    assert seen[0]["root"] == str(ws1)


def test_job_registration_runs_inside_the_shared_section(tmp_path, monkeypatch):
    """The other half of the pair: a job becomes visible to that scan only
    while the shared side is held, so registration and the switch cannot
    interleave."""
    monkeypatch.setenv("STEERLAB_ROOT", str(_make_workspace(tmp_path, "ws")))
    seen = []
    store = state.jobs.store
    original = type(store).insert

    def insert(self, job):
        seen.append(workspace_lock.WORKSPACE_ROOT_LOCK.readers)
        return original(self, job)

    monkeypatch.setattr(type(store), "insert", insert)
    release = threading.Event()

    def work(job):
        release.wait(timeout=30)
        return {}

    job = state.jobs.submit("test:lockcheck", work)
    release.set()
    assert seen and seen[0] >= 1, (
        "JobManager.submit registered a job without the shared "
        "workspace-root lock")

    seen.clear()
    external = state.jobs.record_external(
        "test:lockcheck-external", status="prepared", executor="local")
    assert seen and seen[0] >= 1, (
        "JobManager.record_external registered a job without the shared "
        "workspace-root lock")
    assert external.id and job.id


def test_a_switch_waits_for_an_in_flight_submission():
    """The exclusion the route's holds assume: a switch cannot enter while a
    submission holds the shared side, and proceeds the moment it lets go."""
    lock = workspace_lock.WorkspaceRootLock()
    submitting = threading.Event()
    switch_entered = threading.Event()
    let_go = threading.Event()

    def submission():
        with lock.submitting():
            submitting.set()
            let_go.wait(timeout=30)

    def switch():
        with lock.switching():
            switch_entered.set()

    holder = threading.Thread(target=submission)
    switcher = threading.Thread(target=switch)
    holder.start()
    assert submitting.wait(timeout=10)
    switcher.start()
    # Can only fail if the lock let a switch in over a live submission.
    assert not switch_entered.wait(timeout=0.25)
    let_go.set()
    assert switch_entered.wait(timeout=10)
    holder.join(timeout=10)
    switcher.join(timeout=10)


def test_a_waiting_switch_holds_off_new_submissions():
    """Writer-preferring: a steady stream of submissions must not starve the
    switch, so a submission arriving after the switch queues behind it."""
    lock = workspace_lock.WorkspaceRootLock()
    holding = threading.Event()
    let_go = threading.Event()
    switch_waiting = threading.Event()
    late_entered = threading.Event()

    def first():
        with lock.submitting():
            holding.set()
            let_go.wait(timeout=30)

    def switch():
        switch_waiting.set()
        with lock.switching():
            pass

    def late():
        with lock.submitting():
            late_entered.set()

    holder = threading.Thread(target=first)
    switcher = threading.Thread(target=switch)
    holder.start()
    assert holding.wait(timeout=10)
    switcher.start()
    assert switch_waiting.wait(timeout=10)
    latecomer = threading.Thread(target=late)
    latecomer.start()
    assert not late_entered.wait(timeout=0.25)
    let_go.set()
    assert late_entered.wait(timeout=10)
    for thread in (holder, switcher, latecomer):
        thread.join(timeout=10)


def test_the_shared_side_is_reentrant_on_one_thread():
    """``submit_study`` holds it for the whole submission and calls
    ``JobManager.submit``, which takes it again — without reentrancy that
    inner acquisition deadlocks as soon as a switch starts waiting."""
    lock = workspace_lock.WorkspaceRootLock()
    with lock.submitting():
        with lock.submitting():
            assert lock.readers == 1
        assert lock.readers == 1
    assert lock.readers == 0
