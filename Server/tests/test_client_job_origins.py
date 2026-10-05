"""Remote job origins recorded in the workspace (``client.job_origins``).

The Mac app reads this record before its own preferences, so a job a command
line submitted is importable there without a reconnect. These tests pin the
shared layout, the merge rule, the never-destroy rule, and that writers in
separate processes do not corrupt the file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from steerlab_server.client import job_origins


def _document(root) -> dict:
    with open(job_origins.file_path(str(root)), encoding="utf-8") as handle:
        return json.load(handle)


def test_record_writes_the_shared_layout_and_keeps_it_out_of_git(tmp_path):
    row = job_origins.record(
        str(tmp_path), job_id="job-1", endpoint="http://127.0.0.1:8080",
        serving_root="/srv/runner", experiment="study-a", verb="run",
        operation="submit-bundle", now=0)
    document = _document(tmp_path)
    assert document["schemaVersion"] == 1
    assert document["jobs"] == {"job-1": [row]}
    assert row == {
        "serverIdentity": "http://127.0.0.1:8080",
        "endpoint": "http://127.0.0.1:8080",
        "servingRoot": "/srv/runner",
        "workspaceRoot": os.path.realpath(str(tmp_path)),
        "submittedBy": "steerlab",
        "recordedAt": "1970-01-01T00:00:00Z",
        "experiment": "study-a",
        "verb": "run",
        "operation": "submit-bundle",
    }, "an unknown field is absent rather than null, as the Mac writer encodes it"
    folder = job_origins.directory(str(tmp_path))
    with open(os.path.join(folder, ".gitignore"), encoding="utf-8") as handle:
        assert handle.read() == "*\n"
    assert job_origins.origins_for(str(tmp_path), "job-1") == [row]
    assert job_origins.origins_for(str(tmp_path), "absent") == []


def test_server_identity_matches_the_mac_registry_normalization():
    # ClusterConnectionStore.normalizedEndpointKey, case by case.
    assert job_origins.server_identity("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert job_origins.server_identity("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    assert job_origins.server_identity("HTTP://Runner.Example.ORG") == "http://runner.example.org:80"
    assert job_origins.server_identity("https://runner.example.org") == "https://runner.example.org:443"
    assert job_origins.server_identity("127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_the_same_job_on_the_same_server_is_replaced_and_another_server_is_kept(tmp_path):
    root = str(tmp_path)
    job_origins.record(root, job_id="shared", endpoint="http://first.invalid", verb="run")
    job_origins.record(root, job_id="shared", endpoint="http://first.invalid", verb="validate")
    rows = job_origins.origins_for(root, "shared")
    assert [row["verb"] for row in rows] == ["validate"]
    # Job IDs are unique only per server: a second server's job with the same
    # ID is kept BESIDE the first, so a reader sees the ambiguity.
    job_origins.record(root, job_id="shared", endpoint="http://second.invalid")
    identities = [row["serverIdentity"] for row in job_origins.origins_for(root, "shared")]
    assert identities == ["http://first.invalid:80", "http://second.invalid:80"]


def test_rows_a_newer_client_wrote_survive_a_write(tmp_path):
    root = str(tmp_path)
    os.makedirs(job_origins.directory(root))
    future = {"serverIdentity": "http://a.invalid:80", "endpoint": "http://a.invalid",
              "workspaceRoot": root, "submittedBy": "steerlab", "recordedAt": "x",
              "futureField": {"kept": True}}
    with open(job_origins.file_path(root), "w", encoding="utf-8") as handle:
        json.dump({"schemaVersion": 2, "futureTopLevel": 1,
                   "jobs": {"old": [future, "not-a-row"]}}, handle)
    job_origins.record(root, job_id="new", endpoint="http://b.invalid")
    document = _document(root)
    assert document["schemaVersion"] == 2, "a writer never downgrades the version"
    assert document["futureTopLevel"] == 1
    assert document["jobs"]["old"] == [future, "not-a-row"]
    assert [row["serverIdentity"] for row in document["jobs"]["new"]] == ["http://b.invalid:80"]


def test_an_unreadable_record_is_set_aside_never_destroyed(tmp_path):
    root = str(tmp_path)
    os.makedirs(job_origins.directory(root))
    with open(job_origins.file_path(root), "w", encoding="utf-8") as handle:
        handle.write('{"jobs": {"half-written"')
    assert job_origins.load(root) == {}, "an unreadable record reads as empty"
    job_origins.record(root, job_id="after", endpoint="http://a.invalid")
    assert list(job_origins.load(root)) == ["after"]
    aside = [name for name in os.listdir(job_origins.directory(root))
             if name.startswith("origins.json.unreadable-")]
    assert len(aside) == 1
    with open(os.path.join(job_origins.directory(root), aside[0]), encoding="utf-8") as handle:
        assert handle.read() == '{"jobs": {"half-written"'


def test_a_failed_record_is_a_warning_and_no_workspace_records_nothing(tmp_path):
    warnings = []
    assert job_origins.record_quietly(None, job_id="job", endpoint="http://a.invalid",
                                      warn=warnings.append) is None
    blocker = tmp_path / "file-not-folder"
    blocker.write_text("x")
    assert job_origins.record_quietly(str(blocker), job_id="job", endpoint="http://a.invalid",
                                      warn=warnings.append) is None
    assert len(warnings) == 1
    assert "job job was submitted" in warnings[0]
    assert "reconnect" in warnings[0]


def test_concurrent_writers_in_separate_processes_never_corrupt_the_file(tmp_path):
    """Four processes each record 25 jobs at once. Every record survives —
    no torn file, and no read-modify-write that lost another writer's rows."""
    root = str(tmp_path)
    package = os.path.dirname(os.path.dirname(os.path.abspath(job_origins.__file__)))
    script = textwrap.dedent("""
        import sys
        from steerlab_server.client import job_origins
        root, writer = sys.argv[1], sys.argv[2]
        for index in range(25):
            job_origins.record(root, job_id=f"{writer}-{index}",
                               endpoint="http://127.0.0.1:8080", verb="run")
    """)
    # The checkout's sources, not whatever an editable install points at.
    environment = {**os.environ, "PYTHONPATH": os.path.dirname(package)}
    processes = [subprocess.Popen([sys.executable, "-c", script, root, f"w{writer}"],
                                  env=environment, stderr=subprocess.PIPE)
                 for writer in range(4)]
    for process in processes:
        _, errors = process.communicate(timeout=120)
        assert process.returncode == 0, errors.decode()
    document = _document(root)
    assert len(document["jobs"]) == 100
    assert all(len(rows) == 1 for rows in document["jobs"].values())
    leftovers = [name for name in os.listdir(job_origins.directory(root))
                 if name.startswith(".origins-")]
    assert leftovers == [], "no staging file is left behind"
