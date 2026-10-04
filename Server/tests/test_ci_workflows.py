"""The two GitHub Actions workflows, read as data.

A workflow that does not parse, or that never fires, fails silently: nothing
runs, and nothing says so. These tests hold the workflows to the release
plan's rules — the client qualification runs on pushes to main and on version
tags, on both platforms the installer supports, and every job that checks for
private names gets the list from a repository secret WITHOUT failing a fork
that has none.
"""

import pathlib

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows"
SECRET_STEP = "python scripts/ci/private_names.py --from-ci-secret"


def _workflow(name: str) -> dict:
    document = yaml.safe_load((WORKFLOWS / name).read_text())
    # YAML 1.1 reads the bare key `on` as the boolean True.
    document["triggers"] = document.pop(True) if True in document else document.pop("on")
    return document


def _steps(job: dict) -> list:
    return [step.get("run", "") or step.get("uses", "") for step in job["steps"]]


def test_both_workflows_parse_and_name_their_jobs():
    assert set(_workflow("ci.yml")["jobs"]) >= {
        "python-suite", "results-explorer", "public-scan"}
    assert set(_workflow("client-release.yml")["jobs"]) >= {"linux-client", "macos-client"}


def test_client_qualification_runs_on_pushes_to_main_and_on_version_tags():
    triggers = _workflow("client-release.yml")["triggers"]
    assert "workflow_dispatch" in triggers
    push = triggers["push"]
    assert push["branches"] == ["main"]
    assert push["tags"] == ["v*"]
    # The same paths gate a pull request and a push, and they cover what the
    # release is built from and checked with.
    assert push["paths"] == triggers["pull_request"]["paths"]
    for path in ("Server/**", "scripts/build-client-release.py",
                 "scripts/ci/qualify-client-release.py", "scripts/ci/artifact_scan.py"):
        assert path in push["paths"]


def test_client_qualification_covers_linux_and_apple_silicon():
    jobs = _workflow("client-release.yml")["jobs"]
    assert jobs["linux-client"]["runs-on"] == "ubuntu-latest"
    assert jobs["macos-client"]["runs-on"].startswith("macos-")
    for job in jobs.values():
        steps = _steps(job)
        build = next(i for i, step in enumerate(steps) if "build-client-release.py" in step)
        qualify = next(i for i, step in enumerate(steps) if "qualify-client-release.py" in step)
        assert build < qualify
        assert "--repair" in steps[qualify]
    # The macOS job refuses to qualify the wrong architecture.
    assert any("Darwin/arm64" in step for step in _steps(jobs["macos-client"]))


@pytest.mark.parametrize("workflow, job, before", [
    ("ci.yml", "python-suite", "python -m pytest -q"),
    ("client-release.yml", "linux-client", "build-client-release.py"),
    ("client-release.yml", "macos-client", "build-client-release.py"),
])
def test_the_private_name_list_comes_from_a_secret_before_it_is_needed(workflow, job, before):
    """The list is read from the repository secret into a temporary file
    before the step that checks names. The step itself decides whether the
    list is required: a fork has no secret, the variable is empty, and
    `private_names.py --from-ci-secret` then exits 0 (see
    test_private_names.py)."""
    job = _workflow(workflow)["jobs"][job]
    steps = _steps(job)
    secret = steps.index(SECRET_STEP)
    needed = next(i for i, step in enumerate(steps) if before in step)
    assert secret < needed
    step = job["steps"][secret]
    assert step["env"] == {"STEERLAB_PRIVATE_NAMES": "${{ secrets.STEERLAB_PRIVATE_NAMES }}"}
    # Nothing gates the step on the secret existing: it must run on a fork.
    assert "if" not in step
