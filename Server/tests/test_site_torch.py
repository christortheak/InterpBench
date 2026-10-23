"""The site's torch through bootstrap.sh's lock install (2026-10-22).

On a cluster whose torch index (cu128) did not carry the lock's torch, the
lock install pinned setuptools==84 and a CUDA 13 runtime against the site's
torch 2.11.0+cu128; pip settled the conflict by replacing that torch with
PyPI's CUDA 13 build, and bootstrap reported success. These tests hold the
repair: ``scripts/site_torch.py`` reads the installed torch without importing
it, the lock install drops the pins that torch forbids and holds torch
exactly, a torch that changed anyway fails the step, and a torch built for a
different CUDA than the index provides is reinstalled from the index.

Everything runs against a fake torch distribution on PYTHONPATH; nothing is
installed, downloaded, or built.
"""

import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SITE_TORCH = SCRIPTS / "site_torch.py"
BOOTSTRAP = SCRIPTS / "bootstrap.sh"
bash = shutil.which("bash")

#: What a cu128 torch 2.11.0 requires, in the shape that matters here: an
#: unconditional pin, a marker-gated one that applies everywhere, and an
#: extra that never applies to a plain install.
TORCH_REQUIRES = (
    "filelock",
    "setuptools<82",
    'cuda-toolkit==12.8.1; python_version >= "3.8"',
    'opt-einsum>=3.3; extra == "opt-einsum"',
)


def _fake_torch(site, version="2.11.0+cu128", cuda="12.8"):
    """A torch distribution importlib.metadata and find_spec can see, with
    torch/version.py written as torch writes it."""
    package = site / "torch"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    value = f"'{cuda}'" if cuda else "None"
    (package / "version.py").write_text(
        f"from typing import Optional\n__version__ = '{version}'\n"
        f"cuda: Optional[str] = {value}\n", encoding="utf-8")
    dist = site / f"torch-{version}.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torch\n"
        f"Version: {version}\n"
        + "".join(f"Requires-Dist: {line}\n" for line in TORCH_REQUIRES),
        encoding="utf-8")
    return site


def _site_torch(site, *args, stdin=""):
    return subprocess.run(
        [sys.executable, str(SITE_TORCH), *args], input=stdin, text=True,
        capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": str(site)})


# --- the helper ---------------------------------------------------------------


def test_the_helper_reads_the_installed_version_and_cuda_build(tmp_path):
    site = _fake_torch(tmp_path / "site")
    assert _site_torch(site, "version").stdout.strip() == "2.11.0+cu128"
    assert _site_torch(site, "cuda").stdout.strip() == "12.8"


def test_a_cpu_build_has_no_cuda(tmp_path):
    site = _fake_torch(tmp_path / "site", version="2.11.0+cpu", cuda=None)
    assert _site_torch(site, "cuda").stdout.strip() == ""


def test_the_lock_keeps_what_torch_allows_and_names_what_it_does_not(tmp_path):
    """setuptools==84 and a CUDA 13 runtime against a cu128 torch: both are
    dropped by name; an extra torch never installs constrains nothing."""
    site = _fake_torch(tmp_path / "site")
    lock = ("numpy==2.0.0\nsetuptools==84.0.0\nfilelock==3.32.3\n"
            "cuda-toolkit==13.0.2\nopt-einsum==3.0\nsetuptools-scm==8.0\n")
    proc = _site_torch(site, "filter-lock", stdin=lock)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "numpy==2.0.0", "filelock==3.32.3", "opt-einsum==3.0", "setuptools-scm==8.0"]
    assert ("the lock pins setuptools==84.0.0, which the site's torch 2.11.0+cu128 "
            "does not allow (<82)") in proc.stderr
    assert "the lock pins cuda-toolkit==13.0.2" in proc.stderr
    assert "numpy" not in proc.stderr


def test_an_unknown_question_is_a_usage_error(tmp_path):
    assert _site_torch(_fake_torch(tmp_path / "site"), "nope").returncode == 64


# --- bootstrap.sh, with the helper and a fake pip ------------------------------


def _lock_name():
    return {("Linux", "x86_64"): "requirements-linux-x86_64.lock",
            ("Darwin", "arm64"): "requirements-macos-arm64.lock"}.get(
                (platform.system(), platform.machine()))


def _scriptbin(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _bootstrap(tmp_path, *, cuda="12.8", on_pip=""):
    """Run bootstrap.sh for real against a checkout with the real helper and a
    lock, a fake torch on PYTHONPATH, and a stub interpreter that hands
    site_torch.py to this Python and plays pip for everything else. Each pip
    run copies its -r lock and -c constraint files aside; ``on_pip`` is extra
    shell run inside the stub for each pip call. Returns (process, record)."""
    lock = _lock_name()
    if lock is None or bash is None:
        pytest.skip("bootstrap.sh selects no lock on this platform")
    site = _fake_torch(tmp_path / "site", cuda=cuda)
    server = tmp_path / "repo" / "Server"
    (server / "scripts").mkdir(parents=True)
    shutil.copy(SITE_TORCH, server / "scripts" / "site_torch.py")
    (server / lock).write_text(
        "numpy==2.0.0\ntorch==2.11.0\ntriton==3.6.0\nnvidia-nccl-cu13==2.28.9\n"
        "cuda-toolkit==13.0.2\nsetuptools==84.0.0\nfilelock==3.32.3\n",
        encoding="utf-8")
    record = tmp_path / "record"
    record.mkdir()
    _scriptbin(tmp_path / "fakebin", "conda", "exit 0\n")
    _scriptbin(tmp_path / "env" / "bin", "python", f"""
case "$1" in */site_torch.py) exec {sys.executable} "$@" ;; esac
echo "$*" >> {record}/pip-calls.log
prev=""
for argument in "$@"; do
  [ "$prev" = "-r" ] && cp "$argument" {record}/lock-installed.txt
  [ "$prev" = "-c" ] && cp "$argument" {record}/held.txt
  prev="$argument"
done
{on_pip}
exit 0
""")
    env = {k: v for k, v in os.environ.items()
           if k != "SLURM_JOB_ID" and not k.startswith("STEERLAB_")}
    env.update(PATH=f"{tmp_path / 'fakebin'}:/usr/bin:/bin", HOME=str(tmp_path),
               SLURM_JOB_ID="123456", PYTHONPATH=str(site))
    proc = subprocess.run(
        [bash, str(BOOTSTRAP), "--repo", str(tmp_path / "repo"),
         "--prefix", str(tmp_path / "env"), "--env-file", str(tmp_path / "cluster.env"),
         "--torch-index", "https://example.invalid/whl/cu128"],
        text=True, capture_output=True, check=False, timeout=120, env=env)
    return proc, record, site


def _steps(proc):
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])["steps"]


def test_the_lock_installs_with_the_site_torch_held_and_its_conflicts_named(tmp_path):
    proc, record, _site = _bootstrap(tmp_path)
    steps = _steps(proc)
    assert steps["torchInstall"] == "ok", proc.stdout + proc.stderr
    assert steps["serverInstall"] == "ok", proc.stdout + proc.stderr
    installed = (record / "lock-installed.txt").read_text(encoding="utf-8").splitlines()
    # The torch stack and its CUDA runtime are the site's; setuptools==84 is
    # what this torch forbids.
    assert installed == ["numpy==2.0.0", "filelock==3.32.3"]
    assert (record / "held.txt").read_text(encoding="utf-8") == "torch==2.11.0+cu128\n"
    assert "torch held at 2.11.0+cu128" in proc.stdout
    assert "the lock pins setuptools==84.0.0" in proc.stderr
    # The CUDA runtime never reaches the helper: the filter leaves it out.
    assert "cuda-toolkit" not in proc.stderr
    # Both installs are held: the lock's and the editable package's.
    calls = (record / "pip-calls.log").read_text(encoding="utf-8").splitlines()
    for install in ("-m pip install -r ", "-m pip install -e "):
        assert any(c.startswith(install) and " -c " in c for c in calls), (install, calls)


def test_a_torch_that_changed_anyway_fails_the_step(tmp_path):
    """Held means unchanged: a pip that replaced torch regardless is a failed
    install, never "ok"."""
    replace = (
        'if [ "$prev" = "-r" ] || [[ " $* " == *" -r "* ]]; then\n'
        f'  mv "{tmp_path}/site/torch-2.11.0+cu128.dist-info" '
        f'"{tmp_path}/site/torch-2.14.1.dist-info"\n'
        f'  sed -i.bak "s/^Version: .*/Version: 2.14.1/" "{tmp_path}/site/torch-2.14.1.dist-info/METADATA"\n'
        "fi\n")
    proc, _record, _site = _bootstrap(tmp_path, on_pip=replace)
    assert _steps(proc)["serverInstall"] == "failed"
    assert "torch changed from 2.11.0+cu128 to 2.14.1" in proc.stderr


def test_a_torch_built_for_another_cuda_is_reinstalled_from_the_index(tmp_path):
    rebuild = (
        'if [[ " $* " == *" --force-reinstall "* ]]; then\n'
        f'  sed -i.bak "s/13.0/12.8/" "{tmp_path}/site/torch/version.py"\n'
        "fi\n")
    proc, record, _site = _bootstrap(tmp_path, cuda="13.0", on_pip=rebuild)
    assert _steps(proc)["torchInstall"] == "ok", proc.stdout + proc.stderr
    assert "built for CUDA 13.0, not 12.8" in proc.stdout
    calls = (record / "pip-calls.log").read_text(encoding="utf-8").splitlines()
    assert ("-m pip install --force-reinstall torch==2.11.0 --index-url "
            "https://example.invalid/whl/cu128") in calls


def test_a_torch_that_stays_on_another_cuda_fails_the_step(tmp_path):
    proc, _record, _site = _bootstrap(tmp_path, cuda="13.0")
    assert _steps(proc)["torchInstall"] == "failed"
    assert "built for CUDA 13.0, not 12.8" in proc.stderr
