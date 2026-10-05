"""``scripts/ci/qualify-client-release.py`` — the closure it asserts, held to
the lock it installs from.

The qualification script installs a built client release into scratch and
then asks the installed environment which modules it can import. It used to
require that ``transformers`` be absent, while the client lock pins
``transformers`` on purpose (``test_client_installer.py`` requires it there),
so the script could not pass against its own release. These tests make that
disagreement impossible to reintroduce: the script's two module lists are
checked against the lock and against ``pyproject.toml``.

The full qualification downloads a Python and installs the lock, so it runs
in CI (``.github/workflows/client-release.yml``), not here.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "qualify-client-release.py"
LOCK = ROOT / "Server/steerlab_server/client/resources/client-requirements.lock"
PYPROJECT = ROOT / "Server/pyproject.toml"


def _script():
    spec = importlib.util.spec_from_file_location("qualify_client_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)        # defines; runs nothing until main()
    return module


def _distribution(module_name: str) -> str:
    """An import name as the lock spells its distribution."""
    return module_name.replace("_", "-")


def _locked() -> set:
    return set(re.findall(r"^([a-z0-9_-]+)==", LOCK.read_text(), re.M))


def _requirement_names(requirements) -> set:
    return {re.match(r"[A-Za-z0-9_.-]+", item).group(0).lower().replace("_", "-")
            for item in requirements}


def test_every_module_the_script_expects_is_in_the_client_lock():
    script = _script()
    locked = _locked()
    expected = {_distribution(name) for name in script.CLIENT_MODULES}
    assert expected <= locked, f"not in the client lock: {sorted(expected - locked)}"
    assert "transformers" in script.CLIENT_MODULES     # the original disagreement


def test_no_module_the_script_forbids_is_in_the_client_lock():
    script = _script()
    forbidden = {_distribution(name) for name in script.ENGINE_ONLY_MODULES}
    assert not forbidden & _locked(), (
        f"the script forbids what the lock installs: {sorted(forbidden & _locked())}")
    assert not set(script.CLIENT_MODULES) & set(script.ENGINE_ONLY_MODULES)
    assert {"torch", "fastapi"} <= set(script.ENGINE_ONLY_MODULES)


def test_the_lists_follow_the_declared_dependencies():
    """The client's declared dependencies are what the script expects, and
    every engine extra is something it forbids — so adding an engine
    dependency without telling the qualification fails here."""
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    script = _script()
    assert _requirement_names(project["dependencies"]) == {
        _distribution(name) for name in script.CLIENT_MODULES}
    extras = project["optional-dependencies"]
    engine = set()
    for extra in ("runner", "lora", "gemmascope", "jlens"):
        engine |= _requirement_names(extras[extra])
    forbidden = {_distribution(name) for name in script.ENGINE_ONLY_MODULES}
    assert engine <= forbidden, f"engine extras the script does not forbid: {sorted(engine - forbidden)}"


def test_the_closure_check_refuses_an_environment_that_has_the_engine():
    """The snippet the script runs inside the installed client, run here in an
    environment that DOES carry the engine: it must fail and say which
    modules. (Skipped where the engine is not installed.)"""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("this environment has no engine installed, so there is nothing to refuse")
    result = subprocess.run(
        [sys.executable, "-c", _script().CLOSURE_CHECK],
        text=True, capture_output=True, check=False, cwd=ROOT / "Server")
    assert result.returncode != 0
    assert "engine-only modules are installed in the client runtime" in result.stderr
    assert "torch" in result.stderr
