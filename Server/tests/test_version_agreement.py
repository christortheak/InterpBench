"""One release version, written in three places, held together.

SteerLab ships one version across both engines, and it is kept by hand in
three constants:

* ``Server/pyproject.toml`` — ``[project] version`` (the wheel's);
* ``Server/steerlab_server/__init__.py`` — ``__version__`` (what the client
  and the engine report);
* ``Sources/ExperimentKit/SteerLabVersion.swift`` — ``static let version``
  (what the Mac app and CLI report, and what the app build stamps).

Nothing tied them together, so one could be bumped without the others and
the artifacts of one release would disagree about which release they are.
The build scripts read the same two files with the same patterns used here
(``scripts/build-app.sh``, ``scripts/make-server-payload.sh``), so a change
to how a constant is written fails this test before it breaks a build.
"""

import pathlib
import re

from steerlab_server import __version__

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _only(pattern: str, path: pathlib.Path) -> str:
    found = re.findall(pattern, path.read_text(encoding="utf-8"), re.M)
    assert len(found) == 1, f"expected exactly one version constant in {path.name}, found {found}"
    return found[0]


def test_the_three_version_constants_agree():
    pyproject = _only(r'^version = "([^"]+)"$', ROOT / "Server" / "pyproject.toml")
    module = _only(r'^__version__ = "([^"]+)"$',
                   ROOT / "Server" / "steerlab_server" / "__init__.py")
    swift_file = ROOT / "Sources" / "ExperimentKit" / "SteerLabVersion.swift"
    if not swift_file.is_file():
        # A deployed payload carries Server/ without the Swift sources; the two
        # Python constants are still held together there.
        assert pyproject == module == __version__
        return
    swift = _only(r'^\s*public static let version = "([^"]+)"$', swift_file)
    assert pyproject == module == swift == __version__, (
        f"the release version disagrees: pyproject.toml {pyproject}, "
        f"steerlab_server.__version__ {module}, SteerLabVersion.swift {swift}. "
        "Change all three together.")


def test_the_version_is_one_a_release_can_carry():
    """Digits and dots, with an optional pre-release suffix — the shape the
    app build turns into CFBundleShortVersionString and a tag carries after
    its `v`."""
    assert re.fullmatch(
        r"\d+\.\d+(\.\d+)?([.-]?(alpha|beta|a|b|rc|dev)\.?\d*)?", __version__), __version__
