"""Licenses in artifacts: the third-party notices generator, and the license
metadata of the Python distribution.

A built app links its Swift packages statically and embeds a JavaScript
bundle, and a wheel is installed far from the repository's LICENSE. So the
terms have to travel inside the artifacts. ``scripts/generate-third-party-
notices.py`` collects the license texts of what a build actually used; these
tests run it on a small made-up build and hold it to its one hard rule:
refuse rather than omit. (``test_build_app_script.py`` and
``test_client_release_build.py`` check that the artifacts carry the results.)
"""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "generate-third-party-notices.py"


def _build(tmp_path: pathlib.Path) -> dict:
    """A made-up build: two pinned Swift packages (one vendoring a C library
    and carrying a NOTICE), and a web bundle that reports one npm package."""
    resolved = tmp_path / "Package.resolved"
    resolved.write_text(json.dumps({"pins": [
        {"identity": "zeta-kit", "location": "https://example.org/acme/ZetaKit.git",
         "state": {"revision": "f" * 40, "version": "2.0.0"}},
        {"identity": "alpha-kit", "location": "https://example.org/acme/alpha-kit",
         "state": {"revision": "a" * 40, "version": "1.2.3"}},
    ]}))
    checkouts = tmp_path / "checkouts"
    alpha = checkouts / "alpha-kit"
    (alpha / "Sources/CVendored").mkdir(parents=True)
    (alpha / "Tests").mkdir()
    (alpha / "LICENSE.txt").write_text("Alpha Kit license text\n")
    (alpha / "NOTICE.txt").write_text("Alpha Kit notice text\n")
    (alpha / "Sources/CVendored/LICENSE").write_text("Vendored C library license text\n")
    (alpha / "Tests/LICENSE").write_text("a test fixture's license, not shipped\n")
    # The checkout directory is named after the repository, not the identity.
    zeta = checkouts / "ZetaKit"
    zeta.mkdir()
    (zeta / "LICENSE").write_text("Zeta Kit license text\n")

    bundle = tmp_path / "web"
    bundle.mkdir()
    (bundle / "bundled-packages.json").write_text('["tinyview"]\n')
    package = tmp_path / "node_modules" / "tinyview"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps(
        {"name": "tinyview", "version": "9.9.9", "license": "MIT"}))
    (package / "LICENSE").write_text("Tinyview license text\n")
    return {"--package-resolved": resolved, "--checkouts": checkouts,
            "--web-bundle": bundle, "--web-node-modules": tmp_path / "node_modules",
            "--output": tmp_path / "THIRD-PARTY-NOTICES.txt"}


def _generate(arguments: dict):
    command = [sys.executable, str(GENERATOR)]
    for flag, value in arguments.items():
        command += [flag, str(value)]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_every_pinned_package_and_bundled_package_is_covered(tmp_path):
    arguments = _build(tmp_path)
    result = _generate(arguments)
    assert result.returncode == 0, result.stderr
    assert "2 Swift package(s), 1 npm package(s)" in result.stdout
    text = arguments["--output"].read_text()
    for expected in ("alpha-kit 1.2.3", "revision " + "a" * 40, "Alpha Kit license text",
                     "Alpha Kit notice text",
                     "--- Sources/CVendored/LICENSE ---", "Vendored C library license text",
                     "zeta-kit 2.0.0", "Zeta Kit license text",
                     "tinyview 9.9.9 (npm)", "declared license: MIT", "Tinyview license text"):
        assert expected in text, expected
    assert "not shipped" not in text                      # a package's tests are not linked in
    assert text.index("alpha-kit 1.2.3") < text.index("zeta-kit 2.0.0")   # stable order


@pytest.mark.parametrize("damage, message", [
    (lambda a: (a["--checkouts"] / "ZetaKit" / "LICENSE").unlink(),
     "the package 'zeta-kit' has no license file"),
    (lambda a: (a["--checkouts"] / "alpha-kit").rename(a["--checkouts"] / "elsewhere"),
     "no checkout for the pinned package 'alpha-kit'"),
    (lambda a: (a["--web-bundle"] / "bundled-packages.json").unlink(),
     "bundled-packages.json is missing"),
    (lambda a: (a["--web-node-modules"] / "tinyview" / "LICENSE").unlink(),
     "the npm package 'tinyview' has no license file"),
    (lambda a: (a["--web-bundle"] / "bundled-packages.json").write_text('["tinyview", "absent"]'),
     "the web bundle contains 'absent'"),
])
def test_it_refuses_rather_than_omits(tmp_path, damage, message):
    arguments = _build(tmp_path)
    damage(arguments)
    result = _generate(arguments)
    assert result.returncode == 1
    assert message in result.stderr
    assert not arguments["--output"].exists()


def test_the_embedded_build_reports_what_it_bundled():
    """The seam the generator relies on: the embedded build writes the list of
    npm packages it actually included."""
    config = (ROOT / "results-explorer" / "vite.embed.config.ts").read_text()
    assert "bundled-packages.json" in config
    assert "bundledPackages()" in config


def test_the_python_distribution_declares_its_license():
    tomllib = pytest.importorskip("tomllib")
    document = tomllib.loads((ROOT / "Server" / "pyproject.toml").read_text())
    assert document["project"]["license"] == "Apache-2.0"
    # An SPDX expression needs a setuptools that reads one (PEP 639).
    assert document["build-system"]["requires"] == ["setuptools>=77.0.3"]
    # The expression names the license the repository actually carries.
    assert "Apache License" in (ROOT / "LICENSE").read_text()[:200]
    assert (ROOT / "NOTICE").is_file()
