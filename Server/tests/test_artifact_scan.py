"""``scripts/ci/artifact_scan.py`` — the scan of BUILT artifacts. (The
private-name loader it shares with the neutrality guards has its own tests,
``test_private_names.py``.)

The scan exists because the source scan reads tracked source only, while what
a stranger downloads is a BUILT thing: a compiler writes the build machine's
paths into a binary, and a payload copies whole trees. These tests build a
small artifact-shaped tree, plant each kind of finding in it, and hold the
scan to four promises:

* a planted private term is found wherever it hides — plain bytes, a binary
  blob, UTF-16 text, a file's name, a member of a wheel, a member of a
  tarball;
* a home-folder path is found, and a placeholder one is not;
* a clean tree passes;
* the term itself is never printed, and a missing list is a failure unless
  the caller explicitly allows it.

Every term here is made up, and every home-folder path is assembled at run
time, so this file holds nothing the scan (or the commit hygiene check) hunts.
"""

import gzip
import io
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts" / "ci" / "artifact_scan.py"

TERM = "quuxcluster"                    # made up; "private" only in this test
HOME = "/" + "Users"                    # never spelled whole in this file
REAL_HOME_PATH = f"{HOME}/rosalind/Projects/SteerLab/Sources/File.swift"
PLACEHOLDER_HOME_PATH = f"{HOME}/you/Project/Sources/File.swift"
# A URL whose path merely contains a segment named like a home root.
URL_WITH_A_HOME_SEGMENT = "https://example.org/site/" + "home" + "/page"


def _zip(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _tar_gz(members: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(buffer.getvalue())


def _clean_tree(root: pathlib.Path) -> pathlib.Path:
    """An app-shaped tree with nothing to find: a binary, text, a wheel, a
    tarball, and a placeholder home path that must NOT be a finding."""
    app = root / "Demo.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources/payload").mkdir(parents=True)
    (app / "Contents/MacOS/demo").write_bytes(
        b"\xcf\xfa\xed\xfe" + bytes(range(256)) * 4 + b"\0a plain string\0")
    (app / "Contents/Resources/notes.md").write_text(
        f"Build it from {PLACEHOLDER_HOME_PATH}, or from <checkout>.\n"
        f"See {URL_WITH_A_HOME_SEGMENT} for more.\n")
    (app / "Contents/Resources/payload/demo-1.0-py3-none-any.whl").write_bytes(
        _zip({"demo/__init__.py": b"VALUE = 1\n", "demo-1.0.dist-info/METADATA": b"Name: demo\n"}))
    (app / "Contents/Resources/payload/source.tar.gz").write_bytes(
        _tar_gz({"demo/readme.txt": b"nothing to see\n"}))
    return app


def _list(root: pathlib.Path, text: str = f"# made up\n{TERM}\n") -> pathlib.Path:
    path = root / "private-names.txt"
    path.write_text(text)
    return path


def _scan(*arguments, names_file=None, require=False):
    environment = {"PATH": "/usr/bin:/bin"}
    # Never the machine's own list: point at the given file, or at nothing.
    environment["STEERLAB_PRIVATE_NAMES_FILE"] = str(
        names_file if names_file is not None else "/nonexistent/private-names.txt")
    if require:
        environment["STEERLAB_REQUIRE_PRIVATE_NAMES"] = "1"
    return subprocess.run(
        [sys.executable, str(SCAN), *map(str, arguments)],
        env=environment, text=True, capture_output=True, check=False)


def test_a_clean_tree_passes(tmp_path):
    app = _clean_tree(tmp_path)
    result = _scan(app, names_file=_list(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout
    assert "private names and home-folder paths" in result.stdout


@pytest.mark.parametrize("label, plant, where", [
    ("text", lambda app: (app / "Contents/Resources/notes.md").write_text(
        "Submit it on the QuuxCluster login node.\n"),
     "Demo.app/Contents/Resources/notes.md"),
    ("binary", lambda app: (app / "Contents/MacOS/demo").write_bytes(
        b"\xcf\xfa\xed\xfe" + b"\0" * 300 + b"/opt/QUUXCLUSTER/lib\0" + b"\xff" * 64),
     "Demo.app/Contents/MacOS/demo"),
    ("utf16", lambda app: (app / "Contents/Resources/strings.bin").write_bytes(
        b"\x07\x00" + "host = quuxcluster.example".encode("utf-16-le")),
     "Demo.app/Contents/Resources/strings.bin"),
    ("file name", lambda app: (app / "Contents/Resources/quuxcluster-site.json").write_text("{}"),
     "Demo.app/Contents/Resources/***********-site.json [private-name] in its path"),
    ("wheel member", lambda app: (app / "Contents/Resources/payload/demo-1.0-py3-none-any.whl")
     .write_bytes(_zip({"demo/tests/test_site.py": b"for name in ('quuxcluster',): pass\n"})),
     "demo-1.0-py3-none-any.whl!demo/tests/test_site.py"),
    ("tarball member", lambda app: (app / "Contents/Resources/payload/source.tar.gz")
     .write_bytes(_tar_gz({"demo/site.env": b"export SITE=quuxcluster\n"})),
     "source.tar.gz!demo/site.env"),
])
def test_a_planted_private_name_is_found(tmp_path, label, plant, where):
    app = _clean_tree(tmp_path)
    plant(app)
    result = _scan(app, names_file=_list(tmp_path))
    assert result.returncode == 1, f"{label}: {result.stdout}{result.stderr}"
    assert "[private-name]" in result.stdout
    assert where in result.stdout
    # The finding is located and shown in context, but the term is masked:
    # this output must be safe in a public log.
    assert TERM not in result.stdout.lower()
    assert TERM not in result.stderr.lower()


def test_a_home_folder_path_is_found_and_a_placeholder_is_not(tmp_path):
    app = _clean_tree(tmp_path)
    assert _scan(app, names_file=_list(tmp_path)).returncode == 0   # placeholder only
    (app / "Contents/MacOS/demo").write_bytes(
        b"\xcf\xfa\xed\xfe" + b"\0" * 64 + REAL_HOME_PATH.encode() + b"\0")
    result = _scan(app, names_file=_list(tmp_path))
    assert result.returncode == 1
    assert "Demo.app/Contents/MacOS/demo [home-path] at byte 68" in result.stdout
    assert "rosalind" in result.stdout          # not a private term, so shown


def test_an_allowed_context_accepts_a_term_only_inside_that_token(tmp_path):
    """`allow:<token>` is how the list's owner keeps an identifier that
    happens to contain a term. It accepts the term there and nowhere else."""
    app = _clean_tree(tmp_path)
    notes = app / "Contents/Resources/notes.md"
    names = _list(tmp_path, f"{TERM}\nallow:{TERM}Family\n")
    notes.write_text("case family: quuxclusterFamily\n")
    assert _scan(app, names_file=names).returncode == 0
    notes.write_text("case family: quuxclusterFamily, run on quuxcluster\n")
    result = _scan(app, names_file=names)
    assert result.returncode == 1
    assert result.stdout.count("[private-name]") == 1


def test_a_missing_list_fails_unless_explicitly_allowed(tmp_path):
    app = _clean_tree(tmp_path)
    refused = _scan(app)
    assert refused.returncode == 2
    assert "private names cannot be checked" in refused.stderr
    assert "--allow-missing-list" in refused.stderr

    allowed = _scan("--allow-missing-list", app)
    assert allowed.returncode == 0
    assert "home-folder paths ONLY" in allowed.stdout
    # …and that reduced scan still has teeth.
    (app / "Contents/Resources/log.txt").write_text(f"built in {REAL_HOME_PATH}\n")
    assert _scan("--allow-missing-list", app).returncode == 1

    # A release gate requires the list; the flag cannot talk it out of that.
    required = _scan("--allow-missing-list", app, require=True)
    assert required.returncode == 2
    assert "STEERLAB_REQUIRE_PRIVATE_NAMES" in required.stderr


def test_a_path_that_does_not_exist_is_not_a_clean_scan(tmp_path):
    result = _scan(tmp_path / "nothing-here", names_file=_list(tmp_path))
    assert result.returncode == 2
    assert "no such path" in result.stderr
