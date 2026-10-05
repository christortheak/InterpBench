"""``scripts/ci/public_scan.py`` — the scan of what a commit publishes — and
the archives it used to skip.

The scan skipped every binary file. On 2026-10-05 a committed cross-engine
fixture, a real evidence bundle, turned out to carry the packaging account as
the owner of every member and a per-user temp-folder path in its manifest,
and neither check could see inside it. These tests build a throwaway
repository, plant one thing in an archive (or a temp-folder path in text), and
hold the scan to four promises:

* an owner recorded in a tar member is a finding, and the account is never
  printed;
* a personal path inside an archive is found, whether in a text member, a
  member's name (with or without the leading "/" a packer strips), or an
  archive nested in another, and a member name that is itself a finding is
  never printed;
* an archive that cannot be read, is nested too deep, or is large is a
  finding or is scanned, never a silent skip;
* a clean archive and a placeholder path pass.

Every name is made up and every path is assembled at run time, so this file
holds nothing the scan hunts.
"""

import gzip
import io
import os
import pathlib
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts" / "ci" / "public_scan.py"

OWNER = "rosalind"                       # made up
HOME = "/" + "Users"                     # never spelled whole in this file
HOME_PATH = f"{HOME}/{OWNER}/Projects/study/runs/r"
PLACEHOLDER_HOME_PATH = f"{HOME}/you/Projects/study/runs/r"
TEMP_PATH = ("/private" + "/var/folders" + "/x7/" + "k2m9q4w8v1nb3z0000gn"
             + "/T/steerlab-bundle-fixture-1/runs/r")


def _tar(members: dict, *, owner: str | None = None, uid: int = 0) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.uid = info.gid = uid
            if owner:
                info.uname, info.gname = owner, "staff"
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _tar_gz(members: dict, **owner) -> bytes:
    return gzip.compress(_tar(members, **owner))


def _zip(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _scan(repo: pathlib.Path, files: dict) -> tuple[int, str]:
    for relative, data in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    # Tracked, so no ignore rule on the test machine can hide a file.
    subprocess.run(["git", "add", "-f", "."], cwd=repo, check=True)
    result = subprocess.run([sys.executable, str(SCAN)], cwd=repo,
                            capture_output=True, text=True)
    return result.returncode, result.stdout


def test_an_owner_recorded_in_an_archive_is_a_finding_never_printed(repo):
    code, out = _scan(repo, {"fixtures/bundle.tar.gz": _tar_gz(
        {"runs/r/config.json": b"{}\n", "runs/r/report.json": b"{}\n"},
        owner=OWNER, uid=501)})
    assert code == 1
    assert ("fixtures/bundle.tar.gz [owner] 2 member(s) record an owner "
            "account, the first runs/r/config.json") in out
    assert OWNER not in out


def test_a_numeric_owner_alone_is_a_finding(repo):
    code, out = _scan(repo, {"bundle.tar": _tar({"a.txt": b"hi\n"}, uid=501)})
    assert code == 1
    assert "bundle.tar [owner] 1 member(s)" in out


@pytest.mark.parametrize("files, where", [
    ({"bundle.tar.gz": _tar_gz(
        {"steerlab-evidence.json": f'{{"runDirectory": "{TEMP_PATH}"}}'.encode()})},
     "bundle.tar.gz [path] personal absolute path in steerlab-evidence.json"),
    ({"wheel.zip": _zip({"notes.md": f"built in {HOME_PATH}\n".encode()})},
     "wheel.zip [path] personal absolute path in notes.md"),
    ({"outer.zip": _zip({"inner.tar.gz": _tar_gz(
        {"runs/r/log.txt": f"cwd {TEMP_PATH}\n".encode()})})},
     "outer.zip [path] personal absolute path in inner.tar.gz!runs/r/log.txt"),
])
def test_a_personal_path_inside_an_archive_is_a_finding(repo, files, where):
    code, out = _scan(repo, files)
    assert code == 1
    assert where in out
    assert OWNER not in out and "k2m9q4w8" not in out


# A packer strips the leading "/" (`tarfile.add`, bsdtar), so the stripped
# form is the one a real archive holds; the rooted one is checked too.
@pytest.mark.parametrize("name", [f"{HOME_PATH.lstrip('/')}/report.json",
                                  f"{HOME_PATH}/report.json",
                                  f"{TEMP_PATH.lstrip('/')}/report.json"])
def test_a_member_name_that_is_a_finding_is_found_and_not_printed(repo, name):
    code, out = _scan(repo, {"bundle.tar.gz": _tar_gz(
        {name: b"{}\n"}, owner=OWNER, uid=501)})
    assert code == 1
    assert ("bundle.tar.gz [path] personal absolute path in the header of a "
            "member whose name is itself a finding") in out
    assert ("bundle.tar.gz [owner] 1 member(s) record an owner account, the "
            "first a member whose name is itself a finding") in out
    assert OWNER not in out and "k2m9q4w8" not in out


def test_an_unreadable_archive_is_a_finding_not_a_skip(repo):
    whole = _tar_gz({"runs/r/report.json": b"{}\n" * 4096})
    code, out = _scan(repo, {"bundle.tar.gz": whole[:len(whole) // 2]})
    assert code == 1
    assert "bundle.tar.gz [archive] could not be read" in out


def test_an_encrypted_zip_member_is_a_finding_not_a_crash(repo):
    data = bytearray(_zip({f"{HOME_PATH.lstrip('/')}/notes.md": b"hello\n"}))
    # The encrypted flag, in the local and the central header.
    for signature, flags in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        data[data.find(signature) + flags] |= 0x1
    code, out = _scan(repo, {"wheel.zip": bytes(data)})
    assert code == 1
    assert "wheel.zip [archive] could not be read" in out
    assert "Traceback" not in out and OWNER not in out


def test_an_archive_too_large_to_skip_as_data_is_still_scanned(repo):
    # Incompressible weights put the archive over the 4 MB text limit on disk.
    weights = bytes(range(256)) * 16 + os.urandom(5 * 1024 * 1024)
    archive = _tar_gz({"runs/r/weights.safetensors": weights,
                       "runs/r/config.json": f'"{TEMP_PATH}"'.encode()})
    assert len(archive) > 4 * 1024 * 1024
    code, out = _scan(repo, {"bundle.tar.gz": archive})
    assert code == 1
    assert "bundle.tar.gz [path] personal absolute path in runs/r/config.json" in out


def test_an_archive_nested_too_deep_is_a_finding_not_a_skip(repo):
    innermost = _tar_gz({"a.txt": b"hi\n"})
    nested = _zip({"one.zip": _zip({"two.zip": _zip({"three.tar.gz": innermost})})})
    code, out = _scan(repo, {"outer.zip": nested})
    assert code == 1
    assert ("outer.zip [archive] one.zip!two.zip!three.tar.gz is nested too "
            "deep") in out


def test_a_temp_folder_path_in_text_is_a_finding(repo):
    code, out = _scan(repo, {"notes.md": f"generated in {TEMP_PATH}\n".encode()})
    assert code == 1
    assert "notes.md [path] personal absolute path" in out
    # Prose that names the folder, with no account's id after it, is not one.
    code, out = _scan(repo, {"notes.md": b"a temp root under /var/folders on macOS\n"})
    assert code == 0, out


def test_a_clean_archive_passes(repo):
    code, out = _scan(repo, {
        "bundle.tar.gz": _tar_gz({
            "runs/r/report.json": b'{"runDirectory": "runs/r"}\n',
            "runs/r/notes.md": f"see {PLACEHOLDER_HOME_PATH}\n".encode(),
            "runs/r/weights.bin": bytes(range(256)) * 4}),
        "wheel.zip": _zip({"demo/__init__.py": b"VALUE = 1\n"}),
    })
    assert (code, out.strip()) == (0, "public scan: clean")


def test_an_accepted_archive_finding_is_matched_by_the_archive_path(repo):
    code, out = _scan(repo, {
        "bundle.tar.gz": _tar_gz({"a.txt": b"hi\n"}, owner=OWNER, uid=501),
        "scripts/ci/scan-accepted.txt": b"bundle.tar.gz :: owner\n",
    })
    assert (code, out.strip()) == (0, "public scan: clean")
