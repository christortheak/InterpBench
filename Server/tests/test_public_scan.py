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
  finding or is scanned, never a silent skip, whatever its format: tar with
  or without the ustar magic, gzip, bzip2, xz, or a zip with leading bytes;
* extended attributes recorded in an archive (a tar's extended headers, a
  Mac packer's AppleDouble members) are a finding, never printed;
* a clean archive and a placeholder path pass.

Every name is made up and every path is assembled at run time, so this file
holds nothing the scan hunts.
"""

import bz2
import gzip
import io
import lzma
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


def _v7_tar(members: dict) -> bytes:
    """A tar in the original format, which has no ustar magic: Python's
    tarfile cannot write one, so the magic is blanked and the checksum
    recomputed in each header it wrote."""
    data = bytearray(_tar(members))
    offset = 0
    for name, payload in members.items():
        header = data[offset:offset + 512]
        header[257:265] = b"\0" * 8
        header[148:156] = b" " * 8
        header[148:156] = ("%06o\0 " % sum(header)).encode()
        data[offset:offset + 512] = header
        offset += 512 + -(-len(payload) // 512) * 512
    return bytes(data)


def _pax_tar(members: dict, headers: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.pax_headers = dict(headers)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


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


def _sparse_tar_gz(logical_size: int) -> bytes:
    """A GNU sparse member: 512 stored bytes that read back at
    ``logical_size``. Python's tarfile cannot write one, so the header is
    built by hand."""
    import gzip

    def octal(n, width):
        return ("%0*o" % (width - 1, n)).encode() + b"\0"

    data = b"x" * 512
    h = bytearray(512)
    name = b"runs/r/report.json"
    h[0:len(name)] = name
    h[100:108] = octal(0o644, 8)
    h[108:116] = octal(0, 8)
    h[116:124] = octal(0, 8)
    h[124:136] = octal(len(data), 12)
    h[136:148] = octal(0, 12)
    h[156:157] = b"S"
    h[257:265] = b"ustar  \0"
    h[386:398] = octal(0, 12)
    h[398:410] = octal(len(data), 12)
    h[483:495] = octal(logical_size, 12)
    h[148:156] = b" " * 8
    h[148:156] = ("%06o\0 " % sum(h)).encode()
    return gzip.compress(bytes(h) + data + b"\0" * 1024)


def test_a_sparse_member_is_refused_not_expanded(repo):
    """A few hundred compressed bytes must not expand to hundreds of megabytes
    in the scanner's memory: a sparse member is reported, never read."""
    code, out = _scan(repo, {"bundle.tar.gz": _sparse_tar_gz(200 * 1024 * 1024)})
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


@pytest.mark.parametrize("name, data", [
    ("bundle.tar.xz", lzma.compress(_tar({"runs/r/log.txt": f"cwd {TEMP_PATH}\n".encode()}))),
    ("bundle.tar.bz2", bz2.compress(_tar({"runs/r/log.txt": f"cwd {TEMP_PATH}\n".encode()}))),
    ("bundle.tar", _v7_tar({"runs/r/log.txt": f"cwd {TEMP_PATH}\n".encode()})),
    ("installer.sh", b"#!/bin/sh\nexit 0\n" + _zip({"runs/r/log.txt": f"cwd {TEMP_PATH}\n".encode()})),
    ("outer.zip", _zip({"inner.tar.xz": lzma.compress(_tar({"runs/r/log.txt": f"cwd {TEMP_PATH}\n".encode()}))})),
], ids=["xz", "bzip2", "tar without ustar magic", "zip with leading bytes", "xz in a zip"])
def test_every_archive_format_is_opened(repo, name, data):
    # Each of these was skipped as an ordinary binary: the scan recognized
    # only gzip, a zip that starts at byte 0, and a tar with the ustar magic.
    code, out = _scan(repo, {name: data})
    assert code == 1
    assert f"{name} [path] personal absolute path in " in out
    assert "runs/r/log.txt" in out and "k2m9q4w8" not in out


def test_an_owner_in_a_bzip2_tar_is_found(repo):
    code, out = _scan(repo, {"bundle.tbz": bz2.compress(_tar({"a.txt": b"hi\n"}, owner=OWNER, uid=501))})
    assert code == 1
    assert "bundle.tbz [owner] 1 member(s) record an owner account" in out
    assert OWNER not in out


def test_a_large_zip_with_leading_bytes_is_still_opened(repo):
    # Past the 4 MB text limit a file is read whole only when it is an
    # archive, and a zip with leading bytes shows that only at its end.
    data = (b"\x7fELF" + os.urandom(5 * 1024 * 1024)
            + _zip({"notes.md": f"built in {HOME_PATH}\n".encode()}))
    code, out = _scan(repo, {"tool.bin": data})
    assert code == 1
    assert "tool.bin [path] personal absolute path in notes.md" in out


@pytest.mark.parametrize("key", ["LIBARCHIVE.xattr.com.apple.provenance",
                                 "SCHILY.xattr.com.apple.quarantine"])
def test_extended_attributes_in_a_tar_are_a_finding_never_printed(repo, key):
    # macOS's tar (bsdtar) records each file's extended attributes in an
    # extended header unless told not to; GNU tar does with --xattrs.
    secret = "0083;5f3a;Safari;" + OWNER
    code, out = _scan(repo, {"bundle.tar.gz": gzip.compress(_pax_tar(
        {"runs/r/a.txt": b"hi\n", "runs/r/b.txt": b"hi\n"}, {key: secret}))})
    assert code == 1
    assert ("bundle.tar.gz [xattr] 2 member(s) record extended attributes, "
            "the first runs/r/a.txt") in out
    assert secret not in out and OWNER not in out


@pytest.mark.parametrize("name", ["__MACOSX/runs/r/._report.json", "runs/r/._report.json"])
def test_apple_double_members_are_a_finding(repo, name):
    # `ditto -c -k` and Finder's Compress store a file's extended attributes
    # as an AppleDouble member beside it.
    code, out = _scan(repo, {"bundle.zip": _zip({"runs/r/report.json": b"{}\n",
                                                 name: b"\x00\x05\x16\x07attributes"})})
    assert code == 1
    assert f"bundle.zip [xattr] 1 member(s) record extended attributes, the first {name}" in out


@pytest.mark.parametrize("name, data", [
    ("bundle.tar.xz", lzma.compress(_tar({"runs/r/report.json": b"{}\n" * 4096}))[:-40]),
    ("bundle.tar.bz2", bz2.compress(_tar({"runs/r/report.json": b"{}\n" * 4096}))[:-40]),
    ("bomb.xz", lzma.compress(b"\0" * (65 * 1024 * 1024), preset=0)),
], ids=["truncated xz", "truncated bzip2", "xz over the bound"])
def test_an_unreadable_compressed_stream_is_a_finding(repo, name, data):
    code, out = _scan(repo, {name: data})
    assert code == 1
    assert f"{name} [archive] could not be read" in out


def test_a_tar_that_ends_with_a_zip_is_read_as_the_tar(repo):
    # A zip's records sit at the very end of a tar whose last member is a
    # zip, and a zip reader finds them there. The tar's own headers and
    # members must still be scanned.
    data = _tar({"notes.md": f"built in {HOME_PATH}\n".encode(),
                 "last.zip": _zip({"clean.txt": b"hi\n"})}, owner=OWNER, uid=501)
    code, out = _scan(repo, {"bundle.tar": data})
    assert code == 1
    assert "bundle.tar [owner] 2 member(s)" in out
    assert "bundle.tar [path] personal absolute path in notes.md" in out


def test_an_archive_this_scan_cannot_open_is_a_finding(repo):
    code, out = _scan(repo, {"bundle.tar.zst": b"\x28\xb5\x2f\xfd" + os.urandom(64)})
    assert code == 1
    assert ("bundle.tar.zst [archive] could not be read, so it was not "
            "scanned (a zstd archive, which this scan cannot open)") in out


def test_a_zip_end_record_zipfile_cannot_parse_never_crashes_the_scan(repo):
    # A zip64 locator naming another disk makes Python 3.9's zipfile raise
    # while it is only deciding whether the file is a zip (later versions
    # answer "not a zip"; the unit test above provokes the raise on any
    # version). End to end, the scan finishes and prints no traceback.
    import struct
    data = (b"#!/bin/sh\nexit 0\n" + struct.pack("<4sIQI", b"PK\x06\x07", 1, 0, 2)
            + b"PK\x05\x06" + b"\0" * 18)
    # At the nesting limit the check ran outside any error handling, and the
    # scan of a commit holding this died with a traceback on Python 3.9.
    nested = _zip({"one.zip": _zip({"two.zip": _zip({"tool.sh": data})})})
    code, out = _scan(repo, {"tool.sh": data, "outer.zip": nested,
                             "big.bin": os.urandom(5 * 1024 * 1024) + data})
    assert code in (0, 1) and out.startswith("public scan:"), out
    assert "Traceback" not in out


def test_a_large_binary_with_stray_zip_bytes_near_its_end_is_data(repo):
    # Only an end record that ends the file makes it a zip; the same four
    # bytes inside other data do not. Taken for a zip, a file this size is
    # an archive too large to scan, which fails the check.
    data = b"\x7fELF" + bytes(65 * 1024 * 1024) + b"PK\x05\x06" + b"\xff" * 64
    code, out = _scan(repo, {"tool.bin": data})
    assert (code, out.strip()) == (0, "public scan: clean"), out


def _with_comment(zipped: bytes, comment: bytes) -> bytes:
    """``zipped`` with an archive comment, as a packer writes it."""
    buffer = io.BytesIO(zipped)
    with zipfile.ZipFile(buffer, "a") as archive:
        archive.comment = comment
    return buffer.getvalue()


@pytest.mark.parametrize("tail", ["a trailing newline", "a comment holding the end signature"])
def test_a_zip_with_leading_bytes_is_found_by_any_end_record(repo, tail):
    zipped = _zip({"notes.md": f"built in {HOME_PATH}\n".encode()})
    if tail == "a trailing newline":
        data = b"#!/bin/sh\nexit 0\n" + zipped + b"\n"
    else:
        data = b"#!/bin/sh\nexit 0\n" + _with_comment(zipped, b"see PK\x05\x06 above" + b"x" * 30)
    code, out = _scan(repo, {"installer.sh": data})
    assert code == 1
    # Scanned, or (where this Python's zipfile stops at the signature in the
    # comment) reported unread: never skipped as an ordinary binary.
    assert ("installer.sh [path] personal absolute path in notes.md" in out
            or "installer.sh [archive] could not be read" in out), out


def test_a_zip_appended_after_a_tar_is_scanned_too(repo):
    data = _tar({"clean.txt": b"hi\n"}) + _zip({"notes.md": f"built in {HOME_PATH}\n".encode()})
    code, out = _scan(repo, {"bundle.tar": data})
    assert code == 1
    assert ("bundle.tar [path] personal absolute path in "
            "(zip after the tar)!notes.md") in out


def _scanner_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("public_scan_under_test", SCAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_reader_that_raises_anything_is_unreadable_never_a_crash(monkeypatch):
    # Python 3.9's zipfile raises BadZipFile from is_zipfile on some end
    # records, and IndexError from ZipInfo.is_dir() on an empty name; the
    # suite runs a later Python, so both are provoked here directly.
    scan = _scanner_module()
    data = b"#!/bin/sh\n" + _zip({"a.txt": b"hi\n"})

    def bad_zip(_handle):
        raise zipfile.BadZipFile("spans disks")
    monkeypatch.setattr(scan.zipfile, "is_zipfile", bad_zip)
    assert scan._is_archive(data) is True
    monkeypatch.undo()

    def index_error(self):
        raise IndexError("string index out of range")
    monkeypatch.setattr(zipfile.ZipInfo, "is_dir", index_error)
    with pytest.raises(scan.Unreadable, match="IndexError"):
        scan.archive_members(data)


def test_files_that_only_start_like_an_archive_are_text(repo):
    code, out = _scan(repo, {"notes.md": b"BZh9 is how a bzip2 stream begins.\n",
                             "plain.txt": b"PK is a pair of letters.\n"})
    assert (code, out.strip()) == (0, "public scan: clean"), out


def test_a_clean_archive_in_every_format_passes(repo):
    members = {"runs/r/report.json": b'{"runDirectory": "runs/r"}\n'}
    code, out = _scan(repo, {
        "a.tar.xz": lzma.compress(_tar(members)),
        "a.tar.bz2": bz2.compress(_tar(members)),
        "a.tar": _v7_tar(members),
        "a.pax.tar": _pax_tar(members, {"comment": "made for a test"}),
        "a.sh": b"#!/bin/sh\nexit 0\n" + _zip(members),
    })
    assert (code, out.strip()) == (0, "public scan: clean"), out


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
