#!/usr/bin/env python3
"""Public-tier release scan — the half of the release hygiene checks that
ships (CI runs it on every push; release gate 12's permanent enforcement).

Checks TRACKED files only (`git ls-files`), because CI's question is "what
does this commit publish", not "what is lying around":

  1. secret-shaped content (API tokens, AWS keys, private-key blocks)
  2. personal absolute paths (/Users/<name>, /home/<name>, iCloud containers,
     macOS per-user temp folders)
  3. email addresses
  4. junk that should never be tracked (.DS_Store, __pycache__, *.pyc,
     .build/, DerivedData/)
  5. anything tracked beneath Workspaces/ or workspaces/ (the home-layout
     rule: workspaces live BESIDE the checkout, never inside it)
  6. archive members that record an owner account (tar's uid/gid and login
     and group names)
  7. archive members that record extended attributes: a tar's
     `LIBARCHIVE.xattr.*` or `SCHILY.xattr.*` extended headers, and the
     AppleDouble `._*` or `__MACOSX/` members a Mac packer adds. macOS keeps
     a file's origin there (`com.apple.quarantine`, `com.apple.provenance`,
     the page it was downloaded from), so they can identify the machine
     that packed the archive.

Other binaries are skipped, but an archive is OPENED: a tar (with or without
the ustar magic, plain or compressed with gzip, bzip2, or xz) or a zip (also
one with bytes before it, as a self-extracting archive has), each recognized
by content and never by extension; one in a format the standard library
cannot read (zstd, 7z, rar, lz4, compress) is reported as unscanned rather
than skipped. Checks 1–3 run over every member's name,
link target, and text, and checks 6 and 7 over its header. A packager writes
things into an archive that its source never held, such as the packaging
account as every member's owner or a path from the machine it ran on, and
compression hides them from a text search. A committed fixture archive once
carried both.

There is deliberately NO study/name denylist here — that tier is private and
runs at export time on the research side. This scanner must stay
self-contained (stdlib only) and boring.

Known-and-accepted findings live in scripts/ci/scan-accepted.txt as
`<path> :: <check>` lines — each one a deliberate artifact (e.g. the fake
API tokens the secret-scanner tests use as fixtures). An accepted line that
stops matching anything is reported as stale so the file cannot rot.

Exit 0 clean; exit 1 with one line per finding otherwise.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import re
import subprocess
import sys
import tarfile
import zipfile
import zlib

MAX_CONTENT_BYTES = 4 * 1024 * 1024  # skip anything larger — binaries, data
#: An archive's size, on disk or decompressed, past which it is reported
#: unscanned rather than read: a small archive can expand without limit.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
#: How deep to follow an archive inside an archive.
MAX_ARCHIVE_DEPTH = 3

SECRET_PATTERNS = (
    ("secret", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("secret", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("secret", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("secret", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
)
#: Placeholder identities are fixture data, not people (`/Users/you/…`,
#: `me@cluster.test`); a real username or a real host still flags. Same
#: calibration as the export-time scanner's, and shippable — it names only
#: generic fixture words.
_FAKE_USERS = r"(?:me|you|nobody|someone|anyone|user|test|example|demo|alice|bob|[a-z]{1,2})\b"
PATH_PATTERN = re.compile(
    rf"/Users/(?!<|{_FAKE_USERS})[A-Za-z][\w.-]*"
    rf"|/home/(?!<|{_FAKE_USERS})[A-Za-z][\w.-]*")
# Adjacent-string split so this file's own bytes never carry the contiguous
# token it hunts (the export-time scanner would flag the definition itself).
ICLOUD_PATTERN = re.compile(r"com~apple~" r"CloudDocs")
#: macOS's per-user temp folder, `/var/folders/<xx>/<id>/…` (usually reached
#: through `/private`). The id is fixed for one account on one machine, so a
#: path through it identifies the machine as a home folder does, and it is
#: where a fixture generator's scratch workspace lives.
TEMP_PATTERN = re.compile(r"/var/folders/[\w+-]{2}/[\w+-]{6,}")
# `git@` is exempt: it is the well-known SSH user of every git host
# (git@github.com:owner/repo.git — the scp-form remote URL the update
# check parses), never a person's address.
EMAIL_PATTERN = re.compile(
    rf"\b(?!{_FAKE_USERS}@)(?!git@)[\w.+-]+@"
    r"(?!(?:[\w-]+\.)*(?:example|test|invalid|localhost)\b)"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

JUNK_BASENAMES = {".DS_Store"}
JUNK_SEGMENTS = {"__pycache__", ".build", "DerivedData"}
JUNK_SUFFIXES = (".pyc",)

WORKSPACE_ROOTS = ("Workspaces/", "workspaces/")


def tracked_files(root: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True,
        capture_output=True).stdout
    # Untracked-but-not-ignored files too: a NEW file scans locally the
    # same way it will scan in CI after it is committed. (Learned the hard
    # way — two fresh files passed a local scan untracked, then failed the
    # CI lane on the very next push.)
    out += subprocess.run(
        ["git", "ls-files", "-z", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True).stdout
    return [p.decode("utf-8", "replace") for p in out.split(b"\0") if p]


def load_accepted(root: str) -> set[tuple[str, str]]:
    path = os.path.join(root, "scripts", "ci", "scan-accepted.txt")
    accepted: set[tuple[str, str]] = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if " :: " in line:
                    file_part, check = line.split(" :: ", 1)
                    accepted.add((file_part.strip(), check.strip()))
    return accepted


def text_findings(text: str) -> list[tuple[str, str]]:
    """Checks 1–3 over one text: a ``(check, detail)`` pair per finding."""
    found: list[tuple[str, str]] = []
    for check, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            found.append((check, "secret-shaped content"))
            break
    if (PATH_PATTERN.search(text) or ICLOUD_PATTERN.search(text)
            or TEMP_PATTERN.search(text)):
        found.append(("path", "personal absolute path"))
    email = EMAIL_PATTERN.search(text)
    if email:
        found.append(("email", email.group(0)))
    return found


class Unreadable(Exception):
    """An archive that could not be read whole, and so was not scanned."""


#: One archive member: its name; its header as text (the name, a link's
#: target, and any extended-header values); whether it records an owner
#: account; whether it records extended attributes; and a regular member's
#: bytes (empty for anything else).
Member = tuple[str, str, bool, bool, bytes]

#: Extended-header keys under which tar packers record a file's extended
#: attributes (bsdtar, which macOS's `tar` is, and GNU tar's `--xattrs`).
XATTR_HEADERS = ("LIBARCHIVE.xattr.", "SCHILY.xattr.")

_OVER = f"over {MAX_ARCHIVE_BYTES} bytes decompressed"


def archive_kind(data: bytes) -> str | None:
    """What ``data`` is, judged by its bytes: ``gzip``, ``bzip2``, ``xz``,
    ``zip``, or ``tar``, or ``None`` when it is none of them."""
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    # The stream magic, a block size, then a block's or the end's magic: a
    # text file that happens to start "BZh" is not taken for one.
    if (data[:3] == b"BZh" and data[3:4] in b"123456789" and len(data) >= 10
            and data[4:10] in (b"1AY&SY", b"\x17rE8P\x90")):
        return "bzip2"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    # Before a zip with leading bytes: a tar whose last member is a zip ends
    # in that zip's records, and must still be read as the tar it is.
    if _is_tar(data):
        return "tar"
    if _zip_after_leading_bytes(data):
        return "zip"
    for magic, kind in UNOPENABLE:
        if data.startswith(magic):
            return kind
    return None


#: Archive formats the standard library cannot read. One is reported as
#: unscanned, never skipped as an ordinary binary.
UNOPENABLE = ((b"\x28\xb5\x2f\xfd", "zstd"), (b"7z\xbc\xaf\x27\x1c", "7z"),
              (b"Rar!\x1a\x07", "rar"), (b"\x04\x22\x4d\x18", "lz4"),
              (b"\x1f\x9d", "compress"))


#: Bytes a zip may be followed by and still be read (a newline a tool added).
ZIP_TRAILING_SLACK = 64


def _ends_in_zip_record(data: bytes) -> bool:
    """Whether ``data`` ends where a zip's end record (and the comment it
    declares) ends, give or take a few trailing bytes. Every match in the
    last 64 KiB is tried, from the end, because a comment can hold the
    record's own signature."""
    low = max(0, len(data) - 65557 - ZIP_TRAILING_SLACK)
    end = len(data)
    while True:
        end = data.rfind(b"PK\x05\x06", low, end)
        if end < 0:
            return False
        if len(data) - end >= 22:
            stop = end + 22 + int.from_bytes(data[end + 20:end + 22], "little")
            if 0 <= len(data) - stop <= ZIP_TRAILING_SLACK:
                return True


def _zip_after_leading_bytes(data: bytes) -> bool:
    """A zip with bytes before it (a self-extracting archive, a script with
    an archive appended): zip readers find it from its end record."""
    if not _ends_in_zip_record(data):
        return False
    try:
        return zipfile.is_zipfile(io.BytesIO(data))
    except Exception:  # noqa: BLE001 - it ends like a zip; opening it reports what is wrong
        return True


def archive_members(raw: bytes) -> list[Member] | None:
    """The members of a tar (plain, gzip, bzip2, or xz) or a zip, recognized
    by content and never by extension, or ``None`` when ``raw`` is not an
    archive. A compressed stream that holds no tar is one member, which the
    caller opens in turn if it is an archive itself. Raises
    :class:`Unreadable` rather than return a partial list."""
    try:
        kind = archive_kind(raw)
        if kind in {k for _magic, k in UNOPENABLE}:
            raise Unreadable(f"a {kind} archive, which this scan cannot open")
        if kind in ("gzip", "bzip2", "xz"):
            inner = _decompressed(raw, kind)
            if _is_tar(inner):
                return _tar_members(inner)
            return [(f"({kind} contents)", "", False, False, inner)]
        if kind == "zip":
            return _zip_members(raw)
        if kind == "tar":
            members = _tar_members(raw)
            # A zip appended after the tar's end, which a tar reader stops
            # short of (a zip that is the tar's last MEMBER is followed by
            # the tar's end blocks, so it is not found here twice).
            if _zip_after_leading_bytes(raw):
                members += [("(zip after the tar)!" + name, header, owned,
                             xattrs, payload)
                            for name, header, owned, xattrs, payload
                            in _zip_members(raw)]
            return members
    # A zip member that is encrypted, or compressed by a method this Python
    # cannot read, raises too. Only the error's TYPE is reported: its message
    # can quote a member's name, and that name can be the very path the scan
    # exists to keep out of a public log.
    except Unreadable:
        raise
    # Anything else a reader raises on malformed bytes (an IndexError from
    # Python 3.9's zipfile, say) is the same outcome: unread, so reported.
    except Exception as error:  # noqa: BLE001
        raise Unreadable(type(error).__name__) from error
    return None


def _decompressed(raw: bytes, kind: str) -> bytes:
    """Every stream of a gzip, bzip2, or xz file, never past the bound."""
    if kind == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            inner = stream.read(MAX_ARCHIVE_BYTES + 1)
        if len(inner) > MAX_ARCHIVE_BYTES:
            raise Unreadable(_OVER)
        return inner
    make = bz2.BZ2Decompressor if kind == "bzip2" else lzma.LZMADecompressor
    out = bytearray()
    data = raw
    while data:
        decompressor = make()
        out += decompressor.decompress(
            data, max_length=MAX_ARCHIVE_BYTES + 1 - len(out))
        if len(out) > MAX_ARCHIVE_BYTES:
            raise Unreadable(_OVER)
        if not decompressor.eof:
            raise Unreadable("a truncated stream")
        data = decompressor.unused_data
        if kind == "xz":
            data = data.lstrip(b"\0")   # padding between xz streams
    return bytes(out)


def _is_archive(data: bytes) -> bool:
    return archive_kind(data) is not None


def _is_tar(data: bytes) -> bool:
    """A tar's first header: the ustar magic (POSIX, GNU), or, for the
    original format that has none, a header whose checksum adds up."""
    if data[257:262] == b"ustar":
        return True
    block = data[:512]
    if len(block) < 512 or not block[0]:
        return False
    try:
        recorded = int(block[148:156].replace(b"\0", b" ").strip(), 8)
        int(block[124:136].replace(b"\0", b" ").strip() or b"0", 8)
    except ValueError:
        return False
    # The checksum sums the header with its own field read as spaces; old
    # packers summed signed bytes.
    rest = block[:148] + b" " * 8 + block[156:]
    return recorded in (sum(rest), sum(b - 256 if b > 127 else b for b in rest))


def _apple_double(name: str) -> bool:
    """A member macOS adds to carry a file's extended attributes."""
    return (os.path.basename(name.rstrip("/")).startswith("._")
            or name.startswith("__MACOSX/"))


def _rooted(name: str) -> str:
    """``name`` as an absolute path. A packer strips the leading "/" from an
    absolute member name, and ``Users/<name>/…`` is still a home folder."""
    return "/" + name if name else ""


def _tar_members(data: bytes) -> list[Member]:
    members: list[Member] = []
    budget = MAX_ARCHIVE_BYTES   # what the members may expand to, together
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        if archive.pax_headers:
            # A global extended header applies to every member after it.
            members.append(("(global header)",
                            "\n".join(archive.pax_headers.values()), False,
                            any(key.startswith(XATTR_HEADERS)
                                for key in archive.pax_headers), b""))
        for member in archive:
            # A sparse member reads back at its logical size, which the
            # compressed bytes do not bound: refuse it rather than expand it.
            if member.issparse():
                raise Unreadable("a sparse member")
            if member.isfile():
                budget -= member.size
                if budget < 0:
                    raise Unreadable(f"over {MAX_ARCHIVE_BYTES} bytes decompressed")
            header = "\n".join([_rooted(member.name), _rooted(member.linkname),
                                *member.pax_headers.values()])
            owned = bool(member.uid or member.gid
                         or member.uname or member.gname)
            xattrs = (_apple_double(member.name)
                      or any(key.startswith(XATTR_HEADERS)
                             for key in member.pax_headers))
            payload = b""
            if member.isfile():
                handle = archive.extractfile(member)
                payload = handle.read() if handle is not None else b""
            members.append((member.name, header, owned, xattrs, payload))
    return members


def _zip_members(data: bytes) -> list[Member]:
    # A zip records no owner names, so no member of one is `owned`.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
            raise Unreadable(f"over {MAX_ARCHIVE_BYTES} bytes decompressed")
        members = [(info.filename,
                    _rooted(info.filename) + "\n"
                    + info.comment.decode("utf-8", "replace"),
                    False, _apple_double(info.filename),
                    b"" if info.is_dir() else archive.read(info))
                   for info in infos]
        if archive.comment:
            members.append(("(archive comment)",
                            archive.comment.decode("utf-8", "replace"),
                            False, False, b""))
        return members


def _shown(name: str) -> str:
    """A member's name, as a finding prints it. A name that is itself a
    finding is not printed, because printing it would publish it."""
    return ("a member whose name is itself a finding"
            if text_findings(_rooted(name)) else name)


def scan_members(path: str, members: list[Member], report,
                 depth: int = 1, within: str = "") -> None:
    """Checks 1–3 and 6 over an archive's members, each reported against the
    archive's own path so `scan-accepted.txt` still names a tracked file. A
    member that is itself an archive is opened in turn (``within`` names it).
    """
    owned = [name for name, _header, recorded, _xattrs, _payload in members
             if recorded]
    if owned:
        # The account is never printed: printing it would publish the finding.
        report(path, "owner",
               f"{len(owned)} member(s) record an owner account, the first "
               f"{within}{_shown(owned[0])}")
    attributed = [name for name, _header, _recorded, xattrs, _payload
                  in members if xattrs]
    if attributed:
        # Nor are the attributes: they are what could identify the machine.
        report(path, "xattr",
               f"{len(attributed)} member(s) record extended attributes, the "
               f"first {within}{_shown(attributed[0])}")
    for name, header, _recorded, _xattrs, payload in members:
        shown = within + _shown(name)
        for check, detail in text_findings(header):
            report(path, check, f"{detail} in the header of {shown}")
        if depth >= MAX_ARCHIVE_DEPTH and _is_archive(payload):
            report(path, "archive",
                   f"{shown} is nested too deep, so it was not scanned")
            continue
        try:
            nested = archive_members(payload)
        except Unreadable as error:
            report(path, "archive",
                   f"{shown} could not be read, so it was not scanned "
                   f"({error})")
            continue
        if nested is not None:
            scan_members(path, nested, report, depth + 1, within=shown + "!")
        elif payload and b"\0" not in payload[:8192]:
            for check, detail in text_findings(
                    payload.decode("utf-8", "replace")):
                report(path, check, f"{detail} in {shown}")


def _ends_like_zip(handle) -> bool:
    """Whether a file too large to read whole ends in a zip's end record,
    which is all a zip with leading bytes shows at its start."""
    handle.seek(0, os.SEEK_END)
    handle.seek(max(0, handle.tell() - 65557))
    return _ends_in_zip_record(handle.read())


def main() -> int:
    root = os.getcwd()
    accepted = load_accepted(root)
    used: set[tuple[str, str]] = set()
    findings: list[str] = []

    def report(path: str, check: str, detail: str) -> None:
        if (path, check) in accepted:
            used.add((path, check))
            return
        findings.append(f"{path} [{check}] {detail}")

    for path in tracked_files(root):
        base = os.path.basename(path)
        segments = path.split("/")
        if base in JUNK_BASENAMES or base.endswith(JUNK_SUFFIXES) or (
                set(segments[:-1]) & JUNK_SEGMENTS):
            report(path, "junk", "tracked build/system artifact")
        for ws in WORKSPACE_ROOTS:
            if path.startswith(ws):
                report(path, "workspace",
                       "tracked path beneath a workspaces root")

        full = os.path.join(root, path)
        try:
            with open(full, "rb") as handle:
                raw = handle.read(MAX_CONTENT_BYTES + 1)
                if len(raw) > MAX_CONTENT_BYTES:
                    if not (_is_archive(raw) or _ends_like_zip(handle)):
                        continue  # large, and not an archive: data
                    # An archive is never skipped for size: a real bundle
                    # passes 4 MB with one weights file.
                    handle.seek(len(raw))
                    raw += handle.read(MAX_ARCHIVE_BYTES + 1 - len(raw))
        except OSError:
            continue
        if len(raw) > MAX_ARCHIVE_BYTES:
            report(path, "archive",
                   f"over {MAX_ARCHIVE_BYTES} bytes, so it was not scanned")
            continue
        try:
            members = archive_members(raw)
        except Unreadable as error:
            report(path, "archive",
                   f"could not be read, so it was not scanned ({error})")
            continue
        if members is not None:
            scan_members(path, members, report)
            continue
        if b"\0" in raw[:8192]:
            continue  # binary, and not an archive
        for check, detail in text_findings(raw.decode("utf-8", "replace")):
            report(path, check, detail)

    for entry in sorted(accepted - used):
        findings.append(
            f"{entry[0]} [stale-accept] accepted '{entry[1]}' finding no "
            "longer matches anything — remove the line")

    if findings:
        print(f"public scan: {len(findings)} finding(s)")
        for line in sorted(findings):
            print(f"  {line}")
        return 1
    print("public scan: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
