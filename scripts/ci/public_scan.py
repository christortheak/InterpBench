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

Other binaries are skipped, but an archive (tar, gzip, zip — recognized by
content) is OPENED: checks 1–3 run over every member's name, link target,
and text, and check 6 over its header. A packager writes things into an
archive that its source never held, such as the packaging account as every
member's owner or a path from the machine it ran on, and compression hides
them from a text search. A committed fixture archive once carried both.

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

import gzip
import io
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
#: account; and a regular member's bytes (empty for anything else).
Member = tuple[str, str, bool, bytes]


def archive_members(raw: bytes) -> list[Member] | None:
    """The members of a tar (gzipped or not) or a zip, recognized by content
    and never by extension, or ``None`` when ``raw`` is not an archive.
    Raises :class:`Unreadable` rather than return a partial list."""
    try:
        if raw[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                inner = stream.read(MAX_ARCHIVE_BYTES + 1)
            if len(inner) > MAX_ARCHIVE_BYTES:
                raise Unreadable(f"over {MAX_ARCHIVE_BYTES} bytes decompressed")
            if _is_tar(inner):
                return _tar_members(inner)
            return [("(gzip contents)", "", False, inner)]
        if raw[:4] == b"PK\x03\x04":
            return _zip_members(raw)
        if _is_tar(raw):
            return _tar_members(raw)
    # RuntimeError: a zip member that is encrypted, or compressed by a method
    # this Python cannot read (NotImplementedError is one). Only the error's
    # TYPE is reported: its message can quote a member's name, and that name
    # can be the very path the scan exists to keep out of a public log.
    except (OSError, EOFError, ValueError, RuntimeError, zlib.error,
            tarfile.TarError, zipfile.BadZipFile) as error:
        raise Unreadable(type(error).__name__) from error
    return None


def _is_archive(data: bytes) -> bool:
    return data[:2] == b"\x1f\x8b" or data[:4] == b"PK\x03\x04" or _is_tar(data)


def _is_tar(data: bytes) -> bool:
    return data[257:262] == b"ustar"


def _rooted(name: str) -> str:
    """``name`` as an absolute path. A packer strips the leading "/" from an
    absolute member name, and ``Users/<name>/…`` is still a home folder."""
    return "/" + name if name else ""


def _tar_members(data: bytes) -> list[Member]:
    members: list[Member] = []
    budget = MAX_ARCHIVE_BYTES   # what the members may expand to, together
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
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
            payload = b""
            if member.isfile():
                handle = archive.extractfile(member)
                payload = handle.read() if handle is not None else b""
            members.append((member.name, header, owned, payload))
    return members


def _zip_members(data: bytes) -> list[Member]:
    # A zip records no owner names, so no member of one is `owned`.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
            raise Unreadable(f"over {MAX_ARCHIVE_BYTES} bytes decompressed")
        return [(info.filename,
                 _rooted(info.filename) + "\n"
                 + info.comment.decode("utf-8", "replace"),
                 False, b"" if info.is_dir() else archive.read(info))
                for info in infos]


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
    owned = [name for name, _header, recorded, _payload in members if recorded]
    if owned:
        # The account is never printed: printing it would publish the finding.
        report(path, "owner",
               f"{len(owned)} member(s) record an owner account, the first "
               f"{within}{_shown(owned[0])}")
    for name, header, _recorded, payload in members:
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
                    if not _is_archive(raw):
                        continue  # large, and not an archive: data
                    # An archive is never skipped for size: a real bundle
                    # passes 4 MB with one weights file.
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
