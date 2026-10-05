#!/usr/bin/env python3
"""Scan BUILT artifacts for identifying strings — what the source scan cannot
see.

  usage: artifact_scan.py [--allow-missing-list] [--show-context] <path> [<path> ...]

``public_scan.py`` reads tracked source. A compiler, a packager, and a bundler
stand between that source and what a stranger downloads, and each can add
something the source never held: a compiler writes the build machine's paths
into a binary, a payload copies a tree nobody meant to ship. So this reads the
thing itself. Point it at an app bundle, an extracted client release, a wheel,
or any directory or file, and it reads EVERY file's bytes — executables
included — and looks inside zip and tar archives (a wheel is a zip), for:

  private-name   a term from the private-name list, in any letter case, in a
                 file's bytes or in its path. The list is never in this
                 repository; ``private_names.py`` (beside this file) says
                 where it lives. An ``allow:<token>`` line in that list
                 accepts a term when it sits inside that longer token.
  home-path      a home-folder path (under the macOS or Linux home root) whose
                 user is not a placeholder — the same calibration as
                 ``public_scan.py``.

Both are searched as plain bytes and as UTF-16 text.

Digests are not searched. A hash manifest lists a digest beside each file
name, and a digest is random text: a wheel's ``RECORD`` writes it in base64,
mixed letter case, so a short term turns up inside one by chance. In the
manifests named in ``DIGEST_FIELDS`` (a wheel's ``RECORD``, ``SHA256SUMS``
and ``*.sha256`` files, pip ``--hash=`` values, and the build's resource and
deployment manifests) the digest fields alone are blanked before the search.
The file names beside them are searched like everything else.

Exit 0 clean. Exit 1 with one line per finding: the file (and the archive
member), the check, and the byte offset. Exit 2 when the scan could not be
made: a path that does not exist, a file that could not be read, or no
private-name list.

A private term is NEVER printed, so the output is safe in a public CI log. A
private-name finding names the term by its position in the list ("list entry
3"), which the list's owner can look up and nobody else can. A home-path
finding prints the path, with any private term in it masked.

``--show-context`` adds the text around each private-name finding, the term
masked. It is for a private terminal: when the surrounding word is a common
one, the mask is a fill-in-the-blank, so keep it out of public logs. It is
how to tell a real finding from a harmless longer word that merely contains
a term — and the remedy for the latter is an ``allow:<word>`` line in the
list.

A missing list is a failure, because a scan that cannot check names would
otherwise pass for the wrong reason. ``--allow-missing-list`` turns that into
a home-path-only scan that says so — for a machine that has no list and so
nothing to leak. ``STEERLAB_REQUIRE_PRIVATE_NAMES=1`` overrides the flag: a
release gate sets it, and then a missing list always fails.

Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import private_names  # noqa: E402 - sibling module, found through the path above
from public_scan import _FAKE_USERS  # noqa: E402 - one placeholder calibration

#: The two home roots, assembled so this file never spells a home-folder
#: prefix itself (the commit hygiene check greps for one).
HOME_ROOTS = ("Users", "home")
#: A home-folder path with a real-looking user. Not preceded by a word
#: character, so a URL whose path merely has a segment named like a home
#: root is not one.
HOME_PATH = re.compile(
    rb"(?<![A-Za-z0-9_.~-])/(?:" + "|".join(HOME_ROOTS).encode()
    + rb")/(?!<|" + _FAKE_USERS.encode() + rb")[A-Za-z][\w.-]*")
#: Runs of ASCII text stored as UTF-16, either byte order. Collapsed to ASCII
#: and searched like any other bytes.
UTF16_LE_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){6,}")
UTF16_BE_RUN = re.compile(rb"(?:\x00[\x20-\x7e]){6,}")

#: Hash manifests, recognized by name, and the digest field inside each. The
#: first pattern matches the file's path (an archive member's path when the
#: file is inside an archive); group 1 of the second is the digest, which is
#: blanked before searching. Nothing else in the file is blanked, so a term in
#: a file NAME listed beside a digest is still a finding.
DIGEST_FIELDS = (
    # A wheel's <name>.dist-info/RECORD: `path,sha256=<urlsafe base64>,size`.
    (re.compile(r"(?:^|/)[^/]+\.dist-info/RECORD$"),
     re.compile(rb"(?m),([A-Za-z0-9_]+=[A-Za-z0-9_-]+=*),[0-9]*\r?$")),
    # SHA256SUMS and <file>.sha256: `<hex>  <name>`, or a digest on its own.
    (re.compile(r"(?:^|/)(?:SHA256SUMS|[^/]+\.sha256)$"),
     re.compile(rb"(?m)^([0-9A-Fa-f]{64})(?=[ \t*\r\n]|$)")),
    # A pip requirements lock: `--hash=sha256:<hex>`.
    (re.compile(r"\.(?:lock|txt)$"),
     re.compile(rb"--hash=[A-Za-z0-9]+:([0-9A-Fa-f]+)")),
    # The app's resource manifest and the cluster payload's deployment
    # manifest: `"<path>": "<hex>"`.
    (re.compile(r"(?:^|/)(?:resource|deployment)-manifest\.json$"),
     re.compile(rb':\s*"([0-9a-f]{64})"')),
)


def blank_digests(location: str, data: bytes) -> bytes:
    """``data`` with the digest fields of a hash manifest replaced by NUL
    bytes of the same length (so offsets still mean what they say). Any other
    file is returned unchanged."""
    path = location.rsplit("!", 1)[-1]
    for name, field in DIGEST_FIELDS:
        if not name.search(path):
            continue
        blanked = bytearray(data)
        for match in field.finditer(data):
            start, end = match.span(1)
            blanked[start:end] = b"\x00" * (end - start)
        data = bytes(blanked)
    return data


#: How deep to follow an archive inside an archive (a wheel inside a tarball
#: inside a bundle is depth 2).
MAX_ARCHIVE_DEPTH = 3
#: Findings printed per file, check, and distinct thing found (one list entry,
#: one home path); the rest are counted.
MAX_PER_KIND = 3
CONTEXT_BYTES = 36


@dataclass(frozen=True)
class Finding:
    location: str      # the file, with `!member` for each archive level
    check: str         # "private-name" | "home-path"
    offset: int        # byte offset in the file or member; -1 for its path
    what: str          # the list entry's label, or the home path (masked)
    context: str       # surrounding text, private terms masked

    def line(self, show_context: bool) -> str:
        where = "in its path" if self.offset < 0 else f"at byte {self.offset}"
        line = f"{self.location} [{self.check}] {where}: {self.what}"
        if show_context and self.check == "private-name":
            line += f" — {self.context}"
        return line


class Scanner:
    def __init__(self, names: Optional["private_names.PrivateNames"]) -> None:
        self.names = names
        self.terms = [term.encode("utf-8") for term in names.terms] if names else []
        self.allowed = ([token.encode("utf-8") for token in names.allowed_contexts]
                        if names else [])
        self.findings: list[Finding] = []
        self.suppressed = 0
        self.files = 0
        self.unreadable: list[str] = []

    # -- reporting ----------------------------------------------------------

    def _mask(self, data: bytes) -> str:
        """Printable context with every private term blanked out."""
        lowered = data.lower()
        masked = bytearray(data)
        for term in self.terms:
            start = lowered.find(term)
            while start != -1:
                masked[start:start + len(term)] = b"*" * len(term)
                start = lowered.find(term, start + 1)
        return "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in masked)

    def _context(self, data: bytes, start: int, end: int) -> str:
        low = max(0, start - CONTEXT_BYTES)
        high = min(len(data), end + CONTEXT_BYTES)
        return ("…" if low else "") + self._mask(data[low:high]) + (
            "…" if high < len(data) else "")

    def _report(self, location: str, check: str,
                hits: list[tuple[int, str, str]]) -> None:
        # The location is printed too, and a path can carry a term.
        shown = self._mask(location.encode("utf-8", "surrogateescape"))
        printed: dict[str, int] = {}
        for offset, what, context in hits:
            printed[what] = printed.get(what, 0) + 1
            if printed[what] <= MAX_PER_KIND:
                self.findings.append(Finding(shown, check, offset, what, context))
            else:
                self.suppressed += 1

    # -- the two checks, over one buffer ------------------------------------

    def _private_hits(self, data: bytes) -> list[tuple[int, str, str]]:
        if not self.terms:
            return []
        lowered = data.lower()
        # A term inside an allowed longer token is not a finding: blank the
        # token out (same length, so offsets still mean what they say).
        for token in self.allowed:
            if token in lowered:
                lowered = lowered.replace(token, b"\x00" * len(token))
        hits: list[tuple[int, str, str]] = []
        for term in self.terms:
            label = self.names.label(term.decode("utf-8"))
            start = lowered.find(term)
            while start != -1:
                hits.append((start, label,
                             self._context(data, start, start + len(term))))
                start = lowered.find(term, start + len(term))
        return sorted(hits)

    def _home_hits(self, data: bytes) -> list[tuple[int, str, str]]:
        return [(match.start(), self._mask(match.group(0)),
                 self._context(data, match.start(), match.end()))
                for match in HOME_PATH.finditer(data)]

    def _scan_buffer(self, location: str, data: bytes) -> None:
        private = self._private_hits(data)
        home = self._home_hits(data)
        # UTF-16 text hides from a byte search: every other byte is a NUL.
        for pattern, first in ((UTF16_LE_RUN, 0), (UTF16_BE_RUN, 1)):
            for run in pattern.finditer(data):
                text = run.group(0)[first::2]
                private += [(run.start() + 2 * offset, what, context)
                            for offset, what, context in self._private_hits(text)]
                home += [(run.start() + 2 * offset, what, context)
                         for offset, what, context in self._home_hits(text)]
        if private:
            self._report(location, "private-name", sorted(private))
        if home:
            self._report(location, "home-path", sorted(home))

    def _scan_name(self, location: str, name: str) -> None:
        data = name.encode("utf-8", "surrogateescape")
        private = [(-1, what, context) for _, what, context in self._private_hits(data)]
        home = [(-1, what, context) for _, what, context in self._home_hits(data)]
        if private:
            self._report(location, "private-name", private[:1])
        if home:
            self._report(location, "home-path", home[:1])

    # -- files and archives -------------------------------------------------

    def scan_bytes(self, location: str, data: bytes, depth: int = 0) -> None:
        self.files += 1
        self._scan_buffer(location, blank_digests(location, data))
        if depth >= MAX_ARCHIVE_DEPTH:
            return
        for member, payload in archive_members(data):
            inner = f"{location}!{member}"
            self._scan_name(inner, member)
            self.scan_bytes(inner, payload, depth + 1)

    def scan_path(self, path: str) -> None:
        """One file, or every file under a directory. Locations are reported
        relative to ``path``'s parent, and only each entry's OWN name is
        searched, so the scanning machine's directories never read as
        findings."""
        base = os.path.dirname(os.path.abspath(path))

        def visit(full: str) -> None:
            relative = os.path.relpath(full, base)
            self._scan_name(relative, os.path.basename(full))
            if os.path.islink(full):
                # The link's TARGET is content too: a link into a home folder
                # leaks that path without any file holding it.
                self._scan_buffer(relative + " (link target)",
                                  os.readlink(full).encode("utf-8", "surrogateescape"))
            elif os.path.isfile(full):
                try:
                    with open(full, "rb") as handle:
                        data = handle.read()
                except OSError as error:
                    self.unreadable.append(f"{relative}: {error.strerror}")
                    return
                self.scan_bytes(relative, data)
            # Anything else (a directory, a socket, a device) has a name and
            # no bytes to read.

        if not os.path.isdir(path) or os.path.islink(path):
            visit(path)
            return
        visit(path)
        for directory, subdirectories, files in os.walk(path):
            subdirectories.sort()
            for name in subdirectories + sorted(files):
                visit(os.path.join(directory, name))   # os.walk follows no links


def archive_members(data: bytes) -> Iterator[tuple[str, bytes]]:
    """``(member name, bytes)`` for a zip (so: a wheel), a tar, or a gzip.
    Recognized by content, never by extension. Anything unreadable is treated
    as not an archive — its raw bytes were already searched."""
    try:
        if data[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    if not info.is_dir():
                        yield info.filename, archive.read(info)
        elif data[:2] == b"\x1f\x8b":
            inner = gzip.decompress(data)
            if _is_tar(inner):
                yield from _tar_members(inner)
            else:
                yield "(gzip contents)", inner
        elif _is_tar(data):
            yield from _tar_members(data)
    except (OSError, EOFError, zipfile.BadZipFile, tarfile.TarError, ValueError):
        return


def _is_tar(data: bytes) -> bool:
    return data[257:262] == b"ustar"


def _tar_members(data: bytes) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if member.issym() or member.islnk():
                yield member.name + " (link target)", member.linkname.encode(
                    "utf-8", "surrogateescape")
            elif member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    yield member.name, handle.read()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="artifact_scan.py",
        description="Search built artifacts for private names and home-folder paths.")
    parser.add_argument("paths", nargs="+", metavar="path",
                        help="an app bundle, an extracted release, a wheel, or any file or directory")
    parser.add_argument("--show-context", action="store_true",
                        help="also print the text around each private-name finding, the "
                             "term masked — for a private terminal, not a public log")
    parser.add_argument("--allow-missing-list", action="store_true",
                        help="with no private-name list, check home-folder paths only "
                             "instead of failing (ignored when "
                             f"{private_names.REQUIRE_VARIABLE}=1)")
    args = parser.parse_args(argv)

    missing = [path for path in args.paths if not os.path.lexists(path)]
    if missing:
        print(f"artifact scan: no such path: {', '.join(missing)}", file=sys.stderr)
        return 2

    names = private_names.load()
    if names is None:
        reason = private_names.absent_reason()
        if private_names.is_required():
            print(f"artifact scan: {private_names.REQUIRE_VARIABLE} is set, but there is "
                  f"{reason}. Put the list there and scan again.", file=sys.stderr)
            return 2
        if not args.allow_missing_list:
            print(f"artifact scan: there is {reason}, so private names cannot be "
                  "checked. Put the list there, or pass --allow-missing-list to "
                  "check home-folder paths only.", file=sys.stderr)
            return 2
        print(f"artifact scan: {reason} — checking home-folder paths ONLY")

    scanner = Scanner(names)
    for path in args.paths:
        scanner.scan_path(path)

    checked = ("private names and home-folder paths" if names is not None
               else "home-folder paths only")
    if scanner.unreadable:
        # An unread file is an unscanned file: never report "clean" over it.
        print(f"artifact scan: {len(scanner.unreadable)} file(s) could not be read, "
              "so the scan is incomplete:", file=sys.stderr)
        for line in scanner.unreadable:
            print(f"  {line}", file=sys.stderr)
        return 2
    if scanner.findings:
        locations = {finding.location for finding in scanner.findings}
        print(f"artifact scan: {len(scanner.findings) + scanner.suppressed} finding(s) "
              f"in {len(locations)} file(s), {scanner.files} file(s) read ({checked})")
        for finding in sorted(scanner.findings,
                              key=lambda f: (f.location, f.check, f.offset)):
            print(f"  {finding.line(args.show_context)}")
        if scanner.suppressed:
            print(f"  … and {scanner.suppressed} more of the same in the files above "
                  f"(the first {MAX_PER_KIND} of each are shown)")
        if any(finding.check == "private-name" for finding in scanner.findings):
            print("A list entry is named by its position among the terms of "
                  f"{names.path.name}. To see the text around a finding, scan again "
                  "with --show-context (in a private terminal). If it is only a "
                  "longer, harmless word that happens to contain the term, add the "
                  "line  allow:<that word>  to the list; otherwise remove the name "
                  "from what was built.")
        return 1
    print(f"artifact scan: clean — {scanner.files} file(s) read ({checked})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
