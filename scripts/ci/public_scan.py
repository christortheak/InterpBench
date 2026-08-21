#!/usr/bin/env python3
"""Public-tier release scan — the half of the release hygiene checks that
ships (CI runs it on every push; release gate 12's permanent enforcement).

Checks TRACKED files only (`git ls-files`), because CI's question is "what
does this commit publish", not "what is lying around":

  1. secret-shaped content (API tokens, AWS keys, private-key blocks)
  2. personal absolute paths (/Users/<name>, /home/<name>, iCloud containers)
  3. email addresses
  4. junk that should never be tracked (.DS_Store, __pycache__, *.pyc,
     .build/, DerivedData/)
  5. anything tracked beneath Workspaces/ or workspaces/ (the home-layout
     rule: workspaces live BESIDE the checkout, never inside it)

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

import os
import re
import subprocess
import sys

MAX_CONTENT_BYTES = 4 * 1024 * 1024  # skip anything larger — binaries, data

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
EMAIL_PATTERN = re.compile(
    rf"\b(?!{_FAKE_USERS}@)[\w.+-]+@"
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
            if os.path.getsize(full) > MAX_CONTENT_BYTES:
                continue
            with open(full, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        if b"\0" in raw[:8192]:
            continue  # binary
        text = raw.decode("utf-8", "replace")

        for check, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                report(path, check, "secret-shaped content")
                break
        if PATH_PATTERN.search(text) or ICLOUD_PATTERN.search(text):
            report(path, "path", "personal absolute path")
        email = EMAIL_PATTERN.search(text)
        if email:
            report(path, "email", email.group(0))

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
