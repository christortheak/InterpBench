#!/usr/bin/env python3
"""The private-name list: where it lives and how it is read.

The one Python loader for it. The neutrality guards in ``Server/tests`` and
the artifact scan (``artifact_scan.py``, beside this file) both come here.
Swift twin: ``Tests/ExperimentKitTests/PrivateNames.swift`` — same file, same
rules.

The list holds the site, study, and account names that must not ship. It is
never in this repository, as text or as digests: the terms are short words,
and a short word's hash is recovered by trying words. It lives in a file the
repository never sees:

* ``$STEERLAB_PRIVATE_NAMES_FILE`` when set, otherwise
  ``~/.steerlab/private-names.txt``;
* one term per line, matched case-insensitively as a substring (deliberately
  blunt: a false positive costs one rename, a false negative ships a name);
* blank lines and ``#`` comments are ignored;
* ``allow:<token>`` names a longer token inside which a term is NOT a
  finding — for an identifier that happens to contain one and has to stay.
  Only the artifact scan honors it; the test guards stay strict.

When the list is absent a caller either skips, saying why, or — with
``STEERLAB_REQUIRE_PRIVATE_NAMES=1`` — fails, so a release gate cannot pass
vacuously because a file went missing. A list with no term counts as absent
for the same reason.

Run directly, this prints whether a list was found and how many terms it
holds. It never prints a term for a person to read.

``--from-ci-secret`` is the continuous-integration half. The main repository
keeps the list in a secret and hands it to a job as the environment variable
``STEERLAB_PRIVATE_NAMES``; this writes it to a temporary file, points
``STEERLAB_PRIVATE_NAMES_FILE`` at that file for the rest of the job, and
makes the list REQUIRED. A fork has no secret, so the variable is empty:
nothing is written, nothing is required, and the guards skip.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

FILE_VARIABLE = "STEERLAB_PRIVATE_NAMES_FILE"
REQUIRE_VARIABLE = "STEERLAB_REQUIRE_PRIVATE_NAMES"
#: The list's CONTENT, as a CI job receives it from a repository secret.
SECRET_VARIABLE = "STEERLAB_PRIVATE_NAMES"
DEFAULT_LOCATION = "~/.steerlab/private-names.txt"
ALLOW_PREFIX = "allow:"


@dataclass(frozen=True, repr=False)
class PrivateNames:
    """A list that was found and holds at least one term. Everything is
    lowercase.

    A term is never put into a message: a failing guard's output lands in a
    CI log anyone can read. ``label`` names a term by its position in the
    list instead, which the list's owner can look up and nobody else can."""

    path: Path
    terms: tuple[str, ...]
    allowed_contexts: tuple[str, ...] = ()

    def __repr__(self) -> str:      # a test runner prints arguments on failure
        return f"PrivateNames({len(self.terms)} term(s) at {self.path})"

    def label(self, term: str) -> str:
        return f"list entry {self.terms.index(term) + 1} ({len(term)} letters)"

    def hits(self, text: str) -> list[str]:
        """The LABELS of the terms ``text`` contains — the strict reading the
        test guards use (``allowed_contexts`` is not consulted)."""
        lowered = text.lower()
        return [self.label(term) for term in self.terms if term in lowered]


def list_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get(FILE_VARIABLE) or DEFAULT_LOCATION).expanduser()


def is_required(environ: Optional[Mapping[str, str]] = None) -> bool:
    environ = os.environ if environ is None else environ
    return environ.get(REQUIRE_VARIABLE, "").strip().lower() in ("1", "true", "yes")


def parse(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(terms, allowed_contexts)`` from a list file's text, lowercased and
    de-duplicated in order."""
    terms: list[str] = []
    allowed: list[str] = []
    for line in text.splitlines():
        entry = line.strip().lower()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith(ALLOW_PREFIX):
            token = entry[len(ALLOW_PREFIX):].strip()
            if token and token not in allowed:
                allowed.append(token)
        elif entry not in terms:
            terms.append(entry)
    return tuple(terms), tuple(allowed)


def load(environ: Optional[Mapping[str, str]] = None) -> Optional[PrivateNames]:
    """The list, or ``None`` when the file is missing, unreadable, or holds
    no term."""
    path = list_path(environ)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    terms, allowed = parse(text)
    if not terms:
        return None
    return PrivateNames(path=path, terms=terms, allowed_contexts=allowed)


def absent_reason(environ: Optional[Mapping[str, str]] = None) -> str:
    """One sentence for a skip or a refusal. Names the place, never a term."""
    return (f"no private-name list with at least one term at "
            f"{list_path(environ)} (set {FILE_VARIABLE} to use another file)")


def install_from_ci_secret(environ: Optional[Mapping[str, str]] = None) -> int:
    """Turn the secret a CI job was handed into the file the guards read.

    Writes ``$RUNNER_TEMP/private-names.txt`` (owner-only), then appends the
    two variables to ``$GITHUB_ENV`` so every later step of the job sees
    them. Each term is also handed to the runner as a log mask (the runner
    consumes that line; it is not shown), so that no later tool can print
    one by accident."""
    environ = os.environ if environ is None else environ
    content = environ.get(SECRET_VARIABLE, "")
    terms, allowed = parse(content)
    if not terms:
        print(f"no {SECRET_VARIABLE} secret in this job (a fork, or the secret is "
              "not set): the private-name guards will skip, and the artifact "
              "scan will check home-folder paths only")
        return 0
    job_env = environ.get("GITHUB_ENV")
    temp = environ.get("RUNNER_TEMP")
    if not job_env or not temp:
        print("--from-ci-secret needs GITHUB_ENV and RUNNER_TEMP, which a GitHub "
              "Actions job sets; outside one, put the list in a file and set "
              f"{FILE_VARIABLE} yourself", file=sys.stderr)
        return 2
    path = Path(temp) / "private-names.txt"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content if content.endswith("\n") else content + "\n")
    with open(job_env, "a", encoding="utf-8") as handle:
        handle.write(f"{FILE_VARIABLE}={path}\n{REQUIRE_VARIABLE}=1\n")
    for term in terms:
        print(f"::add-mask::{term}")
    print(f"private-name list: {len(terms)} term(s), {len(allowed)} allowed "
          "context(s), written for this job and REQUIRED from here on")
    return 0


def main(argv: Optional[list] = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--from-ci-secret"]:
        return install_from_ci_secret()
    if arguments:
        print("usage: private_names.py [--from-ci-secret]", file=sys.stderr)
        return 2
    names = load()
    if names is None:
        print(absent_reason())
        return 1 if is_required() else 0
    print(f"private-name list: {len(names.terms)} term(s), "
          f"{len(names.allowed_contexts)} allowed context(s), at {names.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
