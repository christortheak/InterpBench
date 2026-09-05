"""The server's generated regions of ``docs/CLI-REFERENCE.md`` (WP0 step 11).

Swift twin: ``Sources/ExperimentKit/CLIReferenceDocument.swift``. The audit's
disposition (§5.2, §5.3), followed literally:

* only the SYNOPSIS is generated — verb, arity, flags, one-line purpose. Well
  over half the reference document is essay (rationale, incident history, the
  security clause) and that material is the document's real value; every
  hand-written paragraph and semantic flag table survives untouched outside the
  markers;
* generation is NOT a build step. Each engine prints its own regions, the
  result is COMMITTED, and a test compares — the document must be readable on
  GitHub with no toolchain, and a committed artifact diffs reviewably.

Both engines write into the same file and each owns disjoint region ids:
``server-*`` here, ``swift-*`` there. A generator never touches a region it
does not own, so the two can run in either order.
"""

from __future__ import annotations

import os

from . import cli_help
from .cli_envelope import VERB_SPECS

#: Region id → the verb labels it covers, in the document's reading order. A
#: declared verb missing from this map is a generation failure, not a silent
#: omission (``test_cli_reference_regions_match`` asserts the coverage).
REGIONS: dict = {
    "server-experiment": [
        "experiment list", "experiment verify", "experiment extract",
        "experiment validate", "experiment sweep", "experiment run",
        "experiment evaluate", "experiment analyze", "experiment promote",
        "experiment confirm",
    ],
    "server-study": ["study submit"],
    "server-jobs": ["jobs list"],
    "server-model": ["model capabilities"],
    "server-vectors": ["vectors compare", "vectors mirror-poles"],
    "server-site": ["site qualify"],
    "server-data": ["data check"],
    "server-battery": ["battery run"],
}

REGENERATION_NOTE = (
    "<!-- Generated from the declarative verb table — `steerlab-server docs "
    "cli-reference --write`. Edit the table, not this block. -->")

SHARED_FLAG_NOTE = (
    "Every verb above also accepts `--help` (print its arguments and run "
    "nothing), `--json` (one envelope on stdout), and `--out <file>`.")


def begin_marker(region_id: str) -> str:
    return f"<!-- GENERATED:{region_id} BEGIN -->"


def end_marker(region_id: str) -> str:
    return f"<!-- GENERATED:{region_id} END -->"


def _spec(label: str):
    for spec in VERB_SPECS:
        if spec.label == label:
            return spec
    raise KeyError(f"CLI-REFERENCE region names an undeclared verb {label!r}")


def body(region_id: str) -> str:
    """One region's body, markers excluded."""
    specs = [_spec(label) for label in REGIONS[region_id]]
    lines = [REGENERATION_NOTE, "", "```"]
    lines += [cli_help.synopsis(spec, include_shared_flags=False)
              for spec in specs]
    lines += ["```", "", "| Verb | Purpose |", "|---|---|"]
    lines += [f"| `{spec.label}` | {spec.purpose} |" for spec in specs]
    lines += ["", SHARED_FLAG_NOTE]
    return "\n".join(lines)


def generated_bodies() -> dict:
    """Every region this engine owns, by id."""
    return {region_id: body(region_id) for region_id in sorted(REGIONS)}


def extract(document: str, region_id: str) -> str:
    """The committed body of one region, markers excluded."""
    begin, end = begin_marker(region_id), end_marker(region_id)
    start = document.find(begin)
    if start < 0:
        raise ValueError(f"docs/CLI-REFERENCE.md has no region marker {begin}")
    stop = document.find(end, start)
    if stop < 0:
        raise ValueError(
            f"docs/CLI-REFERENCE.md has no closing marker {end} after {begin}")
    return document[start + len(begin):stop].strip("\n")


def drift(document: str) -> list:
    """The region ids whose committed text differs from what the table
    generates. Empty means in sync — the drift gate's whole answer."""
    return [region_id for region_id, text in sorted(generated_bodies().items())
            if extract(document, region_id) != text]


def rewrite(document: str) -> str:
    """Replace this engine's regions, leaving every other byte — including the
    Swift engine's regions — exactly as it was."""
    text = document
    for region_id, generated in sorted(generated_bodies().items()):
        begin, end = begin_marker(region_id), end_marker(region_id)
        start = text.find(begin)
        stop = text.find(end, start) if start >= 0 else -1
        if start < 0 or stop < 0:
            raise ValueError(
                f"docs/CLI-REFERENCE.md has no {begin} … {end} region")
        text = (text[:start + len(begin)] + "\n" + generated + "\n"
                + text[stop:])
    return text


def committed_path() -> str | None:
    """Where the committed document lives in a code checkout, or None outside
    one — a deployed server ships no ``docs/``, and saying so is better than
    writing a document nobody will read."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "docs", "CLI-REFERENCE.md")
    return path if os.path.exists(path) else None
