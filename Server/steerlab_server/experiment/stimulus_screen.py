"""Independence-from-task screen for concept stimulus sets.

The circularity firewall's data half: stimuli used to extract a concept must
not themselves contain the STUDY's task-domain vocabulary, or the "concept"
direction partly encodes the task domain and every downstream effect is
confounded. This screen is a cheap regex tripwire, not a substitute for the
human review pass — it catches the obvious leaks before a set gets pinned.

The screen itself is concept- and domain-agnostic (CLAUDE.md design rule):
WHICH vocabulary is forbidden is workspace DATA, resolved in this order:

  1. an explicit ``--vocabulary FILE`` argument;
  2. the workspace's ``prompts/screens/forbidden-vocabulary.json``,
     auto-discovered by walking up from the concept directory;
  3. a judicial default at
     (``prompts/templates/screens/forbidden-vocabulary-judicial.json``) —
     default DATA for one study program, not a built-in rule, and OPTIONAL:
     it is study material, so a released tree does not carry it. When it is
     absent and nothing else resolved, the screen REFUSES (exit 2) naming
     the repair rather than screening against an empty word list — a clean
     verdict from no vocabulary is worse than no screen at all.

The output always names which vocabulary governed the run. File shape::

    {"note": "...", "lists": {"<name>": {"note": "...", "terms": ["...", ...]}}}

Terms match case-insensitively at word starts, as PREFIXES ("incarcerat"
catches incarcerated/incarceration; "court" catches courtyard) — deliberately
over-broad; a flagged line is a prompt for human judgment, not an automatic
rejection.

Usage:
    python -m steerlab_server.experiment.stimulus_screen <concept-dir> \
        [--vocabulary FILE]

Exit codes: 0 clean · 1 findings to review · 2 usage/setup error.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass

STIMULUS_FILES = ("positive.jsonl", "negative.jsonl", "validation.jsonl")

#: Workspace-authored vocabulary, relative to the workspace root.
WORKSPACE_VOCABULARY_RELPATH = os.path.join(
    "prompts", "screens", "forbidden-vocabulary.json")

#: Shipped default DATA (the judicial list), relative to the repo root.
DEFAULT_VOCABULARY_RELPATH = os.path.join(
    "prompts", "templates", "screens", "forbidden-vocabulary-judicial.json")


@dataclass(frozen=True)
class Vocabulary:
    """A loaded forbidden-vocabulary file: flattened terms + provenance."""

    terms: tuple[str, ...]
    list_names: tuple[str, ...]
    path: str

    @property
    def pattern(self) -> re.Pattern[str]:
        return re.compile(
            r"\b(?:" + "|".join(re.escape(t) for t in self.terms) + r")",
            re.IGNORECASE)


def load_vocabulary(path: str) -> Vocabulary:
    """Load and shape-check a vocabulary file.

    Raises ValueError (bad shape), OSError (unreadable), or
    json.JSONDecodeError (not JSON)."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    lists = data.get("lists") if isinstance(data, dict) else None
    if not isinstance(lists, dict) or not lists:
        raise ValueError(
            'expected {"lists": {"<name>": {"note": "...", "terms": [...]}}} '
            "with at least one named list")
    terms: list[str] = []
    names: list[str] = []
    for name in sorted(lists):
        entry = lists[name]
        entry_terms = entry.get("terms") if isinstance(entry, dict) else None
        if (not isinstance(entry_terms, list) or not entry_terms
                or not all(isinstance(t, str) and t.strip() for t in entry_terms)):
            raise ValueError(
                f"list {name!r} needs a non-empty 'terms' array of strings")
        names.append(name)
        terms.extend(t.strip() for t in entry_terms)
    # Order-preserving dedupe: a term repeated across lists screens once.
    return Vocabulary(terms=tuple(dict.fromkeys(terms)),
                      list_names=tuple(names), path=path)


def discover_workspace_vocabulary(concept_directory: str) -> str | None:
    """Walk up from the concept directory looking for the workspace's
    ``prompts/screens/forbidden-vocabulary.json``; None when absent."""
    directory = os.path.abspath(concept_directory)
    for _ in range(16):
        candidate = os.path.join(directory, WORKSPACE_VOCABULARY_RELPATH)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return None


def default_vocabulary_path() -> str:
    """Where the judicial default WOULD live, resolved relative to this module
    — the repo layout is ``<root>/Server/steerlab_server/experiment/<this
    file>`` (both engines install the server editable, so the tree is
    present). The file itself is study DATA and optional: use
    :func:`default_vocabulary_available` before assuming it is there."""
    here = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(root, DEFAULT_VOCABULARY_RELPATH)


def default_vocabulary_available() -> bool:
    """Whether the optional judicial default is present in this checkout. It
    is excluded from the released tree (study material), so every caller that
    falls back to it must handle its absence."""
    return os.path.isfile(default_vocabulary_path())


def screen_text(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Distinct forbidden terms found in one text (empty = clean)."""
    return sorted({match.group(0).lower() for match in pattern.finditer(text)})


def screen_directory(directory: str, pattern: re.Pattern[str]) -> list[dict]:
    """Screen every stimulus/validation line in a concept directory.

    Returns one finding per flagged line: file, 1-based line number, the
    matched terms, and a preview. Missing files are skipped (the layout
    contract is enforced elsewhere)."""
    findings: list[dict] = []
    for name in STIMULUS_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    text = json.loads(raw).get("text", "")
                except json.JSONDecodeError:
                    findings.append({"file": name, "line": line_number,
                                     "terms": [], "error": "invalid JSON"})
                    continue
                terms = screen_text(text, pattern)
                if terms:
                    findings.append({"file": name, "line": line_number,
                                     "terms": terms,
                                     "preview": text[:100]})
    return findings


def resolve_vocabulary(directory: str,
                       explicit_path: str | None) -> tuple[str, str]:
    """(path, provenance sentence) per the documented resolution order."""
    if explicit_path is not None:
        return explicit_path, f"vocabulary file {explicit_path} (--vocabulary)"
    discovered = discover_workspace_vocabulary(directory)
    if discovered is not None:
        return discovered, f"workspace file {discovered}"
    return default_vocabulary_path(), (
        "shipped judicial-study default "
        f"({DEFAULT_VOCABULARY_RELPATH} — create "
        f"{WORKSPACE_VOCABULARY_RELPATH} in the workspace to replace it)")


def main(argv: list[str]) -> int:
    directory: str | None = None
    vocabulary_arg: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--vocabulary":
            if index + 1 >= len(argv):
                print("--vocabulary requires a file path", file=sys.stderr)
                return 2
            vocabulary_arg = argv[index + 1]
            index += 2
        elif argument.startswith("-"):
            print(f"unknown argument: {argument}", file=sys.stderr)
            print(__doc__.strip(), file=sys.stderr)
            return 2
        elif directory is None:
            directory = argument
            index += 1
        else:
            print(__doc__.strip(), file=sys.stderr)
            return 2
    if directory is None:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    if not os.path.isdir(directory):
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    path, provenance = resolve_vocabulary(directory, vocabulary_arg)
    # The fallback default is study DATA and does not ship. Say so in terms
    # of the repair, instead of an ENOENT against a path the caller never
    # named — the screen refuses rather than passing against no word list.
    if (vocabulary_arg is None and path == default_vocabulary_path()
            and not default_vocabulary_available()):
        print(f"no forbidden-vocabulary list resolved for {directory}",
              file=sys.stderr)
        print("this checkout carries no default list "
              f"({DEFAULT_VOCABULARY_RELPATH} is study data and is not "
              "shipped). Author "
              f"{WORKSPACE_VOCABULARY_RELPATH} in the workspace, or pass "
              "--vocabulary FILE.", file=sys.stderr)
        return 2
    try:
        vocabulary = load_vocabulary(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cannot load vocabulary {path}: {error}", file=sys.stderr)
        return 2
    # ALWAYS name the governing vocabulary — a clean verdict against the
    # wrong word list is worse than no screen at all.
    print(f"vocabulary: {provenance} — "
          f"{len(vocabulary.terms)} term(s) in list(s): "
          f"{', '.join(vocabulary.list_names)}")

    findings = screen_directory(directory, vocabulary.pattern)
    if not findings:
        print(f"{directory}: clean — no forbidden-vocabulary terms "
              "in stimulus text")
        return 0
    for finding in findings:
        if finding.get("error"):
            print(f"{finding['file']}:{finding['line']}: {finding['error']}")
        else:
            print(f"{finding['file']}:{finding['line']}: "
                  f"{', '.join(finding['terms'])} — {finding['preview']!r}")
    print(f"{len(findings)} flagged line(s); review before pinning this set")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
