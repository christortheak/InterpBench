"""Voice lint for multi-agent turns (Wave-2 contract, spec §5, 2026-08-17).

A panel turn is supposed to be ONE participant's document. The measured runs
show three ways that fails, and all three are condition-correlated — steered
arms lose format compliance where baseline arms keep it — so voice
noncompliance is an OUTCOME to record, never noise to hide. Nothing here
blocks, retries, or regenerates: silent regeneration would select on the
dependent variable. The runner stamps each turn record at write time::

    "voiceLint": {"version": 1, "speaksForOthers": true,
                  "otherSpeakerLines": {"Judge Marsden": 3},
                  "thirdPersonSelf": 2}

``otherSpeakerLines`` is omitted when empty — additive-key discipline, the
same rule the ``endpoint`` stamp follows.

**No regex anywhere**, matching the ``turn_endpoint`` house style: scanning is
literal and line-oriented, because two engines byte-agree on literal scans
trivially while regex dialects diverge. This file has a Swift twin
(``Sources/ExperimentKit/MultiAgent/VoiceLint.swift``) that must agree case for
case, and the committed fixture ``prompts/fixtures/voice-lint/cases.jsonl``
(excerpts from the real failing transcripts) is replayed by BOTH engines'
tests.

The failure shapes, and the rules calibrated against them
---------------------------------------------------------

The corpus: the workspace's four ``multiagent-s3-prec-delib-*`` runs of
2026-08-11 (600 turn records; 120 round-4 opinions) and
``20260808T175038808-…-consciousness-a`` (240 records).

**1. Colleague signature blocks** — a judge filing its colleagues' separate
opinions inside its own::

    **WHITFIELD, Judge,** concurring. I join Judge Marsden's opinion in full.
    **CALLOWAY, J.,** dissenting.
    MARSDEN, Circuit Judge, concurring in the judgment.

**2. Screenplay speaker labels / stage directions** — a whole-panel transcript
written by one seat::

    **(Judge A, as presiding judge):** Good morning, Judges B and C.
    **(Judge B):** Certainly. My initial vote is to affirm.

**3. Third-person self-reference** — the speaker naming itself as a third
party ("given Judge A's scale position", written by Judge A).

Pinned decisions (the Swift twin implements each one identically):

* **Name forms come from the agent's own name, never from a vocabulary.** No
  judicial (or any domain) word appears in this file: the "role" half of a
  signature is whatever the scenario's agent name happens to carry. An agent
  ``"Judge Whitfield"`` yields the FULL form ``Judge Whitfield`` and the
  SURNAME form ``Whitfield`` (last whitespace-delimited token). The surname
  form is used only when it is ≥3 characters, so a panel of ``Judge A`` /
  ``Judge B`` does not match every line starting with "A".
* **The two forms carry different evidence, and that is the whole trick.**
  Across 600 records the transcripts use the FULL name to address or discuss a
  colleague ("Judge Calloway raises a compelling point") and the BARE SURNAME
  only in signature position ("CALLOWAY, J., concurring"). So a surname
  followed by a comma is a signature block; a full name followed by a comma is
  legitimate address and is NOT flagged.
* **Only the line's leading content is scanned.** A colleague named mid-line is
  discussion, which the panel is supposed to do.
* Leading Markdown/quotation decoration (``* _ # > - + ~ ` ( [ " '`` and
  whitespace) is stripped first; an opening ``(`` or ``[`` among it marks the
  line BRACKETED, which is what distinguishes a stage direction from prose.
* **Case-insensitivity is ASCII folding**, positions are code points, and
  "whole word" means not adjacent to a Unicode letter — the identical
  conventions ``turn_endpoint`` pins, and literally its functions, so the two
  stamps cannot drift apart.
* **A Markdown list item is a roll-call, not a voice.** ``*   **Judge B:**
  Affirm`` inside a disposition package reports a colleague's vote; the
  ``NAME:`` rule is therefore suppressed on list items. Without this, ten of
  ten "final disposition package" turns — a legitimate turn type — flagged.
* **Third-person self counts the FULL own name only**, and skips positions that
  compliant transcripts genuinely use: line-initial (a heading or the
  speaker's own signature), immediately after a field label ending in ``:``
  (``From: Judge Whitfield`` — 60 occurrences in the corpus), after a
  first-person lead-in (``I, Judge C`` / ``I am Judge C`` / ``I'm …`` /
  ``you are …``), and ``As <name>,`` apposition. Everything else counts. It
  counts, it does not judge: a byline without a colon (``**Opinion by Judge
  A**``) is counted, deliberately, because the alternative is a heuristic that
  guesses at prose.
"""

from __future__ import annotations

import csv

from .turn_endpoint import _fold, _is_letter

#: Stamp schema version. Bump only for a shape change, never for a rule tweak
#: — a rules change is a re-lint of the corpus, which the fixture pins.
VERSION = 1

#: Leading characters removed before a line's content is examined.
_DECORATION = set("*_#>-+~`([\"' \t\r")
#: Characters skipped when looking for the first significant character AFTER a
#: matched name (emphasis markers close there: ``**WHITFIELD, Judge,** …``).
_SKIP = set("*_`~ \t\r")
#: Continuations that make a bare SURNAME at the head of a line a signature.
_SIGNATURE_AFTER = ("", ",", ".", ";")
#: Continuations that make a BRACKETED name a speaker label / stage direction.
_BRACKET_AFTER = ("", ",", ")", "]")
#: First-person lead-ins that make an own-name mention first-person framing.
_FIRST_PERSON = ("i,", "i am", "i'm")
#: Second-person echo of the prompt's identity line.
_SECOND_PERSON = ("you are",)

#: The roll-up artifact. PER-SHARD by construction — each shard can only
#: aggregate the transcripts it ran — so the shard merge rebuilds it over the
#: merged records rather than comparing partials (``sharding._PER_SHARD_FILES``).
VOICE_LINT_FILENAME = "panel-voice-lint.csv"

VOICE_LINT_HEADER = [
    "condition", "speakerName", "turns", "speaksForOthersTurns",
    "speaksForOthersRate", "otherSpeakerLines", "thirdPersonSelfTurns",
    "thirdPersonSelfTotal", "thirdPersonSelfMean",
]


# --- name forms -------------------------------------------------------------

def name_forms(name: str) -> list[tuple[str, str]]:
    """``[(form, kind)]`` for one agent name, longest first.

    Kinds are ``"full"`` and ``"surname"``. Derived from the name alone: the
    linter never knows what domain the panel is in.
    """
    normalized = " ".join(str(name or "").split())
    if not normalized:
        return []
    forms = [(normalized, "full")]
    tokens = normalized.split(" ")
    if len(tokens) > 1 and len(tokens[-1]) >= 3:
        forms.append((tokens[-1], "surname"))
    forms.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    return forms


# --- line scanning ----------------------------------------------------------

def _strip_leading(line: str) -> tuple[str, bool]:
    """The line's content after decoration, and whether a bracket opened it."""
    index = 0
    bracketed = False
    while index < len(line) and line[index] in _DECORATION:
        if line[index] in "([":
            bracketed = True
        index += 1
    return line[index:], bracketed


def is_list_item(line: str) -> bool:
    """A Markdown bullet or ordered-list marker opens this line."""
    index = 0
    while index < len(line) and line[index] in " \t":
        index += 1
    if (index < len(line) and line[index] in "*-+"
            and index + 1 < len(line) and line[index + 1] in " \t"):
        return True
    digits = index
    while digits < len(line) and "0" <= line[digits] <= "9":
        digits += 1
    return (digits > index and digits + 1 < len(line)
            and line[digits] in ".)" and line[digits + 1] in " \t")


def _first_significant(text: str) -> str:
    for char in text:
        if char not in _SKIP:
            return char
    return ""


def line_speaker(line: str, names: list[str]) -> str | None:
    """The agent this LINE signs, labels, or stage-directs as, or ``None``.

    Ambiguity resolves by longest form across all candidate names, so a panel
    holding both ``Judge Marsden`` and ``Marsden`` cannot flag the short one
    for the long one's line.
    """
    head, bracketed = _strip_leading(line)
    if not head:
        return None
    listed = is_list_item(line)
    folded = _fold(head)
    candidates: list[tuple[int, str, str, str]] = []
    for name in names:
        for form, kind in name_forms(name):
            candidates.append((-len(form), form, kind, name))
    candidates.sort(key=lambda c: (c[0], c[1], c[3]))
    for _, form, kind, name in candidates:
        if not folded.startswith(_fold(form)):
            continue
        rest = head[len(form):]
        if rest and _is_letter(rest[0]):
            continue  # a longer word that merely starts with the name
        after = _first_significant(rest)
        if after == ":" and not listed:
            return name          # "Judge B:" — speaker label or salutation
        if kind == "surname" and after in _SIGNATURE_AFTER:
            return name          # "CALLOWAY, J., dissenting."
        if bracketed and after in _BRACKET_AFTER:
            return name          # "(Judge A, as presiding judge):"
        return None              # the name leads the line, but as prose
    return None


def other_speaker_lines(text: str, others: list[str]) -> dict[str, int]:
    """How many lines of ``text`` sign or label as each OTHER participant."""
    counts: dict[str, int] = {}
    names = [name for name in others if " ".join(str(name or "").split())]
    if not names:
        return counts
    for line in (text or "").split("\n"):
        name = line_speaker(line, names)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return counts


# --- third-person self ------------------------------------------------------

def _whole_word_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    folded_text, folded_needle = _fold(text), _fold(needle)
    out: list[int] = []
    index = folded_text.find(folded_needle)
    while index != -1:
        end = index + len(folded_needle)
        before_ok = index == 0 or not _is_letter(text[index - 1])
        after_ok = end >= len(text) or not _is_letter(text[end])
        if before_ok and after_ok:
            out.append(index)
        index = folded_text.find(folded_needle, index + 1)
    return out


def _is_trailing_skippable(char: str) -> bool:
    """Ignored when looking BACK from a match: whitespace and emphasis markers.

    Spelled out rather than delegated to ``str.rstrip()``, whose whitespace set
    is Unicode-wide and would not match the Swift twin's explicit set.
    """
    return char in _SKIP or char == "\n"


def _ends_with_phrase(text: str, end: int, phrase: str) -> bool:
    """``text[:end]`` ends with ``phrase`` (ASCII-folded), on a word boundary."""
    start = end - len(phrase)
    if start < 0 or _fold(text[start:end]) != phrase:
        return False
    return start == 0 or not _is_letter(text[start - 1])


def third_person_self(text: str, speaker: str) -> int:
    """Occurrences of the speaker's own full name in non-first-person prose."""
    name = " ".join(str(speaker or "").split())
    if not name:
        return 0
    body = text or ""
    count = 0
    for index in _whole_word_occurrences(body, name):
        line_start = index
        while line_start > 0 and body[line_start - 1] != "\n":
            line_start -= 1
        content = line_start
        while content < len(body) and body[content] in _DECORATION:
            content += 1
        if content == index:
            continue  # line-initial: a heading, or the speaker's own signature

        before = index
        while before > 0 and _is_trailing_skippable(body[before - 1]):
            before -= 1
        if before > 0 and body[before - 1] == ":":
            continue  # a field label: "From: Judge Whitfield"

        if any(_ends_with_phrase(body, before, phrase)
               for phrase in _FIRST_PERSON + _SECOND_PERSON):
            continue
        if (_ends_with_phrase(body, before, "as")
                and _first_significant(body[index + len(name):]) == ","):
            continue  # "As Judge C, I concur …" — first-person framing
        count += 1
    return count


# --- the stamp --------------------------------------------------------------

def stamp(text: str, *, speaker: str, others: list[str]) -> dict:
    """The turn-record stamp. Never raises, never blocks, never rewrites.

    Swift twin: ``VoiceLint.stamp(in:speaker:others:)``.
    """
    lines = other_speaker_lines(text, others)
    out: dict = {"version": VERSION, "speaksForOthers": bool(lines)}
    if lines:
        out["otherSpeakerLines"] = dict(sorted(lines.items()))
    out["thirdPersonSelf"] = third_person_self(text, speaker)
    return out


# --- aggregation ------------------------------------------------------------

def _fmt(value: float) -> str:
    return f"{value:.6g}"


def csv_rows(records: list[dict]) -> list[list]:
    """One row per (speaker × condition), in sorted order.

    The seat is the unit a panel result is read by, and the condition is the
    axis the failure correlates with — the two together are the finding: "this
    seat, under this arm, stopped writing in its own voice this often".
    """
    cells: dict[tuple[str, str], dict] = {}
    for record in records:
        lint = record.get("voiceLint")
        if not isinstance(lint, dict):
            continue
        key = (str(record.get("condition", "")),
               str(record.get("speakerName") or ""))
        cell = cells.setdefault(key, {"turns": 0, "speaks": 0, "lines": 0,
                                      "selfTurns": 0, "self": 0})
        cell["turns"] += 1
        if bool(lint.get("speaksForOthers")):
            cell["speaks"] += 1
        lines = lint.get("otherSpeakerLines")
        if isinstance(lines, dict):
            cell["lines"] += sum(int(v) for v in lines.values())
        third = int(lint.get("thirdPersonSelf") or 0)
        cell["self"] += third
        if third:
            cell["selfTurns"] += 1
    rows: list[list] = []
    for (condition, speaker) in sorted(cells):
        cell = cells[(condition, speaker)]
        turns = cell["turns"]
        rows.append([
            condition, speaker, turns, cell["speaks"],
            _fmt(cell["speaks"] / turns), cell["lines"], cell["selfTurns"],
            cell["self"], _fmt(cell["self"] / turns),
        ])
    return rows


def write_csv(path: str, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(VOICE_LINT_HEADER)
        for row in rows:
            writer.writerow(row)
