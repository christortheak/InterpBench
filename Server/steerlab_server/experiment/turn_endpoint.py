"""Declared turn endpoints for multi-agent panels (Wave-2 contract, 2026-08-05).

A scenario turn MAY declare ONE endpoint — the quantity that turn is supposed
to produce::

    "endpoint": {"name": "vote", "kind": "choice", "marker": "Vote:",
                 "vocabulary": ["affirm", "reverse", "vacate", "remand"]}
    "endpoint": {"name": "months", "kind": "number", "marker": "Sentence:",
                 "min": 0, "max": 600}

The runner parses each turn's generated text AT WRITE TIME and stamps the
parse onto the turn record, so a panel's votes are stored data rather than
something a viewer mines out of free text later.

**No regex anywhere in this contract.** Parsing is a literal, case-insensitive
scan: find the marker, then read the following 80 characters. Two engines
byte-agree on literal scans trivially; regex dialects diverge (character
classes, boundary semantics, greediness), and this file has a Swift twin
(``Sources/ExperimentKit/MultiAgent/TurnEndpoint.swift``) that must agree
character for character. The committed golden fixture
``prompts/fixtures/panel-endpoints/`` is exercised by BOTH engines' tests.

Pinned edge decisions (the twin implements each one identically):

* **Case-insensitivity is ASCII folding** (A–Z ↔ a–z); every other character
  compares exactly. Unicode lowercasing is not length-preserving (``'İ'``
  lowercases to two code points), which would desynchronise the folded index
  from the original text — and the two engines' Unicode tables need not agree
  on the same release.
* **Positions are Unicode code points**, matching Python string indexing and
  Swift's ``String.UnicodeScalarView``.
* **The window is the 80 code points following the marker**, and a match must
  fit ENTIRELY inside it. A value that starts inside the window and runs past
  it is unparsed, never a truncated number.
* **Whole-word means not adjacent to a letter** (Unicode general category
  L*), checked against the FULL text so a word clipped by the window edge is
  correctly seen as clipped. This is what keeps ``affirm`` out of
  ``reaffirmed``.
* **Choice takes the earliest POSITION**, not the declaration order. Ties
  (only reachable via a duplicated vocabulary) break on longest, then
  declaration order.
* The stamped choice value is the DECLARED vocabulary member, not the matched
  substring, so counts do not fragment across casings.
* **The marker's FIRST occurrence is the only one scanned.** If its window
  holds no value the turn is unparsed even when a later marker would have
  parsed — "the first thing it said" is the endpoint, not the most convenient
  thing it said.
* Number: optional sign, digits, optional ``.``-fraction; the FIRST such
  number in the window, refused when outside a declared ``min``/``max``
  (inclusive). An out-of-range first number is unparsed — the scan does not
  go looking for a second, more agreeable number.

Unparsed is recorded as unparsed, never guessed.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass

#: Code points scanned after the marker.
WINDOW = 80

KINDS = ("choice", "number")

PANEL_ENDPOINTS_HEADER = ["condition", "replicateIndex", "speakerName",
                          "turnTitle", "endpoint", "value", "unparsed"]


class EndpointError(Exception):
    """A malformed endpoint DECLARATION.

    Loud on purpose: the scenario is pinned, reviewed data, and a typo'd
    declaration that silently parses nothing would look exactly like a model
    that never answered.
    """


@dataclass(frozen=True)
class TurnEndpoint:
    name: str
    kind: str
    marker: str
    vocabulary: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, d: dict, *, turn: str = "") -> "TurnEndpoint":
        where = f"turn '{turn}': " if turn else ""
        if not isinstance(d, dict):
            raise EndpointError(f"{where}endpoint must be an object")
        name = str(d.get("name", "")).strip()
        if not name:
            raise EndpointError(f"{where}endpoint needs a name")
        kind = str(d.get("kind", "")).strip()
        if kind not in KINDS:
            raise EndpointError(
                f"{where}endpoint '{name}' has unknown kind '{kind}' — "
                f"expected one of {', '.join(KINDS)}")
        marker = str(d.get("marker", ""))
        if not marker.strip():
            raise EndpointError(
                f"{where}endpoint '{name}' needs a non-empty marker")
        raw_vocabulary = d.get("vocabulary")
        minimum, maximum = d.get("min"), d.get("max")
        if kind == "choice":
            if not isinstance(raw_vocabulary, list) or not raw_vocabulary:
                raise EndpointError(
                    f"{where}choice endpoint '{name}' needs a non-empty "
                    "vocabulary")
            vocabulary = []
            for member in raw_vocabulary:
                text = str(member)
                if not text.strip():
                    raise EndpointError(
                        f"{where}choice endpoint '{name}' has an empty "
                        "vocabulary member")
                vocabulary.append(text)
            if minimum is not None or maximum is not None:
                raise EndpointError(
                    f"{where}choice endpoint '{name}' declares min/max, which "
                    "only a number endpoint reads")
            return cls(name=name, kind=kind, marker=marker,
                       vocabulary=tuple(vocabulary))
        if raw_vocabulary is not None:
            raise EndpointError(
                f"{where}number endpoint '{name}' declares a vocabulary, "
                "which only a choice endpoint reads")
        bounds: list[float | None] = []
        for key, value in (("min", minimum), ("max", maximum)):
            if value is None:
                bounds.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EndpointError(
                    f"{where}number endpoint '{name}' has a non-numeric {key}")
            bounds.append(float(value))
        if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            raise EndpointError(
                f"{where}number endpoint '{name}' has min > max")
        return cls(name=name, kind=kind, marker=marker,
                   minimum=bounds[0], maximum=bounds[1])

    def to_dict(self) -> dict:
        out: dict = {"name": self.name, "kind": self.kind, "marker": self.marker}
        if self.kind == "choice":
            out["vocabulary"] = list(self.vocabulary)
        else:
            if self.minimum is not None:
                out["min"] = self.minimum
            if self.maximum is not None:
                out["max"] = self.maximum
        return out


# --- the parser -------------------------------------------------------------

def _fold(text: str) -> str:
    """ASCII case fold. Length-preserving by construction, so an index into
    the folded string is an index into the original."""
    return "".join(
        chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch for ch in text)


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_digit(ch: str) -> bool:
    # ASCII digits only: a Unicode decimal (e.g. Arabic-Indic) would parse
    # here and then diverge from the Swift twin, or fail float() outright.
    return "0" <= ch <= "9"


def _marker_end(text: str, marker: str) -> int | None:
    """Index just past the FIRST case-insensitive occurrence of ``marker``."""
    index = _fold(text).find(_fold(marker))
    return None if index < 0 else index + len(marker)


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """Whole-word: neither neighbour in the FULL text is a letter."""
    if start > 0 and _is_letter(text[start - 1]):
        return False
    if end < len(text) and _is_letter(text[end]):
        return False
    return True


def _parse_choice(endpoint: TurnEndpoint, text: str, start: int) -> str | None:
    window_end = min(len(text), start + WINDOW)
    folded = _fold(text)
    best: tuple[int, int, int, str] | None = None
    for order, member in enumerate(endpoint.vocabulary):
        needle = _fold(member)
        index = folded.find(needle, start)
        while index != -1 and index + len(needle) <= window_end:
            if _boundary_ok(text, index, index + len(needle)):
                candidate = (index, -len(needle), order, member)
                if best is None or candidate < best:
                    best = candidate
                break
            index = folded.find(needle, index + 1)
    return None if best is None else best[3]


def _parse_number(endpoint: TurnEndpoint, text: str,
                  start: int) -> float | None:
    window_end = min(len(text), start + WINDOW)
    index = start
    while index < window_end:
        cursor = index
        if text[cursor] in "+-":
            cursor += 1
        if cursor >= window_end or not _is_digit(text[cursor]):
            index += 1
            continue
        while cursor < window_end and _is_digit(text[cursor]):
            cursor += 1
        if (cursor + 1 < window_end and text[cursor] == "."
                and _is_digit(text[cursor + 1])):
            cursor += 1
            while cursor < window_end and _is_digit(text[cursor]):
                cursor += 1
        # Clipped by the window edge: the digits continue past it, so what we
        # can see is a prefix of the number, not the number. Reading it would
        # invent a value ("6" out of "600") — the worst possible failure for
        # a numeric endpoint.
        if cursor == window_end and cursor < len(text):
            tail = text[cursor]
            if _is_digit(tail) or (tail == "." and cursor + 1 < len(text)
                                   and _is_digit(text[cursor + 1])):
                return None
        value = float(text[index:cursor])
        if endpoint.minimum is not None and value < endpoint.minimum:
            return None
        if endpoint.maximum is not None and value > endpoint.maximum:
            return None
        return value
    return None


def parse(endpoint: TurnEndpoint, text: str) -> str | float | None:
    """The declared value in ``text``, or ``None`` when it cannot be read."""
    start = _marker_end(text or "", endpoint.marker)
    if start is None:
        return None
    if endpoint.kind == "choice":
        return _parse_choice(endpoint, text, start)
    return _parse_number(endpoint, text, start)


def stamp(endpoint: TurnEndpoint, text: str) -> dict:
    """The turn-record stamp: a value, or an explicit unparsed marker.

    Swift twin: ``TurnEndpointParser.stamp``. The record keeps the full text
    either way — this never replaces the evidence, only indexes it.
    """
    value = parse(endpoint, text)
    if value is None:
        return {"name": endpoint.name, "value": None, "unparsed": True}
    return {"name": endpoint.name, "value": value}


# --- analyze aggregation ----------------------------------------------------

def format_value(value) -> str:
    """CSV/count-key rendering. Integral floats lose the ``.0`` so a 24-month
    sentence counts as ``24``, not ``24.0``."""
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    return str(value)


def csv_rows(records: list[dict]) -> list[list]:
    """One row per endpoint-stamped record, in run order."""
    rows: list[list] = []
    for record in records:
        endpoint = record.get("endpoint")
        if not isinstance(endpoint, dict) or not endpoint.get("name"):
            continue
        unparsed = bool(endpoint.get("unparsed")) or endpoint.get("value") is None
        rows.append([
            record.get("condition", ""),
            record.get("replicateIndex", 0) or 0,
            record.get("speakerName") or "",
            record.get("turnTitle") or "",
            endpoint.get("name"),
            "" if unparsed else format_value(endpoint.get("value")),
            1 if unparsed else 0,
        ])
    return rows


def write_csv(path: str, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PANEL_ENDPOINTS_HEADER)
        for row in rows:
            writer.writerow(row)


def counts(records: list[dict]) -> dict:
    """Value counts per (endpoint × speaker × condition), plus unparsed counts.

    Speaker is the seat, which is the unit a panel result is read by: "who
    voted how, under which arm". Unparsed is counted alongside rather than
    dropped — a seat that stopped answering the question IS a finding.
    """
    out: dict = {}
    for record in records:
        endpoint = record.get("endpoint")
        if not isinstance(endpoint, dict) or not endpoint.get("name"):
            continue
        speaker = record.get("speakerName") or ""
        condition = record.get("condition", "")
        cell = out.setdefault(endpoint["name"], {}).setdefault(
            speaker, {}).setdefault(
            condition, {"values": {}, "unparsed": 0, "n": 0})
        cell["n"] += 1
        if bool(endpoint.get("unparsed")) or endpoint.get("value") is None:
            cell["unparsed"] += 1
        else:
            key = format_value(endpoint.get("value"))
            cell["values"][key] = cell["values"].get(key, 0) + 1
    return out
