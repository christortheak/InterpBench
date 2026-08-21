"""Workspace-declared numeric answer parsers (USABILITY-PLAN Phase-4 item 18).

Numeric-outcome grammars become workspace DATA, not app code: a study names a
parser from ``prompts/parsers/parser-registry.json`` (manifest key
``numericParser``) and the run parses its numeric outcome with that declared
grammar instead of the built-in duration parser. Two kinds exist:

- ``durationMonths``: a unit table as data (token -> months multiplier),
  joiner words as data, compound support ("8 years 3 months" = 99, both
  terms count), ranges under a declared policy. The SHIPPED default entry
  ("sentencing-months", named for the judicial-decision study it was written
  for) reproduces :func:`judicial.parse_months` exactly — fixture-locked on
  both engines.
- ``number``: plain numeric extraction with declared percent and range
  policies.

Cross-engine contract (Swift twin: ``ExperimentKit/ParserRegistry.swift``):
the registry file path, the manifest keys (``numericParser`` +
``parserRegistryHash``), the spec vocabulary (kinds, range/percent policies),
and the report.json ``numericParser`` provenance block are identical. When no
parser is named, behavior is exactly the historical one (``caseFamily ==
"sentencing"`` -> built-in ``parse_months``); no new file is required and
legacy studies gain no violations. That implicit selection is DEPRECATED as
of 2026-08-18: it still works, and every site where it fires now says so
(``manifest.IMPLICIT_CASE_FAMILY_ADVISORY``, CLI advisory code
``deprecatedImplicitSelection``). Declaring a parser is the mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from . import judicial, paths

#: The one workspace location the registry lives at (seeded into new
#: workspaces from the repo template; both engines resolve the same path).
REGISTRY_FILE = "prompts/parsers/parser-registry.json"

KNOWN_KINDS = ("durationMonths", "number")
RANGE_POLICIES = ("mean", "refuse", "first")
PERCENT_POLICIES = ("accept", "refuse", "fraction")


class ParserRegistryError(ValueError):
    """A registry problem, phrased for the researcher: what was looked for,
    what was found, and what to do — never a raw regex or traceback."""


def registry_path(root: str | None = None) -> str:
    base = paths.project_root() if root is None else root
    return os.path.join(base, REGISTRY_FILE)


def registry_live_hash(root: str | None = None) -> str | None:
    """SHA-256 over the registry file's raw bytes, or None when absent —
    the value freeze pins as ``parserRegistryHash``."""
    path = registry_path(root)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_registry(root: str | None = None) -> dict:
    """The parsed registry, or a plain-language refusal."""
    path = registry_path(root)
    if not os.path.isfile(path):
        raise ParserRegistryError(
            f"no parser registry exists at {REGISTRY_FILE} — new workspaces "
            "are seeded with a template; create the file (or remove the "
            "study's numericParser)")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParserRegistryError(
            f"the parser registry {REGISTRY_FILE} is not readable as JSON "
            f"({exc}) — it must be an object like "
            '{"schemaVersion": 1, "parsers": {"<name>": {...}}}') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("parsers"), dict):
        raise ParserRegistryError(
            f"the parser registry {REGISTRY_FILE} must be an object with a "
            '"parsers" object mapping parser names to their declarations — '
            "start from the shipped template")
    if payload.get("schemaVersion") != 1:
        raise ParserRegistryError(
            f"the parser registry {REGISTRY_FILE} declares schemaVersion "
            f"{payload.get('schemaVersion')!r} — this engine reads version 1")
    return payload


def parser_spec(name: str, root: str | None = None) -> dict:
    """The validated spec for one named parser."""
    registry = load_registry(root)
    spec = registry["parsers"].get(name)
    if not isinstance(spec, dict):
        known = ", ".join(sorted(registry["parsers"])) or "none"
        raise ParserRegistryError(
            f"the registry defines no parser named '{name}' — defined: "
            f"{known}")
    validate_spec(name, spec)
    return spec


def validate_spec(name: str, spec: dict) -> None:
    """Shape-check one parser declaration (the pin-time schema-validation
    rule: say what is wrong at the moment of declaration, not at the failing
    run)."""
    kind = spec.get("kind")
    if kind not in KNOWN_KINDS:
        raise ParserRegistryError(
            f"parser '{name}' declares kind {kind!r} — known kinds: "
            + ", ".join(KNOWN_KINDS))
    range_policy = spec.get("range")
    if range_policy is not None and range_policy not in RANGE_POLICIES:
        raise ParserRegistryError(
            f"parser '{name}' declares range {range_policy!r} — declare one "
            "of " + ", ".join(RANGE_POLICIES)
            + " (what to do with '5-7': the midpoint, a parse failure, or "
            "the first value)")
    if kind == "durationMonths":
        units = spec.get("units")
        if not isinstance(units, dict) or not units:
            raise ParserRegistryError(
                f"parser '{name}' is a durationMonths parser but declares no "
                '"units" table — declare unit tokens mapping to a months '
                'multiplier, e.g. {"years": 12, "months": 1}')
        for token, multiplier in units.items():
            if not isinstance(token, str) or not token.strip():
                raise ParserRegistryError(
                    f"parser '{name}': every units key must be a non-empty "
                    "unit token (a word like 'years')")
            if not isinstance(multiplier, (int, float)) \
                    or isinstance(multiplier, bool) or multiplier <= 0:
                raise ParserRegistryError(
                    f"parser '{name}': unit '{token}' has multiplier "
                    f"{multiplier!r} — each unit maps to a positive number "
                    "of months (years: 12, months: 1)")
        joiners = spec.get("joiners")
        if joiners is not None and (
                not isinstance(joiners, list)
                or any(not isinstance(j, str) or not j.strip() for j in joiners)):
            raise ParserRegistryError(
                f"parser '{name}': \"joiners\" must be a list of words that "
                'may join compound terms, e.g. ["and", "und"]')
    else:
        percent_policy = spec.get("percent")
        if percent_policy is not None and percent_policy not in PERCENT_POLICIES:
            raise ParserRegistryError(
                f"parser '{name}' declares percent {percent_policy!r} — "
                "declare one of " + ", ".join(PERCENT_POLICIES)
                + " (what to do with '42%': read 42, a parse failure, or "
                "read 0.42)")


@dataclass(frozen=True)
class ResolvedNumericParser:
    """A ready-to-run declared parser plus the provenance the run stamps
    into report.json (cross-engine block keys: name, kind, registryFile,
    registryHash)."""

    name: str
    kind: str
    registry_hash: str
    parse: Callable[[str], float | None]

    def provenance(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "registryFile": REGISTRY_FILE,
                "registryHash": self.registry_hash}


def resolve(name: str | None, root: str | None = None) -> ResolvedNumericParser | None:
    """Resolve a manifest's ``numericParser`` to a runnable parser, or None
    when the study names none (the legacy path)."""
    name = (name or "").strip()
    if not name:
        return None
    spec = parser_spec(name, root)
    digest = registry_live_hash(root)
    if digest is None:  # unreachable after parser_spec, kept for safety
        raise ParserRegistryError(
            f"no parser registry exists at {REGISTRY_FILE}")
    return ResolvedNumericParser(
        name=name, kind=str(spec["kind"]), registry_hash=digest,
        parse=build_parser(name, spec))


def parser_pin_violations(d: dict, root: str | None = None) -> list[str]:
    """Verify-surface checks for the declared parser + its registry pin
    (Swift ``ParserRegistry.pinViolations`` twin). A study that names no
    parser and pins no hash gets an EMPTY list — legacy manifests gain no
    violations."""
    name = str(d.get("numericParser") or "").strip()
    pinned = d.get("parserRegistryHash")
    if not name:
        if pinned:
            return ["parserRegistryHash is pinned but no numericParser is "
                    "declared — an unused pin certifies nothing; declare the "
                    "parser or remove the pin"]
        return []
    live = registry_live_hash(root)
    if pinned:
        if live is None:
            return [f"pinned parser registry missing at {REGISTRY_FILE}"]
        if live != pinned:
            return [f"parser registry changed since pinning (have "
                    f"{live[:12]}…, pinned {str(pinned)[:12]}…)"]
    try:
        parser_spec(name, root)
    except ParserRegistryError as exc:
        return [f"numericParser '{name}': {exc}"]
    return []


# --- parser construction ------------------------------------------------------


def build_parser(name: str, spec: dict) -> Callable[[str], float | None]:
    """A callable text -> value-or-None (None = parse failure, counted —
    never coerced to 0 or a partial number)."""
    validate_spec(name, spec)
    if spec["kind"] == "durationMonths":
        return _build_duration_parser(spec)
    return _build_number_parser(spec)


def _number_pattern(decimal_comma: bool) -> str:
    return r"(\d+(?:[.,]\d+)?)" if decimal_comma else r"(\d+(?:\.\d+)?)"


def _number_or_word_pattern(decimal_comma: bool) -> str:
    """Digits-or-number-word, one capturing group — formal registers spell
    their numbers out ("ten years and six months' imprisonment"), and a
    digit-only grammar would drop those records non-randomly. durationMonths
    compound/single terms only; ranges and the plain ``number`` kind stay
    digit-only (`judicial._NUMBER_OR_WORD` twin)."""
    digits = r"\d+(?:[.,]\d+)?" if decimal_comma else r"\d+(?:\.\d+)?"
    return ("(" + digits + r"|\b(?:"
            + "|".join(judicial.NUMBER_WORDS) + r")\b)")


def _to_float(token: str, decimal_comma: bool) -> float:
    word = judicial.NUMBER_WORDS.get(token.lower())
    if word is not None:
        return word
    return float(token.replace(",", ".")) if decimal_comma else float(token)


def _alternation(tokens: list[str]) -> str:
    """One capturing alternation over unit tokens, longest first (so
    'monaten' wins over 'mo' regardless of declaration order)."""
    ordered = sorted(tokens, key=lambda token: (-len(token), token))
    return "(" + "|".join(re.escape(token) for token in ordered) + ")"


def _build_duration_parser(spec: dict) -> Callable[[str], float | None]:
    """The declared-unit-grammar generalization of ``judicial.parse_months``:
    compound (larger unit then smaller unit, optional comma/joiner), then
    range, then single — the shipped default table reproduces the built-in
    duration behavior fixture-for-fixture on both engines."""
    units = {str(token).lower(): float(multiplier)
             for token, multiplier in spec["units"].items()}
    minimum = min(units.values())
    minors = [token for token, m in units.items() if m == minimum]
    majors = [token for token, m in units.items() if m != minimum]
    joiners = [str(j).lower() for j in (spec.get("joiners") or [])]
    range_policy = spec.get("range") or "mean"
    decimal_comma = bool(spec.get("decimalComma", True))

    num = _number_pattern(decimal_comma)
    num_or_word = _number_or_word_pattern(decimal_comma)
    any_units = _alternation(list(units))
    joiner_part = ""
    if joiners:
        joiner_part = (r"(?:(?:"
                       + "|".join(re.escape(j) for j in joiners)
                       + r")\b\s*)?")
    compound = None
    if majors and minors:
        compound = re.compile(
            num_or_word + r"\s*" + _alternation(majors) + r"\b\s*,?\s*"
            + joiner_part
            + num_or_word + r"\s*" + _alternation(minors) + r"\b",
            re.IGNORECASE)
    ranged = re.compile(
        num + r"\s*(?:to|-|–|—)\s*" + num + r"\s*" + any_units + r"\b",
        re.IGNORECASE)
    single = re.compile(num_or_word + r"\s*" + any_units + r"\b",
                        re.IGNORECASE)

    def parse(text: str) -> float | None:
        if not text:
            return None
        if compound is not None:
            hit = compound.search(text)
            if hit:
                first = _to_float(hit.group(1), decimal_comma)
                second = _to_float(hit.group(3), decimal_comma)
                return (first * units[hit.group(2).lower()]
                        + second * units[hit.group(4).lower()])
        hit = ranged.search(text)
        if hit:
            if range_policy == "refuse":
                return None
            low = _to_float(hit.group(1), decimal_comma)
            high = _to_float(hit.group(2), decimal_comma)
            value = low if range_policy == "first" else (low + high) / 2.0
            return value * units[hit.group(3).lower()]
        hit = single.search(text)
        if hit:
            return (_to_float(hit.group(1), decimal_comma)
                    * units[hit.group(2).lower()])
        return None

    return parse


def _build_number_parser(spec: dict) -> Callable[[str], float | None]:
    """Plain numeric extraction: the FIRST number in the text, with declared
    range and percent policies (identical logic on both engines —
    fixture-locked)."""
    range_policy = spec.get("range") or "refuse"
    percent_policy = spec.get("percent") or "accept"
    decimal_comma = bool(spec.get("decimalComma", True))
    num = _number_pattern(decimal_comma)
    ranged = re.compile(num + r"\s*(?:to|-|–|—)\s*" + num, re.IGNORECASE)
    single = re.compile(num)

    def apply_percent(value: float, text: str, end: int) -> float | None:
        tail = text[end:].lstrip()
        if tail.startswith("%"):
            if percent_policy == "refuse":
                return None
            if percent_policy == "fraction":
                return value / 100.0
        return value

    def parse(text: str) -> float | None:
        if not text:
            return None
        first = single.search(text)
        if first is None:
            return None
        span = ranged.search(text)
        if span is not None and span.start() <= first.start():
            if range_policy == "refuse":
                return None
            low = _to_float(span.group(1), decimal_comma)
            high = _to_float(span.group(2), decimal_comma)
            value = low if range_policy == "first" else (low + high) / 2.0
            return apply_percent(value, text, span.end())
        return apply_percent(
            _to_float(first.group(1), decimal_comma), text, first.end())

    return parse
