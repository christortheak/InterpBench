"""Which outcome leads a results summary, and the study's declared one.

A study's summary leads with the outcome the study is about. The rule, in
order: the outcome the researcher declared (manifest key ``primaryOutcome``);
else a judged outcome; else a declared choice or numeric outcome; else a
reader or probe score; else a reasoning-style feature; else marker density;
else a surface measure such as word count. Every summary says which rule
chose the headline.

The mapping from outcome names to tiers is DATA, in
``client/resources/headline-outcomes.json``. This module reads it; the Mac
engine and the results explorer read copies generated from the same file
(``scripts/ci/check-headline-outcomes.py``). All three are held to one set of
cases, ``Tests/Fixtures/cross-engine/headline-outcome.json``.

Selection happens when a summary is SHOWN. Nothing here adds, removes, or
reorders the rows an analysis emits, and nothing here computes a statistic.
Pure standard library: no model, no tensor, no network.

Swift twin: ``Sources/ExperimentKit/HeadlineOutcome.swift``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

#: The manifest key a study declares its primary outcome under. Additive and
#: optional: a manifest without it loads, hashes, and verifies as before.
MANIFEST_KEY = "primaryOutcome"

#: The one outcome that comes from the evaluation report.
JUDGED = "judged"

#: The files a judged outcome is read from, in a run directory.
EVALUATION_REPORT_FILES = ("judge-report.json", "coding-report.json")

RULE_DECLARED = "declared"
RULE_DEFAULT_ORDER = "defaultOrder"

SOURCE_ANALYSIS = "analysisRows"
SOURCE_EVALUATION = "evaluationReport"

_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "client", "resources", "headline-outcomes.json")


@lru_cache(maxsize=1)
def mapping() -> dict:
    """The shipped mapping, parsed once."""
    with open(_DATA_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def _entry_index(name: str) -> int | None:
    """The first mapping entry that names ``name``, or None.

    A prefix or suffix entry needs something left over: ``rs_`` alone is not
    a reasoning-style feature, and ``MarkerDensity`` alone names no concept.
    """
    for index, entry in enumerate(mapping()["outcomes"]):
        pattern, kind = entry["name"], entry["match"]
        if kind == "exact" and name == pattern:
            return index
        if kind == "prefix" and name.startswith(pattern) \
                and len(name) > len(pattern):
            return index
        if kind == "suffix" and name.endswith(pattern) \
                and len(name) > len(pattern):
            return index
    return None


def tier_of(name: str) -> str:
    """The tier id of an outcome name; names the mapping does not list fall
    in the last tier, after the surface measures."""
    index = _entry_index(name)
    if index is None:
        return mapping()["unlistedTier"]
    return mapping()["outcomes"][index]["tier"]


def plain_phrase(name: str) -> str | None:
    """Plain words for an outcome name, or None when the mapping does not
    list it. ``{part}`` in a pattern entry is the rest of the name."""
    index = _entry_index(name)
    if index is None:
        return None
    entry = mapping()["outcomes"][index]
    if entry["match"] == "prefix":
        part = name[len(entry["name"]):]
    elif entry["match"] == "suffix":
        part = name[:-len(entry["name"])]
    else:
        part = ""
    return entry["plain"].replace("{part}", part)


def _order_key(name: str) -> tuple:
    """Default-order sort key: tier, then entry order, then the name itself
    (by code point), so three codebases agree on every tie."""
    tiers = [tier["id"] for tier in mapping()["tiers"]]
    index = _entry_index(name)
    tier = tier_of(name)
    return (tiers.index(tier),
            index if index is not None else len(mapping()["outcomes"]),
            [ord(character) for character in name])


@dataclass(frozen=True)
class Headline:
    """What leads a summary, and why.

    ``outcome`` is None when the run has nothing to lead with. ``rule`` is
    ``declared`` or ``defaultOrder`` (None with no outcome). ``declared_absent``
    is True when the study declared a primary outcome this run does not have,
    so the summary fell back; ``chosen_by`` always says so in words.
    """
    outcome: str | None
    tier: str | None
    rule: str | None
    source: str | None
    declared_outcome: str | None
    declared_absent: bool
    chosen_by: str

    def as_payload(self) -> dict:
        """The block an envelope carries. Keys match the Swift twin's."""
        payload = {
            "outcome": self.outcome,
            "tier": self.tier,
            "rule": self.rule,
            "source": self.source,
            "chosenBy": self.chosen_by,
            "declaredAbsent": self.declared_absent,
        }
        if self.declared_outcome is not None:
            payload["declaredOutcome"] = self.declared_outcome
        if self.outcome is not None:
            phrase = plain_phrase(self.outcome)
            if phrase is not None:
                payload["plain"] = phrase
        return payload


def _clean(declared) -> str | None:
    """A declared outcome as stored: a non-empty string, or nothing. A value
    of any other type is read as no declaration rather than as a crash."""
    if not isinstance(declared, str):
        return None
    trimmed = declared.strip()
    return trimmed or None


def select(declared, analysis_outcomes, evaluation_outcomes=()) -> Headline:
    """Choose the headline from what a run HAS.

    ``analysis_outcomes`` are the outcome names of the pooled effect rows;
    ``evaluation_outcomes`` are the names an evaluation report supplies
    (today only ``judged``). A judged outcome is never an analysis row, which
    is why the two arrive separately.
    """
    declared = _clean(declared)
    analysis = [name for name in dict.fromkeys(analysis_outcomes) if name]
    evaluation = [name for name in dict.fromkeys(evaluation_outcomes) if name]
    rules = mapping()["rules"]

    def source_of(name: str) -> str:
        return SOURCE_EVALUATION if name in evaluation else SOURCE_ANALYSIS

    available = evaluation + [name for name in analysis
                              if name not in evaluation]
    if declared is not None and declared in available:
        return Headline(
            outcome=declared, tier=tier_of(declared), rule=RULE_DECLARED,
            source=source_of(declared), declared_outcome=declared,
            declared_absent=False, chosen_by=rules[RULE_DECLARED])
    absent = (f"; the declared primary outcome '{declared}' is not in this run"
              if declared is not None else "")
    if not available:
        return Headline(
            outcome=None, tier=None, rule=None, source=None,
            declared_outcome=declared, declared_absent=declared is not None,
            chosen_by="no outcome to lead with" + absent)
    first = min(available, key=_order_key)
    return Headline(
        outcome=first, tier=tier_of(first), rule=RULE_DEFAULT_ORDER,
        source=source_of(first), declared_outcome=declared,
        declared_absent=declared is not None,
        chosen_by=rules[RULE_DEFAULT_ORDER] + absent)


# --- what a study can produce --------------------------------------------------


def _declares_judging(d: dict) -> bool:
    """The same resolution as ``Manifest.effective_evaluation``: an explicit
    evaluation block decides; without one, a pinned judge plus a pinned rubric
    file is a judging declaration."""
    evaluation = d.get("evaluation")
    if isinstance(evaluation, dict):
        return evaluation.get("kind") == "pairedJudge"
    return bool(d.get("judges")) and bool(
        str(d.get("judgeRubricFile") or "").strip())


def _implicit_numeric_endpoint(d: dict) -> bool:
    """The deprecated implicit numeric endpoint, by the manifest module's own
    rule (kept in one place there)."""
    from . import manifest as manifest_module
    if d.get("caseFamily") != manifest_module.IMPLICIT_ENDPOINT_CASE_FAMILY:
        return False
    if d.get("studyKind") == "multiAgent":
        return True
    return not str(d.get("numericParser") or "").strip()


def _requirement_met(requirement: str, d: dict) -> bool:
    instruments = [str(i) for i in (d.get("outcomeInstruments") or [])]
    declared_parser = bool(str(d.get("numericParser") or "").strip())
    if requirement == "always":
        return True
    if requirement == "never":
        return False
    if requirement == "judging":
        return _declares_judging(d)
    if requirement.startswith("instrument:"):
        return requirement.split(":", 1)[1] in instruments
    if requirement == "declaredNumericParser":
        return declared_parser
    if requirement == "numericEndpoint":
        return declared_parser or _implicit_numeric_endpoint(d)
    if requirement == "readers":
        return "repeReaderScore" in instruments and bool(d.get("readerRefs"))
    if requirement == "reasoningStyleTaxonomy":
        return bool(str(d.get("reasoningStyleTaxonomyPath") or "").strip())
    if requirement == "concepts":
        return bool(d.get("concepts"))
    return False


def producible(d: dict) -> dict:
    """The outcomes this study's settings can produce, in default order.

    ``names`` are complete outcome names. ``patterns`` are families whose
    members cannot be listed from the settings alone (reasoning-style
    features live in the pinned taxonomy file): any name of that shape is
    accepted. Read from the manifest's own declarations; no file is opened.
    """
    names: list[str] = []
    patterns: list[str] = []
    for entry in mapping()["outcomes"]:
        if not _requirement_met(entry["requires"], d):
            continue
        if entry["match"] == "exact":
            names.append(entry["name"])
        elif entry["requires"] == "readers":
            for ref in d.get("readerRefs") or []:
                concept = ref.get("concept") if isinstance(ref, dict) else None
                if concept and entry["name"] + concept not in names:
                    names.append(entry["name"] + concept)
        elif entry["requires"] == "concepts":
            for concept in d.get("concepts") or []:
                concept_name = (concept.get("name")
                                if isinstance(concept, dict) else None)
                if concept_name and concept_name + entry["name"] not in names:
                    names.append(concept_name + entry["name"])
        elif entry["match"] == "prefix":
            patterns.append(entry["name"] + "<name>")
        else:
            patterns.append("<name>" + entry["name"])
    return {"names": names, "patterns": patterns}


def can_produce(d: dict, outcome: str) -> bool:
    """Whether ``outcome`` is one of the study's producible outcomes."""
    listing = producible(d)
    if outcome in listing["names"]:
        return True
    index = _entry_index(outcome)
    if index is None:
        return False
    entry = mapping()["outcomes"][index]
    pattern = (entry["name"] + "<name>" if entry["match"] == "prefix"
               else "<name>" + entry["name"])
    return entry["match"] != "exact" and pattern in listing["patterns"]


def cannot_produce_reason(outcome: str) -> str:
    """Why a study's settings cannot produce ``outcome``, in plain words:
    what the outcome needs, or that nothing reports it. For the refusal a
    declaration verb gives; call it only for an outcome ``can_produce``
    declined."""
    index = _entry_index(outcome)
    if index is None:
        return "it is not an outcome SteerLab reports"
    entry = mapping()["outcomes"][index]
    needs = mapping()["requirements"].get(entry["requires"], "")
    if not needs:
        return "no engine reports it as a paired effect"
    return "it needs " + needs


def producible_choices(d: dict) -> str:
    """The producible outcomes as one retypeable list: ``a | b | rs_<name>``."""
    listing = producible(d)
    return " | ".join(listing["names"] + listing["patterns"])


def settings_summary_line(d: dict) -> str:
    """The primary-outcome line of the generated settings summary. Swift
    twin: ``HeadlineOutcome.settingsSummaryLine`` (same words)."""
    declared = _clean(d.get(MANIFEST_KEY))
    if declared is None:
        return ("- **Primary outcome:** not declared; summaries lead with "
                "the default order")
    phrase = plain_phrase(declared)
    return (f"- **Primary outcome:** {declared}"
            + (f" ({phrase})" if phrase else "")
            + ", declared by the researcher")


# --- what a run has -------------------------------------------------------------


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _source_run_name(run_directory: str) -> str:
    """The directory name of the run a derived run (analyze, evaluate) read,
    or this run's own name when it is a source run itself."""
    try:
        with open(os.path.join(run_directory, "source-run.txt"),
                  encoding="utf-8") as handle:
            stamped = handle.read().strip()
    except OSError:
        stamped = ""
    if not stamped:
        analysis = _read_json(os.path.join(run_directory, "analysis.json"))
        stamped = str((analysis or {}).get("sourceRun") or "")
    name = os.path.basename(stamped.rstrip("/")) if stamped else ""
    return name or os.path.basename(os.path.abspath(run_directory))


def _report_source(report: dict) -> str:
    """The source run an evaluation report names: the server writes
    ``sourceRun`` (a directory name), the Mac ``sourceRunDirectory`` (a
    path). Either way, the directory name."""
    raw = str(report.get("sourceRun") or report.get("sourceRunDirectory")
              or "")
    return os.path.basename(raw.rstrip("/"))


def evaluation_report_for(run_directory: str) -> str | None:
    """The evaluation report that judged the same source run as
    ``run_directory``, or None.

    Looked for in the run directory itself first, then in the sibling
    directories whose name marks them as an evaluation and whose report names
    the same source run; the latest by directory name wins. Read-only.
    """
    for name in EVALUATION_REPORT_FILES:
        own = os.path.join(run_directory, name)
        if os.path.isfile(own):
            return own
    source = _source_run_name(run_directory)
    parent = os.path.dirname(os.path.abspath(run_directory))
    try:
        siblings = sorted(os.listdir(parent), reverse=True)
    except OSError:
        return None
    for sibling in siblings:
        if "-evaluate" not in sibling:
            continue
        for name in EVALUATION_REPORT_FILES:
            path = os.path.join(parent, sibling, name)
            report = _read_json(path) if os.path.isfile(path) else None
            if report is not None and _report_source(report) == source:
                return path
    return None


def for_analysis_run(run_directory: str, analysis_outcomes) -> Headline:
    """The headline of an analysis run: its pooled outcome names, the
    evaluation report for the same source run when there is one, and the
    primary outcome declared in the run's own manifest snapshot."""
    snapshot = _read_json(os.path.join(run_directory, "experiment.json")) or {}
    evaluation = ([JUDGED]
                  if evaluation_report_for(run_directory) is not None else [])
    return select(snapshot.get(MANIFEST_KEY), analysis_outcomes, evaluation)
