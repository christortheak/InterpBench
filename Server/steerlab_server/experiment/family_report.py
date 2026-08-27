"""Cross-family descriptive report — Study B's deliverable
(``docs/SAE-VECTOR-INTERVENTION-PROPOSAL-2026-08-13-r2.md`` §7, §8 P1-7).

One engine-produced artifact that lays the paper's vector families — CAA /
grand-mean, LoRA, OptVec, Gemma Scope SAE — **side by side**: what each
direction is, where it lives, how long it is in units of the residual stream,
how it sits relative to the others at layers they SHARE, and what its
behavioural signature on the common arm was.

**This report is descriptive. It contains no inferential statistics.** No
confidence interval, no p-value, no correction, no significance vocabulary is
computed or copied here, and no cross-family contrast is tested. The paper
reads LoRA vs CAA vs OptVec vs SAE the way this artifact presents them: as
overall behavioural impressions stated qualitatively, beside geometry that is
reported as geometry. The engine already owns the inferential layer
(``analyze``), and its artifacts keep those columns; a reader who wants them
opens the analyze run this report cites. Mixing the two here would invite
exactly the claim §7 demotes — that the sources are statistically comparable.

Three rules carry the validity of the geometry half:

* **Matched layers only.** Two artifacts compare at layer *L* only when BOTH
  carry a nonzero row there. A feature exists only in its own layer's
  dictionary, and directions at different layers live in different bases —
  a cross-layer pair is reported ``notApplicable`` with its reason, never as
  a cosine of zero. Zero would read as "orthogonal", which is a geometric
  claim about two things that were never in the same space.
* **Different models are different bases.** Rows from different ``modelID``s,
  or of different hidden size, are ``notApplicable`` for the same reason.
* **Nothing is recomputed from behaviour.** Behavioural rows are COPIED
  verbatim out of the engine's ``analyze`` artifacts (``effect-sizes.csv``),
  with their source path and file hash stamped. This module never re-derives
  a delta, never re-pairs records, and never re-reads generations.

Inference-free in the other sense too: it loads artifacts off disk and no
model.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import paths
from .optvec_eval import LoadedArtifact, OptVecArtifactError, load_artifact, sha256_file
from .optvec_geometry import cosine_matrix
from .run_config import write_run_config
from ..build_identity import engine_version
from ..steering.vector_store import SUBSTRATE

RUN_TYPE = "family-report"
REPORT_JSON = "family-report.json"
COSINE_CSV = "family-cosines.csv"
SUMMARY_TXT = "family-report.txt"
SCHEMA_VERSION = 1

#: Reproduced at the top of every surface this module writes, because the
#: artifact will outlive the conversation that scoped it.
DESCRIPTIVE_NOTICE = (
    "DESCRIPTIVE REPORT. Vector families are laid side by side — identities, "
    "layers, norms, matched-layer cosines, and behavioural deltas copied "
    "verbatim from engine analyze artifacts. It reports no interval "
    "estimates, no test statistics, and no statistical comparison of one "
    "family against another; differences between families are to be read as "
    "overall impressions, not as measured contrasts. Inferential quantities "
    "live in the analyze artifacts this report cites by path and hash.")

#: ``extractionMethod`` (the sidecar's own vocabulary) → the family label the
#: paper groups by. Deliberately a lookup and not a heuristic: an unknown
#: method becomes its own labelled row (see :func:`family_for`) rather than
#: being folded into whichever family looks closest.
#:
#: ``lat`` and ``repeReaderLAT`` are TWO FAMILIES, not two spellings of one
#: (naming honesty ruling 2026-08-27). They used to share the label
#: "RepE/LAT", which put a direction that borrows the paper's PCA arithmetic
#: and a direction produced by the paper's actual pipeline — task template,
#: template-mediated LAT token, persisted fit parameters, held-out sign and
#: layer selection — in one row of the paper's own side-by-side. A reader of
#: that report could not tell which rows were RepE and which were RepE-shaped.
FAMILY_BY_METHOD = {
    "meanDifference": "CAA",
    "lat": "paired-difference-PCA",
    "repeReaderLAT": "RepE-reader-LAT",
    "emotionGrandMean": "grand-mean",
    "designatedReference": "designated-reference",
    "pinnedArtifact": "pinned-artifact",
    "optvec": "OptVec",
    "gemmaScopeSAE": "SAE",
}

#: The label a row carries when its sidecar stamps no method at all.
UNSTAMPED_FAMILY = "unstamped"

#: Columns copied verbatim out of an ``analyze`` run's ``effect-sizes.csv``.
#: Descriptive quantities and their labels only: what moved, on what endpoint,
#: over how many pairs.
BEHAVIOR_COLUMNS = ("condition", "endpoint", "n", "deltaMean", "modality",
                    "unit", "stratifyBy", "stratum")

#: Columns deliberately NOT copied, by kind. Their NAMES are stamped into the
#: artifact (their values never are) so the omission reads as a decision
#: rather than an oversight: the quantities exist, in the analyze run this
#: report cites by path and hash.
OMITTED_BEHAVIOR_COLUMNS = {
    "ciLower": "interval estimate",
    "ciUpper": "interval estimate",
    "wilcoxonW": "test statistic",
    "wilcoxonP": "test statistic",
    "adjustedP": "test statistic",
    "correction": "multiplicity-correction label",
    "estimand": "inferential-target label",
    "inference": "inferential-procedure label",
}

#: Why every column above is left behind. One sentence, stamped beside them.
OMISSION_REASON = (
    "inferential columns are not copied into a descriptive report; they "
    "remain in the cited analyze artifact")

#: An analyze run's per-condition effect table.
EFFECT_SIZES_CSV = "effect-sizes.csv"

#: ``stratifyBy`` values that mark a POOLED row (the whole condition), as
#: against a per-item / per-stratum breakdown row. An analyze table carries
#: both, keyed only by this column: a report that copied every row would show
#: one condition twenty-five times.
POOLED_STRATA = ("", "pooled")

#: How much of a condition's table is copied. ``pooled`` (the default) takes
#: the condition-level rows; ``all`` takes the per-stratum breakdown too. The
#: choice is stamped beside the rows, with how many were available, so a
#: reader can see that rows were filtered and by what rule.
STRATA_MODES = ("pooled", "all")


class FamilyReportError(ValueError):
    """A family report that cannot be produced as asked."""


# ------------------------------------------------------------------- config


_BEHAVIOR_KEYS = {"analyze": "analyze", "condition": "condition",
                  "endpoints": "endpoints", "strata": "strata"}


@dataclass
class BehaviorRef:
    """Where one row's behavioural signature is read from.

    ``analyze`` is an ``analyze`` RUN DIRECTORY (workspace-relative), not a
    study name: the report cites the exact artifact it copied, and a study
    name would leave "which run?" to whoever reads it later.
    """

    analyze: str
    condition: str
    #: Optional endpoint filter. Absent = every endpoint the analyze artifact
    #: reports for that condition.
    endpoints: list[str] | None = None
    #: ``pooled`` (default) or ``all`` — see :data:`STRATA_MODES`.
    strata: str = "pooled"

    def __post_init__(self) -> None:
        self.strata = str(self.strata or "pooled").strip()
        if self.strata not in STRATA_MODES:
            raise FamilyReportError(
                f"behavior.strata must be one of {', '.join(STRATA_MODES)} "
                f"(got {self.strata!r})")
        self.analyze = str(self.analyze or "").strip()
        self.condition = str(self.condition or "").strip()
        if not self.analyze:
            raise FamilyReportError(
                "behavior.analyze must name an analyze run directory")
        if not self.condition:
            raise FamilyReportError(
                "behavior.condition must name the condition whose rows are "
                "copied — a behavioural join with no condition would copy the "
                "whole table under this row's label")
        if self.endpoints is not None:
            if (not isinstance(self.endpoints, list)
                    or not all(isinstance(e, str) and e.strip()
                               for e in self.endpoints)):
                raise FamilyReportError(
                    "behavior.endpoints must be a list of endpoint names")
            self.endpoints = [e.strip() for e in self.endpoints]

    @classmethod
    def from_dict(cls, payload: dict) -> "BehaviorRef":
        if not isinstance(payload, dict):
            raise FamilyReportError("'behavior' must be a JSON object")
        unknown = sorted(set(payload) - set(_BEHAVIOR_KEYS))
        if unknown:
            raise FamilyReportError(
                "unknown behavior key(s): " + ", ".join(unknown))
        return cls(**{field_name: payload[key]
                      for key, field_name in _BEHAVIOR_KEYS.items()
                      if key in payload})

    def to_dict(self) -> dict:
        payload = {"analyze": self.analyze, "condition": self.condition,
                   "strata": self.strata}
        if self.endpoints is not None:
            payload["endpoints"] = list(self.endpoints)
        return payload


_ENTRY_KEYS = {"reference": "reference", "family": "family", "label": "label",
               "note": "note", "behavior": "behavior"}


@dataclass
class FamilyEntryConfig:
    """One row of the report.

    Either a vector artifact (``reference``), or a BEHAVIOUR-ONLY row — a
    family member with no residual-stream direction to compare. LoRA is the
    live example: a fine-tuned arm is a model variant with adapter weights,
    not a vector, so it has a behavioural signature and no geometry. Such a
    row must declare its own ``family`` and ``label``, since there is no
    sidecar to read them from.
    """

    reference: str | None = None
    family: str | None = None
    label: str | None = None
    note: str | None = None
    behavior: BehaviorRef | None = None

    def __post_init__(self) -> None:
        self.reference = (str(self.reference).strip()
                          if self.reference is not None else None) or None
        for name in ("family", "label", "note"):
            value = getattr(self, name)
            setattr(self, name,
                    (str(value).strip() if value is not None else None) or None)
        if isinstance(self.behavior, dict):
            self.behavior = BehaviorRef.from_dict(self.behavior)
        if self.reference is None:
            if self.behavior is None:
                raise FamilyReportError(
                    "an entry with neither 'reference' nor 'behavior' names "
                    "nothing — give it a vector artifact, or a behavioural "
                    "source for a family member that has no direction")
            if not self.family or not self.label:
                raise FamilyReportError(
                    "a behaviour-only entry must declare both 'family' and "
                    "'label': there is no sidecar to read its identity from, "
                    "and an unlabelled row cannot be attributed to a family")

    @classmethod
    def from_dict(cls, payload: dict) -> "FamilyEntryConfig":
        if isinstance(payload, str):
            # The bare-reference shorthand, for the common all-defaults row.
            return cls(reference=payload)
        if not isinstance(payload, dict):
            raise FamilyReportError(
                "each artifact entry must be an object (or a bare artifact "
                f"reference string), got {payload!r}")
        unknown = sorted(set(payload) - set(_ENTRY_KEYS))
        if unknown:
            raise FamilyReportError(
                "unknown artifact entry key(s): " + ", ".join(unknown))
        return cls(**{field_name: payload[key]
                      for key, field_name in _ENTRY_KEYS.items()
                      if key in payload})

    def to_dict(self) -> dict:
        payload: dict = {}
        for name in ("reference", "family", "label", "note"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.behavior is not None:
            payload["behavior"] = self.behavior.to_dict()
        return payload


_CONFIG_KEYS = {"name": "name", "artifacts": "artifacts",
                "discoverPromotions": "discover_promotions"}


@dataclass
class FamilyReportConfig:
    artifacts: list[FamilyEntryConfig] = field(default_factory=list)
    name: str | None = None
    #: Scan the workspace's mutable variant library for promoted agents that
    #: inject each artifact, so a row can say whether its direction was ever
    #: seated and at what dose. Discovery only — an absent variant library is
    #: an absent fact, never a "not promoted" verdict.
    discover_promotions: bool = True

    def __post_init__(self) -> None:
        self.artifacts = [
            entry if isinstance(entry, FamilyEntryConfig)
            else FamilyEntryConfig.from_dict(entry)
            for entry in self.artifacts]
        if len(self.artifacts) < 2:
            raise FamilyReportError(
                "a cross-family report needs at least 2 entries — one row is "
                "not a side-by-side")
        seen: set[str] = set()
        for entry in self.artifacts:
            if entry.reference is None:
                continue
            if entry.reference in seen:
                raise FamilyReportError(
                    f"artifact {entry.reference!r} appears twice — the same "
                    "direction listed twice would appear as its own perfect "
                    "cross-family match")
            seen.add(entry.reference)
        self.name = (str(self.name).strip() if self.name is not None
                     else None) or None
        if self.name is not None and ("/" in self.name or os.sep in self.name
                                      or self.name in (".", "..")):
            # The name becomes a run-directory slug; a path there would write
            # the report outside runs/.
            raise FamilyReportError(
                f"report name {self.name!r} must be a plain name component")
        self.discover_promotions = bool(self.discover_promotions)

    @classmethod
    def from_dict(cls, payload: dict) -> "FamilyReportConfig":
        if not isinstance(payload, dict):
            raise FamilyReportError(
                "the family-report config must be a JSON object")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS))
        if unknown:
            raise FamilyReportError(
                "unknown family-report config key(s): " + ", ".join(unknown))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise FamilyReportError(
                "'artifacts' must be a list of entries (an artifact reference "
                "string, or an object with 'reference'/'family'/'label'/"
                "'behavior')")
        return cls(**{field_name: payload[key]
                      for key, field_name in _CONFIG_KEYS.items()
                      if key in payload})

    def to_dict(self) -> dict:
        return {"name": self.name,
                "discoverPromotions": self.discover_promotions,
                "artifacts": [entry.to_dict() for entry in self.artifacts]}


def load_config(path: str) -> FamilyReportConfig:
    with open(path, encoding="utf-8") as handle:
        return FamilyReportConfig.from_dict(json.load(handle))


# ------------------------------------------------------------------ identity


def family_for(extraction_method: str | None,
               declared: str | None = None) -> tuple[str, str]:
    """``(family label, where the label came from)``.

    A declared override wins and says so. A recognised ``extractionMethod``
    maps through :data:`FAMILY_BY_METHOD`. An unrecognised method keeps its
    own name — a new recipe must appear as a new family rather than being
    absorbed into an existing one — and an absent method is
    :data:`UNSTAMPED_FAMILY`, not a guess.
    """
    if declared:
        return declared, "declared"
    method = (extraction_method or "").strip()
    if not method:
        return UNSTAMPED_FAMILY, "absent-extraction-method"
    known = FAMILY_BY_METHOD.get(method)
    if known:
        return known, "extractionMethod"
    return method, "unrecognized-extraction-method"


def layers_with_direction(artifact: LoadedArtifact) -> list[int]:
    """Layers where this artifact carries a nonzero row.

    A zero row is "this artifact says nothing here" (an SAE feature or an
    OptVec solution is full-depth with zeros everywhere but its own layer),
    which is why it is excluded rather than compared.
    """
    return [layer for layer in range(artifact.vectors.layer_count)
            if artifact.vectors.norm(layer) > 0.0]


def _residual_unit_norm(norm: float, residual: float | None) -> float | None:
    """The direction's length in units of the residual-stream norm at that
    layer, or ``None`` when no calibration exists. Absent is absent — a row
    with no measured denominator gets no number, never a 0 or a 1."""
    if residual is None or residual <= 0:
        return None
    return float(norm / residual)


def _per_layer_rows(artifact: LoadedArtifact, layers: list[int]) -> list[dict]:
    residuals = artifact.sidecar.residualNormPerLayer or []
    rows = []
    for layer in layers:
        norm = artifact.vectors.norm(layer)
        residual = (float(residuals[layer])
                    if layer < len(residuals) else None)
        row = {"layer": layer, "norm": norm}
        if residual is not None:
            row["residualNorm"] = residual
        units = _residual_unit_norm(norm, residual)
        if units is not None:
            row["normInResidualUnits"] = units
        rows.append(row)
    return rows


def _conventions(artifact: LoadedArtifact) -> dict:
    """The stamps that say what transform the stored bytes have already been
    through. Only stamps that are present appear — a missing convention is a
    fact about the artifact, and the advisories name it."""
    sidecar = artifact.sidecar
    fields = {
        "gemmascopeConvention": sidecar.gemmascopeConvention,
        "rawDecoderNorm": sidecar.rawDecoderNorm,
        "gemmascopeTargetNorm": sidecar.gemmascopeTargetNorm,
        "residualNormSource": sidecar.residualNormSource,
        "neutralProjection": sidecar.neutralProjection,
        "confoundProjection": sidecar.confoundProjection,
        "readingPosition": sidecar.readingPosition,
        "recipeMethod": sidecar.recipeMethod,
        "recipeIdentityHash": sidecar.recipeIdentityHash,
        "neutralMeanSource": sidecar.neutralMeanSource,
        "substrate": sidecar.substrate,
    }
    return {key: value for key, value in fields.items() if value is not None}


def _raw_block(artifact: LoadedArtifact, key: str) -> dict | None:
    block = artifact.raw_sidecar.get(key)
    return block if isinstance(block, dict) else None


def _reference_tail(reference: str) -> str | None:
    """``runs/<run>/<name>`` out of an absolute or relative artifact id.

    Promoted variants written on another machine carry ABSOLUTE artifact
    paths (observed live: six app-promoted agents carried Mac paths), so a
    string comparison against a workspace-relative reference misses every
    one of them. The tail is what both spellings share.
    """
    parts = [p for p in os.path.normpath(reference or "").split(os.sep) if p]
    if "runs" in parts:
        index = len(parts) - 1 - parts[::-1].index("runs")
        return "/".join(parts[index:])
    return None


def _same_artifact(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if os.path.normpath(a) == os.path.normpath(b):
        return True
    tail_a, tail_b = _reference_tail(a), _reference_tail(b)
    return tail_a is not None and tail_a == tail_b


def discover_promotions(reference: str, root: str | None = None) -> list[dict]:
    """Variants in the workspace's variant library that inject ``reference``.

    Read-only discovery over ``runs/model-variants``. Each hit reports the
    variant's name, its selected dose for THIS artifact (layer, α, and
    whether α is in norm units), its LoRA adapter count, and its promotion
    birth certificate's identifying fields when it has one. A hand-created
    variant has no ``promotion`` block and is reported as such — the report
    states what it found, and "no promotion found" is never spelled as "not
    qualified".
    """
    library = os.path.join(paths.runs_directory(root), "model-variants")
    if not os.path.isdir(library):
        return []
    found: list[dict] = []
    for entry in sorted(os.listdir(library)):
        directory = os.path.join(library, entry)
        if not os.path.isdir(directory):
            continue
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for filename in names:
            if not filename.endswith(".json") or filename == "config.json":
                continue
            path = os.path.join(directory, filename)
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict) or "injections" not in payload:
                continue
            injections = payload.get("injections")
            if not isinstance(injections, list):
                continue
            cells = [cell for cell in injections
                     if isinstance(cell, dict)
                     and _same_artifact(str(cell.get("vectorArtifactID") or ""),
                                        reference)]
            if not cells:
                continue
            promotion = payload.get("promotion")
            record = {
                "variant": payload.get("name"),
                "variantPath": path,
                "adapterCount": len(payload.get("adapters") or []),
                "alphaInNormUnits": bool(payload.get("alphaInNormUnits", False)),
                "cells": [{"layer": cell.get("layer"),
                           "alpha": cell.get("alpha"),
                           "concept": cell.get("concept")}
                          for cell in cells],
                "promoted": isinstance(promotion, dict),
            }
            if isinstance(promotion, dict):
                record["promotion"] = {
                    key: promotion.get(key)
                    for key in ("promotedBy", "promotedAt", "experiment",
                                "experimentHash", "sweepRun", "winningCell",
                                "promotionKey", "overrideReason",
                                "selectionOutcome", "substrate")
                    if promotion.get(key) is not None}
            found.append(record)
    return found


#: The qualification artifact's filename (proposal r2 §8 P0-4), looked for
#: first ARTIFACT-scoped (``<name>-…``) and then directory-scoped. Discovery
#: only: a report never invents the file, and its ABSENCE is reported as
#: absence — never as "not qualified", which is a verdict this module has no
#: standing to reach.
QUALIFICATION_FILENAME = "sae-feature-qualification.json"


def discover_qualification(artifact: LoadedArtifact) -> dict | None:
    """A qualification artifact for this vector, or a ``qualification`` block
    on its sidecar. Returns a POINTER (path + hash, or the block verbatim)
    plus the SCOPE it was found at, because a directory-scoped file belongs
    to whatever the import wrote into that run directory — a reader must be
    able to see which of the two associations they are trusting."""
    block = _raw_block(artifact, "qualification")
    if block is not None:
        return {"source": "sidecar", "scope": "artifact",
                "qualification": block}
    directory = os.path.dirname(artifact.path) or "."
    for scope, filename in (
            ("artifact", f"{artifact.name}-{QUALIFICATION_FILENAME}"),
            ("runDirectory", QUALIFICATION_FILENAME)):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return {"source": "runDirectory", "scope": scope,
                    "path": candidate, "sha256": sha256_file(candidate)}
    return None


# ---------------------------------------------------------------- behaviour


def _read_effect_sizes(directory: str) -> tuple[list[dict], list[str], str]:
    """``(rows, header, sha256)`` of an analyze run's ``effect-sizes.csv``."""
    path = os.path.join(directory, EFFECT_SIZES_CSV)
    if not os.path.isfile(path):
        raise FamilyReportError(
            f"'{directory}' carries no {EFFECT_SIZES_CSV} — behavioural rows "
            "are copied from an ANALYZE run directory, and this is not one "
            "(or the analyze stage has not been run for it)")
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, header, sha256_file(path)


def _first_line(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.readline().strip() or None


def load_behavior(behavior: BehaviorRef, root: str | None = None) -> dict:
    """Copy one condition's descriptive rows out of an analyze artifact.

    Verbatim: the values are the strings the engine wrote, parsed to numbers
    only where the column is numeric, and never re-derived. A condition that
    the artifact does not carry REFUSES (naming what it does carry) — a
    silently empty behavioural join looks exactly like a family with no
    effect, which is the one misreading this report must not enable.
    """
    directory = paths.resolve(behavior.analyze, root)
    if not os.path.isdir(directory):
        raise FamilyReportError(
            f"behavioural source '{behavior.analyze}' is not a directory")
    rows, header, digest = _read_effect_sizes(directory)

    conditions = sorted({row.get("condition", "") for row in rows})
    if behavior.condition not in conditions:
        raise FamilyReportError(
            f"analyze run '{behavior.analyze}' has no condition "
            f"{behavior.condition!r} — it reports: "
            + (", ".join(c for c in conditions if c) or "(none)"))

    selected = [row for row in rows if row.get("condition") == behavior.condition]
    if behavior.endpoints is not None:
        available = sorted({row.get("endpoint", "") for row in selected})
        missing = [e for e in behavior.endpoints if e not in available]
        if missing:
            raise FamilyReportError(
                f"analyze run '{behavior.analyze}' condition "
                f"{behavior.condition!r} has no endpoint(s) "
                + ", ".join(repr(m) for m in missing)
                + " — it reports: " + (", ".join(a for a in available if a)
                                       or "(none)"))
        selected = [row for row in selected
                    if row.get("endpoint") in behavior.endpoints]

    available = len(selected)
    if behavior.strata == "pooled":
        pooled = [row for row in selected
                  if (row.get("stratifyBy") or "").strip() in POOLED_STRATA]
        if not pooled:
            raise FamilyReportError(
                f"analyze run '{behavior.analyze}' condition "
                f"{behavior.condition!r} carries no pooled row (every row is "
                "a per-stratum breakdown) — declare strata: \"all\" to copy "
                "the breakdown rows deliberately")
        selected = pooled

    copied = [_copy_behavior_row(row) for row in selected]
    omitted = {column: reason
               for column, reason in OMITTED_BEHAVIOR_COLUMNS.items()
               if column in header}
    unread = [column for column in header
              if column not in BEHAVIOR_COLUMNS
              and column not in OMITTED_BEHAVIOR_COLUMNS]
    source = {
        "analyze": behavior.analyze,
        "path": directory,
        "effectSizesSHA256": digest,
        "sourceRun": _first_line(os.path.join(directory, "source-run.txt")),
        "experimentHash": _first_line(
            os.path.join(directory, "experiment-hash.txt")),
    }
    return {"condition": behavior.condition,
            "endpoints": (list(behavior.endpoints)
                          if behavior.endpoints is not None else None),
            "strata": behavior.strata,
            "rowsAvailable": available,
            "rowsCopied": len(copied),
            "rows": copied,
            "source": {k: v for k, v in source.items() if v is not None},
            "copiedColumns": list(BEHAVIOR_COLUMNS),
            "omittedColumns": omitted,
            "omissionReason": OMISSION_REASON,
            "unreadColumns": unread,
            "provenance": "copied verbatim from the engine's analyze "
                          "artifact; not recomputed"}


def _copy_behavior_row(row: dict) -> dict:
    copied: dict = {}
    for column in BEHAVIOR_COLUMNS:
        if column not in row:
            continue
        raw = row.get(column)
        if raw is None or raw == "":
            continue
        if column in ("n",):
            try:
                copied[column] = int(raw)
                continue
            except (TypeError, ValueError):
                pass
        if column in ("deltaMean",):
            try:
                copied[column] = float(raw)
                continue
            except (TypeError, ValueError):
                pass
        copied[column] = raw
    return copied


# ----------------------------------------------------------------- geometry


def _pair_status(a: dict, b: dict) -> tuple[str, str | None, list[int]]:
    """``(status, reason, shared layers)`` for one pair of vector rows."""
    model_a, model_b = a["modelID"], b["modelID"]
    if model_a and model_b and model_a != model_b:
        return ("notApplicable",
                f"different models ({model_a} vs {model_b}) — their residual "
                "streams are different bases, so a cosine between them is not "
                "a geometric statement", [])
    if a["hiddenSize"] != b["hiddenSize"]:
        return ("notApplicable",
                f"different hidden sizes ({a['hiddenSize']} vs "
                f"{b['hiddenSize']}) — different model bases", [])
    shared = sorted(set(a["layers"]) & set(b["layers"]))
    if not shared:
        return ("notApplicable",
                f"no shared layer with a nonzero direction (layers "
                f"{a['layers'] or '[]'} vs {b['layers'] or '[]'}) — a "
                "direction exists only in its own layer's basis, and a "
                "cross-layer cosine would be a number about two different "
                "spaces", [])
    return "computed", None, shared


def pair_cosines(entries: list[dict]) -> list[dict]:
    """Pairwise, matched-layer cosines over the report's vector rows."""
    pairs: list[dict] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            status, reason, shared = _pair_status(a, b)
            record = {
                "a": a["reference"], "b": b["reference"],
                "labelA": a["label"], "labelB": b["label"],
                "familyA": a["family"], "familyB": b["family"],
                "status": status,
                "layers": [
                    {"layer": layer,
                     "cosine": cosine_matrix(
                         [a["rows"][layer], b["rows"][layer]])[0][1]}
                    for layer in shared],
            }
            if reason is not None:
                record["reason"] = reason
            pairs.append(record)
    return pairs


def cosine_matrices_by_layer(entries: list[dict]) -> dict:
    """One square cosine matrix per layer where at least two rows have a
    direction, over exactly the members present at that layer."""
    by_layer: dict[str, dict] = {}
    layers = sorted({layer for entry in entries for layer in entry["layers"]})
    for layer in layers:
        members = [entry for entry in entries if layer in entry["layers"]]
        widths = {entry["hiddenSize"] for entry in members}
        models = {entry["modelID"] for entry in members if entry["modelID"]}
        if len(members) < 2 or len(widths) > 1 or len(models) > 1:
            # A mixed-basis layer has no single matrix; its pairs still carry
            # their own notApplicable reasons in `pairs`.
            continue
        rows = [entry["rows"][layer] for entry in members]
        by_layer[str(layer)] = {
            "layer": layer,
            "members": [entry["reference"] for entry in members],
            "labels": [entry["label"] for entry in members],
            "families": [entry["family"] for entry in members],
            "matrix": cosine_matrix(rows)}
    return by_layer


# -------------------------------------------------------------------- write


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_new(path: str, text: str) -> None:
    """Write a file that must not already exist.

    Run directories are immutable; the unique-directory helper already makes
    a collision improbable, and this makes an overwrite impossible rather
    than improbable.
    """
    if os.path.exists(path):
        raise FamilyReportError(
            f"refusing to overwrite '{path}' — report artifacts are immutable")
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)


def cosine_csv(pairs: list[dict]) -> str:
    """The cosine matrix in long form: one row per (pair, shared layer).

    Long rather than square because the matrix has a LAYER axis — a pair that
    shares no layer would have no cell in a square table, and writing 0 there
    is precisely the fabrication this report refuses. Such a pair gets one row
    with an empty cosine and its reason in ``status``.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["layer", "a", "b", "labelA", "labelB", "familyA",
                     "familyB", "cosine", "status", "reason"])
    for pair in pairs:
        if pair["status"] != "computed":
            writer.writerow(["", pair["a"], pair["b"], pair["labelA"],
                             pair["labelB"], pair["familyA"], pair["familyB"],
                             "", "notApplicable", pair.get("reason", "")])
            continue
        for cell in pair["layers"]:
            writer.writerow([cell["layer"], pair["a"], pair["b"],
                             pair["labelA"], pair["labelB"], pair["familyA"],
                             pair["familyB"], f"{cell['cosine']:.6f}",
                             "computed", ""])
    return buffer.getvalue()


def _format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


#: How many per-layer lines the readable summary prints before deferring to
#: the JSON/CSV. A full-depth grand-mean artifact has 62 of them, which turns
#: the document a reader is meant to skim into a table dump; the complete
#: numbers are always in ``family-report.json`` / ``family-cosines.csv``.
TEXT_LAYER_LIMIT = 8


def _compact_layers(layers: list[int]) -> str:
    if not layers:
        return "(no nonzero direction)"
    if len(layers) <= TEXT_LAYER_LIMIT:
        return ", ".join(str(layer) for layer in layers)
    return (f"{len(layers)} layers, {layers[0]}–{layers[-1]} "
            f"(full list in {REPORT_JSON})")


def summary_text(report: dict) -> str:
    """The readable face of the same numbers. Deliberately plain: it states
    what each row is and what it did, and adjudicates nothing."""
    compared_layers: dict[str, set] = {}
    for pair in report["pairs"]:
        for cell in pair["layers"]:
            compared_layers.setdefault(pair["a"], set()).add(cell["layer"])
            compared_layers.setdefault(pair["b"], set()).add(cell["layer"])

    lines: list[str] = []
    lines.append(f"SteerLab cross-family descriptive report — "
                 f"{report.get('name') or report['runID']}")
    lines.append(f"generated {report['generatedAt']}  "
                 f"engine {report['engine']}  substrate {report['substrate']}")
    lines.append("")
    lines.append(DESCRIPTIVE_NOTICE)
    lines.append("")

    lines.append("IDENTITY")
    for entry in report["entries"]:
        lines.append(f"  [{entry['family']}] {entry['label']}")
        if entry.get("reference"):
            lines.append(f"      artifact   {entry['reference']}")
            lines.append(f"      method     "
                         f"{entry.get('extractionMethod') or '(unstamped)'}"
                         f"  (family from {entry['familySource']})")
            lines.append(f"      concept    {entry.get('concept') or '—'}")
            lines.append(f"      model      {entry.get('modelID') or '—'}"
                         f" @ {entry.get('revision') or '(unpinned)'}")
            layers = entry["layers"]
            lines.append("      layers     " + _compact_layers(layers))
            shown = entry["perLayer"]
            if len(shown) > TEXT_LAYER_LIMIT:
                # Prefer the layers this row is actually COMPARED at; those
                # are the ones a reader is here for.
                compared = compared_layers.get(entry["reference"], set())
                shown = [row for row in shown if row["layer"] in compared]
            for row in shown[:TEXT_LAYER_LIMIT]:
                units = row.get("normInResidualUnits")
                unit_text = (f", {_format_number(units)} residual-norm units"
                             if units is not None
                             else ", residual-norm units unavailable")
                lines.append(f"        L{row['layer']}: norm "
                             f"{_format_number(row['norm'])}{unit_text}")
            if len(entry["perLayer"]) > len(shown[:TEXT_LAYER_LIMIT]):
                lines.append(f"        (per-layer norms for all "
                             f"{len(entry['perLayer'])} layers in "
                             f"{REPORT_JSON})")
            conventions = entry.get("conventions") or {}
            if conventions:
                lines.append("      stamps     " + ", ".join(
                    f"{key}={_format_number(value)}"
                    for key, value in sorted(conventions.items())))
            sae = entry.get("gemmascopeSource")
            if sae:
                lines.append(
                    f"      SAE        {sae.get('release')}/{sae.get('saeID')}"
                    f" feature {sae.get('feature')} at layer {sae.get('layer')}"
                    f" ({sae.get('constructLabel') or 'unlabelled'})")
            for promotion in entry.get("promotions") or []:
                cells = ", ".join(
                    f"L{cell.get('layer')}@α{cell.get('alpha')}"
                    for cell in promotion["cells"])
                how = (promotion.get("promotion", {}).get("promotedBy")
                       if promotion.get("promoted") else "hand-created variant")
                lines.append(f"      seated in  {promotion['variant']} "
                             f"[{cells}] ({how})")
            if entry.get("qualification"):
                lines.append("      qualification artifact present "
                             f"({entry['qualification'].get('source')})")
        else:
            lines.append("      no residual-stream direction "
                         "(behaviour-only row)")
        if entry.get("note"):
            lines.append(f"      note       {entry['note']}")
    lines.append("")

    lines.append("MATCHED-LAYER COSINES")
    computed = [p for p in report["pairs"] if p["status"] == "computed"]
    if not computed:
        lines.append("  (no pair shares a layer with a nonzero direction)")
    for pair in computed:
        for cell in pair["layers"][:TEXT_LAYER_LIMIT]:
            lines.append(f"  L{cell['layer']}  {pair['labelA']} ↔ "
                         f"{pair['labelB']}: "
                         f"{_format_number(cell['cosine'])}")
        remaining = len(pair["layers"]) - TEXT_LAYER_LIMIT
        if remaining > 0:
            lines.append(f"       … {remaining} further shared layer(s) for "
                         f"{pair['labelA']} ↔ {pair['labelB']} in "
                         f"{COSINE_CSV}")
    for pair in report["pairs"]:
        if pair["status"] == "computed":
            continue
        lines.append(f"  n/a  {pair['labelA']} ↔ {pair['labelB']}: "
                     f"{pair.get('reason', 'not applicable')}")
    lines.append("")

    lines.append("BEHAVIOURAL SIGNATURES (copied from analyze artifacts)")
    any_behavior = False
    for entry in report["entries"]:
        behavior = entry.get("behavior")
        if not behavior:
            continue
        any_behavior = True
        lines.append(f"  [{entry['family']}] {entry['label']} — condition "
                     f"{behavior['condition']} "
                     f"(source {behavior['source']['analyze']}, strata "
                     f"{behavior['strata']}: {behavior['rowsCopied']} of "
                     f"{behavior['rowsAvailable']} row(s))")
        for row in behavior["rows"]:
            delta = row.get("deltaMean")
            stratum = row.get("stratum")
            where = f" [{row.get('stratifyBy')}={stratum}]" if stratum else ""
            lines.append(
                f"      {row.get('endpoint', '?')}{where}: delta "
                f"{_format_number(delta) if delta is not None else '—'}"
                f"  (n={row.get('n', '—')})")
    if not any_behavior:
        lines.append("  (no behavioural sources declared)")
    lines.append("")

    if report["advisories"]:
        lines.append("ADVISORIES")
        for advisory in report["advisories"]:
            lines.append(f"  - {advisory}")
        lines.append("")

    lines.append("PROVENANCE")
    for entry in report["entries"]:
        if not entry.get("reference"):
            continue
        lines.append(f"  {entry['reference']}")
        lines.append(f"      tensors  {entry['tensorSHA256']}")
        lines.append(f"      sidecar  {entry['sidecarSHA256']}")
    cited: set = set()
    for entry in report["entries"]:
        behavior = entry.get("behavior")
        if not behavior:
            continue
        source = behavior["source"]
        key = (source["analyze"], source["effectSizesSHA256"])
        if key in cited:
            continue          # one source cited once, however many rows read it
        cited.add(key)
        lines.append(f"  {source['analyze']}/{EFFECT_SIZES_CSV}")
        lines.append(f"      sha256   {source['effectSizesSHA256']}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- driver


def _vector_entry(entry_config: FamilyEntryConfig,
                  config: FamilyReportConfig,
                  root: str | None,
                  advisories: list[str]) -> dict:
    reference = entry_config.reference or ""
    artifact = load_artifact(reference)
    layers = layers_with_direction(artifact)
    family, family_source = family_for(artifact.sidecar.extractionMethod,
                                       entry_config.family)
    if family_source == "unrecognized-extraction-method":
        advisories.append(
            f"'{reference}' stamps extractionMethod "
            f"{artifact.sidecar.extractionMethod!r}, which this report has no "
            f"family label for — it is reported as its own family, never "
            f"folded into a known one")
    if family_source == "absent-extraction-method":
        advisories.append(
            f"'{reference}' stamps no extractionMethod — reported as family "
            f"'{UNSTAMPED_FAMILY}'; the family it belongs to cannot be read "
            "off the artifact")
    if not layers:
        advisories.append(
            f"'{reference}' has no nonzero row at any layer — it contributes "
            "identity but no geometry")
    if (artifact.sidecar.extractionMethod == "gemmaScopeSAE"
            and artifact.sidecar.gemmascopeConvention is None):
        advisories.append(
            f"'{reference}' is a Gemma Scope import with no "
            "gemmascopeConvention stamp — pre-convention scaling, not "
            "comparable to convention-stamped rows; re-import before evidence "
            "use")
    if not artifact.sidecar.residualNormPerLayer:
        advisories.append(
            f"'{reference}' carries no residualNormPerLayer — its norms are "
            "reported in raw units only, with no residual-norm denominator")

    label = entry_config.label or artifact.name
    record = {
        "reference": reference,
        "path": artifact.path,
        "name": artifact.name,
        "label": label,
        "family": family,
        "familySource": family_source,
        "extractionMethod": artifact.sidecar.extractionMethod,
        "concept": artifact.sidecar.concept,
        "modelID": artifact.sidecar.modelID,
        "revision": artifact.sidecar.revision,
        "stimulusSetHash": artifact.sidecar.stimulusSetHash,
        "extractionDate": artifact.sidecar.extractionDate,
        "layerCount": artifact.vectors.layer_count,
        "hiddenSize": artifact.vectors.hidden_size,
        "layers": layers,
        "perLayer": _per_layer_rows(artifact, layers),
        "conventions": _conventions(artifact),
        "tensorSHA256": artifact.tensor_sha256,
        "sidecarSHA256": artifact.sidecar_sha256,
        # Not serialized: the rows themselves, used for the cosines.
        "rows": {layer: list(artifact.vectors.per_layer[layer])
                 for layer in layers},
    }
    for key in ("gemmascopeSource", "optvec", "promotion", "pinnedFrom",
                "normBackfill"):
        block = _raw_block(artifact, key)
        if block is not None:
            record[key] = block
    qualification = discover_qualification(artifact)
    if qualification is not None:
        record["qualification"] = qualification
    if config.discover_promotions:
        promotions = discover_promotions(reference, root)
        if promotions:
            record["promotions"] = promotions
    if entry_config.note:
        record["note"] = entry_config.note
    return record


def _behavior_only_entry(entry_config: FamilyEntryConfig) -> dict:
    record = {
        "reference": None,
        "label": entry_config.label,
        "family": entry_config.family,
        "familySource": "declared",
        "layers": [],
        "perLayer": [],
        "geometry": "none — this family member carries no residual-stream "
                    "direction (e.g. a fine-tuned adapter), so it appears in "
                    "no cosine",
    }
    if entry_config.note:
        record["note"] = entry_config.note
    return record


def report(config: FamilyReportConfig, root: str | None = None) -> dict:
    """Build the cross-family descriptive report and write its own immutable
    run directory (``family-report.json`` + ``family-cosines.csv`` +
    ``family-report.txt`` + the canonical ``config.json``)."""
    advisories: list[str] = []
    entries: list[dict] = []
    for entry_config in config.artifacts:
        if entry_config.reference is None:
            entries.append(_behavior_only_entry(entry_config))
            continue
        entries.append(_vector_entry(entry_config, config, root, advisories))

    for entry_config, entry in zip(config.artifacts, entries):
        if entry_config.behavior is None:
            continue
        entry["behavior"] = load_behavior(entry_config.behavior, root)

    vector_entries = [entry for entry in entries if entry.get("rows")]
    pairs = pair_cosines(vector_entries)
    by_layer = cosine_matrices_by_layer(vector_entries)

    families: dict[str, list[str]] = {}
    for entry in entries:
        families.setdefault(entry["family"], []).append(entry["label"])
    if len(families) < 2:
        advisories.append(
            "every entry resolved to one family — this is a within-family "
            "table, not a cross-family side-by-side")

    serializable = []
    for entry in entries:
        copy = {key: value for key, value in entry.items() if key != "rows"}
        serializable.append(copy)

    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "runType": RUN_TYPE,
        "claim": "descriptive",
        "notice": DESCRIPTIVE_NOTICE,
        "name": config.name,
        "generatedAt": _now_iso8601(),
        "engine": engine_version(),
        "substrate": SUBSTRATE,
        "config": config.to_dict(),
        "entries": serializable,
        "families": {name: sorted(labels)
                     for name, labels in sorted(families.items())},
        "pairs": pairs,
        "cosineMatricesByLayer": by_layer,
        "advisories": advisories,
    }

    name = (config.name or "").strip() or f"{len(entries)}-rows"
    run_directory = paths.make_unique_run_directory(f"family-report-{name}",
                                                    root)
    payload["runID"] = os.path.basename(os.path.normpath(run_directory))

    _write_new(os.path.join(run_directory, COSINE_CSV), cosine_csv(pairs))
    _write_new(os.path.join(run_directory, SUMMARY_TXT), summary_text(payload))
    _write_new(os.path.join(run_directory, REPORT_JSON),
               json.dumps(payload, indent=2, sort_keys=True) + "\n")

    model_ids = sorted({entry.get("modelID") for entry in entries
                        if entry.get("modelID")})
    write_run_config(
        run_directory, RUN_TYPE,
        model_id=model_ids[0] if len(model_ids) == 1 else None,
        notes={"claim": "descriptive",
               "entryCount": len(entries),
               "families": sorted(families),
               "artifacts": [entry["reference"] for entry in entries
                             if entry.get("reference")],
               "behavioralSources": sorted(
                   {entry["behavior"]["source"]["analyze"]
                    for entry in entries if entry.get("behavior")}),
               "modelIDs": model_ids,
               "advisories": advisories})
    return {"runDirectory": run_directory, **payload}


__all__ = ["BEHAVIOR_COLUMNS", "BehaviorRef", "COSINE_CSV",
           "DESCRIPTIVE_NOTICE", "OMISSION_REASON",
           "FAMILY_BY_METHOD", "FamilyEntryConfig", "FamilyReportConfig",
           "FamilyReportError", "OMITTED_BEHAVIOR_COLUMNS", "REPORT_JSON",
           "RUN_TYPE", "SCHEMA_VERSION", "SUMMARY_TXT", "UNSTAMPED_FAMILY",
           "OptVecArtifactError", "cosine_csv", "cosine_matrices_by_layer",
           "discover_promotions", "discover_qualification", "family_for",
           "layers_with_direction", "load_behavior", "load_config",
           "pair_cosines", "report", "summary_text"]
