"""The SAE feature QUALIFICATION artifact — citable evidence, not a seating
mechanism (SAE-VECTOR-INTERVENTION proposal r2, §6 and §8 P0-4).

Study A asks one question per imported feature: *does this decoder direction
causally move a held-out measure of its intended construct, in both signs,
across a dose ladder, without lexical leakage explaining it away, without a
discriminant control moving with it, and without coherence or format
collapse?* The answer is a decision a human makes, on numbers a run produced.
This module is the durable place that decision and those numbers live.

**What this is NOT.** It is not a second birth certificate and it does not
seat anything. Seating stays exactly where it was for every other vector
family: sweep selects a cell under a declared criterion, ``promote`` mints the
agent with its ``promotion`` block (:mod:`promote`). A qualification artifact
functions like capability-battery evidence — something the promotion chain
CITES (``promotion.qualification``) and something freeze can notice is absent
(a non-blocking advisory). Nothing here can create, block, or alter a
promotion; a promotion that cites nothing behaves exactly as it always did.

**Bound to bytes, not to names.** A feature is only meaningfully qualified if
the thing later injected is the same row of the same dictionary. The record
therefore carries the imported artifact's own ``gemmascopeSource`` identity —
repository + revision, release / saeID / feature, layer, and the raw
``decoderRowHash`` — and :func:`record` re-reads the artifact's sidecar and
refuses to write when the declared identity disagrees with the bytes on disk.
The citation check in ``promote`` re-runs the same comparison, so a
qualification can never travel to a different feature.

**Concept-agnostic** (CLAUDE.md's standing rule): constructs, metrics, and
gate thresholds are DATA. Nothing here knows what "compassion" or "textualism"
means, what a good value is, or which direction counts as success — the
researcher declares the metric name, the measured values, and the decision.
The schema checks SHAPE and internal consistency only. It never computes a
verdict, because a verdict computed from the same numbers it records would
make the artifact self-certifying.

Written IMMUTABLY into its own ``runs/<stamp>-sae-qualification-<feature>/``
directory, beside the canonical ``config.json``, exactly like every other
durable evidence artifact on this engine. A qualification is never edited: a
revised judgement is a new record, and both remain readable.

Cross-engine: server-only, like the whole SAE import path (Gemma Scope is a
PyTorch/HF artifact set). The one key that crosses is
``promotion.qualification`` inside a variant artifact's birth certificate —
additive and optional; a Swift reader must round-trip it untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCHEMA_VERSION = 1

#: The artifact's file name inside its run directory.
FILENAME = "sae-feature-qualification.json"

#: The run type stamped into the run directory's canonical ``config.json``.
RUN_TYPE = "sae-qualification"

#: Where the inputs shape is documented by example (named in the CLI usage).
TEMPLATE_PATH = ("prompts/templates/sae-qualification/"
                 "sae-qualification-inputs-template.json")

#: The only two decisions. Deliberately binary: "promising", "partial", or
#: "pending" would all end up cited as evidence of qualification by someone
#: reading quickly, and the citation check has no way to weigh a hedge.
DECISIONS = ("accept", "reject")

#: Which side of the direction a measurement was taken on. ``baseline`` rows
#: (dose 0, unsteered) are legal and useful, and never satisfy grid coverage.
SIGNS = ("positive", "negative", "baseline")

#: The signs a dose ladder is claimed for, unless the record says otherwise.
DEFAULT_SIGNS = ("positive", "negative")

#: Which way the metric moved. Recorded, never inferred from the value: the
#: sign convention of a researcher-named metric is not knowable here.
DIRECTIONS = ("increase", "decrease", "none")

_TOP_LEVEL_KEYS = (
    "schemaVersion", "artifact", "feature", "doseGrid", "signs",
    "constructProbe", "lexicalLeakage", "discriminantControls",
    "coherenceGate", "doseResponse", "decision", "evidenceRuns", "notes",
    "recordedAt", "recordedBy", "substrate")

#: Keys a caller supplies; the rest are stamped by :func:`record`.
_INPUT_KEYS = tuple(k for k in _TOP_LEVEL_KEYS
                    if k not in ("artifact", "recordedAt", "recordedBy",
                                 "substrate"))

_FEATURE_KEYS = ("repository", "repositoryRevision", "release", "saeID",
                 "feature", "layer", "decoderRowHash", "constructLabel",
                 "modelID", "modelRevision")

#: The identity legs a record MUST declare. Everything else in ``feature`` is
#: copied from the artifact's sidecar (and cross-checked when declared).
_FEATURE_REQUIRED = ("feature", "layer", "decoderRowHash")

_BLOCK_KEYS = ("metric", "higherIsBetter", "heldOutSetHash", "results",
               "notes")
_DISCRIMINANT_KEYS = _BLOCK_KEYS + ("construct",)
_COHERENCE_KEYS = _BLOCK_KEYS + ("threshold", "passed")
_RESULT_KEYS = ("dose", "sign", "value", "direction", "n", "ciLower",
                "ciUpper", "run", "notes")
_DOSE_RESPONSE_KEYS = ("monotone", "signSymmetric", "spearmanRho", "summary")
_DECISION_KEYS = ("decision", "rationale", "date", "decidedBy")
_EVIDENCE_RUN_KEYS = ("label", "path", "describes")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Doses are floats read from JSON; compare them the way the sweep does.
_DOSE_TOLERANCE = 1e-12


class QualificationError(Exception):
    """A qualification record that does not validate must never be written or
    cited: the citation promises a reader can re-read the evidence, and a hash
    over malformed bytes certifies only that they did not change."""


# ---------------------------------------------------------------------------
# Parsed shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """One number, at one dose, on one side of the direction."""

    dose: float
    sign: str
    value: float
    direction: str | None = None
    n: int | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    run: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class MeasurementBlock:
    """A named metric plus its rows. ``construct`` is set only for the
    discriminant controls, whose whole point is naming the OTHER construct
    that must not move."""

    metric: str
    results: tuple[Measurement, ...]
    higher_is_better: bool | None = None
    held_out_set_hash: str | None = None
    notes: str = ""
    construct: str | None = None
    threshold: float | None = None
    passed: bool | None = None


@dataclass(frozen=True)
class Decision:
    decision: str
    rationale: str
    date: str
    decided_by: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


@dataclass(frozen=True)
class Qualification:
    schema_version: int
    feature: dict
    dose_grid: tuple[float, ...]
    signs: tuple[str, ...]
    construct_probe: MeasurementBlock
    lexical_leakage: MeasurementBlock
    discriminant_controls: tuple[MeasurementBlock, ...]
    coherence_gate: MeasurementBlock
    dose_response: dict
    decision: Decision
    artifact: str = ""
    evidence_runs: tuple[dict, ...] = ()
    notes: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def decoder_row_hash(self) -> str:
        return str(self.feature.get("decoderRowHash") or "")

    def summary(self) -> dict:
        """Plain counts for the CLI and any surface that lists records. No
        verdicts — 'enough evidence' is a research judgement."""
        return {
            "schemaVersion": self.schema_version,
            "artifact": self.artifact,
            "feature": self.feature.get("feature"),
            "layer": self.feature.get("layer"),
            "constructLabel": self.feature.get("constructLabel"),
            "decoderRowHash": self.decoder_row_hash,
            "decision": self.decision.decision,
            "decisionDate": self.decision.date,
            "doseGrid": list(self.dose_grid),
            "signs": list(self.signs),
            "constructProbeRows": len(self.construct_probe.results),
            "lexicalLeakageRows": len(self.lexical_leakage.results),
            "discriminantControls": [b.construct
                                     for b in self.discriminant_controls],
            "coherenceGatePassed": self.coherence_gate.passed,
            "doseResponseMonotone": self.dose_response.get("monotone"),
            "evidenceRuns": [dict(r) for r in self.evidence_runs],
        }


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def _reject_unknown(payload: dict, allowed: tuple[str, ...], label: str) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise QualificationError(
            f"{label} has unknown key(s) {', '.join(unknown)} — the schema is "
            f"CLOSED so a typo'd key can never be silently ignored (allowed: "
            f"{', '.join(allowed)})")


def _text(payload: dict, key: str, label: str, *, required: bool = True) -> str:
    value = payload.get(key)
    if value is None or value == "":
        if required:
            raise QualificationError(f"{label}: '{key}' is required")
        return ""
    if not isinstance(value, str):
        raise QualificationError(f"{label}: '{key}' must be a string")
    return value


def _number(payload: dict, key: str, label: str, *,
            required: bool = True) -> float | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise QualificationError(f"{label}: '{key}' is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label}: '{key}' must be a number")
    return float(value)


def _integer(payload: dict, key: str, label: str, *,
             required: bool = True) -> int | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise QualificationError(f"{label}: '{key}' is required")
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationError(f"{label}: '{key}' must be an integer")
    return value


def _boolean(payload: dict, key: str, label: str, *,
             required: bool = True) -> bool | None:
    value = payload.get(key)
    if value is None:
        if required:
            raise QualificationError(f"{label}: '{key}' is required")
        return None
    if not isinstance(value, bool):
        raise QualificationError(f"{label}: '{key}' must be true or false")
    return value


def _enum(payload: dict, key: str, allowed: tuple[str, ...], label: str, *,
          required: bool = True) -> str | None:
    value = payload.get(key)
    if value is None or value == "":
        if required:
            raise QualificationError(f"{label}: '{key}' is required")
        return None
    if value not in allowed:
        raise QualificationError(
            f"{label}: '{key}' must be one of {', '.join(allowed)} "
            f"(got {value!r})")
    return str(value)


def _relative_path(value: str, label: str, key: str) -> str:
    if os.path.isabs(value):
        raise QualificationError(
            f"{label}: '{key}' must be WORKSPACE-RELATIVE — an absolute path "
            "names one machine's filesystem and resolves to nothing on the "
            "cluster (the authoring client's workspace is the source of "
            "truth)")
    return value


def _measurement(payload: object, label: str, *,
                 require_direction: bool) -> Measurement:
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} must be an object")
    _reject_unknown(payload, _RESULT_KEYS, label)
    dose = _number(payload, "dose", label)
    if dose is None or dose < 0:
        raise QualificationError(
            f"{label}: 'dose' must be >= 0 (doses are residual-norm units, "
            "and the sign lives in 'sign')")
    sign = _enum(payload, "sign", SIGNS, label)
    direction = _enum(payload, "direction", DIRECTIONS, label,
                      required=require_direction)
    run = payload.get("run")
    if run is not None:
        if not isinstance(run, str):
            raise QualificationError(f"{label}: 'run' must be a string path")
        run = _relative_path(run, label, "run")
    n = _integer(payload, "n", label, required=False)
    if n is not None and n < 0:
        raise QualificationError(f"{label}: 'n' must be >= 0")
    return Measurement(
        dose=float(dose), sign=str(sign),
        value=float(_number(payload, "value", label)),
        direction=direction, n=n,
        ci_lower=_number(payload, "ciLower", label, required=False),
        ci_upper=_number(payload, "ciUpper", label, required=False),
        run=run or None,
        notes=_text(payload, "notes", label, required=False))


def _block(payload: object, label: str, *, keys: tuple[str, ...],
           require_direction: bool = False,
           require_construct: bool = False,
           require_passed: bool = False) -> MeasurementBlock:
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} must be an object")
    _reject_unknown(payload, keys, label)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise QualificationError(
            f"{label}: 'results' must be an array of measurement rows")
    results = tuple(
        _measurement(row, f"{label}.results[{i}]",
                     require_direction=require_direction)
        for i, row in enumerate(rows))
    construct = None
    if "construct" in keys:
        construct = _text(payload, "construct", label,
                          required=require_construct) or None
    threshold = None
    if "threshold" in keys:
        threshold = _number(payload, "threshold", label, required=False)
    passed = None
    if "passed" in keys:
        passed = _boolean(payload, "passed", label, required=require_passed)
    return MeasurementBlock(
        metric=_text(payload, "metric", label),
        results=results,
        higher_is_better=_boolean(payload, "higherIsBetter", label,
                                  required=False),
        held_out_set_hash=_text(payload, "heldOutSetHash", label,
                                required=False) or None,
        notes=_text(payload, "notes", label, required=False),
        construct=construct, threshold=threshold, passed=passed)


def _dose_grid(payload: dict) -> tuple[float, ...]:
    grid = payload.get("doseGrid")
    if not isinstance(grid, list) or not grid:
        raise QualificationError(
            "'doseGrid' must be a non-empty array of the doses this record "
            "claims to have measured — it is what the coverage check is "
            "checked against")
    values: list[float] = []
    for index, raw in enumerate(grid):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise QualificationError(f"doseGrid[{index}] must be a number")
        value = float(raw)
        if value <= 0:
            raise QualificationError(
                f"doseGrid[{index}] must be > 0 — a dose ladder is the "
                "STEERED doses; record the unsteered arm as a 'baseline' row")
        if any(abs(value - seen) <= _DOSE_TOLERANCE for seen in values):
            raise QualificationError(
                f"doseGrid[{index}] ({value:g}) is a duplicate — a dose "
                "appears once in the claimed grid")
        values.append(value)
    return tuple(values)


def _signs(payload: dict) -> tuple[str, ...]:
    raw = payload.get("signs")
    if raw is None:
        return DEFAULT_SIGNS
    if not isinstance(raw, list) or not raw:
        raise QualificationError(
            "'signs' must be a non-empty array (omit the key to claim both "
            f"signs: {', '.join(DEFAULT_SIGNS)})")
    out: list[str] = []
    for index, value in enumerate(raw):
        if value not in ("positive", "negative"):
            raise QualificationError(
                f"signs[{index}] must be 'positive' or 'negative' — "
                "'baseline' is a row, never a claimed side of the ladder")
        if value in out:
            raise QualificationError(f"signs[{index}] ({value}) is a duplicate")
        out.append(str(value))
    return tuple(out)


def _dose_response(payload: object) -> dict:
    label = "doseResponse"
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} must be an object")
    _reject_unknown(payload, _DOSE_RESPONSE_KEYS, label)
    out: dict = {"monotone": _boolean(payload, "monotone", label)}
    for key, parse in (("signSymmetric", _boolean),
                       ("spearmanRho", _number)):
        value = parse(payload, key, label, required=False)
        if value is not None:
            out[key] = value
    summary = _text(payload, "summary", label, required=False)
    if summary:
        out["summary"] = summary
    return out


def _decision(payload: object) -> Decision:
    label = "decision"
    if not isinstance(payload, dict):
        raise QualificationError(
            f"{label} must be an object "
            '{"decision": "accept"|"reject", "rationale": …, "date": …} — a '
            "record with no decision is a pile of numbers, not evidence")
    _reject_unknown(payload, _DECISION_KEYS, label)
    date = _text(payload, "date", label)
    if not _DATE_RE.match(date):
        raise QualificationError(
            f"{label}: 'date' must be an ISO date (YYYY-MM-DD), got {date!r}")
    return Decision(
        decision=str(_enum(payload, "decision", DECISIONS, label)),
        rationale=_text(payload, "rationale", label),
        date=date,
        decided_by=_text(payload, "decidedBy", label, required=False))


def _feature(payload: object) -> dict:
    label = "feature"
    if not isinstance(payload, dict):
        raise QualificationError(
            f"{label} must be an object carrying the imported artifact's "
            "gemmascopeSource identity")
    _reject_unknown(payload, _FEATURE_KEYS, label)
    out: dict = {
        "feature": _integer(payload, "feature", label),
        "layer": _integer(payload, "layer", label),
        "decoderRowHash": _text(payload, "decoderRowHash", label),
    }
    for key in _FEATURE_KEYS:
        if key in _FEATURE_REQUIRED:
            continue
        value = payload.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise QualificationError(f"{label}: '{key}' must be a string")
        out[key] = value
    if out["feature"] < 0 or out["layer"] < 0:
        raise QualificationError(
            f"{label}: 'feature' and 'layer' are dictionary coordinates and "
            "must be >= 0")
    return out


def _evidence_runs(payload: object) -> tuple[dict, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise QualificationError("'evidenceRuns' must be an array")
    out: list[dict] = []
    for index, row in enumerate(payload):
        label = f"evidenceRuns[{index}]"
        if not isinstance(row, dict):
            raise QualificationError(f"{label} must be an object")
        _reject_unknown(row, _EVIDENCE_RUN_KEYS, label)
        entry = {"path": _relative_path(_text(row, "path", label), label,
                                        "path")}
        for key in ("label", "describes"):
            value = _text(row, key, label, required=False)
            if value:
                entry[key] = value
        out.append(entry)
    return tuple(out)


def from_dict(payload: object) -> Qualification:
    """Validate a qualification payload (shape only — no filesystem)."""
    if not isinstance(payload, dict):
        raise QualificationError(
            "an SAE feature qualification must be a JSON object")
    _reject_unknown(payload, _TOP_LEVEL_KEYS, "the qualification record")
    version = payload.get("schemaVersion", SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise QualificationError("schemaVersion must be an integer")
    if version != SCHEMA_VERSION:
        raise QualificationError(
            f"unsupported schemaVersion {version} — this engine reads "
            f"schemaVersion {SCHEMA_VERSION}")
    artifact = payload.get("artifact")
    if artifact is not None:
        if not isinstance(artifact, str) or not artifact:
            raise QualificationError("'artifact' must be a path string")
        _relative_path(artifact, "the qualification record", "artifact")
    return Qualification(
        schema_version=version,
        artifact=str(artifact or ""),
        feature=_feature(payload.get("feature")),
        dose_grid=_dose_grid(payload),
        signs=_signs(payload),
        construct_probe=_block(payload.get("constructProbe"), "constructProbe",
                               keys=_BLOCK_KEYS, require_direction=True),
        lexical_leakage=_block(payload.get("lexicalLeakage"), "lexicalLeakage",
                               keys=_BLOCK_KEYS),
        discriminant_controls=_discriminants(payload.get("discriminantControls")),
        coherence_gate=_block(payload.get("coherenceGate"), "coherenceGate",
                              keys=_COHERENCE_KEYS, require_passed=True),
        dose_response=_dose_response(payload.get("doseResponse")),
        decision=_decision(payload.get("decision")),
        evidence_runs=_evidence_runs(payload.get("evidenceRuns")),
        notes=_text(payload, "notes", "the qualification record",
                    required=False),
        raw=payload)


def _discriminants(payload: object) -> tuple[MeasurementBlock, ...]:
    if payload is None:
        raise QualificationError(
            "'discriminantControls' is required — an empty array is the way "
            "to say 'none were run', which is a research decision the record "
            "should state rather than leave to inference")
    if not isinstance(payload, list):
        raise QualificationError("'discriminantControls' must be an array")
    blocks = []
    for index, row in enumerate(payload):
        blocks.append(_block(row, f"discriminantControls[{index}]",
                             keys=_DISCRIMINANT_KEYS, require_construct=True))
    constructs = [b.construct for b in blocks]
    duplicate = next((c for c in constructs
                      if constructs.count(c) > 1), None)
    if duplicate is not None:
        raise QualificationError(
            f"discriminantControls names construct {duplicate!r} twice — one "
            "control construct is one block")
    return tuple(blocks)


def from_bytes(data: bytes) -> Qualification:
    try:
        payload = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise QualificationError(
            f"the qualification record is not UTF-8 text: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QualificationError(
            f"the qualification record is not valid JSON: {exc}") from exc
    return from_dict(payload)


def content_hash(data: bytes) -> str:
    """SHA-256 of the FILE BYTES — the same mechanical rule as every other
    pinned input on this engine (markersHash, the candidate roster). A
    citation names bytes, so reformatting the file IS a change."""
    return hashlib.sha256(data).hexdigest()


def load(path: str, root: str | None = None) -> tuple[Qualification, str]:
    """Load + validate a qualification record, returning it with the hash of
    the exact bytes read. ``path`` may be workspace-relative or absolute (a
    human typing a path at a terminal is not a stored reference)."""
    from . import paths
    resolved = paths.resolve(path, root)
    try:
        with open(resolved, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise QualificationError(
            f"no SAE feature qualification at {path}: {exc}") from exc
    return from_bytes(data), content_hash(data)


# ---------------------------------------------------------------------------
# Internal consistency + binding to the artifact's bytes
# ---------------------------------------------------------------------------

def grid_coverage_violations(record: Qualification) -> list[str]:
    """Every claimed (dose, sign) must actually have a construct-probe row.

    A record whose ``doseGrid`` advertises a ladder it did not measure would
    be cited as ladder evidence; the coverage check is what makes the grid a
    claim rather than a decoration. Deliberately narrow: it never asks whether
    the numbers are GOOD."""
    violations: list[str] = []
    for dose in record.dose_grid:
        for sign in record.signs:
            if not any(row.sign == sign
                       and abs(row.dose - dose) <= _DOSE_TOLERANCE
                       for row in record.construct_probe.results):
                violations.append(
                    f"constructProbe has no result at dose {dose:g} on the "
                    f"{sign} side, but doseGrid claims that dose — a claimed "
                    "ladder rung with no measurement cannot be cited")
    return violations


def artifact_sidecar(reference: str, root: str | None = None) -> dict:
    """The sidecar JSON of a vector artifact named by an extension-less
    locator (``<runDir>/<name>``). Raises :class:`QualificationError` naming
    exactly what is missing — a qualification bound to an unreadable artifact
    is bound to nothing."""
    from . import paths
    resolved = paths.resolve_artifact(reference, root)
    sidecar_path = resolved + ".json"
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except OSError as exc:
        raise QualificationError(
            f"vector artifact {reference!r} names no readable sidecar at "
            f"{sidecar_path} (expected the EXTENSION-LESS base path, e.g. "
            f"runs/<run>/sae-feature-62389): {exc}") from exc
    except ValueError as exc:
        raise QualificationError(
            f"vector artifact {reference!r} has an unreadable sidecar "
            f"({sidecar_path}): {exc}") from exc
    if not isinstance(sidecar, dict):
        raise QualificationError(
            f"vector artifact {reference!r} sidecar is not a JSON object")
    return sidecar


def feature_identity(sidecar: dict) -> dict:
    """The identity legs of a DIRECT-ID SAE import, read off its sidecar.

    Returns ``{}`` when the artifact is not one: a CAA/grand-mean/OptVec
    direction has no feature identity to match, and pretending otherwise would
    let a qualification be cited for a vector it never described."""
    source = sidecar.get("gemmascopeSource")
    if not isinstance(source, dict):
        return {}
    if source.get("importPath") != "direct-feature-id":
        return {}
    out: dict = {}
    for key in ("repository", "repositoryRevision", "release", "saeID",
                "constructLabel"):
        value = source.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    for key in ("feature", "layer"):
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    digest = source.get("decoderRowHash")
    if isinstance(digest, str) and digest:
        out["decoderRowHash"] = digest
    model = sidecar.get("modelID")
    if isinstance(model, str) and model:
        out["modelID"] = model
    revision = sidecar.get("revision")
    if isinstance(revision, str) and revision:
        out["modelRevision"] = revision
    return out


def is_direct_id_sae_artifact(sidecar: dict) -> bool:
    """Whether a sidecar describes a feature imported BY ID (the artifacts a
    qualification can describe, and the ones the freeze advisory looks for)."""
    return bool(feature_identity(sidecar))


def resolved_feature_identity(reference: str,
                              root: str | None = None) -> tuple[dict, str]:
    """``(identity, the artifact the identity was read from)``, following a
    MATERIALIZED COPY back to the import it was copied from.

    Necessary because the artifact a study actually injects is usually not the
    import. A pinned concept materializes: every run — sweep included —
    re-verifies the pinned bytes and writes its own copy into the run
    directory, stamped ``pinnedFrom: {path, sha256TensorHash,
    sha256SidecarHash}``. That copy is what ``promote`` matches by recipe
    identity and what a variant's ``vectorArtifactID`` names, and
    :func:`_persist_vectors` does not carry ``gemmascopeSource`` onto it. Read
    naively, an SAE arm would therefore look like an ordinary direction the
    moment it was swept — the citation would refuse and the freeze advisory
    would go quiet, both silently.

    Following the pointer is safe precisely because it is HASH-PINNED: the
    origin's sidecar must hash to the value the copy recorded, or the trail is
    refused rather than trusted. One hop only — a copy of a copy would be a
    provenance chain nothing in the engine writes, and following an
    unbounded chain would be inventing a rule.
    """
    sidecar = artifact_sidecar(reference, root)
    identity = feature_identity(sidecar)
    if identity:
        return identity, reference
    pinned_from = sidecar.get("pinnedFrom")
    origin = (pinned_from or {}).get("path") if isinstance(pinned_from, dict) \
        else None
    if not isinstance(origin, str) or not origin:
        return {}, reference
    from . import paths
    origin_path = paths.resolve_artifact(origin, root) + ".json"
    expected = (pinned_from or {}).get("sha256SidecarHash")
    try:
        with open(origin_path, "rb") as handle:
            data = handle.read()
    except OSError:
        # The origin is gone. Say nothing rather than guess: the copy's own
        # bytes are still hash-pinned by the manifest, but its FEATURE
        # identity is unreadable, and a citation must name bytes it saw.
        return {}, reference
    if isinstance(expected, str) and expected \
            and hashlib.sha256(data).hexdigest() != expected:
        return {}, reference
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, reference
    if not isinstance(payload, dict):
        return {}, reference
    return feature_identity(payload), origin


def identity_violations(declared: dict, actual: dict) -> list[str]:
    """Compare a record's declared feature identity against the artifact's.

    The ``decoderRowHash`` is the load-bearing leg — it is the hash of the RAW
    decoder row, so it survives the storage rescale and identifies the exact
    published bytes. Every other declared field must agree too, but an
    UNDECLARED field is not a violation: a record may name the minimum
    (feature, layer, decoder row) and let the writer fill the rest in."""
    if not actual:
        return ["the cited vector artifact is not a direct-ID Gemma Scope "
                "import (no gemmascopeSource with importPath "
                "'direct-feature-id') — a feature qualification describes one "
                "SAE decoder row and cannot be bound to another kind of "
                "direction"]
    violations: list[str] = []
    for key in sorted(set(declared) & set(actual)):
        if declared[key] != actual[key]:
            violations.append(
                f"feature.{key} declares {declared[key]!r} but the artifact "
                f"records {actual[key]!r}")
    if "decoderRowHash" not in actual:
        violations.append(
            "the cited artifact records no decoderRowHash — it was imported "
            "before the identity stamp existed, so the qualification cannot "
            "be bound to its bytes; re-import the feature")
    return violations


def consistency_violations(record: Qualification, *,
                           artifact: str | None = None,
                           root: str | None = None) -> list[str]:
    """Everything that must hold before a record is written or cited: the
    dose grid is covered, and the artifact it names exists and is the feature
    it claims. Returns the violations; the caller decides whether they refuse
    (``record``, ``promote --qualification``) or merely report (``show``)."""
    violations = grid_coverage_violations(record)
    reference = artifact or record.artifact
    if not reference:
        return violations + [
            "the record names no vector artifact — a qualification that is "
            "not bound to bytes certifies nothing"]
    try:
        sidecar = artifact_sidecar(reference, root)
    except QualificationError as exc:
        return violations + [str(exc)]
    return violations + identity_violations(record.feature,
                                            feature_identity(sidecar))


# ---------------------------------------------------------------------------
# Writing (immutably)
# ---------------------------------------------------------------------------

def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _workspace_relative(path: str, root: str | None) -> str:
    from . import paths
    base = paths.project_root() if root is None else root
    absolute = os.path.abspath(paths.resolve(path, root))
    try:
        relative = os.path.relpath(absolute, os.path.abspath(base))
    except ValueError:  # different drive on Windows; keep what we were given
        return path
    if relative.startswith(os.pardir + os.sep) or relative == os.pardir:
        # Outside the workspace: a stored reference that resolves to nothing
        # anywhere else is worse than none.
        raise QualificationError(
            f"{path!r} resolves outside the workspace ({base}) — pinned "
            "references are workspace-relative so the record reads on any "
            "machine")
    return relative.replace(os.sep, "/")


def record(*, inputs: object, artifact: str, root: str | None = None,
           run_directory: str | None = None) -> dict:
    """Assemble and write one immutable ``sae-feature-qualification.json``.

    ``inputs`` is the researcher's declared evidence (the CLI reads it from a
    JSON file). ``artifact`` names the imported SAE vector artifact by its
    EXTENSION-LESS locator. The write happens only when:

    * the inputs validate against the closed schema;
    * the artifact's sidecar exists and IS a direct-ID Gemma Scope import;
    * every feature-identity leg the record declares agrees with that sidecar
      (the ``decoderRowHash`` above all — it names the exact published bytes);
    * the claimed dose grid is covered by construct-probe rows on every
      claimed sign;
    * a decision with a rationale and a date is present (schema-enforced).

    The written record carries the FULL identity read from the sidecar, so a
    reader never has to go find the artifact to know which feature was
    qualified. Returns ``{"path", "runDirectory", "contentHash", "summary"}``.
    """
    from . import paths
    from .run_config import write_run_config

    if isinstance(inputs, dict) and "artifact" in inputs:
        raise QualificationError(
            "the inputs file must not carry an 'artifact' key — the artifact "
            "is named on the command line and stamped from what was actually "
            "verified, so two possibly-disagreeing references never exist")
    if isinstance(inputs, dict):
        _reject_unknown(inputs, _INPUT_KEYS, "the qualification inputs")
    declared = from_dict(inputs)

    sidecar = artifact_sidecar(artifact, root)
    actual = feature_identity(sidecar)
    violations = identity_violations(declared.feature, actual)
    violations += grid_coverage_violations(declared)
    if violations:
        raise QualificationError(
            "refusing to record this qualification — "
            + "; ".join(violations))

    relative_artifact = _workspace_relative(artifact, root)
    feature_block = dict(actual)
    # Declared-but-unrecorded legs survive (they were checked for agreement
    # above); the artifact's own record wins where both exist.
    for key, value in declared.feature.items():
        feature_block.setdefault(key, value)

    payload = dict(declared.raw)
    payload["schemaVersion"] = SCHEMA_VERSION
    payload["artifact"] = relative_artifact
    payload["feature"] = feature_block
    payload["signs"] = list(declared.signs)
    payload["recordedAt"] = _now_iso8601()
    payload["substrate"] = _substrate()
    payload["recordedBy"] = _engine_version()

    # Re-validate the assembled bytes: what is written is what will later be
    # loaded and cited, and a writer that validates only its INPUT can still
    # emit something the reader refuses.
    blob = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    data = blob.encode("utf-8")
    from_bytes(data)

    directory = run_directory or paths.make_unique_run_directory(
        f"{RUN_TYPE}-{feature_block.get('feature')}", root)
    os.makedirs(directory, exist_ok=True)
    write_run_config(directory, RUN_TYPE,
                     model_id=feature_block.get("modelID"),
                     revision=feature_block.get("modelRevision"))
    path = os.path.join(directory, FILENAME)
    if os.path.exists(path):
        raise QualificationError(
            f"refusing to overwrite {path} — qualification records are "
            "immutable evidence; a revised judgement is a NEW record (both "
            "stay readable, and the promotion cites one by hash)")
    # Exclusive create: two concurrent recorders must not both believe they
    # wrote the file (the same rule the run-directory allocator follows).
    try:
        handle = open(path, "x", encoding="utf-8")
    except FileExistsError as exc:
        raise QualificationError(
            f"refusing to overwrite {path} — qualification records are "
            "immutable evidence") from exc
    with handle:
        handle.write(blob)
    return {"path": path,
            "runDirectory": directory,
            "artifact": relative_artifact,
            "contentHash": content_hash(data),
            "summary": from_bytes(data).summary()}


def _engine_version() -> str:
    from ..build_identity import engine_version
    return engine_version()


def _substrate() -> str:
    from ..steering.vector_store import SUBSTRATE
    return SUBSTRATE


# ---------------------------------------------------------------------------
# Citation (used by promote) and the roster consistency check
# ---------------------------------------------------------------------------

def citation(qualification_path: str, *, artifact_reference: str,
             root: str | None = None) -> dict:
    """The ``promotion.qualification`` block, or a refusal.

    Additive by construction: a promotion that cites nothing never calls this
    and behaves exactly as it did before. What it refuses:

    * a record that does not validate, or whose grid claim is uncovered;
    * a **rejected** feature — a promotion may not cite the evidence that says
      the direction failed qualification as though it supported seating it;
    * a record whose feature identity disagrees with the artifact actually
      being promoted (including the case where that artifact is not an SAE
      import at all). The identity check is the whole point: a citation that
      can travel between features is decoration.

    Containment first, BEFORE the file is opened: this is reachable from the
    HTTP promote route, and a caller-named path must not be able to make the
    server read outside the workspace (security posture: on a shared cluster
    node, assume other local users can reach 127.0.0.1).
    """
    relative = _workspace_relative(qualification_path, root)
    record_obj, digest = load(qualification_path, root)
    identity, origin = resolved_feature_identity(artifact_reference, root)
    violations = grid_coverage_violations(record_obj)
    violations += identity_violations(record_obj.feature, identity)
    if violations:
        via = "" if origin == artifact_reference else f" (via {origin!r})"
        raise QualificationError(
            f"qualification {qualification_path!r} cannot be cited for "
            f"{artifact_reference!r}{via} — " + "; ".join(violations))
    if not record_obj.decision.accepted:
        raise QualificationError(
            f"qualification {qualification_path!r} records decision "
            f"'{record_obj.decision.decision}' ({record_obj.decision.date}): "
            f"{record_obj.decision.rationale} — a promotion cannot cite a "
            "rejected feature as its qualification evidence")
    return {"path": relative,
            "contentHash": digest,
            "decision": record_obj.decision.decision}


def roster_warnings(manifest) -> list[str]:
    """Consistency warnings (never errors) over an SAE candidate roster.

    A candidate whose ``status`` records a qualification OUTCOME but carries
    no ``qualificationArtifact`` pointer has a verdict with nothing behind it:
    the roster says "qualified" and the study cannot show why. It stays a
    warning because the roster is authored by hand, iteratively, and blocking
    on a pointer that is about to be filled in would be an obstacle rather
    than a check. Takes a ``sae_candidates.CandidateManifest``; kept here so
    the roster module stays exactly as it was."""
    warnings: list[str] = []
    for candidate in getattr(manifest, "candidates", ()):
        if candidate.status not in ("qualified", "rejected"):
            continue
        if candidate.qualification_artifact:
            continue
        warnings.append(
            f"candidate '{candidate.construct_label}' (feature "
            f"{candidate.feature_id}, layer {candidate.layer}) has status "
            f"'{candidate.status}' but names no qualificationArtifact — the "
            f"roster records the outcome with no citable evidence behind it; "
            f"record one ({FILENAME}) and point at it")
    return warnings
