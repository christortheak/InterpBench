"""SAE feature candidates as pinned study DATA (SAE-VECTOR-INTERVENTION
proposal r2, §8 P1-6).

An SAE intervention starts on a discovery surface (Neuronpedia) and ends in a
steering condition. Between those sits a decision the study must be able to
re-read years later: *which features were nominated, in what role, on what
evidence, and what happened to each one*. This module is that record — a
plain JSON file in the study WORKSPACE, validated here, hashed by BYTES, and
pinned into the experiment manifest (``saeCandidates`` = ``{path, hash}``) so
that any edit after freeze is a ``verify()`` violation exactly like markers
drift.

Why it is data and not code (CLAUDE.md's concept-agnostic rule): construct
labels are researcher text, roles are a closed vocabulary of *study
functions* (focal vs. control), and nothing here knows about judges, legal
constructs, or any particular concept. A different program with different
constructs uses the same file shape unchanged.

Three rules carry the scientific weight:

- **Neuronpedia is never a runtime dependency for evidence.** The auto-interp
  explanation, top logits, and example activations are captured ONCE at
  discovery time into the entry's ``discovery`` snapshot, with an
  ``accessDate``. Auto-interp labels get regenerated upstream; the snapshot
  is what the paper can cite. The snapshot is REQUIRED for focal roles (a
  focal feature with no recorded discovery evidence is not a candidate, it is
  a number) and optional for controls, whose construct probes are
  deliberately trivial.
- **Discovery is not verification is not qualification.** ``verification``
  records only that a human re-opened the feature on Neuronpedia (with the
  date); ``qualificationArtifact`` points at the durable
  ``sae-feature-qualification.json`` when one exists; ``status`` is the
  candidate's lifecycle position. None of the three is inferred from another.
- **(model, source, layer, featureId) is the identity**, plus the optional
  exact ``gemmaScope: {release, saeID}`` dictionary when an entry declares
  one. A feature exists only in its own layer's dictionary — there is no
  "same feature at another layer" — so that tuple must be unique in a
  manifest. Two entries claiming one feature under different construct labels
  is a nomination collision the researcher must resolve before anything is
  imported. Declaring ``gemmaScope`` is what lets the seating guard tell
  feature 40802 of the 65k dictionary from feature 40802 of the 262k
  dictionary at one layer; without it the guard stays dictionary-blind (see
  the seating section at the foot of this module).

Nothing here imports SAE weights, touches HuggingFace, or reaches the
network: this is the paper trail, and it validates offline.

Cross-engine note: the manifest key ``saeCandidates`` is ADDITIVE and
OPTIONAL. Manifests without it verify and freeze exactly as before, and the
manifest content hash is computed from the raw dict, so an absent key leaves
every existing hash byte-identical. There is no Swift twin today (SAE
imports are server/Gemma work); a Swift twin would need to tolerate — and
ideally re-verify — the key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

#: Study FUNCTION of a candidate, not its semantics (proposal r2 §4).
#: ``focal`` features carry a hypothesis; every other role exists to make a
#: focal result falsifiable.
ROLES = (
    "focal",
    "affectControl",
    "embodiedControl",
    "domainControl",
    "discriminantControl",
    "unrelatedTopicControl",
    "positiveControl",
)

#: Roles whose entries MUST carry a discovery snapshot.
SNAPSHOT_REQUIRED_ROLES = ("focal",)

#: Lifecycle position. ``seated`` means a promoted agent built on this
#: feature entered a study — it is a statement about the variant library, not
#: about the feature's semantics.
STATUSES = ("candidate", "qualified", "rejected", "seated")

#: Verification = a human re-opened the feature on the discovery surface.
VERIFICATION_STATUSES = ("unverified", "verifiedOnNeuronpedia")

_TOP_LEVEL_KEYS = ("schemaVersion", "name", "description", "candidates",
                   "pendingConstructs")
_ENTRY_KEYS = ("constructLabel", "role", "model", "source", "layer",
               "featureId", "neuronpediaUrl", "gemmaScope", "discovery",
               "verification", "qualificationArtifact", "status", "notes")
_GEMMA_SCOPE_KEYS = ("release", "saeID")
_DISCOVERY_KEYS = ("explanationText", "topPositiveLogits", "topNegativeLogits",
                   "exampleActivations", "accessDate")
_VERIFICATION_KEYS = ("status", "date")
_PENDING_KEYS = ("constructLabel", "role", "notes")

SCHEMA_VERSION = 1

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Where the shape is documented (named in every refusal).
TEMPLATE_PATH = "prompts/templates/sae-candidates/sae-candidates-template.json"


class CandidateManifestError(Exception):
    """A candidate manifest that does not validate must never be pinned: the
    pin promises the file is re-readable evidence, and a SHA-256 over
    malformed bytes certifies only that they did not change."""


@dataclass(frozen=True)
class Discovery:
    """Discovery-time snapshot of the upstream surface. Free text and lists —
    this layer checks SHAPE, never semantics."""

    explanation_text: str
    access_date: str
    top_positive_logits: tuple[str, ...] = ()
    top_negative_logits: tuple[str, ...] = ()
    example_activations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GemmaScopeDictionary:
    """The feature's dictionary in GEMMA SCOPE's own vocabulary — the exact
    pair an imported artifact records in ``gemmascopeSource``.

    Optional, because the roster's discovery-surface ``source`` string
    ("gemmascope-2-res-65k") is what a nominator reads off Neuronpedia. When
    it IS declared, the seating guard can tell feature 40802 of the 65k
    dictionary from feature 40802 of the 262k dictionary at the same layer —
    which ``(model, layer, featureId)`` alone cannot.
    """

    release: str
    sae_id: str

    def describe(self) -> str:
        return f"{self.release}/{self.sae_id}"


@dataclass(frozen=True)
class Candidate:
    construct_label: str
    role: str
    model: str
    source: str
    layer: int
    feature_id: int
    neuronpedia_url: str
    status: str
    verification_status: str
    verification_date: str | None = None
    qualification_artifact: str | None = None
    discovery: Discovery | None = None
    notes: str = ""
    #: Optional exact dictionary id (``{"release": …, "saeID": …}``).
    gemma_scope: GemmaScopeDictionary | None = None

    @property
    def identity(self) -> tuple:
        """The tuple that must be unique — a feature exists only in its own
        layer's dictionary.

        The Gemma Scope pair joins it when declared: two entries that differ
        ONLY in dictionary (65k vs 262k under one discovery-surface ``source``
        string) are two different features, and refusing them as duplicates
        would be wrong. Entries without the block keep exactly the old
        four-part identity, so no existing roster changes status.
        """
        dictionary = (None if self.gemma_scope is None else
                      (self.gemma_scope.release, self.gemma_scope.sae_id))
        return (self.model, self.source, self.layer, self.feature_id,
                dictionary)


@dataclass(frozen=True)
class PendingConstruct:
    """A declared control SLOT with no feature nominated yet (proposal r2 §4:
    fear / hunger / unrelated topic). Recorded so an unfilled control is
    visible in the study record instead of being silently absent — it carries
    no feature identity and never collides with a candidate."""

    construct_label: str
    role: str
    notes: str = ""


@dataclass(frozen=True)
class CandidateManifest:
    schema_version: int
    candidates: tuple[Candidate, ...]
    pending: tuple[PendingConstruct, ...] = ()
    name: str = ""
    description: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    # --- loading -----------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes) -> "CandidateManifest":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CandidateManifestError(
                f"the SAE candidate manifest is not UTF-8 text: {exc}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CandidateManifestError(
                f"the SAE candidate manifest is not valid JSON: {exc}")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: object) -> "CandidateManifest":
        if not isinstance(payload, dict):
            raise CandidateManifestError(
                "the SAE candidate manifest must be a JSON object with a "
                f"'candidates' array (see {TEMPLATE_PATH})")
        _reject_unknown(payload, _TOP_LEVEL_KEYS, "the candidate manifest")
        version = payload.get("schemaVersion", SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool):
            raise CandidateManifestError(
                "schemaVersion must be an integer")
        if version != SCHEMA_VERSION:
            raise CandidateManifestError(
                f"unsupported schemaVersion {version} — this engine reads "
                f"schemaVersion {SCHEMA_VERSION}")
        rows = payload.get("candidates")
        if not isinstance(rows, list):
            raise CandidateManifestError(
                "'candidates' must be an array (an empty array is legal — a "
                "manifest with no nominations yet)")
        candidates = tuple(_candidate(row, index)
                           for index, row in enumerate(rows))
        seen: dict[tuple, int] = {}
        for index, candidate in enumerate(candidates):
            first = seen.get(candidate.identity)
            if first is not None:
                dictionary = ("" if candidate.gemma_scope is None else
                              f" / {candidate.gemma_scope.describe()}")
                raise CandidateManifestError(
                    f"candidates[{index}] duplicates candidates[{first}]: "
                    f"{candidate.model} / {candidate.source} / layer "
                    f"{candidate.layer} / feature {candidate.feature_id}"
                    f"{dictionary} is already nominated — one feature is one "
                    "entry (a feature exists only in its own layer's "
                    "dictionary)")
            seen[candidate.identity] = index
        pending_rows = payload.get("pendingConstructs") or []
        if not isinstance(pending_rows, list):
            raise CandidateManifestError("'pendingConstructs' must be an array")
        pending = tuple(_pending(row, index)
                        for index, row in enumerate(pending_rows))
        return cls(schema_version=version, candidates=candidates,
                   pending=pending,
                   name=str(payload.get("name") or ""),
                   description=str(payload.get("description") or ""),
                   raw=payload)

    # --- reporting ---------------------------------------------------------

    def summary(self) -> dict:
        """Counts the CLI prints and the readiness layer reads. Plain data:
        no verdicts, because 'enough controls' is a study-design judgement,
        not a schema property."""
        by_role: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_verification: dict[str, int] = {}
        for candidate in self.candidates:
            by_role[candidate.role] = by_role.get(candidate.role, 0) + 1
            by_status[candidate.status] = by_status.get(candidate.status, 0) + 1
            by_verification[candidate.verification_status] = \
                by_verification.get(candidate.verification_status, 0) + 1
        return {
            "name": self.name,
            "schemaVersion": self.schema_version,
            "count": len(self.candidates),
            "byRole": by_role,
            "byStatus": by_status,
            "byVerification": by_verification,
            "withDiscoverySnapshot": sum(
                1 for c in self.candidates if c.discovery is not None),
            "withQualification": sum(
                1 for c in self.candidates if c.qualification_artifact),
            # How many nominations carry the exact Gemma Scope dictionary —
            # the ones the seating guard can check dictionary-precisely.
            "withGemmaScopeDictionary": sum(
                1 for c in self.candidates if c.gemma_scope is not None),
            "pendingConstructs": [
                {"constructLabel": p.construct_label, "role": p.role}
                for p in self.pending],
        }


# --- entry parsing ---------------------------------------------------------

def _reject_unknown(payload: dict, allowed: tuple[str, ...], label: str) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise CandidateManifestError(
            f"{label} has unknown key(s) {', '.join(unknown)} — the schema is "
            f"CLOSED so a typo'd key can never be silently ignored (allowed: "
            f"{', '.join(allowed)})")


def _text(payload: dict, key: str, label: str, *, required: bool = True) -> str:
    value = payload.get(key)
    if value is None or value == "":
        if required:
            raise CandidateManifestError(f"{label}: '{key}' is required")
        return ""
    if not isinstance(value, str):
        raise CandidateManifestError(f"{label}: '{key}' must be a string")
    return value


def _index(payload: dict, key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateManifestError(
            f"{label}: '{key}' must be an integer (feature and layer indices "
            "are dictionary coordinates, never labels)")
    if value < 0:
        raise CandidateManifestError(f"{label}: '{key}' must be >= 0")
    return value


def _enum(payload: dict, key: str, allowed: tuple[str, ...], label: str) -> str:
    value = _text(payload, key, label)
    if value not in allowed:
        raise CandidateManifestError(
            f"{label}: '{key}' must be one of {', '.join(allowed)} "
            f"(got '{value}')")
    return value


def _date(value: str, label: str, key: str) -> str:
    if not _DATE_RE.match(value):
        raise CandidateManifestError(
            f"{label}: '{key}' must be an ISO date (YYYY-MM-DD), got '{value}'")
    return value


def _string_list(payload: dict, key: str, label: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise CandidateManifestError(
            f"{label}: '{key}' must be an array of strings")
    return tuple(value)


def _discovery(payload: object, label: str) -> Discovery:
    if not isinstance(payload, dict):
        raise CandidateManifestError(f"{label}: 'discovery' must be an object")
    _reject_unknown(payload, _DISCOVERY_KEYS, f"{label} discovery")
    explanation = _text(payload, "explanationText", f"{label} discovery")
    access = _date(_text(payload, "accessDate", f"{label} discovery"),
                   f"{label} discovery", "accessDate")
    return Discovery(
        explanation_text=explanation,
        access_date=access,
        top_positive_logits=_string_list(payload, "topPositiveLogits",
                                         f"{label} discovery"),
        top_negative_logits=_string_list(payload, "topNegativeLogits",
                                         f"{label} discovery"),
        example_activations=_string_list(payload, "exampleActivations",
                                         f"{label} discovery"))


def _verification(payload: object, label: str) -> tuple[str, str | None]:
    if payload is None:
        # Absent = never checked. Recorded explicitly rather than guessed.
        return ("unverified", None)
    if not isinstance(payload, dict):
        raise CandidateManifestError(
            f"{label}: 'verification' must be an object "
            '{"status": …, "date": …}')
    _reject_unknown(payload, _VERIFICATION_KEYS, f"{label} verification")
    status = _enum(payload, "status", VERIFICATION_STATUSES,
                   f"{label} verification")
    raw_date = payload.get("date")
    if raw_date in (None, ""):
        if status != "unverified":
            raise CandidateManifestError(
                f"{label} verification: 'date' is required when status is "
                f"'{status}' — a verification with no date cannot be audited")
        return (status, None)
    if not isinstance(raw_date, str):
        raise CandidateManifestError(
            f"{label} verification: 'date' must be a string")
    return (status, _date(raw_date, f"{label} verification", "date"))


def _gemma_scope(payload: object, label: str) -> GemmaScopeDictionary:
    """The optional exact-dictionary block. BOTH fields are required once the
    block is present: a release without an saeID (or the reverse) names half a
    dictionary, which is no more identifying than omitting it — and it would
    make the seating guard's match silently partial."""
    if not isinstance(payload, dict):
        raise CandidateManifestError(
            f"{label}: 'gemmaScope' must be an object "
            '{"release": …, "saeID": …}')
    _reject_unknown(payload, _GEMMA_SCOPE_KEYS, f"{label} gemmaScope")
    return GemmaScopeDictionary(
        release=_text(payload, "release", f"{label} gemmaScope"),
        sae_id=_text(payload, "saeID", f"{label} gemmaScope"))


def _candidate(payload: object, index: int) -> Candidate:
    label = f"candidates[{index}]"
    if not isinstance(payload, dict):
        raise CandidateManifestError(f"{label} must be an object")
    _reject_unknown(payload, _ENTRY_KEYS, label)
    role = _enum(payload, "role", ROLES, label)
    url = _text(payload, "neuronpediaUrl", label)
    if not (url.startswith("https://") or url.startswith("http://")):
        raise CandidateManifestError(
            f"{label}: 'neuronpediaUrl' must be an http(s) URL — it is the "
            "discovery provenance a reader follows")
    discovery_payload = payload.get("discovery")
    if discovery_payload is None and role in SNAPSHOT_REQUIRED_ROLES:
        raise CandidateManifestError(
            f"{label}: a '{role}' candidate must carry a 'discovery' snapshot "
            "(explanationText + accessDate at minimum) — upstream auto-interp "
            "labels are regenerated, so the snapshot is the only citable "
            "record of why this feature was nominated")
    discovery = None if discovery_payload is None \
        else _discovery(discovery_payload, label)
    qualification = payload.get("qualificationArtifact")
    if qualification is not None and not isinstance(qualification, str):
        raise CandidateManifestError(
            f"{label}: 'qualificationArtifact' must be a path string or null")
    if isinstance(qualification, str) and os.path.isabs(qualification):
        raise CandidateManifestError(
            f"{label}: 'qualificationArtifact' must be WORKSPACE-RELATIVE — an "
            "absolute path names one machine's filesystem and resolves to "
            "nothing on the cluster")
    verification_status, verification_date = _verification(
        payload.get("verification"), label)
    dictionary = payload.get("gemmaScope")
    return Candidate(
        construct_label=_text(payload, "constructLabel", label),
        role=role,
        model=_text(payload, "model", label),
        source=_text(payload, "source", label),
        layer=_index(payload, "layer", label),
        feature_id=_index(payload, "featureId", label),
        neuronpedia_url=url,
        status=_enum(payload, "status", STATUSES, label),
        verification_status=verification_status,
        verification_date=verification_date,
        qualification_artifact=qualification or None,
        discovery=discovery,
        notes=_text(payload, "notes", label, required=False),
        gemma_scope=(None if dictionary is None
                     else _gemma_scope(dictionary, label)))


def _pending(payload: object, index: int) -> PendingConstruct:
    label = f"pendingConstructs[{index}]"
    if not isinstance(payload, dict):
        raise CandidateManifestError(f"{label} must be an object")
    _reject_unknown(payload, _PENDING_KEYS, label)
    return PendingConstruct(
        construct_label=_text(payload, "constructLabel", label),
        role=_enum(payload, "role", ROLES, label),
        notes=_text(payload, "notes", label, required=False))


# --- bytes, hashing, workspace resolution ----------------------------------

def content_hash(data: bytes) -> str:
    """SHA-256 of the FILE BYTES — the same mechanical rule as every other
    measurement-side pin (markersHash, the reasoning-style taxonomy, the
    capability battery). Deliberately not a semantic hash over the parsed
    structure: reformatting the file IS a change the study should notice, and
    a byte hash needs no cross-engine canonicalization contract."""
    return hashlib.sha256(data).hexdigest()


def read_bytes(rel_path: str, root: str | None = None) -> bytes:
    """Raw bytes of a workspace-relative candidate manifest."""
    from . import paths
    with open(paths.resolve(rel_path, root), "rb") as handle:
        return handle.read()


def live_hash(rel_path: str, root: str | None = None) -> str | None:
    """Current hash of the file at ``rel_path``, or None when it is missing —
    the shape the pin checks read (parallel to ``battery.live_hash``)."""
    try:
        return content_hash(read_bytes(rel_path, root))
    except OSError:
        return None


def load(rel_path: str, root: str | None = None) -> tuple[CandidateManifest, str]:
    """Load + validate a workspace-relative candidate manifest, returning it
    with the hash of the exact bytes read. Raises
    :class:`CandidateManifestError` for a missing, unreadable, or invalid
    file — the pin verb and the CLI both refuse on it."""
    try:
        data = read_bytes(rel_path, root)
    except OSError as exc:
        raise CandidateManifestError(
            f"no SAE candidate manifest at {rel_path} — author one under "
            f"prompts/ in the workspace (start from {TEMPLATE_PATH}): {exc}")
    return (CandidateManifest.from_bytes(data), content_hash(data))


def pin_violations(block: object, root: str | None = None) -> list[str]:
    """``verify()`` for the manifest's optional ``saeCandidates`` pin.

    Contract, matching ``markersHash`` / the reasoning-style taxonomy exactly:

    - ABSENT block: no violation, nothing pinned, legacy manifests unaffected;
    - a half-pin (path without hash, or hash without path) certifies nothing
      and is a violation;
    - an ABSOLUTE path is a violation (the Mac workspace is the source of
      truth; stored refs are workspace-relative or they resolve to nothing on
      the cluster);
    - a missing file or drifted bytes after pinning is a violation, exactly
      like stimulus drift;
    - and, ONLY when the hash is clean, the file's SHAPE is checked (the
      PinShapeValidation precedent) — a pin over bytes the engine would refuse
      to read is a pin over nothing.
    """
    if block is None:
        return []
    from . import paths
    base = paths.project_root() if root is None else root
    if not isinstance(block, dict):
        return ["saeCandidates must be an object "
                '{"path": …, "hash": …} naming the workspace-relative SAE '
                "candidate manifest"]
    unknown = sorted(set(block) - {"path", "hash"})
    if unknown:
        return [f"saeCandidates has unknown key(s) {', '.join(unknown)} — the "
                "pin block is path + hash only"]
    rel = block.get("path")
    digest = block.get("hash")
    if not rel or not digest:
        return ["SAE candidate manifest pin is incomplete — saeCandidates.path "
                "and saeCandidates.hash must both be set"]
    if not isinstance(rel, str) or not isinstance(digest, str):
        return ["saeCandidates.path and saeCandidates.hash must both be strings"]
    if os.path.isabs(rel):
        return [f"SAE candidate manifest path '{rel}' is absolute — pinned "
                "inputs are WORKSPACE-RELATIVE so the study resolves on any "
                "machine (the authoring client's workspace is the source of "
                "truth)"]
    path = os.path.join(base, rel)
    if not os.path.exists(path):
        return [f"SAE candidate manifest '{rel}': file missing at {rel} — the "
                "manifest pins it, so the bytes must be there (restore the "
                "file, or drop the saeCandidates block on a duplicate draft)"]
    with open(path, "rb") as handle:
        live = content_hash(handle.read())
    if live != digest:
        return [f"SAE candidate manifest '{rel}' changed since pinning "
                f"(have {live[:12]}…, pinned {digest[:12]}…)"]
    try:
        CandidateManifest.from_bytes(read_bytes(rel, root))
    except CandidateManifestError as exc:
        return [f"SAE candidate manifest '{rel}' is not valid: {exc}"]
    return []


# --- preregistration surface for seated features ---------------------------
#
# The roster's identity is (model, source, layer, featureId), where ``source``
# is the DISCOVERY surface's dictionary name ("gemmascope-2-res-65k"), while
# an imported artifact — and a declared latent condition — record the
# dictionary in GEMMA SCOPE's vocabulary (``release`` =
# "gemma-scope-2-27b-it-res", ``saeID`` = "layer_40_width_65k_l0_medium").
# The two strings are not mechanically comparable and no mapping between them
# is authored anywhere, so the guard used to match on the three fields that
# ARE comparable — model, layer, featureId — which cannot tell feature 40802
# of the 65k dictionary from feature 40802 of the 262k dictionary at one
# layer.
#
# The fix is DATA, not a string mapping (which would go silently wrong at the
# first dictionary whose name did not follow today's convention): a nomination
# may declare its exact ``gemmaScope: {release, saeID}``, and when it does the
# guard matches THAT against what the study seats. The three-field fallback
# survives for entries without the block, so every roster written before this
# behaves exactly as it did — and its dictionary-blindness stays a documented
# human check.


@dataclass(frozen=True)
class SeatedFeature:
    """What a study actually seats, in coordinates a nomination can be
    checked against: identity from the artifact's pinned sidecar (or a latent
    condition's own declaration), never from a manifest restatement."""

    model: str
    feature_id: int
    #: ``None`` only when a latent condition declares no layer and its saeID
    #: does not follow the ``layer_<n>_…`` grammar.
    layer: int | None = None
    #: Gemma Scope's own dictionary id, when the seat records one.
    release: str = ""
    sae_id: str = ""

    @property
    def dictionary(self) -> str:
        return f"{self.release}/{self.sae_id}" if self.release and self.sae_id else ""

    def describe(self) -> str:
        layer = "layer ?" if self.layer is None else f"layer {self.layer}"
        detail = f" in {self.dictionary}" if self.dictionary else ""
        return f"{self.model} {layer} feature {self.feature_id}{detail}"


def nominations(block: object,
                root: str | None = None) -> tuple[Candidate, ...] | None:
    """The pinned roster's nominations, or ``None`` when no usable roster is
    pinned (absent block, or one whose own pin violations
    :func:`pin_violations` already reports — this never double-reports, and
    never turns an unreadable roster into a seating refusal)."""
    if not isinstance(block, dict):
        return None
    rel = block.get("path")
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    try:
        manifest, _digest = load(rel, root)
    except (CandidateManifestError, OSError):
        return None
    return manifest.candidates


def seating_refusal(nominated: tuple[Candidate, ...],
                    seated: SeatedFeature) -> str | None:
    """Why ``seated`` is not preregistered, or ``None`` when it is.

    The rule, in one place because both the vector-condition guard and the
    latent-condition guard must apply exactly the same one:

    - a nomination with a ``gemmaScope`` block matches only when the seated
      feature declares the SAME release/saeID (and the same layer, when both
      say one) — the dictionary is part of the feature's identity;
    - a nomination without the block falls back to (model, layer, featureId),
      the historical match;
    - when only dictionary-declaring nominations are in play and none matches,
      the refusal names BOTH sides, because "not nominated" would send a
      researcher looking for a missing entry that is in fact right there under
      a different dictionary.
    """
    same_feature = [c for c in nominated
                    if c.model == seated.model
                    and c.feature_id == seated.feature_id]
    dictionary_entries = [c for c in same_feature if c.gemma_scope is not None]
    for candidate in same_feature:
        if candidate.gemma_scope is None:
            if seated.layer is not None and candidate.layer == seated.layer:
                return None
            continue
        if (candidate.gemma_scope.release == seated.release
                and candidate.gemma_scope.sae_id == seated.sae_id
                and seated.release and seated.sae_id
                and (seated.layer is None or candidate.layer == seated.layer)):
            return None

    if dictionary_entries:
        nominated_dictionaries = ", ".join(sorted(
            {c.gemma_scope.describe() for c in dictionary_entries}))
        if not seated.dictionary:
            return (f"which the pinned SAE candidate roster nominates only in "
                    f"{nominated_dictionaries}, while what this study seats "
                    "records no release/saeID to match against — import the "
                    "feature BY ID (gemmascope import-id), which stamps the "
                    "dictionary, so the seat and the nomination are comparable")
        return (f"which the pinned SAE candidate roster nominates in "
                f"{nominated_dictionaries}, not the {seated.dictionary} this "
                "study seats — same-numbered features in different Gemma "
                "Scope dictionaries are DIFFERENT features. Nominate the "
                "dictionary being seated (with its own discovery snapshot) "
                "and re-pin the roster, or seat the nominated one")
    return ("which the pinned SAE candidate roster does not nominate — the "
            "roster IS the preregistration of which features this study may "
            "seat. Nominate the feature (with its discovery snapshot) and "
            "re-pin the roster before freezing, or seat a nominated feature")
