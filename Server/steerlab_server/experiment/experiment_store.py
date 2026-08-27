"""Experiment authoring — create / attach / duplicate / edit / freeze (parallel
to Swift ``ExperimentStore`` write methods).

Operates on the raw manifest dict so fields this server doesn't model are
preserved round-trip. Writes Swift-compatible shapes (e.g. the
``readingPosition`` Codable enum form) so a manifest authored here is readable by
the Mac app, and vice-versa.

**Freeze authority:** the server stamps ``status: frozen``, a content hash, the
git commit, and ``frozenBy: "server"``. The content hash is the Python one used
for run-stamping — NOT byte-identical to Swift's ``freezeHash`` — but cross-engine
verification rests on the stimulus/corpus/file **SHA-256s**, which ARE identical,
so either engine can verify the other's frozen experiment. Frozen manifests are
read-only here (iterate by ``duplicate``), exactly like the Swift lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from . import lifecycle_gates, paths
from ..build_identity import engine_version
from .manifest import KNOWN_ORDINAL_AGGREGATIONS as KNOWN_ORDINAL_AGGREGATIONS  # noqa: F401
from .manifest import Manifest
from .manifest import variant_is_evidence_grade

# The closed outcome-instrument vocabulary (Swift
# ``ExperimentStore.knownOutcomeInstruments`` twin — the cross-engine lists
# are pinned identical by tests on both engines). The explicit re-export of
# ``KNOWN_ORDINAL_AGGREGATIONS`` above keeps both authoring vocabularies
# addressable from one module.
KNOWN_OUTCOME_INSTRUMENTS = (
    "sampledText", "answerTokenLogprob", "choiceProbability",
    "repeReaderScore", "ordinalScale")
from ..steering.stimulus_set import StimulusSet, load_texts
from ..steering.vector_math import ExtractionMethod
from ..steering.vector_store import SUBSTRATE as _THIS_SUBSTRATE

#: The sidecar ``extractionMethod`` an OptVec training run stamps (mirrors
#: ``experiment.optvec_train.EXTRACTION_METHOD``; duplicated as a plain string
#: so manifest authoring never imports the torch-bearing training module).
_OPTVEC_METHOD = "optvec"
#: The SOURCE method the Gemma Scope importer stamps (mirrors
#: ``experiment.gemma_scope``; a plain string for the same reason).
_GEMMA_SCOPE_METHOD = "gemmaScopeSAE"


def unknown_outcome_instruments(d: dict) -> list[str]:
    """The declared instruments no engine implements, in declaration order.

    The constant above had ZERO production readers until 2026-08-18: Swift
    enforced the vocabulary at DECLARATION (``set-instruments`` refuses at
    64), which protects only manifests authored through that verb. Authoring
    is Mac-authority, so every manifest arriving HERE arrives as bytes — a
    bundle, an rsync, a hand edit — and every downstream reader is a SET
    MEMBERSHIP test (:data:`tasks.CHOICE_INSTRUMENTS`,
    ``"ordinalScale" in ...``, ``execution_plan.resolve``). An unrecognised
    value therefore dispatches nothing, raises nothing, and the study
    completes having measured only the default sampled text. ``sampledTxt``
    for ``sampledText`` is the whole failure.

    Swift twin: ``ExperimentStore.unknownOutcomeInstruments``.
    """
    return [str(i) for i in (d.get("outcomeInstruments") or [])
            if str(i) not in KNOWN_OUTCOME_INSTRUMENTS]


def unknown_outcome_instrument_problem(d: dict) -> str | None:
    """The plain-language problem for a run-start refusal, or None.

    One rule, both engines (Swift twin:
    ``ExperimentStore.unknownOutcomeInstrumentProblem``) — the sentence is the
    cross-engine contract because the claim is the same claim."""
    unknown = unknown_outcome_instruments(d)
    if not unknown:
        return None
    named = ", ".join(f"'{i}'" for i in unknown)
    return (f"outcomeInstruments declares {named}, which this engine does not "
            "implement — the declared instruments are read by set membership, "
            "so an unrecognised value dispatches nothing and the study would "
            "complete having measured only the default sampled text. Known "
            "instruments: " + ", ".join(KNOWN_OUTCOME_INSTRUMENTS))


def unknown_outcome_instrument_repair(name: str) -> str:
    """THE repair, on both engines: ``set-instruments`` is authoring, and
    authoring is Mac-authority (audit §10.x), so this engine's copy of the
    refusal names the Mac binary too — exactly like the no-rubric sentence."""
    return (f"steerlab-cli experiment set-instruments {name} <"
            + "|".join(KNOWN_OUTCOME_INSTRUMENTS) + ">[,…]")


class ExperimentStoreError(Exception):
    """Authoring/lifecycle refusal.

    Carries an optional GATE id (WP0 step 3). The freeze path computed a
    closed-vocabulary id for every gate and then dropped it on the refusal
    path — ``raise ExperimentStoreError(gate_failures[0][1])`` — so only the
    ``forcedGatesSkipped`` stamp ever named a gate, and only the FIRST of N
    failures survived a refusal at all
    (``docs/WP0-AGENT-SURFACE-AUDIT.md`` §2.4). ``gate`` is the gate the
    message describes; ``gates`` is every gate that failed, in
    :data:`FORCED_GATE_IDS` order — the same order the stamp uses.

    Strictly additive: this class is caught broadly (CLI, HTTP routes,
    tasks), and ``str(exc)`` renders exactly the message it always did, so
    no refusal's human-visible prose or exit code changes. Swift twin:
    ``ExperimentError.freezeRefusal`` / ``FreezeRefusal``.
    """

    def __init__(self, message: str, *, gate: str | None = None,
                 gates: tuple[str, ...] | list[str] | None = None,
                 repair: str = ""):
        super().__init__(message)
        self.gate = gate
        #: The runnable repair (WP0 step 8). Read by the CLI's envelope
        #: builder through ``lifecycle_gates.repair_of``, so a refusal that
        #: knows its own remedy stops handing an agent boilerplate.
        self.repair_action = repair
        #: Every failing gate in vocabulary order; always contains ``gate``.
        self.gates: tuple[str, ...] = tuple(gates) if gates is not None else (
            (gate,) if gate else ())


def _dir(name: str, root: str | None) -> str:
    return os.path.join(paths.experiments_directory(root), name)


def _path(name: str, root: str | None) -> str:
    nested = os.path.join(_dir(name, root), "experiment.json")
    flat = os.path.join(paths.experiments_directory(root), f"{name}.json")
    return nested if os.path.exists(nested) else (flat if os.path.exists(flat) else nested)


def load_raw(name: str, root: str | None = None) -> dict:
    with open(_path(name, root), encoding="utf-8") as handle:
        return json.load(handle)


#: The manifest keys that carry the study's measured surface. Named once so
#: the guard below and its Swift twin (`ExperimentStore.armBearingKeys`) can be
#: read against each other. `variantConditions` is deliberately OUT: the guard
#: is the narrow non-empty → BOTH-empty transition on `concepts`+`conditions`
#: that open-issues §8 describes, and widening it would refuse a legitimate
#: "clear the variant arms" edit that has never gone wrong.
ARM_BEARING_KEYS: tuple[str, ...] = ("concepts", "conditions")


def _clears_every_arm(existing: object, incoming: dict) -> bool:
    """True when this save would take a manifest that HOLDS a measured surface
    to one that holds none at all.

    Not "the document is empty" — a manifest legitimately starts that way and
    stays that way until the first attach. The refusable event is the
    TRANSITION: something on disk had concepts and/or conditions, and what is
    about to replace it has neither."""
    if not isinstance(existing, dict):
        return False
    had = any(existing.get(key) for key in ARM_BEARING_KEYS)
    # The INCOMING side also counts variantConditions: an agentComparison-
    # style save whose whole surface lives in variant conditions is not a
    # disarm — that study type's arms LIVE there. (The guard's first false
    # positive, test_transcript_study, caught at landing 2026-08-20.) Swift
    # twin: `ExperimentStore.holdsAnySurface`.
    clears = not any(
        incoming.get(key) for key in (*ARM_BEARING_KEYS, "variantConditions"))
    return had and clears


def save_raw(d: dict, root: str | None = None, *, freeze_transition: bool = False,
             clearing_arms: bool = False) -> None:
    """Persist a manifest.

    ``clearing_arms`` is the caller DECLARING that dropping every concept and
    condition is the point of this write (open-issues §8). Without it, such a
    save is refused: a document that arrives at both-empty over a populated
    draft is far more often a stale in-memory copy, a skeleton push, or an old
    bundle than an intentional reset, and the loss is silent — the run
    directory's snapshot is what preserved the arms the last time this
    happened, and only by luck.
    """
    name = d["name"]
    path = _path(name, root)
    if os.path.exists(path) and not freeze_transition:
        existing = load_raw(name, root)
        if (not clearing_arms and existing.get("status") == "draft"
                and _clears_every_arm(existing, d)):
            # Frozen/complete manifests never reach here — the status check
            # below refuses them outright — so this rule is DRAFT-only by
            # construction, not by an extra condition that could drift.
            raise ExperimentStoreError(
                f"refusing to save '{name}' with no concepts and no "
                f"conditions over a draft that has "
                f"{len(existing.get('concepts') or [])} concept(s) and "
                f"{len(existing.get('conditions') or [])} condition(s) — a "
                "manifest does not lose its whole measured surface in one "
                "write by accident",
                gate=lifecycle_gates.ARMS_CLEARED,
                repair=(
                    f"steerlab-cli experiment verify {name}  "
                    "# the manifest on disk still holds its arms; re-attach "
                    "what the caller dropped (steerlab-cli experiment attach "
                    f"{name} <concept>… ; steerlab-cli experiment "
                    f"declare-condition {name} …), or author the cleared "
                    "study as its own draft with steerlab-cli experiment "
                    f"create {name}-v2 --model <id>"))
        if existing.get("status") == "frozen":
            # WP0 step 8: typed `statusImmutable`. `gate` here names a
            # LIFECYCLE gate, not a freeze gate — the two vocabularies are
            # disjoint by test, so the CLI's classifier reads it correctly and
            # an agent's `switch` over freeze gates cannot absorb it.
            raise ExperimentStoreError(
                f"'{name}' is frozen and read-only — duplicate it to iterate",
                gate=lifecycle_gates.STATUS_IMMUTABLE,
                repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 "
                        "&& re-apply the change to the duplicate  "
                        "(authoring is Mac-authority)"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ATOMIC (engineer review 2026-07-18): a kill mid-write must never leave
    # experiment.json as invalid JSON — everything downstream (including the
    # judgment-projection recovery path) starts with Manifest.load, so a
    # torn manifest is unrecoverable without hand surgery.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                               prefix=".experiment-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(d, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create(name: str, *, model_id: str, revision: str | None = None,
           description: str = "", root: str | None = None) -> dict:
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name.lower()).strip("-")
    if not safe:
        raise ExperimentStoreError(f"invalid experiment name {name!r}")
    if os.path.exists(_path(safe, root)):
        raise ExperimentStoreError(f"experiment '{safe}' already exists")
    manifest = {
        "name": safe, "status": "draft", "experimentDescription": description,
        "modelID": model_id, "modelRevision": revision, "createdAt": _now(),
        "studyKind": "modelOutput", "concepts": [], "conditions": [],
        "variantConditions": [], "promptMode": "chatAssistant", "temperature": 0.0,
        "maxTokens": 512, "seeds": [0], "qwenThinkingEnabled": False,
    }
    save_raw(manifest, root)
    return manifest


def _reading_position_codable(method_opts: dict, position=None) -> dict:
    """The manifest's ``readingPosition`` block — Swift's synthesized Codable
    enum form, which is what every consumer already reads.

    A DECLARED position (2026-08-25) encodes generically from its identity
    mode and parameter, so ``--reading-position 'mean from token 50'`` and
    ``--pool-from 50`` write the identical bytes: one recipe, one encoding.
    With none declared the legacy pool-only branch runs unchanged, which is
    what keeps every existing manifest byte-identical.
    """
    if position is not None:
        parameter = position.identity_parameter
        return {position.identity_mode:
                ({} if parameter is None else {"_0": parameter})}
    pool = method_opts.get("poolFromToken")
    if pool not in (None, "", 0, "0"):
        return {"meanFromToken": {"_0": int(pool)}}
    return {"lastToken": {}}


def _options_block(method: str, pool_from_token, rendering_block: dict | None,
                   position=None) -> dict:
    """One concept pin's ``options``. The rendering key is written ONLY when a
    chat-template rendering was declared, so an undeclared (or explicitly raw)
    attach produces the exact bytes it always did. A declared reading position
    wins over the method's pool default, which is the point of declaring it."""
    options = {"method": method,
               "readingPosition": _reading_position_codable(
                   {"poolFromToken": pool_from_token}, position)}
    if rendering_block is not None:
        options["extractionRendering"] = rendering_block
    return options


def _declared_reading_position(declaration, pool_from_token,
                               rendering_block: dict | None, method: str):
    """The reading position an attach pins, or ``None`` for the default.

    Every refusal fires HERE, before a manifest is loaded or written, and in
    THIS order — most specific first, so the message names the thing that is
    actually wrong:

    - the two spellings together (:func:`reading_position.declaration_conflict`);
    - a label this engine does not know (typed, naming the vocabulary);
    - any declaration at all on a PINNED ARTIFACT, which carries the position
      it was read at in its own sidecar (checked before the rendering rule
      below, because a pin has no rendering and would otherwise collect the
      wrong refusal);
    - a TEMPLATE-AWARE role under a raw rendering — the declaration-time half
      of a refusal that used to arrive only at extraction. ``rendering_block``
      is ``None`` for both an absent and an explicitly-raw declaration, which
      is exactly the condition those roles cannot resolve under, so the pin is
      answered while the person is still typing.
    """
    from ..steering import reading_position as rp

    conflict = rp.declaration_conflict(declaration, pool_from_token)
    if conflict:
        raise ExperimentStoreError(conflict)
    position = rp.parse_declaration(declaration)
    if position is None:
        return None
    if method == "pinnedArtifact":
        raise ExperimentStoreError(
            "a pinned artifact carries the reading position it was EXTRACTED "
            "at in its own sidecar — declaring "
            f"{rp.DECLARATION_FLAG} on the pin would claim a reading the "
            "bytes do not have. Repair: drop "
            f"{rp.DECLARATION_FLAG}, or attach the concept as a recipe "
            "(--method meanDifference|emotionGrandMean|…) if you mean to "
            "re-derive it at that position")
    if rendering_block is None:
        refusal = rp.templated_rendering_refusal(position)
        if refusal:
            raise ExperimentStoreError(refusal)
    return position


def _extraction_rendering_block(declaration) -> dict | None:
    """The ``extractionRendering`` block an attach writes, or ``None`` to omit
    the key entirely.

    ``None`` for both an absent declaration and an explicit ``{"mode":
    "raw"}``: raw is the legacy rendering, so a manifest that declares it must
    be byte-identical to one that says nothing — which is what keeps every
    pre-option recipe's identity hash, validation scope, and freeze hash
    exactly where they were. A chat-template declaration writes its resolved
    defaults explicitly (``to_dict``), so the manifest says what extraction
    will do without a reader knowing this module's defaults.
    """
    from ..steering.extraction_rendering import parse_declaration
    rendering = parse_declaration(declaration)
    return None if rendering is None else rendering.to_dict()


def attach(name: str, concepts: list[str], *, method: str = "meanDifference",
           pool_from_token: int | None = None, corpus_concepts: list[str] | None = None,
           reference: str | None = None,
           vector_artifact: str | None = None,
           source_concept: str | None = None,
           eval_run: str | None = None,
           extraction_rendering=None,
           reading_position: str | None = None,
           root: str | None = None) -> dict:
    """Pin concepts into a draft manifest.

    Paired methods (meanDifference/lat) pin the concept's positive/negative
    StimulusSet hash. ``emotionGrandMean`` pins the concept's stories.jsonl
    hash AND the grand-mean corpus (population membership + every member's
    hash): the vector is mean(concept) − mean(corpus), so the population is
    part of the recipe. ``corpus_concepts`` widens the population beyond the
    attached targets (targets are always members); the emotion paper's default
    reading position (mean from token 50) applies when none is given.

    ``pinnedArtifact`` pins an EXISTING vector artifact by path + both file
    hashes instead of a stimulus recipe (see :func:`attach_artifact`); it
    takes exactly one concept and ``vector_artifact``.

    ``reading_position`` (2026-08-25) pins WHERE in the stimulus the residual
    stream is read, as one of the cross-engine LABELS (``"last content
    token"``, ``"content offset 2"``, ``"mean content from token 0"``, …). It
    is the study-path writer for the whole vocabulary: before it, a manifest
    could pin only ``lastToken`` or — through ``pool_from_token`` —
    ``meanFromToken``, so every other position was reachable only from the
    ad-hoc ``/api/extract`` route, which pins nothing. Absent keeps the
    default byte-identically; ``pool_from_token`` is the legacy spelling of
    one position and the two may not both be declared.

    Measurement-side pin (2026-07-13): every attach also stamps
    ``validationHash`` — SHA-256 of the concept's held-out validation.jsonl,
    or null when it has none — into the concept's pin block, so the file the
    validate task reads live is under the same drift firewall as the stimuli.
    """
    from .manifest import concept_validation_hash
    # Parsed FIRST, before anything is read or written: a malformed or
    # unsupportable declaration must not half-attach a study.
    rendering_block = _extraction_rendering_block(extraction_rendering)
    position = _declared_reading_position(
        reading_position, pool_from_token, rendering_block, method)
    if method == "pinnedArtifact":
        if rendering_block is not None:
            raise ExperimentStoreError(
                "a pinned artifact carries the rendering it was EXTRACTED "
                "under in its own sidecar — declaring extractionRendering on "
                "the pin would claim a rendering the bytes may not have. "
                "Repair: drop --extraction-rendering, or attach the concept "
                "as a recipe (--method meanDifference|emotionGrandMean|…) if "
                "you mean to re-derive it under that rendering")
        if len(concepts) != 1:
            raise ExperimentStoreError(
                "pinnedArtifact attaches exactly one concept at a time — one "
                "artifact is one direction")
        if not (vector_artifact or "").strip():
            raise ExperimentStoreError(
                "pinnedArtifact needs the artifact path (extension-less, e.g. "
                "runs/<run>/<name>) — the artifact IS the recipe")
        return attach_artifact(name, concepts[0], vector_artifact,
                               source_concept=source_concept,
                               eval_run=eval_run, root=root)
    if method == _OPTVEC_METHOD:
        # "optvec" is a SOURCE method, never an attachable recipe: nothing
        # re-derives an optimized vector from stimuli (and the stimuli do not
        # exist). Say so here rather than letting the paired branch below
        # refuse for a missing stimulus directory the study never had.
        raise ExperimentStoreError(
            "'optvec' is not a recipe a study can re-derive — an optimized "
            "vector enters as a PINNED ARTIFACT. Train it, backfill its "
            "residual norms, then attach with method 'pinnedArtifact' "
            "(--artifact <run>/<name>)")
    if method == _GEMMA_SCOPE_METHOD:
        # Same rule, same reason: "gemmaScopeSAE" is the SOURCE method
        # stamped by the importer, never an attachable recipe. A decoder row
        # is a coordinate in a published dictionary — there are no stimuli to
        # re-derive it from, and none ever existed.
        raise ExperimentStoreError(
            "'gemmaScopeSAE' is not a recipe a study can re-derive — an "
            "imported SAE feature enters as a PINNED ARTIFACT. Import it "
            "(gemmascope import-id), then attach with method "
            "'pinnedArtifact' (--artifact <run>/<name>)")
    d = load_raw(name, root)
    refs = {c["name"]: c for c in d.get("concepts", [])}
    if method == "emotionGrandMean":
        from . import multiconcept
        pool = pool_from_token if pool_from_token is not None else 50
        members = list(dict.fromkeys((corpus_concepts or []) + concepts))
        existing = d.get("grandMeanCorpus") or {}
        members = list(dict.fromkeys((existing.get("concepts") or []) + members))
        hashes: dict[str, str] = {}
        for member in members:
            live = multiconcept.stories_hash(member, root)
            if live is None:
                raise ExperimentStoreError(
                    f"no stories.jsonl for grand-mean corpus member '{member}' "
                    "under prompts/emotions/")
            hashes[member] = live
        for concept in concepts:
            refs[concept] = {
                "name": concept, "stimulusSetHash": hashes[concept],
                "options": _options_block(method, pool, rendering_block,
                                          position),
                "validationHash": concept_validation_hash(
                    concept, paired=False, root=root),
            }
        d["grandMeanCorpus"] = {"concepts": members, "hashes": hashes}
    elif method == "designatedReference":
        # mean(concept stories) − mean(REFERENCE stories), both pooled
        # (METHODS amendment ii). The reference is part of the recipe, so it
        # pins beside the concept; the pooled reading is the method's POLICY
        # — token 50 unless deliberately overridden — because a last-token
        # read on paragraph stories extracts closing-sentence content, not
        # the concept.
        from . import multiconcept
        if not (reference or "").strip():
            raise ExperimentStoreError(
                "designatedReference needs --reference <stories-concept> — "
                "the reference corpus is part of the recipe")
        reference = reference.strip()
        ref_hash = multiconcept.stories_hash(reference, root)
        if ref_hash is None:
            raise ExperimentStoreError(
                f"no stories.jsonl for reference '{reference}' under "
                "prompts/emotions/")
        pool = pool_from_token if pool_from_token is not None else 50
        for concept in concepts:
            live = multiconcept.stories_hash(concept, root)
            if live is None:
                raise ExperimentStoreError(
                    f"no stories.jsonl for concept '{concept}' under "
                    "prompts/emotions/")
            refs[concept] = {
                "name": concept, "stimulusSetHash": live,
                "options": _options_block(method, pool, rendering_block,
                                          position),
                "designatedReference": {"name": reference, "hash": ref_hash},
                "validationHash": concept_validation_hash(
                    concept, paired=False, root=root),
            }
    else:
        for concept in concepts:
            directory = paths.concept_directory(concept, root)
            stimuli = StimulusSet.from_directory(directory)  # raises if missing/empty
            refs[concept] = {
                "name": concept, "stimulusSetHash": stimuli.hash,
                "options": _options_block(method, pool_from_token,
                                          rendering_block, position),
                "validationHash": concept_validation_hash(
                    concept, paired=True, root=root),
            }
    d["concepts"] = list(refs.values())
    # Pin the neutral corpus when present (norm-unit denominator).
    corpus_path = paths.neutral_corpus_path(root)
    if os.path.exists(corpus_path):
        d["neutralCorpusHash"] = load_texts(corpus_path).hash
    save_raw(d, root)
    return d


#: Keys an OptVec sidecar's ``optvec`` block may use to name the EVAL run that
#: read the test split. Tolerant by design: the eval verb owns the spelling,
#: and an unrecognized spelling degrades to "no eval evidence recorded" — an
#: advisory — never to a wrong citation.
_OPTVEC_EVAL_RUN_KEYS = ("evalRun", "evalRunID", "evalRunId", "evalRunDirectory")


def _recorded_optvec_eval_run(block: dict) -> str | None:
    """The eval-run reference an ``optvec`` provenance block records, if any."""
    for key in _OPTVEC_EVAL_RUN_KEYS:
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested in ("runID", "runDirectory", "path"):
                inner = value.get(nested)
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
    return None


def _verify_optvec_eval_run(reference: str, tensor_hash: str,
                            root: str | None) -> dict:
    """Resolve a named OptVec eval run and check it certifies THIS artifact.

    Until 2026-08-10 the reference was recorded verbatim and the freeze
    advisory then *described* it as test-split/capability/anchor/fluency
    evidence without ever looking — evidence trusted by name (review finding).
    The name is the entire validate-equivalent story for an optvec concept,
    so it is now resolved at attach:

    - a reference that names NO run directory refuses (a typo'd citation is
      an input error, catchable now or never);
    - a present ``eval.json`` whose ``artifact.tensorSHA256`` differs from
      the artifact being attached refuses too — that run is evidence for a
      DIFFERENT direction, and stamping it would be the exact dishonesty
      this check exists to close;
    - a run directory with no readable ``eval.json`` (a crashed eval, or one
      not yet imported in full) attaches with ``optvecEvalRunVerified:
      false`` + a reason, and the freeze advisory downgrades it to "not
      verifiable evidence" — loud, never a blocker, because the run may
      legitimately complete later.

    Returns ``{"verified": bool, "reason": str | None}``.
    """
    base = paths.project_root() if root is None else root
    candidates = []
    if os.path.isabs(reference):
        candidates.append(reference)
    else:
        candidates.append(os.path.join(base, reference))
        candidates.append(os.path.join(paths.runs_directory(root),
                                       os.path.basename(reference.rstrip("/"))))
    run_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if run_dir is None:
        raise ExperimentStoreError(
            f"OptVec eval run '{reference}' names no run directory in this "
            "workspace — the eval evidence must exist where it is cited. "
            "Import the eval run (or fix the reference), then re-attach")
    eval_path = os.path.join(run_dir, "eval.json")
    if not os.path.isfile(eval_path):
        return {"verified": False,
                "reason": "run directory exists but has no eval.json (crashed "
                          "or partially imported eval run)"}
    try:
        with open(eval_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "reason": f"eval.json is unreadable: {exc}"}
    recorded = ((payload.get("artifact") or {}).get("tensorSHA256")
                if isinstance(payload.get("artifact"), dict) else None)
    if not isinstance(recorded, str) or not recorded.strip():
        return {"verified": False,
                "reason": "eval.json records no artifact.tensorSHA256 "
                          "(pre-identity eval schema)"}
    if recorded.strip().lower() != tensor_hash.lower():
        raise ExperimentStoreError(
            f"OptVec eval run '{reference}' evaluated tensor "
            f"{recorded.strip()[:12]}…, not this artifact's "
            f"{tensor_hash[:12]}… — it is evidence for a DIFFERENT direction. "
            "Name the eval run that read THIS artifact's test split (or run "
            "the eval verb on it first)")
    return {"verified": True, "reason": None}


def reader_derived_norm_backfill_refusal(rel: str) -> str:
    """The named repair for attaching a reader-derived vector that has not been
    through the residual-norm backfill.

    Not a defect of the artifact but a missing LIFECYCLE STEP, exactly as for
    an OptVec vector: a reader measures a task template's LAT token, not a
    neutral corpus, so its reading direction is born with no denominator and
    alpha in norm units would be meaningless. Cross-engine twin literal (Swift:
    ``ExperimentStore.readerDerivedNormBackfillRefusal``).
    """
    return (f"vector artifact '{rel}' is a RepE-reader-derived direction with "
            "no residualNormSource — a reader measures a task template's LAT "
            "token, not a neutral corpus, so its reading direction is BORN "
            "without a denominator. Run the residual-norm backfill against the "
            "pinned neutral corpus first and attach the BACKFILLED artifact: "
            "α in norm units is meaningless until the denominator is measured")


def attach_artifact(name: str, concept: str, artifact_path: str, *,
                    source_concept: str | None = None,
                    eval_run: str | None = None,
                    root: str | None = None) -> dict:
    """Pin an EXISTING vector artifact into a draft manifest as a concept
    (method ``pinnedArtifact``; cross-engine contract key ``vectorArtifact``).

    Every other attach form pins a RECIPE and lets each run re-derive the
    vector. Some legitimate directions have no such recipe: they are derived
    post-hoc from other artifacts — family-grand-mean centring, for instance,
    re-references a set of vectors against their own family mean — and no
    stimulus set reproduces them. Pinning the BYTES is then the honest
    firewall: this function records the artifact's workspace-relative,
    extension-less locator plus the SHA-256 of BOTH files, and every verify,
    freeze, and materialization re-checks them against the bytes on disk.

    What is copied from the sidecar (so the manifest is self-describing and
    the artifact is never read for facts the manifest should carry): the
    reading position, the source extraction method, the source stimulus hash,
    the designated reference / grand-mean population when the source recipe
    had one, and the residual-norm provenance.

    ``source_concept`` names the concept whose stimuli and held-out
    validation.jsonl the validate probe reads — post-hoc directions are
    usually renamed ("crit" → "crit-gm") while keeping the base concept's
    held-out data. It defaults to ``concept``.

    OPTVEC artifacts (source method ``"optvec"``, OptVec plan §6) are the
    second family with no re-derivable recipe, and the first with no source
    CONCEPT either: the direction was optimized by backprop against hashed
    target/anchor/capability datasets, so there are no stimuli, no
    validation.jsonl, and nothing under ``prompts/`` to hash-check. They are
    accepted on three conditions — the sidecar says ``optvec``, it carries
    the additive ``optvec`` provenance block (what was optimized), and the
    residual norms have been BACKFILLED (an optvec vector is born without a
    denominator) — and their ``optvec:<composite>`` stimulusSetHash travels
    verbatim. ``eval_run`` optionally names the OptVec eval run that read the
    test split; when the sidecar already records one, that is used. It is
    surfaced as a freeze ADVISORY, never a gate: an optvec concept has
    nothing to validate.

    IMPORTED GEMMA SCOPE FEATURES (source method ``"gemmaScopeSAE"``, SAE
    proposal r2) are the third such family and share the optvec exemptions
    wholesale: a decoder row is a coordinate in a published dictionary, so
    there are no stimuli (its ``gemmascope:<release>:<saeID>:<feature>``
    identity travels verbatim), no source concept, and no held-out
    validation.jsonl. Its evidence is the discovery snapshot + qualification
    artifact in the pinned SAE candidate roster, and ``Manifest.verify``
    additionally checks that any feature a CONDITION seats was nominated
    there. Unlike optvec it is born WITH a residual-norm denominator (copied
    from the import's calibration donor), so no backfill step is implied.
    """
    from .manifest import concept_validation_hash
    from ..steering.reading_position import parse_label_strict

    d = load_raw(name, root)
    base = paths.project_root() if root is None else root
    rel = (artifact_path or "").strip()
    for suffix in (".safetensors", ".json"):
        if rel.endswith(suffix):
            rel = rel[:-len(suffix)]
    if os.path.isabs(rel):
        try:
            rel = os.path.relpath(rel, base)
        except ValueError:
            raise ExperimentStoreError(
                f"artifact path '{artifact_path}' is outside the workspace — "
                "pinned inputs must be workspace-relative")
    if rel.startswith(".."):
        raise ExperimentStoreError(
            f"artifact path '{artifact_path}' is outside the workspace — "
            "pinned inputs must be workspace-relative")
    rel = rel.replace(os.sep, "/")
    tensor_path = os.path.join(base, f"{rel}.safetensors")
    sidecar_path = os.path.join(base, f"{rel}.json")
    for path in (tensor_path, sidecar_path):
        if not os.path.isfile(path):
            raise ExperimentStoreError(
                f"no vector artifact at '{rel}' — expected both "
                f"{rel}.safetensors and {rel}.json (the extension-less path "
                "is the artifact locator)")
    # Containment is REAL-path, not lexical (review finding 2026-08-10): a
    # symlink inside the workspace can point at bytes outside it while the
    # string check above passes, and the pin would then record a workspace-
    # relative locator whose bytes a bundle/copy of the workspace does not
    # carry. Same discipline as remote vector localization (safe_paths).
    from ..api.safe_paths import is_contained
    for path in (tensor_path, sidecar_path):
        if not is_contained(path, base):
            raise ExperimentStoreError(
                f"vector artifact '{rel}' resolves outside the workspace "
                f"(symlink target {os.path.realpath(path)}) — a pinned "
                "input's bytes must live under the workspace root, or a "
                "copy of the workspace silently loses them. Move or copy "
                "the artifact into the workspace and re-attach")
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentStoreError(
            f"vector artifact sidecar '{rel}.json' is not readable JSON: {exc}")

    model_id = sidecar.get("modelID")
    if model_id != d.get("modelID"):
        raise ExperimentStoreError(
            f"vector artifact '{rel}' was extracted on model {model_id!r}, "
            f"not this study's {d.get('modelID')!r} — a direction does not "
            "transfer between models")
    revision = sidecar.get("revision")
    if d.get("modelRevision") and revision and revision != d["modelRevision"]:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' was extracted at revision "
            f"{str(revision)[:12]}…, not this study's pinned "
            f"{str(d['modelRevision'])[:12]}…")
    substrate = sidecar.get("substrate")
    if substrate is not None and substrate != _THIS_SUBSTRATE:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' was extracted on substrate "
            f"{substrate!r}; this engine is {_THIS_SUBSTRATE!r} — steering "
            "vectors do not transfer across engines")
    source_method = sidecar.get("extractionMethod")
    if not source_method:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' records no extractionMethod — without "
            "it the study cannot know what its held-out validation MEANS "
            "(contrastive vs population)")
    if not sidecar.get("stimulusSetHash"):
        raise ExperimentStoreError(
            f"vector artifact '{rel}' records no stimulusSetHash — the "
            "direction's data provenance is unpinnable")
    if not sidecar.get("residualNormSource"):
        # For an OPTVEC artifact this is not a defect of the artifact but a
        # missing LIFECYCLE STEP: the training driver deliberately writes no
        # residualNormPerLayer (plan §6, like J-lens directions and Gemma
        # Scope imports), so the refusal names the verb that supplies it.
        if source_method == _OPTVEC_METHOD:
            raise ExperimentStoreError(
                f"vector artifact '{rel}' is an OptVec vector with no "
                "residualNormSource — an optvec vector is BORN without one. "
                "Run the residual-norm backfill against the pinned neutral "
                "corpus first (POST /api/vectors/backfill-norms, or "
                "steering.norm_backfill.backfill_norms) and attach the "
                "BACKFILLED artifact: α in norm units is meaningless until "
                "the denominator is measured")
        if source_method == ExtractionMethod.REPE_READER_LAT.value:
            raise ExperimentStoreError(
                reader_derived_norm_backfill_refusal(rel))
        raise ExperimentStoreError(
            f"vector artifact '{rel}' records no residualNormSource — its "
            "norm denominator (and so its recipe identity, which promotion "
            "matches on) cannot be proved")

    try:
        method = ExtractionMethod(source_method)
    except ValueError:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' records extractionMethod "
            f"{source_method!r}, which this engine does not know — it cannot "
            "resolve where the concept's held-out data lives")
    if method.is_pinned_artifact:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' is itself a materialized pinned "
            "artifact — pin the ORIGINAL it names in pinnedFrom, so the "
            "study cites the bytes' actual origin")

    data_concept = (source_concept or "").strip() or concept
    optvec_block: dict | None = None
    if not method.has_source_concept:
        # Three families invert (or sidestep) the pipeline and have no source
        # concept anywhere in them: an OptVec direction is behavior → vector
        # with no stimulus set, an imported Gemma Scope SAE decoder row is a
        # coordinate in a published dictionary, and a RepE-reader-derived
        # direction's data is the READER's own dataset and held-out split (its
        # reader is pinned separately as a readerRef). Every data-side question
        # below (source stimuli hash-check, the --source-concept hint, the
        # grand-mean population, the designated reference, the held-out
        # validation.jsonl) presumes a source CONCEPT and must be SKIPPED
        # rather than answered with an invention. What is not skipped: the
        # artifact's own identity (model, revision, substrate, both file
        # hashes) and the provenance that says where it came from.
        if (source_concept or "").strip() and \
                (source_concept or "").strip() != concept:
            kind, evidence, _referent = method.source_concept_absence
            raise ExperimentStoreError(
                f"vector artifact '{rel}' is {kind}, which has no source "
                f"concept — '{source_concept}' names stimuli and a held-out "
                "validation.jsonl that play no part in it. Attach it under "
                f"the study's own concept name; its evidence is {evidence}, "
                "not a concept's held-out set")
    if method.is_optvec:
        optvec_block = sidecar.get("optvec")
        if not isinstance(optvec_block, dict) or not optvec_block:
            raise ExperimentStoreError(
                f"vector artifact '{rel}' declares extractionMethod 'optvec' "
                "but carries no 'optvec' provenance block — a stripped "
                "sidecar cannot say WHAT was optimized (objective, λs, "
                "datasets, seed, chosen checkpoint, training run), and an "
                "optvec vector with no recorded objective certifies nothing. "
                "Re-attach the artifact the training run wrote (or its "
                "norm-backfilled copy, which preserves the block verbatim)")
    # The pin must be one verify() could pass the moment it is written:
    # attach never records a hash the very next verify would reject.
    from . import multiconcept
    if not method.has_source_concept:
        # The identity hash is carried through VERBATIM, because nothing on
        # disk under prompts/ can be compared against it: for optvec the
        # composite "optvec:<sha256 over the split files>" (the splits are
        # pinned in the training run), for a Gemma Scope import the
        # "gemmascope:<release>:<saeID>:<feature>" dictionary coordinate (the
        # published SAE is the referent, and the roster records what
        # justified nominating that feature).
        live = sidecar["stimulusSetHash"]
        _kind, _evidence, where = method.source_concept_absence
    elif method.uses_story_corpus:
        live = multiconcept.stories_hash(data_concept, root)
        where = f"prompts/emotions/{data_concept}/stories.jsonl"
    else:
        try:
            live = StimulusSet.from_directory(
                paths.concept_directory(data_concept, root)).hash
        except Exception:  # noqa: BLE001 - absence IS the message below
            live = None
        where = f"prompts/concepts/{data_concept}/"
    if live is None:
        suggestion = (sidecar.get("familyGrandMeanCentring") or {}).get(
            "baseConcept") if isinstance(
                sidecar.get("familyGrandMeanCentring"), dict) else None
        hint = (f" (the artifact names base concept '{suggestion}' — try "
                f"--source-concept {suggestion})") if suggestion else ""
        raise ExperimentStoreError(
            f"no stimulus data at {where} for concept '{data_concept}', so "
            f"the study could never validate '{concept}' — pass the concept "
            f"the artifact was derived from{hint}")
    if live != sidecar["stimulusSetHash"]:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' was extracted from stimuli hashing "
            f"{sidecar['stimulusSetHash'][:12]}…, but {where} now hashes "
            f"{live[:12]}… — restore the bytes the artifact was built on, or "
            "pass the right --source-concept")

    # The READING POSITION is copied from the sidecar so held-out activations
    # are read where the vector was read — FAITHFULLY, which used to mean
    # something much narrower than it says. The pin was built by asking the
    # parsed position for its ``requested_start_index`` and re-deriving a
    # position from that integer, and only two of the eight positions survive
    # that round trip: ``meanFromToken(k)`` (the one with a start index) and
    # ``lastToken`` (everything else answers None). ``offsetFromEnd``,
    # ``lastContentToken``, ``turnCloseToken``, ``postInstruction``,
    # ``contentOffset`` and ``meanContentFromToken`` all collapsed to
    # last-token, so the manifest contradicted its own sidecar and the next
    # verify refused the pin attach had just written (external review round 5).
    #
    # STRICT parse, and a named refusal on anything unreadable — the same
    # answer the rendering copy below gives, and for the same reason: attach
    # must never write a pin the next verify rejects, and guessing a position
    # would launder a wrong guess into the recipe identity. An ABSENT label is
    # the legacy artifact, and stays the default (last-token) — matching the
    # Swift twin, ``ExperimentStore.attachArtifactPin``, which leaves
    # ``ExtractionOptions``' default in place when the sidecar records none.
    recorded_position_label = sidecar.get("readingPosition")
    recorded_position = None
    if recorded_position_label is not None:
        recorded_position = parse_label_strict(recorded_position_label)
        if recorded_position is None:
            raise ExperimentStoreError(
                f"vector artifact '{rel}' records reading position "
                f"'{recorded_position_label}', which this engine cannot parse "
                "— re-attach on the engine that wrote it")

    entry: dict = {
        "name": concept,
        "stimulusSetHash": sidecar["stimulusSetHash"],
        "options": {
            "method": ExtractionMethod.PINNED_ARTIFACT.value,
            "readingPosition": _reading_position_codable(
                {}, recorded_position),
        },
        "vectorArtifact": {
            "path": rel,
            "sha256TensorHash": _sha256_path(tensor_path),
            "sha256SidecarHash": _sha256_path(sidecar_path),
            "sourceMethod": source_method,
            "sourceConcept": data_concept,
            "residualNormSource": sidecar["residualNormSource"],
        },
        # A source-concept-less concept (optvec, imported SAE feature) pins
        # validation EXPLICITLY NULL (never merely absent, which reads as a
        # legacy attach): there is no held-out validation.jsonl to pin, and a
        # file appearing later under this name would be unrelated to the
        # direction.
        "validationHash": (
            None if not method.has_source_concept else concept_validation_hash(
                data_concept, paired=not method.uses_story_corpus, root=root)),
    }
    if method.is_optvec:
        # The optimization's own identity, copied so the manifest is
        # self-describing about what this direction was trained to do and
        # WHICH evidence run certifies it (freeze surfaces the latter as an
        # advisory). Absent keys stay absent — never a guessed reference.
        assert optvec_block is not None
        for source_key, entry_key in (("layer", "optvecLayer"),
                                      ("runID", "optvecTrainingRun"),
                                      ("seed", "optvecSeed")):
            if optvec_block.get(source_key) is not None:
                entry["vectorArtifact"][entry_key] = optvec_block[source_key]
        recorded_eval = eval_run or _recorded_optvec_eval_run(optvec_block)
        if recorded_eval:
            verification = _verify_optvec_eval_run(
                str(recorded_eval),
                entry["vectorArtifact"]["sha256TensorHash"], root)
            entry["vectorArtifact"]["optvecEvalRun"] = str(recorded_eval)
            entry["vectorArtifact"]["optvecEvalRunVerified"] = \
                verification["verified"]
            if not verification["verified"]:
                entry["vectorArtifact"]["optvecEvalRunUnverifiedReason"] = \
                    verification["reason"]
    # The RENDERING travels with the reading position, for exactly the reason
    # the position does: verify compares the two canonically, so an artifact
    # extracted through the chat template must not attach into a manifest that
    # says raw. A raw (or absent) recording writes NO key, so a legacy artifact
    # still attaches to byte-identical manifest JSON. Swift twin:
    # ``ExperimentStore.attachArtifactPin``.
    from ..steering.extraction_rendering import ExtractionRenderingError
    from ..steering.extraction_rendering import from_json as rendering_from_json
    try:
        recorded_rendering = rendering_from_json(
            sidecar.get("extractionRendering"))
    except ExtractionRenderingError as exc:
        raise ExperimentStoreError(
            f"vector artifact '{rel}' records an extractionRendering this "
            f"engine cannot read ({exc}) — re-attach on the engine that "
            "wrote it")
    if not recorded_rendering.is_raw:
        entry["options"]["extractionRendering"] = recorded_rendering.to_dict()
    if sidecar.get("neutralCorpusHash"):
        entry["vectorArtifact"]["normCorpusHash"] = sidecar["neutralCorpusHash"]
    if method.is_designated_reference:
        ref = sidecar.get("designatedReference")
        if not (isinstance(ref, dict) and ref.get("name") and ref.get("hash")):
            raise ExperimentStoreError(
                f"vector artifact '{rel}' is a designated-reference direction "
                "but records no designatedReference {name, hash} — the "
                "reference is part of what validation compares against")
        live_ref = multiconcept.stories_hash(ref["name"], root)
        if live_ref != ref["hash"]:
            raise ExperimentStoreError(
                f"the artifact's reference '{ref['name']}' stories hash "
                f"{ref['hash'][:12]}… does not match the bytes on disk "
                f"({(live_ref or 'missing')[:12]}…) — restore them before "
                "pinning the artifact")
        entry["designatedReference"] = {"name": ref["name"], "hash": ref["hash"]}
    if method.is_grand_mean:
        population = sidecar.get("grandMeanPopulation")
        if not isinstance(population, dict) or not population:
            raise ExperimentStoreError(
                f"vector artifact '{rel}' is a grand-mean direction but "
                "records no grandMeanPopulation — the population IS the "
                "comparison, so validation cannot be reproduced without it")
        existing = d.get("grandMeanCorpus") or {}
        members = list(dict.fromkeys(
            (existing.get("concepts") or []) + sorted(population)))
        hashes = dict(existing.get("hashes") or {})
        for member, digest in sorted(population.items()):
            live_member = multiconcept.stories_hash(member, root)
            if live_member != digest:
                raise ExperimentStoreError(
                    f"grand-mean population member '{member}' hashes "
                    f"{(live_member or 'missing')[:12]}… on disk but "
                    f"{str(digest)[:12]}… in the artifact — restore the bytes "
                    "the artifact was built on")
            hashes[member] = digest
        d["grandMeanCorpus"] = {"concepts": members, "hashes": hashes}

    refs = {c["name"]: c for c in d.get("concepts", [])}
    refs[concept] = entry
    d["concepts"] = list(refs.values())
    corpus_path = paths.neutral_corpus_path(root)
    if os.path.exists(corpus_path):
        d["neutralCorpusHash"] = load_texts(corpus_path).hash
    save_raw(d, root)
    return d


#: Every DECLARATION in a manifest that reads a pinned concept BY NAME and
#: would be left dangling if the pin went away, as
#: ``(label, extractor)`` — the extractor answers the declaration names that
#: reference one concept. Named once so :func:`concept_dependents` and its
#: Swift twin (``ExperimentStore.conceptDependents``) can be read against each
#: other, and so a new concept-referencing block is added in ONE place.
#:
#: The audit behind the list (every concept-name reference in the schema) and
#: what is deliberately OUT of it:
#:
#: * ``grandMeanCorpus.concepts`` / ``.hashes`` — HANDLED, not gated: the
#:   corpus is dropped when the last grand-mean target leaves, and a corpus
#:   wider than the remaining targets is kept (the population is part of the
#:   remaining vectors' recipe).
#: * ``concepts[].designatedReference.name`` and
#:   ``concepts[].vectorArtifact.sourceConcept`` — they name an on-disk DATA
#:   concept (``prompts/emotions/<n>/stories.jsonl``, ``prompts/concepts/<n>/``),
#:   which need never be attached; detaching a pin cannot dangle them.
#: * ``validationControls[].concept`` — a discriminant control is refused if it
#:   IS a study concept, so it is never a dependent of one.
#: * ``readerRefs[].concept`` — binds to a fitted reader ARTIFACT, checked
#:   against the artifact's own concept, never against the pin list.
#: * ``conditions[].selection`` and ``variantConditions[].artifact.promotion``
#:   embed a criterion copy whose ``choicePromptsFiles`` is concept-keyed —
#:   but those are stamped PROVENANCE (what a past sweep selected on), and the
#:   live reference in the same condition is its slot, which is row one.
_CONCEPT_DEPENDENT_SOURCES: tuple = (
    ("condition", lambda d, c: [
        str(cond.get("name") or "?")
        for cond in (d.get("conditions") or [])
        if any((slot or {}).get("concept") == c
               for slot in (cond.get("slots") or []))]),
    ("sweep selection instrument", lambda d, c: [
        f"sweep.selection.objective.{key}[{c}]"
        for key in ("choicePromptsFiles", "choicePromptsHashes")
        if isinstance((((d.get("sweep") or {}).get("selection") or {})
                       .get("objective") or {}).get(key), dict)
        and c in ((((d.get("sweep") or {}).get("selection") or {})
                   .get("objective") or {})[key])]),
    ("variant condition", lambda d, c: [
        str(variant.get("name") or "?")
        for variant in (d.get("variantConditions") or [])
        if isinstance(variant.get("fromPromotion"), dict)
        and (variant["fromPromotion"].get("concept") or "") == c]),
    ("perturbation policy", lambda d, c: [
        "perturbationPolicy"
        for policy in [d.get("perturbationPolicy")]
        if isinstance(policy, dict) and (policy.get("concept") or "") == c]),
)


def concept_dependents(d: dict, concept: str) -> list[str]:
    """Every declaration in ``d`` that names ``concept``, as
    ``"<label> '<name>'"`` strings in :data:`_CONCEPT_DEPENDENT_SOURCES` order.

    Empty means the pin can be removed without orphaning anything. Swift twin:
    ``ExperimentStore.conceptDependents(_:in:)`` — the strings are the
    cross-engine contract, because the refusal built from them is.
    """
    found: list[str] = []
    for label, extract in _CONCEPT_DEPENDENT_SOURCES:
        for name in extract(d, concept):
            found.append(f"{label} '{name}'")
    return found


#: THE repair for :data:`lifecycle_gates.CONCEPT_IN_USE`, on both engines.
#: Authoring is Mac-authority (audit §10.x), so the server's copy names that
#: binary too. Swift twin: ``ExperimentStore.conceptInUseRepair``.
def concept_in_use_repair(name: str) -> str:
    return (f"remove or re-declare those conditions first: steerlab-cli "
            f"experiment declare-condition {name} <condition> … (re-declare "
            f"onto a concept that stays), then steerlab-cli experiment detach "
            f"{name} <concept>…")


#: THE repair for detaching a concept the manifest never pinned. Swift twin:
#: ``ExperimentStore.conceptNotPinnedRepair``.
def concept_not_pinned_repair(name: str) -> str:
    return (f"steerlab-cli experiment list  (result.experiments[].concepts "
            f"names what '{name}' pins), then steerlab-cli experiment detach "
            f"{name} <one of those>")


def detach(name: str, concepts: list[str], root: str | None = None) -> dict:
    """Remove pinned concepts from a DRAFT manifest — the inverse of
    :func:`attach`, and the reason it exists: before it, a pinned concept could
    not be removed or re-pointed HEADLESSLY at all, so a draft carried whatever
    it was first attached with and a re-pointing across many drafts was not
    expressible as a command.

    All-or-nothing (the ``add_conditions`` rule, for the same reason): every
    named concept is checked against the whole manifest BEFORE anything is
    written, so a two-concept detach cannot land the first and refuse on the
    second, leaving a draft nobody asked for.

    Two typed refusals:

    * a concept the manifest does not pin — ``missingPrerequisite``, naming
      what IS pinned, because the commonest cause is a typo and the list is
      the answer;
    * a concept some DECLARATION still reads by name —
      ``conceptInUse``, naming every dependent. Detaching anyway would leave
      a dangling reference that only the next ``verify`` finds, and a run in
      between would have measured a study nobody declared. That is the
      silent-drop class this engine refuses on principle.

    The grand-mean corpus follows the pins: when the last grand-mean target
    leaves, the corpus goes with it (nothing left to define). A corpus wider
    than the remaining targets is deliberately KEPT — the population is part
    of the remaining vectors' recipe. Swift twin:
    ``ExperimentStore.detachConcepts``.
    """
    wanted = [c.strip() for c in concepts]
    if not wanted or any(not c for c in wanted):
        raise ExperimentStoreError(
            "detach needs at least one concept name — "
            f"'{name}' keeps every pin it has")
    d = load_raw(name, root)
    # Status FIRST, before the pin list is even consulted. `save_raw` would
    # refuse a frozen manifest anyway, but only after the two refusals below
    # had had their chance — so a frozen study with a dependent condition
    # would have answered `conceptInUse` here and `statusImmutable` on the
    # Mac (whose `updateDraft` checks status before it mutates). Same input,
    # same gate id, either engine.
    status = str(d.get("status") or "draft")
    if status != "draft":
        raise ExperimentStoreError(
            f"'{name}' is {status} and read-only — duplicate it to iterate",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 && "
                    f"steerlab-cli experiment detach {name} <concept>…  "
                    "(frozen studies are immutable; the duplicate is a draft "
                    "again)"))
    pinned = [str(c.get("name") or "") for c in (d.get("concepts") or [])]
    for concept in wanted:
        if concept not in pinned:
            raise ExperimentStoreError(
                f"concept '{concept}' is not pinned to '{name}' — pinned: "
                + (", ".join(pinned) if pinned else "(none)"),
                gate=lifecycle_gates.MISSING_PREREQUISITE,
                repair=concept_not_pinned_repair(name))
    for concept in wanted:
        dependents = concept_dependents(d, concept)
        if dependents:
            raise ExperimentStoreError(
                f"concept '{concept}' is still declared by "
                + ", ".join(dependents)
                + " — remove or re-declare those conditions first",
                gate=lifecycle_gates.CONCEPT_IN_USE,
                repair=concept_in_use_repair(name))
    removed = set(wanted)
    d["concepts"] = [c for c in (d.get("concepts") or [])
                     if str(c.get("name") or "") not in removed]
    if not any(_is_grand_mean_ref(c) for c in d["concepts"]):
        d.pop("grandMeanCorpus", None)
    # THE intentional clear-all flow, the same one `remove_condition` declares
    # (open-issues §8): detaching the last concept of a condition-less draft
    # is a researcher removing one named pin they authored, one call at a
    # time — declared intent, not a stale document landing on a populated one.
    save_raw(d, root, clearing_arms=True)
    return d


def _is_grand_mean_ref(ref: dict) -> bool:
    """Whether a stored concept pin was attached by the grand-mean recipe —
    the only pins the ``grandMeanCorpus`` block exists for.
    ``designatedReference`` is deliberately NOT one: it pins its reference
    corpus inside the concept ref itself. Same predicate as
    ``ExtractionMethod.is_grand_mean`` / Swift ``ExtractionMethod.isGrandMean``,
    read off the STORED string because this path never inflates a Manifest."""
    return str(((ref or {}).get("options") or {}).get("method") or "") \
        == "emotionGrandMean"


def _sha256_path(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def pin_model_revision(name: str, revision: str, root: str | None = None) -> dict:
    """Draft-only model-revision auto-pin (twin of Swift
    ``ExperimentTasks.loadContainer(pinning:)``): the first model-loading task
    writes the revision it actually resolved back into the manifest, BEFORE
    any artifacts exist — so the identity the manifest requires
    (:func:`recipe_identity.required_identity` reads ``modelRevision``) and
    the identity every extraction sidecar stamps (``model.revision``) agree
    on the concrete revision. A no-op when the manifest already pins one or
    is not a draft (frozen manifests are immutable; freeze gates on the pin,
    so a frozen-null manifest is legacy only)."""
    d = load_raw(name, root)
    if d.get("modelRevision") or d.get("status") != "draft":
        return d
    d["modelRevision"] = revision
    save_raw(d, root)
    return d


# =============================================================================
# The sweep GRID (`set-sweep-grid`) — Swift twins throughout
# =============================================================================

#: The sweep block's fields `set-sweep-selection` owns, as the flags that write
#: them. `set-sweep-grid` owns exactly what is left — the two axes, the two
#: instrument files, and the per-cell budget — and points here rather than
#: silently writing a field it does not own. ``devPromptsHash`` /
#: ``batteryHash`` are owned by NEITHER: they are pinned at freeze from the
#: bytes on disk and never authored. Swift twin:
#: ``ExperimentStore.sweepSelectionOwnedFlags``.
SWEEP_SELECTION_OWNED_FLAGS: tuple[str, ...] = (
    "--objective", "--choice-prompts", "--capability-tolerance",
    "--coherence-floor", "--control-margin", "--control-apply-to",
    "--control-top-k",
)

#: The engine defaults a draft with no sweep block starts from — identical to
#: Swift ``ExperimentManifest.SweepSpec.init`` and to
#: ``tasks.DEFAULT_SWEEP_LAYER_FRACTIONS`` / ``DEFAULT_SWEEP_ALPHAS``.
_DEFAULT_SWEEP_BLOCK: dict = {
    "layerFractions": [0.5, 0.7, 0.85],
    "alphas": [0.05, 0.08, 0.1, 0.13],
    "devPromptsFile": "prompts/dev/dev-prompts.jsonl",
    "batteryFile": "prompts/batteries/basic.jsonl",
    "maxTokens": 80,
}


def sweep_selection_owns_repair(name: str, flag: str) -> str:
    """THE pointer for asking ``set-sweep-grid`` to write a field
    ``set-sweep-selection`` owns. Swift twin:
    ``ExperimentStore.sweepSelectionOwnsRepair``."""
    return (f"steerlab-cli experiment set-sweep-selection {name} {flag} "
            "<value>  (the selection RULE is that verb's; set-sweep-grid "
            "writes the layer × alpha grid the rule then picks a winner from)")


def _number_text(value: float) -> str:
    """Locale-independent rendering for a number inside refusal prose, so the
    twin literals compare byte-for-byte across engines. Swift twin:
    ``ExperimentStore.numberText``."""
    if value == value and abs(value) != float("inf") \
            and float(value) == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _first_non_ascending(values) -> float | None:
    """The first entry that is not strictly greater than the one before it, or
    None when the list ascends. Swift twin:
    ``ExperimentStore.firstNonAscending``."""
    for index, value in enumerate(values):
        if index and not value > values[index - 1]:
            return value
    return None


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def sweep_grid_problem(layer_fractions, alphas, dev_prompts_file: str,
                       battery_file: str, max_tokens: int) -> str | None:
    """Why one declared grid cannot be swept, or None when it can. Every check
    is on the DECLARATION, so it answers at authoring time rather than after a
    model has loaded.

    The two ascent rules are the ones the shape of a dose-response makes
    non-negotiable. Alphas are a LADDER: the sweep reports a dose-response
    curve, and a ladder declared out of order reads as a curve that doubles
    back, while a repeated rung is a cell paid for twice and reported once.
    Layer fractions are the same argument one axis over, with an extra wrinkle
    — ``resolve_sweep_layers`` sorts and deduplicates them anyway, so an
    unsorted or repeated declaration is a grid whose written form and run form
    disagree.

    Swift twin: ``ExperimentStore.sweepGridProblem`` — the strings are the
    cross-engine contract, because the refusal built from them is.
    """
    if not layer_fractions:
        return "the layer axis is empty — a grid names at least one depth"
    bad = next((f for f in layer_fractions
                if not _finite(f) or f < 0 or f > 1), None)
    if bad is not None:
        return f"layer fractions are depths in [0, 1] — got {_number_text(bad)}"
    out = _first_non_ascending(layer_fractions)
    if out is not None:
        return (f"the layer axis does not ascend at {_number_text(out)} — "
                "declare depths in increasing order, each one once (the sweep "
                "sorts and deduplicates them, so an unordered declaration "
                "names a grid it will not run)")
    if not alphas:
        return "the alpha axis is empty — a grid names at least one dose"
    bad = next((a for a in alphas if not _finite(a) or a <= 0), None)
    if bad is not None:
        return ("alphas are residual-norm units above 0 — got "
                f"{_number_text(bad)} (0 is the baseline cell, which every "
                "sweep runs anyway)")
    out = _first_non_ascending(alphas)
    if out is not None:
        return (f"the alpha ladder does not ascend at {_number_text(out)} — "
                "declare doses in increasing order, each one once (a ladder "
                "that doubles back is not a dose-response)")
    if max_tokens <= 0:
        return f"max tokens must be above 0 — got {max_tokens}"
    if not dev_prompts_file.strip():
        return "the dev-prompts file is required — the sweep generates on it"
    if not battery_file.strip():
        return ("the capability-battery file is required — the sweep scores "
                "every cell on it")
    return None


def depth_fraction_for_layer(layer: int, depth: int) -> float:
    """The depth fraction that resolves to exactly ``layer`` in a network of
    ``depth`` blocks — the MIDPOINT of the fraction band that truncates to it,
    ``(layer + 0.5) / depth``.

    The midpoint is not an aesthetic choice. ``resolve_sweep_layers``
    truncates (``int(depth * f)``), so every fraction in
    ``[L/depth, (L+1)/depth)`` names layer L, and the two ends of that band are
    one rounding error from naming L-1 or L+1 instead. The midpoint sits half a
    layer from either edge, which is ~10¹⁴ times the double-rounding error of
    the round trip. Checked anyway at the call site rather than asserted here.

    Swift twin: ``ExperimentStore.depthFraction(forLayer:depth:)``.
    """
    return (float(layer) + 0.5) / float(depth)


def cached_layer_count(model_id: str, root: str | None = None) -> int | None:
    """The pinned model's depth, from any vector sidecar already on disk for
    it — None when nothing has been extracted for it yet.

    Deliberately catalog-only, exactly as Swift
    ``SweepPanelModel.cachedLayerCount`` is: loading a 27B model to answer
    "what is 0.66 of this network?" is the reason that question went
    unanswered. The import is local because this module is on the light-install
    path and ``catalog`` walks the tree.
    """
    from . import catalog
    for artifact in catalog.list_vectors(root):
        if artifact.modelID == model_id and artifact.layerCount > 0:
            return int(artifact.layerCount)
    return None


def sweep_grid_repair(name: str) -> str:
    """THE repair for a grid no engine could sweep. Swift twin:
    ``ExperimentStore.sweepGridRepair``."""
    return (f"steerlab-cli experiment set-sweep-grid {name} "
            "--layer-fractions 0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13  "
            "(both axes ascend, each value once; alphas are residual-norm "
            "units above 0)")


def absolute_layers_need_depth_repair(name: str) -> str:
    """THE repair for absolute layers with no depth to read them against.
    Extraction is named first because it makes the ORIGINAL request
    answerable; fractions are the answer that needs no model at all. Swift
    twin: ``ExperimentStore.absoluteLayersNeedDepthRepair``."""
    return (f"steerlab-cli experiment extract {name}  (any vector for the "
            f"pinned model states its depth) && steerlab-cli experiment "
            f"set-sweep-grid {name} --layers <L>,…  ; or declare the grid in "
            "depth fractions, which need no model: steerlab-cli experiment "
            f"set-sweep-grid {name} --layer-fractions 0.5,0.7,0.85")


def absolute_layers_out_of_range_repair(name: str, depth: int) -> str:
    """THE repair for an absolute layer outside the pinned model. Swift twin:
    ``ExperimentStore.absoluteLayersOutOfRangeRepair``."""
    return (f"steerlab-cli experiment set-sweep-grid {name} "
            f"--layers <0…{depth - 1}>,…  ; or declare depths instead, which "
            "survive a change of model: steerlab-cli experiment set-sweep-grid "
            f"{name} --layer-fractions 0.5,0.7,0.85")


def set_sweep_grid(name: str, *, layer_fractions=None, layers=None,
                   alphas=None, dev_prompts_file: str | None = None,
                   battery_file: str | None = None,
                   max_tokens: int | None = None,
                   layer_count: int | None = None,
                   root: str | None = None) -> dict:
    """Declare the sweep's GRID on a draft — the layer × alpha ladder, the two
    instrument files it reads, and its per-cell token budget.

    The gap this closes is the one the passenger-concept problem sat in: the
    grid was reachable only from the Mac app's Optimizations panel, so the ONLY
    headless way to obtain one was to ``duplicate`` a study that already had it
    — which carries the donor's concepts along with its grid, and a concept
    that rode in that way is swept but not citable.

    Layers are declarable both ways. ``layer_fractions`` are depths in [0, 1]
    and are what the manifest stores: they are the portable form, and the sweep
    resolves them at run time against the model it actually loaded.
    ``layers`` is the spelling a researcher reading a paper has ("L28"), and it
    is converted HERE against the pinned model's depth as this workspace knows
    it, with both forms reported back. The manifest deliberately gains no
    second key for the absolute layers: an axis with two stored spellings is an
    axis that can disagree with itself, and the depth is a property of the
    pinned model, which the manifest already names.

    Alphas are RESIDUAL-NORM UNITS, the house convention everywhere in this
    schema. 0 is the implied baseline cell and is never a rung.

    ``None`` arguments leave their field as it stands, so this verb can move
    one axis without restating the block. Returns the manifest document with a
    non-persisted ``_sweepGrid`` report of what the declaration resolves to;
    callers read it and drop it. Swift twin: ``ExperimentStore.setSweepGrid``.
    """
    d = load_raw(name, root)
    status = str(d.get("status") or "draft")
    if status != "draft":
        raise ExperimentStoreError(
            f"'{name}' is {status} and read-only — duplicate it to iterate",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 && "
                    f"steerlab-cli experiment set-sweep-grid {name}-v2 …  "
                    "(frozen studies are immutable; the duplicate is a draft "
                    "again)"))
    spec = dict(d.get("sweep") or {})
    for key, value in _DEFAULT_SWEEP_BLOCK.items():
        spec.setdefault(key, list(value) if isinstance(value, list) else value)
    model_id = str(d.get("modelID") or "")
    depth = layer_count if layer_count is not None \
        else cached_layer_count(model_id, root)
    fractions = list(layer_fractions) if layer_fractions is not None \
        else list(spec["layerFractions"])
    if layers is not None:
        absolute = [int(layer) for layer in layers]
        if not depth or depth <= 0:
            raise ExperimentStoreError(
                f"absolute layers were declared for '{name}', but nothing in "
                f"this workspace states how deep '{model_id}' is — no "
                "extracted vector for it carries a layer count, and a layer "
                "index means nothing without one",
                gate=lifecycle_gates.MISSING_PREREQUISITE,
                repair=absolute_layers_need_depth_repair(name))
        out = _first_non_ascending(absolute)
        if out is not None:
            # Checked on the INDICES, before conversion: the same rule fires
            # on the fractions below, but it would say "the layer axis does
            # not ascend at 0.42" about a declaration that named 11.
            raise ExperimentStoreError(
                f"the layer axis does not ascend at {int(out)} — declare "
                "blocks in increasing order, each one once (the sweep sorts "
                "and deduplicates them, so an unordered declaration names a "
                "grid it will not run)",
                gate=lifecycle_gates.SWEEP_GRID_RULE,
                repair=absolute_layers_out_of_range_repair(name, depth))
        bad = next((layer for layer in absolute
                    if layer < 0 or layer >= depth), None)
        if bad is not None:
            raise ExperimentStoreError(
                f"layer {bad} is outside '{model_id}', which has {depth} "
                f"block(s) — legal layers are 0…{depth - 1}",
                gate=lifecycle_gates.SWEEP_GRID_RULE,
                repair=absolute_layers_out_of_range_repair(name, depth))
        fractions = [depth_fraction_for_layer(layer, depth)
                     for layer in absolute]
        # Defensive, and cheap: the midpoint rule is exact for every depth a
        # language model has, but "exact by argument" is not the standard this
        # schema holds its own writes to.
        from .manifest import resolve_sweep_layers
        round_trip = resolve_sweep_layers(depth, fractions)
        if round_trip != sorted(absolute):
            raise ExperimentStoreError(
                f"the declared layers {absolute} do not survive conversion to "
                f"depth fractions at depth {depth} (they would name "
                f"{round_trip}) — declare the grid as fractions instead",
                gate=lifecycle_gates.SWEEP_GRID_RULE,
                repair=absolute_layers_out_of_range_repair(name, depth))
    ladder = list(alphas) if alphas is not None else list(spec["alphas"])
    dev = dev_prompts_file if dev_prompts_file is not None \
        else str(spec["devPromptsFile"])
    battery = battery_file if battery_file is not None \
        else str(spec["batteryFile"])
    budget = int(max_tokens) if max_tokens is not None \
        else int(spec["maxTokens"])
    problem = sweep_grid_problem(fractions, ladder, dev, battery, budget)
    if problem:
        raise ExperimentStoreError(
            problem, gate=lifecycle_gates.SWEEP_GRID_RULE,
            repair=sweep_grid_repair(name))
    spec["layerFractions"] = [float(f) for f in fractions]
    spec["alphas"] = [float(a) for a in ladder]
    spec["devPromptsFile"] = dev
    spec["batteryFile"] = battery
    spec["maxTokens"] = budget
    # A grid edit re-opens the two block-level pins: they certify the bytes of
    # files this verb may have just re-pointed, and a pin kept over a path that
    # moved is a claim about a file nobody read. Freeze re-pins from disk when
    # they are absent, which is the one moment a pin is allowed to be minted.
    if dev_prompts_file is not None:
        spec.pop("devPromptsHash", None)
    if battery_file is not None:
        spec.pop("batteryHash", None)
    d["sweep"] = spec
    save_raw(d, root)
    resolved: list[int] = []
    collapsed = 0
    if depth and depth > 0:
        from .manifest import resolve_sweep_layers
        resolved = resolve_sweep_layers(depth, spec["layerFractions"])
        collapsed = max(0, len(set(spec["layerFractions"])) - len(resolved))
    report = dict(d)
    report["_sweepGrid"] = {
        "layerCount": depth if depth and depth > 0 else None,
        "resolvedLayers": resolved,
        "collapsedFractions": collapsed,
        "declaredAbsoluteLayers": layers is not None,
    }
    return report


def set_protocol(name: str, fields: dict, root: str | None = None) -> dict:
    d = load_raw(name, root)
    allowed = {"experimentDescription", "taskDescription", "outcomeMeasures", "promptMode",
               "systemPrompt", "qwenThinkingEnabled", "temperature", "maxTokens", "seeds",
               # studyType: the researcher's declared study type (authoring
               # vocabulary: conceptStudy | agentComparison | confirmAgent |
               # multiAgent) — persisted verbatim; studyKind stays the
               # engine-facing run-path switch.
               "taskPromptsFile", "taskPromptsHash", "studyKind", "studyType",
               "multiAgentScenarioPath",
               "multiAgentScenarioHash", "multiAgentIncludeBaseline", "evaluation",
               "judgeRubricFile", "judgeRubricHash", "judges", "humanValidation",
               "capabilityBatteryFile", "capabilityBatteryHash",
               "reasoningStyleTaxonomyPath", "reasoningStyleTaxonomyHash",
               # Declared record-exclusion rules (closed vocabulary,
               # validated by verify(); joined at analyze) — measurement
               # declarations, so draft-editable like the other protocol
               # fields and frozen with the manifest.
               "exclusionRules"}
    for key, value in fields.items():
        if key in allowed:
            d[key] = value
    save_raw(d, root)
    return d


def pin_reasoning_style_taxonomy(name: str, path: str,
                                 root: str | None = None) -> dict:
    """Pin a reasoning-style taxonomy into a draft manifest (parallel to
    Swift ``pinReasoningStyleTaxonomy`` / the ``set-style-taxonomy`` verb):
    validates the taxonomy LOADS on this engine (both-engine loadability is
    the contract), then stamps ``reasoningStyleTaxonomyPath`` +
    ``reasoningStyleTaxonomyHash`` (SHA-256 of the file bytes). Drift after
    pinning is a verify() violation."""
    import hashlib

    from .reasoning_style import Taxonomy, TaxonomyError
    base = paths.project_root() if root is None else root
    resolved = path if os.path.isabs(path) else os.path.join(base, path)
    try:
        with open(resolved, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ExperimentStoreError(
            f"no taxonomy file at {path} — author one under prompts/taxonomies/ "
            f"(see prompts/templates/reasoning-style/): {exc}")
    try:
        Taxonomy.from_bytes(data)  # validation gate
    except TaxonomyError as exc:
        raise ExperimentStoreError(str(exc))
    d = load_raw(name, root)
    d["reasoningStyleTaxonomyPath"] = path
    d["reasoningStyleTaxonomyHash"] = hashlib.sha256(data).hexdigest()
    save_raw(d, root)
    return d


def pin_sae_candidates(name: str, path: str, root: str | None = None) -> dict:
    """Pin an SAE candidate roster into a DRAFT manifest (proposal r2 §8 P1-6).

    The attach-equivalent moment for the candidate manifest: validate that the
    file LOADS on this engine (a pin over bytes the engine would refuse to
    read certifies nothing), then stamp ``saeCandidates`` =
    ``{path, hash}`` with the SHA-256 of the file bytes. Drift, disappearance,
    or a later edit is a verify() violation from that moment on — which is the
    point: the roster records which features a study MAY seat and the
    discovery evidence behind each nomination, so it must be fixed before
    behavior is measured, like every other measurement-side input.

    The path is WORKSPACE-RELATIVE by rule (the Mac workspace is the source of
    truth; an absolute path resolves to nothing on the cluster). Frozen
    manifests are immutable — duplicate and re-pin instead."""
    from . import sae_candidates as sae_candidates_mod
    if os.path.isabs(path):
        raise ExperimentStoreError(
            f"SAE candidate manifest path '{path}' is absolute — pin it "
            "workspace-relative (e.g. prompts/sae/candidates.json) so the "
            "study resolves on any machine")
    d = load_raw(name, root)
    if d.get("status") != "draft":
        raise ExperimentStoreError(
            f"cannot pin an SAE candidate manifest into '{name}': it is "
            f"{d.get('status')} — duplicate the experiment and pin there")
    try:
        _, digest = sae_candidates_mod.load(path, root)
    except sae_candidates_mod.CandidateManifestError as exc:
        raise ExperimentStoreError(str(exc))
    d["saeCandidates"] = {"path": path, "hash": digest}
    save_raw(d, root)
    return d


#: The repair for a NEW condition declaration that names no ``alphaInNormUnits``
#: — in BOTH spellings a caller can write it in, because the two authoring
#: surfaces are the manifest document (this API, and the ``/api/authoring/
#: {name}/condition`` route that fronts it) and the Mac CLI verb. Swift twin:
#: ``ExperimentManifest.alphaUnitsRepairAction``
#: (``Sources/ExperimentKit/ExperimentStore.swift``) — the same two spellings,
#: written out independently so neither engine can quietly follow the other.
ALPHA_UNITS_REPAIR = (
    'declare the α units explicitly: add "alphaInNormUnits": true '
    "(α in residual-stream-norm units — the project convention) or "
    "false (raw α) to the condition, or declare the arm with "
    "`steerlab-cli experiment declare-condition <study> <condition> "
    "--slots <concept>:<layer>:<alpha> --alpha-units norm|raw`")


def _condition_entry(condition: dict) -> dict:
    """Project one DECLARED condition into the manifest's stored shape.

    ``alphaInNormUnits`` is REQUIRED here (Phase-0 gap G6,
    ``docs/PORTABILITY-CONTRACTS.md``). It used to default to ``False`` while
    the Swift engine's ``Condition.init`` defaulted to ``true``, so the same
    client call authored a different study depending on which engine served it
    — and α units are dose semantics, not a display setting: the same α number
    is a different intervention in each convention. Neither engine picks a
    default now; both refuse a new declaration that does not say.

    READING an existing manifest is untouched: ``Manifest.from_dict`` still
    reads a legacy key-less condition as ``False``, which is the reading this
    engine has always given it. Reinterpreting a frozen study's dose is the one
    thing this repair must not do.
    """
    if "alphaInNormUnits" not in condition:
        raise ExperimentStoreError(
            f"condition {condition.get('name', '?')!r} declares no "
            "'alphaInNormUnits', so the α it names has no unit — this engine "
            "would read raw α and the Mac engine residual-norm units for the "
            f"same document. {ALPHA_UNITS_REPAIR}",
            repair=ALPHA_UNITS_REPAIR)
    entry = {
        "name": condition["name"],
        "slots": [{"concept": s["concept"], "layer": int(s["layer"]), "alpha": float(s["alpha"])}
                  for s in condition.get("slots", [])],
        "bandWidth": int(condition.get("bandWidth", 1)),
        "alphaInNormUnits": bool(condition["alphaInNormUnits"]),
    }
    # Sweep-selection provenance (cross-engine contract): the sweep stamps how
    # its `<concept>-recommended` cell was chosen — run, resolved criterion,
    # dev-split hash, winning cell, metrics, control. Preserved verbatim; this
    # block is what `promote` later copies into an agent's birth certificate.
    if isinstance(condition.get("selection"), dict):
        entry["selection"] = condition["selection"]
    return entry


def add_condition(name: str, condition: dict, root: str | None = None) -> dict:
    return add_conditions(name, [condition], root)


def add_conditions(name: str, conditions: list[dict],
                   root: str | None = None) -> dict:
    """All-or-nothing projection of several conditions: one load, one save
    (review 2026-08-03 round 3, P2 — per-condition saves could land concept
    A's `<concept>-recommended` and die before concept B's, leaving a
    partial projection in the manifest with no completion marker)."""
    d = load_raw(name, root)
    conds = list(d.get("conditions", []))
    for condition in conditions:
        entry = _condition_entry(condition)
        conds = [c for c in conds if c.get("name") != entry["name"]]
        conds.append(entry)
    d["conditions"] = conds
    save_raw(d, root)
    return d


def remove_condition(name: str, condition_name: str, root: str | None = None) -> dict:
    d = load_raw(name, root)
    d["conditions"] = [c for c in d.get("conditions", []) if c.get("name") != condition_name]
    # THE intentional clear-all flow on this engine: removing the last named
    # condition of a concept-less draft is a researcher deleting one arm they
    # named, one call at a time — declared intent, not a document arriving
    # from somewhere stale. (open-issues §8)
    save_raw(d, root, clearing_arms=True)
    return d


def replace_draft_manifest(name: str, document: object,
                           root: str | None = None) -> dict:
    """One-click server-draft sync (2026-07-21 incident, part 3): install the
    caller's manifest document as this server's DRAFT copy of ``name``.

    The remote-freeze identity check refuses to freeze when the server's
    same-named copy is not the manifest on screen — this is the remedy the
    app offers ("Update the server's copy"). Draft manifests only, on both
    sides of the wire:

    - a FROZEN server copy refuses (same firewall as bundle import's
      frozen-manifest protection — duplicate to iterate, never overwrite);
    - a non-draft INCOMING document refuses: frozen status is stamped by a
      freeze authority (``POST /api/authoring/{name}/freeze``), never
      installed by upload.

    **Merge semantics for server-side auto-pins (2026-08-06 incident):** a
    server-executed verb may have written into the server's copy state the
    Mac has not adopted home yet — the ``SweepConditionAdoption`` family:
    the auto-pinned ``modelRevision`` (``_pin_model_revision``), the
    capability-battery pin pair, and sweep-projected ``<concept>-recommended``
    conditions. A push whose document merely OMITS one of those must not
    strip it: the omission means "the Mac never saw it", not "remove it" —
    stripping the revision pin is exactly what made a healthy chain's
    resume refuse (the 2026-08-06 replication run). Preserved fields are merged
    into the stored document and NAMED in the response (``preserved``) so
    the caller can adopt them into its own draft; an EXPLICIT ``null``
    (key present) still clears — deliberate unpinning stays possible. A
    same-named incoming condition always wins (the workspace is authority
    over content it has seen).

    A missing server copy is created — syncing an unpaired server's first
    draft is the same affordance. Returns the stored status plus this
    engine's canonical body hash (sha256 of the sorted-key compact JSON of
    the MERGED document, informational: the app re-fetches and compares
    documents itself)."""
    if not isinstance(document, dict) or not document:
        raise ExperimentStoreError("manifest body must be a JSON object")
    if document.get("name") != name:
        raise ExperimentStoreError(
            f"manifest body names {document.get('name')!r} but the route "
            f"names '{name}' — refusing an ambiguous sync")
    if document.get("status") != "draft":
        raise ExperimentStoreError(
            "only a DRAFT manifest can be pushed as the server's copy — "
            "frozen manifests are stamped by the server's own gated freeze, "
            "never installed by upload (duplicate to iterate)")
    path = _path(name, root)
    existing = None
    if os.path.exists(path):
        try:
            existing = load_raw(name, root)
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "frozen":
            raise ExperimentStoreError(
                f"refusing to overwrite frozen manifest '{name}' with a "
                "pushed draft (freeze firewall) — duplicate to iterate")
    document = dict(document)
    preserved = _merge_server_pins(document, existing)
    save_raw(document, root)
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {
        "name": name,
        "status": document.get("status"),
        "canonicalBodyHash": hashlib.sha256(canonical).hexdigest(),
    }
    if preserved:
        result["preserved"] = preserved
    return result


def adopt_evidence_revision(run_directory: str,
                            root: str | None = None) -> dict:
    """Python twin of the Mac's ``EvidenceRevisionAdoption`` (2026-08-06):
    reconcile an imported run's manifest-snapshot ``modelRevision`` with the
    same-named LOCAL experiment. The raw ``bundles.import_bundle`` path used
    to skip this reconciliation entirely — that run's manual
    recovery then had ``analyze`` refuse on the auto-pinned revision's epoch
    diff, exactly the confusion adoption exists to prevent.

    Same rules as the Swift enum: adopting completes the researcher's own
    declared intent (draft, unpinned, same model → pin the evidence's
    revision); a CONFLICTING pin or model mismatch is reported loudly and
    never overwritten; frozen manifests are immutable. Returns
    ``{"outcome": ..., ...detail}`` with the Swift outcome vocabulary."""
    try:
        with open(os.path.join(run_directory, "experiment.json"),
                  encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"outcome": "noEvidenceRevision"}
    if not isinstance(snapshot, dict):
        return {"outcome": "noEvidenceRevision"}
    name = snapshot.get("name")
    revision = str(snapshot.get("modelRevision") or "").strip()
    if not (isinstance(name, str) and name and revision):
        return {"outcome": "noEvidenceRevision"}
    try:
        local = load_raw(name, root)
    except (OSError, ValueError):
        return {"outcome": "noLocalExperiment", "experiment": name}
    if local.get("modelID") != snapshot.get("modelID"):
        return {"outcome": "modelMismatch", "experiment": name,
                "localModel": local.get("modelID"),
                "evidenceModel": snapshot.get("modelID")}
    existing = str(local.get("modelRevision") or "").strip()
    if existing:
        if existing == revision:
            return {"outcome": "alreadyPinned", "experiment": name,
                    "revision": revision}
        return {"outcome": "conflict", "experiment": name,
                "localRevision": existing, "evidenceRevision": revision}
    if local.get("status") != "draft":
        return {"outcome": "notADraft", "experiment": name}
    local["modelRevision"] = revision
    try:
        save_raw(local, root)
    except (OSError, ExperimentStoreError) as exc:
        return {"outcome": "saveFailed", "experiment": name,
                "message": str(exc)}
    return {"outcome": "adopted", "experiment": name, "revision": revision}


def _merge_server_pins(document: dict, existing: object) -> dict:
    """Merge server-side auto-pins the incoming document OMITS (key absent)
    into ``document`` in place; returns the ``preserved`` report (empty =
    nothing merged). Explicit ``null`` keys are the caller clearing a pin
    on purpose and are honored."""
    if not isinstance(existing, dict):
        return {}
    preserved: dict = {}
    if existing.get("modelRevision") and "modelRevision" not in document:
        document["modelRevision"] = existing["modelRevision"]
        preserved["modelRevision"] = existing["modelRevision"]
    if (existing.get("capabilityBatteryFile")
            and existing.get("capabilityBatteryHash")
            and "capabilityBatteryFile" not in document
            and "capabilityBatteryHash" not in document):
        document["capabilityBatteryFile"] = existing["capabilityBatteryFile"]
        document["capabilityBatteryHash"] = existing["capabilityBatteryHash"]
        preserved["capabilityBattery"] = {
            "file": existing["capabilityBatteryFile"],
            "hash": existing["capabilityBatteryHash"]}
    incoming_conditions = document.get("conditions")
    if incoming_conditions is not None \
            and not isinstance(incoming_conditions, list):
        return preserved  # malformed conditions: let validation refuse it
    incoming_names = {c.get("name") for c in incoming_conditions or []
                      if isinstance(c, dict)}
    restored: list[str] = []
    for condition in existing.get("conditions") or []:
        if not (isinstance(condition, dict)
                and isinstance(condition.get("selection"), dict)
                and str(condition.get("name") or "").endswith("-recommended")
                and condition.get("name") not in incoming_names):
            continue
        document.setdefault("conditions", []).append(condition)
        restored.append(condition["name"])
    if restored:
        preserved["conditions"] = restored
    return preserved


def duplicate(name: str, new_name: str, root: str | None = None) -> dict:
    source = load_raw(name, root)
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in new_name.lower()).strip("-")
    if not safe:
        raise ExperimentStoreError(f"invalid name {new_name!r}")
    if os.path.exists(_path(safe, root)):
        raise ExperimentStoreError(f"experiment '{safe}' already exists")
    copy = dict(source)
    copy["name"] = safe
    copy["status"] = "draft"
    copy["createdAt"] = _now()
    for key in ("frozenAt", "freezeHash", "gitCommit", "frozenBy", "appVersion",
                "freezeForced", "forcedGatesSkipped"):
        copy.pop(key, None)
    save_raw(copy, root)
    return copy


def _complete_validate_runs(scope_hash: str, root: str | None):
    """Yield ``(run_directory, evidence)`` for every COMPLETE validate run
    whose scope matches, newest first, REGARDLESS of substrate:
    ``validation-evidence.json`` must name task "validate" (schema 1) with the
    matching scope hash, AND a non-empty validation report must exist. A bare
    hash file is not enough — the run must actually have produced validation
    content.

    Matching is SCOPE-based, not name-based (parity with Swift): a duplicated
    experiment with identical pins inherits the original's validate evidence —
    the scope hash, not the experiment name, is what the evidence certifies.
    The same-substrate gate stays in ``_matching_validate_evidence``; the raw
    scan also feeds the cross-substrate advisory."""
    from . import run_status
    runs = paths.runs_directory(root)
    if not os.path.isdir(runs):
        return
    for entry in sorted(os.listdir(runs), reverse=True):
        if "-validate" not in entry:
            continue
        rundir = os.path.join(runs, entry)
        # A run whose own status says it did not complete is never evidence
        # (retention 2026-07-24). Belt-and-braces today — a partial validate
        # run has no validation-evidence.json — but partial run directories
        # now legitimately exist in runs/, and a gate that refuses them only
        # by accident is one refactor away from accepting one.
        if run_status.is_partial(rundir):
            continue
        try:
            with open(os.path.join(rundir, "validation-evidence.json"), encoding="utf-8") as h:
                evidence = json.load(h)
        except (OSError, json.JSONDecodeError):
            continue
        if (evidence.get("schemaVersion") != 1 or evidence.get("task") != "validate"
                or evidence.get("validationScopeHash") != scope_hash):
            continue
        report_file = evidence.get("reportFile") or "validation-report.json"
        try:
            with open(os.path.join(rundir, report_file), encoding="utf-8") as h:
                report = json.load(h)
        except (OSError, json.JSONDecodeError):
            continue
        if "concepts" in report:  # actual validation content, not an empty stub
            yield rundir, evidence


def _matching_validate_evidence(scope_hash: str, root: str | None) -> dict | None:
    """The evidence dict of the newest COMPLETE, SCOPE-matched validate run
    this engine may rely on, else None.

    Same-substrate rule (explicit, not accidental): evidence stamped with a
    different ``substrate`` never counts — CUDA/HF activations do not match
    MLX/Metal, so vectors must be validated on the engine that freezes/runs
    the study. Evidence without the stamp is legacy from THIS engine (the
    historical filename divergence made cross-engine evidence impossible).
    Newest run wins so re-validation supersedes stale evidence."""
    for _rundir, evidence in _complete_validate_runs(scope_hash, root):
        substrate = evidence.get("substrate")
        if substrate is None or substrate == _THIS_SUBSTRATE:
            return evidence
    return None


def vacuous_validate_evidence_problem(name: str, manifest: Manifest,
                                      evidence: dict | None) -> str | None:
    """The ``validateEvidence`` gate's refusal text when the matching validate
    run scored NO held-out probe for one or more pinned concepts — VACUOUS
    evidence, not validation (2026-08-17 firewall repair).

    A ``validate`` run for a concept with no ``prompts/concepts/<c>/
    validation.jsonl`` — the DEFAULT state, since workspace seeding creates
    none — was a no-op that exited 0 and satisfied this gate, so an unforced,
    unstamped freeze was indistinguishable from a validated one while
    ``data check`` called the same missing file a blocker. ``validate`` now
    stamps ``vacuousConcepts``; this reads the stamp back.

    Returns None when there is no evidence (the plain missing-evidence gate
    covers that), when the evidence predates the stamp (LEGACY runs keep
    passing, by design — only newly vacuous ones stop), or when every
    eligible concept was probed. Swift twin:
    ``ExperimentStore.vacuousValidationEvidenceProblem``."""
    if not evidence:
        return None
    stamped = evidence.get("vacuousConcepts")
    if not isinstance(stamped, list) or not stamped:
        return None
    from .manifest import held_out_probe_relpath
    by_name = {c.name: c for c in manifest.concepts}
    # Evidence is matched by SCOPE, so only concepts this manifest still pins
    # matter (a duplicate that dropped one inherits the evidence).
    vacuous = sorted(str(c) for c in stamped if str(c) in by_name)
    if not vacuous:
        return None
    paths_ = [p for p in (held_out_probe_relpath(by_name[c]) for c in vacuous)
              if p]
    return (
        f"cannot freeze '{name}': the matching validate run scored NO held-out "
        f"probe for concept(s) {', '.join(vacuous)} — it is VACUOUS evidence, "
        "not validation. Author the never-named scenarios "
        f"({', '.join(paths_)}) as {{\"text\": …, \"expresses\": true|false}} "
        f"rows and re-run 'steerlab-server experiment validate {name}', or "
        "force-freeze to record an unvalidated experiment")


def _validate_evidence_engine(rundir: str, evidence: dict) -> str | None:
    """Best-known engine of a validate run: the evidence's own ``substrate``
    stamp, else the run's canonical ``config.json`` stamp (covers evidence
    written before the substrate field existed), else None (unknowable —
    pre-stamp runs from either engine)."""
    engine = evidence.get("substrate")
    if isinstance(engine, str) and engine:
        return engine
    try:
        with open(os.path.join(rundir, "config.json"), encoding="utf-8") as h:
            config = json.load(h)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("substrate")
    return value if isinstance(value, str) and value else None


def cross_substrate_validation_advisory(scope_hash: str,
                                        root: str | None = None) -> str | None:
    """WS7.1 non-blocking advisory: scope-matched validate evidence exists,
    but the best evidence this engine can name came from the OTHER engine —
    the study should re-validate on THIS substrate before its runs count.

    Returns the advisory string, or None when (a) the evidence this engine's
    gate relies on is same-engine (or its engine is unknowable — legacy runs
    are never accused), or (b) no scope-matched evidence exists at all (the
    freeze gate's plain missing-evidence message already covers that)."""
    runs = list(_complete_validate_runs(scope_hash, root))
    for rundir, evidence in runs:
        substrate = evidence.get("substrate")
        if substrate is None or substrate == _THIS_SUBSTRATE:
            # The run the gate would rely on. Its config.json can still
            # reveal a foreign engine behind a legacy unstamped evidence file.
            engine = _validate_evidence_engine(rundir, evidence)
            if engine is not None and engine != _THIS_SUBSTRATE:
                return _cross_substrate_message(engine)
            return None
    for rundir, evidence in runs:
        engine = _validate_evidence_engine(rundir, evidence)
        if engine is not None and engine != _THIS_SUBSTRATE:
            return _cross_substrate_message(engine)
    return None


def _cross_substrate_message(engine: str) -> str:
    return (f"validation evidence was produced on {engine}; runs on "
            f"{_THIS_SUBSTRATE} should re-validate on-substrate")


def judging_custody_advisory(d: dict) -> str | None:
    """Plain-language note about WHERE this panel's judging would run on
    this host, or None when there is nothing notable to say.

    Silent on the two unsurprising cases — a purely local panel, and an
    inline panel on a credentialed host — because an advisory that fires
    on every study is one nobody reads. It speaks up for `deferred` and
    `refused`, which are the states a researcher would otherwise discover
    only from the artifacts.

    Evaluated against the CURRENT host, so the same manifest can honestly
    say different things on a Mac and on a cluster node. Cross-engine twin:
    `ExperimentStore.judgingCustodyAdvisory`.
    """
    from . import judging_custody
    roster = judging_custody.roster_from_judge_entries(d.get("judges"))
    if not roster:
        return None
    plan = judging_custody.custody_plan(roster)
    if plan["disposition"] in ("local", "inline"):
        return None
    if plan["disposition"] == "refused":
        return (f"judging would REFUSE on this host: {plan['reason']}. "
                "Pin an all-local or all-external panel, or place a judge "
                "key here.")
    return (
        f"judging would DEFER to the Mac on this host: {plan['reason']}. "
        "That is a legitimate design (keyless is the default posture) — "
        "but if you expected inline judging, note the key file holds ONE "
        "kind, so a mixed panel needs a matching credential for every "
        "external judge.")


def optvec_pinned_concepts(d: dict) -> list[tuple[str, dict]]:
    """``[(concept name, vectorArtifact block)]`` for every concept pinned to
    an OptVec artifact. Pure manifest reading (no filesystem, no torch), so
    the freeze gates and advisories can both ask it."""
    out: list[tuple[str, dict]] = []
    for concept in d.get("concepts") or []:
        if not isinstance(concept, dict) or not concept.get("name"):
            continue
        if ((concept.get("options") or {}).get("method")
                != ExtractionMethod.PINNED_ARTIFACT.value):
            continue
        block = concept.get("vectorArtifact")
        if isinstance(block, dict) and block.get("sourceMethod") == _OPTVEC_METHOD:
            out.append((str(concept["name"]), block))
    return out


def optvec_exempt_from_validate_gate(d: dict) -> bool:
    """Whether the validate-evidence freeze gate has nothing to ask of this
    manifest because every concept it declares is an OptVec direction.

    THE RULE (OptVec plan §6, decided here): an optvec concept has nothing to
    validate. ``validate`` scores a held-out probe against the recipe's class
    means; an optvec vector has no stimuli, no classes and no
    validation.jsonl, so there is no probe to run and a validate run could
    never exist — gating on one would make an optvec confirm study
    freezable only under ``--force``, i.e. permanently non-citable, which
    would be a stamp about the FIREWALL rather than about the science. The
    evidence that certifies the direction is the OptVec eval run's
    ``eval.json`` (test split, untouched by gradients and by checkpoint
    selection), surfaced by :func:`freeze_advisories`.

    Deliberately narrow. It applies only when concepts exist and EVERY one is
    optvec-pinned: a mixed study still owes a validate run for its ordinary
    concepts, and a variant study still owes per-condition battery evidence
    (which is joined to the validate run), so both keep the gate.
    """
    concepts = [c for c in (d.get("concepts") or [])
                if isinstance(c, dict) and c.get("name")]
    if not concepts or d.get("variantConditions"):
        return False
    return len(optvec_pinned_concepts(d)) == len(concepts)


def _adapter_variant_advisories(d: dict, root: str | None = None) -> list[str]:
    """Non-blocking advisories about ADAPTER variant conditions (Swift twin
    ``ExperimentStore.adapterVariantAdvisories``):

    * an EXPLORATORY adapter — v1 sidecar, no sidecar, or a v2 sidecar that
      does not claim evidence grade. Legal (pilots, robustness arms), but it
      cannot carry a citable intervention;
    * an evidence-grade adapter with NO ``matchedControl`` declared. Amendment
      2: every evidence-grade stance adapter trains alongside an S0-analog
      control on an identical schedule, and WHICH neutralization is manifest
      data declared before training. An adapter arm with no declared control
      is an intervention with no counterfactual.

    Advisories must never block or sink a freeze, so every filesystem read
    here is best-effort."""
    advisories: list[str] = []
    for vc in d.get("variantConditions") or []:
        if not isinstance(vc, dict) or isinstance(vc.get("fromPromotion"), dict):
            continue
        artifact = vc.get("artifact") or {}
        if not artifact.get("adapters"):
            continue
        label = vc.get("name", "?")
        try:
            evidence_grade = variant_is_evidence_grade(vc, root)
        except Exception:  # noqa: BLE001 - advisories never block a freeze
            evidence_grade = bool(
                (vc.get("trainingProvenance") or {}).get("evidenceGrade"))
        if not evidence_grade:
            advisories.append(
                f"variant '{label}' uses an EXPLORATORY adapter (no "
                "evidence-grade training provenance: the adapter's sidecar "
                "is pre-v2 or does not claim evidence grade) — fine for "
                "pilots and robustness arms; a citable adapter intervention "
                "must be trained through the evidence-grade path")
            continue
        block = vc.get("trainingProvenance")
        control = block.get("matchedControl") if isinstance(block, dict) else None
        if not isinstance(control, dict) or not control:
            advisories.append(
                f"variant '{label}' pins an evidence-grade adapter with NO "
                "matchedControl declared — an adapter intervention without "
                "its matched (neutralized/shuffled-label) control arm has no "
                "counterfactual; declare "
                "trainingProvenance.matchedControl before training")
    return advisories


def _sae_qualification_advisories(d: dict, root: str | None = None) -> list[str]:
    """Non-blocking advisories about SEATED SAE-feature arms (SAE proposal r2
    §6 / §8 P0-4).

    An agent whose injected vector is a DIRECT-ID Gemma Scope import is a
    steering condition built on a feature chosen for what an auto-interp label
    said it means. That semantic claim is exactly what the qualification
    protocol tests, and the durable record is what the paper can cite. When
    the seated agent's promotion cites no such record, say so at the moment
    the study stops being editable.

    ADVISORY, never a gate — the proposal is explicit that qualification is
    citable evidence, not a seating mechanism: seating stays sweep → promote.
    A pilot arm on an unqualified feature is a legitimate thing to run, and a
    refusal here would make it freezable only under ``--force`` (i.e.
    permanently non-citable for a reason that is about the paperwork).

    Scoped to variant conditions — the SEATS. An SAE artifact merely attached
    as a concept is the sweep's INPUT, and flagging it would fire on every
    correctly-run SAE study before its promotion exists.

    Every filesystem read is best-effort: advisories must never block or sink
    a freeze."""
    from . import sae_qualification

    advisories: list[str] = []
    for vc in d.get("variantConditions") or []:
        if not isinstance(vc, dict) or isinstance(vc.get("fromPromotion"), dict):
            # Forward references resolve at RUN time; there is no artifact to
            # read here, by design.
            continue
        artifact = vc.get("artifact")
        if not isinstance(artifact, dict):
            continue
        promotion = artifact.get("promotion")
        cited = (promotion.get("qualification")
                 if isinstance(promotion, dict) else None)
        for injection in artifact.get("injections") or []:
            if not isinstance(injection, dict):
                continue
            reference = injection.get("vectorArtifactID")
            if not isinstance(reference, str) or not reference:
                continue
            try:
                # A seated agent injects the run's MATERIALIZED COPY of the
                # import, which carries no gemmascopeSource of its own — the
                # resolver follows its hash-pinned pinnedFrom, so an SAE arm
                # does not stop looking like one the moment it is swept.
                source, _origin = sae_qualification.resolved_feature_identity(
                    reference, root)
            except Exception:  # noqa: BLE001 - advisories never block a freeze
                continue
            if not source or isinstance(cited, dict):
                continue
            advisories.append(
                f"variant '{vc.get('name', '?')}' seats a Gemma Scope SAE "
                f"feature (feature {source.get('feature')}, layer "
                f"{source.get('layer')}, {source.get('release')}/"
                f"{source.get('saeID')}) whose promotion cites NO "
                "qualification artifact — the feature's semantic claim rests "
                "on an auto-interp label alone. Record a "
                f"{sae_qualification.FILENAME} (held-out construct probe, "
                "lexical leakage, discriminant controls, coherence/format, "
                "dose response, accept/reject) and re-promote citing it "
                "(promote --qualification …) for a citable arm")
            break  # one advisory per variant, not one per injection
    return advisories


def _validation_lookup_advisories(d: dict, root: str | None = None) -> list[str]:
    """Non-blocking advisories about MISFILED held-out sets (2026-08-19).

    A concept's ``validation.jsonl`` has two possible homes and the RECIPE
    decides which is canonical — paired recipes read ``prompts/concepts/``,
    the grand-mean recipe reads ``prompts/emotions/``. The lookup falls back
    to the other home rather than reading a misfiled set as absent, so the
    hash IS pinned; what the researcher still needs to be told is that the
    file is not where its recipe says it lives (or that it is in both
    places). Swift twin:
    ``ExperimentStore.validationLookupAdvisories``."""
    from .manifest import (Manifest, concept_machinery_operative,
                           resolve_validation_file,
                           validation_lookup_advisory)
    if not concept_machinery_operative(d):
        return []
    advisories: list[str] = []
    for concept in Manifest.from_dict(d).concepts:
        if not concept.effective_method.has_source_concept:
            continue  # nothing to validate — no home, no advisory
        location = resolve_validation_file(
            concept.data_concept,
            paired=not concept.effective_method.uses_story_corpus, root=root)
        advisory = validation_lookup_advisory(concept.name, location)
        if advisory:
            advisories.append(advisory)
    return advisories


def freeze_advisories(d: dict, root: str | None = None) -> list[str]:
    """Non-blocking freeze advisories (parallel to Swift
    ``ExperimentStore.freezeAdvisories``). Never a gate: freeze prints these
    loudly and proceeds/refuses on the GATES alone. Advisories: cross-substrate
    validate evidence (WS7.1); legacy pre-measurement-pin concept attaches; and
    the non-citable marker on a forced freeze."""
    advisories: list[str] = []
    if d.get("concepts") or d.get("conditions") or d.get("variantConditions"):
        try:
            scope = Manifest.from_dict(d).validation_scope_hash()
            advisory = cross_substrate_validation_advisory(scope, root)
        except Exception:  # advisories must never block or sink a freeze
            advisory = None
        if advisory:
            advisories.append(advisory)
    # OptVec-pinned concepts (plan §6): the validate gate does not apply to
    # them (nothing to validate — see optvec_pinned_concepts), so the
    # evidence that DOES certify the direction is named here instead, at the
    # moment of freezing. Loud, never a gate: this is the "validate-equivalent
    # evidence, advisory first" resolution of §6's open design point.
    for concept_name, block in optvec_pinned_concepts(d):
        evidence = block.get("optvecEvalRun")
        training = block.get("optvecTrainingRun")
        trained = f" (trained by run '{training}')" if training else ""
        if evidence and block.get("optvecEvalRunVerified") is False:
            # Named but NOT verifiable at attach: say so instead of
            # describing contents nobody has seen (evidence is only as good
            # as what was checked, and here nothing was).
            reason = block.get("optvecEvalRunUnverifiedReason") or \
                "unverified"
            advisories.append(
                f"concept '{concept_name}' pins an OptVec vector{trained} "
                f"naming eval run '{evidence}', but the reference could NOT "
                f"be verified at attach ({reason}) — treat it as no "
                "verifiable eval evidence: confirm the run completed and "
                "certifies this artifact's tensor hash, then re-attach so "
                "the citation is checked")
        elif evidence:
            verified = (" — verified at attach: its eval.json certifies "
                        "this artifact's tensor hash"
                        if block.get("optvecEvalRunVerified") is True else
                        " — recorded before attach-time verification; the "
                        "reference was never checked against this artifact "
                        "(re-attach to verify)")
            advisories.append(
                f"concept '{concept_name}' pins an OptVec vector{trained}: it "
                "has no stimuli and no held-out validation.jsonl, so the "
                "validate gate does not apply. Its validate-equivalent "
                f"evidence is the OptVec eval run '{evidence}' (test-split "
                "shift, anchor drift, capability, fluency in eval.json) — "
                "cite that, and the confirm study's own conditions, never "
                f"the training run's val split{verified}")
        else:
            advisories.append(
                f"concept '{concept_name}' pins an OptVec vector{trained} "
                "with NO eval evidence recorded: the validate gate does not "
                "apply to an optvec concept (no stimuli, no held-out set), "
                "and nothing in the artifact names an eval run either — so "
                "this freeze rests on the training run alone, whose val "
                "split is selected-on and not citable. Run the OptVec eval "
                "verb on the test split and re-attach so the evidence is "
                "named")
    # Trained-adapter arms (LoRA readiness §0 amendments 1 + 2). Both are
    # non-blocking BY DESIGN: an exploratory adapter is a legitimate pilot
    # arm, and a missing matched control is a design choice the researcher
    # must be able to make loudly rather than be refused for. They fire at
    # the moment of freezing because that is when the study stops being
    # editable.
    advisories += _adapter_variant_advisories(d, root)
    # Misfiled held-out sets (2026-08-19): the dual-root lookup FOUND a
    # concept's validation.jsonl under the other recipe's root, or under
    # both. Non-blocking — the file is read and pinned either way — but
    # freeze is the last moment the filing can still be corrected.
    try:
        advisories += _validation_lookup_advisories(d, root)
    except Exception:  # noqa: BLE001 - advisories never block or sink a freeze
        pass
    # Seated SAE-feature arms with no cited qualification evidence (SAE
    # proposal r2 §6). Non-blocking by design — see the helper.
    try:
        advisories += _sae_qualification_advisories(d, root)
    except Exception:  # noqa: BLE001 - advisories never block or sink a freeze
        pass
    # A study that reads the residual stream but keeps no token ids. Advisory,
    # not a gate: retention changes nothing that is measured, so refusing
    # would be refusing a legitimate design. But it fires HERE, at freeze,
    # because retention is not retroactive — once the run completes without
    # the ids, the only way to get an exactly-replayable record is to run it
    # again.
    if d.get("jlensReadout") and not d.get("recordTokenIDs"):
        advisories.append(
            "this study declares a jlensReadout but not recordTokenIDs, so "
            "the exact sampled sequence will not be kept, so its generations "
            "cannot be replayed faithfully afterwards: retrospective "
            "J-space analysis needs the EXACT sampled ids fed back, and "
            "re-deriving them from the stored text is not a round trip (a "
            "generation that stops naturally ends with <end_of_turn>, the "
            "streamer skips it, and re-tokenizing returns a shorter "
            "sequence). The online trace still records what it was armed "
            "for; what is lost is the ability to go back and read a layer, "
            "token, or position the readout block did not name. Set "
            "recordTokenIDs unless the extra bytes matter")
    # Legacy concept pins (attached before 2026-07-13) carry no
    # validationHash key: verify passes (no violation), but the study should
    # re-attach so the measurement-side inputs enter the drift firewall.
    if any(isinstance(c, dict) and "validationHash" not in c
           for c in d.get("concepts") or []):
        advisories.append(
            "measurement-side inputs unpinned (markers/validation) — "
            "re-attach to pin")
    # Legacy conditions carrying no `alphaInNormUnits` (Phase-0 gap G6). A NEW
    # declaration without the key is refused (`_condition_entry`), but a
    # document that already holds one still freezes — reinterpreting a dose
    # nobody re-declared is exactly what this repair must not do. What it gets
    # instead is the ambiguity said out loud, at the last moment the study can
    # still be edited: this engine reads such a condition as RAW α, and the Mac
    # engine refuses to open the manifest at all, so the study's dose semantics
    # depend on which engine you ask.
    keyless = [str(c.get("name", "?")) for c in d.get("conditions") or []
               if isinstance(c, dict) and "alphaInNormUnits" not in c]
    if keyless:
        advisories.append(
            "condition(s) " + ", ".join(f"'{name}'" for name in keyless)
            + " declare no 'alphaInNormUnits': this engine reads them as RAW "
            "α (the reading they were authored under, kept as it is), and the "
            "Mac engine cannot read the manifest at all until the key is "
            "present. Re-declare the arm to state the units — a frozen study "
            "keeps whatever reading it was measured under, so this is worth "
            "settling BEFORE the freeze, not after. " + ALPHA_UNITS_REPAIR)
    # F1: panel-script authoring problems that fail SILENTLY at run time —
    # duplicate output labels, and {{outputs.X}} references no earlier turn
    # produces. Advisory, not a gate: they make prompts quietly wrong rather
    # than making the study invalid, and the researcher may be mid-iteration.
    if d.get("studyKind") == "multiAgent" and d.get("multiAgentScenarioPath"):
        try:
            from . import multi_agent as _ma
            spath = d["multiAgentScenarioPath"]
            if not os.path.isabs(spath):
                spath = os.path.join(paths.project_root() if root is None else root, spath)
            scenario, _ = _ma.load_scenario(spath)
            advisories.extend(_ma.advisories(scenario))
        except Exception:  # advisories must never block or sink a freeze
            pass
    # A declared chain with NO gates freezes legally but is worth a loud
    # word (stage 5): the frozen object then runs every stage to completion
    # with no scientific stop conditions.
    if isinstance(d.get("pipeline"), dict) and not (
            isinstance(d["pipeline"].get("gates"), dict)
            and d["pipeline"]["gates"]):
        advisories.append(
            "pipeline declares no gates — the chain will run every stage "
            "to completion with no scientific stop conditions; declare "
            "pipeline.gates for evidence-grade chains")
    # Carried non-operative configuration (2026-07-19): preserved by the
    # type picker's never-delete promise, but invisible to this kind's
    # verification, snapshot, and bundle — say so at the moment of
    # freezing instead of letting it read as covered (Swift twin).
    carried: list[str] = []
    if d.get("studyKind") == "multiAgent":
        if d.get("concepts"):
            carried.append(f"{len(d['concepts'])} concept(s)")
        if d.get("conditions"):
            carried.append(f"{len(d['conditions'])} injection condition(s)")
        if d.get("variantConditions"):
            carried.append(f"{len(d['variantConditions'])} agent condition(s)")
        if d.get("taskPromptsFile"):
            carried.append("a task-prompts pin")
    else:
        if d.get("multiAgentScenarioPath"):
            carried.append("a pinned multi-agent scenario")
        from .manifest import concept_machinery_operative
        if not concept_machinery_operative(d):
            if d.get("concepts"):
                carried.append(f"{len(d['concepts'])} concept(s)")
            if d.get("conditions"):
                carried.append(
                    f"{len(d['conditions'])} injection condition(s)")
    if carried:
        advisories.append(
            "carries configuration for another study type ("
            + ", ".join(carried)
            + ") — preserved, but NOT verified, snapshotted, or bundled "
            "for this study kind; switch the study type (and duplicate) "
            "to use it")
    # An indistinct judge panel (finding 4) is a freeze GATE (judgeValidity);
    # surfaced here too so a DRAFT shows the problem before the freeze
    # attempt refuses.
    indistinct = judge_panel_indistinct_problem(d)
    if indistinct:
        advisories.append(indistinct)
    # A judged SWEEP whose local judge cannot load inside the chain
    # (finding 1, live incident 2026-07-22) is a freeze GATE (judgeValidity);
    # surfaced here too so a DRAFT shows the problem before freeze refuses.
    # A foreign local judge with no revision/dtype pin is a freeze GATE
    # (judgeValidity); surfaced here so a DRAFT shows it before the freeze
    # attempt refuses.
    unpinned_judge = unpinned_foreign_local_judge_problem(d)
    if unpinned_judge:
        advisories.append(unpinned_judge)
    pipeline_judge_problem = local_judge_pipeline_problem(d)
    if pipeline_judge_problem:
        advisories.append(pipeline_judge_problem)
    # An evaluate stage with such judges ROUTES to the post-generation
    # judge fan-out (2026-07-23) — informational, never a gate.
    fanout_note = local_judge_fanout_note(d)
    if fanout_note:
        advisories.append(fanout_note)
    # WHERE this panel's judging will happen (2026-07-24). Informational,
    # never a gate — deferring to the Mac is a legitimate design, and the
    # keyless cluster is the DEFAULT custody posture. It is surfaced here
    # because the fork was previously invisible until after the fact, and
    # the surprising case (a mixed panel deferring wholesale despite a
    # deliberately pushed key) is exactly the one worth seeing before a
    # study is frozen rather than after a run.
    custody = judging_custody_advisory(d)
    if custody:
        advisories.append(custody)
    advisories.extend(
        _adapter_config_pin_advisories(d.get("variantConditions"), root))
    if d.get("freezeForced"):
        skipped = ", ".join(d.get("forcedGatesSkipped") or []) or "none"
        advisories.append(
            f"forced freeze — gates skipped: {skipped} — non-citable")
    return advisories


def _adapter_config_pin_advisories(variant_configs, root=None) -> list:
    """`adapter_config.json` pinned? — advisory, deliberately not a gate.

    Weights pinned and config unpinned means the config can be edited —
    rank, target modules, scaling, WHICH layers the adapter even touches —
    while the agent's declared identity stays byte-identical. That is a real
    hole (external review round 6).

    It is an advisory and not a freeze refusal because every agent minted
    before 2026-08-16 lacks the field, the value is RECOVERABLE from bytes
    already on disk, and refusing would push whole studies onto `--force`,
    which stamps them non-citable — a worse outcome than a loud advisory for
    a repairable gap. The enforcement that bites is downstream: a J-lens
    report over an unpinned agent is DOWNGRADED off `qualified`
    (`jlens/probe.py`), because that is where the configuration being
    unverified actually changes what may be claimed.
    """
    advisories = []
    for vc in variant_configs or []:
        artifact = vc.get("artifact") or {}
        for adapter in artifact.get("adapters") or []:
            if adapter.get("configHash"):
                continue
            advisories.append(
                f"variant '{vc.get('name', '?')}' adapter "
                f"'{adapter.get('name') or adapter.get('adapterDirectory', '?')}' "
                f"pins its WEIGHTS but not its adapter_config.json (no "
                f"configHash) — the configuration can change without changing "
                f"the agent's declared identity. The hash is computable from the "
                f"adapter directory. Repair: re-save the Agent, then "
                f"RE-ATTACH it to this study — re-saving the library Agent "
                f"alone does not repair a condition already attached, which "
                f"carries its own pinned copy. If the study is FROZEN, "
                f"duplicate it, attach the repaired Agent, verify, and freeze "
                f"again. A J-lens readout over this agent cannot claim "
                f"'qualified'")
    return advisories
#: The CLOSED cross-engine vocabulary of freeze gates (Swift holds the
#: identical literal as ``FreezeGate.vocabulary``; the twin tests are
#: ``test_gate_id_vocabulary_is_closed`` here and
#: ``FreezeGateVocabularyTests.matchesServerLiteral`` there). Since WP0 step 3
#: these ids name a REFUSAL as well as ordering the ``forcedGatesSkipped``
#: stamp — ``ExperimentStoreError.gate``/``.gates`` — and
#: ``forcedGatesSkipped`` may still only ever contain these values.
#: "measurementPins" is the measurement-pin GATE: this engine emits it for an
#: unloadable study dtype and for an unqualified J-lens readout (the doc
#: comment that claimed it was never emitted was stale — audit §2.4). Pin
#: DRIFT remains a verify() violation on both engines, never skippable, and
#: unpinned legacy inputs remain a non-blocking advisory.
FORCED_GATE_IDS = ("revision", "validateEvidence", "batteryEvidence",
                   "judgeValidity", "variantValidity", "gitClean",
                   "measurementPins")


def _check_jlens_readout(name: str, d: dict, root: str | None) -> None:
    """A frozen study's J-lens readout must be fully pinned and QUALIFIED.

    The readout decides what gets measured, so it belongs to the
    ``measurementPins`` gate. What it must establish before a study can cite it:

    * the lens artifact, by id AND content hash — a re-import from a different
      upstream commit keeps the id and changes every number;
    * a QUALIFICATION that passes for this study's exact model, revision, and
      dtype. Geometry cannot see dtype, so an unqualified or differently-typed
      runtime would read the residual through a Jacobian never validated
      against it;
    * an EVIDENCE-tier model. A testing-tier lens exercises the path and can
      never produce evidence, so freezing a study on one would mint a citable
      artifact from a tier that is defined as non-citable;
    * the declared choices and conventions, via the configuration hash, so a
      post-hoc change to layers or watchlist is a verify violation rather than
      a convenience.
    """
    block = d.get("jlensReadout")
    if not block:
        return
    if not isinstance(block, dict):
        # A present-but-malformed block used to reach `.get` on a string and
        # raise AttributeError — an internal exception where a named refusal
        # belongs (external review round 5).
        raise ExperimentStoreError(
            f"cannot freeze '{name}': jlensReadout is present but is a "
            f"{type(block).__name__}, not an object — a readout block is a "
            f"JSON object of pinned declarations")
    from ..jlens import importer as jlens_importer
    from ..jlens import lens_store as jlens_store

    # `qualificationID` is a FROZEN-study pin (external review round 3). An
    # unpinned study resolves whichever qualification is newest, so appending
    # one later silently changes which acceptance a frozen study rests on —
    # the exact-pin resolution added in round 2 only helps when the field is
    # actually there. Exploratory (unfrozen) runs may still omit it: run start
    # resolves newest and warns.
    missing = [k for k in ("lensID", "lensSHA256", "layers", "configHash",
                          "tokenizerHash", "qualificationID")
               if not block.get(k)]
    if missing:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': jlensReadout is missing {missing} — a "
            f"readout that is not fully pinned cannot be reproduced. "
            f"'qualificationID' names the exact acceptance this study rests "
            f"on; without it a later re-qualification moves the ground under "
            f"an already-frozen study")
    if not block.get("watchlist") and not block.get("topK"):
        raise ExperimentStoreError(
            f"cannot freeze '{name}': jlensReadout declares neither a token "
            f"watchlist nor a top-k width, so it would record nothing")

    # The readout rides on GENERATION. A deterministic choice/logprob-only
    # study runs none, so an armed readout there records nothing and closes
    # with an empty trace (external review round 3). Caught at freeze as well
    # as at run start, because freeze is where the study stops being editable.
    from . import execution_plan

    if not execution_plan.resolve(d.get("outcomeInstruments")).generates_sampled_text:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': jlensReadout is declared but this "
            f"study's outcome instruments generate no sampled text "
            f"({d.get('outcomeInstruments')}) — the readout observes the "
            f"generation path, so it would record nothing. Add a "
            f"sampled-text instrument, or drop the readout block")

    model_id = d.get("modelID")
    entry = jlens_importer.SUPPORTED.get(model_id or "")
    if entry is None:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': jlensReadout is declared but "
            f"'{model_id}' has no supported lens — this feature is Gemma-only")
    if entry.get("tier") != "evidence":
        raise ExperimentStoreError(
            f"cannot freeze '{name}': '{model_id}' is a {entry.get('tier')}-tier "
            f"model for J-lens work. It exercises the path and produces no "
            f"evidence, so a frozen study cannot select it")

    try:
        record = jlens_store.resolve(block["lensID"], root)
    except Exception as exc:  # noqa: BLE001
        raise ExperimentStoreError(
            f"cannot freeze '{name}': pinned J-lens '{block['lensID']}' is not "
            f"imported in this workspace ({type(exc).__name__})") from exc

    revision, dtype = d.get("modelRevision"), d.get("dtype")
    if not revision or not dtype:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': jlensReadout needs a pinned model "
            f"revision AND dtype — geometry cannot see dtype, so a "
            f"qualification is only meaningful against both")
    # The layers the study will ARM, and its explicit pin when it has one.
    # Without the layers a qualification that exercised three mid-stack layers
    # licensed a study arming any layer; without the pin, re-qualifying a
    # runtime broke every frozen study that named an older still-valid record
    # (external review round 2).
    armed_layers = [int(x) for x in (block.get("layers") or [])]
    qualification = record.qualification_for(
        model_id, revision, dtype, d.get("quantization"),
        layers=armed_layers,
        qualification_id=block.get("qualificationID") or None)
    if qualification is None:
        # Stage 4 landed 2026-08-15, so the refusal names the real verb again.
        # The GATE is unchanged — it always demanded a passing qualification
        # for the exact runtime — and what changed is only that the evidence
        # it demands is now producible. Between 2026-07-31 and that date the
        # message said honestly that no verb existed rather than directing the
        # researcher at one that did not; a test pins this wording, so the
        # message and the test move together.
        pinned = block.get("qualificationID")
        raise ExperimentStoreError(
            f"cannot freeze '{name}': J-lens '{block['lensID']}' has no passing "
            f"qualification for {model_id}@{revision[:12]}…/{dtype}"
            + (f" matching the pinned id {pinned!r}" if pinned else "")
            + (f" covering layers {armed_layers}" if armed_layers else "")
            + ". A qualification must be bound to the lens bytes it was "
              "measured against and must cover the layers being armed. "
            f"Qualify this exact runtime first — 'steerlab-server jlens "
            f"qualify {block['lensID']} {model_id} --revision {revision[:12]}…' "
            f"(it needs the model resident, so it is a GPU job) — or "
            f"force-freeze, which stamps the study non-citable")



def model_output_surfaces_operative(d: dict) -> bool:
    """Whether the MODEL-OUTPUT freeze surfaces apply to this manifest.

    The one decision shared by readiness and freeze, so they cannot give
    opposite answers about the same manifest (external review round 13).

    A multi-agent study runs a SCENARIO. Under the app's never-delete rule it
    may carry concepts, injection conditions, agents, and a J-lens readout
    from before a kind switch — none of which it executes. `freeze_advisories`
    already tells the researcher that carried configuration is "preserved, but
    NOT verified, snapshotted, or bundled for this study kind"; the gates then
    verified it anyway and refused the freeze. Both statements came out of the
    same function.

    So carried model-output state may ADVISE, and may not block. What still
    applies to every kind — a panel loads a model and is judged like any other
    study — is deliberately outside this: pinned revision, loadable dtype,
    judge validity, and git cleanliness.
    """
    return (d.get("studyKind") or "modelOutput") == "modelOutput"


def _evaluate_freeze_gates(name: str, d: dict, manifest: Manifest,
                           root: str | None) -> list[tuple[str, str]]:
    """Evaluate EVERY freeze gate, returning ``[(gate_id, message)]`` for each
    that would refuse — in the historical refusal order, so a non-force freeze
    raises the same first error it always did, and a forced freeze can name
    everything it skipped. Read-only: no gate mutates the manifest."""
    failures: list[tuple[str, str]] = []
    if not d.get("modelRevision"):
        failures.append((
            "revision",
            f"cannot freeze '{name}': model revision not pinned and "
            f"{d['modelID']} not in the local HF cache — load it once or "
            "freeze --force"))
    # `measurementPins` was RESERVED in the cross-engine gate vocabulary for
    # exactly this: an input that determines what gets measured rather than
    # what gets loaded. A study dtype is the first one to use it.
    symbolic = symbolic_revision_problem(d)
    if symbolic:
        failures.append((
            "revision", f"cannot freeze '{name}': {symbolic}"))
    bad_dtype = unloadable_study_dtype_problem(d)
    if bad_dtype:
        failures.append(("measurementPins", f"cannot freeze '{name}': {bad_dtype}"))
    # Model-output surfaces: gated on the study kind that USES them. A panel
    # carrying these from a kind switch executes none of them, and refusing
    # its freeze over them contradicts the advisory this same module emits.
    operative = model_output_surfaces_operative(d)
    if operative:
        try:
            _check_jlens_readout(name, d, root)
        except ExperimentStoreError as exc:
            failures.append(("measurementPins", str(exc)))
        try:
            _check_variant_validity(name, d, root)
        except ExperimentStoreError as exc:
            failures.append(("variantValidity", str(exc)))
    # Judge validity is NOT model-output-only: a panel's turns are flattened
    # into generations and judged like any other study's output.
    try:
        _check_judged_evaluation(name, d)
    except ExperimentStoreError as exc:
        failures.append(("judgeValidity", str(exc)))
    evidence = None
    if (operative
            and (d.get("concepts") or d.get("conditions")
                 or d.get("variantConditions"))
            and not optvec_exempt_from_validate_gate(d)):
        evidence = _matching_validate_evidence(manifest.validation_scope_hash(), root)
        if evidence is None:
            failures.append((
                "validateEvidence",
                f"cannot freeze '{name}': no validate run matches its exact "
                f"pins — run 'steerlab-server experiment validate {name}' "
                "first, or force-freeze"))
        else:
            # Evidence EXISTS but probed nothing: same gate id, a remedy
            # naming the missing files (2026-08-17).
            vacuous = vacuous_validate_evidence_problem(name, manifest, evidence)
            if vacuous:
                failures.append(("validateEvidence", vacuous))
    if operative and d.get("variantConditions"):
        try:
            _check_battery_evidence(name, d, evidence, root)
        except ExperimentStoreError as exc:
            failures.append(("batteryEvidence", str(exc)))
    try:
        _check_git_pin_cleanliness(name, d, root)
    except ExperimentStoreError as exc:
        failures.append(("gitClean", str(exc)))
    return failures


def _freeze_gate_repair(gate: str, name: str) -> str:
    """The RUNNABLE repair for a freeze-gate refusal on THIS engine.

    Gate-5 dry run #2 (P2/P3): every freeze-gate refusal here carried one
    boilerplate string — "satisfy the named gate, or freeze --force to record
    an explicitly non-citable experiment" — which names no command, and whose
    only concrete token (`freeze --force`) is a verb this CLI does not have.

    Each repair below names the engine that can actually satisfy its gate.
    The evidence gates name THIS one: ``_matching_validate_evidence`` accepts
    only evidence stamped ``python-hf-transformers`` and there is no
    run-substrate seam here, so evidence from the other engine can never
    satisfy them. Everything else is an authoring act, which is Mac-authority.
    """
    freeze_again = (f"steerlab-cli experiment freeze {name}  "
                    "(authoring is Mac-authority)")
    repairs = {
        "revision": f"steerlab-cli experiment create {name} --model <id> "
                    "--revision <commit> on the Mac, or load the model once "
                    f"here so it is cached, then {freeze_again}",
        "measurementPins": "repoint the invalid measurement pin at a loadable "
                           f"value on the Mac, then {freeze_again}",
        "validateEvidence": f"steerlab-server experiment validate {name}  "
                            "(this gate reads evidence stamped "
                            "python-hf-transformers; evidence from the other "
                            f"engine will not satisfy it), then {freeze_again}",
        "variantValidity": "re-save the variant with hashed adapter weights "
                           "and re-attach it on the Mac, then "
                           f"{freeze_again}",
        "batteryEvidence": f"steerlab-server experiment validate {name}  "
                           "(each variant condition runs the pinned battery), "
                           f"then {freeze_again}",
        "judgeValidity": f"steerlab-cli experiment pin-rubric {name} "
                         "prompts/rubrics/default-paired-v1.md --judges "
                         f"a:local,b:claude on the Mac, then {freeze_again}",
        "gitClean": "commit the pinned inputs in the workspace git repo, then "
                    f"{freeze_again}",
    }
    return repairs.get(
        gate, f"satisfy the '{gate}' gate, then {freeze_again}")


def freeze(name: str, *, force: bool = False, cached_revision=None,
           root: str | None = None) -> dict:
    """Gate then stamp a draft as frozen (parallel to Swift ``freeze``).

    Gates (unless ``force``): all pinned hashes verify; a model revision is
    pinned (auto-pinned from the HF cache via ``cached_revision`` if absent); and
    a matching ``validate`` run exists when the experiment uses concept vectors.

    ``force`` is LOUD and STAMPED (2026-07-13): every gate is still evaluated,
    each failing gate prints a warning naming what was skipped, and the frozen
    manifest carries ``freezeForced: true`` + ``forcedGatesSkipped``
    (:data:`FORCED_GATE_IDS` vocabulary) so a forced freeze is permanently
    distinguishable — freeze_advisories marks it non-citable. The always-run
    ``verify()`` of the pins is never skippable.

    Lifecycle order after the gates pass (force included — the pinned/
    snapshot and workspace auto-commit no longer diverge under force, matching
    Swift): (1) runs/-resident inputs are copied into pinned/ and repointed
    (``_pin_external_inputs``) and the manifest re-verifies; (2) ALL pinned
    inputs are snapshotted into experiments/<name>/pinned/
    (``_snapshot_pinned_inputs``, no path rewriting); (3) the workspace is
    auto-committed when it is its own git work-tree root
    (``_auto_commit_workspace``); (4) the freeze stamps — status, frozenAt,
    freezeHash, frozenBy, appVersion, gitCommit (now the auto-commit's HEAD)
    — land and the canonical bytes + preregistration are written; (5) a
    follow-up ``freeze <name> (stamp)`` commit (Swift parity) lands the frozen
    manifest / canonical bytes / preregistration so a standalone workspace
    ends CLEAN. gitCommit deliberately stays the step-3 commit — the one that
    contains the pinned bytes it vouches for.
    """
    d = load_raw(name, root)
    if d.get("status") != "draft":
        # Typed since gate-5 dry run #2 (P3): this was the last status guard
        # in the module left untyped, so re-freezing — the commonest possible
        # retry — answered `verbFailed`/70, indistinguishable from a crash,
        # while `save_raw` and `confirmation.attach_perturbations` already
        # said `statusImmutable`/65 with a runnable repair. Prose unchanged.
        raise ExperimentStoreError(
            f"'{name}' is already {d.get('status')}",
            gate=lifecycle_gates.STATUS_IMMUTABLE,
            repair=(f"steerlab-cli experiment duplicate {name} {name}-v2 && "
                    f"steerlab-cli experiment freeze {name}-v2  "
                    "(a frozen study is immutable; the duplicate is a draft "
                    "again, and authoring is Mac-authority)"))
    if not d.get("modelRevision") and cached_revision is not None:
        d["modelRevision"] = cached_revision(d["modelID"])
    # A frozen variant study must NAME the battery it relied on: an unpinned
    # manifest gates against the live default, and the default file can later
    # change with no pin to flag the drift. Pin the default (mirrors Swift and
    # the validate-time auto-pin) BEFORE the scope/freeze hashes are computed,
    # so the frozen manifest is self-describing.
    # Every model-output-only PIN is scoped the same way the gates are: a
    # panel that carries agents or concepts across a kind switch must not have
    # a battery, marker rubric, training provenance or parser registry stamped
    # into its frozen manifest for configuration it never executes (external
    # review round 14). The predicate governs the whole freeze transaction,
    # not just gate evaluation.
    model_output = model_output_surfaces_operative(d)
    if model_output and d.get("variantConditions") and not d.get("capabilityBatteryHash"):
        from . import battery as battery_mod
        digest = battery_mod.live_hash(battery_mod.DEFAULT_BATTERY_FILE, root)
        if digest is not None:
            d.setdefault("capabilityBatteryFile", battery_mod.DEFAULT_BATTERY_FILE)
            d["capabilityBatteryHash"] = digest
    # Local-judge revision pin (cross-engine contract key
    # "judges[].revision", 2026-07-23): a local judge resolving to the STUDY
    # model inherits the study's pinned revision when its own is blank —
    # the judging path then loads exactly the pinned bytes. Stamped BEFORE
    # the freeze hash so the frozen manifest is self-describing.
    _pin_local_judge_revisions(d)
    # Measurement-side pin (cross-engine contract key "markersHash"): freeze
    # is the pin moment for the score-time markers rubric — stamp the
    # aggregate hash of every attached concept's markers.json (null when none
    # exists) BEFORE the freeze hash is computed, so the frozen manifest is
    # self-describing and later drift is a verify() violation.
    from .manifest import markers_aggregate_hash
    # Pin only when the key is ABSENT. A confirmation draft inherits its
    # parent's pin, and re-pinning at freeze would recompute the hash from the
    # CURRENT bytes — so markers.json drifting between the two freezes was
    # silently overwritten and the verify() violation erased by the very act
    # of freezing. Swift has always guarded this (`markersHash == nil`) and
    # says so in a comment; the server did not (external review round 15).
    #
    # `not in` rather than `is None`: an explicitly pinned null means "no
    # markers existed at pin time", and a later-appearing markers.json is a
    # violation, not something to quietly pin now.
    if model_output and "markersHash" not in d:
        d["markersHash"] = markers_aggregate_hash(
            [c.get("name") for c in d.get("concepts") or [] if c.get("name")],
            root)
    # Adapter training-provenance pin (cross-engine contract key
    # "variantConditions[].trainingProvenance", LoRA readiness §0 amendment 1):
    # freeze is the pin moment for a trained adapter's DATASET — stamped from
    # the adapter's own sidecar, BEFORE the freeze hash, so the frozen
    # manifest is self-describing and later drift in the training data is a
    # verify() violation.
    if model_output:
        _pin_training_provenance(d, root)
    # Numeric-parser registry pin (cross-engine contract key
    # "parserRegistryHash"): freeze is the pin moment for the registry the
    # named parser reads — stamped only when absent, BEFORE the freeze hash,
    # so later drift is a verify() violation, never a silent re-pin. A study
    # that names no parser gets no new key (legacy bytes unchanged).
    if model_output and d.get("numericParser") and not d.get("parserRegistryHash"):
        from . import parser_registry
        digest = parser_registry.registry_live_hash(root)
        if digest is not None:
            d["parserRegistryHash"] = digest
    # Sweep-input pins (cross-engine contract keys "sweep.devPromptsHash" +
    # "sweep.batteryHash", firewall closure 2026-07-20): freeze is the pin
    # moment for the files the sweep SELECTS on — stamped only when the key
    # is absent and the file exists, BEFORE the freeze hash, so later drift
    # is a verify() violation, never a silent re-pin. Only an OPERATIVE,
    # declared sweep block gains keys (legacy bytes unchanged elsewhere);
    # paths resolve declared-or-default exactly as the sweep run resolves
    # them. The ex-post provenance hash (selection.devPromptsHash) is
    # unchanged — sweep start refuses on a pin mismatch, so pin and
    # provenance can only agree. A MISSING input file REFUSES the freeze
    # (second pass, 2026-07-20), force included: this is pin-surface
    # integrity — the never-skippable class, like verify() itself — not an
    # evidence gate. Freezing with an absent pin would leave sweep start's
    # legacy-unpinned fallback open to whatever bytes later appear at the
    # path, and no forcedGatesSkipped stamp can neutralize data accepted
    # silently at run time. Legacy manifests already frozen with absent
    # hashes keep verifying clean — only NEW freezes refuse. Carried-inert
    # sweeps (machinery not operative) neither pin nor block, as before.
    from .manifest import (concept_machinery_operative,
                           sweep_choice_pin_entries, sweep_input_pin_surface)
    if isinstance(d.get("sweep"), dict) and concept_machinery_operative(d):
        sweep_block = d["sweep"]
        missing_sweep_inputs: list[str] = []

        def _file_hash(rel: str) -> str | None:
            try:
                with open(paths.resolve(rel, root), "rb") as handle:
                    return hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                return None

        for file_key, hash_key, default, label in sweep_input_pin_surface():
            if hash_key in sweep_block:
                continue  # never silently re-pin
            rel = sweep_block.get(file_key) or default
            digest = _file_hash(rel)
            if digest is not None:
                sweep_block[hash_key] = digest
            else:
                missing_sweep_inputs.append(
                    f"{label} file '{rel}' is missing, so freeze cannot "
                    f"pin it — an operative sweep selects on that file; "
                    f"create the file, or remove/repoint the sweep's "
                    f"{file_key}, before freezing")
        # Choice instruments (review 2026-08-02, P1: the files that
        # determine the WINNING CELL were the one sweep input not pinned at
        # freeze). Same contract: pin-when-absent, never re-pin, a missing
        # file refuses the freeze — force included.
        for concept, rel, pinned, label in sweep_choice_pin_entries(sweep_block):
            if pinned:
                continue
            digest = _file_hash(rel)
            if digest is None:
                missing_sweep_inputs.append(
                    f"{label} file '{rel}' is missing, so freeze cannot "
                    f"pin it — an operative sweep selects on that file; "
                    "create the file, or remove/repoint the declaration, "
                    "before freezing")
                continue
            objective = sweep_block["selection"]["objective"]
            if concept is None:
                objective["choicePromptsHash"] = digest
            else:
                objective.setdefault("choicePromptsHashes", {})[concept] = digest
        if missing_sweep_inputs:
            raise ExperimentStoreError(
                "cannot freeze:\n  - " + "\n  - ".join(missing_sweep_inputs))

    manifest = Manifest.from_dict(d)
    violations = manifest.verify(root)
    if violations:
        raise ExperimentStoreError("cannot freeze:\n  - " + "\n  - ".join(violations))
    # Non-blocking advisories, printed BEFORE the gates so a refusal (e.g.
    # "no validate run matches") still explains why foreign-looking evidence
    # was not counted. Loud, never a refusal (parallel to Swift's
    # freeze-readiness advisories).
    for advisory in freeze_advisories(d, root):
        print(f"freeze '{name}' advisory: {advisory}", file=sys.stderr)
    # Evaluate EVERY gate even under force, so what force skips is known,
    # printed, and stamped — a silent force freeze is indistinguishable from
    # a clean one, which is exactly the non-citability hole being closed.
    gate_failures = _evaluate_freeze_gates(name, d, manifest, root)
    if force:
        for gate_id, message in gate_failures:
            print(f"freeze '{name}' FORCE WARNING: skipping failing gate "
                  f"'{gate_id}' — {message}", file=sys.stderr)
    elif gate_failures:
        # The MESSAGE stays the first failure's, byte-for-byte: a refusal's
        # prose and exit code are unchanged by WP0 step 3. What was
        # unrecoverable without parsing prose — WHICH gate declined, and how
        # many did — now rides the structured fields, so a forced freeze no
        # longer tells the researcher more than a refusal does (§2.4
        # divergence 4).
        raise ExperimentStoreError(
            gate_failures[0][1],
            gate=gate_failures[0][0],
            gates=tuple(g for g in FORCED_GATE_IDS
                        if g in {gid for gid, _ in gate_failures}),
            repair=_freeze_gate_repair(gate_failures[0][0], name))

    # Frozen studies must be self-contained: pinned scenario/variant inputs
    # that live under gitignored runs/ are copied into the experiment
    # directory (byte-identical, hash-checked) so the git-tracked manifest
    # never points at an unversioned file. Happens BEFORE the freeze hash is
    # stamped because it rewrites the pinned paths.
    _pin_external_inputs(name, d, root)
    manifest = Manifest.from_dict(d)
    violations = manifest.verify(root)
    if violations:
        raise ExperimentStoreError(
            "cannot freeze (after pinning inputs):\n  - " + "\n  - ".join(violations))

    # No-git reproducibility floor: snapshot EVERY pinned input into
    # experiments/<name>/pinned/ (verification bytes; the manifest keeps
    # pointing at the canonical prompts/ paths — hashes prove identity),
    # then make freeze = commit + stamp in one gesture when the artifact
    # root is its own git work tree, so the gitCommit stamped below
    # actually CONTAINS the pinned bytes. Runs under force too (2026-07-13):
    # a forced freeze must not ALSO lose its reproducibility floor (Swift
    # keeps both; the engines now agree).
    _snapshot_pinned_inputs(name, d, root)
    _auto_commit_workspace(name, root)

    d["status"] = "frozen"
    d["frozenAt"] = _now()
    d["freezeHash"] = manifest.content_hash()
    d["frozenBy"] = "server"
    d["appVersion"] = engine_version()
    d["gitCommit"] = _git_commit(root)
    if force:
        # Freeze stamps (excluded from the canonical payload, like frozenAt):
        # what force skipped, permanently. An empty list means force was used
        # but every gate would have passed anyway.
        d["freezeForced"] = True
        d["forcedGatesSkipped"] = [gate_id for gate_id, _ in gate_failures]
    save_raw(d, root, freeze_transition=True)
    _write_freeze_canonical(name, d, root)
    _write_preregistration(d, root)
    _auto_commit_workspace(name, root,
                           message=f"freeze {name} (stamp)", quiet=True)
    return d


def _write_freeze_canonical(name: str, d: dict, root: str | None) -> None:
    """Persist the exact canonical bytes the freeze hash was computed over.

    Swift cannot reproduce Python's json.dumps canonicalization byte-for-byte,
    so post-freeze verification of a SERVER-frozen manifest on the Swift side
    works from this file instead of skipping: sha256(bytes) must equal
    freezeHash, and the parsed content must JSON-equal the manifest minus the
    volatile freeze stamps."""
    directory = _dir(name, root)
    if not os.path.isdir(directory):
        return  # legacy flat-file manifest
    payload = {k: v for k, v in d.items()
               if k not in ("status", "frozenAt", "freezeHash", "gitCommit",
                            "frozenBy", "appVersion", "createdAt",
                            "freezeForced", "forcedGatesSkipped")}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    with open(os.path.join(directory, "freeze-canonical.json"), "w",
              encoding="utf-8") as handle:
        handle.write(blob)


def _pin_training_provenance(d: dict, root: str | None) -> None:
    """Stamp each adapter variant's ``trainingProvenance`` from the adapter's
    own training sidecar (LoRA readiness contract §9).

    Pin-when-absent, per KEY (the ``markersHash`` pattern, one level down):
    a value already in the manifest is NEVER overwritten — that would be a
    silent re-pin, and it would also destroy the ex ante ``matchedControl``
    declaration (amendment 2), which is researcher data the trainer knows
    nothing about. Only sidecars that carry the v2 provenance schema are
    pinned from; a v1/exploratory adapter has nothing to pin and gets no new
    keys, so legacy manifest bytes are unchanged.

    Runs BEFORE the freeze hash, so the frozen manifest covers the dataset
    pins and post-freeze drift in a training file is a verify() violation.
    """
    from .manifest import (TRAINING_PROVENANCE_SCHEMA_VERSION,
                           variant_adapter_sidecar)
    for vc in d.get("variantConditions") or []:
        if not isinstance(vc, dict) or isinstance(vc.get("fromPromotion"), dict):
            continue  # forward references pin at run time, by design
        path, sidecar = variant_adapter_sidecar(vc.get("artifact") or {}, root)
        if not path or not isinstance(sidecar, dict):
            continue
        try:
            version = int(sidecar.get("schemaVersion") or 0)
        except (TypeError, ValueError):
            version = 0
        if version < TRAINING_PROVENANCE_SCHEMA_VERSION:
            continue
        try:
            with open(path, "rb") as handle:
                sidecar_hash = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            continue
        dataset = sidecar.get("dataset") if isinstance(sidecar.get("dataset"), dict) else {}
        block = vc.get("trainingProvenance")
        if not isinstance(block, dict):
            block = {}
        stamped = {
            "datasetBundleID": dataset.get("bundleID"),
            "adapterSidecarHash": sidecar_hash,
            "evidenceGrade": bool(sidecar.get("evidenceGrade")),
        }
        # Path and hash are ONE pin: stamping a path the sidecar could not
        # hash would mint a half-pin that certifies nothing and refuses at
        # verify. A sidecar with an incomplete dataset block stamps neither —
        # and if it also claims evidence grade, the variantValidity gate says
        # so in as many words.
        dataset_path = str(dataset.get("manifestPath") or "").strip()
        dataset_hash = str(dataset.get("manifestHash") or "").strip()
        if dataset_path and dataset_hash:
            stamped["datasetManifestPath"] = dataset_path
            stamped["datasetManifestHash"] = dataset_hash
        for key, value in stamped.items():
            block.setdefault(key, value)
        # The matched control is DECLARED, never derived: an absent key
        # becomes an explicit null so the frozen manifest says "no control
        # was declared" instead of leaving it open to later reinterpretation.
        block.setdefault("matchedControl", None)
        vc["trainingProvenance"] = block


def _check_variant_validity(name: str, d: dict, root: str | None = None) -> None:
    """Modality arms need a validity story (WORK-PLAN Phase E, promoted by the
    modality-axis decision): a variant condition's interventions must be fully
    pinned — adapter content hashes, system-prompt hash, vector artifact ids —
    or the condition is unverifiable the moment it freezes. ``freeze --force``
    skips this loudly, like the other evidence gates; the always-run verify()
    of the artifact-file hash is never skippable."""
    for vc in d.get("variantConditions") or []:
        label = vc.get("name", "?")
        if isinstance(vc.get("fromPromotion"), dict):
            # Forward-referenced (stage 4): the artifact does not exist at
            # freeze time BY DESIGN — its pins land at run time and are
            # recorded in the run directory (forward-resolutions.json).
            # verify() enforces the declaration shape (attached concept,
            # exactly one identity).
            continue
        artifact = vc.get("artifact") or {}
        if not vc.get("artifactHash"):
            raise ExperimentStoreError(
                f"cannot freeze '{name}': variant '{label}' has no pinned "
                "artifactHash")
        for adapter in artifact.get("adapters") or []:
            if not adapter.get("adapterHash"):
                raise ExperimentStoreError(
                    f"cannot freeze '{name}': variant '{label}' adapter "
                    f"'{adapter.get('name', '?')}' has no adapterHash — re-save "
                    "the variant with hashed adapter weights, or freeze --force")
        if (artifact.get("systemPrompt") or "").strip() and \
                not artifact.get("systemPromptHash"):
            raise ExperimentStoreError(
                f"cannot freeze '{name}': variant '{label}' has a system prompt "
                "but no systemPromptHash — re-save the variant, or freeze --force")
        for injection in artifact.get("injections") or []:
            if not injection.get("vectorArtifactID"):
                raise ExperimentStoreError(
                    f"cannot freeze '{name}': variant '{label}' has an injection "
                    f"for '{injection.get('concept', '?')}' without a "
                    "vectorArtifactID pin")
        # Trained-adapter arms owe the same story about their TRAINING DATA
        # (LoRA readiness §0 amendment 1). An evidence-grade adapter whose
        # dataset is not pinned into the manifest is unverifiable the moment
        # it freezes: the training files could change afterwards with nothing
        # to flag the drift. Exploratory adapters are legal and produce an
        # advisory instead (freeze_advisories), never a refusal.
        if artifact.get("adapters") and variant_is_evidence_grade(vc, root):
            block = vc.get("trainingProvenance")
            has_pin = isinstance(block, dict) and \
                str(block.get("datasetManifestHash") or "").strip()
            if not has_pin:
                raise ExperimentStoreError(
                    f"cannot freeze '{name}': variant '{label}' uses an "
                    "evidence-grade adapter but carries no "
                    "trainingProvenance.datasetManifestHash — its training "
                    "data would stay outside the freeze pin surface. Re-attach "
                    "the variant so freeze can pin the dataset manifest from "
                    "the adapter's sidecar, or freeze --force")


def _resolved_judge_identity(judge: dict, study_model: str) -> tuple[str, str, str]:
    """A judge's RESOLVED identity ``(kind, model, provider)`` — what will
    actually run, not what the manifest happens to spell. Cross-engine rules:
    a LOCAL judge with a blank model resolves to the STUDY model; a claude
    judge with a blank model resolves to the default Claude judge model;
    openrouter judges have no defaults (their own verify rules apply)."""
    kind = str(judge.get("kind") or "claude").strip() or "claude"
    model = str(judge.get("model") or "").strip()
    provider = str(judge.get("provider") or "").strip()
    if kind == "openrouter":
        from . import paired_judge
        provider = paired_judge.canonical_openrouter_provider(provider)
    if kind == "local" and not model:
        model = study_model
    elif kind == "claude" and not model:
        from . import paired_judge
        model = paired_judge.DEFAULT_JUDGE_MODEL
    return (kind, model, provider)


def judge_panel_indistinct_problem(d: dict) -> str | None:
    """External review 2026-07-22 (finding 4): two blank-model local judges
    both resolve to the study model at temperature 0 — identical
    deterministic judges whose perfect agreement is guaranteed by
    construction, satisfying a count-only panel gate while providing zero
    independence. Returns the plain-language problem (identical wording on
    both engines) when a panel of >= 2 named judges collapses to fewer than
    2 DISTINCT resolved identities, else None. Shared by the freeze gate
    (judgeValidity) and the pre-freeze advisories/data check."""
    judges = [j for j in (d.get("judges") or [])
              if isinstance(j, dict) and j.get("name")]
    if len(judges) < 2:
        return None
    study_model = str(d.get("modelID") or "")
    identities: dict[tuple[str, str, str], list[str]] = {}
    for judge in judges:
        identity = _resolved_judge_identity(judge, study_model)
        identities.setdefault(identity, []).append(str(judge["name"]))
    if len(identities) >= 2:
        return None
    (kind, model, provider), names = next(iter(identities.items()))
    quoted = [f"'{n}'" for n in names]
    joined = (" and ".join(quoted) if len(quoted) == 2
              else ", ".join(quoted[:-1]) + " and " + quoted[-1])
    quantifier = "both" if len(quoted) == 2 else "all"
    if kind == "local" and model == study_model:
        what = "the study model at temperature 0"
    elif provider:
        what = f"the {kind} judge '{model}' via '{provider}'"
    else:
        what = f"the {kind} judge '{model}'"
    return (f"judges {joined} {quantifier} resolve to the same deterministic "
            f"judge ({what}) — they would agree perfectly by construction; "
            "use judges with different models, kinds, or providers")


def _pipeline_stage_list(d: dict) -> list[str]:
    """The declared pipeline's stage list, or [] when no pipeline is
    declared or the block is malformed (malformed blocks have their own
    verify violations)."""
    block = d.get("pipeline")
    if not isinstance(block, dict):
        return []
    try:
        from .pipeline_spec import resolve_pipeline
        return list(resolve_pipeline(block).stages)
    except Exception:  # noqa: BLE001 - malformed pipeline refuses elsewhere
        return []


#: Canonical dtype names a judge may pin, and the aliases that resolve to
#: them. Duplicated from `steering.model_loader` DELIBERATELY: the manifest
#: firewall must not import torch to validate a manifest. A test asserts the
#: two agree, and the Swift twin (`ExperimentStore.judgeDtypeVocabulary`)
#: carries the same set.
JUDGE_DTYPE_VOCABULARY = ("bfloat16", "float16", "float32")
_JUDGE_DTYPE_ALIASES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16",
    "float32": "float32", "fp32": "float32",
}


def normalize_judge_dtype(value: str | None) -> str | None:
    """Canonical spelling of a judge dtype alias, or None if unrecognized."""
    return _JUDGE_DTYPE_ALIASES.get((value or "").strip().lower())


def _is_commit_like(revision: str) -> bool:
    """Whether a revision names FIXED bytes rather than a moving ref.

    Hexadecimal — the shape of a git commit hash, full or abbreviated.
    Branch names (`main`, `master`), `HEAD`, `refs/...` paths, and
    conventional tags (`v1.0`, `latest`) all fail it, which is the point: a
    branch is re-pointed by definition, and a tag can be moved, so neither
    identifies the bytes a run used (external review round 5, finding 4).

    Honest residual: a tag whose name happens to be hexadecimal would pass.
    No format check can distinguish that from a short hash — only asking the
    hub could — and it is not a shape anyone tags in practice.
    """
    stripped = revision.strip()
    if not stripped:
        return False
    try:
        int(stripped, 16)
    except ValueError:
        return False
    return True


def symbolic_revision_problem(d: dict) -> str | None:
    """Revision pins that name a moving ref instead of a commit.

    Applies to the STUDY revision and to every local judge's. `"main"`
    passed the old gate — it only required non-emptiness — and the loader
    then recorded the symbolic name it was handed rather than the commit it
    resolved to, so two runs a week apart could record the same "pin" and
    have run different weights.

    Cross-engine wording (Swift twin:
    `ExperimentStore.symbolicRevisionProblem`).
    """
    offenders: list[str] = []
    study = str(d.get("modelRevision") or "").strip()
    if study and not _is_commit_like(study):
        offenders.append(f"the study model pins '{study}'")
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "openrouter").strip() or "openrouter") != "local":
            continue
        revision = str(judge.get("revision") or "").strip()
        if revision and not _is_commit_like(revision):
            offenders.append(f"judge '{judge['name']}' pins '{revision}'")
    if not offenders:
        return None
    return ("revision pin(s) name a moving reference rather than a commit: "
            + "; ".join(offenders)
            + ". A branch or tag is re-pointed by definition, so it cannot "
            "identify the weights a run used — two runs a week apart would "
            "record the same pin having loaded different bytes. Use the "
            "commit hash (the app's Resolve button reads it from whichever "
            "substrate will run the model)")


def study_model_judge_pin_conflict(d: dict) -> str | None:
    """A study-model local judge declaring pins that differ from the study's.

    Scoped to studies declaring a **judgeScore sweep** (external review
    round 5, finding 1). There, such a judge has no independent identity:
    the sweep judges with the already-HELD study model rather than loading
    anything, so a divergent `revision`/`dtype` is silently ignored while
    remaining in the criterion provenance.

    Deliberately NOT a blanket rule. `evaluate` genuinely LOADS a declared
    judge revision, so judging with a different checkpoint of the study repo
    is a legitimate design there — `_pin_local_judge_revisions` has always
    preserved a declared revision for exactly that reason. The defect is the
    SWEEP path silently ignoring what evaluate honors: one manifest, two
    identities, depending on the verb.

    Forbidding divergence is preferred over verify-and-stamp: it is
    checkable while authoring, rather than producing an artifact merely
    honest about having judged with something else. Pins that AGREE with the
    study stay legal — redundant, not wrong.

    Cross-engine wording (Swift twin:
    `ExperimentStore.studyModelJudgePinConflict`).
    """
    selection = (d.get("sweep") or {}).get("selection") \
        if isinstance(d.get("sweep"), dict) else None
    objective = (selection or {}).get("objective") \
        if isinstance(selection, dict) else None
    if ((objective or {}).get("metric")
            if isinstance(objective, dict) else None) != "judgeScore":
        return None
    study_model = str(d.get("modelID") or "")
    study_revision = str(d.get("modelRevision") or "").strip()
    study_dtype = str(d.get("dtype") or "").strip()
    offenders: list[str] = []
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "openrouter").strip() or "openrouter") != "local":
            continue
        declared = str(judge.get("model") or "").strip()
        # Blank model AND explicit study model both resolve to the study
        # model (`sweep_selection.resolve_local_judge_model`).
        if declared and declared != study_model:
            continue
        revision = str(judge.get("revision") or "").strip()
        if revision and revision != study_revision:
            offenders.append(
                f"'{judge['name']}' pins revision '{revision}' but the study "
                + (f"is pinned at '{study_revision}'" if study_revision
                   else "has no revision pinned"))
        dtype = str(judge.get("dtype") or "").strip()
        if dtype and normalize_judge_dtype(dtype) != \
                normalize_judge_dtype(study_dtype):
            offenders.append(
                f"'{judge['name']}' pins dtype '{dtype}' but the study "
                + (f"is pinned at '{study_dtype}'" if study_dtype
                   else "pins none (the device decides)"))
    if not offenders:
        return None
    return ("this study selects on judgeScore, and local judge(s) resolving "
            "to the STUDY model cannot pin a different identity: " + "; ".join(offenders)
            + ". Such a judge IS the study model — a sweep judges with the "
            "already-held weights and never loads anything else, so the "
            "divergent pin would be silently ignored. Drop the pin to "
            "inherit the study's, or name a different model to make it a "
            "genuinely separate judge")


def unloadable_study_dtype_problem(d: dict) -> str | None:
    """A study-level `dtype` outside the closed vocabulary.

    The Mac is the AUTHORING surface and the cluster is the measurement one,
    so this is validated here even though only the server consumes the key —
    a manifest must not reach the cluster carrying a dtype that refuses at load
    after a queue wait. Cross-engine wording (Swift twin:
    `ExperimentStore.unloadableStudyDtypeProblem`).
    """
    spelled = str(d.get("dtype") or "").strip()
    if not spelled or normalize_judge_dtype(spelled) is not None:
        return None
    return (f"study dtype '{spelled}' is not one this engine can load — the "
            "loader accepts only " + ", ".join(JUDGE_DTYPE_VOCABULARY)
            + " (aliases bf16/fp16/fp32). Leave it unset to let the device "
            "decide, which is what every study did before this pin existed")


def unpinned_foreign_local_judge_problem(d: dict) -> str | None:
    """Foreign local judges whose model bytes are not pinned.

    A local judge resolving to the STUDY model inherits the study's pinned
    revision, so "the same judge" across two sessions is a fact. A local
    judge naming a DIFFERENT model has no such pin to inherit — and freeze
    deliberately leaves its revision blank (`_pin_local_judge_revisions`).
    That was tolerable while a judgment artifact merely RECORDED what
    loaded, but targeted retry compares recorded identities to decide
    whether verdicts from an earlier session may be REUSED: two sessions
    can each load a different default revision while both records say
    ``null``, and null == null passes (external review round 2, finding 3).

    Requiring the pin is the cheaper of the two fixes and matches the
    pin-everything discipline everywhere else. `dtype` is required with it
    on this engine because the loader takes one and a bf16-vs-fp16 judge is
    a different judge.

    Returns the plain-language problem, or None when every foreign local
    judge is pinned. Cross-engine wording.
    """
    study_model = str(d.get("modelID") or "")
    offenders: list[str] = []
    unknown: list[str] = []
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "openrouter").strip() or "openrouter") != "local":
            continue
        # A dtype OUTSIDE the closed vocabulary is checked for every local
        # judge, pinned or not: the loader refuses it at run time, and
        # discovering that on a compute node after a queue wait is exactly
        # the failure this firewall exists to move forward in time
        # (external review round 4, finding 2).
        spelled = str(judge.get("dtype") or "").strip()
        if spelled and normalize_judge_dtype(spelled) is None:
            unknown.append(f"'{judge['name']}' declares dtype '{spelled}'")
        declared = str(judge.get("model") or "").strip()
        if not declared or declared == study_model:
            continue
        missing = [field for field in ("revision", "dtype")
                   if not str(judge.get(field) or "").strip()]
        if missing:
            offenders.append(
                f"'{judge['name']}' (model '{declared}') is missing "
                + " and ".join(missing))
    if unknown:
        return ("local judge(s) declare a dtype this engine cannot load: "
                + "; ".join(unknown) + ". The loader accepts only "
                + ", ".join(JUDGE_DTYPE_VOCABULARY)
                + " (aliases bf16/fp16/fp32). An unrecognized value used to "
                "load float32 silently, so the pin would be a false claim")
    if not offenders:
        return None
    return ("local judge(s) naming a model other than the study model must "
            "pin the exact bytes that will judge: " + "; ".join(offenders)
            + ". Without a revision pin two judging sessions can load "
            "different defaults while both records say 'none', so a "
            "resumed evaluation cannot prove its reused verdicts came from "
            "the same judge. Pin judges[].revision and judges[].dtype, or "
            "use the study model as judge")


def _foreign_local_judges(d: dict) -> list[str]:
    """Local judges whose declared model differs from the study model,
    rendered ``'name' (model 'id')``."""
    study_model = str(d.get("modelID") or "")
    offenders: list[str] = []
    for judge in (d.get("judges") or []):
        if not (isinstance(judge, dict) and judge.get("name")):
            continue
        if (str(judge.get("kind") or "claude").strip() or "claude") != "local":
            continue
        declared = str(judge.get("model") or "").strip()
        if declared and declared != study_model:
            offenders.append(f"'{judge['name']}' (model '{declared}')")
    return offenders


def local_judge_pipeline_problem(d: dict) -> str | None:
    """Finding 1 gate, fan-out era (2026-07-23): a judged SWEEP inside the
    chain still cannot use a local judge whose model differs from the study
    model — sweep judging is interleaved with the selection, not a
    separable post-stage, so no fan-out exists for it. The EVALUATE stage
    no longer refuses (it routes to the post-generation judge fan-out —
    see :func:`local_judge_fanout_note`). Returns the sweep problem text,
    or None."""
    stages = _pipeline_stage_list(d)
    if "sweep" not in stages:
        return None
    selection = (d.get("sweep") or {}).get("selection") \
        if isinstance(d.get("sweep"), dict) else None
    objective = (selection or {}).get("objective") \
        if isinstance(selection, dict) else None
    metric = (objective or {}).get("metric") \
        if isinstance(objective, dict) else None
    if metric != "judgeScore":
        return None
    offenders = _foreign_local_judges(d)
    if not offenders:
        return None
    study_model = str(d.get("modelID") or "")
    return ("the declared pipeline's sweep stage holds ONE model — the "
            f"study model '{study_model}' — but local judge(s) "
            + ", ".join(offenders) + " resolve to a different model, which "
            "cannot load inside the chain (the judge fan-out covers the "
            "evaluate stage only). Leave a local judge's model empty to "
            "judge with the study model, pin claude/openrouter judges, or "
            "select on logprobShift")


def local_judge_fanout_note(d: dict) -> str | None:
    """Routing information (never a gate, 2026-07-23): a declared pipeline
    whose EVALUATE stage pins local judges resolving to models other than
    the study model will judge them as a post-generation fan-out — the
    chain emits blinded packets, one worker job per distinct judge model
    judges them, and the merge resumes the chain. Available on Slurm
    run-first pipeline submissions; elsewhere the packets await deferred
    (Mac) judging. Returns the note, or None."""
    stages = _pipeline_stage_list(d)
    if "evaluate" not in stages:
        return None
    offenders = _foreign_local_judges(d)
    if not offenders:
        return None
    return ("the pipeline's evaluate stage will judge local judge(s) "
            + ", ".join(offenders) + " as a post-generation judge fan-out "
            "(one worker job per distinct judge model; available on Slurm "
            "run-first pipeline submissions — elsewhere the emitted packets "
            "await deferred judging)")


def _pin_local_judge_revisions(d: dict) -> None:
    """Freeze-time pin for LOCAL judge revisions (cross-engine contract key
    ``judges[].revision``, 2026-07-23, omit-when-nil): a local judge that
    resolves to the STUDY model inherits the study's pinned revision when
    its own is blank — the judging path then loads exactly the pinned
    bytes. A local judge declaring a DIFFERENT model keeps its blank
    revision (there is no study pin to inherit; the judgment artifact
    stamps what actually loaded). Never overwrites a declared revision."""
    study_revision = d.get("modelRevision")
    if not study_revision:
        return
    study_model = str(d.get("modelID") or "")
    for judge in (d.get("judges") or []):
        if not isinstance(judge, dict):
            continue
        if (str(judge.get("kind") or "claude").strip() or "claude") != "local":
            continue
        if judge.get("revision"):
            continue
        declared = str(judge.get("model") or "").strip()
        if not declared or declared == study_model:
            judge["revision"] = study_revision


def _check_judged_evaluation(name: str, d: dict) -> None:
    """Judged studies need a versioned criterion and a real panel (evidence
    tier): a pairedJudge evaluation must pin its rubric as a hashed FILE
    (prompts/rubrics/) and declare >= 2 judges with DISTINCT resolved
    identities, or inter-judge agreement — the check that the criterion
    measures anything — is unreportable (or trivially perfect).
    ``freeze --force`` skips this loudly, like the other evidence gates."""
    evaluation = d.get("evaluation") or {}
    # Judge-evaluated = a pairedJudge evaluation OR an explicit judges panel
    # (Swift's rule exactly — declaring judges intends judged evaluation, and
    # the two engines' gates must agree or a manifest freezes on one engine
    # and not the other).
    if evaluation.get("kind") != "pairedJudge" and not d.get("judges"):
        return
    if not (d.get("judgeRubricFile") and d.get("judgeRubricHash")):
        raise ExperimentStoreError(
            f"cannot freeze '{name}': evaluation uses pairedJudge but no judge "
            "rubric is pinned — set judgeRubricFile + judgeRubricHash "
            "(rubrics live in prompts/rubrics/), or freeze --force")
    judges = [j for j in (d.get("judges") or [])
              if isinstance(j, dict) and j.get("name")]
    if len(judges) < 2:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': evaluation uses pairedJudge with "
            f"{len(judges)} pinned judge(s) — judged studies need at least 2 "
            "judges for agreement statistics, or freeze --force")
    indistinct = judge_panel_indistinct_problem(d)
    if indistinct:
        raise ExperimentStoreError(f"cannot freeze '{name}': {indistinct}")
    pipeline_problem = local_judge_pipeline_problem(d)
    if pipeline_problem:
        raise ExperimentStoreError(f"cannot freeze '{name}': {pipeline_problem}")
    unpinned = unpinned_foreign_local_judge_problem(d)
    if unpinned:
        raise ExperimentStoreError(f"cannot freeze '{name}': {unpinned}")
    study_model_conflict = study_model_judge_pin_conflict(d)
    if study_model_conflict:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': {study_model_conflict}")


def _check_battery_evidence(name: str, d: dict, evidence: dict | None,
                            root: str | None) -> None:
    """Variant freeze gate, evidence half: the scope-matched validate evidence
    must contain capability-battery results for the baseline and EVERY variant
    condition, produced from the battery the manifest pins (or the live
    default battery when unpinned). Hash pins alone say the artifact bytes are
    stable; this says the variant still answers concept-unrelated probes."""
    from . import battery as battery_mod
    results = {r.get("condition"): r
               for r in (evidence or {}).get("batteryResults") or []
               if isinstance(r, dict)}
    # Forward-referenced conditions (stage 4) are exempt: their agent does
    # not exist at validate time, so their battery evidence is produced by
    # the RUN's per-condition battery, not by freeze-time validation.
    required = ["baseline"] + [
        vc.get("name", "?") for vc in d.get("variantConditions") or []
        if not isinstance(vc.get("fromPromotion"), dict)]
    missing = [c for c in required
               if c not in results or results[c].get("accuracy") is None]
    if missing:
        raise ExperimentStoreError(
            f"cannot freeze '{name}': validate evidence has no capability-"
            f"battery results for condition(s) {', '.join(missing)} — run "
            f"'experiment validate {name}' (each variant condition runs the "
            "pinned battery), or freeze --force")
    expected = d.get("capabilityBatteryHash") or battery_mod.live_hash(
        battery_mod.DEFAULT_BATTERY_FILE, root)
    if expected:
        drifted = sorted(c for c in required
                         if results[c].get("batteryHash") != expected)
        if drifted:
            raise ExperimentStoreError(
                f"cannot freeze '{name}': capability battery drifted since "
                f"validation for condition(s) {', '.join(drifted)} — "
                f"re-run 'experiment validate {name}', or freeze --force")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return slug or "unnamed"


def _pin_external_inputs(name: str, d: dict, root: str | None) -> None:
    """Copy pinned inputs that live under gitignored ``runs/`` into the
    experiment's own directory (``experiments/<name>/pinned/``), byte-identical,
    and repoint the manifest. A frozen, git-committed manifest must never
    depend on a file the repository does not version.

    Concept and agent artifacts are relocated only for the study kind that
    RUNS them. A panel carrying either across a kind switch would otherwise
    have its carried configuration silently rewritten and an irrelevant
    artifact committed — or, when the stale file is simply gone, die in
    ``copyfile`` — immediately after freeze printed the advisory promising
    that carried state is "not verified, snapshotted, or bundled for this
    study kind" (external review round 14).
    """
    base = paths.project_root() if root is None else root
    model_output = model_output_surfaces_operative(d)

    def _relative(path: str) -> str:
        if os.path.isabs(path):
            try:
                return os.path.relpath(path, base)
            except ValueError:
                return path
        return path

    def _under_runs(path: str) -> bool:
        rel = _relative(path).replace(os.sep, "/")
        return rel.startswith("runs/")

    def _copy_in(path: str, dest_name: str) -> str:
        src = path if os.path.isabs(path) else os.path.join(base, path)
        pinned_dir = os.path.join(_dir(name, root), "pinned")
        os.makedirs(pinned_dir, exist_ok=True)
        dest = os.path.join(pinned_dir, dest_name)
        shutil.copyfile(src, dest)
        return os.path.relpath(dest, base)

    # The scenario is the PANEL's input, not a kind-neutral one — the mirror
    # of the carried-agent case, and my round-14 comment calling it
    # kind-neutral was simply wrong. A model-output study carrying a scenario
    # across a kind switch executes none of it: relocating a stale one failed
    # the copy, and relocating a present one rewrote and committed inactive
    # configuration (external review round 15).
    scenario_path = d.get("multiAgentScenarioPath")
    if not model_output and scenario_path and _under_runs(scenario_path):
        d["multiAgentScenarioPath"] = _copy_in(scenario_path, "scenario.json")
    # Artifact-pinned concepts: the vector bytes are the pinned input, and
    # they live in gitignored runs/ by construction. Both files move together
    # (byte-identical, so both pinned hashes still verify) and the locator is
    # repointed at the pair's new extension-less path.
    for concept in (d.get("concepts") or []) if model_output else []:
        artifact = (concept or {}).get("vectorArtifact")
        if not isinstance(artifact, dict) or not artifact.get("path"):
            continue
        rel = str(artifact["path"])
        if not _under_runs(f"{rel}.safetensors"):
            continue
        stem = f"vector-{_slugify(concept.get('name', 'unnamed'))}"
        moved = _copy_in(f"{rel}.safetensors", f"{stem}.safetensors")
        _copy_in(f"{rel}.json", f"{stem}.json")
        artifact["path"] = moved[:-len(".safetensors")]
    for vc in (d.get("variantConditions") or []) if model_output else []:
        artifact_path = vc.get("artifactPath")
        if artifact_path and _under_runs(artifact_path):
            vc["artifactPath"] = _copy_in(
                artifact_path, f"variant-{_slugify(vc.get('name', 'unnamed'))}.json")


def _snapshot_sources(d: dict, root: str | None) -> list[str]:
    """The freeze SNAPSHOT's source list: the pin surface, plus the neutral
    corpus whenever the manifest names one.

    Why the snapshot is wider than the pin surface here (2026-08-18, WP0
    residual (c)). :func:`pinned_input_entries` guards the neutral corpus with
    ``model_output and machinery``, which excludes exactly the studies that
    keep ``neutralCorpusHash`` without operating the concept machinery — a
    compare-agents study whose promoted agents' α is in norm units, and a
    panel carrying the pin forward. Their manifests still NAME a corpus hash,
    so the bytes behind that name still have to survive beside the frozen
    manifest; whether this engine re-derives vectors from them is a different
    question. Widening the SNAPSHOT is additive and freeze-time-only — the
    git-cleanliness gate and the bundle packer keep reading
    :func:`pinned_input_paths` unchanged, so no study that freezes today stops
    freezing and no already-frozen study changes. Swift twin: the
    ``neutralCorpusHash != nil`` plan in ``snapshotPinnedInputs``."""
    sources = list(pinned_input_paths(d, root))
    if d.get("neutralCorpusHash"):
        corpus = paths.neutral_corpus_path(root)
        if os.path.exists(corpus) and not any(
                os.path.realpath(s) == os.path.realpath(corpus)
                for s in sources):
            sources.append(corpus)
    return sources


def _snapshot_pinned_inputs(name: str, d: dict, root: str | None) -> None:
    """Freeze-time reproducibility floor (no-git workspaces included): copy
    EVERY pinned input — files, and directories recursively (concept stimulus
    dirs, grand-mean stories) — into ``experiments/<name>/pinned/``, mirroring
    the workspace-relative layout. Verification-bytes copies only: the
    manifest's prompts/-resident paths are NOT rewritten (the SHA-256 pins
    prove identity; the snapshot just guarantees the bytes survive beside the
    frozen manifest). Paths already under pinned/ (e.g. placed there by
    ``_pin_external_inputs``) are skipped."""
    base = paths.project_root() if root is None else root
    base_real = os.path.realpath(base)
    pinned_dir = os.path.join(_dir(name, root), "pinned")
    pinned_real = os.path.realpath(pinned_dir)
    for src in _snapshot_sources(d, root):
        src_real = os.path.realpath(src)
        if src_real == pinned_real or src_real.startswith(pinned_real + os.sep):
            continue  # already snapshotted
        rel = os.path.relpath(src_real, base_real)
        if rel.startswith(".."):
            continue  # outside the workspace — hash-pinned but not mirrorable
        dest = os.path.join(pinned_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isdir(src_real):
            shutil.copytree(src_real, dest, dirs_exist_ok=True)
        else:
            shutil.copyfile(src_real, dest)


def _git_toplevel(base: str) -> str | None:
    """The enclosing git work-tree root of ``base``, or None when ``base`` is
    not inside a work tree (or git is unavailable)."""
    try:
        out = subprocess.run(["git", "-C", base, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _auto_commit_workspace(name: str, root: str | None,
                           message: str | None = None,
                           quiet: bool = False) -> None:
    """Freeze = commit + stamp in one gesture: when the artifact root IS the
    root of its own git work tree, ``git add -A`` + ``git commit`` everything
    so the gitCommit stamped moments later actually contains the pinned bytes
    (including the snapshot just written). Freeze calls this twice (Swift
    parity): once before stamping (``freeze <name>``) and once after
    (``freeze <name> (stamp)``, via ``message``/``quiet``) so the frozen
    manifest, canonical bytes, and preregistration are committed too and the
    workspace ends clean.

    Safety rule (server-side; Swift instead scopes its auto-commit to `.`
    within the workspace and skips the legacy code-checkout root — it does
    NOT apply this subdirectory check): when the workspace is a SUBDIRECTORY
    of a larger repository — e.g. today's source checkout —
    auto-committing someone's whole repo as a freeze side effect would be
    hostile, so the commit is skipped and the skip is logged (once — the
    stamp pass sets ``quiet``); the cleanliness gate already refused dirty
    pins in that case. Commit failure or a non-git root is a silent skip —
    auto-commit never blocks a freeze."""
    base = paths.project_root() if root is None else root
    top = _git_toplevel(base)
    if top is None:
        return  # not a git work tree — nothing to commit
    if os.path.realpath(top) != os.path.realpath(base):
        if not quiet:
            print(f"freeze '{name}': auto-commit skipped — artifact root {base!r} "
                  f"is a subdirectory of the git repository at {top!r}, and "
                  "committing that whole repo as a side effect would be hostile. "
                  "Commit the pinned inputs yourself.", file=sys.stderr)
        return
    try:
        subprocess.run(["git", "-C", base, "add", "-A"],
                       capture_output=True, timeout=30)
        subprocess.run(["git", "-C", base, "commit", "-m",
                        message or f"freeze {name}"],
                       capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass  # never block the freeze; the stamp falls back to prior HEAD


def _write_preregistration(d: dict, root: str | None) -> None:
    """Export preregistration.md beside the frozen manifest — the study's
    settings-chosen-before-measurement statement, generated at the freeze
    instant from the frozen dict so it cannot disagree with what was frozen."""
    directory = _dir(d["name"], root)
    if not os.path.isdir(directory):
        return  # legacy flat-file manifest; no directory to write into
    lines = [
        f"# Preregistration: {d['name']}",
        "",
        f"- **Frozen at:** {d.get('frozenAt')}",
        f"- **Freeze hash:** `{d.get('freezeHash')}`",
        f"- **Git commit:** `{d.get('gitCommit')}`",
        f"- **Model:** {d.get('modelID')} @ `{d.get('modelRevision')}`",
        f"- **Phase:** {d.get('phase') or 'unspecified'}",
        f"- **Case family:** {d.get('caseFamily') or 'unspecified'}",
        f"- **Outcome instruments:** "
        f"{', '.join(d.get('outcomeInstruments') or []) or 'sampledText'}",
        f"- **Sampling:** temperature {d.get('temperature', 0)}, "
        f"samplesPerItem {d.get('samplesPerItem', 1)}, "
        f"seedPolicy {d.get('seedPolicy') or 'manifestSeeds'}",
        "",
        "## Conditions",
        "",
    ]
    for condition in d.get("conditions") or []:
        slots = ", ".join(
            f"{s.get('concept')}@L{s.get('layer')} α={s.get('alpha')}"
            for s in condition.get("slots") or []) or "none (baseline)"
        control = condition.get("controlType")
        suffix = f" [{control}]" if control else ""
        lines.append(f"- **{condition.get('name')}**{suffix}: {slots}")
    if d.get("promotionRule"):
        pr = d["promotionRule"]
        lines += [
            "",
            "## Promotion rule (screen → confirm)",
            "",
            f"- FDR threshold: {pr.get('fdrThreshold', 0.05)} "
            "(Benjamini–Hochberg across concepts)",
            f"- Dose-monotonicity required: {pr.get('doseMonotone', True)}",
            f"- Must exceed matched-norm random floor: "
            f"{pr.get('exceedsRandomFloor', True)}",
            f"- Capability gate: {pr.get('capabilityGate') or 'none'}",
        ]
    if d.get("humanBaseline"):
        hb = d["humanBaseline"]
        lines += [
            "",
            "## Human baseline",
            "",
            f"- `{hb.get('path')}` (SHA-256 `{hb.get('hash')}`)",
            "- Residual: R = delta_model − delta_human, per endpoint.",
        ]
    lines += [
        "",
        "## Statistics",
        "",
        "- Paired to each item's same-case baseline; percentile bootstrap CIs "
        "on the mean paired difference; Wilcoxon signed-rank as the "
        "nonparametric companion.",
        "- Multiple comparisons: BH-FDR across concepts at screen; Holm within "
        "the pre-registered family at confirm.",
        "",
        "*Generated at freeze; do not edit. Duplicate the experiment to change "
        "anything.*",
        "",
    ]
    with open(os.path.join(directory, "preregistration.md"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _git_commit(root: str | None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=paths.project_root() if root is None else root,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class PinnedInput:
    """One manifest-declared file input: absolute path, a human-readable
    label for error messages, and whether the manifest REQUIRES the file to
    exist (a declared pin — a study artifact without it is broken) or merely
    reads it when present (e.g. the experiments/<name>/pinned snapshot, a
    grand-mean concept's optional markers directory)."""
    path: str
    label: str
    required: bool


def _variant_artifact_dependencies(
        vc: dict, root: str | None = None) -> list[tuple[str, str]]:
    """The files a variant condition's agent artifact REFERENCES: each
    injection's vector pair, each adapter directory, and the neutral-PC
    basis it projects against.

    Unlike the rest of `pinned_input_entries` this reads a file — it has to,
    because the dependency list exists only inside the artifact. An artifact
    that cannot be read or decoded contributes nothing: its own entry is
    already `required`, so packaging refuses, and guessing dependency paths
    from an unparsed file would be worse than the honest refusal. Parallel
    to Swift ``ExperimentStore.variantArtifactDependencies``."""
    artifact_path = (vc or {}).get("artifactPath")
    if not artifact_path:
        return []
    name = (vc or {}).get("name", "?")
    try:
        with open(paths.resolve(artifact_path, root), encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(artifact, dict):
        return []

    out: list[tuple[str, str]] = []
    for inj in artifact.get("injections") or []:
        # `vectorArtifactID` is an extension-less locator; the store opens
        # the two files beside it.
        ref = (inj or {}).get("vectorArtifactID")
        if not ref:
            continue
        concept = (inj or {}).get("concept", "?")
        for suffix in ("safetensors", "json"):
            out.append((f"{ref}.{suffix}",
                        f"variant '{name}' vector '{concept}' (.{suffix})"))
    for adapter in artifact.get("adapters") or []:
        directory = (adapter or {}).get("adapterDirectory")
        if directory:
            out.append((directory, f"variant '{name}' adapter"))
    basis = artifact.get("neutralPCBasisPath")
    if basis:
        out.append((basis, f"variant '{name}' neutral-PC basis"))
    return out


def _panel_agent_dependencies(
        scenario_rel: str | None, root: str | None = None) -> list[tuple[str, str]]:
    """Agent artifacts a panel script names, and what those artifacts point at.

    Like ``_variant_artifact_dependencies`` this reads files, because the
    dependency list exists only inside them: the scenario names its seats'
    variants, and each variant names its vectors, adapters and neutral basis.
    An unreadable scenario contributes nothing — its own entry is already
    required, so packaging refuses on that instead of guessing.
    """
    if not scenario_rel:
        return []
    try:
        with open(paths.resolve(scenario_rel, root), encoding="utf-8") as handle:
            scenario = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(scenario, dict):
        return []

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for agent in scenario.get("agents") or []:
        artifact = (agent or {}).get("variantArtifactPath")
        if not artifact or artifact in seen:
            continue
        seen.add(artifact)
        seat = (agent or {}).get("name") or (agent or {}).get("id") or "?"
        out.append((artifact, f"panel seat '{seat}' agent artifact"))
        # Reuse the variant walker: a seat's artifact has exactly the same
        # shape and the same dependency surface as a variant condition's.
        out.extend(
            _variant_artifact_dependencies({"artifactPath": artifact,
                                            "name": f"seat '{seat}'"}, root))
    return out


def pinned_input_entries(d: dict, root: str | None = None) -> list[PinnedInput]:
    """THE pin-surface enumeration: every file input the manifest declares,
    as structured entries. Single source of truth shared by the freeze
    cleanliness gate (`_check_git_pin_cleanliness`), the freeze snapshot
    (`_snapshot_pinned_inputs`), and the run-bundle packer
    (`bundles._experiment_files`) — add a new pin kind HERE and it is
    git-gated, snapshotted, and packed automatically; a pin kind that
    bypasses this function is exactly the silent-bundle-gap bug this
    function exists to prevent. Pure path assembly (existence is not checked
    here); a directory entry means "the whole directory". Parallel to Swift
    ``ExperimentStore.pinnedInputEntries``."""
    from . import multiconcept
    from .manifest import resolve_validation_file, validation_file_path

    base = paths.project_root() if root is None else root
    out: list[PinnedInput] = []

    def _add(rel: str | None, label: str, required: bool) -> None:
        if not rel:
            return
        path = rel if os.path.isabs(rel) else os.path.join(base, rel)
        out.append(PinnedInput(path=path, label=label, required=required))

    # The pin surface is the OPERATIVE surface for the study kind
    # (2026-07-19, engineer finding): configuration CARRIED from another
    # type — the picker's never-delete promise — is neither verified nor
    # packaged, so a stale hidden concept can never block a multi-agent
    # freeze or bloat its evidence bundle. Kind-agnostic pins (judging,
    # taxonomy, human data, snapshot) stay on both surfaces. Parallel to
    # Swift ``pinnedInputEntries``.
    model_output = d.get("studyKind", "modelOutput") != "multiAgent"
    # Finer rule within model-output (2026-07-19): concept machinery is
    # inert for compare-agents studies without forward references.
    from .manifest import concept_machinery_operative
    machinery = concept_machinery_operative(d)

    for concept in (d.get("concepts") or []) if machinery else []:
        name = concept.get("name")
        if not name:
            continue
        method = (concept.get("options") or {}).get("method") or "meanDifference"
        artifact = concept.get("vectorArtifact") \
            if isinstance(concept.get("vectorArtifact"), dict) else {}
        if method == "pinnedArtifact":
            # The BYTES are this concept's primary pinned input — both files,
            # required: a bundle without them cannot steer at all. The data
            # entries below still travel, keyed on the SOURCE concept, because
            # validate reads them.
            artifact_path = str(artifact.get("path") or "")
            if artifact_path:
                _add(f"{artifact_path}.safetensors",
                     f"concept '{name}' pinned vector artifact", required=True)
                _add(f"{artifact_path}.json",
                     f"concept '{name}' pinned vector artifact sidecar",
                     required=True)
            method = str(artifact.get("sourceMethod") or "meanDifference")
            data_name = str(artifact.get("sourceConcept") or name)
            if method in (_OPTVEC_METHOD, _GEMMA_SCOPE_METHOD):
                # An optvec or imported-SAE concept's pin surface is the two
                # artifact files and nothing else: neither has a stimulus
                # directory, stories.jsonl or validation.jsonl to travel with
                # a bundle — an optimized vector has no stimuli, and a Gemma
                # Scope decoder row is a coordinate in a published dictionary
                # (attach refuses both as re-derivable recipes for the same
                # reason). Claiming a REQUIRED stimulus directory here would
                # make every such study unpackageable (bundles refuse a
                # missing required input) and would put a nonexistent path
                # in front of the git cleanliness gate. The scoring markers
                # rubric may still exist under the study's own concept name,
                # so it stays on the surface — optional, because these
                # studies score by logprob instrument, not markers.
                _add(os.path.join("prompts", "concepts", name),
                     f"concept '{name}' markers directory", required=False)
                continue
            if data_name != name:
                # The marker rubric is keyed on the STUDY's concept name (the
                # scorer reads prompts/concepts/<name>/markers.json), while
                # the stimuli below are the source concept's — so a renamed
                # derived direction needs both directories on the surface.
                _add(os.path.join("prompts", "concepts", name),
                     f"concept '{name}' markers directory", required=False)
        else:
            data_name = name
        paired = method not in ("emotionGrandMean", "designatedReference")
        # Paired concepts pin their whole stimulus directory; grand-mean
        # concepts keep stimuli under prompts/emotions/ but may still carry
        # a markers.json in prompts/concepts/<name>/ (read when present).
        _add(os.path.join("prompts", "concepts", data_name),
             f"concept '{name}' stimulus directory", required=paired)
        if not paired:
            _add(multiconcept.stories_path(data_name, root),
                 f"concept '{name}' stories.jsonl", required=True)
        ref = (concept.get("designatedReference") or {}).get("name")
        if method == "designatedReference" and ref:
            _add(multiconcept.stories_path(ref, root),
                 f"concept '{name}' designated reference '{ref}' "
                 "stories.jsonl", required=True)
        # Measurement-side pin: a non-null validationHash names exact bytes;
        # an unpinned validation file is still read LIVE by the validate verb
        # (legacy manifests), so it travels when present. Paired concepts'
        # unpinned files are covered by their directory entry above.
        #
        # The path is the one the dual-root lookup FOUND (canonical home
        # first, the other recipe's home as fallback) so a misfiled set
        # travels with the bundle and faces the git-cleanliness gate
        # truthfully; when nothing is on disk the canonical home is named,
        # because that is where the missing file belongs.
        located = resolve_validation_file(data_name, paired=paired, root=root)
        canonical_validation = validation_file_path(
            data_name, paired=paired, root=root)
        if concept.get("validationHash"):
            _add(located.path if located else canonical_validation,
                 f"concept '{name}' validation.jsonl", required=True)
        elif not paired:
            _add(located.path if located else canonical_validation,
                 f"concept '{name}' validation.jsonl", required=False)
    corpus = (d.get("grandMeanCorpus") or {}) if machinery else {}
    for concept in corpus.get("concepts") or []:
        _add(multiconcept.stories_path(concept, root),
             f"grand-mean corpus member '{concept}' stories.jsonl",
             required=True)
    if model_output:
        _add(d.get("taskPromptsFile"), "task prompts", required=True)
        _add(d.get("capabilityBatteryFile"), "capability battery", required=True)
        if d.get("numericParser"):
            from . import parser_registry
            _add(parser_registry.REGISTRY_FILE, "parser registry", required=True)
        if machinery and d.get("neutralCorpusHash"):
            _add(paths.neutral_corpus_path(root), "neutral corpus", required=True)
        for condition in (d.get("conditions") or []) if machinery else []:
            _add((condition or {}).get("neutralPCBasisPath"),
                 f"condition '{(condition or {}).get('name', '?')}' "
                 "neutral-PC basis", required=True)
        for ref in d.get("readerRefs") or []:
            _add((ref or {}).get("path"),
                 f"reader '{(ref or {}).get('concept', '?')}' artifact",
                 required=True)
        # Declared validation controls are pinned INPUTS (C2): their stimuli
        # travel with the bundle and are git-gated at freeze, exactly like a
        # study concept's. Parallel to Swift ``pinnedInputEntries``.
        for control in (d.get("validationControls") or []) if machinery else []:
            concept = (control or {}).get("concept")
            if concept:
                _add(os.path.join("prompts", "concepts", concept),
                     f"validation control '{concept}' stimulus directory",
                     required=True)
        for vc in d.get("variantConditions") or []:
            _add((vc or {}).get("artifactPath"),
                 f"variant '{(vc or {}).get('name', '?')}' artifact",
                 required=True)
            # ...and everything that artifact POINTS AT. Packing the agent's
            # JSON alone shipped a bundle whose injections named vectors
            # absent on the far side; the study then failed after the queue
            # wait and the model load. Required, so an unresolvable
            # dependency refuses packaging here instead (B3, fail closed).
            # Parallel to Swift ``variantArtifactDependencies``.
            for dep_path, dep_label in _variant_artifact_dependencies(vc, root):
                _add(dep_path, dep_label, required=True)
            # A trained adapter's DATASET MANIFEST is a pinned input like any
            # other (LoRA readiness §0 amendment 1): verify() re-hashes it, so
            # it must be git-gated, snapshotted, and packed with the study.
            # Declared-only — legacy adapter variants gain no entry.
            _add(((vc or {}).get("trainingProvenance") or {})
                 .get("datasetManifestPath"),
                 f"variant '{(vc or {}).get('name', '?')}' training dataset "
                 "manifest", required=True)
    else:
        scenario_rel = d.get("multiAgentScenarioPath")
        _add(scenario_rel, "multi-agent scenario", required=True)
        # ...and every AGENT artifact the panel script names, plus everything
        # those point at. The scenario carries its agents by path INSIDE the
        # JSON, so packing the script alone shipped a bundle whose seats named
        # variants absent on the far side — the exact failure the
        # variantConditions branch above was fixed for, reproduced here
        # because this branch enumerates only the script. It surfaced on the
        # cluster as a FileNotFoundError for runs/model-variants/... after the
        # queue wait and the model load. Required, so an unresolvable seat
        # refuses PACKAGING instead (fail closed).
        for path, label in _panel_agent_dependencies(scenario_rel, root):
            _add(path, label, required=True)
    _add(d.get("judgeRubricFile"), "judge rubric", required=True)
    _add(d.get("reasoningStyleTaxonomyPath"), "reasoning-style taxonomy",
         required=True)
    # The SAE candidate roster is a pinned INPUT like the taxonomy: verify()
    # re-hashes it, so it must be git-gated at freeze, snapshotted into
    # pinned/, and packed into an evidence bundle. Declared-only — a study
    # with no roster gains no entry.
    sae_block = d.get("saeCandidates")
    _add((sae_block or {}).get("path") if isinstance(sae_block, dict) else None,
         "SAE candidate manifest", required=True)
    _add((d.get("humanBaseline") or {}).get("path"), "human baseline",
         required=True)
    _add((d.get("humanValidation") or {}).get("path"), "human validation",
         required=True)
    # Sweep inputs are manifest-declared data ON the verify() hash-pin
    # surface (firewall closure 2026-07-20: sweep.devPromptsHash +
    # sweep.batteryHash, pinned at freeze): a DECLARED file is required for
    # the declared sweep to run at all; engine-default paths (absent keys)
    # are the packer's optional-extras concern.
    sweep = (d.get("sweep") if isinstance(d.get("sweep"), dict) else {}) \
        if machinery else {}
    if sweep:
        _add(sweep.get("devPromptsFile"), "sweep dev prompts", required=True)
        _add(sweep.get("batteryFile"), "sweep battery", required=True)
        objective = (sweep.get("selection") or {}).get("objective") or {}
        # Choice instruments are inputs only when the objective READS them
        # (logprobShift) — a stale path under another metric is inert, and
        # a required-but-unread file must not refuse packaging.
        if objective.get("metric") == "logprobShift":
            _add(objective.get("choicePromptsFile"), "sweep choice prompts",
                 required=True)
            # Per-concept instruments (choicePromptsFiles, 2026-08-02):
            # every map value is a declared, required input — a bundle
            # without one cannot run the sweep it declares.
            choice_map = objective.get("choicePromptsFiles")
            if isinstance(choice_map, dict):
                for concept, rel in sorted(choice_map.items()):
                    if isinstance(rel, str):
                        _add(rel, f"sweep choice prompts '{concept}'",
                             required=True)
    if d.get("name"):
        _add(os.path.join("experiments", d["name"], "pinned"),
             "pinned-input snapshot", required=False)
    return out


def pinned_input_paths(d: dict, root: str | None = None) -> list[str]:
    """Absolute paths of every input the manifest declares — the files the
    stamped gitCommit must actually contain for the frozen manifest's hashes
    to be resolvable to bytes forever. Existence-filtered, deduplicated view
    over :func:`pinned_input_entries`; shared by the freeze cleanliness gate
    and the freeze snapshot."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in pinned_input_entries(d, root):
        real = os.path.realpath(entry.path)
        if real in seen or not os.path.exists(entry.path):
            continue
        seen.add(real)
        out.append(entry.path)
    return out


def _check_git_pin_cleanliness(name: str, d: dict, root: str | None) -> None:
    """Freeze-time reproducibility gate (policy as mechanism): every pinned
    input must be COMMITTED — tracked and unmodified — in the workspace's git
    repo, or the gitCommit freeze stamps cannot resolve the pinned hashes to
    bytes (the 'untracked manifest with completed runs' hole, caught at the
    only moment it matters). Skipped when the workspace is not a git work
    tree (unverifiable, e.g. test roots), AND when the workspace is its own
    work-tree root — there freeze auto-commits the pins itself
    (``_auto_commit_workspace``), so dirty pins are not a violation but the
    normal input to the freeze gesture. The gate therefore only fires where
    auto-commit is safety-skipped: a workspace that is a SUBDIRECTORY of a
    larger repository. ``freeze --force`` skips loudly."""
    base = paths.project_root() if root is None else root
    top = _git_toplevel(base)
    if top is None:
        return  # not a git work tree — unverifiable
    if os.path.realpath(top) == os.path.realpath(base):
        return  # standalone workspace repo — auto-commit will commit the pins
    pinned = pinned_input_paths(d, root)
    if not pinned:
        return
    try:
        status = subprocess.run(
            ["git", "-C", base, "status", "--porcelain", "--"] + pinned,
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    if status.returncode != 0:
        return  # e.g. a pinned path outside the repo — unverifiable, not dirty
    offending = sorted({
        line[3:].strip() for line in status.stdout.splitlines() if line.strip()
    })
    if offending:
        shown = ", ".join(offending[:8]) + (" …" if len(offending) > 8 else "")
        raise ExperimentStoreError(
            f"cannot freeze '{name}': pinned input(s) not committed to git — "
            f"{shown} — the stamped gitCommit must contain the bytes the "
            "manifest's hashes name (commit them, or freeze --force)")
