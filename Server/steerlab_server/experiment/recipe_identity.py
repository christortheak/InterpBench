"""Canonical full-recipe identity for extraction artifacts — the identity
Promote matches on, closing the provenance hole where two extractions could
"match" while representing different recipes (different reading position,
neutral projection, norm denominator, or grand-mean population) with the
newest silently winning.

CANONICAL FORM (cross-engine contract; this doc comment is mirrored verbatim
in ``Sources/ExperimentKit/RecipeIdentity.swift``):

recipeIdentityHash = SHA-256 hex of the UTF-8 canonical JSON of the recipe:
sorted keys (recursively), compact separators ("," and ":"), explicit nulls
for every absent field, raw UTF-8 (no ASCII escaping, no forward-slash
escaping). Top-level keys, in sorted order:

- "concept": the concept name.
- "extractionRendering": THE ONE OPTIONAL TOP-LEVEL KEY — present only when
  the recipe declares a CHAT-TEMPLATE extraction rendering, omitted entirely
  otherwise. This is deliberate and load-bearing: every recipe written before
  the rendering option existed rendered raw, and adding an explicit-null key
  would have changed every one of their identity hashes, breaking promotion
  for every frozen experiment. Absent = legacy raw, and an explicitly
  declared ``{"mode": "raw"}`` canonicalizes to absent because it IS the
  legacy semantics said out loud. When present the value is
  {"addGenerationPrompt": bool, "mode": "chatTemplate",
  "qwenThinkingEnabled": bool, "systemPrompt": string|null} with every inner
  field explicit (an identity may not depend on a default a later version
  could change).
- "extractionMethod": the substrate-independent method name in the MANIFEST
  vocabulary ("meanDifference" | "lat" | "emotionGrandMean"). Sidecar
  recipeMethod values map caaMeanDifference→meanDifference, repeLAT→lat,
  emotionGrandMean→emotionGrandMean; any other recorded method travels
  verbatim (it can never equal a manifest method).
- "grandMeanPopulation": for emotionGrandMean only — the FULL comparison
  population as [[conceptName, storiesSha256], …] sorted by conceptName then
  hash (code-point order); null for every other method.
- "methodParameters": method-specific recipe parameters, or null.
  designatedReference: {"referenceHash": storiesSha256, "referenceName":
  conceptName} — the designated reference IS part of the recipe, so two
  vectors built against different references must never share an identity
  (external review 2026-07-31, finding 2). Null for every other method
  (LAT is the fixed first principal component on both engines), which
  preserves every pre-existing recipe hash.
- "modelID": the HF model id.
- "neutralProjection": {"basisHash": string|null, "count": int|null,
  "explainedVariance": decimal-string|null, "mode": "none" | "legacyPooled" |
  "tokenBankFixedCount" | "tokenBankExplainedVariance"} — all inner fields
  explicit. explainedVariance is the decimal string exactly as recorded (a
  string, so float formatting can never diverge across engines).
- "normCorpusHash": SHA-256 of the pinned neutral corpus when
  residualNormSource is "neutral-corpus" or "neutral-token-bank"; null
  otherwise.
- "readingPosition": {"mode": "lastToken" | "meanFromToken" |
  "offsetFromEnd" | "lastContentToken" | "turnCloseToken" |
  "postInstruction", "parameter": int|null} — the pool-from token index for
  meanFromToken, the backward offset for offsetFromEnd, the post-instruction
  index for postInstruction, null for the rest. ``offsetFromEnd`` with
  parameter 0 canonicalizes to ``{"mode": "lastToken", "parameter": null}``:
  it names the identical token, so declaring it that way must not split an
  identity away from an otherwise-identical last-token recipe.
- "residualNormSource": the canonical source token ("neutral-corpus" |
  "extraction-stimuli" | "neutral-token-bank"). A sidecar value is
  canonicalized by truncating at the first space (the Swift experiment writer
  historically embedded a corpus-hash prefix after it), and the Swift
  grand-mean self-measured label "multi-concept-corpus" canonicalizes to
  "extraction-stimuli" (the server records the same denominator recipe —
  norms measured on the extraction stimuli themselves — under that name).
- "revision": the pinned model revision, or null.
- "schema": 1 (the integer literal).
- "stimulusSetHash": the concept's pinned stimulus-set hash.

Substrate is deliberately OUTSIDE this identity: it remains a separate match
criterion (a CUDA artifact must never satisfy an MLX recipe silently),
exactly as before.

ARTIFACT-PINNED concepts (manifest method ``pinnedArtifact``) travel through
the same canonical form with ``extractionMethod`` = ``"pinnedArtifact"``: the
identity then says "these exact pinned bytes, materialized for this model at
this revision and read at this position", which is the only recipe there is
when the direction was derived post-hoc. Their ``residualNormSource`` /
``normCorpusHash`` come from the PIN BLOCK (copied from the artifact's sidecar
at attach), not from the study's neutral corpus — the norms are the artifact's,
so the denominator provenance must be too.
"""

from __future__ import annotations

import hashlib
import json

SCHEMA = 1

_METHOD_MAP = {
    "caaMeanDifference": "meanDifference",
    "repeLAT": "lat",
    "emotionGrandMean": "emotionGrandMean",
}


def canonical_json(components: dict) -> str:
    """The canonical JSON string (see the canonical-form contract above).
    ``components`` is the flat field dict produced by
    :func:`required_identity` / :func:`candidate_identity`; the Swift twin
    (`RecipeIdentity.canonicalJSON`) hand-builds byte-identical output."""
    population = components.get("grandMeanPopulation")
    payload = {
        "concept": components["concept"],
        "extractionMethod": components["extractionMethod"],
        "grandMeanPopulation": (
            sorted([list(pair) for pair in population])
            if population is not None else None),
        "methodParameters": components.get("methodParameters"),
        "modelID": components["modelID"],
        "neutralProjection": {
            "basisHash": components.get("projectionBasisHash"),
            "count": components.get("projectionCount"),
            "explainedVariance": components.get("projectionExplainedVariance"),
            "mode": components["projectionMode"],
        },
        "normCorpusHash": components.get("normCorpusHash"),
        "readingPosition": {
            "mode": components["readingPositionMode"],
            "parameter": components.get("readingPositionParameter"),
        },
        "residualNormSource": components["residualNormSource"],
        "revision": components.get("revision"),
        "schema": SCHEMA,
        "stimulusSetHash": components["stimulusSetHash"],
    }
    # The ONE optional key. Absent (and explicitly-raw) recipes must hash
    # byte-identically to every recipe written before this option existed, so
    # the key is added only for a chat-template rendering. See the
    # canonical-form contract above. Swift twin: `RecipeIdentity.canonicalJSON`
    # appends the same fragment between "extractionMethod" and
    # "grandMeanPopulation" in sorted-key order.
    rendering = components.get("extractionRendering")
    if rendering is not None:
        payload["extractionRendering"] = rendering
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def identity_hash(components: dict) -> str:
    return hashlib.sha256(canonical_json(components).encode("utf-8")).hexdigest()


def diff_fields(required: dict, candidate: dict) -> list[str]:
    """Human-readable differences between two identities, as dotted canonical
    field paths with both values — e.g. ``"revision (manifest: null,
    artifact: 8f2a66d1c9e0…)"`` or ``"neutralProjection.mode (manifest: none,
    artifact: legacyPooled)"``. Compares the CANONICAL payloads (the exact
    bytes the hash covers), so the diff sees precisely the normalization the
    match saw — population sorting, method-vocabulary mapping, explicit
    nulls. Empty means the identities hash identically. The Swift twin
    (`RecipeIdentity.diffFields`) emits the same wording shape."""
    diffs: list[str] = []

    def walk(path: str, a, b) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(f"{path}.{key}" if path else key, a.get(key), b.get(key))
        elif a != b:
            diffs.append(
                f"{path} (manifest: {_display(a)}, artifact: {_display(b)})")

    walk("", json.loads(canonical_json(required)),
         json.loads(canonical_json(candidate)))
    return diffs


def _display(value) -> str:
    """Compact single-token rendering for diff messages: explicit ``null``,
    long strings (hashes, revisions) truncated to a 12-char prefix, compound
    values as truncated canonical JSON."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value if len(value) <= 16 else value[:12] + "…"
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return text if len(text) <= 48 else text[:44] + "…"


def required_identity(manifest, ref) -> dict:
    """The full-recipe identity this manifest's pinned recipe demands for one
    concept. Deterministic from pins alone (no filesystem reads): the
    extraction paths derive the norm denominator from exactly these pins, so
    the prediction here matches what a faithful extraction stamps. Raises
    ``ValueError`` when the manifest's own pins are incomplete."""
    reading_mode, reading_parameter = canonical_reading(
        ref.options.reading_position)
    pc_count = ref.options.neutral_pc_count or 0
    # A pinned neutral corpus is the norm denominator on both engines
    # (extract / extract_grand_mean use it whenever present); without one,
    # norms come from the extraction stimuli themselves.
    source = "neutral-corpus" if manifest.neutral_corpus_hash else "extraction-stimuli"
    if ref.options.method.is_pinned_artifact:
        # An artifact-pinned concept CARRIES its denominator: the norms come
        # from the pinned artifact, not from anything this study measures, so
        # the identity must demand the artifact's provenance rather than the
        # study's neutral corpus. Both values were copied from the sidecar at
        # attach (and re-checked at verify), so this is still pins-only.
        block = ref.vector_artifact or {}
        source = block.get("residualNormSource") or source
    norm_corpus_hash = (manifest.neutral_corpus_hash
                        if source == "neutral-corpus" else None)
    if ref.options.method.is_pinned_artifact and source == "neutral-corpus":
        norm_corpus_hash = (ref.vector_artifact or {}).get(
            "normCorpusHash") or manifest.neutral_corpus_hash
    population = None
    if ref.options.method.is_grand_mean:
        corpus = manifest.grand_mean_corpus
        if corpus is None:
            raise ValueError(
                f"grand-mean concept '{ref.name}' has no pinned "
                "grandMeanCorpus — re-attach with method emotionGrandMean")
        population = []
        for member in corpus.concepts:
            member_hash = corpus.hashes.get(member)
            if not member_hash:
                raise ValueError(
                    f"grandMeanCorpus member '{member}' has no pinned hash — "
                    "re-attach before promoting")
            population.append([member, member_hash])
    method_parameters = None
    if ref.options.method.is_designated_reference:
        pin = ref.designated_reference or {}
        if not pin.get("name") or not pin.get("hash"):
            raise ValueError(
                f"designated-reference concept '{ref.name}' has no pinned "
                "reference — re-attach with --reference before promoting")
        method_parameters = {"referenceHash": pin["hash"],
                             "referenceName": pin["name"]}
    return {
        "concept": ref.name,
        "modelID": manifest.model_id,
        "revision": manifest.model_revision,
        "extractionMethod": ref.options.method.value,
        "stimulusSetHash": ref.stimulus_set_hash,
        "readingPositionMode": reading_mode,
        "readingPositionParameter": reading_parameter,
        "projectionMode": "legacyPooled" if pc_count > 0 else "none",
        "projectionCount": pc_count if pc_count > 0 else None,
        "projectionExplainedVariance": None,
        "projectionBasisHash": None,
        "residualNormSource": source,
        "normCorpusHash": norm_corpus_hash,
        "grandMeanPopulation": population,
        "methodParameters": method_parameters,
        "extractionRendering": rendering_fragment(
            getattr(ref.options, "extraction_rendering", None)),
    }


def canonical_reading(position) -> tuple[str, int | None]:
    """A reading position's ``(mode, parameter)`` for the identity.

    ``offsetFromEnd(0)`` canonicalizes to ``("lastToken", None)``: it names
    the identical token, so declaring the offset form must not split an
    identity away from an otherwise-identical last-token recipe. Swift twin:
    ``RecipeIdentity.canonicalReading``.
    """
    mode = position.identity_mode
    parameter = position.identity_parameter
    if mode == "offsetFromEnd" and parameter == 0:
        return ("lastToken", None)
    return (mode, parameter)


def rendering_fragment(rendering) -> dict | None:
    """The identity fragment for an extraction rendering, or ``None`` to omit
    the key (absent, or an explicitly declared legacy raw)."""
    from ..steering.extraction_rendering import canonical_identity_fragment
    return canonical_identity_fragment(rendering)


def candidate_identity(sidecar: dict) -> tuple[dict | None, list[str]]:
    """Compute the identity from an artifact's recorded provenance (the raw
    sidecar dict). Returns ``(components, [])`` when every canonical field is
    provable, else ``(None, sorted_missing_field_names)`` — the caller must
    refuse, never guess."""
    missing: set[str] = set()

    recipe_method = sidecar.get("recipeMethod")
    if recipe_method is not None:
        method = _METHOD_MAP.get(recipe_method, recipe_method)
    else:
        method = sidecar.get("extractionMethod")
    if method is None:
        missing.add("extractionMethod")

    reading = _parse_reading_label(sidecar.get("readingPosition"))
    if reading is None:
        missing.add("readingPosition")

    rendering, rendering_ok = _parse_sidecar_rendering(sidecar)
    if not rendering_ok:
        missing.add("extractionRendering")

    projection = None
    description = sidecar.get("neutralProjection")
    if description is None:
        # Pre-neutralProjection sidecars recorded the legacy pooled
        # projection as bare "top-K neutral PCs".
        description = sidecar.get("confoundProjection")
    if description is not None:
        projection = _parse_projection(description)
    if projection is None:
        missing.add("neutralProjection")

    source = None
    recorded = sidecar.get("residualNormSource")
    if isinstance(recorded, str) and recorded.split():
        # The Swift experiment writer embeds a corpus-hash prefix after a
        # space; the grand-mean self-measured label unifies with the
        # server's (see the canonical-form contract).
        token = recorded.split()[0]
        source = "extraction-stimuli" if token == "multi-concept-corpus" else token
    else:
        missing.add("residualNormSource")

    norm_corpus_hash = None
    if source in ("neutral-corpus", "neutral-token-bank"):
        norm_corpus_hash = sidecar.get("neutralCorpusHash")
        if norm_corpus_hash is None:
            missing.add("normCorpusHash")

    population = None
    if method == "emotionGrandMean":
        recorded_population = sidecar.get("grandMeanPopulation")
        if isinstance(recorded_population, dict) and recorded_population:
            population = [[name, value]
                          for name, value in recorded_population.items()]
        else:
            missing.add("grandMeanPopulation")

    method_parameters = None
    if method == "designatedReference":
        recorded_ref = sidecar.get("designatedReference")
        if isinstance(recorded_ref, dict) and recorded_ref.get("name") \
                and recorded_ref.get("hash"):
            method_parameters = {"referenceHash": recorded_ref["hash"],
                                 "referenceName": recorded_ref["name"]}
        else:
            missing.add("designatedReference")

    if missing:
        return None, sorted(missing)
    reading_mode, reading_parameter = reading
    projection_mode, projection_count, projection_ev = projection
    return {
        "concept": sidecar.get("concept"),
        "modelID": sidecar.get("modelID"),
        "revision": sidecar.get("revision"),
        "extractionMethod": method,
        "stimulusSetHash": sidecar.get("stimulusSetHash"),
        "readingPositionMode": reading_mode,
        "readingPositionParameter": reading_parameter,
        "projectionMode": projection_mode,
        "projectionCount": projection_count,
        "projectionExplainedVariance": projection_ev,
        "projectionBasisHash": None,
        "residualNormSource": source,
        "normCorpusHash": norm_corpus_hash,
        "grandMeanPopulation": population,
        "methodParameters": method_parameters,
        "extractionRendering": rendering,
    }, []


def _parse_reading_label(label) -> tuple[str, int | None] | None:
    """STRICT reading-position parse for identity — unlike
    ``reading_position.from_label`` this never falls back to last-token: an
    unrecognized label makes the field unprovable."""
    from ..steering.reading_position import parse_label_strict
    position = parse_label_strict(label)
    if position is None:
        return None
    return canonical_reading(position)


def _parse_sidecar_rendering(sidecar: dict) -> tuple[dict | None, bool]:
    """``(identity fragment, ok)`` from an artifact's stamped
    ``extractionRendering``.

    Absent is LEGACY RAW — provable, and it contributes nothing to the
    identity, which is exactly why every pre-option artifact keeps its hash.
    An unparseable block is NOT provable: the caller must refuse rather than
    silently read it as raw, because reading a templated artifact as raw is
    the confusion this whole option exists to end.
    """
    from ..steering.extraction_rendering import (ExtractionRenderingError,
                                                 canonical_identity_fragment)
    from ..steering.extraction_rendering import from_json as parse
    raw = sidecar.get("extractionRendering")
    if raw is None:
        return None, True
    try:
        return canonical_identity_fragment(parse(raw)), True
    except ExtractionRenderingError:
        return None, False


def _parse_projection(description) -> tuple[str, int | None, str | None] | None:
    """Parse a sidecar's neutral-projection description into the canonical
    (mode, count, explainedVariance) triple. None = unrecognized — the caller
    reports the field as unprovable, never guesses. Mirrors
    ``RecipeIdentity.parseProjection`` exactly."""
    if not isinstance(description, str):
        return None
    if description == "none":
        return ("none", None, None)
    text = description
    if text.startswith("legacy-pooled "):
        text = text[len("legacy-pooled "):]
    if text.startswith("top-") and text.endswith(" neutral PCs"):
        middle = text[len("top-"):-len(" neutral PCs")]
        if middle.isdigit():
            return ("legacyPooled", int(middle), None)
        return None
    if (description.startswith("token-bank fixed-count ")
            and description.endswith(" PCs")):
        middle = description[len("token-bank fixed-count "):-len(" PCs")]
        if middle.isdigit():
            return ("tokenBankFixedCount", int(middle), None)
        return None
    if description.startswith("token-bank explained-variance "):
        value = description[len("token-bank explained-variance "):].strip()
        if value:
            return ("tokenBankExplainedVariance", None, value)
        return None
    return None
