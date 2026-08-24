"""Read + verify experiment manifests (parallel to Swift ``ExperimentManifest``
/ ``ExperimentStore``).

An experiment pins **stimuli by hash + extraction options** — the recipe, not
the vector bytes — plus a model revision. Runs re-derive vectors deterministically
and stamp the experiment's content hash, so drift in a pinned stimulus file
surfaces as a verification failure rather than a silent change (the proposal's
circularity firewall). This module reproduces the load + the verify gate; it
does **not** re-implement the Swift freeze byte-for-byte (Python stamps run
directories with its own content hash). Cross-engine safety rests on the
stimulus/corpus **SHA-256s matching**, which they do (see
:mod:`steerlab_server.steering.stimulus_set`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import NamedTuple

from ..steering import vector_math as vm
from ..steering.extraction_rendering import RAW_RENDERING, ExtractionRendering
from ..steering.extraction_rendering import from_json as rendering_from_json
from ..steering.reading_position import (LAST_TOKEN, ReadingPosition, from_label,
                                         mean_from_token)
from ..steering.stimulus_set import StimulusSet, load_texts
from ..steering.vector_store import SUBSTRATE as _THIS_SUBSTRATE
from . import paths

# The closed ``ordinalAggregation`` vocabulary for the ``ordinalScale``
# outcome instrument (kept torch-free here; ``logprob.ORDINAL_AGGREGATIONS``
# mirrors it next to the aggregation math, and Swift
# ``ExperimentStore.knownOrdinalAggregations`` is the cross-engine twin).
KNOWN_ORDINAL_AGGREGATIONS = ("expectedValue", "argmax")

#: How many imported SAE features ONE condition may mix when the manifest
#: declares no ``maxSAEMixtureFeatures`` (SAE-VECTOR-INTERVENTION proposal r2
#: §8 P2-10: "preregistered SPARSE multi-feature mixtures"). A cap as DATA
#: with a default, not a rule in code: a denser mixture stays perfectly legal,
#: it just has to be declared in the manifest — before behavior is measured —
#: instead of arrived at. Counts SAE-backed slots only, so it can never change
#: the verdict on a condition that seats none.
DEFAULT_MAX_SAE_MIXTURE_FEATURES = 4

# Coherence-gate wording (2026-07-22 incident: a frozen study whose pipeline
# declared evaluate carried ``evaluation: null`` and died at the evaluate
# stage after generation). Cross-engine identical string — Swift twin:
# ``ExperimentStore.evaluateWithoutJudgingViolation``.
EVALUATE_WITHOUT_JUDGING_MESSAGE = (
    "the pipeline declares evaluate but the study declares no paired "
    "judging — pin at least one judge and a rubric, or remove evaluate "
    "from the pipeline")

#: The value of ``caseFamily`` that still selects a measurement instrument.
#: The ONE deprecated implicit selection left in the manifest vocabulary.
IMPLICIT_ENDPOINT_CASE_FAMILY = "sentencing"

#: What every site that fires the deprecated trigger says, on both engines and
#: at every site, byte for byte. ONE sentence to match on: an agent that has to
#: learn four spellings of the same deprecation has not been told anything.
#: Swift twin: ``ExperimentStore.implicitCaseFamilyAdvisory``.
#:
#: Advisory, never a refusal — the whole point of the deprecation is that
#: manifests which already depend on the trigger keep producing the same
#: numbers. What changes is that they say so, in the run log, in the run
#: directory's ``advisories.txt``, and in the CLI envelope under
#: ``deprecatedImplicitSelection``.
IMPLICIT_CASE_FAMILY_ADVISORY = (
    "caseFamily 'sentencing' selected the built-in duration endpoint "
    "implicitly — declare numericParser instead; this implicit selection is "
    "deprecated. The shipped registry entry 'sentencing-months' "
    "(prompts/parsers/parser-registry.json) reproduces this parser exactly.")


def implicit_case_family_endpoint(manifest) -> bool:
    """Does the deprecated magic trigger ACTUALLY fire for this manifest?

    ONE definition for every site that asks, because two would drift and the
    advisory would then be right at one site and wrong at another.

    Model-output studies: true only when the study declares no
    ``numericParser`` — the declared mechanism always wins — AND names the one
    case family with a built-in parser. A study that declares a parser gets no
    advisory, because nothing was selected implicitly.

    Multi-agent studies: true on the case family ALONE. Their trigger site is
    the panel-effects decomposition's built-in ``months`` endpoint
    (``tasks._run_multi_agent_study``), which a declared ``numericParser`` does
    not displace — so a manifest that declares one still fires the trigger
    there, and the advisory has to say so.

    Swift twin: ``ExperimentManifest.usesImplicitCaseFamilyEndpoint``.
    """
    if manifest.case_family != IMPLICIT_ENDPOINT_CASE_FAMILY:
        return False
    if manifest.study_kind == "multiAgent":
        return True
    return not (manifest.numeric_parser or "").strip()


@dataclass
class ExtractionOptions:
    method: vm.ExtractionMethod = vm.ExtractionMethod.MEAN_DIFFERENCE
    reading_position: ReadingPosition = LAST_TOKEN
    neutral_pc_count: int | None = None
    #: HOW the stimulus reaches the model (``extractionRendering``). ABSENT is
    #: legacy raw and is never retro-applied: a manifest without the key
    #: renders, hashes, and verifies exactly as it always has. See
    #: :mod:`steerlab_server.steering.extraction_rendering`.
    extraction_rendering: ExtractionRendering = RAW_RENDERING

    @classmethod
    def from_json(cls, d: dict | None) -> "ExtractionOptions":
        d = d or {}
        method = vm.ExtractionMethod(d.get("method", "meanDifference")) \
            if d.get("method") else vm.ExtractionMethod.MEAN_DIFFERENCE
        return cls(method=method,
                   reading_position=_parse_reading_position(d.get("readingPosition")),
                   neutral_pc_count=d.get("neutralPCCount"),
                   extraction_rendering=rendering_from_json(
                       d.get("extractionRendering")))


@dataclass
class ConceptRef:
    name: str
    stimulus_set_hash: str
    options: ExtractionOptions = field(default_factory=ExtractionOptions)
    # Measurement-side pin (cross-engine contract key "validationHash"):
    # SHA-256 of the concept's held-out validation.jsonl at attach time, or
    # null when the concept had none. ``validation_pinned`` distinguishes a
    # legacy pin block (key absent — verify passes, freeze advises) from a
    # deliberate null pin (key present — a validation file APPEARING later is
    # drift, exactly like stimulus drift).
    validation_hash: str | None = None
    validation_pinned: bool = False
    # designatedReference concepts only: {"name", "hash"} of the pinned
    # reference stories corpus (cross-engine contract key
    # "designatedReference"). The vector is mean(concept stories) −
    # mean(reference stories), so the reference is recipe data and drifting
    # reference bytes are a verify violation exactly like stimulus drift.
    designated_reference: dict | None = None
    # ARTIFACT-PINNED concepts only (method "pinnedArtifact"; cross-engine
    # contract key "vectorArtifact"). A recipe concept pins stimuli and
    # RE-DERIVES its vector every run; an artifact-pinned concept pins the
    # VECTOR BYTES, because the direction was derived post-hoc from other
    # artifacts (e.g. family-grand-mean centring) and no stimulus recipe can
    # reproduce it. Keys:
    #
    #   path                — workspace-relative, EXTENSION-LESS locator
    #                         (ArtifactIdentity convention: <path>.safetensors
    #                         + <path>.json)
    #   sha256TensorHash    — SHA-256 of <path>.safetensors raw bytes
    #   sha256SidecarHash   — SHA-256 of <path>.json raw bytes
    #   sourceMethod        — the extractionMethod recorded in that sidecar.
    #                         It decides the DATA questions the lifecycle asks
    #                         (where stimuli/validation live, whether
    #                         validation is contrastive or population-based).
    #   sourceConcept       — the concept whose stimuli and held-out
    #                         validation.jsonl the probe reads; defaults to
    #                         the manifest concept name. Post-hoc derived
    #                         directions are usually renamed ("crit" →
    #                         "crit-gm") while keeping the base concept's
    #                         held-out data.
    #   residualNormSource  — the artifact's norm denominator provenance, and
    #   normCorpusHash        the neutral corpus it was measured on. Copied
    #                         from the sidecar at attach so the manifest is
    #                         self-describing about α-in-norm units and the
    #                         recipe identity needs no filesystem read.
    #                         Omitted when the sidecar records none.
    #
    # BOTH file hashes are re-checked against the bytes at every verify() and
    # again at materialization: drift refuses loudly, exactly like stimulus
    # drift.
    vector_artifact: dict | None = None

    @property
    def is_pinned_artifact(self) -> bool:
        return self.options.method.is_pinned_artifact

    @property
    def effective_method(self) -> vm.ExtractionMethod:
        """The method whose DATA semantics apply — the artifact's recorded
        source method for a pinned concept, else the declared method. Ask
        this whenever the question is "where do this concept's stimuli /
        validation live" or "what does validation MEAN here"; ask
        ``options.method`` when the question is "how is the vector produced"
        (a pinned concept produces nothing — it materializes)."""
        if not self.is_pinned_artifact:
            return self.options.method
        recorded = (self.vector_artifact or {}).get("sourceMethod")
        try:
            return vm.ExtractionMethod(recorded)
        except ValueError:
            return vm.ExtractionMethod.MEAN_DIFFERENCE

    @property
    def data_concept(self) -> str:
        """The concept name under which this concept's stimuli and held-out
        validation.jsonl live (``vectorArtifact.sourceConcept`` for a pinned
        concept, else its own name)."""
        if self.is_pinned_artifact:
            recorded = (self.vector_artifact or {}).get("sourceConcept")
            if isinstance(recorded, str) and recorded.strip():
                return recorded.strip()
        return self.name


@dataclass
class GrandMeanCorpus:
    """The pinned population for grand-mean extraction. A grand-mean vector is
    mean(concept stories) − mean(ALL corpus stories), so the vector depends on
    every member of the corpus — membership and every member's stories.jsonl
    hash must be pinned, not just the target concept's own file."""
    concepts: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class Slot:
    concept: str
    layer: int
    #: α when steering, λ when ablating.
    alpha: float
    #: "add" (steer) or "ablate". None means "add", and an explicit "add" is
    #: never written back: manifest bytes are the content hash, so a key
    #: appearing on every existing condition would re-identify every frozen
    #: study in the workspace. Cross-engine twin of Swift `Slot.mode`.
    mode: str | None = None

    @property
    def effective_mode(self) -> str:
        return self.mode or "add"

    @property
    def is_ablation(self) -> bool:
        return self.effective_mode == "ablate"


@dataclass
class Condition:
    name: str
    slots: list[Slot] = field(default_factory=list)
    band_width: int = 1
    #: False here and True in Swift's `Condition.init` — an IN-CODE default,
    #: not a document one, and deliberately left alone by the Phase-1a G6
    #: repair: the only production caller that omits it is the synthesized
    #: slot-less baseline (`effective_conditions`), which carries no α for
    #: the unit to apply to. Every condition READ from a document passes the
    #: value explicitly (`from_dict`), and every NEW declaration must state it
    #: (`experiment_store._condition_entry`).
    alpha_in_norm_units: bool = False
    # "randomMatchedNorm": each slot injects a deterministic random direction
    # norm-matched to the named concept's vector at that layer — the magnitude/
    # noise control cell, expressible as data instead of ad-hoc code.
    #
    # "randomDirectionAblation": the ABLATION analogue. Norm-matching is
    # meaningless for a projection — ablation removes whatever is present
    # regardless of the direction's length — so the control removes a random
    # DIRECTION instead. It answers the question a concept ablation raises: is
    # the effect specific to this direction, or does removing any rank-1
    # subspace of the residual stream do it? Same seeding convention, so the
    # cell is reproducible across re-runs of a frozen study.
    control_type: str | None = None
    # Neutral-PC basis pin (cross-engine contract key "neutralPCBasisHash"):
    # SHA-256 of the basis FILE's raw bytes. Previously written but never
    # checked; verify() now enforces it against the bytes at
    # ``neutralPCBasisPath`` (a directory path resolves to its
    # neutral-pc-basis.json, mirroring ``neutral.load_basis``).
    neutral_pc_basis_path: str | None = None
    neutral_pc_basis_hash: str | None = None


@dataclass
class VariantCondition:
    """A condition backed by a saved ModelVariant artifact (base model +
    adapters + injections + neutral basis + prompt settings). Parallel to Swift
    ``ExperimentManifest.VariantCondition``.

    FORWARD-REFERENCED form (seamless pipeline stage 4, 2026-07-18):
    ``from_promotion`` = ``{"concept": <name>}`` declares "the agent this
    experiment's sweep promotes for CONCEPT under the declared criterion"
    INSTEAD of a concrete artifact pin — declarable (and freezable) before
    the agent exists. It resolves at RUN time against promotion evidence
    (the birth certificate: criterion promotion of the concept whose sweep
    run + winning cell match the manifest's current selection evidence,
    under the current epoch), and the resolved path + hash are recorded in
    the run directory (``forward-resolutions.json``) — runs are the
    immutable evidence, so a frozen manifest never needs editing. A
    condition declares exactly ONE identity: fromPromotion and a concrete
    artifact pin together are a verify violation.

    TRAINING PROVENANCE (cross-engine contract key ``trainingProvenance``,
    cluster-LoRA readiness §0 amendment 1 + contract §9): an adapter enters a
    study as a VARIANT, so the manifest is where a trained adapter's
    provenance becomes evidence. The block pins the training DATASET into the
    freeze ``verify()`` surface — post-freeze drift in the dataset manifest
    (or in the adapter's own sidecar) is a violation, exactly like stimulus
    drift — and declares the ex ante matched control (amendment 2). Kept a
    plain dict like the neighbouring optional blocks (``artifact``,
    ``from_promotion``), validated where used. Shape::

        {"datasetBundleID", "datasetManifestPath", "datasetManifestHash",
         "adapterSidecarHash", "evidenceGrade",
         "matchedControl": {"variant", "kind"} | null}
    """
    name: str
    artifact_path: str
    artifact_hash: str
    artifact: dict = field(default_factory=dict)
    from_promotion: dict | None = None
    training_provenance: dict | None = None


@dataclass
class EvaluationSpec:
    kind: str = "none"          # none | pairedJudge
    judge_model: str = ""
    judge_prompt: str = ""
    structured_prompt: str | None = None


@dataclass
class JudgeRef:
    """One member of the judge panel (manifest key ``judges``). ``kind`` is
    ``claude`` (Anthropic API), ``openrouter`` (any OpenRouter-served model,
    provider-pinned), or ``local`` (a served open-weight judge); ``model``
    may be null (kind default — except openrouter, which has no default and
    must pin an explicit slug). ``provider`` is openrouter-only and
    REQUIRED there: the same slug can be served by different backends with
    different outputs, so an unpinned provider is not a pinned judge. Each
    judgment record is stamped with ``name`` so per-judge outputs and
    inter-judge agreement are reportable.

    LOCAL judges additionally pin ``revision`` (model commit) and, on this
    engine, ``dtype`` (the loader takes one) — cross-engine manifest keys
    (2026-07-23), omit-when-nil so legacy manifests keep their bytes. A
    blank local-judge revision is pinned from the STUDY pin at freeze when
    the judge resolves to the study model; judgment artifacts stamp the
    judge model+revision that actually judged."""
    name: str
    kind: str = "claude"        # claude | openrouter | local
    model: str | None = None
    provider: str | None = None  # openrouter: the pinned serving provider
    revision: str | None = None  # local: pinned model commit
    dtype: str | None = None     # local: loader dtype (server engine only)


@dataclass
class HumanValidation:
    """A pinned, human-labeled subset for judge validation (manifest key
    ``humanValidation``): JSONL rows in the SAME id-keyed shape as judge
    outputs (``promptID``, ``seed``, ``condition``, ``outcome``). Pinned by
    hash like every other input; evaluation reports judge-vs-human agreement
    per judge when present."""
    path: str
    hash: str


@dataclass
class HumanBaseline:
    """A pinned human-effect table (CSV) for the alien-residual computation
    R = delta_model − delta_human. Pinned by hash like every other input."""
    path: str
    hash: str


@dataclass
class ReaderRef:
    """A pinned RepE reader artifact (measurement instrument) used as the
    ``repeReaderScore`` outcome instrument. Pinned by file hash like
    ``humanBaseline``; additionally substrate-gated in :meth:`Manifest.verify`
    because a reader fitted on another engine's activations measures nothing
    on this one."""
    path: str
    hash: str
    concept: str


@dataclass
class PromotionRule:
    """Screen→confirm promotion gate (study guide funnel). All criteria must
    hold for a concept to enter the confirm phase."""
    fdr_threshold: float = 0.05
    dose_monotone: bool = True
    exceeds_random_floor: bool = True
    capability_gate: str | None = None


@dataclass
class Manifest:
    name: str
    model_id: str
    model_revision: str | None = None
    #: The numeric precision this study's model is pinned to run in
    #: (``bfloat16``/``float16``/``float32``, or the bf16/fp16/fp32 aliases).
    #: Absent means "let the device decide" — `default_dtype` resolves it,
    #: which is what every study did before this key existed.
    #:
    #: Greedy decoding is not precision-proof: at a near-tie between two
    #: tokens, bf16 and fp16 round differently, the argmax flips, and the
    #: continuation diverges. The same argument that requires a JUDGE to pin
    #: its dtype applies at least as strongly to the model that produced the
    #: text being judged.
    #:
    #: SERVER-HONORED, Swift-validated-but-unconsumed — the same shape as
    #: ``judges[].dtype``. MLX study models are quantized repos with no
    #: loader dtype to set; the Mac validates the pin at freeze (it is the
    #: authoring surface) so a manifest cannot reach the cluster carrying a
    #: dtype that will refuse there after a queue wait.
    dtype: str | None = None
    status: str = "draft"
    description: str = ""
    task_description: str | None = None
    prompt_mode: str = "chatAssistant"
    system_prompt: str | None = None
    qwen_thinking_enabled: bool = False
    temperature: float = 0.0
    max_tokens: int = 512
    seeds: list[int] = field(default_factory=lambda: [0])
    task_prompts_file: str | None = None
    task_prompts_hash: str | None = None
    neutral_corpus_hash: str | None = None
    # Study-level markers pin (cross-engine contract key "markersHash"):
    # aggregate hash over every attached concept's markers.json (see
    # :func:`markers_aggregate_hash`), pinned at FREEZE from the resolved
    # files; null when no concept has one. ``markers_pinned`` distinguishes
    # legacy manifests (key absent) from a deliberate null pin.
    markers_hash: str | None = None
    markers_pinned: bool = False
    # J-lens readout pin (cross-engine contract key "jlensReadout"; SERVER-ONLY
    # by rule — imported lens artifacts are PyTorch/HF-native, so the Swift
    # engine renders this block and never produces one). The whole block is
    # measurement-side: it decides WHAT gets read out of the residual stream,
    # so every field in it is pinned at freeze and drift is a verify()
    # violation. Absent = no readout, no violation.
    jlens_readout: dict | None = None
    # Retain the exact sampled token ids on every generation record
    # (cross-engine contract key "recordTokenIDs"; Swift carries it verbatim).
    #
    # NOT a measurement-side pin: it cannot change any measured value, only
    # what the run KEEPS. What it buys is that a completed run stays
    # replayable — teacher-forced replay of the realized sequence reproduces
    # the states a stepped decode wrote into the KV cache, but only if the
    # EXACT ids are fed back, and re-deriving them from the stored text is not
    # a round trip (a generation that stops naturally ends with
    # <end_of_turn>, the streamer skips it, and re-tokenizing silently returns
    # a shorter sequence — measured on gemma-3-4b-it 2026-08-15).
    #
    # Useful well beyond J-lens work, which is why it is a study-level choice
    # rather than part of the readout block: any post-hoc activation analysis
    # (probes, readers, SAE latents, optvec J-space) needs the same exact
    # sequence, and the id COUNT is the only way to tell a generation that
    # stopped on EOS from one that hit maxTokens — a distinction the stored
    # text cannot express and `wordCount` only approximates.
    #
    # Off by default: it is additive bytes (~7 per token) on every record, and
    # a study that never looks inside the model does not need it. Retention is
    # not retroactive, so `data check` advises when a study's own declarations
    # imply it will be wanted.
    record_token_ids: bool = False
    # Reasoning-style taxonomy pin (cross-engine contract keys
    # "reasoningStyleTaxonomyPath" + "reasoningStyleTaxonomyHash"): a
    # versioned feature file under prompts/taxonomies/, pinned at set time by
    # SHA-256 of its raw bytes. Drift after pinning is a verify() violation
    # like every other measurement-side input; ABSENT (both None) = no
    # reasoning-style scoring, no violation.
    reasoning_style_taxonomy_path: str | None = None
    reasoning_style_taxonomy_hash: str | None = None
    # Declared numeric-answer parser (cross-engine contract keys
    # "numericParser" + "parserRegistryHash"): the name of an entry in the
    # workspace parser registry (prompts/parsers/parser-registry.json) that
    # parses this study's numeric outcome instead of the built-in duration
    # parser. Freeze pins the registry file's SHA-256; drift after pinning is
    # a verify() violation. ABSENT = the historical behavior (caseFamily ==
    # "sentencing" -> built-in parse_months), no violation, no advisory.
    numeric_parser: str | None = None
    parser_registry_hash: str | None = None
    # SAE candidate-manifest pin (cross-engine contract key "saeCandidates":
    # {path, hash}; SERVER-ONLY today — SAE import is Gemma/PyTorch work, so
    # no Swift twin writes this key). The candidate roster decides WHICH
    # features a study may seat and records the discovery evidence that
    # justified each nomination, so it is a measurement-side input: freeze
    # pins it and drift afterwards is a verify() violation, exactly like
    # markers. ABSENT = no roster, no violation (additive and optional —
    # every legacy manifest verifies and freezes unchanged, and the content
    # hash is computed from raw so absent means byte-identical).
    sae_candidates: dict | None = None
    # Declared record-exclusion rules (cross-engine contract key
    # "exclusionRules"): a closed rule vocabulary applied at ANALYSIS time
    # (failedAttentionCheck / unparseableEndpoint / outOfRange — see
    # exclusions.py), joined against per-item attentionCheck declarations.
    # Manifest data, so freeze pins them through the ordinary content hash.
    # ABSENT = today's behavior exactly: no exclusion, no stamp, unchanged
    # manifest bytes.
    exclusion_rules: list = field(default_factory=list)
    concepts: list[ConceptRef] = field(default_factory=list)
    grand_mean_corpus: GrandMeanCorpus | None = None
    conditions: list[Condition] = field(default_factory=list)
    # Current study objects (parallel to ExperimentStore.swift:219).
    study_kind: str = "modelOutput"            # modelOutput | multiAgent
    variant_conditions: list[VariantCondition] = field(default_factory=list)
    multi_agent_scenario_path: str | None = None
    multi_agent_scenario_hash: str | None = None
    multi_agent_include_baseline: bool = True
    evaluation: EvaluationSpec | None = None
    # Judge-rubric versioning (evidence tier): the rubric is a git-versioned
    # FILE under prompts/rubrics/, pinned by hash; the judge panel is pinned
    # as data. See prompts/rubrics/README.md for the cross-engine contract.
    judge_rubric_file: str | None = None
    judge_rubric_hash: str | None = None
    judges: list[JudgeRef] = field(default_factory=list)
    human_validation: HumanValidation | None = None
    # Capability-battery-as-evidence: the battery a variant study's validate
    # run must execute per condition, pinned like task prompts.
    capability_battery_file: str | None = None
    capability_battery_hash: str | None = None
    # Science-layer fields (alien-stance program; WORK-PLAN Phase D).
    phase: str | None = None                   # shakedown | screen | confirm | triangulate | panel
    # A provenance LABEL. Free text, decoded from every manifest that carries
    # it, printed in report/preregistration output — and behaviorless, with one
    # DEPRECATED exception: the value "sentencing" still implicitly selects the
    # built-in duration endpoint where no ``numericParser`` is declared. That
    # trigger keeps working for manifests that already depend on it (2026-08-18)
    # and now announces itself at every site where it fires — see
    # :data:`IMPLICIT_CASE_FAMILY_ADVISORY`. Declare ``numericParser`` instead;
    # the workspace registry's shipped ``sentencing-months`` entry reproduces
    # the built-in parser exactly.
    case_family: str | None = None
    outcome_instruments: list[str] = field(default_factory=list)
    # e.g. ["answerTokenLogprob", "sampledText"]; empty = sampledText only.
    samples_per_item: int = 1                  # stochastic samples per (condition, prompt)
    seed_policy: str = "manifestSeeds"         # manifestSeeds | derivedSHA256
    # Joint logprobs favor shorter options; unequal scored-option token counts
    # are an instrument-design smell the run refuses unless acknowledged.
    acknowledge_unequal_option_lengths: bool = False
    human_baseline: HumanBaseline | None = None
    reader_refs: list[ReaderRef] = field(default_factory=list)
    promotion_rule: PromotionRule | None = None
    screen_task_prompts_hash: str | None = None  # confirm phase: the screen pool it must be disjoint from
    freeze_hash: str | None = None
    frozen_by: str | None = None
    raw: dict = field(default_factory=dict)

    # --- IO ----------------------------------------------------------------

    @classmethod
    def load(cls, name: str, root: str | None = None) -> "Manifest":
        path = os.path.join(paths.experiments_directory(root), name, "experiment.json")
        if not os.path.exists(path):
            path = os.path.join(paths.experiments_directory(root), f"{name}.json")
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        concepts = [ConceptRef(name=c["name"], stimulus_set_hash=c["stimulusSetHash"],
                               options=ExtractionOptions.from_json(c.get("options")),
                               validation_hash=c.get("validationHash"),
                               validation_pinned=("validationHash" in c),
                               designated_reference=(
                                   c.get("designatedReference")
                                   if isinstance(c.get("designatedReference"), dict)
                                   else None),
                               vector_artifact=(
                                   c.get("vectorArtifact")
                                   if isinstance(c.get("vectorArtifact"), dict)
                                   else None))
                    for c in d.get("concepts", [])]
        grand_mean_corpus = None
        if isinstance(d.get("grandMeanCorpus"), dict):
            gmc = d["grandMeanCorpus"]
            grand_mean_corpus = GrandMeanCorpus(
                concepts=[str(c) for c in (gmc.get("concepts") or [])],
                hashes={str(k): str(v) for k, v in (gmc.get("hashes") or {}).items()})
        conditions = []
        for c in d.get("conditions", []):
            slots = [Slot(concept=s["concept"], layer=int(s["layer"]),
                          alpha=float(s["alpha"]), mode=s.get("mode"))
                     for s in c.get("slots", [])]
            conditions.append(Condition(
                name=c["name"], slots=slots, band_width=int(c.get("bandWidth", 1)),
                # DELIBERATELY still False for a key-less condition (Phase-0
                # gap G6, ``docs/PORTABILITY-CONTRACTS.md``). Declaring a NEW
                # condition without the key is refused now
                # (``experiment_store._condition_entry``), but READING one is
                # not the same act: a manifest already pinned or frozen with a
                # key-less condition was measured under THIS reading, and
                # flipping it to True would silently reinterpret its doses —
                # every α in that study would mean something else than it did
                # when the study ran. ``freeze_advisories`` surfaces the
                # ambiguity instead (the Mac engine cannot read such a
                # condition at all, so the reading is engine-dependent).
                alpha_in_norm_units=bool(c.get("alphaInNormUnits", False)),
                control_type=c.get("controlType"),
                neutral_pc_basis_path=c.get("neutralPCBasisPath"),
                neutral_pc_basis_hash=c.get("neutralPCBasisHash")))
        variants = [VariantCondition(
            name=v["name"], artifact_path=v.get("artifactPath", ""),
            artifact_hash=v.get("artifactHash", ""), artifact=v.get("artifact", {}),
            from_promotion=(v.get("fromPromotion")
                            if isinstance(v.get("fromPromotion"), dict)
                            else None),
            training_provenance=(v.get("trainingProvenance")
                                 if isinstance(v.get("trainingProvenance"), dict)
                                 else None))
            for v in d.get("variantConditions", [])]
        evaluation = None
        if isinstance(d.get("evaluation"), dict):
            ev = d["evaluation"]
            evaluation = EvaluationSpec(
                kind=ev.get("kind", "none"), judge_model=ev.get("judgeModel", ""),
                judge_prompt=ev.get("judgePrompt", ""),
                structured_prompt=ev.get("structuredPrompt"))
        judges = []
        for j in (d.get("judges") or []):
            if isinstance(j, dict) and j.get("name"):
                judges.append(JudgeRef(name=str(j["name"]),
                                       kind=str(j.get("kind", "claude")),
                                       model=j.get("model"),
                                       provider=j.get("provider"),
                                       revision=j.get("revision"),
                                       dtype=j.get("dtype")))
        human_validation = None
        if isinstance(d.get("humanValidation"), dict):
            hv = d["humanValidation"]
            if hv.get("path") and hv.get("hash"):
                human_validation = HumanValidation(path=hv["path"], hash=hv["hash"])
        human_baseline = None
        if isinstance(d.get("humanBaseline"), dict):
            hb = d["humanBaseline"]
            if hb.get("path") and hb.get("hash"):
                human_baseline = HumanBaseline(path=hb["path"], hash=hb["hash"])
        reader_refs = []
        for r in (d.get("readerRefs") or []):
            if isinstance(r, dict) and r.get("path") and r.get("hash"):
                reader_refs.append(ReaderRef(path=r["path"], hash=r["hash"],
                                             concept=str(r.get("concept", ""))))
        promotion_rule = None
        if isinstance(d.get("promotionRule"), dict):
            pr = d["promotionRule"]
            promotion_rule = PromotionRule(
                fdr_threshold=float(pr.get("fdrThreshold", 0.05)),
                dose_monotone=bool(pr.get("doseMonotone", True)),
                exceeds_random_floor=bool(pr.get("exceedsRandomFloor", True)),
                capability_gate=pr.get("capabilityGate"))
        return cls(
            name=d["name"], model_id=d["modelID"], model_revision=d.get("modelRevision"),
            dtype=d.get("dtype"),
            status=d.get("status", "draft"), description=d.get("experimentDescription", ""),
            task_description=d.get("taskDescription"),
            prompt_mode=d.get("promptMode") or "chatAssistant",
            system_prompt=d.get("systemPrompt"),
            qwen_thinking_enabled=bool(d.get("qwenThinkingEnabled", False)),
            temperature=float(d.get("temperature", 0.0)),
            max_tokens=int(d.get("maxTokens", 512)),
            seeds=[int(s) for s in (d.get("seeds") or [0])],
            task_prompts_file=d.get("taskPromptsFile"),
            task_prompts_hash=d.get("taskPromptsHash"),
            neutral_corpus_hash=d.get("neutralCorpusHash"),
            markers_hash=d.get("markersHash"),
            markers_pinned=("markersHash" in d),
            jlens_readout=(d.get("jlensReadout") or None),
            record_token_ids=bool(d.get("recordTokenIDs") or False),
            reasoning_style_taxonomy_path=d.get("reasoningStyleTaxonomyPath"),
            reasoning_style_taxonomy_hash=d.get("reasoningStyleTaxonomyHash"),
            numeric_parser=d.get("numericParser"),
            parser_registry_hash=d.get("parserRegistryHash"),
            sae_candidates=(d.get("saeCandidates")
                            if d.get("saeCandidates") is not None else None),
            exclusion_rules=list(d.get("exclusionRules") or []),
            concepts=concepts, grand_mean_corpus=grand_mean_corpus,
            conditions=conditions,
            study_kind=d.get("studyKind") or "modelOutput",
            variant_conditions=variants,
            multi_agent_scenario_path=d.get("multiAgentScenarioPath"),
            multi_agent_scenario_hash=d.get("multiAgentScenarioHash"),
            multi_agent_include_baseline=bool(d.get("multiAgentIncludeBaseline", True)),
            evaluation=evaluation,
            judge_rubric_file=d.get("judgeRubricFile"),
            judge_rubric_hash=d.get("judgeRubricHash"),
            judges=judges,
            human_validation=human_validation,
            capability_battery_file=d.get("capabilityBatteryFile"),
            capability_battery_hash=d.get("capabilityBatteryHash"),
            phase=d.get("phase"),
            case_family=d.get("caseFamily"),
            outcome_instruments=[str(i) for i in (d.get("outcomeInstruments") or [])],
            samples_per_item=max(1, int(d.get("samplesPerItem", 1))),
            seed_policy=d.get("seedPolicy") or "manifestSeeds",
            acknowledge_unequal_option_lengths=bool(
                d.get("acknowledgeUnequalOptionLengths", False)),
            human_baseline=human_baseline,
            reader_refs=reader_refs,
            promotion_rule=promotion_rule,
            screen_task_prompts_hash=d.get("screenTaskPromptsHash"),
            freeze_hash=d.get("freezeHash"), frozen_by=d.get("frozenBy"), raw=d)

    # --- effective evaluation ----------------------------------------------

    def effective_evaluation(self) -> tuple["EvaluationSpec | None", str | None]:
        """``(spec, source)`` — the paired-judge evaluation this manifest
        EFFECTIVELY declares (2026-07-22 incident: the app's rubric-FILE +
        judges path pinned ``judges`` and ``judgeRubricFile`` but never wrote
        an ``evaluation`` block, so a frozen study died at the evaluate stage
        after generation).

        Resolution rule (cross-engine, Swift twin
        ``ExperimentStore.effectiveEvaluation``): an explicit ``evaluation``
        block always wins (``source == "manifest"``, unchanged semantics —
        including an explicit ``kind: "none"``). With no block, at least one
        pinned judge PLUS a pinned ``judgeRubricFile`` ARE a paired-judge
        declaration: the spec is synthesized with the documented defaults
        (kind pairedJudge; judgeModel/judgePrompt empty — the panel carries
        the judges and the pinned file carries the rubric text, hash-verified
        at read time; no structured fields) and ``source ==
        "pinnedRubric"``. ``(None, None)`` = no judging declared."""
        if self.evaluation is not None:
            return self.evaluation, "manifest"
        if self.judges and self.judge_rubric_file:
            return (EvaluationSpec(kind="pairedJudge", judge_model="",
                                   judge_prompt="", structured_prompt=None),
                    "pinnedRubric")
        return None, None

    # --- SAE latent conditions ---------------------------------------------

    @property
    def sae_latent_conditions(self) -> list[dict]:
        """Raw ``saeLatentConditions`` entries (proposal r2 §8 P2-9).

        A read-through view rather than a parsed field: the block is ADDITIVE
        and optional, the manifest content hash is computed from the raw dict,
        and an absent key must leave every existing hash byte-identical. Parse
        + validate with :func:`steerlab_server.experiment.sae_latent.parse`.
        """
        entries = self.raw.get("saeLatentConditions")
        return [e for e in entries if isinstance(e, dict)] \
            if isinstance(entries, list) else []

    # --- firewall ----------------------------------------------------------

    def _verify_jlens_readout(self, base: str) -> list[str]:
        """Re-check the pinned lens against what is on disk.

        Verifying the CONFIG HASH alone would not be enough: it covers the
        researcher's declared choices (layers, watchlist, conventions) but not
        the artifact those choices are read through. A lens re-imported from a
        different upstream commit keeps the same lensID and changes every
        number the study would report, so the artifact hash is pinned too.
        """
        block = self.jlens_readout
        if not block:
            return []
        violations: list[str] = []
        lens_id = block.get("lensID")
        pinned_hash = block.get("lensSHA256")
        if not lens_id or not pinned_hash:
            return ["jlensReadout pin is incomplete — both lensID and "
                    "lensSHA256 are required (a half-pin certifies nothing)"]
        try:
            from ..jlens import lens_store

            record = lens_store.resolve(lens_id, base)
        except Exception as exc:  # noqa: BLE001 - absence IS the violation
            return [f"pinned J-lens '{lens_id}' is not importable in this "
                    f"workspace ({type(exc).__name__}) — import it before "
                    f"running, or the readout cannot be reproduced"]
        live = record.source.tensorSHA256
        if live and live != pinned_hash:
            violations.append(
                f"J-lens '{lens_id}' changed since pinning "
                f"(have {live[:12]}…, pinned {pinned_hash[:12]}…)")
        declared = block.get("layers") or []
        unknown = sorted(set(declared) - set(record.sourceLayers))
        if unknown:
            violations.append(
                f"jlensReadout pins layers {unknown} that are not fitted "
                f"source layers of '{lens_id}'")
        # Tokenizer drift is lens drift one level down: the readout is indexed
        # by token ID, so a changed vocabulary re-points every watched token at
        # a different piece while every recorded number still looks plausible.
        pinned_tok = block.get("tokenizerHash")
        if pinned_tok and self.model_id:
            try:
                from ..jlens.derive import tokenizer_identity_hash

                live_tok = tokenizer_identity_hash(
                    self.model_id, self.model_revision)
            except Exception:  # noqa: BLE001 - unresolvable is not drift
                live_tok = None
            if live_tok and live_tok != pinned_tok:
                violations.append(
                    f"tokenizer changed since pinning "
                    f"(have {live_tok[:12]}…, pinned {pinned_tok[:12]}…) — "
                    f"every watched token ID now names a different piece")
        return violations

    def verify(self, root: str | None = None) -> list[str]:
        """Re-hash all pinned inputs, reporting drift (parallel to Swift
        ``ExperimentStore.verify``). Empty list = verified."""
        violations: list[str] = []
        base = paths.project_root() if root is None else root

        # Multi-agent studies pin a scenario; model-output studies pin concepts
        # or variants (must have at least one of them).
        if self.study_kind == "multiAgent":
            if self.multi_agent_scenario_path and self.multi_agent_scenario_hash:
                violations += _verify_file_hash(
                    base, self.multi_agent_scenario_path, self.multi_agent_scenario_hash,
                    f"multi-agent scenario '{self.multi_agent_scenario_path}'")
            else:
                violations.append("multi-agent study needs a pinned scenario")
        elif not self.concepts and not self.variant_conditions:
            violations.append("no concepts or variants attached")

        # Executable condition names must be UNIQUE. `resume.record_key`
        # leads with the condition NAME, so two executed conditions sharing
        # one alias their records into a single key — resume skipping treats
        # one condition's work as the other's, shard membership double-counts,
        # and merge completeness is computed against the wrong cell list.
        #
        # Checked against the EXECUTION matrix, not the declared collections
        # (external review round 11): the implicit baseline is in no declared
        # collection, so a variant condition named "baseline" collided while
        # passing a declared-collections scan; and carried-but-inert
        # collections (injection conditions in an agent comparison,
        # model-output config in a multi-agent study) are never executed, so
        # duplicates there would have been false refusals.
        counts: dict[str, int] = {}
        for label in effective_condition_names(self):
            if label:
                counts[label] = counts.get(label, 0) + 1
        duplicates = sorted(n for n, count in counts.items() if count > 1)
        if duplicates:
            violations.append(
                f"duplicate condition name(s) {duplicates} among the "
                f"conditions this study EXECUTES — condition names are record "
                f"identity (resume keys, shard ownership, merge completeness "
                f"all lead with them), so two cannot be told apart in the "
                f"outputs; rename one. Note the baseline condition is implicit "
                f"and always executes")

        # studyType contract (2026-07-19): LLM-authored JSON gets loud
        # feedback — an unknown vocabulary value, or a label contradicting
        # the engine-facing studyKind, is a violation (absent is legal;
        # legacy manifests derive from content).
        declared_type = self.raw.get("studyType")
        if declared_type is not None:
            type_kinds = {"conceptStudy": "modelOutput",
                          "agentComparison": "modelOutput",
                          "confirmAgent": "modelOutput",
                          "multiAgent": "multiAgent"}
            if declared_type not in type_kinds:
                violations.append(
                    f"unknown studyType {declared_type!r} — one of "
                    "conceptStudy, agentComparison, confirmAgent, multiAgent")
            elif type_kinds[declared_type] != self.study_kind:
                violations.append(
                    f"studyType {declared_type!r} contradicts studyKind "
                    f"{self.study_kind!r} — a study runs its studyKind; "
                    "fix whichever is wrong")

        # F2 (2026-07-19): pins are checked for the study kind that USES
        # them — a multi-agent study CARRIES model-output configuration
        # (concepts, task prompts, agents …) untouched but never runs it,
        # so its drift must not block verification or freezing (it
        # surfaces as a freeze advisory, and packaging skips it too).
        model_output = self.study_kind != "multiAgent"
        # Finer rule within model-output (2026-07-19): concept machinery is
        # inert for compare-agents studies without forward references.
        machinery = concept_machinery_operative(self.raw)

        # Pipeline block (chain runner, stage 5): the chain is preregistered
        # DATA, so a malformed declaration — unknown/out-of-order stages,
        # bad thresholds, gates for absent stages — is a verify violation
        # HERE, not a refusal discovered at first cluster submission.
        if self.raw.get("pipeline") is not None:
            from . import pipeline_spec
            declared_stages: tuple[str, ...] = ()
            try:
                declared_stages = pipeline_spec.resolve_pipeline(
                    self.raw.get("pipeline")).stages
            except ValueError as exc:
                violations.append(f"pipeline block invalid: {exc}")
            # Coherence gate (2026-07-22 incident): a chain declaring
            # 'evaluate' with no effective paired-judge declaration is a
            # GUARANTEED runtime failure after generation — a verify
            # violation here, never a surprise at the evaluate stage.
            # Judges + a pinned rubric file COUNT as the declaration (the
            # engines synthesize the spec from those pins), so only
            # manifests that could never have produced a judged report trip.
            if "evaluate" in declared_stages:
                spec, _source = self.effective_evaluation()
                if spec is None or spec.kind != "pairedJudge":
                    violations.append(EVALUATE_WITHOUT_JUDGING_MESSAGE)

        # Task prompts drift (when pinned) — plus scripted-transcript pin
        # checks: transcript items ride in the pinned prompts file, so their
        # schema, the model family's chat-template constraints (the model is
        # known here), and the rawCompletion incompatibility surface at
        # verify/freeze time, not first at run start.
        if model_output and self.task_prompts_file and self.task_prompts_hash:
            violations += _verify_file_hash(
                base, self.task_prompts_file, self.task_prompts_hash,
                f"task prompts '{self.task_prompts_file}'")
            violations += self._transcript_violations(base)

        # Judge rubric drift (when pinned): the rubric file IS the evaluation
        # criterion, so drift is a violation exactly like task prompts. A
        # half-pin (file without hash or vice versa) certifies nothing.
        if self.judge_rubric_file and self.judge_rubric_hash:
            violations += _verify_file_hash(
                base, self.judge_rubric_file, self.judge_rubric_hash,
                f"judge rubric '{self.judge_rubric_file}'")
        elif self.judge_rubric_file or self.judge_rubric_hash:
            violations.append(
                "judge rubric pin is incomplete — judgeRubricFile and "
                "judgeRubricHash must both be set")

        # Human-validation subset drift (when pinned) — judge-vs-human
        # agreement is only meaningful against the exact labeled rows. A
        # hash-clean file is ALSO parsed through the same row parser
        # evaluate uses (review 2026-08-02): a pin authored by pasted JSON
        # or a bundle can hash a file the parser refuses.
        if self.human_validation:
            hv_violations = _verify_file_hash(
                base, self.human_validation.path, self.human_validation.hash,
                f"human validation '{self.human_validation.path}'")
            violations += hv_violations
            if not hv_violations:
                from . import human_validation as hv_rows
                hv_path = self.human_validation.path
                resolved = hv_path if os.path.isabs(hv_path) \
                    else os.path.join(base, hv_path)
                try:
                    with open(resolved, "rb") as handle:
                        hv_rows.parse_rows(handle.read(), hv_path)
                except RuntimeError as exc:
                    violations.append(str(exc))
                except OSError:
                    pass  # missing-file already reported by the hash check

        # Capability-battery drift (when pinned): the battery certifies the
        # variant conditions' capability floor, so it pins like task prompts.
        # A hash-clean battery is ALSO shape-checked (2026-07-19, Swift
        # PinShapeValidation twin): a present-but-malformed battery used to
        # pass verify and die at task load. Shape fires only when the hash
        # matches — drift/missing are already their own violations.
        if model_output and self.capability_battery_file and self.capability_battery_hash:
            battery_problems = _verify_file_hash(
                base, self.capability_battery_file, self.capability_battery_hash,
                f"capability battery '{self.capability_battery_file}'")
            violations += battery_problems
            if not battery_problems:
                violations += _battery_shape_violations(
                    base, self.capability_battery_file)
        elif model_output and (self.capability_battery_file or self.capability_battery_hash):
            violations.append(
                "capability battery pin is incomplete — capabilityBatteryFile "
                "and capabilityBatteryHash must both be set")

        # Human-baseline table drift (when pinned) — the alien-residual R is
        # only meaningful against the exact table the study froze. A
        # hash-clean baseline is ALSO shape-checked against the analyze
        # loader's contract (2026-07-19, Swift PinShapeValidation twin);
        # shape fires only when the hash matches.
        if self.human_baseline:
            baseline_problems = _verify_file_hash(
                base, self.human_baseline.path, self.human_baseline.hash,
                f"human baseline '{self.human_baseline.path}'")
            violations += baseline_problems
            # Same trigger as the hash check (both engines run this
            # unconditionally for a pinned baseline): shape follows hash.
            if not baseline_problems:
                violations += _human_baseline_shape_violations(
                    base, self.human_baseline.path)

        # Measurement-side pins (firewall closure, 2026-07-13): markers.json
        # is read LIVE at score time and each concept's validation.jsonl LIVE
        # at validate time, so drift in either is a verification violation
        # exactly like stimulus drift. Legacy manifests without the pin keys
        # verify clean (freeze appends a non-blocking advisory instead).
        if machinery and self.markers_pinned:
            live_markers = markers_aggregate_hash(
                [c.name for c in self.concepts], root)
            if self.markers_hash is None:
                if live_markers is not None:
                    violations.append(
                        "markers.json appeared after pinning (markersHash "
                        "pinned null) — re-freeze on a duplicate to pin it")
            elif live_markers is None:
                violations.append(
                    "pinned markers.json missing — no attached concept has a "
                    "markers.json anymore")
            elif live_markers != self.markers_hash:
                violations.append(
                    f"markers.json changed since pinning "
                    f"(have {live_markers[:12]}…, pinned {self.markers_hash[:12]}…)")
        # J-lens readout pin: the lens IS a measurement instrument, so the
        # SAME rule as markers and the reasoning-style taxonomy applies —
        # drift or disappearance after pinning is a violation, and a half-pin
        # certifies nothing. Absent = no readout, no violation.
        #
        # Under `model_output` for the same F2 reason its neighbours are: a
        # multi-agent study may CARRY a readout block from before a kind
        # switch, and the panel path never arms one, so blocking verification
        # (and therefore freeze) over an instrument this study cannot run
        # refuses a legal manifest. Found by sweeping every helper this branch
        # added against studyKind after external review round 12.
        if model_output:
            violations += self._verify_jlens_readout(base)
        # Reasoning-style taxonomy pin: the taxonomy IS a measurement
        # instrument, so drift or disappearance after pinning is a violation
        # exactly like markers drift; a half-pin certifies nothing. Absent
        # (both keys None) = no reasoning-style scoring, no violation.
        if self.reasoning_style_taxonomy_path and self.reasoning_style_taxonomy_hash:
            violations += _verify_file_hash(
                base, self.reasoning_style_taxonomy_path,
                self.reasoning_style_taxonomy_hash,
                f"reasoning-style taxonomy '{self.reasoning_style_taxonomy_path}'")
        elif self.reasoning_style_taxonomy_path or self.reasoning_style_taxonomy_hash:
            violations.append(
                "reasoning-style taxonomy pin is incomplete — "
                "reasoningStyleTaxonomyPath and reasoningStyleTaxonomyHash "
                "must both be set")
        # SAE candidate roster: the file records WHICH features the study may
        # seat and the discovery evidence behind each nomination, so drift or
        # disappearance after pinning is a violation exactly like markers
        # drift, and a half-pin certifies nothing. ABSENT = no roster, no
        # violation (the whole block is optional and additive).
        from . import sae_candidates as _sae_candidates
        violations += _sae_candidates.pin_violations(
            self.raw.get("saeCandidates"), root)
        # …and the features a condition actually SEATS must be the features
        # the roster nominated (proposal r2 §8 P2-10). See
        # _verify_sae_seating.
        violations += self._verify_sae_seating(base, machinery, root)
        # SAE LATENT conditions (proposal r2 §8 P2-9): a declared mechanism,
        # not a pinned file — so what is checked is the DECLARATION. An unknown
        # mode, a missing dose, a missing server-only marker, a latent
        # condition misfiled inside `conditions` (where it would execute as a
        # slotless baseline under a steered arm's name), or a name colliding
        # with another condition are all violations here, before any behavior
        # is measured. ABSENT key = no latent conditions = no violations.
        from . import sae_latent as _sae_latent
        violations += _sae_latent.condition_violations(self.raw)
        # Declared numeric parser: the registry entry IS a measurement
        # instrument, so a missing registry, an undefined/malformed parser,
        # or registry drift after pinning is a violation like markers drift.
        # A study that names no parser (and pins no hash) gains NOTHING here.
        if model_output:
            from . import parser_registry
            violations += parser_registry.parser_pin_violations(self.raw, root)
        # Sweep-input pins (firewall closure, 2026-07-20 — cross-engine
        # contract keys "sweep.devPromptsHash" + "sweep.batteryHash"): the
        # sweep's dev prompts and capability battery decide which cell WINS,
        # so they are measurement-side inputs exactly like markers — a
        # pinned hash is enforced against the file bytes (freeze pins them;
        # sweep start additionally refuses to select on drifted inputs).
        # Legacy manifests without the keys verify clean; a declared sweep
        # resolves absent file paths to the engine defaults, exactly as the
        # sweep run does (tasks._sweep_with_spec).
        violations += self._sweep_input_pin_violations(base, machinery)
        # Declared exclusion rules are measurement declarations: a typo'd
        # rule id or a malformed range surfaces at verify — before any
        # behavior is measured — never first at analyze. Absent key = no
        # rules = nothing to check (Swift twin: ExperimentStore.verify →
        # ExclusionEngine.violations).
        from . import exclusions
        violations += exclusions.rule_violations(self.raw)
        # Artifact-pinned concepts: the VECTOR BYTES are the pinned input, so
        # both artifact files are re-hashed here exactly like a stimulus file
        # (and the block's own completeness is checked — a half-pin certifies
        # nothing).
        violations += self._verify_vector_artifact_pins(base, machinery)
        for concept in (self.concepts if machinery else []):
            if not concept.validation_pinned:
                continue
            if not concept.effective_method.has_source_concept:
                # A source-concept-less direction has NOTHING to validate: no
                # stimuli, no held-out validation.jsonl. An optvec vector's
                # evidence is its eval run's eval.json (test-split
                # shift/anchor/capability/fluency; plan §6); an imported SAE
                # decoder row's is the discovery snapshot + qualification
                # artifact in the pinned candidate roster. attach pins the
                # hash explicitly null so neither is mistaken for a legacy
                # unpinned attach; checking a validation FILE that can never
                # exist would be inventing an obligation.
                continue
            # Dual-root lookup (2026-08-19): the canonical home wins, the
            # other recipe's home is the fallback. A file in the CANONICAL
            # home resolves exactly as before, so verify()'s verdict on an
            # already-frozen manifest is unchanged; only a misfiled set —
            # previously invisible to the pin — now participates.
            paired = not concept.effective_method.uses_story_corpus
            location = resolve_validation_file(
                concept.data_concept, paired=paired, root=root)
            canonical_rel = validation_relpath(
                concept.data_concept, paired=paired)
            if concept.validation_hash is None:
                if location is not None:
                    violations.append(
                        f"concept '{concept.name}': validation.jsonl appeared "
                        "after pinning (validationHash pinned null) — "
                        f"re-attach to pin it (found at {location.relpath})")
                continue
            if location is None:
                violations.append(
                    f"concept '{concept.name}': pinned validation.jsonl "
                    f"missing at {canonical_rel}")
                continue
            val_path = location.path
            with open(val_path, "rb") as handle:
                live = hashlib.sha256(handle.read()).hexdigest()
            if live != concept.validation_hash:
                violations.append(
                    f"concept '{concept.name}' validation.jsonl changed since "
                    f"pinning (have {live[:12]}…, pinned "
                    f"{concept.validation_hash[:12]}…)")

        # Mixed arm modes refuse (engineer finding 2026-07-19): when agent
        # conditions exist, BOTH run engines execute agents only, so a
        # HAND-DECLARED injection condition would silently vanish from the
        # evidence. Two narrow exemptions (fourth round — the mere
        # presence of a `selection` dict is not proof):
        # - the canonical empty "baseline" (the agent path executes an
        #   equivalent baseline; a CUSTOM-named empty condition still
        #   refuses),
        # - a sweep-stamped condition whose provenance is STRUCTURALLY
        #   CONSISTENT: names a sweep run, one slot whose layer/alpha
        #   equal winningCell, named "<concept>-recommended" — AND whose
        #   EXECUTABLE TWIN is among the agent arms (fifth round:
        #   self-consistency alone did not prove the promoted agent
        #   actually runs; sixth round: the twin needs complete
        #   birth-certificate identity — concept + sweepRun + cell — and
        #   an unbound forward reference no longer counts; seventh round,
        #   2026-07-20: the artifact must EXECUTE the identity it
        #   certifies — exactly one injection, at exactly that
        #   concept/layer/alpha — see
        #   _sweep_stamped_executable_twin). Run-dir
        #   existence is deliberately not required (runs are per-substrate
        #   trees; imports reference the other engine's evidence).
        if machinery and self.variant_conditions:
            hand_declared = []
            for c in self.raw.get("conditions") or []:
                c = c or {}
                if _is_canonical_baseline(c):
                    continue
                if not _is_sweep_stamped(c):
                    hand_declared.append(c.get("name", "?"))
                    continue
                if not _sweep_stamped_executable_twin(
                        c, self.raw.get("variantConditions") or []):
                    slot = ((c.get("slots") or [{}])[0] or {})
                    concept = slot.get("concept", "?")
                    sweep_run = str(
                        (c.get("selection") or {}).get("sweepRun")
                        or "?").strip()
                    violations.append(
                        f"recommended condition '{c.get('name', '?')}': the "
                        f"promoted agent for concept '{concept}' is not "
                        "among this study's arms — the recommendation is "
                        f"bound to sweep run '{sweep_run}', and a study "
                        "with agent conditions runs agents only, so the "
                        "recommended cell would never execute. Promote "
                        "that sweep's agent and add it (its birth "
                        "certificate must match this concept, sweep run, "
                        "and winning cell, and its artifact must inject "
                        "exactly that cell — one injection, same "
                        "concept/layer/alpha; an unbound fromPromotion "
                        "forward reference does not count), or remove the "
                        "stale recommendation")
            if hand_declared:
                violations.append(
                    "mixed arm modes: hand-declared injection condition(s) "
                    + ", ".join(hand_declared)
                    + " would NOT execute — a study with agent conditions "
                    "runs agents only. Remove one side or split into two "
                    "studies")

        # The DEGENERATE inert case (2026-08-11): a declared agentComparison
        # with no agent arms at all but carrying injection conditions runs
        # baseline only, silently — the mixed-arm check above cannot see it
        # because the machinery is inert and there are no variants. Run
        # start and submission preflight refuse through the same rule.
        inert_problem = inert_conditions_problem(self.raw)
        if inert_problem:
            violations.append(inert_problem)

        # Neutral-PC basis pins on conditions: the basis bytes are part of the
        # injection recipe (directions projected out of the vector), so a
        # pinned hash must match the file — previously written, never checked.
        for condition in (self.conditions if machinery else []):
            if not condition.neutral_pc_basis_hash:
                continue
            if not condition.neutral_pc_basis_path:
                violations.append(
                    f"condition '{condition.name}' pins neutralPCBasisHash "
                    "without neutralPCBasisPath — an unresolvable pin "
                    "certifies nothing")
                continue
            basis_path = condition.neutral_pc_basis_path
            resolved = basis_path if os.path.isabs(basis_path) \
                else os.path.join(base, basis_path)
            if os.path.isdir(resolved):
                basis_path = os.path.join(basis_path, "neutral-pc-basis.json")
            violations += _verify_file_hash(
                base, basis_path, condition.neutral_pc_basis_hash,
                f"condition '{condition.name}' neutral-PC basis")

        # RepE reader instruments: each pinned reader must match its hash, be a
        # reader artifact at all, be fitted on THIS substrate (activations do
        # not transfer between engines), and be fitted on the study's model.
        if (model_output and "repeReaderScore" in self.outcome_instruments
                and not self.reader_refs):
            violations.append(
                "outcomeInstruments includes repeReaderScore but no readerRefs "
                "are pinned")
        for ref in (self.reader_refs if model_output else []):
            violations += _verify_file_hash(
                base, ref.path, ref.hash, f"reader '{ref.concept}'")
            reader_path = ref.path if os.path.isabs(ref.path) \
                else os.path.join(base, ref.path)
            if not os.path.exists(reader_path):
                continue  # missing-file violation already reported above
            # THROUGH THE REAL LOADER (review 2026-08-02, P1): a loose JSON
            # inspection let an artifact missing `template`/`probe` pass
            # verification and fail much later at evaluate. What the scorer
            # cannot load, verify refuses NOW.
            from ..steering import repe_reader
            try:
                artifact = repe_reader.load_reader(reader_path)
            except repe_reader.RepeReaderError as exc:
                violations.append(f"reader '{ref.concept}': {exc}")
                continue
            # The COMPLETE binding through the ONE helper the runtime
            # scorers also call (review 2026-08-02: two hand-kept subsets
            # meant the runtime accepted readers verify would flag).
            violations += repe_reader.binding_problems(
                artifact, ref_concept=ref.concept, model_id=self.model_id,
                model_revision=self.model_revision,
                substrate=_THIS_SUBSTRATE)

        # Funnel: a confirm-phase study must use an item pool DISJOINT from the
        # screen pool it references (study guide: held-out items for Phase 2).
        if model_output and self.phase == "confirm":
            if not self.screen_task_prompts_hash:
                violations.append(
                    "confirm-phase study must pin screenTaskPromptsHash "
                    "(the screen pool it is held out from)")
            elif (self.task_prompts_hash
                  and self.task_prompts_hash == self.screen_task_prompts_hash):
                violations.append(
                    "confirm-phase item pool is IDENTICAL to the screen pool "
                    "(held-out items required)")

        # Thinking-mode gate: a reasoning model's answer distribution is a
        # marginal over sampled reasoning paths, so the answer-token logprob
        # instrument (conditional on NO reasoning prefix) is only valid with
        # thinking off. Thinking-mode arms must use sampled answers over
        # samplesPerItem instead.
        if model_output and self.qwen_thinking_enabled and (
                set(self.outcome_instruments)
                & {"answerTokenLogprob", "choiceProbability", "ordinalScale"}):
            violations.append(
                "outcomeInstruments includes an answer-token instrument but "
                "qwenThinkingEnabled is true — thinking-mode answers are marginals "
                "over reasoning paths; disable thinking or use sampled answers")

        # Ordinal-scale contract: the collapse of the ladder distribution to
        # one position is an instrument-design choice — DECLARED in the
        # manifest, never silently defaulted (workbench rule). Swift twin:
        # ExperimentStore.modelOutputPinViolations.
        declared_aggregation = self.raw.get("ordinalAggregation")
        if (model_output and "ordinalScale" in self.outcome_instruments
                and declared_aggregation is None):
            violations.append(
                "outcomeInstruments includes ordinalScale but ordinalAggregation "
                "is not declared — declare 'expectedValue' "
                "(probability-weighted mean ladder position) or 'argmax' "
                "(position of the highest-probability option)")
        if (model_output and declared_aggregation is not None
                and declared_aggregation not in KNOWN_ORDINAL_AGGREGATIONS):
            violations.append(
                f"unknown ordinalAggregation '{declared_aggregation}' — one of "
                + ", ".join(KNOWN_ORDINAL_AGGREGATIONS))

        # Validation-control SHAPE is a schema question, not a runtime one:
        # nothing here needs a model, so a malformed frozen manifest must not
        # consume a cluster allocation before the problem surfaces. The
        # runtime checks in `_extract_validation_controls` stay as defence in
        # depth. Swift gets this for free — its typed decoding refuses a
        # control with no stimulusSetHash or options at LOAD.
        raw_controls = self.raw.get("validationControls")
        if raw_controls is not None and not isinstance(raw_controls, list):
            # `{}` is FALSY, so `or []` silently read a dict as "no controls
            # declared" — a malformed manifest verified clean and validated
            # with no discriminant evidence at all.
            violations.append(
                "validationControls must be an array of control objects (got "
                f"{type(raw_controls).__name__})")
            raw_controls = []
        for control in raw_controls or []:
            if not isinstance(control, dict):
                violations.append(
                    "validationControls entry is not an object (got "
                    f"{type(control).__name__})")
                continue
            concept = control.get("concept")
            if not isinstance(concept, str) or not concept.strip():
                violations.append(
                    "validationControls entry declares no usable 'concept' "
                    "(expected a non-empty string)")
                continue
            hash_value = control.get("stimulusSetHash")
            if hash_value is not None and (
                    not isinstance(hash_value, str) or not hash_value.strip()):
                violations.append(
                    f"validation control '{concept}' stimulusSetHash must be "
                    "a non-empty string")
            options = control.get("options")
            if options is not None and not isinstance(options, dict):
                # A truthy non-dict failed only at extraction, AFTER model
                # allocation; a falsy one (`[]`) was silently converted into
                # DEFAULT extraction options by ExtractionOptions.from_json,
                # reading the control at a position it was never authored for.
                violations.append(
                    f"validation control '{concept}' options must be an "
                    f"object (got {type(options).__name__})")
            if not control.get("stimulusSetHash"):
                violations.append(
                    f"validation control '{concept}' declares no "
                    "stimulusSetHash — a control is a complete pinned recipe "
                    "reference, so its stimulus bytes must be pinned like any "
                    "other input")
            if control.get("options") is None:
                violations.append(
                    f"validation control '{concept}' declares no extraction "
                    "options — a control must carry its OWN recipe; "
                    "inheriting or defaulting one reads it at a position it "
                    "was not authored for")
            if control.get("validationLayer") is not None:
                violations.append(
                    f"validation control '{concept}' declares "
                    "validationLayer, which nothing reads: the cosine matrix "
                    "compares every cell at ONE layer, so a per-control layer "
                    "would conflate concept differences with depth "
                    "differences. Remove the field; declare the study-wide "
                    "validationLayer instead")

        # The validation read layer is a measurement declaration (D4):
        # declaring an index AND a fraction is ambiguous, and the two
        # disagree on any model whose depth differs from the one assumed.
        from . import validation_layer as _vl
        layer_problem = _vl.violation(
            self.raw.get("validationLayer"),
            self.raw.get("validationLayerFraction"),
            declared_layers=self.raw.get("validationLayers"),
            declared_fractions=self.raw.get("validationLayerFractions"))
        if layer_problem:
            violations.append(layer_problem)

        # Stochastic sampling policy: samplesPerItem > 1 requires temperature > 0
        # and a deterministic per-record seed derivation.
        if self.samples_per_item > 1:
            if not self.temperature or self.temperature <= 0:
                violations.append(
                    "samplesPerItem > 1 requires temperature > 0 "
                    "(greedy decoding makes every sample identical)")
            if self.seed_policy != "derivedSHA256":
                violations.append(
                    "samplesPerItem > 1 requires seedPolicy 'derivedSHA256' "
                    "(per-record seeds derived from condition/prompt/sampleIndex)")

        from . import multiconcept

        # Data-side questions ask the EFFECTIVE method and the DATA concept:
        # an artifact-pinned concept keeps the base concept's stimuli and
        # held-out data (crit-gm reads crit's), so its pins are checked
        # against those bytes.
        grand_mean_names = [c.data_concept for c in self.concepts
                            if machinery and c.effective_method.is_grand_mean]
        for concept in (self.concepts if machinery else []):
            data_name = concept.data_concept
            if not concept.effective_method.has_source_concept:
                # No stimuli exist to drift. An optvec vector was optimized
                # against hashed target/anchor/capability DATASETS and its
                # stimulusSetHash is the composite "optvec:<hash>" over them;
                # an imported Gemma Scope SAE decoder row is a coordinate in a
                # published dictionary and its stimulusSetHash is the
                # "gemmascope:<release>:<saeID>:<feature>" identity. A
                # stimulus-directory lookup here would refuse either concept
                # for missing data it never had. Both identities travel
                # verbatim in the sidecar, which is itself hash-pinned by
                # _verify_vector_artifact_pins.
                continue
            if concept.effective_method.uses_story_corpus:
                # Story-corpus concept: its own stories.jsonl must match the pin…
                live_hash = multiconcept.stories_hash(data_name, root)
                if live_hash is None:
                    violations.append(
                        f"concept '{concept.name}': no stories.jsonl under prompts/emotions/")
                elif live_hash != concept.stimulus_set_hash:
                    violations.append(
                        f"concept '{concept.name}' stories changed since pinning "
                        f"(have {live_hash[:12]}…, pinned {concept.stimulus_set_hash[:12]}…)")
                continue
            directory = paths.concept_directory(data_name, root)
            try:
                live = StimulusSet.from_directory(directory)
            except Exception as exc:  # noqa: BLE001 - report, don't crash verify
                violations.append(f"concept '{concept.name}': {exc}")
                continue
            if live.hash != concept.stimulus_set_hash:
                violations.append(
                    f"concept '{concept.name}' stimuli changed since pinning "
                    f"(have {live.hash[:12]}…, pinned {concept.stimulus_set_hash[:12]}…)")

        # designatedReference: the reference corpus is recipe data — its pin
        # must exist and its live bytes must match, exactly like stimuli.
        for concept in (self.concepts if machinery else []):
            if not concept.effective_method.is_designated_reference:
                continue
            ref = concept.designated_reference or {}
            ref_name, ref_hash = ref.get("name"), ref.get("hash")
            if not ref_name or not ref_hash:
                violations.append(
                    f"designated-reference concept '{concept.name}' has no "
                    "pinned reference (designatedReference.name/hash) — "
                    "re-attach with --reference")
                continue
            live_ref = multiconcept.stories_hash(ref_name, root)
            if live_ref is None:
                violations.append(
                    f"concept '{concept.name}' reference '{ref_name}': "
                    "stories.jsonl missing under prompts/emotions/")
            elif live_ref != ref_hash:
                violations.append(
                    f"concept '{concept.name}' reference '{ref_name}' stories "
                    f"changed since pinning (have {live_ref[:12]}…, pinned "
                    f"{ref_hash[:12]}…)")

        # …and the vector also depends on every OTHER corpus member: the whole
        # pinned population must exist unchanged, and every grand-mean target
        # must be a member of it.
        if grand_mean_names:
            if self.grand_mean_corpus is None:
                violations.append(
                    "grand-mean concepts attached but no grandMeanCorpus pinned "
                    "(the population the grand mean is computed over)")
            else:
                for name in grand_mean_names:
                    if name not in self.grand_mean_corpus.concepts:
                        violations.append(
                            f"grand-mean concept '{name}' is not a member of the "
                            "pinned grandMeanCorpus")
                for member in self.grand_mean_corpus.concepts:
                    pinned = self.grand_mean_corpus.hashes.get(member)
                    if not pinned:
                        violations.append(
                            f"grandMeanCorpus member '{member}' has no pinned hash")
                        continue
                    live_hash = multiconcept.stories_hash(member, root)
                    if live_hash is None:
                        violations.append(
                            f"grandMeanCorpus member '{member}': stories.jsonl missing")
                    elif live_hash != pinned:
                        violations.append(
                            f"grandMeanCorpus member '{member}' stories changed since "
                            f"pinning (have {live_hash[:12]}…, pinned {pinned[:12]}…)")
        if machinery and self.neutral_corpus_hash:
            corpus_path = paths.neutral_corpus_path(root)
            try:
                live = load_texts(corpus_path)
                if live.hash != self.neutral_corpus_hash:
                    violations.append(
                        f"neutral corpus changed since pinning "
                        f"(have {live.hash[:12]}…, pinned {self.neutral_corpus_hash[:12]}…)")
            except Exception as exc:  # noqa: BLE001
                violations.append(f"neutral corpus: {exc}")

        attached = {c.name for c in self.concepts}
        for variant in (self.variant_conditions if model_output else []):
            if variant.from_promotion is not None:
                # Forward-referenced (stage 4): the DECLARATION is what
                # verify can check — a named, attached concept, and exactly
                # one identity. The artifact pin lands at run time and is
                # recorded in the run directory.
                concept = str(variant.from_promotion.get("concept") or "")
                if not concept:
                    violations.append(
                        f"variant '{variant.name}' fromPromotion names no "
                        "concept")
                elif concept not in attached:
                    violations.append(
                        f"variant '{variant.name}' forward-references "
                        f"concept '{concept}', which is not attached to "
                        "this experiment")
                if (variant.artifact_path or variant.artifact_hash
                        or variant.artifact):
                    violations.append(
                        f"variant '{variant.name}' declares BOTH "
                        "fromPromotion and a concrete artifact — a "
                        "condition has exactly one identity")
                continue
            # Variant base model must match the study's model (else it would run
            # the wrong model), and its artifact must not have drifted.
            variant_base = (variant.artifact or {}).get("baseModelID")
            if variant_base and variant_base != self.model_id:
                violations.append(
                    f"variant '{variant.name}' uses {variant_base}, not the study "
                    f"model {self.model_id}")
            if variant.artifact_path and variant.artifact_hash:
                violations += _verify_file_hash(
                    base, variant.artifact_path, variant.artifact_hash,
                    f"variant '{variant.name}'")
            # Training-dataset pins (LoRA readiness §0 amendment 1): a
            # trained adapter's dataset manifest and provenance sidecar are
            # pinned inputs like stimuli — drift after freeze is a violation,
            # never a silent re-pin.
            violations += _verify_training_provenance(base, variant, root)

        # Frozen-manifest content drift (parallel to Swift's freezeHash ==
        # manifestHash check). Only checkable for SERVER-frozen manifests — the
        # server can't reproduce Swift's manifestHash, so a Swift-frozen manifest
        # is verified by its pinned inputs alone, not the freeze hash.
        if (self.status == "frozen" and self.frozen_by == "server"
                and self.freeze_hash and self.freeze_hash != self.content_hash()):
            violations.append("manifest content changed after freeze (hash mismatch)")
        return violations

    def _seated_sae_concepts(self, base: str) -> dict:
        """``{concept name: SeatedFeature-or-None}`` for every attached concept
        backed by an imported Gemma Scope SAE decoder row.

        The identity is read from the PINNED SIDECAR BYTES (the hash-verified
        record), never from a manifest copy — a manifest that restated the
        feature id could disagree with the artifact it points at, and the
        preregistration check must speak about the vector that actually
        steers. ``None`` marks an SAE-sourced artifact carrying no
        ``gemmascopeSource`` provenance block (a report-ranked import, which
        records no feature identity a roster can match).

        Sidecars that cannot be read are omitted entirely:
        ``_verify_vector_artifact_pins`` already reports the drift/absence,
        and one defect should produce one violation.
        """
        seated: dict = {}
        for concept in self.concepts:
            if not concept.is_pinned_artifact:
                continue
            if not concept.effective_method.is_gemma_scope_sae:
                continue
            block = concept.vector_artifact or {}
            rel = str(block.get("path") or "")
            if not rel:
                continue
            path = f"{rel}.json" if os.path.isabs(rel) \
                else os.path.join(base, f"{rel}.json")
            try:
                with open(path, encoding="utf-8") as handle:
                    sidecar = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            source = sidecar.get("gemmascopeSource")
            identity = None
            if isinstance(source, dict):
                layer, feature = source.get("layer"), source.get("feature")
                if (isinstance(layer, int) and not isinstance(layer, bool)
                        and isinstance(feature, int)
                        and not isinstance(feature, bool)):
                    from . import sae_candidates as _sae_candidates
                    # The dictionary travels too when the import recorded one:
                    # a same-numbered feature in the 65k and the 262k
                    # dictionary at one layer are different features.
                    identity = _sae_candidates.SeatedFeature(
                        model=str(sidecar.get("modelID") or ""),
                        feature_id=feature, layer=layer,
                        release=str(source.get("release") or ""),
                        sae_id=str(source.get("saeID") or ""))
            seated[concept.name] = identity
        return seated

    def _verify_sae_seating(self, base: str, machinery: bool,
                            root: str | None) -> list[str]:
        """Preregistration guard for conditions that seat imported SAE
        features (proposal r2 §8 P2-10, "prohibit post-outcome feature
        selection" made mechanical).

        A condition's slots compose ADDITIVELY — several at once ARE the
        linear mix ``h + Σ αᵢ·vᵢ`` — so an SAE mixture needs no new mechanism,
        only the two declarations that keep it preregistered:

        - **Roster membership.** When the manifest pins an SAE candidate
          roster (``saeCandidates``), every SAE feature a condition seats must
          be nominated in it. The roster IS the preregistration of which
          features the study may seat, so seating an unnominated feature is a
          VIOLATION, not an advisory: it is the shape post-outcome feature
          selection takes. No roster pinned = no check (additive; every
          existing manifest is untouched). This covers BOTH mechanisms — a
          vector condition seating an imported decoder row, and a
          ``saeLatentConditions`` entry, which names its release/saeID/feature
          outright. A latent condition steers a feature exactly as a seated
          vector does, so leaving it outside the guard would have left the
          preregistration trivially avoidable by declaring the intervention in
          the other key.
        - **Sparsity.** The proposal's mixtures are SPARSE. A condition may
          seat at most :data:`DEFAULT_MAX_SAE_MIXTURE_FEATURES` SAE features
          unless the manifest declares ``maxSAEMixtureFeatures`` — so
          exceeding the default is a recorded decision, never an accident.
          The cap counts only SAE-backed slots and so can never change the
          verdict on a condition that seats none.

        Both checks run on CONDITION SLOTS, whether one or many: a single-slot
        SAE condition seats a feature just as a mixture does, and a guard that
        looked only at mixtures would be trivially avoidable by splitting one.
        """
        if not machinery:
            return []
        from . import sae_candidates as _sae_candidates
        nominated = _sae_candidates.nominations(
            self.raw.get("saeCandidates"), root)
        violations = self._verify_sae_latent_seating(nominated)

        conditions = self.raw.get("conditions") or []
        if not conditions:
            return violations
        seated = self._seated_sae_concepts(base)
        if not seated:
            return violations

        cap = self.raw.get("maxSAEMixtureFeatures")
        if cap is None:
            cap = DEFAULT_MAX_SAE_MIXTURE_FEATURES
        elif not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            violations.append(
                f"maxSAEMixtureFeatures must be a positive integer (have "
                f"{cap!r}) — it caps how many SAE features ONE condition may "
                f"mix; omit the key for the default of "
                f"{DEFAULT_MAX_SAE_MIXTURE_FEATURES}")
            cap = None

        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            name = condition.get("name")
            slots = condition.get("slots") or []
            sae_slots = [s for s in slots
                         if isinstance(s, dict) and s.get("concept") in seated]
            if not sae_slots:
                continue
            if cap is not None and len(sae_slots) > cap:
                origin = ("declared maxSAEMixtureFeatures"
                          if self.raw.get("maxSAEMixtureFeatures") is not None
                          else "the default — declare maxSAEMixtureFeatures "
                               "to raise it")
                violations.append(
                    f"condition '{name}' mixes {len(sae_slots)} SAE features, "
                    f"over the cap of {cap} ({origin}) — the preregistered "
                    "mixtures are SPARSE, and a denser mixture is a different "
                    "claim, which should be declared before behavior is "
                    "measured")
            if nominated is None:
                continue
            # One concept, one violation: a condition that named the same
            # concept in two slots would otherwise report it twice.
            for concept_name in dict.fromkeys(s["concept"] for s in sae_slots):
                identity = seated[concept_name]
                if identity is None:
                    violations.append(
                        f"condition '{name}' seats concept '{concept_name}', "
                        "a Gemma Scope SAE artifact carrying no "
                        "gemmascopeSource provenance (a report-ranked "
                        "import), so the feature it holds cannot be matched "
                        "against the pinned candidate roster. Import the "
                        "feature BY ID (gemmascope import-id) and re-attach, "
                        "so what the study seats is checkable against what it "
                        "preregistered")
                    continue
                refusal = _sae_candidates.seating_refusal(nominated, identity)
                if refusal:
                    violations.append(
                        f"condition '{name}' seats concept '{concept_name}' = "
                        f"SAE {identity.describe()}, {refusal}")
        return violations

    def _verify_sae_latent_seating(self, nominated) -> list[str]:
        """Roster membership for ``saeLatentConditions`` (review finding 3c).

        A latent condition declares its feature in Gemma Scope's OWN
        vocabulary (release / saeID / feature / optional layer), so the match
        against a nomination carrying a ``gemmaScope`` block is exact — no
        artifact, no sidecar, and no dictionary-name translation in between.

        Only the roster check applies here: the mixture cap counts injected
        vector slots, and a latent condition has none (it is one feature's
        encode-edit-decode, declared one per condition).
        """
        if nominated is None:
            return []
        entries = self.raw.get("saeLatentConditions")
        if not isinstance(entries, list) or not entries:
            return []
        from . import sae_candidates as _sae_candidates
        from . import sae_latent as _sae_latent

        violations: list[str] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            feature = entry.get("feature")
            if isinstance(feature, bool) or not isinstance(feature, int):
                continue  # already a condition_violations refusal
            name = str(entry.get("name") or f"saeLatentConditions[{index}]")
            release = str(entry.get("release") or "")
            sae_id = str(entry.get("saeID") or "")
            layer = entry.get("layer")
            if isinstance(layer, bool) or not isinstance(layer, int):
                # Absent is legal — the SAE's dictionary lives at one layer,
                # which its id names. Fall back to the same grammar the run
                # path uses so the guard reads the same layer the run will.
                from . import gemma_scope as _gemma_scope
                layer = _gemma_scope.parse_sae_id(sae_id).get("layer")
            seated = _sae_candidates.SeatedFeature(
                model=self.model_id, feature_id=feature, layer=layer,
                release=release, sae_id=sae_id)
            refusal = _sae_candidates.seating_refusal(nominated, seated)
            if refusal:
                violations.append(
                    f"{_sae_latent.CONDITION_KEY} '{name}' intervenes on SAE "
                    f"{seated.describe()}, {refusal}")
        return violations

    def _verify_vector_artifact_pins(self, base: str,
                                     machinery: bool) -> list[str]:
        """Drift/completeness violations for artifact-pinned concepts.

        The pinned bytes ARE the recipe here, so both files are re-hashed at
        every verify and the refusal names the file and both hashes — the
        same rule and the same loudness as stimulus drift. Contradictions
        that are pure DATA (a sidecar recorded against another model, or a
        sourceMethod that disagrees with the sidecar's own record) are
        violations too: they are checkable on either engine and can never
        become true later. Substrate is deliberately NOT checked here — a
        manifest must stay verifiable on the engine that authored it, and
        the foreign-substrate refusal belongs at the load path
        (``require_native_substrate``), where it fires before any steering.
        """
        violations: list[str] = []
        for concept in (self.concepts if machinery else []):
            block = concept.vector_artifact
            if not concept.is_pinned_artifact:
                if block:
                    violations.append(
                        f"concept '{concept.name}' pins a vectorArtifact but "
                        f"declares method '{concept.options.method.value}' — "
                        "a pinned artifact is only read under method "
                        "'pinnedArtifact'")
                continue
            if not isinstance(block, dict) or not block.get("path"):
                violations.append(
                    f"concept '{concept.name}': method 'pinnedArtifact' with "
                    "no vectorArtifact.path — there is nothing to materialize")
                continue
            path = str(block["path"])
            tensor_pin = block.get("sha256TensorHash")
            sidecar_pin = block.get("sha256SidecarHash")
            if not tensor_pin or not sidecar_pin:
                violations.append(
                    f"concept '{concept.name}' vectorArtifact pin is "
                    "incomplete — sha256TensorHash and sha256SidecarHash "
                    f"must both be set for '{path}' (a half-pin certifies "
                    "nothing)")
                continue
            tensor_problems = _verify_file_hash(
                base, f"{path}.safetensors", tensor_pin,
                f"concept '{concept.name}' vector artifact")
            violations += tensor_problems
            sidecar_problems = _verify_file_hash(
                base, f"{path}.json", sidecar_pin,
                f"concept '{concept.name}' vector artifact sidecar")
            violations += sidecar_problems
            if sidecar_problems:
                continue
            sidecar_path = f"{path}.json" if os.path.isabs(path) \
                else os.path.join(base, f"{path}.json")
            try:
                with open(sidecar_path, encoding="utf-8") as handle:
                    sidecar = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                violations.append(
                    f"concept '{concept.name}' vector artifact sidecar "
                    f"'{path}.json' is not readable JSON ({exc})")
                continue
            recorded_method = sidecar.get("extractionMethod")
            if block.get("sourceMethod") != recorded_method:
                violations.append(
                    f"concept '{concept.name}' vectorArtifact.sourceMethod "
                    f"{block.get('sourceMethod')!r} contradicts the pinned "
                    f"sidecar's extractionMethod {recorded_method!r} — "
                    "re-attach the artifact")
            if sidecar.get("modelID") != self.model_id:
                violations.append(
                    f"concept '{concept.name}' vector artifact was extracted "
                    f"on {sidecar.get('modelID')!r}, not this study's model "
                    f"{self.model_id!r} — a direction does not transfer "
                    "between models")
            recorded_revision = sidecar.get("revision")
            if (self.model_revision and recorded_revision
                    and recorded_revision != self.model_revision):
                violations.append(
                    f"concept '{concept.name}' vector artifact was extracted "
                    f"at revision {str(recorded_revision)[:12]}…, not this "
                    f"study's pinned {self.model_revision[:12]}…")
            recorded_reading = sidecar.get("readingPosition")
            if recorded_reading and recorded_reading != \
                    concept.options.reading_position.label:
                violations.append(
                    f"concept '{concept.name}' reading position "
                    f"'{concept.options.reading_position.label}' contradicts "
                    f"the pinned artifact's '{recorded_reading}' — held-out "
                    "activations must be read where the vector was read")
        return violations

    def _sweep_input_pin_violations(self, base: str,
                                    machinery: bool) -> list[str]:
        """Drift/missing violations for the sweep-input pins, keyed off the
        DECLARED sweep block (Swift twin: the sweep branch of
        ``ExperimentStore.modelOutputPinViolations``). Two-state contract:
        key absent = legacy/unpinned (clean — freeze pins it); key present =
        hash-enforced against the file the sweep would actually read
        (declared path, else the engine default)."""
        sweep = self.raw.get("sweep")
        if not machinery or not isinstance(sweep, dict):
            return []
        violations: list[str] = []
        for file_key, hash_key, default, label in sweep_input_pin_surface():
            pinned = sweep.get(hash_key)
            if not pinned:
                continue
            rel = sweep.get(file_key) or default
            violations += _verify_file_hash(base, rel, pinned,
                                            f"{label} '{rel}'")
        # Choice instruments (review 2026-08-02, P1): same two-state
        # contract, per declared instrument.
        for _concept, rel, pinned, label in sweep_choice_pin_entries(sweep):
            if not pinned:
                continue
            violations += _verify_file_hash(base, rel, pinned,
                                            f"{label} '{rel}'")
        return violations

    def _transcript_violations(self, base: str) -> list[str]:
        """Scripted-transcript checks over the pinned task-prompts file:
        per-item schema, the study model family's chat-template constraints,
        and the rawCompletion incompatibility. Messages are the cross-engine
        contract (Swift twin: ``ExperimentTasks.transcriptPinViolations``).
        Unreadable/unparseable files add nothing here — the hash pin (checked
        by the caller) and the run loop's own loader carry those failures."""
        from . import prompt_render

        path = self.task_prompts_file if os.path.isabs(self.task_prompts_file) \
            else os.path.join(base, self.task_prompts_file)
        violations: list[str] = []
        any_transcript = False
        try:
            with open(path, encoding="utf-8") as handle:
                for i, raw in enumerate(handle):
                    line = raw.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "transcript" not in obj:
                        continue
                    any_transcript = True
                    item_id = obj.get("id", f"prompt-{i}")
                    violation = prompt_render.transcript_schema_violation(
                        obj["transcript"], item_id)
                    if violation is None:
                        violation = prompt_render.transcript_family_violation(
                            obj["transcript"], item_id, self.model_id)
                    if violation:
                        violations.append(violation)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return []
        if any_transcript and self.prompt_mode == prompt_render.RAW_COMPLETION:
            violations.append(prompt_render.TRANSCRIPT_RAW_COMPLETION_MESSAGE)
        return violations

    def content_hash(self) -> str:
        """Hash of the WHOLE pinned manifest (parallel to Swift ``manifestHash``)
        — conditions, task prompts, evaluation, seeds, temperature, system
        prompt, variants, scenario, everything — minus the volatile freeze
        stamps. This is the freezeHash + run-provenance stamp; two materially
        different studies must not collide. (Not byte-identical to Swift's value;
        cross-engine safety rests on the stimulus SHA-256s.)"""
        payload = {k: v for k, v in self.raw.items()
                   if k not in ("status", "frozenAt", "freezeHash", "gitCommit",
                                "frozenBy", "appVersion", "createdAt",
                                "freezeForced", "forcedGatesSkipped")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def validation_scope_hash(self) -> str:
        """Narrow hash for validate-evidence matching: model+revision+concepts+
        neutral corpus only. Conditions are deliberately OUT of scope, so adding
        a condition never invalidates a validate run (CLAUDE.md)."""
        def _concept_scope(c: "ConceptRef") -> dict:
            entry = {"name": c.name, "stimulusSetHash": c.stimulus_set_hash,
                     "method": c.options.method.value,
                     "readingPosition": c.options.reading_position.label}
            if c.is_pinned_artifact:
                # The pinned BYTES are what validate measured, so a different
                # artifact at the same path is a different scope — evidence
                # from the old bytes must not certify the new ones. Appended
                # only for pinned concepts, so every legacy scope hash is
                # unchanged.
                block = c.vector_artifact or {}
                entry["vectorArtifact"] = {
                    "path": block.get("path"),
                    "sha256TensorHash": block.get("sha256TensorHash"),
                    "sha256SidecarHash": block.get("sha256SidecarHash")}
            return entry

        payload = {
            "modelID": self.model_id, "modelRevision": self.model_revision,
            "concepts": [_concept_scope(c) for c in self.concepts],
            "neutralCorpusHash": self.neutral_corpus_hash,
        }
        if self.grand_mean_corpus is not None:
            # The population defines the grand-mean vectors: changing corpus
            # membership or any member's stories invalidates validation.
            payload["grandMeanCorpus"] = {
                "concepts": sorted(self.grand_mean_corpus.concepts),
                "hashes": dict(sorted(self.grand_mean_corpus.hashes.items())),
            }
        if self.variant_conditions:
            # Variant studies validate the capability battery per condition:
            # battery drift (a different pinned battery) invalidates the
            # evidence, so the pin enters the scope. Conditions themselves stay
            # out of scope — freeze separately requires per-condition battery
            # results in the matched evidence, so an ADDED condition surfaces
            # there, not here.
            payload["capabilityBatteryHash"] = self.capability_battery_hash
        depths = {k: self.raw[k] for k in (
            "validationLayer", "validationLayerFraction",
            "validationLayers", "validationLayerFractions")
            if self.raw.get(k) is not None}
        if depths:
            # Appended ONLY when a validation depth is declared (D4):
            # evidence validated at one depth does not certify a manifest
            # that now declares different depths. Absent when undeclared, so
            # legacy scope hashes are unchanged. Swift twin:
            # ``ValidationScope.validationDepths``.
            payload["validationDepths"] = depths
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _is_canonical_baseline(condition: dict) -> bool:
    """The canonical baseline Add Baseline creates: named "baseline", no
    slots — the agent path executes an equivalent, so it never vanishes."""
    return condition.get("name") == "baseline" and not condition.get("slots")


def _is_sweep_stamped(condition: dict) -> bool:
    """STRUCTURALLY VALIDATED sweep provenance (Swift ``isSweepStamped``
    twin): the selection block must name a sweep run, and the condition
    must BE the winning cell — one slot whose layer/alpha equal
    winningCell, named "<concept>-recommended"."""
    selection = condition.get("selection")
    if not isinstance(selection, dict):
        return False
    if not str(selection.get("sweepRun") or "").strip():
        return False
    cell = selection.get("winningCell")
    slots = condition.get("slots") or []
    if not isinstance(cell, dict) or len(slots) != 1:
        return False
    slot = slots[0] or {}
    try:
        if (int(slot.get("layer")) != int(cell.get("layer"))
                or float(slot.get("alpha")) != float(cell.get("alpha"))):
            return False
    except (TypeError, ValueError):
        return False
    return condition.get("name") == f"{slot.get('concept')}-recommended"


def _sweep_stamped_executable_twin(condition: dict,
                                   variant_conditions: list) -> bool:
    """Whether a sweep-stamped "-recommended" condition's EXECUTABLE TWIN
    is among the study's agent arms (Swift
    ``sweepStampedExecutableTwinPresent`` twin — fifth engineer round,
    2026-07-19: structural self-consistency alone proved nothing about
    what RUNS; sixth round, same day: the twin now requires COMPLETE
    birth-certificate identity). The twin is a CONCRETE variant whose
    artifact carries a promotion birth certificate with all three
    identity legs:

    - concept + cell, as EXECUTED: the artifact carries EXACTLY ONE
      injection, and that injection's concept/layer/alpha equal the
      condition's slot / ``selection.winningCell`` (seventh round,
      2026-07-20: the certificate alone proved what was CLAIMED, not
      what RUNS — an artifact certifying L4/α2 while injecting L9/α9,
      or carrying a second injection, passed). ``promote`` on both
      engines mints exactly one injection stamped from the same
      layer/alpha values as the certificate's winningCell, so exact
      float equality on alpha is correct here (both sides are the same
      stamp, never recomputed). The promotion certificate itself
      carries NO concept field (a known shape gap) — concept identity
      is read from the injection;
    - sweepRun: the certificate's sweepRun is NON-EMPTY and equal to
      ``selection.sweepRun`` (a stamped selection always names its run,
      so the old "when both carry one" let an empty certificate run
      match any recommendation);
    - cell: winningCell layer/alpha equal to selection.winningCell.

    A ``fromPromotion`` forward reference NO LONGER exempts a STAMPED
    condition: it binds to no selection — its future sweep may pick a
    different cell. (Pipeline chains are unaffected: frozen chains never
    carry mid-chain stamps, and chains resolve forward refs from their
    OWN sweep at run time.)"""
    selection = condition.get("selection") or {}
    slots = condition.get("slots") or []
    slot = (slots[0] or {}) if slots else {}
    concept = slot.get("concept")
    cell = selection.get("winningCell") or {}
    condition_sweep = str(selection.get("sweepRun") or "").strip()
    if not condition_sweep:
        return False
    for variant in variant_conditions or []:
        variant = variant or {}
        if isinstance(variant.get("fromPromotion"), dict):
            continue  # unbound — no longer an executable twin
        artifact = variant.get("artifact")
        promotion = artifact.get("promotion") \
            if isinstance(artifact, dict) else None
        if not isinstance(promotion, dict):
            continue
        injections = artifact.get("injections") or []
        if len(injections) != 1 or not isinstance(injections[0], dict):
            continue  # promote mints exactly one; more or fewer is not it
        injection = injections[0]
        if injection.get("concept") != concept:
            continue
        try:
            if (int(injection.get("layer")) != int(cell.get("layer"))
                    or float(injection.get("alpha"))
                    != float(cell.get("alpha"))):
                continue  # certifies one cell, injects another
        except (TypeError, ValueError):
            continue
        promoted_cell = promotion.get("winningCell")
        if not isinstance(promoted_cell, dict):
            continue
        try:
            if (int(promoted_cell.get("layer")) != int(cell.get("layer"))
                    or float(promoted_cell.get("alpha"))
                    != float(cell.get("alpha"))):
                continue
        except (TypeError, ValueError):
            continue
        promoted_sweep = str(promotion.get("sweepRun") or "").strip()
        if not promoted_sweep or promoted_sweep != condition_sweep:
            continue
        return True
    return False


#: "confirmAgent" is a LEGACY ALIAS (a top-level type until 2026-07-19,
#: now the concept study's confirm phase) — still accepted in shipped
#: manifests and LLM packs, canonicalized by ``effective_study_type``.
STUDY_TYPE_KINDS = {"conceptStudy": "modelOutput",
                    "agentComparison": "modelOutput",
                    "confirmAgent": "modelOutput",
                    "multiAgent": "multiAgent"}
_STUDY_TYPE_ALIASES = {"confirmAgent": "conceptStudy"}


def effective_study_type(d: dict) -> str:
    """The researcher-facing study type (Swift ``StudyIntent.derive`` twin):
    a DECLARED, kind-consistent ``studyType`` wins; otherwise derive from
    content — perturbation policy => confirmAgent; concepts/sweep =>
    conceptStudy; variants => agentComparison."""
    kind = d.get("studyKind") or "modelOutput"
    declared = d.get("studyType")
    if declared in STUDY_TYPE_KINDS and STUDY_TYPE_KINDS[declared] == kind:
        return _STUDY_TYPE_ALIASES.get(declared, declared)
    if kind == "multiAgent":
        return "multiAgent"
    # A perturbation policy is the concept study's CONFIRM phase
    # (2026-07-19 fold-in) — same machinery, same type.
    if d.get("perturbationPolicy"):
        return "conceptStudy"
    # Injection conditions count as concept-study evidence too: a manifest
    # carrying them without a DECLARED type is running the machinery.
    if d.get("concepts") or d.get("conditions") or d.get("sweep"):
        return "conceptStudy"
    if d.get("variantConditions"):
        return "agentComparison"
    return "conceptStudy"


def effective_conditions(manifest) -> list:
    """The condition list a RUN actually executes — the one truth.

    Extracted from the run loop so that verification, sharding, and expected-
    record enumeration cannot disagree with execution about what a study runs
    (external review round 11). Two rules, both load-bearing:

    * A variant comparison runs **baseline + variants only**. Declared
      injection conditions are carried but INERT (Swift's
      `runVariantComparison` twin), so counting them as executable invents
      cells that never emit.
    * Otherwise the concept machinery decides whether declared conditions are
      operative at all, and a baseline is PREPENDED when absent — the
      implicit baseline that no declared collection contains.

    A MULTI-AGENT study executes neither collection: it runs a scenario, and
    its record identity is the panel's own fixed vocabulary (see
    :func:`effective_condition_names`). It may legally CARRY model-output
    configuration under the app's never-delete rule, so evaluating it against
    the model-output matrix invented conditions it never runs — a carried
    variant named "baseline" resolved to ["baseline", "baseline"] and refused
    a legal manifest (external review round 12).

    Returns `Condition` objects; use :func:`effective_condition_names` when
    only the record-identity vocabulary is needed.
    """
    if (manifest.raw.get("studyKind") or "modelOutput") != "modelOutput":
        return []
    if manifest.variant_conditions:
        return [Condition(name="baseline", slots=[])]
    machinery = concept_machinery_operative(manifest.raw)
    conditions = list(manifest.conditions) if machinery else []
    if not any(c.name == "baseline" for c in conditions):
        conditions = [Condition(name="baseline", slots=[])] + conditions
    return conditions


def effective_condition_names(manifest) -> list[str]:
    """Every name that will key a record, in execution order.

    `resume.record_key` leads with the condition name, so this list IS the
    record-identity vocabulary of a run: shard membership, resume skipping,
    and merge completeness all speak it.
    """
    if (manifest.raw.get("studyKind") or "modelOutput") != "modelOutput":
        # A panel run's conditions are the scenario's own, not the manifest's:
        # "configured", plus "baseline" when a baseline play-through is
        # included (tasks._run_multi_agent_impl). Carried model-output
        # configuration is inert and keys no record.
        return (["configured", "baseline"]
                if manifest.multi_agent_include_baseline else ["configured"])
    names = [c.name for c in effective_conditions(manifest)]
    names += [vc.name for vc in (manifest.variant_conditions or [])]
    names += [spec.get("name") for spec in (manifest.sae_latent_conditions or [])
              if isinstance(spec, dict) and spec.get("name")]
    return names


def concept_machinery_operative(d: dict) -> bool:
    """Whether concepts / injection conditions / markers / sweep inputs are
    OPERATIVE for this manifest (Swift ``conceptMachineryOperative`` twin —
    the finer rule under studyKind, 2026-07-19): concept studies and
    confirmations run the machinery; a compare-agents study runs it only
    when forward-referenced agents make its own sweep necessary. Inert
    carried configuration must not verify, package, or EXECUTE."""
    if (d.get("studyKind") or "modelOutput") != "modelOutput":
        return False
    effective = effective_study_type(d)
    if effective == "conceptStudy":
        return True
    return any((vc or {}).get("fromPromotion")
               for vc in d.get("variantConditions") or [])


def inert_conditions_problem(d: dict) -> str | None:
    """A study that would SILENTLY measure baseline only (observed live
    2026-08-11: the c20-* fan-out burned a 4-shard GPU allocation producing
    12 baseline records instead of 96): a modelOutput manifest whose
    DECLARED study type keeps the concept machinery inert — a declared
    agentComparison with no fromPromotion arms — while attaching NO variant
    conditions to run instead, yet still carrying non-baseline injection
    conditions. Both engines' run paths drop those conditions by the
    2026-07-19 inert-machinery rule, sharded and unsharded alike, so every
    declared arm vanishes from the evidence without a word. Returns the
    plain-language problem (shared by verify, run start, and submission
    preflight — one rule, three surfaces), or None.

    Deliberately narrow: a manifest WITH variant conditions keeps the
    2026-07-19 behavior (agents run; carried injection conditions are inert,
    surfaced by the mixed-arm verify check), and a baseline-only manifest
    with no injection conditions is a legal, if unusual, study."""
    if (d.get("studyKind") or "modelOutput") != "modelOutput":
        return None
    if concept_machinery_operative(d):
        return None
    if d.get("variantConditions"):
        return None
    dropped = [str((c or {}).get("name", "?"))
               for c in (d.get("conditions") or [])
               if not _is_canonical_baseline(c or {})]
    if not dropped:
        return None
    return (
        f"declared studyType '{d.get('studyType')}' keeps the concept "
        "machinery inert and the manifest attaches no agent (variant) "
        "conditions, so a run would execute BASELINE ONLY — silently "
        f"dropping injection condition(s) {', '.join(dropped)}. Remove the "
        "studyType declaration (content derivation would make this a "
        "concept study), declare 'conceptStudy', or attach the agent "
        "conditions the declaration promises")


def no_measured_conditions_problem(d: dict) -> str | None:
    """A concept study that would measure NOTHING: the concept machinery is
    operative, yet the manifest declares no injection condition, no agent
    (variant) condition and no SAE latent arm — so a run executes the
    implicit BASELINE alone and ``analyze`` then reports zero effect sizes,
    both exiting 0. A pipeline that "succeeds" end to end and measures
    nothing (WP0 dry run #0, P0-2).

    Sibling of :func:`inert_conditions_problem` (2026-08-11), which catches
    the other road to a silent baseline-only run: there the arms were
    DECLARED and dropped, here they were never declared. Same remedy shape —
    say what is missing, and name the sanctioned spelling of a deliberate
    baseline-only run.

    Deliberately narrow: a study with agent arms runs them, and a declared
    ``agentComparison`` keeps the machinery inert (that IS the explicit
    baseline-only declaration), so neither trips this. Swift twin:
    ``ExperimentStore.noMeasuredConditionsProblem``."""
    if (d.get("studyKind") or "modelOutput") != "modelOutput":
        return None
    if not concept_machinery_operative(d):
        return None
    if d.get("variantConditions") or d.get("saeLatentConditions"):
        return None
    if any(not _is_canonical_baseline(c or {})
           for c in (d.get("conditions") or [])):
        return None
    concepts = d.get("concepts") or []
    if not concepts:
        # No concepts either: nothing was ever DERIVED, so there is no arm
        # that "vanished" and nothing to promise. That is the plain
        # baseline-only manifest the 2026-08-11 rule already calls legal, if
        # unusual — this refusal is about a manifest that attached concepts
        # and then never gave them an arm.
        return None
    return (
        f"study '{d.get('name')}' runs the concept machinery "
        f"({len(concepts)} concept(s) attached) but declares NO injection "
        "condition, NO agent (variant) condition and NO SAE latent arm, so a "
        "run would execute the implicit BASELINE only and measure nothing — "
        "analyze would then report zero effect sizes. Add an arm: sweep "
        f"layer×alpha and 'experiment promote {d.get('name')} <concept>' "
        "mints one from the winning cell, or declare a condition naming a "
        "concept, layer and alpha. For a DELIBERATE baseline-only run, "
        "declare studyType 'agentComparison' — the manifest then says "
        "baseline-only instead of implying it")


def inert_machinery_note(d: dict) -> dict | None:
    """Structured record of INERT carried concept machinery, for the runs
    that legally proceed (agent arms exist, so :func:`inert_conditions_problem`
    does not refuse): a declared agent comparison whose manifest still
    carries concepts or injection conditions runs agents+baseline only, by
    the 2026-07-19 rule. That is correct — and it must be LOUD and stamped,
    not silent: the 2026-08-11 c20-* incident cost two full GPU rounds
    partly because a baseline-only result looked completed and ordinary.
    The run start logs this note and ``config.json``'s open ``notes`` dict
    carries it (key ``inertConceptMachinery``, Swift twin
    ``inertMachineryNote``), so a baseline-only run is self-describing.

    Returns ``{"declaredStudyType", "inertConditions", "inertConcepts"}``
    (names, canonical baseline excluded), or None when the machinery is
    operative, the study is not modelOutput, or nothing is carried."""
    if (d.get("studyKind") or "modelOutput") != "modelOutput":
        return None
    if concept_machinery_operative(d):
        return None
    conditions = [str((c or {}).get("name", "?"))
                  for c in (d.get("conditions") or [])
                  if not _is_canonical_baseline(c or {})]
    concepts = [str((c or {}).get("name", "?"))
                for c in (d.get("concepts") or [])]
    if not conditions and not concepts:
        return None
    return {"declaredStudyType": d.get("studyType"),
            "inertConditions": conditions,
            "inertConcepts": concepts}


def validation_file_path(concept: str, *, paired: bool, root: str | None = None) -> str:
    """The CANONICAL held-out validation.jsonl for one concept under the given
    recipe: beside its stimuli for paired methods, beside its stories for
    grand-mean concepts (mirrors ``tasks._validate_impl``).

    This is the home the recipe OWNS — where a file should be filed and where
    a refusal points. It is not by itself the lookup: see
    :func:`resolve_validation_file`, which falls back to the other recipe's
    home so a misfiled set is found loudly instead of read as absent."""
    from . import multiconcept
    if paired:
        return os.path.join(paths.concept_directory(concept, root), "validation.jsonl")
    return multiconcept.validation_path(concept, root)


def validation_relpath(concept: str, *, paired: bool) -> str:
    """Workspace-relative form of :func:`validation_file_path` — the string a
    refusal or advisory names. Swift twin:
    ``ExperimentStore.conceptValidationRelativePath``."""
    return f"prompts/{'concepts' if paired else 'emotions'}/{concept}/validation.jsonl"


class ValidationSetLocation(NamedTuple):
    """Where one concept's held-out ``validation.jsonl`` was actually FOUND.

    ``path`` is the file the pin hashes and the probe reads; ``canonical``
    is where this recipe's home says it belongs. ``used_fallback`` is true
    when the two differ (the set is filed under the OTHER recipe's root);
    ``both_present`` is true when a file sits in each home — ambiguous
    filing, worth its own advisory even though the canonical one wins.
    Swift twin: ``ExperimentStore.ValidationSetLocation``."""
    path: str
    relpath: str
    canonical_path: str
    canonical_relpath: str
    fallback_relpath: str
    used_fallback: bool
    both_present: bool


def resolve_validation_file(concept: str, *, paired: bool,
                            root: str | None = None) -> ValidationSetLocation | None:
    """THE dual-root lookup for a concept's held-out set (2026-08-19).

    Resolution order is deterministic and the canonical home ALWAYS wins:

    1. the recipe's canonical home (:func:`validation_file_path`),
    2. failing that, the OTHER recipe's home.

    Returns None when neither holds a file — the historical "absent" state,
    unchanged. Only WHERE we look changed: a set filed under the wrong
    recipe's root used to be silently invisible (no hash pinned, no error),
    which is a measurement-side pin quietly missing from the firewall.
    Callers that find a fallback must say so — :func:`validation_lookup_advisory`
    is the one wording. Swift twin:
    ``ExperimentStore.resolveConceptValidation``."""
    canonical = validation_file_path(concept, paired=paired, root=root)
    fallback = validation_file_path(concept, paired=not paired, root=root)
    canonical_present = os.path.isfile(canonical)
    fallback_present = os.path.isfile(fallback) and fallback != canonical
    if not canonical_present and not fallback_present:
        return None
    used_fallback = not canonical_present
    canonical_relpath = validation_relpath(concept, paired=paired)
    fallback_relpath = validation_relpath(concept, paired=not paired)
    return ValidationSetLocation(
        path=fallback if used_fallback else canonical,
        relpath=fallback_relpath if used_fallback else canonical_relpath,
        canonical_path=canonical,
        canonical_relpath=canonical_relpath,
        fallback_relpath=fallback_relpath,
        used_fallback=used_fallback,
        both_present=canonical_present and fallback_present)


def validation_lookup_advisory(concept_name: str,
                               location: ValidationSetLocation | None) -> str | None:
    """The LOUD, non-fatal note a dual-root lookup owes its caller: the file
    was not in this recipe's home, or it is in both. None when the filing is
    unambiguous (canonical only) or nothing was found.

    One wording, shared by every point of use (validate, extract, freeze,
    data-readiness) so the same mistake reads the same everywhere. Swift twin:
    ``ExperimentStore.validationLookupAdvisory`` (intent-twinned prose, not a
    byte-identical literal — these are engine-local log/advisory lines)."""
    if location is None:
        return None
    if location.both_present:
        return (
            f"concept '{concept_name}': validation.jsonl exists under BOTH "
            f"recipe roots ({location.canonical_relpath} and "
            f"{location.fallback_relpath}) — this recipe reads and pins the "
            f"canonical {location.canonical_relpath}; delete or merge the "
            "other so the held-out set is unambiguous")
    if location.used_fallback:
        return (
            f"concept '{concept_name}': validation.jsonl was found under the "
            f"OTHER recipe's root ({location.relpath}) and is being read and "
            f"pinned from there — this recipe's canonical home is "
            f"{location.canonical_relpath}; move it there so the pin names "
            "the file the recipe owns")
    return None


def owes_held_out_probe(concept) -> bool:
    """Whether a pinned concept OWES a held-out probe at validate time.

    False for directions that were never read OFF a concept's stimuli — an
    OptVec vector (evidence: the eval run's eval.json) and an imported Gemma
    Scope SAE decoder row (evidence: the roster's discovery snapshot +
    qualification). Asking those for a validation.jsonl invents an obligation
    they can never meet. Swift twin: ``ExperimentStore.owesHeldOutProbe``."""
    return concept.effective_method.has_source_concept


def held_out_probe_relpath(concept) -> str | None:
    """Workspace-relative ``validation.jsonl`` a pinned concept's held-out
    probe reads — the DATA concept's file, in the family its EFFECTIVE method
    implies. None when the concept owes no probe. Named in refusals so the
    remedy is a real path. Swift twin: ``ExperimentStore.heldOutProbePath``."""
    if not owes_held_out_probe(concept):
        return None
    family = "emotions" if concept.effective_method.uses_story_corpus \
        else "concepts"
    return f"prompts/{family}/{concept.data_concept}/validation.jsonl"


def concept_validation_hash(concept: str, *, paired: bool,
                            root: str | None = None) -> str | None:
    """SHA-256 of a concept's validation.jsonl raw bytes, or None when absent
    — the value ``attach`` pins as the concept's ``validationHash``.

    Hashes the file the dual-root lookup actually FOUND, canonical home
    first (:func:`resolve_validation_file`). The three-state pin semantics
    are untouched: a hash when a file exists in either home, an explicit
    null when neither does."""
    location = resolve_validation_file(concept, paired=paired, root=root)
    if location is None:
        return None
    with open(location.path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


#: Default dev-prompts path a sweep resolves when the spec declares none —
#: keep identical to ``tasks._sweep_with_spec``'s fallback and Swift
#: ``SweepSpec()``'s default.
DEFAULT_SWEEP_DEV_PROMPTS_FILE = "prompts/dev/dev-prompts.jsonl"


def sweep_input_pin_surface() -> tuple:
    """The sweep-input pin surface (cross-engine contract, 2026-07-20):
    ``(fileKey, hashKey, engine default path, label)`` per pinned input of a
    declared sweep block. Shared by verify (drift), freeze (pin-when-absent),
    and the sweep-start refusal, so a new sweep input added HERE is pinned,
    verified, and enforced everywhere at once. Choice instruments live under
    ``selection.objective`` and carry per-concept structure, so they have
    their own parallel enumeration: :func:`sweep_choice_pin_entries`."""
    from . import battery as battery_mod
    return (
        ("devPromptsFile", "devPromptsHash",
         DEFAULT_SWEEP_DEV_PROMPTS_FILE, "sweep dev prompts"),
        ("batteryFile", "batteryHash",
         battery_mod.DEFAULT_BATTERY_FILE, "sweep capability battery"),
    )


def sweep_choice_pin_entries(sweep: dict) -> list[tuple]:
    """The choice-instrument pin surface (review 2026-08-02, P1: the files
    that determine the WINNING CELL were the one sweep input not pinned at
    freeze — a choice file could change between freeze and execution with
    no manifest drift). ``(concept_or_None, rel_path, pinned_hash_or_None,
    label)`` per declared instrument; concept ``None`` is the singular
    ``choicePromptsFile`` (pin key ``choicePromptsHash``), a concept names
    a ``choicePromptsFiles`` entry (pin map ``choicePromptsHashes``).
    Shared by freeze (pin-when-absent), verify (drift), and the sweep-start
    refusal — the same three-consumer contract as
    :func:`sweep_input_pin_surface`. Swift twin:
    ``ExperimentStore.sweepChoicePinEntries``."""
    if not isinstance(sweep, dict):
        return []
    objective = (sweep.get("selection") or {}).get("objective") or {}
    if not isinstance(objective, dict):
        return []
    if objective.get("metric") != "logprobShift":
        # Only logprobShift READS choice instruments (review 2026-08-02
        # round 2, P2): a stale path carried under judgeScore/markerDensity
        # is inert at execution, so pinning it would let dead declarations
        # block freezes over files nothing reads.
        return []
    entries: list[tuple] = []
    singular = objective.get("choicePromptsFile")
    if isinstance(singular, str) and singular.strip():
        entries.append((None, singular, objective.get("choicePromptsHash"),
                        "sweep choice prompts"))
    mapping = objective.get("choicePromptsFiles")
    if isinstance(mapping, dict):
        hashes = objective.get("choicePromptsHashes")
        hashes = hashes if isinstance(hashes, dict) else {}
        for concept in sorted(mapping):
            rel = mapping[concept]
            if isinstance(rel, str):
                entries.append((concept, rel, hashes.get(concept),
                                f"sweep choice prompts '{concept}'"))
    return entries


def markers_aggregate_hash(concept_names: list[str],
                           root: str | None = None) -> str | None:
    """Study-level aggregate hash over every attached concept's markers.json
    (cross-engine contract for the manifest key ``markersHash``).

    Rule (both engines MUST implement identically): for each DISTINCT concept
    name, sorted ascending by Unicode code point, whose
    ``prompts/concepts/<name>/markers.json`` exists, emit the line
    ``"<name>\\t<sha256-hex-of-raw-bytes>\\n"``; the aggregate is the SHA-256
    hex digest of the UTF-8 concatenation of those lines. Returns None when no
    attached concept has a markers.json (the pin is then null)."""
    lines: list[str] = []
    for name in sorted(set(concept_names)):
        path = os.path.join(paths.concept_directory(name, root), "markers.json")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        lines.append(f"{name}\t{digest}\n")
    if not lines:
        return None
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


#: The sidecar schema version at which a trained adapter carries evidence-grade
#: training provenance (LoRA readiness contract §7). Anything below it — or no
#: sidecar at all — is an EXPLORATORY adapter: legal in a study, never
#: evidence, and surfaced as a non-blocking freeze advisory.
TRAINING_PROVENANCE_SCHEMA_VERSION = 2


def variant_adapter_reference(artifact: dict) -> str | None:
    """The workspace reference of a variant's FIRST adapter directory, or None
    when the variant applies no adapter (a pure injection/prompt agent).
    Mirrors the reference resolution in ``model_variant.missing_artifacts``."""
    for adapter in (artifact or {}).get("adapters") or []:
        if not isinstance(adapter, dict):
            continue
        ref = adapter.get("adapterDirectory") or adapter.get("artifactPath") or ""
        if ref:
            return str(ref)
    return None


def variant_adapter_sidecar(artifact: dict,
                            root: str | None = None) -> tuple[str | None, dict | None]:
    """``(absolute sidecar path, parsed sidecar)`` for a variant's adapter.

    The path is returned even when the file is absent or unreadable (the
    caller reports the absence — a pinned-but-missing sidecar is a verify
    violation, an unpinned missing one is merely a legacy adapter). ``(None,
    None)`` means the variant declares no adapter at all."""
    from . import model_variant

    ref = variant_adapter_reference(artifact)
    if not ref:
        return None, None
    path = model_variant.adapter_sidecar_path(paths.resolve_artifact(ref, root))
    try:
        with open(path, encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return path, None
    return path, (parsed if isinstance(parsed, dict) else None)


def sidecar_is_evidence_grade(sidecar: dict | None) -> bool:
    """Whether a training sidecar claims EVIDENCE-GRADE provenance: schema v2
    or later AND ``evidenceGrade: true``. A v1 sidecar has no such key and no
    such claim, so it is exploratory by construction."""
    if not isinstance(sidecar, dict):
        return False
    try:
        version = int(sidecar.get("schemaVersion") or 0)
    except (TypeError, ValueError):
        return False
    return (version >= TRAINING_PROVENANCE_SCHEMA_VERSION
            and bool(sidecar.get("evidenceGrade")))


def variant_is_evidence_grade(variant: dict, root: str | None = None) -> bool:
    """Whether an adapter variant is evidence-grade, from EITHER side of the
    pin: the manifest's own ``trainingProvenance.evidenceGrade`` (checkable on
    any engine, including a Mac that never sees the server's runs/ tree) or
    the adapter sidecar's stamp where the file is readable. The union is
    deliberate — the gate must not become unenforceable just because the
    artifact tree is remote, and it must not be dodgeable by deleting a
    manifest key."""
    block = variant.get("trainingProvenance")
    if isinstance(block, dict) and bool(block.get("evidenceGrade")):
        return True
    _, sidecar = variant_adapter_sidecar(variant.get("artifact") or {}, root)
    return sidecar_is_evidence_grade(sidecar)


def _verify_training_provenance(base: str, variant: "VariantCondition",
                                root: str | None) -> list[str]:
    """Drift/completeness violations for a variant's ``trainingProvenance``
    block (LoRA readiness §0 amendment 1: the dataset-bundle hash joins the
    freeze pin surface). Absent block = no violations; a PRESENT block is
    checked exactly like stimuli — a missing file or changed bytes refuses,
    and a half-pin (path without hash, or hash without path) certifies
    nothing and refuses too."""
    block = variant.training_provenance
    if not isinstance(block, dict):
        return []
    violations: list[str] = []
    label = f"variant '{variant.name}' trainingProvenance"
    manifest_path = str(block.get("datasetManifestPath") or "").strip()
    manifest_hash = str(block.get("datasetManifestHash") or "").strip()
    if manifest_path and not manifest_hash:
        violations.append(
            f"{label} names dataset manifest '{manifest_path}' with no "
            "datasetManifestHash — a half-pin certifies nothing")
    elif manifest_hash and not manifest_path:
        violations.append(
            f"{label} pins a datasetManifestHash with no "
            "datasetManifestPath — a half-pin certifies nothing")
    elif manifest_path:
        violations += _verify_file_hash(
            base, manifest_path, manifest_hash, f"{label} dataset manifest")
    sidecar_hash = str(block.get("adapterSidecarHash") or "").strip()
    if sidecar_hash:
        path, _ = variant_adapter_sidecar(variant.artifact or {}, root)
        if path is None:
            violations.append(
                f"{label} pins an adapterSidecarHash but the variant "
                "declares no adapter directory")
        elif not os.path.exists(path):
            violations.append(
                f"{label} adapter sidecar: file missing at {path}")
        else:
            with open(path, "rb") as handle:
                live = hashlib.sha256(handle.read()).hexdigest()
            if live != sidecar_hash:
                violations.append(
                    f"{label} adapter sidecar changed since pinning "
                    f"(have {live[:12]}…, pinned {sidecar_hash[:12]}…)")
    return violations


def _verify_file_hash(base: str, rel_path: str, pinned: str, label: str) -> list[str]:
    path = rel_path if os.path.isabs(rel_path) else os.path.join(base, rel_path)
    if not os.path.exists(path):
        return [f"{label}: file missing at {rel_path}"]
    with open(path, "rb") as handle:
        live = hashlib.sha256(handle.read()).hexdigest()
    if live != pinned:
        return [f"{label} changed since pinning (have {live[:12]}…, pinned {pinned[:12]}…)"]
    return []


# --- pin SHAPE checks at verify (cross-engine with Swift's ---------------
# PinShapeValidation, 2026-07-19): a pin certifies identity (SHA-256), so a
# present-but-malformed file used to pass verify and die much later — the
# battery at task load, the baseline at analyze. Verify now ALSO checks the
# shape the consumer requires, but ONLY when the hash matches: drift and
# missing files are already their own violations (never double-reported).
# The checks mirror the CONSUMER of each file, never a stricter invented
# schema: the battery mirrors ``battery.load_battery`` (the loader every
# battery run uses on this engine), the baseline mirrors
# ``residuals.load_human_baseline`` / ``HUMAN_BASELINE_FIELDS``.

def _battery_shape_violations(base: str, rel_path: str) -> list[str]:
    """Shape problems in a hash-clean pinned capability battery, by running
    THE LOADER every battery run uses (``battery.load_battery``, 2026-07-20
    — the old reimplementation drifted loose: it accepted non-string
    prompts, coerced answers, and any grading name, so a battery could
    verify here and refuse on Swift). Loader errors — including an empty
    battery, which would silently score every condition 0-of-0 — surface
    verbatim with the standard remedy."""
    from . import battery as _battery
    remedy = ('each line must be a JSON object like '
              '{"prompt": "…", "answer": "…"} (optional "grading") — start '
              'from prompts/batteries/basic.jsonl')
    try:
        _battery.load_battery(rel_path, root=base)
    except (OSError, UnicodeDecodeError):
        return [f"the capability battery '{rel_path}' is not readable as "
                f"UTF-8 text; {remedy}"]
    except ValueError as err:
        return [f"the capability battery '{rel_path}' is not shaped the "
                f"way battery runs read it ({err}); {remedy}"]
    return []


def _human_baseline_shape_violations(base: str, rel_path: str) -> list[str]:
    """Shape problems in a hash-clean pinned human-baseline CSV, mirroring
    the analyze loader (``residuals.load_human_baseline``: ``csv.DictReader``
    + ``float()``): the required columns must all be present (extra columns
    are fine), and every data row's numeric fields must parse. Names the
    required and found column lists, or the first bad data row and field."""
    import csv as _csv
    import io as _io

    from .residuals import HUMAN_BASELINE_FIELDS

    path = rel_path if os.path.isabs(rel_path) else os.path.join(base, rel_path)
    try:
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return [f"the analyze step reads '{rel_path}' as a UTF-8 CSV — the "
                "file is not readable as one"]
    # No BOM stripping here: the analyze loader itself does none, so a
    # BOM'd header would genuinely fail at analyze — the missing-columns
    # branch below reports it (mirror the consumer, never a looser rule).
    reader = _csv.DictReader(_io.StringIO(text))
    found = list(reader.fieldnames or [])
    missing = [f for f in HUMAN_BASELINE_FIELDS if f not in found]
    if missing:
        required = ", ".join(HUMAN_BASELINE_FIELDS)
        have = ("columns " + ", ".join(found)) if found \
            else "no header columns at all"
        return [f"the analyze step reads human-baseline columns {required} — "
                f"'{rel_path}' has {have}; add the missing column(s) "
                f"({', '.join(missing)}) or start from Create from template "
                "(extra columns are fine)"]
    numeric = [f for f in HUMAN_BASELINE_FIELDS if f != "endpoint"]
    for index, row in enumerate(reader):
        for fld in numeric:
            value = row.get(fld)
            if value is None:
                return [f"the analyze step reads {fld} as a number on every "
                        f"data row — '{rel_path}' data row {index + 1} has "
                        f"no {fld} value; fill the cell in (blank lines are "
                        "fine, blank cells are not)"]
            try:
                float(value)
            except (TypeError, ValueError):
                return [f"the analyze step reads {fld} as a number on every "
                        f"data row — '{rel_path}' data row {index + 1} has "
                        f"{fld} '{value.strip()}', which is not a number; "
                        "fix the value (quoted numbers and scientific "
                        "notation are fine)"]
    return []


#: Swift Codable enum case name → the label-building constructor. Cases with
#: an associated value carry it under ``_0`` in Swift's synthesized encoding.
_READING_POSITION_CASES = {
    "lastToken": lambda _: LAST_TOKEN,
    "meanFromToken": lambda k: mean_from_token(int(k)),
    "offsetFromEnd": lambda k: _offset_from_end(int(k)),
    "lastContentToken": lambda _: _last_content_token(),
    "turnCloseToken": lambda _: _turn_close_token(),
    "postInstruction": lambda i: _post_instruction(int(i)),
}


def _offset_from_end(k):
    from ..steering.reading_position import offset_from_end
    return offset_from_end(k)


def _post_instruction(i):
    from ..steering.reading_position import post_instruction
    return post_instruction(i)


def _last_content_token():
    from ..steering.reading_position import LAST_CONTENT_TOKEN
    return LAST_CONTENT_TOKEN


def _turn_close_token():
    from ..steering.reading_position import TURN_CLOSE_TOKEN
    return TURN_CLOSE_TOKEN


def _parse_reading_position(value) -> ReadingPosition:
    """Parse the manifest ``readingPosition`` (Swift Codable enum or a label).

    Swift synthesizes an enum-with-associated-value as ``{"lastToken": {}}`` or
    ``{"meanFromToken": {"_0": k}}``; older/looser data may use the label
    string. Named template-aware roles (``lastContentToken``,
    ``turnCloseToken``, ``postInstruction``) travel the same two ways.
    """
    if value is None:
        return LAST_TOKEN
    if isinstance(value, str):
        return from_label(value)
    if isinstance(value, dict):
        for case, make in _READING_POSITION_CASES.items():
            if case not in value:
                continue
            inner = value[case]
            parameter = inner.get("_0") if isinstance(inner, dict) else inner
            try:
                return make(parameter)
            except (TypeError, ValueError):
                return LAST_TOKEN
    return LAST_TOKEN
