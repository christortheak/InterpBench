"""Template-mediated RepE reader LAT (Zou et al., arXiv:2310.01405 §3.1,
App. C.1).

A reader is a fitted **measurement instrument**, not a steering vector: task
template + LAT token position + PCA direction with training normalization +
held-out sign/layer selection + held-out scalar accuracy (see
``docs/REPE-IMPLEMENTATION-BRIEF.md``). Pipeline:

    stimulus → render task template → capture hidden state at the LAT token
    → per-pair differences → centered PCA → PC1 signed on the HELD-OUT split
    → ScalarProbe fitted on the TRAIN activations

**Three split roles, and the reason they are named on the artifact.** The
held-out split does double duty: it fixes the sign AND ranks the layers, so
``heldOutAccuracy`` is the score of the winner of two selections made on the
very rows it is computed over — a model-selection statistic, not an untouched
estimate of generalization. A dataset that wants the latter reserves rows with
``split: "finalTest"``; nothing that fits or selects reads them, and they are scored
once, at the end, as ``finalTestAccuracy``. Every artifact stamps an
``evidenceRoles`` block plus :data:`EVIDENCE_ROLE_NOTE` so no consumer has to
know any of this to report the numbers honestly.

Inference renders the *same* template under the *same* rendering, captures the
*same* token position, and scores through the stored probe (training
center/scale) — never a raw cosine-to-vector shortcut. A reader can *derive* a
steering vector (:func:`derive_steering_vector`), but the derived artifact
stamps its reader provenance so "reading-vector activation addition" is never
conflated with the paper's full control experiments.

**Faithful, and where it still departs.** Implemented from the paper and the
reference implementation: the task template (§3.1) with its LAT token at the
rendered scaffold's final position (``rep_token=-1``); PCA over paired
differences with ``n_components=1`` and mean-centering only
(``repe/rep_readers.py``, ``PCARepReader``); the difference construction
``[::2] − [1::2]`` over per-pair randomized orientation
(``rep_reading_pipeline.py`` plus the dataset builder's ``random.shuffle(d)``);
BOTH contrast constructions — the supervised content contrast and the paper's
unsupervised T+/T− instruction pair (§3.1 step 1b); held-out sign AND layer
selection (the paper's step 4, e.g. its 25 ARC-Challenge validation examples);
and either rendering — raw scaffold or the family chat template, the repo's
``user_tag``/``assistant_tag`` analogue.

The departures that REMAIN, each deliberate and each stamped:

- **``latToken`` supports only ``"final"``.** The registry schema carries the
  field so a second position is a data change, but only ``final`` is
  implemented, and any other value is refused rather than silently coerced.
- **Scoring is a calibrated scalar probe**, not the paper's logistic /
  ``pca_model.transform`` pipeline. The center and scale come from the train
  projections and are persisted, which reproduces "normalize test activations
  with the training parameters"; the classifier on top is our midpoint rule.
- **Reader artifacts are substrate-specific by rule** (:data:`SUBSTRATE`).
  Activations do not transfer between PyTorch and MLX, so a Swift/MLX reader
  must be re-fitted here. The paper has one substrate.
- **Datasets are authored, not borrowed.** The paper fits on published corpora
  (TQA, ARC); a study here fits on its own pinned pairs.

Concept-agnostic by design: concepts, templates, and stimuli enter as data.
Template rendering goes through the family-aware renderer
(``experiment.prompt_render.render_reader``) — that module is tokenizer/family
convention, not experiment logic, and keeping one rendering authority is what
prevents the double-BOS / hand-tokenized-scaffold bug class.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ..experiment.prompt_render import (READER_CHAT_TEMPLATE_RENDERING_CONVENTION,
                                        READER_RENDERING_CONVENTION,
                                        reader_rendering_convention, render_reader)
from . import extractor
from . import vector_math as vm
from .extraction_rendering import (ExtractionRendering, RAW_RENDERING,
                                   from_json as _rendering_from_json)
from .reading_position import LAST_TOKEN, ReadingPosition
from .vector_math import ScalarProbe
from .vector_store import SUBSTRATE  # single shared definition (re-exported here)

ARTIFACT_TYPE = "repe-reader-lat"
CONTROL_MODE = "reading-vector activation addition"

#: WHAT the two activations in a pair differ by. Both are first-class: the
#: paper describes both, and neither is a degraded form of the other. Swift
#: twin: ``RepEReader.ContrastMode``.
#:
#: - ``supervisedContent``: two DIFFERENT stimuli under ONE template — the
#:   concept-present stimulus against its matched control. This is what every
#:   reader fitted before 2026-08-27 did, so an absent stamp means this.
#: - ``unsupervisedTemplatePair``: ONE stimulus under TWO templates — the
#:   paper's experimental / reference instruction pair (T+ / T−). The stimulus
#:   is held fixed and the INSTRUCTION carries the contrast, so the direction
#:   cannot be a content artifact of two different texts.
SUPERVISED_CONTENT = "supervisedContent"
UNSUPERVISED_TEMPLATE_PAIR = "unsupervisedTemplatePair"
CONTRAST_MODES = (SUPERVISED_CONTENT, UNSUPERVISED_TEMPLATE_PAIR)

#: HOW a layer's PC1 sign was fixed. Swift twin: ``RepEReader.SignConvention``.
#:
#: - ``heldOutPairAgreement``: the paper's step 4 — the held-out split decides.
#: - ``trainMajority``: ``get_signs``' train-label agreement, which is what the
#:   reference implementation ships and what every reader fitted before
#:   2026-08-27 used. An absent stamp means this.
HELD_OUT_PAIR_AGREEMENT = "heldOutPairAgreement"
TRAIN_MAJORITY = "trainMajority"
SIGN_CONVENTIONS = (HELD_OUT_PAIR_AGREEMENT, TRAIN_MAJORITY)

#: The fewest held-out pairs that may decide a sign. Below it the held-out vote
#: is one or two coin flips wearing the authority of a validation split, so the
#: fit falls back to train-majority AND says so in the artifact
#: (``signFallbackReason``) rather than quietly pretending.
MINIMUM_HELD_OUT_PAIRS_FOR_SIGN_SELECTION = 2

#: Default seed for the unsupervised mode's per-row orientation draw.
#: Deterministic and stamped: the reference implementation shuffles each pair
#: with an unseeded ``random.shuffle``, which makes its direction
#: irreproducible; we keep the paper's ± symmetry and drop its
#: irreproducibility. ``231001405`` is the paper's arXiv id.
DEFAULT_ORIENTATION_SEED = 231_001_405

#: The three SPLIT ROLES a dataset row can take, and what each one is allowed
#: to touch. Swift twin owed: ``RepEReader`` knows only train / not-train, so
#: it would read a ``finalTest`` row as HELD OUT — a dataset that uses the
#: role must be fitted on this engine until that parity lands (brief, §3).
#:
#: - ``"train"`` (the default, and what an absent ``split`` means) — fits PC1
#:   and the probe's center/scale.
#: - ``"finalTest"`` (case-insensitive) — FINAL EVALUATION. Read by nothing
#:   that fits or selects: not the direction, not the sign, not the layer
#:   recommendation, not the probe's normalization. Scored once, after the
#:   fit, as ``finalTestAccuracy``.
#: - anything else (``"test"``, ``"heldOut"``, ``"validation"``, ``"dev"`` …)
#:   — HELD OUT, exactly as every non-``train`` value has always been: it
#:   fixes the sign (paper step 4) and ranks the layers, which is precisely
#:   why its accuracy is a SELECTION statistic and not a final-test estimate.
#:
#: ``"test"`` stays held out ON PURPOSE. It is the spelling the app's corpus
#: writer (``ConceptBuilder``), its import parser and the authoring prompt
#: have always used for held-out rows, on both engines; repurposing that word
#: would have changed the meaning of every existing corpus. The final-
#: evaluation role got a new word instead (2026-09-05).
TRAIN_SPLIT = "train"
#: Lower-case, because ``parse_pairs`` lower-cases ``split`` before comparing;
#: the canonical spelling in a file is ``finalTest``.
FINAL_TEST_SPLIT = "finaltest"

#: What each accuracy on the artifact IS, in one word. Roles, not adjectives:
#: a number's role says which decisions were allowed to read the rows it was
#: computed on, and that is the only thing that makes it evidence of anything.
FIT_ROLE = "fit"
SELECTION_ROLE = "selection"
VALIDATION_ROLE = "validation"
FINAL_EVALUATION_ROLE = "finalEvaluation"

#: Which split decided something: the values of ``signSelectedBy`` and
#: ``layerRecommendedBy`` in an ``evidenceRoles`` block.
BY_HELD_OUT = "heldOut"
BY_TRAIN = "train"

#: Where an ``evidenceRoles`` block came from. ``"stamped"`` = written by the
#: engine that fitted (or by an artifact that already carried one);
#: ``"derivedFromLegacyStamps"`` = reconstructed at decode from a schema-1/2
#: artifact's ``signConvention`` and ``layerRecommendationBasis``, which is a
#: reading of the artifact, not a record it kept.
EVIDENCE_ROLES_STAMPED = "stamped"
EVIDENCE_ROLES_DERIVED = "derivedFromLegacyStamps"

#: What a reader must be told about this instrument's ACCURACIES, in one
#: paragraph, wherever they are surfaced. The sibling of
#: :data:`LAYER_RECOMMENDATION_NOTE`, and for the same reason: the artifact
#: says what its own numbers mean, so no consumer has to know the fit's
#: internals to report them honestly.
EVIDENCE_ROLE_NOTE = (
    "heldOutAccuracy is a MODEL-SELECTION statistic: the same held-out rows "
    "chose the direction's sign and ranked the layers, so it is not an "
    "untouched final-test estimate. Final generalization evidence is "
    "finalTestAccuracy, scored on rows marked split 'finalTest' that no "
    "fitting or selection step read. When finalTestAccuracy is absent the "
    "dataset reserved no such rows, and the result must be reported as "
    "selection/validation evidence, not as a final test.")

#: What a reader must be told about this instrument's layer, in one sentence,
#: wherever the recommendation is surfaced.
LAYER_RECOMMENDATION_NOTE = (
    "recommendedLayer is the argmax of held-out accuracy over the layers "
    "fitted together; it is a RECOMMENDATION. The layer a study reads is "
    "declared in the manifest and is never selected automatically at use time.")

_UINT64 = 0xFFFF_FFFF_FFFF_FFFF


class RepeReaderError(Exception):
    pass


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_text(text: str) -> str:
    """Whitespace-collapsed, case-folded text — the comparison form for the
    cross-split leakage check ONLY. Nothing that reaches a model is normalized
    this way: the stimulus is rendered verbatim, as it always was."""
    return " ".join(str(text).split()).casefold()


def _split_mix64(state: int) -> tuple[int, int]:
    """SplitMix64 — the shared cross-engine PRNG for the orientation draw.

    Chosen because it is four lines in both languages and produces
    bit-identical streams, so an unsupervised fit is reproducible across
    substrates from the seed alone. Swift twin: ``RepEReader.splitMix64``.
    """
    state = (state + 0x9E37_79B9_7F4A_7C15) & _UINT64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _UINT64
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _UINT64
    return state, z ^ (z >> 31)


def orientation_signs(count: int, seed: int) -> list[float]:
    """``count`` orientations in ±1, drawn in row order from ``seed``. Swift
    twin: ``RepEReader.orientationSigns``."""
    state = seed & _UINT64
    signs: list[float] = []
    for _ in range(max(0, count)):
        state, value = _split_mix64(state)
        signs.append(1.0 if value & 1 == 0 else -1.0)
    return signs


# --- task templates ----------------------------------------------------------

@dataclass(frozen=True)
class InstructionPair:
    """The paper's T+ / T− instruction pair (§3.1 step 1b).

    A template that declares one is a TEMPLATE-PAIR template: the same stimulus
    is rendered twice, once under each instruction, and the difference is
    H(T+) − H(T−). Both strings substitute into the template's
    ``{{instruction}}`` slot.

    Hygiene rules apply exactly as they do to a scaffold: an instruction that
    names the study concept makes the reader a concept-word detector, so the
    shipped example names none.
    """

    #: T+ — the instruction under which the concept is present.
    experimental: str
    #: T− — the matched reference instruction.
    reference: str

    def to_dict(self) -> dict:
        return {"experimental": self.experimental, "reference": self.reference}

    @classmethod
    def from_dict(cls, d: dict) -> "InstructionPair":
        return cls(experimental=str(d.get("experimental", "")),
                   reference=str(d.get("reference", "")))


#: The instruction slot a template-pair template must carry.
INSTRUCTION_SLOT = "{{instruction}}"


@dataclass(frozen=True)
class TaskTemplate:
    """One registry entry from ``prompts/templates/<id>.json``.

    ``hash`` is the SHA-256 of the file's raw bytes (stimulus-set convention):
    changing the template changes every artifact fitted through it.
    ``divergence`` marks deliberate departures from the paper (e.g. the
    unnamed clean-room scaffold, which never names the concept).
    ``instruction_pair`` is absent for a single-template (supervised content)
    template and present for a T+/T− template-pair template; absent is the
    legacy shape, so every template file written before 2026-08-27 keeps its
    hash.
    """

    id: str
    text: str
    concept_slot: bool
    lat_token: str
    hash: str
    divergence: str | None = None
    instruction_pair: InstructionPair | None = None

    @property
    def is_template_pair(self) -> bool:
        """Whether this template carries the paper's T+/T− instruction pair."""
        return self.instruction_pair is not None

    @property
    def reading_position(self) -> ReadingPosition:
        if self.lat_token != "final":
            raise RepeReaderError(
                f"template {self.id!r}: unsupported latToken {self.lat_token!r} "
                "(only 'final' is implemented)")
        return LAST_TOKEN

    def validate_instruction_slot(self) -> None:
        """Schema coherence between the instruction pair and the slot. Called
        by :func:`load_template` so a malformed registry file is refused at
        load rather than producing a scaffold with a literal
        ``{{instruction}}`` in it."""
        has_slot = INSTRUCTION_SLOT in self.text
        if self.instruction_pair is not None:
            if not has_slot:
                raise RepeReaderError(
                    f"template {self.id!r} declares an instructionPair but its text "
                    f"has no {INSTRUCTION_SLOT} slot — the T+/T− instructions would "
                    f"never reach the model. Repair: add {INSTRUCTION_SLOT} to the "
                    "text, or drop instructionPair to make this a single-template "
                    "reader")
            if not self.instruction_pair.experimental or \
                    not self.instruction_pair.reference:
                raise RepeReaderError(
                    f"template {self.id!r}: instructionPair needs both "
                    "'experimental' (T+) and 'reference' (T−) — one empty "
                    "instruction makes the contrast a rendering artifact. Repair: "
                    "write both instructions")
            if self.instruction_pair.experimental == self.instruction_pair.reference:
                raise RepeReaderError(
                    f"template {self.id!r}: instructionPair's experimental and "
                    "reference instructions are identical — every difference would "
                    "be exactly zero. Repair: write two instructions that differ in "
                    "the quality under study")
        elif has_slot:
            raise RepeReaderError(
                f"template {self.id!r} has a {INSTRUCTION_SLOT} slot but declares no "
                "instructionPair — nothing would fill it. Repair: add an "
                "instructionPair with 'experimental' and 'reference', or remove the "
                "slot")

    def render(self, *, stimulus: str, concept: str | None = None,
               instruction: str | None = None) -> str:
        """Pure slot substitution; the family-aware scaffold pass happens in
        :func:`render_scaffold`. ``instruction`` fills ``{{instruction}}`` and
        is required exactly when the template declares an instruction pair."""
        if "{{stimulus}}" not in self.text:
            raise RepeReaderError(f"template {self.id!r} has no {{{{stimulus}}}} slot")
        text = self.text
        if self.concept_slot:
            if not concept:
                raise RepeReaderError(
                    f"template {self.id!r} names the concept but none was given")
            text = text.replace("{{concept}}", concept)
        elif "{{concept}}" in text:
            raise RepeReaderError(
                f"template {self.id!r} declares conceptSlot=false but contains "
                "a {{concept}} slot")
        if self.is_template_pair:
            if not instruction:
                raise RepeReaderError(
                    f"template {self.id!r} is a T+/T− template-pair template but no "
                    "instruction was given — render it once per instruction")
            text = text.replace(INSTRUCTION_SLOT, instruction)
        elif INSTRUCTION_SLOT in text:
            raise RepeReaderError(
                f"template {self.id!r} declares no instructionPair but contains an "
                f"{INSTRUCTION_SLOT} slot")
        return text.replace("{{stimulus}}", stimulus)

    def to_dict(self) -> dict:
        out = {"id": self.id, "conceptSlot": self.concept_slot, "text": self.text,
               "latToken": self.lat_token, "hash": self.hash}
        if self.divergence is not None:
            out["divergence"] = self.divergence
        if self.instruction_pair is not None:
            out["instructionPair"] = self.instruction_pair.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict, *, hash: str | None = None) -> "TaskTemplate":
        pair = d.get("instructionPair")
        return cls(id=str(d["id"]), text=str(d["text"]),
                   concept_slot=bool(d.get("conceptSlot", False)),
                   lat_token=str(d.get("latToken", "final")),
                   hash=str(hash if hash is not None else d.get("hash", "")),
                   divergence=d.get("divergence"),
                   instruction_pair=(InstructionPair.from_dict(pair)
                                     if isinstance(pair, dict) else None))


def parse_template(data: bytes | str, *, source: str) -> TaskTemplate:
    """Parse one template from BYTES, with every rule :func:`load_template`
    applies except the filename one — the in-memory seam an upload needs.

    ``parse_pairs`` exists for exactly this reason on the corpus side: an
    uploaded pairs file is validated in memory so a rejected upload never
    reaches (or clobbers) its canonical home. Templates had no equivalent, and
    the reader-fit route therefore checked an uploaded template for JSON-ness,
    an ``id`` and a ``text`` and nothing more — then replaced the canonical
    ``prompts/readers/<concept>/pairs.jsonl`` and queued a job that discovered,
    minutes later, that the template could not render. Review round 6,
    finding 4.

    ``hash`` is SHA-256 over the raw bytes, exactly as the file loader computes
    it, so a template validated here and persisted verbatim keeps one identity.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RepeReaderError(f"{source}: invalid template JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise RepeReaderError(f"{source}: template must be a JSON object")
    for key in ("id", "text"):
        if key not in obj:
            raise RepeReaderError(f"{source}: template missing {key!r}")
    template = TaskTemplate.from_dict(obj, hash=_sha256_hex(raw))
    template.validate_instruction_slot()
    return template


def load_template(path: str) -> TaskTemplate:
    """Load one template JSON; hash = SHA-256 over the file's raw bytes."""
    if not os.path.exists(path):
        raise RepeReaderError(f"missing template file: {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    try:
        obj = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RepeReaderError(f"{path}: invalid template JSON: {exc}") from exc
    for key in ("id", "text"):
        if key not in obj:
            raise RepeReaderError(f"{path}: template missing {key!r}")
    template = TaskTemplate.from_dict(obj, hash=_sha256_hex(data))
    expected = os.path.basename(path)[:-len(".json")] if path.endswith(".json") else None
    if expected and template.id != expected:
        raise RepeReaderError(
            f"{path}: template id {template.id!r} does not match filename "
            f"{expected!r} — the registry is one file per id")
    template.validate_instruction_slot()
    return template


def render_scaffold(template: TaskTemplate, *, stimulus: str,
                    concept: str | None, model_id: str,
                    instruction: str | None = None,
                    rendering: ExtractionRendering | None = None) -> str:
    """Template substitution + the family-aware scaffold pass (amendment A)."""
    return render_reader(
        template.render(stimulus=stimulus, concept=concept, instruction=instruction),
        model_id=model_id, rendering=rendering)


# --- reader dataset ----------------------------------------------------------

@dataclass(frozen=True)
class ReaderPair:
    """One row: unrendered stimuli + the template that will render them
    (REPE-IMPLEMENTATION-BRIEF §3). Storing the raw stimulus keeps the corpus
    re-renderable per model family while the hash pins the bytes.

    Two shapes, one file format:

    - **content pair** — ``positive_stimulus`` + ``negative_stimulus``: two
      different texts under one template (the legacy and still-default shape).
    - **single stimulus** — ``stimulus``: ONE text, rendered under a
      template-pair template's T+ and T− instructions. The contrast is the
      instruction, so a second text would be a confound, not data.
    """

    positive_stimulus: str
    negative_stimulus: str
    concept: str
    template_id: str
    id: str | None = None
    topic: str | None = None
    #: One of the three roles in :data:`TRAIN_SPLIT` / :data:`FINAL_TEST_SPLIT`
    #: / anything-else-is-held-out. Lower-cased at parse, so ``"FinalTest"``
    #: is ``"finaltest"``.
    split: str = TRAIN_SPLIT
    stimulus: str | None = None

    @property
    def is_template_pair_row(self) -> bool:
        return self.stimulus is not None

    @property
    def is_final_test(self) -> bool:
        """A FINAL-TEST row: excluded from the fit, from sign selection, from
        the layer recommendation and from the probe's normalization."""
        return self.split == FINAL_TEST_SPLIT

    @property
    def content_key(self) -> str:
        """The row's TEXT identity, ignoring ``id`` and ``topic``.

        Whitespace-normalized and case-folded; a content pair's identity is
        BOTH its stimuli (swapping them is a different row), a template-pair
        row's is its single stimulus. Convention borrowed from
        ``lora_data.Row.content_key``, which exists for the same reason: the
        same passage under two ids is still the same passage in both splits.
        """
        if self.stimulus is not None:
            return _normalized_text(self.stimulus)
        return "\x00".join((_normalized_text(self.positive_stimulus),
                            _normalized_text(self.negative_stimulus)))


@dataclass(frozen=True)
class ReaderDataset:
    concept: str
    pairs: tuple[ReaderPair, ...]
    hash: str

    @property
    def train(self) -> list[ReaderPair]:
        return [p for p in self.pairs if p.split == TRAIN_SPLIT]

    @property
    def held_out(self) -> list[ReaderPair]:
        """The SELECTION rows: every split that is neither ``train`` nor
        ``finalTest`` — ``test``, ``heldOut``, ``validation`` and the rest,
        exactly as every non-``train`` value has always been read."""
        return [p for p in self.pairs
                if p.split not in (TRAIN_SPLIT, FINAL_TEST_SPLIT)]

    @property
    def final_test(self) -> list[ReaderPair]:
        """The FINAL-TEST rows: fitted on by nothing, selected on by nothing,
        scored once at the end."""
        return [p for p in self.pairs if p.is_final_test]

    @property
    def split_counts(self) -> dict:
        return {"train": len(self.train), "heldOut": len(self.held_out),
                "finalTest": len(self.final_test)}

    @property
    def shape(self) -> str:
        """``"singleStimulus"`` or ``"contentPair"``. A dataset is one shape or
        the other, never both: the two produce different differences, so a
        mixed file has no single meaning."""
        if self.pairs and self.pairs[0].is_template_pair_row:
            return "singleStimulus"
        return "contentPair"


def _row_label(pair: ReaderPair, position: int) -> str:
    """How a row is named in a refusal: its ``id`` when it has one, else its
    1-based position in the file (rows are optional-id by schema)."""
    return f"row {pair.id!r}" if pair.id else f"row #{position}"


def check_split_overlap(dataset: ReaderDataset, *,
                        source: str = "<reader pairs>") -> dict:
    """Refuse a dataset whose held-out or test rows repeat a train row's text,
    and return the stamp that records the check ran.

    The rule, its content-identity convention and its wording follow
    ``lora_data.load_split_rows`` / ``lora_data.Row.content_key``, which
    refuses a validation row present in training for exactly this reason:
    evidence scored on rows the fit already read measures memorization. A
    reader's rows are stimuli rather than training examples, so identity is the
    stimulus text — BOTH stimuli of a content pair, the single stimulus of a
    template-pair row — normalized for whitespace and case (:func:
    `_normalized_text`).

    **EXACT duplicates only.** Near-duplicates — a paraphrase, one edited
    clause, the same scenario with a renamed subject — are NOT assessed here
    and the stamp does not claim they were: ``exactDuplicatesAcrossSplits`` is
    the only key it carries. A fuzzy check needs a threshold, and an unstated
    threshold in a leakage gate is worse than an honest gap.

    Swift twin owed: ``RepEReader`` has no split-overlap check.
    """
    by_key: dict[str, tuple[ReaderPair, int]] = {}
    for position, pair in enumerate(dataset.pairs, 1):
        if pair.split == TRAIN_SPLIT:
            by_key.setdefault(pair.content_key, (pair, position))
    for position, pair in enumerate(dataset.pairs, 1):
        if pair.split == TRAIN_SPLIT:
            continue
        found = by_key.get(pair.content_key)
        if found is None:
            continue
        train_pair, train_position = found
        raise RepeReaderError(
            f"{source}: {_row_label(pair, position)} (split {pair.split!r}) has "
            f"the same stimulus text as train "
            f"{_row_label(train_pair, train_position)} — a held-out or test row "
            "identical to a train row is a leak: the fit already read those "
            "words, so the accuracy scored on them measures memorization, not "
            "generalization. Repair: rewrite or drop the duplicate row so each "
            "split holds distinct stimuli")
    return {"exactDuplicatesAcrossSplits": 0}


def _require_row_keys(obj: dict, keys, source: str) -> None:
    for key in keys:
        if key not in obj:
            raise RepeReaderError(f"{source}: row missing {key!r}")


def parse_pairs(data: bytes | str, *, source: str = "<reader pairs>") -> ReaderDataset:
    """Parse pairs JSONL from raw bytes; hash = SHA-256 over those bytes
    (stimulus_set convention). Rows must share one concept — a reader measures
    exactly one — and one shape. This is the in-memory seam under
    :func:`load_pairs`, so an uploaded corpus can be fully validated *before*
    any canonical file is written; ``source`` names the origin in error
    messages."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    pairs: list[ReaderPair] = []
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RepeReaderError(f"{source}: invalid JSON line: {exc}") from exc
        split = str(obj.get("split") or "train").lower()
        # Shape first (which keys a row OWES depends on it), then the two keys
        # every row owes whatever its shape.
        if "stimulus" in obj:
            if "positiveStimulus" in obj or "negativeStimulus" in obj:
                raise RepeReaderError(
                    f"{source}: row declares both 'stimulus' and a positive/negative "
                    "pair — a template-pair row holds ONE stimulus (the T+/T− "
                    "instructions carry the contrast). Repair: drop 'stimulus' for a "
                    "content pair, or drop 'positiveStimulus'/'negativeStimulus' for "
                    "a template pair")
            _require_row_keys(obj, ("concept", "templateID"), source)
            pairs.append(ReaderPair(
                positive_stimulus="", negative_stimulus="",
                stimulus=str(obj["stimulus"]),
                concept=str(obj["concept"]), template_id=str(obj["templateID"]),
                id=obj.get("id"), topic=obj.get("topic"), split=split))
            continue
        _require_row_keys(obj, ("positiveStimulus", "negativeStimulus",
                                "concept", "templateID"), source)
        pairs.append(ReaderPair(
            positive_stimulus=str(obj["positiveStimulus"]),
            negative_stimulus=str(obj["negativeStimulus"]),
            concept=str(obj["concept"]), template_id=str(obj["templateID"]),
            id=obj.get("id"), topic=obj.get("topic"), split=split))
    if not pairs:
        raise RepeReaderError(f"empty reader pairs file: {source}")
    concepts = {p.concept for p in pairs}
    if len(concepts) > 1:
        raise RepeReaderError(
            f"{source}: mixed concepts {sorted(concepts)} — one reader dataset "
            "per concept")
    if len({p.is_template_pair_row for p in pairs}) > 1:
        raise RepeReaderError(
            f"{source}: mixes content-pair rows (positiveStimulus/negativeStimulus) "
            "with template-pair rows (stimulus) — the two produce different "
            "differences, so one file cannot mean both. Repair: split them into two "
            "datasets")
    dataset = ReaderDataset(concept=pairs[0].concept, pairs=tuple(pairs),
                            hash=_sha256_hex(data))
    check_split_overlap(dataset, source=source)
    return dataset


def load_pairs(path: str) -> ReaderDataset:
    """Load ``pairs.jsonl`` from disk (delegates to :func:`parse_pairs`)."""
    if not os.path.exists(path):
        raise RepeReaderError(f"missing reader pairs file: {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    return parse_pairs(data, source=path)


def resolve_contrast_mode(dataset: ReaderDataset, template: TaskTemplate) -> str:
    """Which contrast a (dataset, template) combination declares — derived,
    never passed loosely, so the fit cannot be asked for a construction its
    inputs do not support. Swift twin: ``RepEReader.resolveContrastMode``."""
    single = dataset.shape == "singleStimulus"
    if not single and not template.is_template_pair:
        return SUPERVISED_CONTENT
    if single and template.is_template_pair:
        return UNSUPERVISED_TEMPLATE_PAIR
    if template.is_template_pair:
        raise RepeReaderError(
            f"template {template.id!r} declares a T+/T− instructionPair but the "
            "dataset holds content pairs (positiveStimulus/negativeStimulus) — under "
            "a template pair the contrast is the INSTRUCTION and a second stimulus "
            "would be a confound. Repair: fit these pairs through a single-template "
            "reader template, or rewrite the dataset as one-stimulus ('stimulus') "
            "rows")
    raise RepeReaderError(
        "the dataset holds one-stimulus ('stimulus') rows but template "
        f"{template.id!r} declares no instructionPair — there is nothing to contrast "
        "the stimulus against. Repair: choose a template-pair template (one with "
        "'instructionPair'), or rewrite the dataset as positiveStimulus/"
        "negativeStimulus content pairs")


# --- evidence roles ----------------------------------------------------------

def derive_evidence_roles(*, sign_convention: str,
                          layer_recommendation_basis: str | None,
                          held_out_accuracy: float | None,
                          final_test_accuracy: float | None,
                          train_pair_count: int, held_out_pair_count: int,
                          final_test_pair_count: int) -> dict:
    """WHICH DECISIONS each accuracy's rows were allowed to make.

    The finding this answers (2026-09-05): the held-out split does double duty
    here. It fixes the direction's sign (:func:`fit_direction`, the paper's
    step 4) AND it ranks the layers (:func:`stamp_layer_recommendation`), so
    ``heldOutAccuracy`` is the score of the winner of two selections made on
    those very rows. Reporting it as an out-of-sample estimate overstates it,
    and until now nothing on the artifact said so.

    The rule: ``heldOutAccuracy`` is ``selection`` whenever the held-out rows
    signed the direction OR ranked the layers, and ``validation`` only when
    they did neither (a fit whose sign fell back to train-label majority and
    whose recommendation had no held-out basis). It is omitted entirely when
    there is no held-out accuracy to label.

    ``finalTestAccuracy`` is the only ``finalEvaluation`` number, and it exists only
    when the dataset reserved ``split: "finalTest"`` rows.

    Swift twin owed: ``RepEReader`` stamps no evidence roles yet.
    """
    sign_by = BY_HELD_OUT if sign_convention == HELD_OUT_PAIR_AGREEMENT else BY_TRAIN
    if layer_recommendation_basis == "heldOutAccuracy":
        layer_by = BY_HELD_OUT
    elif layer_recommendation_basis == "trainAccuracy":
        layer_by = BY_TRAIN
    else:
        layer_by = None
    roles: dict = {"trainAccuracy": FIT_ROLE}
    if held_out_accuracy is not None:
        roles["heldOutAccuracy"] = (
            SELECTION_ROLE if (sign_by == BY_HELD_OUT or layer_by == BY_HELD_OUT)
            else VALIDATION_ROLE)
    if final_test_accuracy is not None:
        roles["finalTestAccuracy"] = FINAL_EVALUATION_ROLE
    roles["signSelectedBy"] = sign_by
    roles["layerRecommendedBy"] = layer_by
    roles["splitCounts"] = {"train": int(train_pair_count),
                            "heldOut": int(held_out_pair_count),
                            "finalTest": int(final_test_pair_count)}
    return roles


def evidence_roles_from_json(d: dict) -> tuple[dict, str]:
    """``(roles, basis)`` for a raw artifact dict.

    A stamped block is returned VERBATIM — including a wrong one, which
    :func:`validate_evidence_roles` is there to catch on the way out rather
    than silently repairing on the way in. A block-less artifact (schema 1, and
    every schema-2 artifact fitted before 2026-09-05) has its roles derived
    from the stamps it does carry, and the basis says so.
    """
    stamped = d.get("evidenceRoles")
    if isinstance(stamped, dict):
        return dict(stamped), str(d.get("evidenceRolesBasis")
                                  or EVIDENCE_ROLES_STAMPED)
    return (derive_evidence_roles(
        sign_convention=str(d.get("signConvention") or TRAIN_MAJORITY),
        layer_recommendation_basis=d.get("layerRecommendationBasis"),
        held_out_accuracy=d.get("heldOutAccuracy"),
        final_test_accuracy=d.get("finalTestAccuracy"),
        train_pair_count=int(d.get("trainPairCount") or 0),
        held_out_pair_count=int(d.get("heldOutPairCount") or 0),
        final_test_pair_count=int(d.get("finalTestPairCount") or 0)),
        EVIDENCE_ROLES_DERIVED)


def validate_evidence_roles(roles: dict, *,
                            source: str = "reader artifact") -> None:
    """Refuse a roles block that promotes a selection statistic to a final
    test, or demotes the final test to something else.

    Serialization runs this, so a hand-written fixture (or a future decoder
    bug) cannot put a relabelled score into a file that looks like every other
    reader artifact. Two labels are load-bearing and both are checked."""
    if roles.get("heldOutAccuracy") == FINAL_EVALUATION_ROLE:
        raise RepeReaderError(
            f"{source} labels heldOutAccuracy {FINAL_EVALUATION_ROLE!r}: those "
            "rows chose the direction's sign and ranked the layers, so the "
            "label would present a selection statistic as a final-test result. "
            f"Repair: label heldOutAccuracy {SELECTION_ROLE!r} or "
            f"{VALIDATION_ROLE!r}, and report final-test evidence as "
            "finalTestAccuracy over rows marked split 'finalTest'")
    if "finalTestAccuracy" in roles and roles["finalTestAccuracy"] != FINAL_EVALUATION_ROLE:
        raise RepeReaderError(
            f"{source} labels finalTestAccuracy {roles['finalTestAccuracy']!r}: "
            "finalTestAccuracy is the ONLY final-evaluation statistic a reader has — "
            "rows marked split 'finalTest' are read by no fitting or selection step, "
            "which is the whole of what makes them one. Repair: label "
            f"finalTestAccuracy {FINAL_EVALUATION_ROLE!r}, or drop it if the dataset "
            "reserved no split 'finalTest' rows")


# --- reader artifact ---------------------------------------------------------

@dataclass
class ReaderArtifact:
    """The fitted instrument (REPE-IMPLEMENTATION-BRIEF §4): one per concept × layer × template ×
    model × substrate. The full template record is embedded so inference is
    standalone and drift-proof; ``templateID``/``templateHash`` remain the
    registry pins.

    **Schema growth rule.** Every field added after schema 1 decodes with an
    explicit LEGACY default, named here, so a reader artifact written by an
    older engine keeps loading and keeps meaning what it meant:
    ``contrastMode`` absent = ``supervisedContent``, ``signConvention`` absent
    = ``trainMajority``, ``extractionRendering`` absent = raw,
    ``pc1ExplainedVarianceOfDifferences`` absent = the legacy
    ``pc1ExplainedVariance`` under basis ``alternatedRows``,
    ``pc1PowerIteration`` absent = fitted before the convergence diagnostic
    existed (2026-08-28), NOT "converged", ``finalTestAccuracy``/``finalTestPairCount``
    absent = the dataset reserved no ``split: "finalTest"`` rows (and, on an
    artifact fitted before 2026-09-05, that the split role did not yet exist),
    ``evidenceRoles`` absent = derive them from ``signConvention`` and
    ``layerRecommendationBasis`` at decode, ``splitOverlap`` absent = the
    cross-split leakage check had not been written when this was fitted, NOT
    "no overlap".

    Schema version stays **2**: every key here is additive and absent-means-
    legacy, and the Swift decoder reads the keys it knows and ignores the rest,
    so a version bump would buy nothing and break the twin's pin.
    """

    model_id: str
    revision: str | None
    concept: str
    layer: int
    template: TaskTemplate
    dataset_hash: str
    probe: ScalarProbe
    #: Share of the DIFFERENCE CLOUD's variance that PC1 explains.
    #:
    #: Before 2026-08-27 this number was computed over the ALTERNATED rows —
    #: the ± symmetrized copies PCA is actually fitted on — where the cloud's
    #: mean is near zero by construction and the ratio is systematically
    #: flattering. A reader reasonably assumes it describes the differences
    #: themselves, so it now does. ``explained_variance_basis`` says which of
    #: the two a given artifact carries.
    #: None when the cloud has no variance to apportion — see
    #: ``explained_variance_basis == "degenerateDifferenceCloud"``.
    difference_cloud_explained_variance: float | None
    train_accuracy: float
    held_out_accuracy: float | None
    train_pair_count: int
    held_out_pair_count: int
    #: ``"differenceCloud"`` (fitted here), ``"degenerateDifferenceCloud"``
    #: (fitted here, but every difference was identical so there is no variance
    #: to apportion and the value is absent), or ``"alternatedRows"`` (decoded
    #: from a pre-2026-08-27 artifact's ``pc1ExplainedVariance``, which
    #: measured the ± symmetrized copies).
    explained_variance_basis: str = "differenceCloud"
    #: Convergence health of the Gram power iteration that produced PC1
    #: (2026-08-28 audit, F5). Absent on every artifact written before
    #: 2026-08-28 — and absent is not "converged": it means the fit predates
    #: the diagnostic and the question was never asked.
    pc1_power_iteration: vm.PowerIterationDiagnostic | None = None
    contrast_mode: str = SUPERVISED_CONTENT
    sign_convention: str = TRAIN_MAJORITY
    #: Held-out paired-discrimination accuracy of the CHOSEN sign, when the
    #: held-out split fixed it. None under ``trainMajority``.
    sign_held_out_accuracy: float | None = None
    #: Why the held-out sign rule stood down, when it did.
    sign_fallback_reason: str | None = None
    #: The seed the unsupervised orientation draw used. None under
    #: ``supervisedContent`` (whose alternation needs no RNG).
    orientation_seed: int | None = None
    #: The layer with the highest held-out accuracy in the SET this artifact
    #: was fitted with — a RECOMMENDATION, never a selection.
    recommended_layer: int | None = None
    recommended_layer_accuracy: float | None = None
    #: ``"heldOutAccuracy"``, or ``"trainAccuracy"`` when the fit had no
    #: held-out pairs to recommend from.
    layer_recommendation_basis: str | None = None
    #: FINAL-TEST accuracy: the probe's accuracy over rows marked
    #: ``split: "finalTest"``, which no fitting or selection step read. None when
    #: the dataset reserved none — and then this reader has no final-test
    #: evidence at all, only selection/validation evidence.
    final_test_accuracy: float | None = None
    final_test_pair_count: int = 0
    #: The cross-split leakage check's result (:func:`check_split_overlap`).
    #: ``{"exactDuplicatesAcrossSplits": 0}`` — the only value that can be
    #: stamped, because a nonzero count refuses the fit. Absent = fitted before
    #: the check existed, which is not the same as "checked and clean".
    split_overlap: dict | None = None
    #: The stamped ``evidenceRoles`` block, or None to derive it from this
    #: artifact's own stamps at serialization. A block decoded from a file is
    #: kept VERBATIM so a mislabelled one is refused rather than corrected.
    evidence_roles: dict | None = None
    evidence_roles_basis: str = EVIDENCE_ROLES_STAMPED
    substrate: str = SUBSTRATE
    #: HOW the scaffold reached the model. None = legacy raw, which is why a
    #: raw fit encodes byte-identically to a pre-2026-08-27 one.
    extraction_rendering: dict | None = None
    rendering_convention: str = READER_RENDERING_CONVENTION
    extraction_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def lat_token_position(self) -> str:
        return self.template.lat_token

    @property
    def reading_position(self) -> ReadingPosition:
        return self.template.reading_position

    @property
    def resolved_evidence_roles(self) -> dict:
        """The roles block this artifact stands behind: the stamped one when it
        has one, else derived from its own stamps. Consumers read this, never
        the raw field, so a reader fitted before the block existed still says
        which split chose its sign."""
        if self.evidence_roles is not None:
            return self.evidence_roles
        return derive_evidence_roles(
            sign_convention=self.sign_convention,
            layer_recommendation_basis=self.layer_recommendation_basis,
            held_out_accuracy=self.held_out_accuracy,
            final_test_accuracy=self.final_test_accuracy,
            train_pair_count=self.train_pair_count,
            held_out_pair_count=self.held_out_pair_count,
            final_test_pair_count=self.final_test_pair_count)

    def refresh_evidence_roles(self) -> None:
        """Re-derive and STAMP the roles from the current stamps.

        Called by :func:`stamp_layer_recommendation`, because the layer
        recommendation is decided over the whole fitted set AFTER each
        artifact exists: until it lands, a fit whose sign fell back to train
        majority looks like one whose held-out rows decided nothing, and its
        held-out accuracy would be labelled ``validation`` when the layer
        ranking is about to make it ``selection``.
        """
        self.evidence_roles = self.resolved_evidence_roles
        self.evidence_roles_basis = EVIDENCE_ROLES_STAMPED

    @property
    def resolved_extraction_rendering(self) -> ExtractionRendering:
        """The rendering actually applied — absent resolves to legacy raw."""
        if not self.extraction_rendering:
            return RAW_RENDERING
        return _rendering_from_json(self.extraction_rendering)

    def to_dict(self) -> dict:
        # Refuse on the way OUT: a relabelled block reaches serialization from
        # a fixture or a decoder, never from `derive_evidence_roles`, and the
        # file is where the mislabelling would do its damage.
        roles = self.resolved_evidence_roles
        validate_evidence_roles(roles)
        out = {
            "artifactType": ARTIFACT_TYPE,
            "schemaVersion": 2,
            "modelID": self.model_id,
            "revision": self.revision,
            "substrate": self.substrate,
            "concept": self.concept,
            "layer": self.layer,
            "templateID": self.template.id,
            "templateHash": self.template.hash,
            "template": self.template.to_dict(),
            "datasetHash": self.dataset_hash,
            "latTokenPosition": self.lat_token_position,
            "readingPosition": self.reading_position.label,
            "probe": self.probe.to_dict(),
            # The legacy `pc1ExplainedVariance` key is deliberately NOT
            # written: the number's basis changed, and writing the old key
            # with the new semantics would make every pre-existing consumer
            # silently wrong instead of visibly out of date.
            "pc1ExplainedVarianceOfDifferences":
                self.difference_cloud_explained_variance,
            # Always written, even when the value above is absent: it is what
            # tells a decoder that a missing number is a degenerate cloud
            # rather than a truncated artifact.
            "pc1ExplainedVarianceBasis": self.explained_variance_basis,
            # Convergence health of the iteration that produced PC1 (F5).
            # Additive and OUTSIDE any identity: `recipe_identity` reads a
            # closed list of sidecar keys and no reader field is in it, so this
            # cannot move a recipeIdentityHash. It does change the bytes — and
            # so the readerHash — of NEWLY fitted readers, which is correct:
            # they carry information older ones do not.
            "pc1PowerIteration": (None if self.pc1_power_iteration is None
                                  else self.pc1_power_iteration.to_dict()),
            "trainAccuracy": self.train_accuracy,
            "heldOutAccuracy": self.held_out_accuracy,
            "trainPairCount": self.train_pair_count,
            "heldOutPairCount": self.held_out_pair_count,
            # Final-test evidence. Both absent when the dataset reserved no
            # 'finalTest' rows — absent means "none were reserved", which is
            # exactly what the note tells a reader to do about it.
            "finalTestAccuracy": self.final_test_accuracy,
            "finalTestPairCount": (None if self.final_test_pair_count == 0
                              else self.final_test_pair_count),
            "splitOverlap": self.split_overlap,
            # WHICH DECISIONS each accuracy's rows were allowed to make, and
            # whether this artifact recorded that or we reconstructed it.
            # Additive and outside every identity, exactly like
            # pc1PowerIteration: no reader field is in `recipe_identity`'s
            # closed sidecar list, so this cannot move a recipeIdentityHash. It
            # does change the bytes — and so the readerHash — of NEWLY fitted
            # readers, which is correct: they carry information older ones do
            # not.
            "evidenceRoles": roles,
            "evidenceRolesBasis": self.evidence_roles_basis,
            "evidenceRoleNote": EVIDENCE_ROLE_NOTE,
            "contrastMode": self.contrast_mode,
            "signConvention": self.sign_convention,
            "signHeldOutAccuracy": self.sign_held_out_accuracy,
            "signFallbackReason": self.sign_fallback_reason,
            "orientationSeed": self.orientation_seed,
            "recommendedLayer": self.recommended_layer,
            "recommendedLayerAccuracy": self.recommended_layer_accuracy,
            "layerRecommendationBasis": self.layer_recommendation_basis,
            "extractionRendering": self.extraction_rendering,
            "renderingConvention": self.rendering_convention,
            "extractionDate": self.extraction_date,
        }
        if self.recommended_layer is not None:
            out["layerRecommendationNote"] = LAYER_RECOMMENDATION_NOTE
        if self.template.divergence is not None:
            out["templateDivergence"] = self.template.divergence
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "ReaderArtifact":
        if d.get("artifactType") != ARTIFACT_TYPE:
            raise RepeReaderError(
                f"not a {ARTIFACT_TYPE} artifact (artifactType="
                f"{d.get('artifactType')!r})")
        template = TaskTemplate.from_dict(d["template"],
                                          hash=d.get("templateHash"))
        if "pc1ExplainedVarianceOfDifferences" in d:
            explained = float(d["pc1ExplainedVarianceOfDifferences"])
            basis = str(d.get("pc1ExplainedVarianceBasis", "differenceCloud"))
        elif "pc1ExplainedVariance" in d:
            # A pre-2026-08-27 artifact: the number is over the ALTERNATED
            # rows, and the basis stamp says so rather than relabelling it.
            explained = float(d["pc1ExplainedVariance"])
            basis = "alternatedRows"
        elif "pc1ExplainedVarianceBasis" in d:
            # A degenerate difference cloud: the basis is stamped and the
            # number is deliberately absent, because zero variance has no share
            # to apportion and writing 0 would read as "PC1 explains nothing" —
            # the opposite of what a one-point cloud means.
            explained = None
            basis = str(d["pc1ExplainedVarianceBasis"])
        else:
            raise RepeReaderError(
                "reader artifact carries neither 'pc1ExplainedVarianceOfDifferences', "
                "its 'pc1ExplainedVarianceBasis', nor the legacy "
                "'pc1ExplainedVariance' — one of them is required")
        roles, roles_basis = evidence_roles_from_json(d)
        return cls(
            model_id=str(d["modelID"]), revision=d.get("revision"),
            concept=str(d["concept"]), layer=int(d["layer"]), template=template,
            dataset_hash=str(d["datasetHash"]),
            probe=ScalarProbe.from_dict(d["probe"]),
            difference_cloud_explained_variance=explained,
            explained_variance_basis=basis,
            pc1_power_iteration=(
                None if d.get("pc1PowerIteration") is None
                else vm.PowerIterationDiagnostic.from_dict(d["pc1PowerIteration"])),
            train_accuracy=float(d["trainAccuracy"]),
            held_out_accuracy=(None if d.get("heldOutAccuracy") is None
                               else float(d["heldOutAccuracy"])),
            train_pair_count=int(d.get("trainPairCount", 0)),
            held_out_pair_count=int(d.get("heldOutPairCount", 0)),
            contrast_mode=str(d.get("contrastMode") or SUPERVISED_CONTENT),
            sign_convention=str(d.get("signConvention") or TRAIN_MAJORITY),
            sign_held_out_accuracy=(None if d.get("signHeldOutAccuracy") is None
                                    else float(d["signHeldOutAccuracy"])),
            sign_fallback_reason=d.get("signFallbackReason"),
            orientation_seed=(None if d.get("orientationSeed") is None
                              else int(d["orientationSeed"])),
            recommended_layer=(None if d.get("recommendedLayer") is None
                               else int(d["recommendedLayer"])),
            recommended_layer_accuracy=(
                None if d.get("recommendedLayerAccuracy") is None
                else float(d["recommendedLayerAccuracy"])),
            layer_recommendation_basis=d.get("layerRecommendationBasis"),
            final_test_accuracy=(None if d.get("finalTestAccuracy") is None
                           else float(d["finalTestAccuracy"])),
            final_test_pair_count=int(d.get("finalTestPairCount") or 0),
            split_overlap=(d.get("splitOverlap")
                           if isinstance(d.get("splitOverlap"), dict) else None),
            evidence_roles=roles, evidence_roles_basis=roles_basis,
            substrate=str(d.get("substrate", "")),
            extraction_rendering=(d.get("extractionRendering")
                                  if isinstance(d.get("extractionRendering"), dict)
                                  else None),
            rendering_convention=str(d.get("renderingConvention", "")),
            extraction_date=str(d.get("extractionDate", "")))


def save_reader(artifact: ReaderArtifact, directory: str,
                name: str | None = None) -> str:
    """Write ``<name>.json`` into a run directory; returns the file path."""
    os.makedirs(directory, exist_ok=True)
    name = name or f"reader-{artifact.concept}-layer{artifact.layer}"
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, indent=2, sort_keys=True)
    return path


def save_readers(artifacts: list[ReaderArtifact], directory: str) -> list[str]:
    return [save_reader(a, directory) for a in artifacts]


def load_reader(path: str) -> ReaderArtifact:
    if not os.path.exists(path):
        raise RepeReaderError(f"missing reader artifact: {path}")
    with open(path, encoding="utf-8") as handle:
        try:
            return ReaderArtifact.from_dict(json.load(handle))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RepeReaderError(f"{path}: unreadable reader artifact: {exc}") from exc


# --- fit ----------------------------------------------------------------------

@dataclass
class FittedDirection:
    """One layer's direction, with the provenance of how its sign was fixed."""

    component: list[float]
    difference_cloud_explained_variance: float | None
    sign_convention: str
    sign_held_out_accuracy: float | None
    sign_fallback_reason: str | None
    #: Convergence health of the Gram power iteration that produced PC1
    #: (2026-08-28 audit, F5). None only when the PCA path returned no
    #: diagnostic at all, which no current path does.
    power_iteration: vm.PowerIterationDiagnostic | None = None


def held_out_sign_fallback_reason(*, held_out_pair_count: int, decided: int,
                                  agree: int, disagree: int) -> str:
    """Why the held-out sign rule stood down — stamped into the artifact, so a
    fit that fell back cannot be mistaken for one that did not. Swift twin:
    ``RepEReader.heldOutSignFallbackReason``.

    A dataset with ``finalTest`` rows and no held-out rows takes the ordinary
    "no held-out pairs" branch: final-test rows are read by no selection step,
    so they cannot sign the direction, and the repair is the same — mark some
    OTHER rows held out.
    """
    if held_out_pair_count == 0:
        return ("no held-out pairs (every row is split 'train'): the sign follows "
                "train-label majority, the reference implementation's get_signs. "
                "Repair: mark some rows with a non-'train' split so the paper's "
                "held-out sign selection can run")
    if decided < MINIMUM_HELD_OUT_PAIRS_FOR_SIGN_SELECTION:
        return (f"{decided} held-out pair(s) projected off zero, below the minimum "
                f"{MINIMUM_HELD_OUT_PAIRS_FOR_SIGN_SELECTION}: a one-pair vote is a "
                "coin flip wearing a validation split's authority, so the sign "
                "follows train-label majority instead")
    return (f"held-out pairs split evenly ({agree} for, {disagree} against): the "
            "held-out set does not discriminate at this layer, so the sign follows "
            "train-label majority. Read this layer's heldOutAccuracy before trusting "
            "its direction")


def explained_variance_of_difference_cloud(differences: list,
                                           component) -> float | None:
    """PC1's share of the DIFFERENCE CLOUD's variance: the centered
    differences' projection energy over their total energy.

    Computed on the differences themselves, NOT on the ± alternated copies PCA
    is fitted on — the alternated cloud is centered near zero by construction,
    which inflates the ratio into a number that means less than a reader
    assumes. Swift twin:
    ``RepEReader.explainedVarianceOfDifferenceCloud``.
    """
    a = np.asarray(differences, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] == 0:
        return None
    centered = (a - a.mean(axis=0)).astype(np.float32)
    total = float(np.square(centered).sum(dtype=np.float32))
    if total <= 0:
        # No variance to apportion (every difference identical). None, not 0:
        # 0 would read as "PC1 explains nothing", the opposite of the truth.
        return None
    projections = centered @ np.asarray(component, dtype=np.float32)
    captured = float(np.square(projections).sum(dtype=np.float32))
    return captured / total


def fit_direction(pos_train: list[list[float]], neg_train: list[list[float]],
                  pos_held: list[list[float]], neg_held: list[list[float]],
                  *, contrast_mode: str,
                  orientation_seed: int) -> FittedDirection:
    """PC1 of the paired differences, signed by the HELD-OUT split when it can
    be (paper step 4) and by train-label majority when it cannot.

    Construction by contrast mode:

    - ``supervisedContent``: each difference is L2-normalized before PCA (OUR
      departure, not the paper's — see :func:`vector_math.direction`), then
      enters in alternating ± orientation. Alternation matters: once magnitudes
      are normalized away, labeled differences all point the same way, and
      centering would subtract the shared concept direction out of the data
      entirely.
    - ``unsupervisedTemplatePair``: the reference implementation's own
      construction — NO normalization, per-row random orientation
      (``random.shuffle(d)`` then ``[::2] − [1::2]``), mean-centering, PCA with
      ``n_components=1``. The orientation draw is seeded and the seed is
      stamped, which is the one place we improve on the reference: its shuffle
      is unseeded, so its direction cannot be reproduced.

    Why the sign cannot come from the probe's own accuracy: :func:`scalar_probe`
    derives ``orientation`` from the train class means, so flipping the
    direction flips the orientation too and leaves every score identical. Sign
    selection therefore scores the PAIRED discrimination — does a held-out
    (positive − negative) difference project positive? — which is exactly what
    ``get_signs`` asks of the train split, asked of held-out data instead.

    **Final-test rows never reach this function.** ``pos_held``/``neg_held``
    are the SELECTION rows only; rows marked ``split: "finalTest"`` are not passed,
    which is the whole of what makes their accuracy a final-test estimate.

    Swift twin: ``RepEReader.fitDirection``.
    """
    pos = np.asarray(pos_train, dtype=np.float32)
    neg = np.asarray(neg_train, dtype=np.float32)
    if pos.shape != neg.shape:
        raise RepeReaderError(
            f"unpaired activations: positive {pos.shape} vs negative {neg.shape}")
    diffs: list[np.ndarray] = []
    for d in (pos - neg).astype(np.float32):
        norm = vm.l2_norm(d.tolist())
        if norm <= 0:
            continue
        if contrast_mode == SUPERVISED_CONTENT:
            diffs.append((d / np.float32(norm)).astype(np.float32))
        else:
            diffs.append(d)
    if len(diffs) < 2:
        raise RepeReaderError("need at least 2 non-degenerate train pairs")

    if contrast_mode == SUPERVISED_CONTENT:
        oriented = [d if i % 2 == 0 else -d for i, d in enumerate(diffs)]
    else:
        signs = orientation_signs(len(diffs), orientation_seed)
        oriented = [d if s > 0 else -d for d, s in zip(diffs, signs)]

    result = vm.principal_components_with_variance(
        [o.tolist() for o in oriented], 1)
    if not result.components:
        raise RepeReaderError("degenerate pair differences (no PC1)")
    pc = result.components[0]
    explained = explained_variance_of_difference_cloud(
        [d.tolist() for d in diffs], pc)

    # --- sign: held-out first (paper step 4), train majority as fallback
    held_diffs = [(np.asarray(p, dtype=np.float32) - np.asarray(n, dtype=np.float32))
                  for p, n in zip(pos_held, neg_held)]
    held_scores = [vm.dot(d.tolist(), pc) for d in held_diffs]
    agree = sum(1 for s in held_scores if s > 0)
    disagree = sum(1 for s in held_scores if s < 0)
    decided = agree + disagree

    convention = HELD_OUT_PAIR_AGREEMENT
    sign_held_out_accuracy: float | None = None
    fallback_reason: str | None = None

    if decided >= MINIMUM_HELD_OUT_PAIRS_FOR_SIGN_SELECTION and agree != disagree:
        flip = disagree > agree
        sign_held_out_accuracy = max(agree, disagree) / decided
    else:
        convention = TRAIN_MAJORITY
        fallback_reason = held_out_sign_fallback_reason(
            held_out_pair_count=len(held_diffs), decided=decided,
            agree=agree, disagree=disagree)
        scores = [vm.dot(d.tolist(), pc) for d in diffs]
        positive_scores = sum(1 for s in scores if s > 0)
        n = len(scores)
        if positive_scores * 2 == n:
            flip = vm.dot(pc, vm.mean_difference(pos_train, neg_train)) < 0
        else:
            flip = positive_scores * 2 < n
    # Flipping the component does not change the held-out accuracy just
    # recorded: it IS the accuracy of the chosen sign, the larger of the two.
    if flip:
        pc = [-x for x in pc]
    return FittedDirection(
        component=pc, difference_cloud_explained_variance=explained,
        sign_convention=convention, sign_held_out_accuracy=sign_held_out_accuracy,
        sign_fallback_reason=fallback_reason,
        # Sign-invariant by construction (‖G(−w) − λ(−w)‖ = ‖Gw − λw‖), so the
        # flip above cannot move it.
        power_iteration=(result.diagnostics[0] if result.diagnostics else None))


def _pair_accuracy(probe: ScalarProbe, positive: list[list[float]],
                   negative: list[list[float]]) -> float | None:
    total = len(positive) + len(negative)
    if total == 0:
        return None
    correct = sum(1 for row in positive if probe.classifies_positive(row))
    correct += sum(1 for row in negative if not probe.classifies_positive(row))
    return correct / total


def stamp_layer_recommendation(artifacts: list[ReaderArtifact]) -> list[ReaderArtifact]:
    """Stamp the argmax-held-out-accuracy layer into every artifact of a set
    (the paper's step 4, layer half). A RECOMMENDATION: nothing downstream may
    read it as a selection, because which layer a study reads is a declarable
    choice recorded in its manifest. Swift twin:
    ``RepEReader.stampLayerRecommendation``.

    This is also where ``evidenceRoles`` is stamped, and it has to be: ranking
    the layers by ``heldOutAccuracy`` is the SECOND selection those rows make,
    so the role of ``heldOutAccuracy`` is not knowable until this function has
    run over the whole set."""
    if len(artifacts) <= 1:
        # One layer is never "recommended" over anything, but its roles are
        # still owed.
        for artifact in artifacts:
            artifact.refresh_evidence_roles()
        return artifacts
    basis = ("heldOutAccuracy"
             if any(a.held_out_accuracy is not None for a in artifacts)
             else "trainAccuracy")

    def score(a: ReaderArtifact) -> float:
        if basis == "heldOutAccuracy":
            return -1.0 if a.held_out_accuracy is None else a.held_out_accuracy
        return a.train_accuracy

    # Ties go to the LOWER layer index: an arbitrary tiebreak, but a stated and
    # stable one, so two fits of the same data recommend the same layer on both
    # engines.
    best = min(artifacts, key=lambda a: (-score(a), a.layer))
    for artifact in artifacts:
        artifact.recommended_layer = best.layer
        artifact.recommended_layer_accuracy = score(best)
        artifact.layer_recommendation_basis = basis
        artifact.refresh_evidence_roles()
    return artifacts


def fit_texts(dataset: ReaderDataset, template: TaskTemplate, *, model_id: str,
              rendering: ExtractionRendering | None = None) -> list[str]:
    """The rendered texts a fit reads, in the order :func:`fit_activations`
    expects: train rows, then held-out rows, then FINAL-TEST rows, positive/T+
    before negative/T− within each row. Pure (no model), so the rendering
    contract is testable.

    Test rows come last so that a dataset without any renders exactly the text
    list this function has always produced — the ordering is a schema, and
    appending is the only change to it that cannot move an existing fit."""
    contrast_mode = resolve_contrast_mode(dataset, template)
    texts: list[str] = []
    for pair in dataset.train + dataset.held_out + dataset.final_test:
        if contrast_mode == SUPERVISED_CONTENT:
            for stimulus in (pair.positive_stimulus, pair.negative_stimulus):
                texts.append(render_scaffold(
                    template, stimulus=stimulus, concept=dataset.concept,
                    model_id=model_id, rendering=rendering))
            continue
        if pair.stimulus is None or template.instruction_pair is None:
            raise RepeReaderError(
                f"row {pair.id!r} carries no 'stimulus' for a template-pair fit")
        for instruction in (template.instruction_pair.experimental,
                            template.instruction_pair.reference):
            texts.append(render_scaffold(
                template, stimulus=pair.stimulus, concept=dataset.concept,
                model_id=model_id, instruction=instruction, rendering=rendering))
    return texts


def validate_fit_renders(dataset: ReaderDataset, template: TaskTemplate, *,
                         model_id: str,
                         rendering: ExtractionRendering | None = None) -> None:
    """Everything the fit needs from a template, checked WITHOUT a model.

    The LAT-token vocabulary, the ``{{stimulus}}``/``{{concept}}``/
    ``{{instruction}}`` slot rules, the contrast-mode resolution, and the
    scaffold's marker hygiene — every one of them, over every row of the real
    dataset, exactly as :func:`fit` will exercise them. It is the same code
    path (:func:`fit_texts`), not a re-statement of it, which is the only kind
    of pre-flight worth trusting.

    Why it exists (review round 6, finding 4): the reader-fit route replaced
    the canonical pairs corpus and queued a job BEFORE anything had tried to
    render the template. A template that parsed as JSON but could not render
    took the good corpus with it and reported the failure minutes later, from
    a job. Callers run this first, and every failure — including the
    ``ValueError`` the scaffold pass raises — arrives as one typed
    :class:`RepeReaderError`.
    """
    try:
        template.reading_position  # noqa: B018 — the latToken vocabulary check
        fit_texts(dataset, template, model_id=model_id, rendering=rendering)
    except RepeReaderError:
        raise
    except ValueError as exc:
        raise RepeReaderError(str(exc)) from exc


def fit_activations(dataset: ReaderDataset, template: TaskTemplate,
                    captured: list[list[list[float]]], *, model_id: str,
                    revision: str | None, layers: list[int] | None = None,
                    orientation_seed: int = DEFAULT_ORIENTATION_SEED,
                    extraction_rendering: ExtractionRendering | None = None
                    ) -> list[ReaderArtifact]:
    """Pure fit over pre-captured LAT-token activations — the unit-testable
    math half. ``captured[text_index][layer]`` follows the rendered-text order
    :func:`fit_texts` produces. Swift twin: ``RepEReader.fit(dataset:…)``."""
    for pair in dataset.pairs:
        if pair.template_id != template.id:
            raise RepeReaderError(
                f"pair {pair.id!r} pins template {pair.template_id!r} but the fit "
                f"uses {template.id!r}")
    contrast_mode = resolve_contrast_mode(dataset, template)
    # A hand-built dataset (one not routed through `parse_pairs`) reaches the
    # fit here, so the leakage gate runs here too rather than trusting the
    # loader to have been used.
    split_overlap = check_split_overlap(dataset, source="reader dataset")
    train = dataset.train
    held = dataset.held_out
    test = dataset.final_test
    if len(train) < 2:
        raise RepeReaderError(
            f"need at least 2 train pairs, have {len(train)} "
            "(rows default to split 'train')")
    if len(captured) != 2 * (len(train) + len(held) + len(test)):
        raise RepeReaderError(
            f"captured {len(captured)} activations for "
            f"{len(train) + len(held) + len(test)} pairs — expected two per "
            "pair")
    layer_count = len(captured[0]) if captured else 0
    if layer_count == 0:
        raise RepeReaderError("no layers captured")
    chosen = list(range(layer_count)) if layers is None else list(layers)

    n_train = len(train)
    rendering_block = (None if extraction_rendering is None
                       or extraction_rendering.is_raw
                       else extraction_rendering.to_dict())
    artifacts: list[ReaderArtifact] = []
    for layer in chosen:
        if not 0 <= layer < layer_count:
            raise RepeReaderError(f"layer {layer} out of range 0..{layer_count - 1}")
        pos_train = [captured[2 * i][layer] for i in range(n_train)]
        neg_train = [captured[2 * i + 1][layer] for i in range(n_train)]
        pos_held = [captured[2 * (n_train + i)][layer] for i in range(len(held))]
        neg_held = [captured[2 * (n_train + i) + 1][layer] for i in range(len(held))]
        # The final-test rows, kept in their own names from here down so that
        # every selection step below can be read to not touch them.
        n_selected = n_train + len(held)
        pos_test = [captured[2 * (n_selected + i)][layer] for i in range(len(test))]
        neg_test = [captured[2 * (n_selected + i) + 1][layer]
                    for i in range(len(test))]

        fitted = fit_direction(pos_train, neg_train, pos_held, neg_held,
                               contrast_mode=contrast_mode,
                               orientation_seed=orientation_seed)
        # Training normalization: center at the train activation mean, score
        # scale/center from the train projections — the "fit params" the paper's
        # inference reuses on new text.
        center = vm.mean(pos_train + neg_train)
        probe = vm.scalar_probe(fitted.component, pos_train, neg_train,
                                activation_center=center)
        train_accuracy = _pair_accuracy(probe, pos_train, neg_train)
        held_accuracy = _pair_accuracy(probe, pos_held, neg_held)
        # Scored LAST, through the finished instrument: the direction, its
        # sign, the probe's center and scale are all fixed by now, so this
        # number is the only one on the artifact that no decision read.
        final_test_accuracy = _pair_accuracy(probe, pos_test, neg_test)
        artifacts.append(ReaderArtifact(
            model_id=model_id, revision=revision,
            concept=dataset.concept, layer=layer, template=template,
            dataset_hash=dataset.hash, probe=probe,
            difference_cloud_explained_variance=(
                fitted.difference_cloud_explained_variance),
            explained_variance_basis=(
                "differenceCloud"
                if fitted.difference_cloud_explained_variance is not None
                else "degenerateDifferenceCloud"),
            pc1_power_iteration=fitted.power_iteration,
            train_accuracy=float(train_accuracy or 0.0),
            held_out_accuracy=held_accuracy,
            train_pair_count=n_train, held_out_pair_count=len(held),
            final_test_accuracy=final_test_accuracy, final_test_pair_count=len(test),
            split_overlap=split_overlap,
            contrast_mode=contrast_mode,
            sign_convention=fitted.sign_convention,
            sign_held_out_accuracy=fitted.sign_held_out_accuracy,
            sign_fallback_reason=fitted.sign_fallback_reason,
            orientation_seed=(orientation_seed
                              if contrast_mode == UNSUPERVISED_TEMPLATE_PAIR
                              else None),
            extraction_rendering=rendering_block,
            rendering_convention=reader_rendering_convention(extraction_rendering)))
    return stamp_layer_recommendation(artifacts)


def fit(model, dataset: ReaderDataset, template: TaskTemplate,
        *, layers: list[int] | None = None,
        orientation_seed: int = DEFAULT_ORIENTATION_SEED,
        extraction_rendering: ExtractionRendering | None = None
        ) -> list[ReaderArtifact]:
    """Fit one reader per layer from the dataset's train split; held-out rows
    (any ``split`` value that is neither ``train`` nor ``test``) fix each
    layer's SIGN and rank the layers, which is why their accuracy is SELECTION
    evidence; rows marked ``split: "finalTest"`` are read by none of that and are
    scored once at the end as ``finalTestAccuracy``.

    Rendering goes through :func:`render_scaffold` (family-aware, amendment A)
    and then the declared ``extraction_rendering``; activations are captured by
    the same extraction path every other recipe uses, at the template's LAT
    token position.
    """
    texts = fit_texts(dataset, template, model_id=model.model_id,
                      rendering=extraction_rendering)
    captured = extractor.activations(
        model, texts, template.reading_position,
        extraction_rendering or RAW_RENDERING).values
    return fit_activations(
        dataset, template, captured, model_id=model.model_id,
        revision=getattr(model, "revision", None), layers=layers,
        orientation_seed=orientation_seed,
        extraction_rendering=extraction_rendering)


# --- exact inference ---------------------------------------------------------

def score_activation(reader: ReaderArtifact, activation) -> float:
    """Pure probe scoring for a pre-captured LAT-token activation."""
    return reader.probe.score(activation)


def score_texts(model, reader: ReaderArtifact, texts: list[str]) -> list[float]:
    """The paper's inference, exactly: render the SAME template around each new
    stimulus under the SAME rendering the fit used, capture the LAT token at
    the reader's layer, normalize with the training parameters (inside the
    probe), and project. Not cosine-to-vector.

    A template-pair reader scores under its EXPERIMENTAL (T+) instruction: the
    direction was fitted as H(T+) − H(T−), so T+ is the rendering that matches
    the direction's construction.

    **A TEMPLATE-PAIR SCORE IS RELATIVE, NOT A PRESENCE TEST** (2026-08-28
    audit, F4). This paragraph replaces an earlier claim that T+ is "the
    rendering the probe's center and scale were calibrated on", which was
    affirmatively wrong: for ``unsupervisedTemplatePair`` the probe is fitted by
    :func:`vector_math.scalar_probe` over BOTH renderings of the same stimuli,
    so ``projectionCenter`` is the MIDPOINT of the T+ and T− train projection
    means. Inference renders new text under T+ only. Every score therefore
    carries a systematic positive offset of about
    ``|pos_mean − neg_mean| / (2·projectionScale)`` — bounded by 1 by the scale
    floor, but always positive — so:

    - **Comparisons are valid, thresholds are not.** Scores of the same reader
      across conditions, arms, or items differ by exactly the quantity of
      interest: the constant offset cancels in the difference. That is how the
      ``repeReaderScore`` outcome instrument uses them (``tasks.py``), and it is
      the only supported reading.
    - **``score > 0`` does NOT mean the concept is present.** Neutral text
      rendered under T+ scores positive systematically. Do not threshold a
      template-pair score at zero, and in particular do not reach for
      ``ScalarProbe.classifies_positive`` on this path — the probe's own
      accuracy statistics (``trainAccuracy``/``heldOutAccuracy``) are fitted and
      evaluated over BOTH renderings, where the midpoint IS the right threshold,
      which is why they stay meaningful while a single-rendering label does not.

    Supervised-content readers are unaffected: their fit and their inference
    render the same way, so their center is calibrated on the distribution they
    score. Twin prose: ``RepEReader.scoreTexts``.
    """
    if reader.substrate != SUBSTRATE:
        raise RepeReaderError(
            f"reader was fitted on substrate {reader.substrate!r}; this engine is "
            f"{SUBSTRATE!r} — reader artifacts are substrate-specific, re-fit here")
    if reader.model_id != model.model_id:
        raise RepeReaderError(
            f"reader was fitted on model {reader.model_id!r}; the loaded model is "
            f"{model.model_id!r} — a reader is a per-model measurement instrument, "
            "re-fit it for this model")
    rendering = reader.resolved_extraction_rendering
    instruction = (reader.template.instruction_pair.experimental
                   if reader.template.instruction_pair is not None else None)
    rendered = [render_scaffold(reader.template, stimulus=text,
                                concept=reader.concept, model_id=model.model_id,
                                instruction=instruction, rendering=rendering)
                for text in texts]
    captured = extractor.activations(
        model, rendered, reader.reading_position, rendering).values
    scores: list[float] = []
    for values in captured:
        if reader.layer >= len(values):
            raise RepeReaderError(
                f"reader layer {reader.layer} out of range for a "
                f"{len(values)}-layer capture — wrong model?")
        scores.append(score_activation(reader, values[reader.layer]))
    return scores


def score_text(model, reader: ReaderArtifact, text: str) -> float:
    return score_texts(model, reader, [text])[0]


# --- derive-steering conversion (REPE-IMPLEMENTATION-BRIEF §6) ------------------------------------

def derive_steering_sidecar(reader: ReaderArtifact, *, reader_file_name: str,
                            reader_bytes: bytes):
    """Pure half of the reader → steering-vector conversion:
    ``(vectors, sidecar)``.

    The reading direction **in reading orientation** is placed at the reader's
    layer (zeros below, Gemma-Scope import convention); the sidecar records
    ``source: repe-reader-lat`` + reader id/hash + ``controlMode``, because "we
    steered with a RepE reader direction" is a different claim from "we
    reproduced RepE control".

    **Whose sign the bytes carry — a two-wave story, and the second wave
    changed the answer.**

    *Wave one (audit finding 1, 2026-08-27).* ``ScalarProbe.score`` computes
    ``orientation · (a·direction − center)``, so a stored ``direction`` points
    at "more concept" only when ``orientation == +1``. Back when every fitted
    direction was signed by TRAIN-label majority, ``orientation`` — derived
    from the train class means — agreed with that choice, and folding it into
    the bytes was simply restating the training labels' verdict in the one
    place a steering vector can hold a sign. Readers whose PC1 came out
    anti-aligned carried ``orientation == −1``, and shipping their raw
    direction injected the concept BACKWARDS while every provenance stamp said
    forwards. Applying the orientation fixed that.

    *Wave two (the RepE wave, and review round 6).* Sign authority then moved
    to the HELD-OUT split: :func:`fit_direction` flips PC1 by held-out pair
    agreement and stamps ``signConvention: "heldOutPairAgreement"``. The
    ``direction`` on such a reader is ALREADY the held-out-chosen sign, while
    ``probe.orientation`` still comes from the TRAIN class means. When the two
    splits disagree the orientation is ``−1``, and applying it re-flipped the
    vector to the very direction held-out REJECTED — the wave-one repair,
    turned inside out by the wave that followed it.

    **So the rule is convention-aware:**

    - ``heldOutPairAgreement`` — the fitted direction is authoritative and
      ships UNFLIPPED. A train/held-out disagreement (``orientation == −1``) is
      not discarded: it is stamped as ``trainHeldOutSignDisagreement: true``,
      because "the training labels would have signed this the other way" is a
      fact about the direction's stability that a reader of the artifact is
      owed. Agreement stamps ``false`` — the field is present, either way,
      whenever held-out did the signing.
    - ``trainMajority`` (and every legacy schema-1 artifact, whose absent
      ``signConvention`` reads as train-majority) — wave one stands: the
      orientation is applied, and ``trainHeldOutSignDisagreement`` is absent
      because held-out never voted.

    ``readerProbeOrientation`` records the orientation either way, so what the
    conversion did is recoverable from the artifact alone.

    **The sidecar also carries the reader's ``evidenceRoles``**, because the
    question a derived vector raises first is which rows chose the sign its
    bytes now carry — and the reader file that could answer it does not travel
    with a bundle. Swift twin: ``RepEReader.deriveSteeringArtifact`` (parity
    owed for ``readerEvidenceRoles``).
    """
    from .vector_store import ConceptVectors, SteeringVectorSidecar

    probe_direction = list(reader.probe.direction)
    if not probe_direction:
        raise RepeReaderError("reader probe has an empty direction")
    orientation = float(reader.probe.orientation)
    held_out_signed = reader.sign_convention == HELD_OUT_PAIR_AGREEMENT
    if held_out_signed:
        direction = probe_direction
    else:
        direction = ([-x for x in probe_direction] if orientation < 0
                     else probe_direction)
    hidden = len(direction)
    per_layer = [[0.0] * hidden for _ in range(reader.layer)] + [direction]
    vectors = ConceptVectors(per_layer=per_layer)
    sidecar = SteeringVectorSidecar.make(
        model_id=reader.model_id, revision=reader.revision, concept=reader.concept,
        stimulus_set_hash=reader.dataset_hash, vectors=vectors,
        extraction_method=vm.ExtractionMethod.REPE_READER_LAT.value,
        reading_position=reader.reading_position)
    sidecar.recipeMethod = vm.ExtractionMethod.REPE_READER_LAT.value
    # PARTIAL by construction: zeros below the reader's layer, then one row.
    # Stamped so nothing downstream reads this artifact's layerCount as the
    # model's depth (review round 6, finding 2).
    sidecar.coversModelDepth = False
    sidecar.source = ARTIFACT_TYPE
    sidecar.readerID = reader_file_name
    sidecar.readerHash = _sha256_hex(reader_bytes)
    sidecar.controlMode = CONTROL_MODE
    sidecar.readerLayer = reader.layer
    sidecar.readerTemplateID = reader.template.id
    sidecar.readerTemplateHash = reader.template.hash
    sidecar.readerContrastMode = reader.contrast_mode
    sidecar.readerSignConvention = reader.sign_convention
    sidecar.readerEvidenceRoles = reader.resolved_evidence_roles
    sidecar.readerProbeOrientation = orientation
    sidecar.signConvention = reader.sign_convention
    if held_out_signed:
        sidecar.trainHeldOutSignDisagreement = orientation < 0
    if reader.extraction_rendering:
        sidecar.extractionRendering = dict(reader.extraction_rendering)
    return vectors, sidecar


def derive_steering_vector(reader_path: str, run_directory: str,
                           *, name: str | None = None) -> str:
    """Convert a reader into a standard steering-vector artifact — an explicit,
    provenance-stamped conversion. Returns ``<run_directory>/<name>``."""
    from .vector_store import save

    if not os.path.exists(reader_path):
        raise RepeReaderError(f"missing reader artifact: {reader_path}")
    with open(reader_path, "rb") as handle:
        reader_bytes = handle.read()
    reader = ReaderArtifact.from_dict(json.loads(reader_bytes.decode("utf-8")))
    vectors, sidecar = derive_steering_sidecar(
        reader, reader_file_name=os.path.basename(reader_path),
        reader_bytes=reader_bytes)
    name = name or f"{reader.concept}-repe-reader"
    save(vectors, sidecar, run_directory, name)
    return os.path.join(run_directory, name)


def binding_problems(reader: ReaderArtifact, *, ref_concept: str,
                     model_id: str, model_revision: str | None,
                     substrate: str = SUBSTRATE) -> list[str]:
    """The COMPLETE reader↔study binding, in one place (review 2026-08-02:
    verify and the runtime scorers each carried their own subset, so the
    runtime accepted readers verify would flag — and a forced freeze or the
    permissive draft path made that gap live). A reader is activation-,
    model-, revision-, and substrate-specific:

    - substrate must be THIS engine's (activations do not transfer);
    - modelID must equal the study model;
    - a revision is REQUIRED (an unattributable reader cannot be bound to
      exact fitted bytes) and must equal the study's pin when one exists;
    - the artifact's concept must be the concept the ref claims.

    Used by ``Manifest.verify`` (violations) and ``tasks._reader_scorers``
    (refusals). Swift twin: ``ExperimentStore.readerBindingProblems``."""
    problems: list[str] = []
    if reader.substrate != substrate:
        problems.append(
            f"reader '{ref_concept}' was fitted on substrate "
            f"{reader.substrate!r}, not this engine ({substrate!r}) — "
            "reader artifacts are substrate-specific and must be re-fitted")
    if reader.model_id != model_id:
        problems.append(
            f"reader '{ref_concept}' was fitted on {reader.model_id}, "
            f"not the study model {model_id}")
    if not reader.revision:
        problems.append(
            f"reader '{ref_concept}': artifact carries no model revision — "
            "readers bind to exact fitted bytes")
    elif model_revision and reader.revision != model_revision:
        problems.append(
            f"reader '{ref_concept}' was fitted on revision "
            f"{reader.revision[:12]}…, not the study's pinned "
            f"{model_revision[:12]}…")
    if reader.concept != ref_concept:
        problems.append(
            f"reader '{ref_concept}': the pinned artifact is for concept "
            f"'{reader.concept}' — the ref names the wrong instrument")
    return problems
