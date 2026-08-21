"""Faithful RepE reader LAT (Zou et al., arXiv:2310.01405 §3.1, App. C.1).

A reader is a fitted **measurement instrument**, not a steering vector: task
template + LAT token position + PCA direction with training normalization +
held-out scalar accuracy (see ``docs/REPE-IMPLEMENTATION-BRIEF.md``). Pipeline:

    stimulus → render task template → capture hidden state at the LAT token
    → normalized pair differences → centered PCA → PC1 oriented by labels
    → ScalarProbe fitted on the TRAIN activations

Inference renders the *same* template, captures the *same* token position, and
scores through the stored probe (training center/scale) — never a raw
cosine-to-vector shortcut. A reader can *derive* a steering vector
(:func:`derive_steering_vector`), but the derived artifact stamps its reader
provenance so "reading-vector activation addition" is never conflated with the
paper's full control experiments.

Concept-agnostic by design: concepts, templates, and stimuli enter as data.
Reader artifacts are substrate-specific (:data:`SUBSTRATE`) — activations do
not transfer between engines, so a Swift/MLX reader must be re-fitted here.

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

from ..experiment.prompt_render import READER_RENDERING_CONVENTION, render_reader
from . import extractor
from . import vector_math as vm
from .reading_position import LAST_TOKEN, ReadingPosition
from .vector_math import ScalarProbe
from .vector_store import SUBSTRATE  # single shared definition (re-exported here)

ARTIFACT_TYPE = "repe-reader-lat"
CONTROL_MODE = "reading-vector activation addition"


class RepeReaderError(Exception):
    pass


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- task templates ----------------------------------------------------------

@dataclass(frozen=True)
class TaskTemplate:
    """One registry entry from ``prompts/templates/<id>.json``.

    ``hash`` is the SHA-256 of the file's raw bytes (stimulus-set convention):
    changing the template changes every artifact fitted through it.
    ``divergence`` marks deliberate departures from the paper (e.g. the
    unnamed clean-room scaffold, which never names the concept).
    """

    id: str
    text: str
    concept_slot: bool
    lat_token: str
    hash: str
    divergence: str | None = None

    @property
    def reading_position(self) -> ReadingPosition:
        if self.lat_token != "final":
            raise RepeReaderError(
                f"template {self.id!r}: unsupported latToken {self.lat_token!r} "
                "(only 'final' is implemented)")
        return LAST_TOKEN

    def render(self, *, stimulus: str, concept: str | None = None) -> str:
        """Pure slot substitution; the family-aware scaffold pass happens in
        :func:`render_scaffold`."""
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
        return text.replace("{{stimulus}}", stimulus)

    def to_dict(self) -> dict:
        out = {"id": self.id, "conceptSlot": self.concept_slot, "text": self.text,
               "latToken": self.lat_token, "hash": self.hash}
        if self.divergence is not None:
            out["divergence"] = self.divergence
        return out

    @classmethod
    def from_dict(cls, d: dict, *, hash: str | None = None) -> "TaskTemplate":
        return cls(id=str(d["id"]), text=str(d["text"]),
                   concept_slot=bool(d.get("conceptSlot", False)),
                   lat_token=str(d.get("latToken", "final")),
                   hash=str(hash if hash is not None else d.get("hash", "")),
                   divergence=d.get("divergence"))


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
    return template


def render_scaffold(template: TaskTemplate, *, stimulus: str,
                    concept: str | None, model_id: str) -> str:
    """Template substitution + the family-aware scaffold pass (amendment A)."""
    return render_reader(template.render(stimulus=stimulus, concept=concept),
                         model_id=model_id)


# --- reader dataset ----------------------------------------------------------

@dataclass(frozen=True)
class ReaderPair:
    """One paired row: unrendered stimuli + the template that will render them
    (brief §2). Storing the raw stimulus keeps the corpus re-renderable per
    model family while the hash pins the bytes."""

    positive_stimulus: str
    negative_stimulus: str
    concept: str
    template_id: str
    id: str | None = None
    topic: str | None = None
    split: str = "train"


@dataclass(frozen=True)
class ReaderDataset:
    concept: str
    pairs: tuple[ReaderPair, ...]
    hash: str

    @property
    def train(self) -> list[ReaderPair]:
        return [p for p in self.pairs if p.split == "train"]

    @property
    def held_out(self) -> list[ReaderPair]:
        return [p for p in self.pairs if p.split != "train"]


def parse_pairs(data: bytes | str, *, source: str = "<reader pairs>") -> ReaderDataset:
    """Parse pairs JSONL from raw bytes; hash = SHA-256 over those bytes
    (stimulus_set convention). Rows must share one concept — a reader measures
    exactly one. This is the in-memory seam under :func:`load_pairs`, so an
    uploaded corpus can be fully validated *before* any canonical file is
    written; ``source`` names the origin in error messages."""
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
        for key in ("positiveStimulus", "negativeStimulus", "concept", "templateID"):
            if key not in obj:
                raise RepeReaderError(f"{source}: row missing {key!r}")
        pairs.append(ReaderPair(
            positive_stimulus=str(obj["positiveStimulus"]),
            negative_stimulus=str(obj["negativeStimulus"]),
            concept=str(obj["concept"]), template_id=str(obj["templateID"]),
            id=obj.get("id"), topic=obj.get("topic"),
            split=str(obj.get("split") or "train").lower()))
    if not pairs:
        raise RepeReaderError(f"empty reader pairs file: {source}")
    concepts = {p.concept for p in pairs}
    if len(concepts) > 1:
        raise RepeReaderError(
            f"{source}: mixed concepts {sorted(concepts)} — one reader dataset "
            "per concept")
    return ReaderDataset(concept=pairs[0].concept, pairs=tuple(pairs),
                         hash=_sha256_hex(data))


def load_pairs(path: str) -> ReaderDataset:
    """Load ``pairs.jsonl`` from disk (delegates to :func:`parse_pairs`)."""
    if not os.path.exists(path):
        raise RepeReaderError(f"missing reader pairs file: {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    return parse_pairs(data, source=path)


# --- reader artifact ---------------------------------------------------------

@dataclass
class ReaderArtifact:
    """The fitted instrument (brief §4): one per concept × layer × template ×
    model × substrate. The full template record is embedded so inference is
    standalone and drift-proof; ``templateID``/``templateHash`` remain the
    registry pins."""

    model_id: str
    revision: str | None
    concept: str
    layer: int
    template: TaskTemplate
    dataset_hash: str
    probe: ScalarProbe
    pc1_explained_variance: float
    train_accuracy: float
    held_out_accuracy: float | None
    train_pair_count: int
    held_out_pair_count: int
    substrate: str = SUBSTRATE
    rendering_convention: str = READER_RENDERING_CONVENTION
    extraction_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def lat_token_position(self) -> str:
        return self.template.lat_token

    @property
    def reading_position(self) -> ReadingPosition:
        return self.template.reading_position

    def to_dict(self) -> dict:
        out = {
            "artifactType": ARTIFACT_TYPE,
            "schemaVersion": 1,
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
            "pc1ExplainedVariance": self.pc1_explained_variance,
            "trainAccuracy": self.train_accuracy,
            "heldOutAccuracy": self.held_out_accuracy,
            "trainPairCount": self.train_pair_count,
            "heldOutPairCount": self.held_out_pair_count,
            "renderingConvention": self.rendering_convention,
            "extractionDate": self.extraction_date,
        }
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
        return cls(
            model_id=str(d["modelID"]), revision=d.get("revision"),
            concept=str(d["concept"]), layer=int(d["layer"]), template=template,
            dataset_hash=str(d["datasetHash"]),
            probe=ScalarProbe.from_dict(d["probe"]),
            pc1_explained_variance=float(d["pc1ExplainedVariance"]),
            train_accuracy=float(d["trainAccuracy"]),
            held_out_accuracy=(None if d.get("heldOutAccuracy") is None
                               else float(d["heldOutAccuracy"])),
            train_pair_count=int(d.get("trainPairCount", 0)),
            held_out_pair_count=int(d.get("heldOutPairCount", 0)),
            substrate=str(d.get("substrate", "")),
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

def _pc1_oriented(positive: list[list[float]], negative: list[list[float]]
                  ) -> tuple[list[float], float]:
    """RepE App. C.1 direction: L2-normalized pair differences → centered PCA
    → PC1, sign oriented by the paired labels (majority of normalized diffs
    project positive; ties fall back to the class-mean criterion). Mirrors the
    inner LAT math of :func:`vector_math.direction` but returns the *unit* PC
    plus its explained variance — the reader does not norm-match to CAA."""
    pos = np.asarray(positive, dtype=np.float32)
    neg = np.asarray(negative, dtype=np.float32)
    if pos.shape != neg.shape:
        raise RepeReaderError(
            f"unpaired activations: positive {pos.shape} vs negative {neg.shape}")
    diffs: list[np.ndarray] = []
    for d in (pos - neg).astype(np.float32):
        norm = vm.l2_norm(d.tolist())
        if norm > 0:
            diffs.append((d / np.float32(norm)).astype(np.float32))
    if len(diffs) < 2:
        raise RepeReaderError("need at least 2 non-degenerate train pairs")
    # Alternate orientation before PCA so the shared concept direction is not
    # centered out of normalized labeled pairs (same convention as vm.direction).
    oriented = [d if i % 2 == 0 else -d for i, d in enumerate(diffs)]
    result = vm.principal_components_with_variance(
        [o.tolist() for o in oriented], 1)
    if not result.components:
        raise RepeReaderError("degenerate pair differences (no PC1)")
    pc = result.components[0]
    explained = result.explained_variance[0]
    scores = [vm.dot(d.tolist(), pc) for d in diffs]
    positive_scores = sum(1 for s in scores if s > 0)
    n = len(scores)
    if positive_scores * 2 == n:
        flip = vm.dot(pc, vm.mean_difference(positive, negative)) < 0
    else:
        flip = positive_scores * 2 < n
    if flip:
        pc = [-x for x in pc]
    return pc, explained


def _pair_accuracy(probe: ScalarProbe, positive: list[list[float]],
                   negative: list[list[float]]) -> float | None:
    total = len(positive) + len(negative)
    if total == 0:
        return None
    correct = sum(1 for row in positive if probe.classifies_positive(row))
    correct += sum(1 for row in negative if not probe.classifies_positive(row))
    return correct / total


def fit(model, dataset: ReaderDataset, template: TaskTemplate,
        *, layers: list[int] | None = None) -> list[ReaderArtifact]:
    """Fit one reader per layer from the dataset's train split; held-out rows
    (any other ``split`` value) score the instrument it was not fitted on.

    Rendering goes through :func:`render_scaffold` (family-aware, amendment A);
    activations are captured by the same extraction path every other recipe
    uses, at the template's LAT token position.
    """
    for pair in dataset.pairs:
        if pair.template_id != template.id:
            raise RepeReaderError(
                f"pair {pair.id!r} pins template {pair.template_id!r} but the fit "
                f"uses {template.id!r}")
    train = dataset.train
    held = dataset.held_out
    if len(train) < 2:
        raise RepeReaderError(
            f"need at least 2 train pairs, have {len(train)} "
            "(rows default to split 'train')")

    ordered = train + held
    texts: list[str] = []
    for pair in ordered:
        for stimulus in (pair.positive_stimulus, pair.negative_stimulus):
            texts.append(render_scaffold(template, stimulus=stimulus,
                                         concept=dataset.concept,
                                         model_id=model.model_id))
    captured = extractor.activations(model, texts, template.reading_position).values
    layer_count = len(captured[0]) if captured else 0
    if layer_count == 0:
        raise RepeReaderError("no layers captured")
    chosen = list(range(layer_count)) if layers is None else list(layers)

    n_train = len(train)
    artifacts: list[ReaderArtifact] = []
    for layer in chosen:
        if not 0 <= layer < layer_count:
            raise RepeReaderError(f"layer {layer} out of range 0..{layer_count - 1}")
        pos_train = [captured[2 * i][layer] for i in range(n_train)]
        neg_train = [captured[2 * i + 1][layer] for i in range(n_train)]
        pos_held = [captured[2 * (n_train + i)][layer] for i in range(len(held))]
        neg_held = [captured[2 * (n_train + i) + 1][layer] for i in range(len(held))]

        pc, explained = _pc1_oriented(pos_train, neg_train)
        # Training normalization: center at the train activation mean, score
        # scale/center from the train projections — the "fit params" the paper's
        # inference reuses on new text.
        center = vm.mean(pos_train + neg_train)
        probe = vm.scalar_probe(pc, pos_train, neg_train, activation_center=center)
        train_accuracy = _pair_accuracy(probe, pos_train, neg_train)
        held_accuracy = _pair_accuracy(probe, pos_held, neg_held)
        artifacts.append(ReaderArtifact(
            model_id=model.model_id, revision=getattr(model, "revision", None),
            concept=dataset.concept, layer=layer, template=template,
            dataset_hash=dataset.hash, probe=probe,
            pc1_explained_variance=explained,
            train_accuracy=float(train_accuracy or 0.0),
            held_out_accuracy=held_accuracy,
            train_pair_count=n_train, held_out_pair_count=len(held)))
    return artifacts


# --- exact inference ---------------------------------------------------------

def score_activation(reader: ReaderArtifact, activation) -> float:
    """Pure probe scoring for a pre-captured LAT-token activation."""
    return reader.probe.score(activation)


def score_texts(model, reader: ReaderArtifact, texts: list[str]) -> list[float]:
    """The paper's inference, exactly: render the SAME template around each new
    stimulus, capture the LAT token at the reader's layer, normalize with the
    training parameters (inside the probe), and project. Not cosine-to-vector.
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
    rendered = [render_scaffold(reader.template, stimulus=text,
                                concept=reader.concept, model_id=model.model_id)
                for text in texts]
    captured = extractor.activations(model, rendered, reader.reading_position).values
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


# --- derive-steering conversion (brief §6) ------------------------------------

def derive_steering_vector(reader_path: str, run_directory: str,
                           *, name: str | None = None) -> str:
    """Convert a reader into a standard steering-vector artifact — an explicit,
    provenance-stamped conversion, because "we steered with a RepE reader
    direction" is a different claim from "we reproduced RepE control".

    The unit reading direction is placed at the reader's layer (zeros below,
    Gemma-Scope import convention); the sidecar records
    ``source: repe-reader-lat`` + reader id/hash + ``controlMode``.
    Returns ``<run_directory>/<name>`` (artifact base path).
    """
    from .vector_store import ConceptVectors, SteeringVectorSidecar, save

    if not os.path.exists(reader_path):
        raise RepeReaderError(f"missing reader artifact: {reader_path}")
    with open(reader_path, "rb") as handle:
        reader_bytes = handle.read()
    reader = ReaderArtifact.from_dict(json.loads(reader_bytes.decode("utf-8")))

    direction = list(reader.probe.direction)
    hidden = len(direction)
    per_layer = [[0.0] * hidden for _ in range(reader.layer)] + [direction]
    vectors = ConceptVectors(per_layer=per_layer)
    sidecar = SteeringVectorSidecar.make(
        model_id=reader.model_id, revision=reader.revision, concept=reader.concept,
        stimulus_set_hash=reader.dataset_hash, vectors=vectors,
        extraction_method="repeReaderLAT",
        reading_position=reader.reading_position)
    sidecar.source = ARTIFACT_TYPE
    sidecar.readerID = os.path.basename(reader_path)
    sidecar.readerHash = _sha256_hex(reader_bytes)
    sidecar.controlMode = CONTROL_MODE
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
