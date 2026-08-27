"""Read-only discovery of on-disk artifacts for the UI (parallel to Swift
``VectorCatalog`` scanning).

Walks the canonical data tree (``STEERLAB_ROOT``) to enumerate steering-vector
artifacts in ``runs/``, run directories and their outputs, experiment manifests,
and concept stimulus sets. Pure filesystem reads — no GPU, no model — so the web
client can populate pickers and browsers cheaply.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass

from . import paths
from .manifest import Manifest


@dataclass
class VectorArtifact:
    runDirectory: str
    name: str
    concept: str
    modelID: str
    revision: str | None
    layerCount: int
    hiddenSize: int
    method: str | None
    reading: str | None
    residualNormSource: str | None
    hasResidualNorms: bool
    extracted: str | None
    # Freshness-index pins: without the stimulus hash a client cannot tell a
    # reusable artifact from a stale one (unverifiable classifies as stale).
    stimulusSetHash: str | None = None
    neutralCorpusHash: str | None = None
    # Recipe method verbatim from the sidecar (e.g. "caaMeanDifference",
    # "emotionGrandMean"). The Swift freshness index normalizes method
    # identity as recipeMethod ?? extractionMethod on BOTH substrates —
    # without this field a server-built CAA vector (recipeMethod
    # "caaMeanDifference", extractionMethod "meanDifference") classifies as
    # "different extraction method" and shows permanently stale. ``method``
    # stays the flattened extractionMethod for compatibility.
    recipeMethod: str | None = None
    # Which engine extracted the vectors ("python-hf-transformers" here,
    # "swift-mlx" for the Mac app; None = legacy/unknown). On a shared tree the
    # catalog otherwise lists foreign-engine vectors indistinguishably, and
    # injection paths refuse foreign-substrate artifacts.
    substrate: str | None = None
    # Per-layer norms straight from the sidecar so the client can render the
    # same injection preview (vector norm / residual norm / Δnorm) it computes
    # locally. Field names are a pinned contract with the Swift decoder —
    # they match the sidecar keys exactly. One float per layer: small enough
    # to inline in GET /api/vectors.
    normsPerLayer: list[float] | None = None
    residualNormPerLayer: list[float] | None = None
    # The denominator CONVENTION those norms were measured under (sidecar key
    # ``residualNormConvention``). Absent = legacy artifact; never guessed.
    # The app's alpha control renders it so a researcher can see whether the
    # denominator behind a norm-unit alpha is the stamped whole-corpus rule.
    residualNormConvention: str | None = None
    # Grand-mean recipe membership is load-bearing for agent optimization:
    # the app pins and re-derives the corpus recipe, never the remote vector
    # bytes. Without this field a server-built grand-mean vector can steer in
    # Playground but cannot honestly seed an optimization.
    comparisonConcepts: list[str] | None = None
    selectedTopics: list[str] | None = None
    selectedSplits: list[str] | None = None
    grandMeanPopulation: dict[str, str] | None = None
    # designatedReference recipes only: the reference pin {"name", "hash"}
    # verbatim from the sidecar. Load-bearing for agent optimization the
    # same way comparisonConcepts is (review 2026-08-02 round 5, item 5):
    # the reference corpus is part of the recipe the app pins and
    # re-derives, and without it a server-built designated-reference vector
    # cannot honestly seed an optimization.
    designatedReference: dict[str, str] | None = None
    # Workspace-relative form of ``id`` ("runs/<run>/<name>") — the reference
    # a CLIENT should store. The Mac workspace is the source of truth and the
    # server a substitutable runner, so refs written into workspace data
    # (variants, manifests) must resolve on both substrates; this engine's
    # resolver joins relative refs under its root, and the app localizes the
    # artifact bytes on selection. ``id`` stays the absolute catalog form for
    # compatibility (chat sends, existing refs). None only in synthetic/test
    # constructions that predate the field.
    workspaceRelativeID: str | None = None
    # SHA-256 of the two files that ARE the artifact (2026-08-06). The client
    # localizes the pair into its own workspace, and until these existed the
    # only binding between a catalog row and the bytes on the client's disk
    # was that a file of that name existed: a truncated download, a
    # half-written cache, or a same-named artifact from a different
    # extraction all passed. With them the client verifies what it already
    # has, verifies what it just fetched, and refuses rather than overwrite
    # local truth. None when the file cannot be read (a listing must not die
    # on one unreadable artifact) or on pre-field servers, where the client
    # keeps its historical existence-only behaviour, loudly.
    sidecarSha256: str | None = None
    tensorSha256: str | None = None
    # Whether ``layerCount`` is the MODEL's depth or only this artifact's own
    # row count — verbatim from the sidecar's ``coversModelDepth``. See
    # :func:`covers_model_depth` for what absent means.
    coversModelDepth: bool | None = None

    @property
    def id(self) -> str:
        return os.path.join(self.runDirectory, self.name)

    @property
    def states_model_depth(self) -> bool:
        """Whether this row may be read as a statement of the model's depth."""
        return covers_model_depth(
            covers=self.coversModelDepth, extraction_method=self.method,
            recipe_method=self.recipeMethod)


#: Extraction methods whose artifacts are PARTIAL by construction — their
#: ``layerCount`` is a row count, not a model depth. ``repeReaderLAT`` writes
#: zeros below the reader's layer and stops there (the Gemma-Scope import
#: convention), so a reader fitted at block 13 of a 42-block model writes
#: ``layerCount: 14``.
PARTIAL_DEPTH_METHODS = frozenset({"repeReaderLAT"})


def covers_model_depth(*, covers: bool | None, extraction_method: str | None,
                       recipe_method: str | None) -> bool:
    """Whether an artifact's ``layerCount`` is the MODEL's depth.

    Not every steering-vector artifact is one row per block. Reader-derived
    directions are partial by construction, and reading a model's depth off one
    of them reports the reader's layer plus one — which then converts absolute
    sweep layers against a network that does not exist.

    The discriminator is EXPLICIT where the writer knows: ``coversModelDepth``
    on the sidecar, stamped forward. Absent is read by method, and the two
    legacy meanings are different:

    - absent on a reader-derived artifact (``repeReaderLAT``) = PARTIAL. Every
      artifact of that family ever written is partial, so the method is a
      sound witness for the ones written before the stamp existed.
    - absent anywhere else = FULL. CAA, LAT, grand-mean, designated-reference,
      OptVec and J-lens artifacts all carry one row per block, as do the
      artifacts old enough to carry no ``extractionMethod`` at all — the family
      predates the reader entirely.

    Swift twin: ``SteeringVectorSidecar.coversModelDepth`` (the resolved
    property, not the stored optional).
    """
    if covers is not None:
        return bool(covers)
    return not (str(extraction_method or "") in PARTIAL_DEPTH_METHODS
                or str(recipe_method or "") in PARTIAL_DEPTH_METHODS)


def _finite_float_list(value) -> list[float] | None:
    """Return a JSON-safe float list, or ``None`` when the sidecar contains
    non-finite values.

    Python's ``json`` parser accepts NaN/Infinity by default, but Starlette's
    response renderer correctly refuses to emit them. Catalog discovery is a
    summary endpoint, so one bad artifact must not make ``GET /api/vectors``
    unusable.
    """
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


@functools.lru_cache(maxsize=4096)
def _sha256_of_file(path: str, mtime_ns: int, size: int) -> str | None:
    """SHA-256 of ``path``, memoized on its identity AS OF the caller's stat.

    ``mtime_ns``/``size`` are unused in the body ON PURPOSE: they are cache-key
    material. A file rewritten in place gets a new key, so a stale digest can
    never be served; the bounded LRU keeps a long-lived server from holding a
    digest per artifact forever.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _artifact_sha256(path: str) -> str | None:
    """Cost decision (2026-08-06): the catalog hashes eagerly, at list time.

    A steering pair is ~1.3 MB, so a cold listing of N artifacts reads ~1.3N MB
    once. The alternatives were worse: making the hash a second per-artifact
    endpoint costs the client a round trip per row exactly when it is deciding
    what to fetch, and omitting it re-opens the existence-only hole this field
    closes. The read happens ONCE per file per process — ``runs/`` artifacts are
    immutable by the project's own rule, so the (path, mtime, size) key never
    turns over in practice, and a rewritten file re-hashes rather than lying.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return _sha256_of_file(path, stat.st_mtime_ns, stat.st_size)


def list_vectors(root: str | None = None) -> list[VectorArtifact]:
    """Every ``<name>.safetensors`` + ``<name>.json`` sidecar pair under runs/."""
    runs = paths.runs_directory(root)
    out: list[VectorArtifact] = []
    if not os.path.isdir(runs):
        return out
    for entry in sorted(os.listdir(runs)):
        run_dir = os.path.join(runs, entry)
        if not os.path.isdir(run_dir):
            continue
        for fname in sorted(os.listdir(run_dir)):
            if not fname.endswith(".json"):
                continue
            name = fname[:-len(".json")]
            tensor_path = os.path.join(run_dir, f"{name}.safetensors")
            if not os.path.exists(tensor_path):
                continue
            try:
                with open(os.path.join(run_dir, fname), encoding="utf-8") as handle:
                    sidecar = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if "modelID" not in sidecar or "layerCount" not in sidecar:
                continue  # not a vector sidecar (e.g. config.json, report.json)
            norms = _finite_float_list(sidecar.get("normsPerLayer"))
            residual_norms = _finite_float_list(sidecar.get("residualNormPerLayer"))
            out.append(VectorArtifact(
                runDirectory=run_dir, name=name,
                concept=sidecar.get("concept", name), modelID=sidecar["modelID"],
                revision=sidecar.get("revision"), layerCount=int(sidecar["layerCount"]),
                hiddenSize=int(sidecar.get("hiddenSize", 0)),
                method=sidecar.get("extractionMethod"),
                reading=sidecar.get("readingPosition"),
                residualNormSource=sidecar.get("residualNormSource"),
                hasResidualNorms=bool(residual_norms),
                # Full timestamp, not a [:10] date — same-day re-extractions
                # are indistinguishable otherwise (review 2026-08-03, P3);
                # clients shorten for display.
                extracted=sidecar.get("extractionDate") or None,
                stimulusSetHash=sidecar.get("stimulusSetHash"),
                neutralCorpusHash=sidecar.get("neutralCorpusHash"),
                recipeMethod=sidecar.get("recipeMethod"),
                substrate=sidecar.get("substrate"),
                normsPerLayer=norms,
                residualNormPerLayer=residual_norms,
                residualNormConvention=sidecar.get("residualNormConvention"),
                comparisonConcepts=sidecar.get("comparisonConcepts"),
                selectedTopics=sidecar.get("selectedTopics"),
                selectedSplits=sidecar.get("selectedSplits"),
                grandMeanPopulation=sidecar.get("grandMeanPopulation"),
                designatedReference=sidecar.get("designatedReference"),
                workspaceRelativeID=os.path.join("runs", entry, name),
                # Hashed AFTER the sidecar parsed: only rows that are really
                # vector artifacts pay the read.
                sidecarSha256=_artifact_sha256(os.path.join(run_dir, fname)),
                tensorSha256=_artifact_sha256(tensor_path),
                coversModelDepth=(
                    None if sidecar.get("coversModelDepth") is None
                    else bool(sidecar["coversModelDepth"]))))
    return out


@dataclass
class ReaderSummary:
    """One RepE reader artifact (``artifactType: repe-reader-lat``) under runs/.
    Readers are measurement instruments, listed separately from steering
    vectors; ``id`` is the artifact JSON path (what ``/api/reader/score``
    takes as ``readerID``)."""

    runDirectory: str
    name: str
    concept: str
    modelID: str
    revision: str | None
    substrate: str | None
    layer: int
    templateID: str | None
    templateHash: str | None
    templateDivergence: str | None
    datasetHash: str | None
    latTokenPosition: str | None
    trainAccuracy: float | None
    heldOutAccuracy: float | None
    extracted: str | None
    #: WHAT the pair differences contrast ("supervisedContent" /
    #: "unsupervisedTemplatePair") and HOW the sign was fixed
    #: ("heldOutPairAgreement" / "trainMajority"). Absent on a reader fitted
    #: before 2026-08-27, which means supervisedContent + trainMajority — the
    #: listing shows the stamp it has, never a guessed one.
    contrastMode: str | None = None
    signConvention: str | None = None
    #: The set's argmax-held-out-accuracy layer — a RECOMMENDATION. The layer a
    #: study reads is declared in its manifest; nothing selects it here.
    recommendedLayer: int | None = None

    @property
    def id(self) -> str:
        return os.path.join(self.runDirectory, f"{self.name}.json")


def list_readers(root: str | None = None) -> list[ReaderSummary]:
    """Every ``repe-reader-lat`` artifact JSON under runs/."""
    runs = paths.runs_directory(root)
    out: list[ReaderSummary] = []
    if not os.path.isdir(runs):
        return out
    for entry in sorted(os.listdir(runs)):
        run_dir = os.path.join(runs, entry)
        if not os.path.isdir(run_dir):
            continue
        for fname in sorted(os.listdir(run_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(run_dir, fname), encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if artifact.get("artifactType") != "repe-reader-lat":
                continue
            try:
                out.append(ReaderSummary(
                    runDirectory=run_dir, name=fname[:-len(".json")],
                    concept=str(artifact.get("concept", "")),
                    modelID=str(artifact.get("modelID", "")),
                    revision=artifact.get("revision"),
                    substrate=artifact.get("substrate"),
                    layer=int(artifact.get("layer", 0)),
                    templateID=artifact.get("templateID"),
                    templateHash=artifact.get("templateHash"),
                    templateDivergence=artifact.get("templateDivergence"),
                    datasetHash=artifact.get("datasetHash"),
                    latTokenPosition=artifact.get("latTokenPosition"),
                    trainAccuracy=artifact.get("trainAccuracy"),
                    heldOutAccuracy=artifact.get("heldOutAccuracy"),
                    extracted=(artifact.get("extractionDate") or "")[:10] or None,
                    contrastMode=artifact.get("contrastMode"),
                    signConvention=artifact.get("signConvention"),
                    recommendedLayer=artifact.get("recommendedLayer")))
            except (TypeError, ValueError):
                continue
    return out


@dataclass
class RunFileEntry:
    """One top-level file of a run directory with its byte size, so a remote
    client can size-gate previews and label "other files" without a round
    trip per file."""

    name: str
    size: int


@dataclass
class RunSummary:
    id: str
    path: str
    task: str | None
    hasReport: bool
    hasGenerations: bool
    hasCosineMatrix: bool
    vectorNames: list[str]
    files: list[str]
    # name+size sibling of ``files`` (which stays a plain string list — the
    # web workbench and the Swift RemoteRunRecord decode read it as-is).
    fileEntries: list[RunFileEntry] | None = None
    # The canonical per-run config.json stamps the local Results browser
    # shows (RunMetadata / write_run_config contract). Read tolerantly:
    # legacy runs without a stamp (or with a foreign-shaped config.json)
    # list every field as None — absence is shown, never invented.
    runType: str | None = None
    createdAt: str | None = None
    modelID: str | None = None
    revision: str | None = None
    experiment: str | None = None
    substrate: str | None = None
    appVersion: str | None = None


_STAMP_FIELDS = ("runType", "createdAt", "modelID", "revision", "experiment",
                 "substrate", "appVersion")


def _read_run_stamp(run_dir: str) -> dict[str, str | None]:
    """The config.json stamp fields, tolerantly: missing file, unreadable
    JSON, non-object top level, or non-string values all degrade to None
    per field (mirrors the Swift ``RunBrowser.readStamp``)."""
    out: dict[str, str | None] = {field: None for field in _STAMP_FIELDS}
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as h:
            data = json.load(h)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    for field in _STAMP_FIELDS:
        value = data.get(field)
        if isinstance(value, str):
            out[field] = value
    return out


def list_runs(root: str | None = None) -> list[RunSummary]:
    runs = paths.runs_directory(root)
    out: list[RunSummary] = []
    if not os.path.isdir(runs):
        return out
    for entry in sorted(os.listdir(runs), reverse=True):
        run_dir = os.path.join(runs, entry)
        if not os.path.isdir(run_dir):
            continue
        files = sorted(f for f in os.listdir(run_dir)
                       if os.path.isfile(os.path.join(run_dir, f)))
        task = None
        if "task.txt" in files:
            try:
                with open(os.path.join(run_dir, "task.txt"), encoding="utf-8") as h:
                    task = h.read().strip()
            except OSError:
                pass
        entries = []
        for fname in files:
            try:
                size = os.path.getsize(os.path.join(run_dir, fname))
            except OSError:
                size = 0
            entries.append(RunFileEntry(name=fname, size=size))
        out.append(RunSummary(
            id=entry, path=run_dir, task=task,
            hasReport="report.json" in files,
            hasGenerations="generations.jsonl" in files,
            hasCosineMatrix="cosine-matrix.csv" in files,
            vectorNames=[f[:-len(".safetensors")] for f in files
                         if f.endswith(".safetensors")],
            files=files,
            fileEntries=entries,
            **_read_run_stamp(run_dir)))
    return out


@dataclass
class ConceptSummary:
    name: str
    positiveCount: int
    negativeCount: int
    hasValidation: bool
    hasMarkers: bool
    # Cross-engine content hash of the paired stimuli (see
    # ``authoring.stimulus_content_hash``): the Swift Concept Lab compares it
    # against the local workspace's texts to surface local↔server drift.
    # None when both sides are empty (or on pre-field servers).
    contentHash: str | None = None


def list_concepts(root: str | None = None) -> list[ConceptSummary]:
    from . import authoring

    directory = paths.concepts_directory(root)
    out: list[ConceptSummary] = []
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        concept_dir = os.path.join(directory, name)
        if not os.path.isdir(concept_dir):
            continue
        content_hash = authoring.stimulus_content_hash(
            authoring._read_texts(os.path.join(concept_dir, "positive.jsonl")),
            authoring._read_texts(os.path.join(concept_dir, "negative.jsonl")))
        out.append(ConceptSummary(
            name=name,
            positiveCount=_line_count(os.path.join(concept_dir, "positive.jsonl")),
            negativeCount=_line_count(os.path.join(concept_dir, "negative.jsonl")),
            hasValidation=os.path.exists(os.path.join(concept_dir, "validation.jsonl")),
            hasMarkers=os.path.exists(os.path.join(concept_dir, "markers.json")),
            contentHash=content_hash))
    return out


def concept_preview(name: str, limit: int = 5, root: str | None = None) -> dict:
    concept_dir = paths.concept_directory(name, root)
    return {
        "name": name,
        "positive": _sample_texts(os.path.join(concept_dir, "positive.jsonl"), limit),
        "negative": _sample_texts(os.path.join(concept_dir, "negative.jsonl"), limit),
        "positiveCount": _line_count(os.path.join(concept_dir, "positive.jsonl")),
        "negativeCount": _line_count(os.path.join(concept_dir, "negative.jsonl")),
    }


def list_experiments(root: str | None = None) -> list[dict]:
    directory = paths.experiments_directory(root)
    out: list[dict] = []
    if not os.path.isdir(directory):
        return out
    names = set()
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "experiment.json")):
            names.add(entry)
        elif entry.endswith(".json"):
            names.add(entry[:-len(".json")])
    for name in sorted(names):
        try:
            m = Manifest.load(name, root)
            out.append(experiment_detail(m))
        except Exception as exc:  # noqa: BLE001
            out.append({"name": name, "status": "unreadable", "error": str(exc)})
    return out


def experiment_detail(manifest: Manifest) -> dict:
    # Sweep-selection provenance travels VERBATIM from the raw manifest: the
    # ``Manifest.conditions`` dataclasses deliberately don't model the
    # ``selection`` block (it is a pinned cross-engine JSON contract, not
    # engine state), so a client that must mark the sweep-recommended winner
    # (Screens) reads it here, straight from ``manifest.raw``.
    raw_conditions = {c.get("name"): c for c in manifest.raw.get("conditions", [])
                      if isinstance(c, dict)}
    conditions = []
    for cond in manifest.conditions:
        entry = {"name": cond.name, "bandWidth": cond.band_width,
                 "normUnits": cond.alpha_in_norm_units,
                 "slots": [{"concept": s.concept, "layer": s.layer, "alpha": s.alpha}
                           for s in cond.slots]}
        if cond.control_type is not None:
            entry["controlType"] = cond.control_type
        selection = raw_conditions.get(cond.name, {}).get("selection")
        if isinstance(selection, dict):
            entry["selection"] = selection
        conditions.append(entry)
    detail = {
        "name": manifest.name, "status": manifest.status, "modelID": manifest.model_id,
        "modelRevision": manifest.model_revision, "description": manifest.description,
        "promptMode": manifest.prompt_mode, "temperature": manifest.temperature,
        "maxTokens": manifest.max_tokens, "seeds": manifest.seeds,
        "taskPromptsFile": manifest.task_prompts_file,
        "studyKind": manifest.study_kind,
        "concepts": [{"name": c.name, "hash": c.stimulus_set_hash[:12],
                      "method": c.options.method.value,
                      "reading": c.options.reading_position.label}
                     for c in manifest.concepts],
        "conditions": conditions,
        "variantConditions": [{"name": v.name, "artifactPath": v.artifact_path,
                               "baseModelID": v.artifact.get("baseModelID")}
                              for v in manifest.variant_conditions],
        "multiAgentScenarioPath": manifest.multi_agent_scenario_path,
        "evaluation": (None if manifest.evaluation is None else
                       {"kind": manifest.evaluation.kind,
                        "judgeModel": manifest.evaluation.judge_model}),
        "freezeHash": manifest.freeze_hash,
    }
    # Confirmation-study manifests carry a declared perturbation policy; it is
    # manifest DATA (same verbatim rule as ``selection``).
    if isinstance(manifest.raw.get("perturbationPolicy"), dict):
        detail["perturbationPolicy"] = manifest.raw["perturbationPolicy"]
    # The declared sweep spec (grid, dev split, battery, optional ``selection``
    # criterion) is manifest DATA under the same verbatim rule: the Screens
    # client renders the declared selection rule from it.
    if isinstance(manifest.raw.get("sweep"), dict):
        detail["sweep"] = manifest.raw["sweep"]
    return detail


def _line_count(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _sample_texts(path: str, limit: int) -> list[str]:
    out: list[str] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out
