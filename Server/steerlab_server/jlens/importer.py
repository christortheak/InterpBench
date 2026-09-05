"""Import a published J-lens: hash the upstream, convert once, write the record.

Acquisition (bytes into the HF cache) and import (conversion + record into the
workspace) are separate operations on purpose — plan §11.0.1. This module is
the second one, and it is offline by construction: it resolves the pinned
snapshot from the local cache and never reaches the network.

The conversion is the reason this stage exists at all. The reference loader
promotes every layer to float32 at construction, so the published fp16 tensor
becomes roughly twice its size resident, for every layer, regardless of how
many are armed. Converting once to per-layer safetensors lets generation read a
single layer at its stored width.
"""

from __future__ import annotations

import glob
import os

from ..experiment import paths
from . import backend as backend_mod
from .schemas import (ConvertedRef, FitProvenance, JLensError, JLensRecord,
                      SourceRef, sha256_bytes, sha256_file, write_record)

#: CURATED lenses, keyed by runtime model id (plan §3.1). Filenames are DATA:
#: upstream names the tensor after the checkpoint variant, so no rule derives
#: them from the model id.
#:
#: ``tier`` records THIS PROJECT'S SCOPE DECISION about where its evidence comes
#: from — not a claim about what a model is scientifically capable of. The study
#: chose 27B (CLAUDE.md, 2026-07-27), so 27B is the evidence tier and everything
#: else is testing tier *here*. A 12B J-lens result could be perfectly good
#: science; it would simply not be THIS study's evidence, and the tier is what
#: keeps a run from being cited as if the decision had been otherwise. Change the
#: decision and the tier changes with it — it is a policy field, revisable, not a
#: verdict on the model.
#:
#: This table is no longer the boundary of what can be imported (2026-09-05).
#: The published repository carries lenses for some forty models — Gemma,
#: Llama, Qwen, OLMo, GPT-OSS and more — and any of them can be acquired and
#: imported: the folder and tensor are read from the published
#: ``config.yaml`` that names the model (:func:`published_entries`), and the
#: TIER, which no upstream file can supply, is declared by the researcher at
#: import and stamped on the record with its provenance. What the table still
#: does is fix the tier of the models this study has decided about, so a
#: curated model's tier can never be re-declared on the command line.
SUPPORTED: dict[str, dict] = {
    "google/gemma-3-27b-it": {
        "folder": "gemma-3-27b-it/jlens/Salesforce-wikitext",
        "tensor": "gemma-3-27b-it_jacobian_lens.pt",
        "config": "config.yaml",
        "tier": "evidence",
    },
    "google/gemma-3-12b-it": {
        "folder": "gemma-3-12b-it/jlens/Salesforce-wikitext",
        "tensor": "gemma-3-12b-it_jacobian_lens.pt",
        "config": "config.yaml",
        "tier": "testing",
    },
    "google/gemma-3-4b-it": {
        "folder": "gemma-3-4b-it/jlens/Salesforce-wikitext",
        "tensor": "gemma-3-4b-it_jacobian_lens.pt",
        "config": "config.yaml",
        "tier": "testing",
    },
}

LENS_REPO = "neuronpedia/jacobian-lens"

#: The tiers a researcher may declare for a model outside the curated table.
TIERS = ("evidence", "testing")

#: Every published lens folder's config, and nothing else: the glob an
#: acquisition fetches to learn WHICH folder holds a model's lens before
#: fetching that folder's gigabytes. ~40 files of ~2.5 KB.
PUBLISHED_CONFIG_GLOB = "*/jlens/*/config.yaml"
TENSOR_SUFFIX = "_jacobian_lens.pt"
CONFIG_FILE = "config.yaml"


def supported_models() -> list[str]:
    return sorted(SUPPORTED)


def published_entries(*, snapshot: str | None = None,
                      offline: bool = True) -> dict[str, dict]:
    """Every published lens whose ``config.yaml`` is in the local cache (or in
    ``snapshot``), keyed by the HF model id the config names.

    Read from the files themselves: ``hf_model_name`` identifies the model,
    the folder is where the config sits, and the tensor is the one
    ``*_jacobian_lens.pt`` beside it (``None`` until acquired, or when the
    folder holds several). A model whose id appears under two folders is
    recorded as ambiguous rather than resolved to either.
    """
    snap = snapshot or _cached_snapshot(PUBLISHED_CONFIG_GLOB, offline=offline)
    entries: dict[str, dict] = {}
    pattern = os.path.join(snap, "*", "jlens", "*", CONFIG_FILE)
    for config_path in sorted(glob.glob(pattern)):
        folder_abs = os.path.dirname(config_path)
        folder = os.path.relpath(folder_abs, snap).replace(os.sep, "/")
        fit, _ = _parse_config(config_path)
        if not fit.modelID:
            continue
        tensors = sorted(os.path.basename(p) for p in
                         glob.glob(os.path.join(folder_abs, "*" + TENSOR_SUFFIX)))
        entry = {"folder": folder,
                 "tensor": tensors[0] if len(tensors) == 1 else None,
                 "tensorCandidates": tensors,
                 "config": CONFIG_FILE, "tier": None,
                 "corpus": fit.corpus, "fitDtype": fit.dtype,
                 "promptsFitted": fit.promptsFitted}
        if fit.modelID in entries:
            previous = entries[fit.modelID]
            previous.setdefault("ambiguous", [previous["folder"]]).append(folder)
            continue
        entries[fit.modelID] = entry
    return entries


def resolve_published(model_id: str, *, snapshot: str | None = None,
                      offline: bool = True) -> dict:
    """The published entry for one model, or a refusal that names what IS
    published so the next command can be exact."""
    try:
        entries = published_entries(snapshot=snapshot, offline=offline)
    except JLensError as exc:
        raise JLensError(
            f"'{model_id}' is not in the curated table and no published lens "
            f"configs are cached to look it up in — acquire it first "
            f"(steerlab-server jlens acquire {model_id}), which fetches the "
            f"published configs and then the lens. Underlying: {exc}") from exc
    entry = entries.get(model_id)
    if entry is None:
        raise JLensError(
            f"'{model_id}' has no published lens in {LENS_REPO} (curated: "
            f"{supported_models()}; published and cached: "
            f"{sorted(entries)}). `steerlab-server jlens supported "
            f"--published` lists what upstream carries.")
    if entry.get("ambiguous"):
        raise JLensError(
            f"'{model_id}' is named by more than one published folder "
            f"({entry['ambiguous']}) — refusing to pick one silently")
    return entry


def entry_for(model_id: str, *, snapshot: str | None = None,
              offline: bool = True) -> dict:
    """The lens entry for ``model_id``: the curated row when there is one,
    else the published folder its config names."""
    curated = SUPPORTED.get(model_id)
    if curated is not None:
        return curated
    return resolve_published(model_id, snapshot=snapshot, offline=offline)


def is_curated(model_id: str) -> bool:
    return model_id in SUPPORTED


def tier_of(model_id: str | None, record=None) -> tuple[str, str]:
    """``(tier, source)`` for a model — the curated row when there is one,
    else the tier the lens record carries from its import declaration, else
    ``("unknown", "none")``.

    One resolution for every consumer (qualify, freeze, run start, the
    J-space report), so a declared tier is honoured everywhere or nowhere.
    A curated row always wins: a declaration made before the study decided
    about a model does not survive the decision.
    """
    curated = SUPPORTED.get(model_id or "")
    if curated is not None and curated.get("tier"):
        return curated["tier"], "curated"
    declared = getattr(record, "tier", None) if record is not None else None
    if declared:
        return declared, getattr(record, "tierSource", None) or "declared"
    return "unknown", "none"


def resolve_tier(model_id: str, declared: str | None) -> tuple[str, str]:
    """The tier an import stamps, from the table or the researcher's flag.

    A curated model takes its row's tier and refuses a flag that says
    otherwise — the row is policy, changed by editing it, not per command.
    An uncurated model REQUIRES the flag: nothing upstream can say whether
    this study treats a model as evidence or rehearsal.
    """
    curated = SUPPORTED.get(model_id)
    if curated is not None:
        if declared is not None and declared != curated["tier"]:
            raise JLensError(
                f"'{model_id}' is curated at tier '{curated['tier']}' and "
                f"--tier {declared} contradicts it — the table row is this "
                f"project's policy; change the row, not the flag")
        return curated["tier"], "curated"
    if declared is None:
        raise JLensError(
            f"'{model_id}' has no curated row, so its evidence tier must be "
            f"declared at import: --tier evidence (this study may cite lens "
            f"results on it) or --tier testing (rehearsal only; freeze "
            f"refuses it). Nothing upstream can make that decision.")
    if declared not in TIERS:
        raise JLensError(
            f"--tier {declared!r} is not a tier; choose one of {list(TIERS)}")
    return declared, "declared"


def lens_id_for(model_id: str, corpus: str | None = "wikitext") -> str:
    """Stable, filesystem-safe identity for a lens artifact directory.

    The suffix names the corpus the lens was FITTED on, read from the
    published config's ``dataset.name`` (``Salesforce/wikitext`` → ``wikitext``,
    ``NeelNanda/pile-10k`` → ``pile-10k``) — not assumed: the published set
    is not all wikitext. A config that names no corpus gets ``unknown-corpus``
    rather than a claim. The default keeps the historical id for callers
    that only have a model id.
    """
    if not corpus:
        slug = "unknown-corpus"
    else:
        slug = corpus.split(":", 1)[0].rsplit("/", 1)[-1].strip() or "unknown-corpus"
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in slug)
    return model_id.replace("/", "--") + "--jlens-" + safe


def _cached_snapshot(pattern: str, *, offline: bool = True) -> str:
    """Resolve the pinned snapshot from the local HF cache.

    ``pattern`` is a repo-relative glob — a lens folder (``<folder>/*``) or
    :data:`PUBLISHED_CONFIG_GLOB`. Offline by default: import must not
    silently pull a different snapshot than the one acquisition pinned. When
    nothing is cached the error names the acquisition step rather than a hub
    traceback.
    """
    from huggingface_hub import snapshot_download

    if not pattern.endswith("*"):
        pattern = pattern + "/*"
    try:
        return snapshot_download(LENS_REPO, allow_patterns=[pattern],
                                 local_files_only=offline)
    except Exception as exc:  # noqa: BLE001 — remapped to an actionable error
        raise JLensError(
            f"no cached lens files for '{pattern}' — acquire it first "
            f"(steerlab-server jlens acquire, or bootstrap.sh --with-jlens). "
            f"Underlying: {exc}") from exc


def _parse_config(path: str) -> tuple[FitProvenance, str | None]:
    if not os.path.exists(path):
        return FitProvenance(), None
    raw = open(path, "rb").read()
    try:
        import yaml

        cfg = yaml.safe_load(raw.decode("utf-8")) or {}
    except Exception:  # noqa: BLE001 — a lens without a parseable config still imports
        return FitProvenance(), sha256_bytes(raw)
    fit = cfg.get("fit") or {}
    dataset = cfg.get("dataset") or {}
    results = cfg.get("results") or {}
    corpus = dataset.get("name")
    if corpus and dataset.get("config"):
        corpus = f"{corpus}:{dataset['config']}"
    # revisionKnown stays False unless upstream actually pins one. The runtime
    # revision is NOT a substitute — that would relabel unknown as known.
    revision = cfg.get("revision") or fit.get("revision")
    return FitProvenance(
        modelID=cfg.get("hf_model_name"),
        revision=revision,
        revisionKnown=bool(revision),
        dtype=fit.get("dtype"),
        corpus=corpus,
        promptsFitted=results.get("prompts_fitted"),
        maxSeqLen=fit.get("max_seq_len"),
    ), sha256_bytes(raw)


def convert_to_per_layer(source, directory: str, *, dtype: str | None = None) -> ConvertedRef:
    """Write one safetensors tensor per source layer, preserving stored dtype.

    Keys are ``layer_<i>`` — the same convention the steering-vector store uses,
    so nothing has to learn a second layout.
    """
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    tensors = {}
    stored_dtype = dtype
    for layer in source.source_layers:
        j = source.jacobian(layer)
        if dtype is not None:
            import torch

            j = j.to(getattr(torch, dtype))
        stored_dtype = stored_dtype or str(j.dtype).replace("torch.", "")
        tensors[f"layer_{layer}"] = j.contiguous()
    path = os.path.join(directory, "jacobians.safetensors")
    save_file(tensors, path)
    return ConvertedRef(path=path, dtype=stored_dtype or "unknown",
                        sha256=sha256_file(path), layerCount=len(tensors))


def import_lens(model_id: str, *, root: str | None = None,
                source=None, snapshot: str | None = None,
                offline: bool = True, tier: str | None = None) -> JLensRecord:
    """Import the published lens for ``model_id`` into the workspace lens store.

    ``tier`` is the researcher's declaration for a model outside the curated
    table (required there, refused where it contradicts a curated row — see
    :func:`resolve_tier`). ``source``/``snapshot`` are injection points for
    tests (a StubBackend and a synthetic snapshot directory); production
    passes neither.
    """
    resolved_tier, tier_source = resolve_tier(model_id, tier)
    entry = entry_for(model_id, snapshot=snapshot, offline=offline)
    snap = snapshot or _cached_snapshot(entry["folder"], offline=offline)
    folder = os.path.join(snap, entry["folder"])
    tensor_name = entry.get("tensor")
    if tensor_name is None:
        candidates = entry.get("tensorCandidates") or []
        raise JLensError(
            f"the published folder '{entry['folder']}' holds "
            f"{len(candidates)} '*{TENSOR_SUFFIX}' file(s) ({candidates}) — "
            f"an import needs exactly one; acquire the folder first, or add a "
            f"curated row naming the tensor")
    tensor_path = os.path.join(folder, tensor_name)
    config_path = os.path.join(folder, entry["config"])

    if source is None:
        if not os.path.exists(tensor_path):
            raise JLensError(
                f"expected lens tensor '{tensor_name}' in {folder} — the "
                f"acquisition step verifies this, so a miss here means the cache "
                f"was modified or the table's filename is wrong")
        source = backend_mod.CheckpointLensSource(tensor_path)

    fit, config_hash = _parse_config(config_path)
    if fit.modelID and fit.modelID != model_id:
        raise JLensError(
            f"lens config identifies '{fit.modelID}' but this import is for "
            f"'{model_id}' — refusing a mislabeled artifact")

    layers = source.source_layers
    if not layers:
        raise JLensError("lens has no fitted source layers")
    if layers != list(range(layers[0], layers[-1] + 1)):
        raise JLensError(
            f"lens source layers are not contiguous ({layers[0]}..{layers[-1]} "
            f"with gaps) — refusing an ambiguous layer mapping rather than "
            f"guessing what the missing indices mean")
    # The reference fits sources 0..target-1 and defines transport AT the target
    # as the identity, so the target is one past the last source.
    target = layers[-1] + 1

    # A curated model keeps the id it has always had — study manifests pin
    # lens ids, and the curated folders are all wikitext. An uncurated lens is
    # named by the corpus its own config declares.
    lens_id = (lens_id_for(model_id) if is_curated(model_id)
               else lens_id_for(model_id, fit.corpus))
    directory = paths.jlens_lens_directory(lens_id, root)
    os.makedirs(directory, exist_ok=True)
    converted = convert_to_per_layer(source, directory)

    record = JLensRecord(
        lensID=lens_id,
        source=SourceRef(
            repo=LENS_REPO, folder=entry["folder"], tensorFile=tensor_name,
            configFile=entry["config"],
            commit=os.path.basename(snap.rstrip("/")) or None,
            tensorSHA256=sha256_file(tensor_path) if os.path.exists(tensor_path) else None,
            configSHA256=config_hash),
        fit=fit,
        sourceLayers=layers,
        dModel=int(source.d_model),
        targetLayer=target,
        nPrompts=int(source.n_prompts),
        converted=converted,
        configHash=config_hash,
        referencePackage=backend_mod.REFERENCE_PACKAGE,
        referenceCommit=backend_mod.REFERENCE_COMMIT,
        tier=resolved_tier,
        tierSource=tier_source,
    )
    write_record(record, os.path.join(directory, "lens.json"))
    return record
