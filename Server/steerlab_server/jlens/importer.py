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

import os

from ..experiment import paths
from . import backend as backend_mod
from .schemas import (ConvertedRef, FitProvenance, JLensError, JLensRecord,
                      SourceRef, sha256_bytes, sha256_file, write_record)

#: Supported lenses, keyed by runtime model id (plan §3.1). Filenames are DATA:
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


def supported_models() -> list[str]:
    return sorted(SUPPORTED)


def entry_for(model_id: str) -> dict:
    try:
        return SUPPORTED[model_id]
    except KeyError:
        raise JLensError(
            f"'{model_id}' has no supported lens — this feature is Gemma-only and "
            f"limited to {supported_models()}. The supported set is data, not a "
            f"constant: add a table row with the exact folder and filenames "
            f"verified against the repository tree.") from None


def lens_id_for(model_id: str) -> str:
    """Stable, filesystem-safe identity for a lens artifact directory."""
    return model_id.replace("/", "--") + "--jlens-wikitext"


def _cached_snapshot(folder: str, *, offline: bool = True) -> str:
    """Resolve the pinned snapshot from the local HF cache.

    Offline by default: import must not silently pull a different snapshot than
    the one acquisition pinned. When nothing is cached the error names the
    acquisition step rather than a hub traceback.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(LENS_REPO, allow_patterns=[folder + "/*"],
                                 local_files_only=offline)
    except Exception as exc:  # noqa: BLE001 — remapped to an actionable error
        raise JLensError(
            f"no cached lens for '{folder}' — acquire it first "
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
                offline: bool = True) -> JLensRecord:
    """Import the published lens for ``model_id`` into the workspace lens store.

    ``source``/``snapshot`` are injection points for tests (a StubBackend and a
    synthetic snapshot directory); production passes neither.
    """
    entry = entry_for(model_id)
    snap = snapshot or _cached_snapshot(entry["folder"], offline=offline)
    folder = os.path.join(snap, entry["folder"])
    tensor_path = os.path.join(folder, entry["tensor"])
    config_path = os.path.join(folder, entry["config"])

    if source is None:
        if not os.path.exists(tensor_path):
            raise JLensError(
                f"expected lens tensor '{entry['tensor']}' in {folder} — the "
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

    lens_id = lens_id_for(model_id)
    directory = paths.jlens_lens_directory(lens_id, root)
    os.makedirs(directory, exist_ok=True)
    converted = convert_to_per_layer(source, directory)

    record = JLensRecord(
        lensID=lens_id,
        source=SourceRef(
            repo=LENS_REPO, folder=entry["folder"], tensorFile=entry["tensor"],
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
    )
    write_record(record, os.path.join(directory, "lens.json"))
    return record
