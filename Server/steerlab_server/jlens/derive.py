"""Derive a model-depth steering direction for one exact token.

    v_l = J_l^T (g . u_t)

where ``u_t`` is the token's raw output-head row and ``g`` is the model's
final-RMSNorm gain — ``1 + norm.weight`` on an offset-parameterized norm
(Gemma, Qwen3.5, …), ``norm.weight`` on a direct one (Llama, Qwen3, …), as
OBSERVED from the architecture ``config.json`` names (:mod:`norm_convention`;
no weights are loaded for it). The gain fold is not a refinement: measured on
gemma-3-4b-it its weights run ~6.6-9.5, so ``g`` is ~7.6-10.5 and omitting it
would rescale the direction by about an order of magnitude, unevenly per
element (Stage 1a, plan §11.1).

Two properties worth stating because they shape the whole module:

**No resident model.** Derivation needs exactly one embedding row and one norm
vector, which safetensors can slice out of the cached snapshot: ~10 KB read
instead of an 8 GB (4B) or 54 GB (27B) load, in well under a second. Deriving
is therefore a CPU job schedulable anywhere, never a GPU one.

**Fails closed.** If the lens's source layers plus the proven identity target
do not cover the full runtime depth, nothing is written. The consumer-side
guards that would make a partial artifact safe do not exist:
``variant_injections`` silently CLAMPS an out-of-range layer and silently SKIPS
a zero-norm row, so a partial artifact yields a run that completes, reports as
steered, and produces a null indistinguishable from a real no-effect result
(plan §6.1).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

from ..experiment import paths
from ..steering import vector_store
from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar
from . import lens_store
from . import norm_convention as norm_convention_mod
from .schemas import (CANONICAL_READOUT, DIRECTION_CONVENTION, JLensError,
                      JLensRecord)

RECIPE_METHOD = "jlensTokenDirection"
SOURCE = "neuronpedia-jacobian-lens"

#: Prefix on the compatibility ``stimulusSetHash``. A derived direction has no
#: stimulus set, and the cross-engine sidecar requires that field: the Gemma
#: Scope import established the precedent of a prefixed source-recipe identity
#: rather than a schema break, and this follows it. The canonical value is
#: ``derivationIdentityHash``; this field exists so Swift can still decode.
STIMULUS_PREFIX = "jlens:"

_EMBED_KEYS = ("language_model.model.embed_tokens.weight",
               "model.embed_tokens.weight",
               "model.language_model.embed_tokens.weight")
_NORM_KEYS = ("language_model.model.norm.weight",
              "model.norm.weight",
              "model.language_model.norm.weight")
_LM_HEAD_KEYS = ("lm_head.weight", "language_model.lm_head.weight")


def _snapshot_dir(model_id: str, revision: str | None = None) -> str:
    from ..steering.model_loader import hf_hub_dir

    repo = "models--" + model_id.replace("/", "--")
    base = os.path.join(hf_hub_dir(), repo, "snapshots")
    if revision:
        path = os.path.join(base, revision)
        if os.path.isdir(path):
            return path
    found = sorted(glob.glob(os.path.join(base, "*")))
    if not found:
        raise JLensError(
            f"'{model_id}' is not in the local HF cache — install it before "
            f"deriving (the derivation reads two tensors from the snapshot)")
    if revision:
        raise JLensError(
            f"revision {revision!r} of '{model_id}' is not cached; found "
            f"{[os.path.basename(p) for p in found]}")
    return found[-1]


def _weight_map(snapshot: str) -> dict:
    index = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as handle:
            return json.load(handle).get("weight_map", {})
    single = os.path.join(snapshot, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt") as handle:
            return {k: "model.safetensors" for k in handle.keys()}
    raise JLensError(f"no safetensors index or single-file weights in {snapshot}")


def _pick(weight_map: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in weight_map:
            return key
    return None


def observed_norm_convention(model_id: str,
                             revision: str | None = None) -> dict:
    """The final-norm gain convention of the cached snapshot's architecture,
    observed without loading weights (:func:`norm_convention.from_config`)."""
    from . import norm_convention

    return norm_convention.from_config(_snapshot_dir(model_id, revision))


def read_token_row_gain_and_convention(model_id: str, token_id: int,
                                       revision: str | None = None):
    """``(u_t, g, convention)`` read directly from the cached snapshot — no
    model load.

    Gemma **ties** its embeddings, so the output-head row is
    ``embed_tokens.weight[t]`` and ``lm_head.weight`` may not exist as a key at
    all; an untied head is read from ``lm_head.weight`` first. Resolve the tie
    explicitly rather than by a bare lookup that would KeyError on one family
    or the other. Gemma's sqrt(d) input scaling applies to the embedding on
    the way IN and must not be applied here.
    """
    import torch
    from safetensors import safe_open

    from . import norm_convention

    snapshot = _snapshot_dir(model_id, revision)
    weight_map = _weight_map(snapshot)

    embed_key = _pick(weight_map, _LM_HEAD_KEYS) or _pick(weight_map, _EMBED_KEYS)
    if embed_key is None:
        raise JLensError(
            f"no output-head or embedding tensor in {snapshot} — looked for "
            f"{_LM_HEAD_KEYS + _EMBED_KEYS}")
    norm_key = _pick(weight_map, _NORM_KEYS)
    if norm_key is None:
        raise JLensError(
            f"no final-norm tensor in {snapshot} — looked for {_NORM_KEYS}")

    with safe_open(os.path.join(snapshot, weight_map[embed_key]),
                   framework="pt") as handle:
        sliced = handle.get_slice(embed_key)
        vocab_size = sliced.get_shape()[0]
        if not 0 <= token_id < vocab_size:
            raise JLensError(
                f"token id {token_id} out of range for '{model_id}' "
                f"(vocabulary size {vocab_size})")
        u_t = sliced[token_id:token_id + 1][0].to(torch.float32)

    with safe_open(os.path.join(snapshot, weight_map[norm_key]),
                   framework="pt") as handle:
        norm_w = handle.get_tensor(norm_key).to(torch.float32)

    # Which fold the final norm applies is observed from the architecture the
    # snapshot names (Stage 1a probed the live module; this does the same
    # against a weightless instance of its class), never assumed from a name.
    convention = norm_convention.from_config(snapshot)
    gain = norm_convention.gain_from_weight(norm_w, convention["convention"])
    return u_t, gain, convention


def read_token_row_and_gain(model_id: str, token_id: int,
                            revision: str | None = None):
    """``(u_t, g)`` — :func:`read_token_row_gain_and_convention` without the
    stamp, for callers that only need the numbers."""
    u_t, gain, _ = read_token_row_gain_and_convention(model_id, token_id,
                                                      revision)
    return u_t, gain


def runtime_layer_count(model_id: str, revision: str | None = None) -> int:
    snapshot = _snapshot_dir(model_id, revision)
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    text = config.get("text_config") or config
    layers = text.get("num_hidden_layers")
    if not layers:
        raise JLensError(f"could not read num_hidden_layers from {snapshot}")
    return int(layers)


def cached_revision_of(model_id: str) -> str | None:
    from ..steering.model_loader import cached_revision

    return cached_revision(model_id)


def derivation_identity_hash(*, lens_id: str, lens_source_hash: str | None,
                             model_id: str, revision: str | None,
                             token_id: int, direction_convention: str,
                             readout_convention: str) -> str:
    """Canonical identity for a derived direction.

    Everything that changes the BYTES is in here, and nothing that does not.
    This replaces the stimulus-set identity that a stimulus-extracted artifact
    would carry, so verification has something with equivalent force.
    """
    payload = json.dumps({
        "lensID": lens_id,
        "lensSourceSHA256": lens_source_hash,
        "modelID": model_id,
        "revision": revision,
        "tokenID": int(token_id),
        "directionConvention": direction_convention,
        "readoutConvention": readout_convention,
        "recipeMethod": RECIPE_METHOD,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_piece(piece: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in (piece or "")]
    out = "".join(keep).strip("-") or "token"
    return out[:40]


def _artifact_name(name: str | None, piece: str | None, token_id: int) -> str:
    """The artifact filename, with the token id guaranteed present.

    The id is appended here, server-side, and not only in the app's builder: the
    CLI and the route can both name a direction, and an invariant enforced in one
    client is not an invariant. Two directions labelled "courage" could be token
    23648 (" courage") and 236755 ("c"); the suffix is what keeps that visible in
    the name itself, where every downstream consumer can see it.

    Idempotent — a caller that already supplied the suffix does not get it twice.
    """
    suffix = f"-id{token_id}"
    base = (name or "").strip()
    if not base:
        base = f"jlens-token-{safe_piece(piece or '')}"
    if base.endswith(suffix):
        return base
    return base + suffix


def derive_direction(lens_id: str, token_id: int, *, model_id: str,
                     revision: str | None = None, root: str | None = None,
                     name: str | None = None,
                     piece: str | None = None,
                     concept: str | None = None) -> dict:
    """Derive, verify, and persist a model-depth J-lens token direction.

    ``concept`` is an OPTIONAL association, and it is deliberately separate from
    ``name``. The artifact's filename always carries the token id, so the bytes
    stay unambiguous; ``concept`` is only the grouping key that concept-scoped
    views filter on.

    Defaulting it to the name — rather than to a real concept — is the point.
    Filing a one-token direction under a stimulus-defined concept of the same
    label would present them, in every concept-grouped view, as two extractions
    of one thing. They are not: one is a curated stimulus set with held-out
    validation, the other is a single vocabulary token. Doing that silently is
    the plan's last non-goal ("treating a J-lens token direction as evidence
    that a broad psychological concept has been isolated"), so association is a
    choice a researcher makes explicitly or not at all.
    """
    import torch

    record: JLensRecord = lens_store.resolve(lens_id, root)
    revision = revision or cached_revision_of(model_id)
    n_layers = runtime_layer_count(model_id, revision)

    if record.dModel <= 0:
        raise JLensError(f"lens '{lens_id}' has no d_model recorded")
    if record.targetLayer != n_layers - 1:
        raise JLensError(
            f"lens '{lens_id}' targets layer {record.targetLayer} but "
            f"'{model_id}' has {n_layers} layers (target would be "
            f"{n_layers - 1}) — refusing an ambiguous layer mapping")

    u_t, g, gain_convention = read_token_row_gain_and_convention(
        model_id, token_id, revision)
    if u_t.numel() != record.dModel:
        raise JLensError(
            f"token row width {u_t.numel()} != lens d_model {record.dModel}")
    w_t = g * u_t

    rows: dict[int, torch.Tensor] = {}
    for layer in record.sourceLayers:
        j = lens_store.load_layer(record, layer, root=root).to(torch.float32)
        rows[layer] = j.T @ w_t
    # Transport AT the target is the identity by construction, so the effective
    # row IS the direction there. Materialized only because the lens's own
    # target-layer semantics prove it (plan §6.1 step 3).
    rows[record.targetLayer] = w_t.clone()

    defined = sorted(rows)
    if defined != list(range(n_layers)):
        raise JLensError(
            f"lens '{lens_id}' covers layers {defined[0]}..{defined[-1]} "
            f"({len(defined)} of {n_layers}) — refusing to write a partial "
            f"artifact. Injection silently clamps an out-of-range layer and "
            f"silently skips a zero-norm row, so a partial direction would "
            f"produce a clean-looking null instead of an error.")
    for layer, vec in rows.items():
        if not torch.isfinite(vec).all():
            raise JLensError(f"derived row for layer {layer} is not finite")
        if float(vec.norm()) <= 0:
            raise JLensError(f"derived row for layer {layer} is all zeros")

    identity = derivation_identity_hash(
        lens_id=lens_id, lens_source_hash=record.source.tensorSHA256,
        model_id=model_id, revision=revision, token_id=token_id,
        direction_convention=DIRECTION_CONVENTION,
        readout_convention=CANONICAL_READOUT)

    vectors = ConceptVectors(per_layer=[rows[i].tolist() for i in range(n_layers)])
    artifact_name = _artifact_name(name, piece, token_id)
    # Absent an explicit association, the grouping key IS the artifact name, so
    # the direction groups only with itself.
    grouping_concept = (concept or "").strip() or artifact_name
    sidecar = SteeringVectorSidecar(
        modelID=model_id,
        concept=grouping_concept,
        stimulusSetHash=STIMULUS_PREFIX + identity,
        layerCount=vectors.layer_count,
        hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate=record.importedAt,
        revision=revision,
        substrate=vector_store.SUBSTRATE,
        extractionMethod=RECIPE_METHOD,
        recipeMethod=RECIPE_METHOD,
        source=SOURCE,
    )

    run_dir = paths.make_unique_run_directory("jlens-direction", root)
    vector_store.save(vectors, sidecar, run_dir, artifact_name)

    # Additive J-lens provenance beside the ordinary sidecar fields. Written
    # after save() so the base contract is exactly what every other artifact
    # writes and this is a strict extension of it.
    sidecar_path = os.path.join(run_dir, f"{artifact_name}.json")
    with open(sidecar_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.update({
        "derivationIdentityHash": identity,
        "sourceLensID": lens_id,
        "sourceLensCommit": record.source.commit,
        "sourceLensSHA256": record.source.tensorSHA256,
        "sourceLensConfigSHA256": record.source.configSHA256,
        "tokenID": int(token_id),
        "tokenPiece": piece,
        # Recorded so a reader can tell a deliberate association from the
        # default self-grouping without re-deriving the rule.
        "conceptAssociation": (concept or "").strip() or None,
        "availableSourceLayers": [record.sourceLayers[0], record.sourceLayers[-1]],
        "definedLayers": defined,
        "identityTargetLayer": record.targetLayer,
        "coverage": "complete",
        "directionConvention": DIRECTION_CONVENTION,
        "readoutConvention": CANONICAL_READOUT,
        # Observed on the architecture, stamped here so the direction says
        # which gain it folded (an offset/direct mix-up reorders coordinates).
        "finalNormConvention": (
            f"{gain_convention['convention']} "
            f"({norm_convention_mod.describe(gain_convention['convention'])}; "
            f"observed on {gain_convention.get('architecture')})"),
        "finalNormClass": gain_convention.get("className"),
        "referencePackage": record.referencePackage,
        "referenceCommit": record.referenceCommit,
    })
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, sidecar_path)

    return {
        "runDirectory": run_dir,
        "name": artifact_name,
        "concept": grouping_concept,
        "conceptAssociation": (concept or "").strip() or None,
        "vectorArtifactID": os.path.join(run_dir, artifact_name),
        "lensID": lens_id,
        "tokenID": int(token_id),
        "layerCount": vectors.layer_count,
        "definedLayers": defined,
        "identityTargetLayer": record.targetLayer,
        "derivationIdentityHash": identity,
        # Norm-unit alpha needs a pinned neutral-corpus calibration for this
        # exact model/revision/dtype. Absent is a normal state, not a failure:
        # run the ordinary backfill before using alphaInNormUnits (§6.2).
        "residualNormsPresent": False,
    }


#: Files that define tokenizer identity. A readout is indexed by token ID, so a
#: tokenizer change silently re-points every watched token at a different piece
#: — the vocabulary is the readout's coordinate system, not an incidental
#: detail of loading.
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "added_tokens.json")


def tokenizer_identity_hash(model_id: str, revision: str | None = None) -> str | None:
    """SHA-256 over the snapshot's tokenizer files, or None when none exist.

    Order is fixed and content-addressed, so the value is stable across
    machines and depends on nothing but the bytes.
    """
    import hashlib

    snapshot = _snapshot_dir(model_id, revision)
    digest = hashlib.sha256()
    found = False
    for name in _TOKENIZER_FILES:
        path = os.path.join(snapshot, name)
        if not os.path.exists(path):
            continue
        found = True
        digest.update(name.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest() if found else None
