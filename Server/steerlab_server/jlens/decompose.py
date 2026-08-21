"""Read an existing concept vector back as J-lens token atoms — the support readout.

What this answers: **what vocabulary is this steering vector made of?** A concept
vector is a direction in the residual stream with a label a researcher chose. The
lens supplies a direction per vocabulary token in the same space at the same
layer, so the vector can be expressed as a non-negative sparse combination of
them, and the tokens carrying that combination are a readout of the vector in the
model's own words. It applies retroactively to every vector already in a library
and needs no generation run.

It is also a validity check sharper than cross-concept cosine: a vector labelled
``impartiality`` whose support is affect words has a problem that a cosine matrix
against other concepts would not surface.

Convention is exactly ``derive.py``'s, so a support token and a derived direction
mean the same thing:

    atom_t(l) = J_l^T (g . u_t),   g = 1 + norm.weight,  u_t = tied embed row

Energy is never reported alone (hard rule, enforced by :func:`_layer_report` and
its test)
-----------------------------------------------------------------------------
The reconstructed fraction of a vector's squared norm looks like the headline
number and cannot carry that weight. The atom dictionary is overcomplete by ~100x
(one atom per vocabulary token, in d_model dimensions) and non-negative pursuit
over it reconstructs a *random* direction respectably: measured on gemma-3-4b-it,
real concept vectors reached 3-17% at k=200 while a norm-matched Gaussian and a
coordinate-shuffled twin reached 11-32% through the identical solver
(``docs/spikes/jlens-decomposition-energy-curve.py``, 2026-07-29). A bare energy
figure is therefore uninterpretable, and it happens to land near the published
~6-7% for reasons that have nothing to do with concepts.

So every layer report carries ``nullEnergyFraction`` — a norm-matched random
direction pushed through the same solver at the same budget — and the schema has
no way to express one without the other. The support is the signal; the energy is
context for it.

Two further cautions the same measurement established, recorded here because they
shape how a reader should use the output:

* **The atom cone is narrow.** Embedding rows share a strong common component, so
  the atoms do not span; pursuit runs out of positively-correlated atoms at a few
  hundred. Reconstructible energy has a ceiling unrelated to concepts that
  plausibly moves with d_model, which makes cross-MODEL energy comparison unsafe.
  ``coneExhaustedAt`` reports where it stopped rather than truncating in silence.
* **Readability is layer-dependent** and was late, not middle, at 4B. Support at
  a layer where the lens does not resolve is lexical noise that still looks like
  a list of tokens. Ask for several layers and compare.

Memory
------
The dictionary is never materialized. Correlations against the residual come from
the identity ``atom_t^T r = (g . u_t)^T (J r)``, i.e. ``(U diag(g)) @ (J r)`` — a
[vocab, d] matvec against the gain-scaled head we already read, with no [d, vocab]
product held anywhere. Atom norms take one chunked pass and are discarded per
chunk. Peak is the head itself (~2.7 GB at 4B, ~5.6 GB at 27B in float32) rather
than twice it, which is what lets this run on a Mac at all.
"""

from __future__ import annotations

import hashlib
import json
import os

from ..experiment import paths
from .schemas import JLensError, JLensRecord, sha256_file
from . import derive as derive_mod
from . import lens_store

ARTIFACT_TYPE = "jlens-support"
SCHEMA_VERSION = 1

#: Default sparsity budget. 25 is the occupancy the workspace paper reports, so
#: it is the honest default rather than a budget picked to flatter a curve.
DEFAULT_BUDGET = 25

#: Fixed seed for the matched-norm random null. The null must be reproducible
#: from the record alone, and it is stamped as ``nullSeed``.
NULL_SEED = 20260729

_VOCAB_CHUNK = 32768


def _device(prefer: str | None = None):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------

def _refine(gram, rhs, target_sq: float, coeffs, iterations: int = 600):
    """FISTA with non-negative projection on a fixed support.

    Guarded on both ends. The Lipschitz constant comes from a power iteration on
    the support Gram — MPS has no SVD, and the spectral norm is what the step
    size needs; using the Gram DIAGONAL instead (~1 for unit atoms) diverges the
    moment the support is correlated, which in a 100x overcomplete dictionary is
    always. And the result is accepted only if it reduced the objective, so a
    refinement can never return something worse than it was handed.
    """
    import torch

    probe = torch.ones(gram.shape[0], device=gram.device, dtype=gram.dtype)
    for _ in range(200):
        probe = gram @ probe
        norm = probe.norm()
        if not torch.isfinite(norm) or norm <= 0:
            return coeffs.clamp_min(0)
        probe = probe / norm
    lipschitz = float((probe @ (gram @ probe)).abs()) * 1.2
    step = 1.0 / (lipschitz + 1e-9)

    def objective(x) -> float:
        return float(x @ (gram @ x) - 2.0 * (rhs @ x)) + target_sq

    best = coeffs.clamp_min(0)
    best_value = objective(best)
    x, y, t = best.clone(), best.clone(), 1.0
    for _ in range(iterations):
        x_next = (y - step * (gram @ y - rhs)).clamp_min(0)
        if not torch.isfinite(x_next).all():
            break
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        x, t = x_next, t_next
        value = objective(x)
        if value < best_value:
            best, best_value = x.clone(), value
    return best


def _pursue(target, budget: int, correlate, atoms_for):
    """Non-negative matching pursuit, then one refinement over the final support.

    ``correlate(residual)`` returns unit-atom correlations for the whole
    vocabulary; ``atoms_for(indices)`` materializes just the chosen columns.
    Matching pursuit (rather than a gradient step across the whole support) keeps
    the residual monotonically non-increasing without needing a step size at all.

    Returns ``(indices, coefficients, energy_fraction, exhausted_at)``.
    """
    import torch

    residual = target.clone()
    total = float(target.dot(target))
    if total <= 0:
        raise JLensError("vector is all zeros at this layer — nothing to decompose")
    chosen: list[int] = []
    coefficients: list[float] = []
    exhausted_at: int | None = None
    seen: set[int] = set()

    for step in range(1, budget + 1):
        correlation = correlate(residual)
        if seen:
            correlation[torch.tensor(sorted(seen), device=correlation.device)] = \
                float("-inf")
        best = int(torch.argmax(correlation))
        value = float(correlation[best])
        if not value > 0:
            exhausted_at = step
            break
        seen.add(best)
        chosen.append(best)
        coefficients.append(value)
        residual = residual - value * atoms_for([best])[:, 0]

    if not chosen:
        return [], [], 0.0, exhausted_at

    columns = atoms_for(chosen)
    coeffs = torch.tensor(coefficients, device=columns.device, dtype=columns.dtype)
    refined = _refine(columns.T @ columns, columns.T @ target, total, coeffs)
    residual = target - columns @ refined
    fraction = 1.0 - float(residual.dot(residual)) / total

    # A quantity bounded in [0,1] by construction. Outside it means the solver
    # diverged, and a milder divergence than the one that produced -3.3e23 during
    # development would have looked exactly like data.
    if not 0.0 <= fraction <= 1.0 + 1e-6:
        raise JLensError(
            f"reconstructed fraction {fraction} is outside [0,1] — the solver "
            f"diverged and the readout would be meaningless")
    order = sorted(range(len(chosen)), key=lambda i: -float(refined[i]))
    return ([chosen[i] for i in order], [float(refined[i]) for i in order],
            min(fraction, 1.0), exhausted_at)


# --------------------------------------------------------------------------
# lens algebra
# --------------------------------------------------------------------------

class _LayerAtoms:
    """Unit-norm atom access at one layer, without materializing the dictionary."""

    def __init__(self, jacobian, head, device):
        import torch

        self._j = jacobian.to(device=device, dtype=torch.float32)
        self._head = head                      # [vocab, d], gain already folded
        self._device = device
        self._norms = self._atom_norms()

    def _atom_norms(self):
        import torch

        vocab = self._head.shape[0]
        norms = torch.empty(vocab, device=self._device, dtype=torch.float32)
        for start in range(0, vocab, _VOCAB_CHUNK):
            block = self._head[start:start + _VOCAB_CHUNK].to(self._device)
            # Row t of ``block @ J`` is ``w_t^T J`` = ``(J^T w_t)^T`` = the atom.
            # Using ``J`` transposed here instead computes ``J w_t``, which is a
            # different vector: the norms would then disagree with the
            # correlations, and the refinement zeroes out everything it cannot
            # reconcile.
            norms[start:start + _VOCAB_CHUNK] = (block @ self._j).norm(dim=1)
            del block
        return norms

    def correlate(self, residual):
        """Unit-atom correlations for every token: ``(U diag(g)) @ (J r) / ||atom||``."""
        import torch

        projected = self._j @ residual.to(self._device)
        vocab = self._head.shape[0]
        out = torch.empty(vocab, device=self._device, dtype=torch.float32)
        for start in range(0, vocab, _VOCAB_CHUNK):
            block = self._head[start:start + _VOCAB_CHUNK].to(self._device)
            out[start:start + _VOCAB_CHUNK] = block @ projected
            del block
        # Degenerate atoms (a zero embedding row, or one the Jacobian annihilates)
        # are unreachable rather than infinitely attractive.
        safe = self._norms > 0
        out[safe] /= self._norms[safe]
        out[~safe] = float("-inf")
        return out

    def columns(self, indices: list[int]):
        """The chosen atoms as unit-norm columns ``[d, len(indices)]``."""
        import torch

        rows = self._head[torch.tensor(indices)].to(self._device)
        columns = (rows @ self._j).T           # column i = J^T w_t, as in derive
        return columns / self._norms[torch.tensor(indices, device=self._device)]


def gain_scaled_head(model_id: str, revision: str | None):
    """``U diag(g)`` as ``[vocab, d]`` — one read, shared across every layer.

    Delegates the tie/gain conventions to :mod:`derive` rather than restating
    them: Gemma ties its embeddings so the output row IS the embedding row,
    RMSNorm is ``1 + weight``, and the sqrt(d) input scaling must not appear.
    """
    import torch
    from safetensors import safe_open

    snapshot = derive_mod._snapshot_dir(model_id, revision)
    weight_map = derive_mod._weight_map(snapshot)
    embed_key = (derive_mod._pick(weight_map, derive_mod._LM_HEAD_KEYS)
                 or derive_mod._pick(weight_map, derive_mod._EMBED_KEYS))
    norm_key = derive_mod._pick(weight_map, derive_mod._NORM_KEYS)
    if embed_key is None or norm_key is None:
        raise JLensError(
            f"no output-head or final-norm tensor in {snapshot} — this is the "
            f"same read {derive_mod.RECIPE_METHOD} performs, so a miss here "
            f"means the snapshot is incomplete")
    with safe_open(os.path.join(snapshot, weight_map[embed_key]),
                   framework="pt") as handle:
        head = handle.get_tensor(embed_key).to(torch.float32)
    with safe_open(os.path.join(snapshot, weight_map[norm_key]),
                   framework="pt") as handle:
        gain = 1.0 + handle.get_tensor(norm_key).to(torch.float32)
    if head.shape[1] != gain.numel():
        raise JLensError(
            f"head width {head.shape[1]} != final-norm width {gain.numel()}")
    return head * gain


# --------------------------------------------------------------------------
# the readout
# --------------------------------------------------------------------------

def support_identity_hash(*, lens_id: str, lens_source_hash: str | None,
                          model_id: str, revision: str | None,
                          vector_sha256: str | None, layers: list[int],
                          budget: int) -> str:
    """Canonical identity for a support readout — everything that moves the numbers."""
    payload = json.dumps({
        "lensID": lens_id, "lensSourceSHA256": lens_source_hash,
        "modelID": model_id, "revision": revision,
        "vectorSHA256": vector_sha256, "layers": sorted(layers),
        "budget": int(budget), "nullSeed": NULL_SEED,
        "artifactType": ARTIFACT_TYPE, "schemaVersion": SCHEMA_VERSION,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _layer_report(layer: int, *, pieces, indices, coefficients, fraction,
                  null_fraction: float, exhausted_at, null_exhausted_at) -> dict:
    """One layer's readout. ``nullEnergyFraction`` is a required argument on
    purpose — the schema cannot express an energy figure without its control."""
    total = sum(coefficients) or 1.0
    return {
        "layer": layer,
        "support": [
            {"tokenID": int(token), "piece": piece, "coefficient": coefficient,
             # Share of the reconstructed direction's length. Atoms are unit-norm,
             # so a coefficient IS a length and the shares are comparable.
             "share": coefficient / total}
            for token, piece, coefficient in zip(indices, pieces, coefficients)],
        "energyFraction": fraction,
        "nullEnergyFraction": null_fraction,
        # Stated rather than left to be computed, because the whole point is that
        # nobody reads the fraction without it.
        "energyOverNull": fraction - null_fraction,
        "coneExhaustedAt": exhausted_at,
        "nullConeExhaustedAt": null_exhausted_at,
    }


def decompose(*, lens_id: str, vector_directory: str, vector_name: str,
              layers: list[int] | None = None, budget: int = DEFAULT_BUDGET,
              root: str | None = None, device: str | None = None,
              tokenizer=None, progress=None) -> dict:
    """Decompose a stored concept vector into J-lens token atoms, layer by layer.

    ``vector_directory``/``vector_name`` are the catalog's ``(runDirectory, name)``
    pair. Layers default to every fitted source layer of the lens, which is the
    honest default: readability varies by layer and picking one for the caller
    would hide that.
    """
    import torch

    record: JLensRecord = lens_store.resolve(lens_id, root)
    sidecar_path = os.path.join(vector_directory, f"{vector_name}.json")
    vectors_path = os.path.join(vector_directory, f"{vector_name}.safetensors")
    if not (os.path.isfile(sidecar_path) and os.path.isfile(vectors_path)):
        raise JLensError(f"no vector artifact '{vector_name}' in {vector_directory}")
    with open(sidecar_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)

    model_id = sidecar.get("modelID")
    revision = sidecar.get("revision")
    if not model_id:
        raise JLensError(f"'{vector_name}' has no modelID — cannot pick a lens for it")
    if record.fit.modelID and record.fit.modelID != model_id:
        raise JLensError(
            f"lens '{lens_id}' was fitted on '{record.fit.modelID}' but the "
            f"vector was extracted on '{model_id}' — atoms and vector would live "
            f"in different spaces")
    if record.dModel and sidecar.get("hiddenSize") not in (None, record.dModel):
        raise JLensError(
            f"lens d_model {record.dModel} != vector hidden size "
            f"{sidecar.get('hiddenSize')}")
    substrate = sidecar.get("substrate")
    if substrate not in (None, "python-hf-transformers"):
        raise JLensError(
            f"'{vector_name}' was extracted by '{substrate}'. Activations do not "
            f"transfer across substrates and the J-lens is Python-only, so its "
            f"atoms cannot be compared with an MLX-extracted direction.")

    wanted = sorted(set(layers)) if layers else list(record.sourceLayers)
    unknown = [layer for layer in wanted if layer not in record.sourceLayers]
    if unknown:
        raise JLensError(
            f"layers {unknown} are not fitted source layers of '{lens_id}' "
            f"(have {record.sourceLayers[0]}..{record.sourceLayers[-1]}; the "
            f"target layer {record.targetLayer} has no Jacobian by construction)")
    if budget < 1:
        raise JLensError("budget must be at least 1")

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    target_device = _device(device)
    head = gain_scaled_head(model_id, revision)
    generator = torch.Generator().manual_seed(NULL_SEED)

    from safetensors import safe_open

    reports: list[dict] = []
    with safe_open(vectors_path, framework="pt", device="cpu") as handle:
        for layer in wanted:
            key = f"layer_{layer}"
            try:
                vector = handle.get_tensor(key).to(torch.float32)
            except Exception as exc:  # noqa: BLE001 — remapped to an actionable error
                raise JLensError(
                    f"'{vector_name}' has no {key} — it has "
                    f"{sidecar.get('layerCount')} layers") from exc
            if progress is not None:
                progress(f"decomposing layer {layer} of {wanted[-1]}")

            atoms = _LayerAtoms(lens_store.load_layer(record, layer, root=root),
                                head, target_device)
            target = vector.to(target_device)
            indices, coefficients, fraction, exhausted = _pursue(
                target, budget, atoms.correlate, atoms.columns)

            # The matched-norm random null, through the identical solver at the
            # identical budget. Not optional: see the module docstring.
            noise = torch.randn(vector.numel(), generator=generator)
            noise = (noise * (vector.norm() / noise.norm())).to(target_device)
            _, _, null_fraction, null_exhausted = _pursue(
                noise, budget, atoms.correlate, atoms.columns)

            pieces = tokenizer.convert_ids_to_tokens(indices) if indices else []
            reports.append(_layer_report(
                layer, pieces=pieces, indices=indices, coefficients=coefficients,
                fraction=fraction, null_fraction=null_fraction,
                exhausted_at=exhausted, null_exhausted_at=null_exhausted))
            del atoms
            if target_device.type == "mps":
                torch.mps.empty_cache()
            elif target_device.type == "cuda":
                torch.cuda.empty_cache()

    vector_sha = sha256_file(vectors_path)
    return {
        "artifactType": ARTIFACT_TYPE,
        "schemaVersion": SCHEMA_VERSION,
        "lensID": record.lensID,
        "sourceLensSHA256": record.source.tensorSHA256,
        "sourceLensCommit": record.source.commit,
        "lensFitPrompts": record.nPrompts,
        "lensFitCorpus": record.fit.corpus,
        "modelID": model_id,
        "revision": revision,
        "vector": {
            "runDirectory": vector_directory,
            "name": vector_name,
            "concept": sidecar.get("concept"),
            "sha256": vector_sha,
            "extractionMethod": sidecar.get("extractionMethod"),
            "recipeMethod": sidecar.get("recipeMethod"),
        },
        "budget": int(budget),
        "nullSeed": NULL_SEED,
        "directionConvention": record.directionConvention,
        "solver": "non-negative matching pursuit + projected-gradient refinement",
        "device": target_device.type,
        "supportIdentityHash": support_identity_hash(
            lens_id=record.lensID, lens_source_hash=record.source.tensorSHA256,
            model_id=model_id, revision=revision, vector_sha256=vector_sha,
            layers=wanted, budget=budget),
        "layers": reports,
    }


def write_readout(readout: dict, root: str | None = None) -> str:
    """Persist a readout into its own run directory and return the directory."""
    from ..experiment.run_config import write_run_config

    name = readout["vector"]["name"]
    run_dir = paths.make_unique_run_directory(f"jlens-support-{derive_mod.safe_piece(name)}",
                                              root)
    write_run_config(run_dir, "jlens-support", model_id=readout.get("modelID"),
                     revision=readout.get("revision"),
                     notes={"lensID": readout.get("lensID"),
                            "budget": readout.get("budget"),
                            "supportIdentityHash": readout.get("supportIdentityHash")})
    with open(os.path.join(run_dir, "support.json"), "w", encoding="utf-8") as handle:
        json.dump(readout, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csv(readout, os.path.join(run_dir, "support.csv"))
    return run_dir


def _write_csv(readout: dict, path: str) -> None:
    """Flat table — one row per (layer, support token), for reading anywhere."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "rank", "tokenID", "piece", "coefficient",
                         "share", "energyFraction", "nullEnergyFraction"])
        for report in readout["layers"]:
            for rank, row in enumerate(report["support"], start=1):
                writer.writerow([report["layer"], rank, row["tokenID"],
                                 row["piece"], f"{row['coefficient']:.6f}",
                                 f"{row['share']:.6f}",
                                 f"{report['energyFraction']:.6f}",
                                 f"{report['nullEnergyFraction']:.6f}"])
