"""Position-resolved J-space readout over a prompt — one condition, one pass.

The question this answers is the one the online recorder cannot: **how does a
single agent's J-space evolve as it reads?** The recorder discards every prompt
position but the last, by design — its frame is prediction alignment, and
prompt positions predict tokens nobody sampled. So a within-prompt trajectory
has to come from a forward pass over the realized text, which is exactly what
teacher-forced reconstruction is, and which is faithful for the reason
``recorder.py`` now states.

**One condition is the first-class case**, not a degenerate comparison. "What
is in the baseline agent's J-Space while it works through one of the study's
own items" is a whole question by itself, and the axis it varies along —
position — is a *within-token* axis:
``logit_t = ‖z‖ · ‖g ⊙ u_t‖ · cos(z, g ⊙ u_t)``, and the per-token norm is
constant across positions, so tracking ONE token across the prompt is clean
with no control condition at all. Tracking DIFFERENT tokens against each other
at one position is not, which is why this reports per-token trajectories rather
than cross-token rankings.

Three readouts, deliberately layered from most to least committed
-----------------------------------------------------------------
* **Direction projections** (``--directions``) — cosine against a concept
  vector you already extracted, at each position. No vocabulary, no token
  list, no discretization; the least committed and the most defensible answer
  to "when did X appear". This is the paper's own concept score (an inner
  product against a J-lens vector, no full-vocabulary ranking).
* **Pinned-token ranks** (``--pin``) — full-vocabulary rank of a few chosen
  tokens across the prompt. The paper's usage, and robust: rank holds the
  token's own unembedding norm constant, so its movement is alignment.
  Emphatically NOT a watchlist averaged into a construct score.
* **Top-k** — what is up there, per position and layer. The discovery view.
  Read it knowing the top of this distribution is tilted toward tokens with
  large unembedding norms (including untrained ones), which is why the ranks
  above are the measurement and this is the illustration.

Every J-lens number carries its **logit-lens companion** (the same readout with
``J_l`` set to the identity). For a single-agent trajectory that stops being a
nicety and becomes the primary control available: if lens and companion move
together you are reading the token that is locally present, not what the model
is poised to say later.

Bounded by construction — N prompts × armed layers × positions — so it sits
outside the study's trace budget, and captures through hooks on the armed
layers rather than ``output_hidden_states`` (62 layers × 2000 positions ×
5376 × 4 B ≈ 2.7 GB at 27B; four armed layers is ~170 MB).

Server-only, Gemma-only (CLAUDE.md, hard requirement).
"""

from __future__ import annotations

import csv
import json
import os

from ..experiment import model_variant as _model_variant
from ..steering.intervention import LayerIntervention
from .schemas import JLensError

RUN_TYPE = "jlens-probe"
PROBE_JSON = "probe.json"
TOPK_CSV = "probe-topk.csv"
TRAJECTORY_CSV = "probe-trajectory.csv"
SCHEMA_VERSION = 1

#: Read every Nth position. 1 is every position; larger strides make a long
#: case affordable when the shape matters more than the resolution.
DEFAULT_POSITION_STRIDE = 1

#: Full-vocabulary projections are the cost here, one per (armed layer,
#: scored position) and doubled by the companion. Refused above this rather
#: than discovered as a wedged node — the same reasoning as the readout
#: Budget's compute ceiling, scaled for a bounded diagnostic.
MAX_PROJECTIONS = 200_000

#: Adapter weight filenames, by the format that writes them. Swift's fine-tune
#: panel writes MLX's ``adapters.safetensors``; the server's LoRA trainer
#: writes PEFT's ``adapter_model.safetensors``. A verifier that knew only one
#: would refuse the other engine's agents — and one that GUESSED between them
#: could hash the MLX file while the PEFT loader reads the other (external
#: review round 5).

#: Re-exported from `model_variant`, which owns the adapter-identity rules
#: now that the probe and the study-run path share one verifier. Kept under
#: these names so existing importers and stamped artifacts are unaffected.
ADAPTER_CONTENT_HASH_ALGORITHM = _model_variant.ADAPTER_CONTENT_HASH_ALGORITHM
ADAPTER_CONTENT_HASH_SPEC = _model_variant.ADAPTER_CONTENT_HASH_SPEC


class ProbeError(JLensError):
    """A probe that cannot be run honestly is not run."""


class _AllPositionCapture(LayerIntervention):
    """Read-only capture of every position at the armed layers.

    Returns the hidden state unchanged and never mutates in place — the same
    contract the online recorder holds, for the same reason: a capture that
    perturbed the forward pass would be reading a state the model would not
    otherwise have had.
    """

    def __init__(self, layers):
        self.layers = {int(x) for x in layers}
        self.rows: dict[int, object] = {}

    def apply(self, h, layer: int, offset: int):
        if layer in self.layers:
            import torch

            with torch.no_grad():
                self.rows[layer] = h[0].detach().to(torch.float32).cpu()
        return h


def resolve_pins(tokenizer, words=(), token_ids=()) -> dict[int, str]:
    """``{token id: piece}`` for the pinned tokens.

    A word is resolved only when it is a SINGLE token in its leading-space
    form, and refused otherwise with the components named. Resolving a
    multi-token word by picking a piece is the silent mis-selection
    ``token-options`` exists to prevent: "sympathy" splits into 'sym' +
    'pathy' on this tokenizer, and a direction or a rank trajectory for
    'pathy' would be labelled sympathy everywhere downstream.
    """
    pinned: dict[int, str] = {}
    for raw in words:
        word = raw if raw.startswith(" ") else " " + raw
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            pieces = [tokenizer.decode([int(i)]) for i in ids]
            raise ProbeError(
                f"{word!r} is {len(ids)} tokens ({pieces}) on this tokenizer, "
                f"not one — pin an exact id with --pin-id instead of letting "
                f"this choose a fragment for you (steerlab-server jlens "
                f"token-options <model> {raw.strip()!r} lists them)")
        pinned[int(ids[0])] = word
    for token in token_ids:
        pinned.setdefault(int(token), tokenizer.decode([int(token)]))
    return pinned


def _sha256_file(path: str) -> str | None:
    import hashlib

    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def load_directions(references, root=None) -> list[dict]:
    """Concept vectors to project the residual onto, PINNED TO THEIR BYTES.

    A locator is not an identity. Vector artifacts live in mutable library
    subtrees (``runs/model-variants/`` and friends are editable in place by
    rule), so the same ``<runDir>/<name>`` can hold different numbers next
    week — and a trajectory computed against the new bytes would report the
    old provenance. The tensor and sidecar hashes travel into the artifact
    alongside the model binding and recipe identity, so a reader can tell
    which direction produced a curve (external review, 2026-08-16).
    """
    from ..experiment import paths
    from ..steering import vector_store

    out: list[dict] = []
    for reference in references:
        resolved = paths.resolve_artifact(reference, root)
        try:
            vectors, sidecar = vector_store.load(
                os.path.dirname(resolved), os.path.basename(resolved))
        except Exception as exc:  # noqa: BLE001 - remapped to a typed refusal
            raise ProbeError(
                f"could not load direction {reference!r}: {exc}") from exc
        out.append({
            "reference": reference, "name": os.path.basename(resolved),
            "vectors": vectors, "sidecar": sidecar,
            "concept": sidecar.concept,
            "extractionMethod": sidecar.extractionMethod,
            "recipeMethod": getattr(sidecar, "recipeMethod", None),
            "modelID": sidecar.modelID, "revision": sidecar.revision,
            "stimulusSetHash": sidecar.stimulusSetHash,
            "tensorSHA256": _sha256_file(resolved + ".safetensors"),
            "sidecarSHA256": _sha256_file(resolved + ".json"),
        })
    return out


def _require_hash(value, what: str):
    """A provenance block full of ``null`` looks pinned and is not.

    Refusing is the only honest option: the caller asked for a trajectory
    attributable to an agent, and an unhashable input means it cannot be
    attributed (external review round 3).
    """
    if not value:
        raise ProbeError(
            f"could not hash {what} — a probe records provenance so a "
            f"trajectory can be attributed to exact bytes, and an unpinnable "
            f"input would leave a provenance block that LOOKS pinned while "
            f"containing null")
    return value


def downgrade_for_unpinned_agent(stamp: str, claim: str, identity: dict):
    """A qualified LENS does not make a report over an UNPINNED AGENT citable.

    Qualification decides whether the readout is trustworthy. It says nothing
    about the agent being read: an unpinned `adapter_config.json` can change
    rank, target modules, scaling — which layers the adapter even touches —
    while the agent's declared identity stays byte-identical. So the claim is
    the WEAKER of the two, not the lens's alone (external review round 6).

    This is the enforcement that bites for the configHash gap; freeze only
    advises, because the field is recomputable and gating it would push
    existing studies onto `--force`.
    """
    unpinned = [a.get("adapterDirectory", "?")
                for a in identity.get("adapters") or []
                if a.get("configurationUnpinned")]
    if not unpinned or claim != "qualified":
        return stamp, claim
    return (f"{stamp} — DOWNGRADED: this agent's adapter configuration is "
            f"unpinned ({', '.join(unpinned)}), so the condition being read "
            f"is not pinned even though the lens is"), "exploratory"


def _variant_identity(variant, path: str, root=None) -> dict:
    """Everything that makes this agent THIS agent, by hash.

    The variant JSON, every injection's vector artifact, the neutral basis the
    projection uses, and the adapter CONTENT. A trajectory is a function of all
    of them, so each is hashed from the bytes on disk rather than copied from
    the variant's own declaration — a declared ``adapterHash`` is a claim about
    an adapter, not a measurement of the one that will be loaded, and a
    retrained adapter written to the same path keeps the claim.
    """
    from ..experiment import model_variant, paths

    out: dict = {
        "variantSHA256": _require_hash(_sha256_file(path), f"variant {path!r}"),
        "baseModelID": getattr(variant, "base_model_id", None),
        "baseRevision": getattr(variant, "base_revision", None),
        "alphaInNormUnits": getattr(variant, "alpha_in_norm_units", None),
    }

    # ONE implementation of "what is this adapter, measured", shared with the
    # study-run path (external review round 8). It reads the adapters as the
    # dicts they are and hashes the live files; a mismatch refuses.
    try:
        adapters = model_variant.verified_adapter_identity(variant, root)
    except model_variant.AdapterIdentityError as exc:
        raise ProbeError(str(exc)) from exc
    out["adapters"] = adapters

    basis = getattr(variant, "neutral_pc_basis_path", None)
    if basis:
        # Resolved the way EXECUTION resolves it (resolve_artifact, which
        # rebases a reference recorded on another machine), and hashed at the
        # file the basis actually is — the path may name a directory, and
        # hashing a directory as a file silently yields None.
        resolved = paths.resolve_artifact(basis, root)
        target = resolved
        if os.path.isdir(resolved):
            target = os.path.join(resolved, "neutral-pc-basis.json")
        out["neutralPCBasisPath"] = basis
        out["neutralPCBasisSHA256"] = _require_hash(
            _sha256_file(target), f"neutral basis {basis!r} (at {target!r})")

    vectors = []
    for cell in (getattr(variant, "injections", None) or []):
        reference = (cell or {}).get("vectorArtifactID") if isinstance(cell, dict) \
            else getattr(cell, "vectorArtifactID", None)
        if not reference:
            continue
        resolved = paths.resolve_artifact(reference, root)
        vectors.append({
            "reference": reference,
            "tensorSHA256": _require_hash(
                _sha256_file(resolved + ".safetensors"),
                f"injection vector {reference!r}"),
            "sidecarSHA256": _require_hash(
                _sha256_file(resolved + ".json"),
                f"injection sidecar {reference!r}"),
        })
    out["injectionVectors"] = vectors
    return out


def _require_direction_model(directions, model_id: str, revision) -> None:
    """A direction extracted on another model is not a direction here.

    Residual bases are model-specific: projecting a gemma-3-4b vector onto a
    27B residual is a shape error at best and a meaningless cosine at worst.
    Revision mismatch is a WARNING's worth of risk, not a refusal, so only the
    model id is enforced.
    """
    wrong = [d for d in directions
             if d["modelID"] and d["modelID"] != model_id]
    if wrong:
        detail = ", ".join(f"{d['name']} ({d['modelID']})" for d in wrong)
        raise ProbeError(
            f"direction(s) {detail} were extracted on a different model than "
            f"{model_id} — residual bases are model-specific, so the cosine "
            f"would be a number about nothing")


def _cosine(a, b) -> float:
    import torch

    denom = float(a.norm()) * float(b.norm())
    return float(torch.dot(a, b)) / denom if denom > 0 else 0.0


def probe(model_id: str, *, prompt: str, lens_id: str | None = None,
          layers: list[int] | None = None, top_k: int = 10,
          pin_words=(), pin_ids=(), directions=(),
          variant_path: str | None = None,
          prompt_mode: str | None = None, system_prompt: str | None = None,
          position_stride: int = DEFAULT_POSITION_STRIDE,
          max_tokens: int | None = None, revision: str | None = None,
          root: str | None = None, model=None, device: str | None = None,
          dtype: str | None = None, log=None) -> dict:
    """Read one CONDITION's prompt at every (armed layer × position).

    The condition is an effective one, not a bare model: ``variant_path``
    supplies the agent (its adapter, its stored injections, its prompt mode
    and system prompt), and the prompt goes through the SHARED renderer.

    That is not decoration. Rendering is what a study run does, and a raw
    ``tokenizer(prompt)`` produces a different token sequence than the chat
    template — different length, different positions, a different BOS story —
    so a trajectory read off raw text is not about any run's positions
    (external review, 2026-08-16). Absent a variant this reads the BASE model
    under the declared rendering, which is a legitimate condition and is
    stamped as such rather than left to look like an agent.
    """
    import torch

    from ..experiment import paths, prompt_render
    from ..experiment.run_config import write_run_config
    from ..experiment.optvec_jspace import evidence_tier_for, qualification_state
    from ..steering import model_loader
    from . import importer, lens_store
    from .qualification import _default_layers
    from .readout import LensReadout, ReadoutConfig

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    if not (prompt or "").strip():
        raise ProbeError("a probe needs a prompt to read")

    lens = lens_id or importer.lens_id_for(model_id)
    record = lens_store.resolve(lens, root)
    if model is None:
        emit(f"loading {model_id} …")
        model = model_loader.load(model_id, revision, dtype=dtype,
                                  device=device)
    revision = revision or getattr(model, "revision", None)

    armed = sorted(set(layers)) if layers else _default_layers(record)
    unknown = [l for l in armed if l not in record.sourceLayers]
    if unknown:
        raise ProbeError(
            f"layers {unknown} are not fitted source layers of '{lens}' "
            f"(have {record.sourceLayers[0]}..{record.sourceLayers[-1]})")

    tokenizer = model.tokenizer
    pinned = resolve_pins(tokenizer, pin_words, pin_ids)
    loaded_directions = load_directions(directions, root)
    _require_direction_model(loaded_directions, model_id, revision)

    # The effective condition: agent first, then the rendering it declares.
    variant = injections = adapter = None
    variant_identity: dict = {}
    if variant_path:
        from ..experiment import model_variant

        resolved_variant = paths.resolve(variant_path, root)
        variant = model_variant.ModelVariant.from_file(resolved_variant)
        # The agent gets the same treatment the directions do: an artifact
        # locator is not an identity, and variant artifacts live in a
        # deliberately MUTABLE library subtree (runs/model-variants/), so the
        # same path can hold a different agent next week (external review
        # round 2). Editing a variant or the vectors it references would
        # otherwise change a trajectory with indistinguishable provenance.
        variant_identity = _variant_identity(variant, resolved_variant, root)
        base = getattr(variant, "base_model_id", None)
        if base and base != model_id:
            raise ProbeError(
                f"variant {variant.name!r} was built for {base}, not "
                f"{model_id} — its injections live in that model's residual "
                f"basis and mean nothing in this one")
        injections = model_variant.variant_injections(variant, root=root)
        prompt_mode = prompt_mode or variant.prompt_mode
        if system_prompt is None:
            system_prompt = variant.system_prompt
    rendered = prompt_render.render(
        tokenizer, prompt, model_id=model_id,
        prompt_mode=prompt_mode or prompt_render.CHAT_ASSISTANT,
        system_prompt=system_prompt,
        qwen_thinking_enabled=bool(
            getattr(variant, "qwen_thinking_enabled", False)))
    ids = list(rendered.input_ids)
    if max_tokens and len(ids) > max_tokens:
        ids = ids[:max_tokens]
    encoded = {"input_ids": torch.tensor([ids])}
    scored = list(range(0, len(ids), max(1, position_stride)))
    # A vocabulary readout is wanted only when something reads the vocabulary.
    # A direction-only probe needs J_l and never touches the output head, so
    # it costs ZERO full-vocabulary projections — the previous arithmetic
    # charged it two per cell for a readout nobody asked for.
    wants_vocabulary = bool(pinned) or top_k > 0
    if not (wants_vocabulary or loaded_directions):
        raise ProbeError(
            "nothing to read: pass --pin/--pin-id, --top-k, or --directions")
    # ONE projection per (cell, lens) and one per (cell, companion); top-k and
    # pinned ranks are both derived from it. Counting two while computing four
    # is what this ceiling exists to prevent (external review, 2026-08-16).
    projections = (len(scored) * len(armed) * 2) if wants_vocabulary else 0
    if projections > MAX_PROJECTIONS:
        raise ProbeError(
            f"{projections} full-vocabulary projections "
            f"({len(scored)} position(s) × {len(armed)} layer(s), doubled by "
            f"the companion) exceeds {MAX_PROJECTIONS} — raise --stride, arm "
            f"fewer layers, or truncate with --max-tokens. Lowering --top-k "
            f"does NOT reduce this: k selects from the result, it does not "
            f"avoid the matmul. A --directions-only probe costs none of it")

    config = ReadoutConfig(layers=armed, watchlist=sorted(pinned) or [],
                           topK=max(0, top_k), topKLayers=armed,
                           logitLensCompanion=True,
                           transportOnly=not wants_vocabulary)
    readout = LensReadout.build(record=record, config=config, model=model,
                                root=root)

    emit(f"reading {len(scored)} position(s) × {len(armed)} layer(s) "
         f"({projections} projections)")
    capture = _AllPositionCapture(armed)
    device_obj = next(model.model.parameters()).device
    # Injectors FIRST, the read-only capture last — the standing order, so at
    # an injection layer the capture observes the post-intervention residual
    # and at later layers the downstream one.
    #
    # Injection here uses the deployed gate (prompt_token_count = the prompt's
    # own length), which fires at the FINAL prompt position only. That is what
    # a study run does during prefill, and it is why a steered agent's earlier
    # positions are identical to baseline — a fact this artifact stamps rather
    # than leaves to be rediscovered.
    session_hooks = []
    if injections:
        from ..steering.plan import Edit, Mode, interventions as build_chain

        session_hooks = build_chain(
            [Edit(layer=cell.layer, vector=cell.vector, strength=cell.alpha,
                  mode=Mode(cell.mode), concept=cell.concept)
             for cell in injections],
            prompt_token_count=len(ids))
    if variant is not None:
        from ..experiment import model_variant

        adapter = model_variant.apply_adapter(model, variant, root=root)
    try:
        with torch.no_grad():
            model.hooked.reset_offsets()
            with model.hooked.session(session_hooks + [capture]):
                out = model.model(
                    input_ids=encoded["input_ids"].to(device_obj))
    finally:
        if adapter is not None:
            from ..experiment import model_variant

            model_variant.remove_adapter(model, adapter)
    missing = [l for l in armed if l not in capture.rows]
    if missing:
        raise ProbeError(
            f"no residual captured at layer(s) {missing} — the hook did not "
            f"fire, so the readout would be over nothing")

    pin_list = sorted(pinned)
    cells: list[dict] = []
    trajectory: list[dict] = []
    # The readouts run OUTSIDE the capture's forward pass, so they need their
    # own no_grad: the output head's parameters may require grad, and without
    # this every projection builds an autograd graph that is never used —
    # wasted memory per cell, at 27B across every position.
    with torch.no_grad():
      for layer in armed:
          states = capture.rows[layer]
          for position in scored:
              # The capture holds CPU rows on purpose (device tensors for every
              # position at 27B is the memory this exists to avoid); the readout
              # needs its own device, so move one position at a time.
              h_cpu = states[position].to(torch.float32)
              h = h_cpu.to(device=readout.device)
              row = {"layer": layer, "position": position,
                     "tokenID": ids[position]}
              if wants_vocabulary:
                  # The two projections, computed once each; the top-k and the
                  # pinned ranks are both READ OFF them.
                  lens_logits = readout.logits(h, layer)
                  comp_logits = readout.logits(h, layer, use_jacobian=False)
                  if config.topK > 0:
                      lens_ids, lens_values = readout.topk_of(lens_logits,
                                                              config.topK)
                      comp_ids, comp_values = readout.topk_of(comp_logits,
                                                              config.topK)
                      cells.append({
                          "layer": layer, "position": position,
                          "tokenID": ids[position],
                          "topKIDs": [int(t) for t in lens_ids.cpu()],
                          "topKLogits": [float(v) for v in lens_values.cpu()],
                          "companionTopKIDs": [int(t) for t in comp_ids.cpu()],
                          "companionTopKLogits": [float(v)
                                                  for v in comp_values.cpu()],
                      })
                  if pin_list:
                      row["pinnedRanks"] = dict(zip(
                          (str(t) for t in pin_list),
                          readout.ranks_of(lens_logits, pin_list)))
                      row["companionPinnedRanks"] = dict(zip(
                          (str(t) for t in pin_list),
                          readout.ranks_of(comp_logits, pin_list)))
              if loaded_directions:
                  per_direction = {}
                  for direction in loaded_directions:
                      vector = direction["vectors"].per_layer[layer]
                      v = torch.tensor(vector, dtype=torch.float32)
                      transported_h = readout.transported(h, layer).cpu()
                      transported_v = readout.transported(
                          v.to(device=readout.device), layer).cpu()
                      per_direction[direction["name"]] = {
                          "cosine": _cosine(h_cpu, v),
                          "transportedCosine": _cosine(transported_h,
                                                       transported_v),
                      }
                  row["directions"] = per_direction
              trajectory.append(row)

    rendered_mode = prompt_mode or prompt_render.CHAT_ASSISTANT
    tier = evidence_tier_for(record)
    stamp, claim = qualification_state(record, model)
    stamp, claim = downgrade_for_unpinned_agent(stamp, claim, variant_identity)
    run_directory = paths.make_unique_run_directory(
        f"{RUN_TYPE}-{model_id.replace('/', '--')}", root)
    write_run_config(run_directory, RUN_TYPE, model_id=model_id,
                     revision=revision, dtype=getattr(model, "dtype", None),
                     notes={"lensID": lens, "evidenceTier": tier,
                            "qualification": stamp, "claim": claim,
                            "armedLayers": armed, "positions": len(scored),
                            "variant": getattr(variant, "name", None),
                            "isBaseModel": variant is None})

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "runType": RUN_TYPE,
        "runID": os.path.basename(run_directory),
        "evidenceTier": tier,
        "qualification": stamp,
        "claim": claim,
        "model": {"modelID": model_id, "revision": revision,
                  "dtype": getattr(model, "dtype", None)},
        "lens": {"lensID": lens, "sourceSHA256": record.source.tensorSHA256,
                 "targetLayer": record.targetLayer},
        "configuration": {
            "armedLayers": armed, "topK": config.topK,
            "positionStride": position_stride, "maxTokens": max_tokens,
            "logitLensCompanion": True,
            "pinned": {str(k): v for k, v in pinned.items()},
            # Pinned to BYTES, not to a locator: vector artifacts live in
            # mutable library subtrees, so the same path can hold different
            # numbers later.
            "directions": [{"reference": d["reference"], "name": d["name"],
                            "concept": d["concept"],
                            "extractionMethod": d["extractionMethod"],
                            "recipeMethod": d["recipeMethod"],
                            "modelID": d["modelID"], "revision": d["revision"],
                            "stimulusSetHash": d["stimulusSetHash"],
                            "tensorSHA256": d["tensorSHA256"],
                            "sidecarSHA256": d["sidecarSHA256"]}
                           for d in loaded_directions],
            "projections": projections,
            "projectionConvention":
                "one full-vocabulary projection per (cell, lens) and one per "
                "(cell, companion); top-k and pinned ranks are derived from "
                "them, never recomputed. A --directions-only probe costs none",
        },
        # The effective CONDITION this read is about. A probe with no variant
        # is the base model under the declared rendering — a real condition,
        # named as one, so an artifact can never be mistaken for an agent's.
        "condition": {
            "variantPath": variant_path,
            "variantName": getattr(variant, "name", None),
            **variant_identity,
            "isBaseModel": variant is None,
            "promptMode": rendered_mode,
            "systemPromptPresent": bool(system_prompt),
            "injections": [
                {"layer": cell.layer, "alpha": cell.alpha,
                 "mode": cell.mode, "concept": cell.concept}
                for cell in (injections or [])],
            "injectionGate":
                "deployed semantics: injection fires at the FINAL prompt "
                "position only during prefill, so an injected agent's earlier "
                "positions are identical to baseline by construction",
            "adapterApplied": adapter is not None,
        },
        "prompt": {"text": prompt, "renderedText": rendered.text,
                   "tokenCount": len(ids),
                   "tokenIDs": ids,
                   "pieces": [tokenizer.decode([t]) for t in ids],
                   "renderingConvention":
                       "the SHARED prompt renderer, so positions correspond "
                       "to what a study run sees; raw tokenization would "
                       "produce a different sequence"},
        "cells": cells,
        "trajectory": trajectory,
        "conventions": {
            "companion": "every J-lens number carries the same readout with "
                         "J_l = identity; moving together means the token is "
                         "locally present, not that it is poised to be said",
            "rank": "full-vocabulary, 1-based, ties take the best rank",
            "statistics": "counts, ranks and cosines only; no null, no CI",
        },
        "runDirectory": run_directory,
    }
    path = os.path.join(run_directory, PROBE_JSON)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csvs(report, run_directory, tokenizer)
    report["paths"] = {"report": path,
                       "topKCSV": os.path.join(run_directory, TOPK_CSV),
                       "trajectoryCSV": os.path.join(run_directory,
                                                     TRAJECTORY_CSV)}
    emit(f"{claim} [{tier}] → {run_directory}")
    return report


def _write_csvs(report: dict, run_directory: str, tokenizer) -> None:
    with open(os.path.join(run_directory, TOPK_CSV), "w", encoding="utf-8",
              newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "position", "tokenPiece", "rank",
                         "lensTokenID", "lensPiece", "lensLogit",
                         "companionTokenID", "companionPiece",
                         "companionLogit"])
        for cell in report["cells"]:
            piece = tokenizer.decode([cell["tokenID"]])
            for rank in range(len(cell["topKIDs"])):
                companion = (cell["companionTopKIDs"][rank]
                             if rank < len(cell["companionTopKIDs"]) else None)
                writer.writerow([
                    cell["layer"], cell["position"], piece, rank + 1,
                    cell["topKIDs"][rank],
                    tokenizer.decode([cell["topKIDs"][rank]]),
                    f"{cell['topKLogits'][rank]:.4f}",
                    companion if companion is not None else "",
                    tokenizer.decode([companion]) if companion is not None else "",
                    (f"{cell['companionTopKLogits'][rank]:.4f}"
                     if rank < len(cell["companionTopKLogits"]) else ""),
                ])

    pinned = report["configuration"]["pinned"]
    directions = [d["name"] for d in report["configuration"]["directions"]]
    with open(os.path.join(run_directory, TRAJECTORY_CSV), "w",
              encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["layer", "position", "tokenPiece"]
        for token, piece in sorted(pinned.items()):
            header += [f"rank{piece.strip() or token}",
                       f"companionRank{piece.strip() or token}"]
        for name in directions:
            header += [f"cos:{name}", f"transportedCos:{name}"]
        writer.writerow(header)
        for row in report["trajectory"]:
            line = [row["layer"], row["position"],
                    tokenizer.decode([row["tokenID"]])]
            for token, _piece in sorted(pinned.items()):
                line.append((row.get("pinnedRanks") or {}).get(token, ""))
                line.append((row.get("companionPinnedRanks") or {}).get(token, ""))
            for name in directions:
                cell = (row.get("directions") or {}).get(name, {})
                line.append(f"{cell.get('cosine', 0.0):.5f}")
                line.append(f"{cell.get('transportedCosine', 0.0):.5f}")
            writer.writerow(line)
