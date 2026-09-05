# Intervention Scope

**What each intervention actually changes — and what a result from it can and
cannot claim.**

This engine can arm four residual-stream mechanisms. They share one protocol
(`LayerIntervention.apply(h, layer, offset)`) and one site (a decoder block's
output), and they are otherwise *not alike*: the additive path edits the final
prompt position and each decode step's last position; ablation edits **every**
position at its layers; the training-time injector edits one teacher-forced
pass's answer positions, or all of them. A methods section that describes them
with one sentence — "we steered the residual stream at layer L" — is wrong for
at least two of the four, and the error is not cosmetic: it changes what the
result is evidence *for*.

So the description is data, not prose that has to be remembered.
`steering.intervention.InterventionScope` is a frozen record with plain-string
fields; each class fills one from its own configuration; and every run stamps
the inventory for its whole condition matrix into
`intervention-scope.json` beside `config.json`. This page is that record's
human half. The two are meant to be edited together.

---

## 1. The paths at a glance

| Path | Class | Token positions | Dose | Matched control |
|---|---|---|---|---|
| `additive` | `steering/injector.py` `VectorInjector` | final prompt position on the last prefill chunk, then the last position of every decode pass | α (per-layer residual-norm units when the study declares them) | `randomMatchedNorm` |
| `ablation` | `steering/ablator.py` `SubspaceAblator` | every position, prefill and decode | λ, a dimensionless fraction of the projected component | `randomDirectionAblation` |
| `trainableAdditive` | `steering/trainable_injector.py` `TrainableVectorInjector` / `AdditiveDeltaProbe` | each item's answer position, or every non-pad position, in one teacher-forced pass | α as an absolute L2 norm at the layer | S0 shuffled-target null + `randomMatchedNorm` (declared by the confirm study) |
| `saeLatent` | `steering/sae_latent.py` `SAELatentIntervention` | same as `additive` — it reuses that gate | β in **latent** units | none synthesized; the baseline arm is the contrast |

Every path's site is the same tensor: **the residual stream at the block output
of the named layer(s)** — what a forward hook receives from decoder block *L*,
before block *L+1* reads it (`steering/hooks.py`). Nothing in this package hooks
inside attention or the MLP.

---

## 2. `additive` — `VectorInjector`

| | |
|---|---|
| **Hook / site** | residual stream at the block output of the named layers |
| **Layers** | the condition's declared layer widened by `bandWidth`; consolidated into one injector when every ADD edit has its own layer |
| **Positions** | final prompt position on the last prefill chunk, then the last position of every decode pass |
| **Prefill** | fires once, at the true prompt end; suppressed on every prefill chunk whose last position is mid-prompt (`should_inject`, gated on `prompt_token_count`) |
| **Decode** | fires on every decode pass, at that pass's single new position |
| **Centering** | not applicable — centering is an ablation-direction transform, and this engine refuses it on a steering injection (`api/routes.py`, `experiment/model_variant.py`) |
| **Dose units** | α at the layer, applied here as an absolute offset. A study declaring `alphaInNormUnits` converts its α through that layer's residual-norm denominator — the study's declared denominator convention — before the vector reaches this class |
| **Control** | `randomMatchedNorm`: the same layers at the same α with a seeded isotropic-Gaussian direction rescaled to the concept vector's L2 norm at each layer (algorithm stamp `gaussian-isotropic-v1`). Scaffolded by `experiment/control_matrix.py::control_matrix_conditions`, substituted at run time by `experiment/tasks.py::_condition_injections` |
| **Pinned by** | `tests/test_injection_fires_per_token.py::test_apply_adds_alpha_v_at_last_position_only`, `::test_should_not_inject_on_mid_prompt_prefill_chunk`, `::test_injection_fires_on_every_decode_step_of_a_sampled_generation`; descriptor binding in `tests/test_intervention_scope.py::test_the_injector_moves_only_the_position_its_descriptor_names` |

**The gate's state is part of the description.** Without a `prompt_token_count`
the class fires at the last position of *every* forward pass — correct only
when prefill is single-chunk, and silently mis-sited when it is not. The
descriptor prints a different `positions` sentence in that case rather than
letting a chunked run be reported as a prompt-end injection.

**Claim limit.** *Adds a fixed offset at these positions and nowhere else. A
behavioural change shows the offset moved this model's output here; it does not
locate the concept, and equal residual-norm doses are not equal potency across
layers or models.*

---

## 3. `ablation` — `SubspaceAblator`

| | |
|---|---|
| **Hook / site** | residual stream at the block output of the named layers |
| **Layers** | every layer of the network by default — a removal at one layer is usually rewritten by the layers above it, so ablation takes no band |
| **Positions** | **every position, prefill and decode** |
| **Prefill** | applies at every position of every prefill chunk — deliberately ungated, the inverse of the additive path's gate |
| **Decode** | applies at every position of every decode pass |
| **Centering** | `none` or `neutralMean`, declared per edit where the direction is resolved. The ablator carries the declaration and never applies it: by the time a direction reaches this class the transform is already in the numbers. A layer whose directions were expressed in two conventions is reported as `mixed(…)`, because orthonormalization dissolves the row/concept correspondence and there is nothing left to resolve it against |
| **Dose units** | λ, a dimensionless fraction of the projected component (1 = full removal, 2 = reflection). **No residual-norm denominator** — the removal is already scaled by whatever the stream carries, so α's comparability machinery has nothing to make comparable |
| **Control** | `randomDirectionAblation`: the same layers at the same λ with a random *direction* removed instead. Norm matching means nothing to a projection, so the control varies the direction and asks whether removing any rank-1 subspace does this (`experiment/control_matrix.py::ablation_control_conditions`) |
| **Pinned by** | `tests/test_ablator.py::test_every_position_is_ablated_including_mid_prompt_chunks`, `::test_full_ablation_zeroes_the_projection`, `::test_concentric_ablations_at_a_layer_become_one_subspace`; descriptor binding in `tests/test_intervention_scope.py::test_the_ablator_moves_every_position_its_descriptor_claims` |

**Rank is reported after orthonormalization**, per layer — two nearly parallel
directions remove a rank-1 subspace, and the record should say the rank that
was removed, not the count that was declared.

**Claim limit.** *Removes the component along the ablated subspace at this site
only. It does not show the model cannot represent the concept — in another
basis, at another site, or reconstructed by the layers above — and λ is not
comparable to α.*

---

## 4. `trainableAdditive` — `TrainableVectorInjector`, `AdditiveDeltaProbe`

| | |
|---|---|
| **Hook / site** | residual stream at the block output of one layer (no band: the optimizer trains one layer at a time) |
| **Positions** | `from_response` — each item's answer position in one teacher-forced full-sequence pass; `all` — every non-pad position in that pass |
| **Prefill / decode** | neither: one full-sequence pass with no KV cache. This class never runs under stepped decode; the finished vector deploys through `VectorInjector`, and §2 is what a measured run executes |
| **Centering** | not applicable |
| **Dose units** | α as an absolute L2 norm at the layer — the direction is projected onto that sphere inside `apply()`. Converting a norm-unit α into absolute units is the training driver's job, because the residual-norm denominator is a per-model calibration, not a property of the injector. `AdditiveDeltaProbe` carries **no dose**: its delta is zero-initialized and it reads a derivative |
| **Control** | declared by the confirm study, never synthesized at run time: the **S0 shuffled-target null** (the identical optimization against permuted labels) alongside the ordinary `randomMatchedNorm` cell (`experiment/control_matrix.py::optvec_confirm_conditions`). A matched-norm random direction alone is far too weak a null against a gradient search over ℝ^d, which can fit noise |
| **Pinned by** | `tests/test_trainable_injector.py::test_from_response_injects_exactly_at_the_supplied_positions`, `::test_all_mode_injects_at_every_non_pad_position`, `::test_exported_vector_under_the_deployed_injector_matches_training_path` |

**The two modes are two different interventions.** `from_response` is the
train-time equivalent of deployed semantics for a single-token forced-choice
readout: the injected position is the one whose logits predict the answer
token. `all` is an explicit variant — a vector optimized under `all` and
deployed under prompt-end semantics is a train/deploy mismatch, which is why it
is never the default.

**Claim limit.** *A direction selected ON behaviour is a screen, not a result:
it is one sample from an equivalence class, so only loadings stable across
seeds are interpretable, and the citable numbers come from a separate confirm
study run through the deployed additive path.*

---

## 5. `saeLatent` — `SAELatentIntervention`

| | |
|---|---|
| **Hook / site** | residual stream at the block output of the named layer |
| **Positions** | identical to §2 — it reuses `VectorInjector.should_inject` rather than restating the arithmetic |
| **Centering** | not applicable |
| **Dose units** | β in **latent** units: the feature's own activation scale, set by the dictionary's normalization. Not α, not residual-norm units, and not comparable across features. Records stamp `latentUnits: true` so no reader mistakes it for a dose in norm units |
| **Control** | none synthesized by this engine. The contrast is the baseline arm; `clamp` with β = 0 is the per-feature removal cell. Dose calibration against a pinned corpus (so β could be expressed as a percentile of a feature's own activation distribution) is declared future work |
| **Pinned by** | `tests/test_sae_latent.py::test_gate_is_literally_the_injectors_gate`, `::test_applies_at_the_last_position_only`, `::test_fires_on_every_decode_step` |

The edit is **state-dependent** — the SAE's JumpReLU gate decides whether
anything happens at all — so the same β is a different residual-stream delta at
every position. That is why it gets its own row rather than a footnote on §2.

**Claim limit.** *Edits one dictionary feature's latent activation and decodes
only the induced delta at this site. It shows what THIS dictionary's feature
does here; β is in latent units, comparable neither across features nor to α.*

---

## 6. What an ablation result does and does not show

An ablation is an orthogonal projection: at the layers it covers, and at every
token position, it removes the component of the residual stream that lies in
the ablated subspace. λ = 1 leaves `h'·v̂ = 0` for each ablated direction.

That is a statement about **this site**. It is not a statement about the model.

Specifically, a behavioural change under ablation does not show:

- **that the model cannot represent the concept.** The projection is applied to
  one tensor at one set of layers. Anything the model computes downstream of
  those layers can reintroduce a component along the removed direction — and
  routinely does, which is why ablation covers every layer by default rather
  than one. "Covers every layer" is still not "cannot represent": it is
  "cannot carry it in this basis at these sites during this forward pass".
- **that the direction is the concept.** A direction is a coordinate chosen by
  an extraction recipe. Removing it removes whatever else shares that
  coordinate, including the residual mean if the direction was never centered
  against it — which is exactly the failure the mean-alignment preflight warns
  about, where λ = 1 collapses generation into single-token repetition.
- **that the effect is specific.** That is what the `randomDirectionAblation`
  control is for: does removing *any* rank-1 subspace of the residual stream
  produce this? Norm-matching a random vector answers a different question and
  answers it vacuously — scaling a direction does not change its projection.

And a *null* result under ablation does not show the concept is unused: it is
equally consistent with the model reconstructing the component above the
ablated layers, or with the behaviour depending on a direction the extraction
did not recover.

The honest form of an ablation claim names the site and the control: *"removing
this direction at every layer and every position changed the outcome by X,
where removing a random direction of the same rank changed it by Y."*

## 7. Dose comparability

**α is comparable only in the sense its denominator makes it comparable.** When
a study declares `alphaInNormUnits`, α is divided through by the typical
residual-stream norm at that layer, so "α = 1" means "an offset the size of a
typical activation there". That makes α roughly commensurable *across layers of
one model, under one denominator convention*. It does not make equal α equal
**potency**: the same normalized offset moves behaviour by different amounts at
different depths, and by different amounts in different models, because what
downstream layers do with a perturbation is not a function of its size. A
dose-response ladder within one arm is evidence; "the same α" across models or
layers is a *coincidence of units*, not a matched intervention.

**λ is not α.** λ scales a removal that is already sized by the stream's own
content, so it has no denominator and needs none. λ = 1 is total removal
whatever the activation's magnitude; α = 1 is an offset whose effect depends on
what was already there. Reporting them in one column, or converting between
them, states a relationship that does not exist.

**β is neither.** It is in the dictionary's latent units. Two features'
β = 5 are unrelated quantities, and no residual-norm denominator relates either
of them to α.

**Absolute α (the training path) is a third thing again.** It is the L2 norm the
optimized direction is projected onto at one layer, in raw units — the
conversion from norm units happens in the driver, and the vector that leaves
training carries the absolute number.

## 8. Where a run records this

`intervention-scope.json`, written once at run start into the run directory by
`experiment/intervention_scope.py::stamp_run` (one call site, in the run driver
in `experiment/tasks.py`). Shape:

```json
{
  "schemaVersion": 1,
  "experiment": "<name>",
  "conditions": [
    {"condition": "baseline", "interventionState": {}, "scopes": []},
    {"condition": "fear-a1",
     "interventionState": {"slots": [...], "bandWidth": 1,
                           "alphaInNormUnits": true, "controlType": null},
     "scopes": [{"path": "additive", "site": "...", "layers": [14],
                 "positions": "...", "prefill": "...", "decode": "...",
                 "centering": "notApplicable", "doseUnits": "...",
                 "control": "...", "claimLimits": "...",
                 "detail": {"alphaPerLayer": {"14": 0.8}, ...}}]}
  ]
}
```

Notes a reader needs:

- `scopes` is in **chain order** — the ablator first, then any SAE latent
  intervention, then the injectors — so a condition that both ablates and adds
  shows two rows with two different position claims. That is the design, not a
  duplication.
- An arm that changes no residual stream (baseline, and every carried condition
  of a study whose concept machinery is inert) has `"scopes": []`. An arm the
  stamp could not resolve — an agent artifact that will not load, a condition
  naming an unextracted concept — carries `"unresolved": "<reason>"` instead of
  being dropped. The stamp never raises: the condition loop owns every
  resolution failure at the point the arm would have executed, and describing a
  run must not change what it does.
- It is written **once**, before any generation compute, and never rewritten —
  so a run that dies mid-matrix still says what it had armed, and a resumed run
  keeps its original stamp.
- Every shard of a sharded run writes the whole matrix, so the file is
  byte-identical across shards; the merge verifies that and carries one copy
  into the merged run.
- It is a **sidecar**, not a `config.json` key: that key set is the closed
  cross-engine schema (`experiment/run_config.py`), and engine-specific
  provenance lands beside it, as `substrate.json` already does.

## 9. Swift parity

**Mirrored in behaviour today.** The Swift engine implements the same
mechanics, and the tests on both sides pin them against each other:

- `Sources/SteeringKit/Injection/VectorInjector.swift` — same last-position
  firing, same `shouldInject` chunked-prefill gate.
- `Sources/SteeringKit/Injection/SubspaceAblator.swift` — same every-position
  removal, same `dependenceTolerance`, same modified Gram-Schmidt in float64.
- `Sources/SteeringKit/Injection/InterventionPlan.swift` — same
  single-ablator-first chain, same one-subspace-per-layer rule.
- `ChatService.currentInjections` — same `neutralMean` centering semantics and
  the same mean-alignment advisory threshold.

**Owed on the Swift side.** The *descriptor* is Python-only: Swift has no
`InterventionScope`, no `scope()` on its intervention types, and no
`intervention-scope.json`. Until the constants in
`Server/steerlab_server/steering/intervention.py` are ported verbatim (they are
deliberately written as module-level constants for exactly this reason), a
Swift-executed run stamps no sidecar — and a reader must not read that absence
as a claim about what a Mac run did. The trainable path has no Swift twin at
all: optimization is server-only.
