# `optvec jspace` — reading a propagated injection through a J-lens

**What this document is.** The reader's guide to the artifact
`steerlab-server optvec jspace` writes (`jspace.json` +
`jspace-records.jsonl`): what each number is computed from, which of three
different questions it belongs to, and — the part that keeps this verb honest —
what none of the numbers establish. The implementation is
`Server/steerlab_server/experiment/optvec_jspace.py`; its module docstring says
the same things in the same words.

**Why it exists.** An external review found the same three tables being read as
three different kinds of claim: an arithmetic identity read as evidence about
the model, a readout of a difference vector read as a change in the model's
output, and one random comparator read as a significance test. Every number
here was correct; the reading was not. So the report now separates the
questions, carries the terms that make its own partition complete, and states
its floors in the artifact rather than in a reviewer's memory.

---

## §1 — What the verb computes

Two probe passes per artifact, teacher-forced over identical token sequences,
read at each item's answer position:

```
baseline   h_base          (no intervention)
steered    h_steered       (alpha * v injected at layer L)
delta      = h_steered - h_base                  at each observation layer l >= L
direct     = alpha * v                           (the dosed vector itself)
emergent   = delta - direct                      (what the model computed from it)
```

The J-lens supplies a LINEAR map `J_l` from a source-layer residual into the
final pre-normalization basis. Everything in the report is a function of
`J_l(delta)`, `J_l(direct)` and `J_l(emergent)`, plus a matched-norm random
comparator pushed through the identical pipeline.

Plus, per artifact, a third kind of pass: `nullDraws` matched-norm RANDOM
directions, each its own seeded draw with its own steered pass.

---

## §2 — Three questions, and which fields answer which

The report answers the first, gestures at the second, and does not touch the
third. The `quantities` block in every report says this too, verbatim.

### (1) Arithmetic fidelity — does the bookkeeping hold?

| Field | Where |
|---|---|
| `linearityResidualL2`, `maxLinearityResidualL2` | per record / per layer |
| `deltaEnergy`, `directEnergy`, `emergentEnergy` and their `…CrossTerm` | per record |
| `meanDeltaEnergy`, `directEnergy`, `meanEmergentEnergy`, `meanDeltaEnergyCrossTerm` | per layer |

`J(x+y) = J(x) + J(y)` holds for **any** linear map and **any** split of the
delta. `linearityResidualL2` is therefore a check that the transport was
computed correctly — nothing more. A residual at 1e-7 does **not** mean the
lens explains the model's behavior; it means the arithmetic is right.

The energies are a partition of one squared norm, and squared norms are not
additive over a sum:

```
||d + e||^2 = ||d||^2 + ||e||^2 + 2<d, e>
deltaEnergy =  directEnergy + emergentEnergy + crossTerm
```

The cross term is reported (`ENERGY_PARTITION`, stamped at
`instrument.energyPartition`) because without it a reader computes
`directEnergy / deltaEnergy`, gets a number between 0 and 1, and calls it "the
share of the effect the vector accounts for". It is not a share, not a
percentage, and not causal — the third addend is missing from that quotient and
can be either sign. A signed quantity takes a null but never a ratio, so
`…CrossTermNullRatio` does not exist.

### (2) Readout predictiveness — does the lens's picture mean anything here?

| Field | Where |
|---|---|
| `topKDelta`, `topKEmergent` | per layer |
| `cosineDeltaDirect`, `meanCosineDeltaDirect` | per record / per layer |
| `evidenceTier`, `qualification`, `claim`, `lens.compatibility` | report |

`topKDelta` is the lens's **normalized** readout of the MEAN over items of the
`(steered - baseline)` residual. It is a readout of a DIFFERENCE VECTOR. It is
not the difference between the normalized readouts of the two states —
normalization does not distribute over a subtraction — and it is not an
observed change in any logit the model produced. Same for `topKEmergent` over
`delta - direct`.

`J_l` is an averaged Jacobian estimator fitted on the lens's own prompts.
Whether it transfers to this probe set's prompt distribution at this dose is an
empirical question, and the verb does not answer it: `jlens qualify` does. Until
a runtime has a passing qualification record, every readout is stamped
`claim: "exploratory"` and is not citable evidence.

### (3) Behavioral / causal evidence — absent

**No field in this report measures the intervention's effect on the model's
OUTPUTS.** Not token probabilities, not choices, not generated text. This verb
reads residual streams. A causal claim about behavior needs a behavioral run
(`optvec eval`, a study run, a battery) that measures outputs under the same
dose, and it needs its own controls.

---

## §3 — Null draws: a comparator, not a test

Every energy is reported beside a matched-norm random sibling, and the writer
refuses a report where any key ending in `Energy` lacks its `…Null` and
`…NullRatio`. That rule is about never showing an energy naked. It is not a
significance test, and the report says so (`NULL_INFERENCE_NOTE`, stamped at
`null.inferenceNote`).

`nullDraws` (default 1, integer >= 1) buys *k* independent draws per artifact:

- Draw 0 uses `nullSeed + artifact ordinal` — the seed a single-draw run has
  always used, so a `nullDraws: 1` run reproduces every number an older run
  produced, key for key.
- Draw *k* uses `nullSeed + artifact ordinal + k * NULL_DRAW_STRIDE`
  (`NULL_DRAW_STRIDE` = 1000003, far larger than any plausible artifact count,
  so draws never collide across artifacts).
- Every draw's seed is stamped: `vectors[].null.draws = [{draw, seed}, …]`,
  `drawCount`, and `nullSeeds` on every per-item record.

What each level reports:

| Level | Null value | Also |
|---|---|---|
| per layer (`vectors[].layers[]`) | `…EnergyNull` = MEAN over draws | `…EnergyNullDraws` (each entry that draw's mean over items), `…EnergyNullSD` (sample SD, `null` for one draw), `…EnergyNullExceedanceCount` |
| per item (`jspace-records.jsonl`) | `…EnergyNull` = MEAN over draws for that item | `…EnergyNullDraws` only |

Per-item rows deliberately carry no SD and no exceedance count: a two- or
three-draw spread on a single item reads as that item's error bar, which it is
not.

**Guidance.**

- Choose `nullDraws` BEFORE looking at any result. A draw count raised after
  seeing a ratio you did not like is not a null.
- With *k* draws the smallest one-sided exceedance fraction that can be
  resolved at all is `1/(k+1)`. One draw resolves nothing finer than 1/2. Ten
  draws resolve 1/11. Read `…NullExceedanceCount` as a rank, never as a
  p-value.
- The draws share one pipeline and one probe set, so they are not independent
  replications of the experiment — they bound one thing only: how much of the
  observed energy an arbitrary direction of the same length would have produced
  in this same setup.
- Cost is linear: each draw is one more full steered pass per artifact.

---

## §4 — The precision floor

Residual rows are captured at the runtime dtype and cast to float32 inside the
pass. **The cast does not recover precision the runtime dtype already
discarded.** On a bf16 runtime the subtraction that isolates the emergent term
is a subtraction of two numbers that were each rounded to roughly 3 decimal
digits before anyone touched them.

So each artifact reports, at the injection layer (`vectors[].realizedDose`):

| Field | Meaning |
|---|---|
| `nominalDirectNorm` | `alpha * ||v||` — the dose that was asked for |
| `meanRealizedDeltaNorm` | mean over items of `||delta_L||` — the change that actually appeared |
| `meanRealizationErrorNorm` | mean `||delta_L - direct||` — the gap between them |
| `realizationErrorToDoseRatio` | that gap in units of the dose |
| `meanBaselineResidualNorm` | mean `||h_base(L)||` — the stream the dose was added to |
| `captureDtype` | the runtime dtype the rows were captured at, before the float32 cast |
| `dtypeEpsilon` | `torch.finfo(dtype).eps`, or `null` when the dtype is not a float this engine recognizes |

At `l = L` the delta IS the dosed vector by construction, so every bit of
`meanRealizationErrorNorm` is realization noise. That number is the empirical
noise floor for THIS dose and THIS dtype. **A low-dose result must be read
against it, not against zero.** If `realizationErrorToDoseRatio` is not small,
the dose is at or below the runtime's own arithmetic noise and the emergent
term at deeper layers is measuring rounding as much as computation.

The injection layer is captured for this block even when the declared
`observationLayers` skip it; `observationLayers` still reports exactly what was
asked for.

---

## §5 — Config keys

Closed-key JSON: an unknown key refuses rather than being ignored, because a
typo'd `observationLayer` that fell back to the default ladder would answer a
different question under the same filename.

| Key | Type | Default | Notes |
|---|---|---|---|
| `vectorArtifacts` | list of strings | required | extension-less artifact paths; >= 2 adds the family tables; duplicates refuse |
| `lensID` | string | required | the imported lens; there is no default |
| `probeItems` | `{path, sha256}` | required | hashed choice-row JSONL; the readout is taken at each item's answer position |
| `observationLayers` | list of ints | default ladder | all must be >= L and fitted source layers of the lens |
| `alphaMultiple` | float > 0 | 1.0 | dose as a multiple of the artifact's own norm |
| `nullSeed` | int | 20260810 | base seed for draw 0 |
| `nullDraws` | int >= 1 | 1 | independent matched-norm random draws per artifact; non-int and < 1 refuse |
| `seed` | int | 0 | inert — no sampling occurs; recorded and stamped `seedInert` |
| `topK` | int >= 1 | 10 | width of the token tables |
| `microbatchSize` | int >= 1 | 4 | capture batching only |
| `modelID`, `revision`, `device`, `dtype` | strings | from the artifact | runtime resolution |
| `promptMode`, `systemPrompt`, `qwenThinkingEnabled` | | | rendering, identical to a study run's |
| `name` | string | artifact name | run-directory label |

Default observation ladder: `L`, then `L+4`, `L+8`, … while the layer is a
fitted source layer of the lens, plus the lens's deepest fitted source layer if
the stride missed it. Below `L` the delta is exactly zero by construction; the
lens's target layer has no Jacobian.

---

## §6 — Provenance

- `engine: {version, buildCommit}` — the code that produced the artifact, so an
  impact ledger can classify a finding by producing revision rather than by
  date. `buildCommit` is `null` when no identity source resolves; it is never a
  guess.
- `evidenceTier` / `qualification` / `claim` — the lens's tier and its
  qualification against the runtime that actually ran, stamped by the engine.
- `lens` — lens id, fit model, source tensor SHA-256, fitted layer range,
  readout and direction conventions, and the compatibility report.
- `model` — model id, revision, dtype, device, and the per-artifact
  verification.
- `probeItems` — path, SHA-256 and item count.
- `config.json` `notes` carries the same engine stamp, the draw count and a
  compact per-(artifact, layer) summary.

Run directories are immutable: a corrected readout is a NEW run against the
same pinned probe set and lens, never an edit of an existing one. Two runs are
told apart by their `engine` stamp and their `runID`.
