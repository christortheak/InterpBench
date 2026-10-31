# P5 intervention policies

Implementation and format contract for P5 of the phase-1 plan. A policy changes
an agent's behavior; a study measurement only records it. P5 execution is Python
Compute. The app, native CLI, portable client, and workbench API share authoring.
Native MLX policy execution, dynamically padded batches, and extra model passes
are not part of this slice.

Policies are immutable, self-contained JSON documents. Publication resolves and
embeds exact probe bytes, selected vector values with source hashes, and optional
expert provider source/assets. Agent attachment copies the exact policy text and
its SHA-256 into a new agent version; it never rewrites the source agent or run.
An absent policy field retains historical agent encoding. Existing study and
panel attachment owners then carry the policy with the agent through freeze and
transport, without hidden dependencies on mutable library files.

A policy declares model/revision, tokenizer, rendering, residual coordinates,
one decision site, prefill/decode schedule, token positions, actions, rules,
error handling, and a decision-record budget. Rules are fixed, threshold, or
bounded affine mappings of a weighted combination of named probe scores. Probe
scores are predictive quantities, not assumed probabilities. Same-site residual
policies read the pre-action tensor; logits policies may read a probe at a
residual site from the same consumed token, never an earlier cached score.

At a site the order is: pre-action study readings; all policy decisions from the
same snapshot; the existing legacy chain; policy subspace removal; ordered policy
additions; post-action study readings. The new policy action phase is explicitly
separate from the legacy chain. Conditional removal can therefore remove an
existing legacy offset. Multiple policy ablations at a site share one subspace
and must request the same per-position removal fraction, or execution explains
the conflict. Addition uses the direction as stored at the policy's one layer,
with no layer band or neutral-basis projection, and a bounded strength in
residual-norm units (below). Removal uses a dimensionless fraction of the
normalized subspace projection.

At token selection, policies run after the existing Hugging Face processors and
finite-logit guard, before temperature and sampling filters. Biases add first;
allowed-token sets and forced single token IDs intersect. They do not unmask
already forbidden tokens or override EOS, cancellation, or length limits. An
empty allowed set or incompatible forced choices is an explicit execution error.
A text string is never treated as one token without tokenization.

Providers execute in process and return typed, declared action decisions. They
receive tensor inputs, named score tensors, response-local state, a dedicated
RNG, and pinned assets. This is trusted expert code, not a sandbox. Invalid action
IDs, impossible sites/timing, out-of-bound strengths, and non-finite decisions
cannot silently alter the declared policy. An explicit skip-policy error mode
records a failed decision; default stop preserves partial evidence. Recording
budgets bound traces independently of execution and count omitted events.

Independent tests must cover arithmetic, rule combinations, same-snapshot
ordering, state/RNG isolation, site alignment, token conflicts, noninterference,
custom providers, immutable authoring, exact-byte attachment, cross-client
round-trips, and ordinary/panel generation. No app install, cluster deploy, live
research job, or model download is authorized by implementation alone.

## Strength units

From policy schema version 2, an addition's strength is in residual-norm units,
the units of a static agent's α. A strength α adds α × R / ‖v‖ × v at each
selected position, where v is the stored direction and R is the vector
sidecar's stamped mean residual norm at the policy layer. R and its averaging
convention are the ones static steering reads for that vector and layer, so the
addition's norm is α × R. At a `residualPost` site, a fixed policy at α over both
stages at `lastPosition` adds what a static agent at α adds, with no band or
neutral basis, at the same positions and to activation precision: the static
injector rounds α·v to the activation dtype before adding, while a policy adds in
float32 and rounds once, so in bf16 a few elements can differ by one unit in the
last place. A `residualPre` policy acts at the block's input instead. 0.1, a
tenth of the layer's typical residual norm, is the usual starting strength (the
Playground's default when no promoted agent exists); a full norm (1.0) usually
makes generation incoherent. Fixed, threshold, and affine values, action bounds,
and the strengths an expert provider returns are all in these units.

Review records what execution needs on the action: `strengthUnits`,
`residualNorm`, `residualNormConvention` (`null` for a sidecar measured before
the convention was stamped), and `vectorNorm`, beside the `source` that names the
vector artifact and the SHA-256 of its sidecar. A residual-norm addition without
that source is refused. The policy's hash covers these fields, so execution and
evidence never reread the vector library. Attachment review rereads the sidecar
when it is in the workspace and refuses if its residual norm or convention
differs from what the policy recorded. A vector with no stamped
residual norm at the policy layer, or a direction given as numbers, is refused
(`policyResidualNormMissing`). The repair is to measure the norms with
`vectors backfill-norms`, which writes a new vector artifact, or to choose
`"strengthUnits": "rawMultiplier"` on that action. A raw multiplier multiplies the
stored direction directly and is not comparable with a static agent's α; it is
an explicit, recorded choice for one action, never a default.

Every published version-1 policy keeps its meaning forever: its additions
multiply the stored direction directly, and frozen studies pin those bytes by
hash. Version 1 has no unit fields, and one written into it is refused rather
than reinterpreted. Review no longer authors version 1
(`policyVersionOneSettings`): set `"schemaVersion": 2`, and add
`"strengthUnits": "rawMultiplier"` to an action to keep the old meaning. An
engine that knows only version 1 refuses a version-2 policy, and clients check
for the `policy-v2` capability before submitting one. Ablation keeps its
dimensionless fraction and token actions keep their own scales in every
version.

## Span policies: steering while the model reads

A span policy acts while the model **reads** a declared part of the prompt, an
instruction sentence for example, and at no other time. It sets
`"positions": "span"`, declares the part in `"span"`, and declares its dose in
`"spanDose"`. It needs schema version 2 and the stages `["prefill"]`, at a
residual site (`residualPre` or `residualPost`):

```json
{"positions": "span", "stages": ["prefill"],
 "span": {"kind": "substring", "text": "Ignore all prior rules."},
 "spanDose": "perPosition"}
```

The declaration uses the prompt-span grammar (`experiment/prompt_span.py`):
`substring` (with an optional 1-based `occurrence`), `anchors` (a `start` and
an `end`, both included), `userMessage` (the user text, never the system
prompt), or `allPrompt` (every prompt position, template markers included).
Matching is exact. Review writes the canonical declaration and the default
dose, so the policy's hash covers both. The span restricts whatever strengths
the policy produces: fixed, threshold, and affine rules and an expert
provider all work, and only the span's tokens receive them.

**Dose.** With `perPosition` (the default), each span token gets the strength,
so the summed dose grows with the span's length. With `fixedTotal`, the
strength is a total: each of the span's n tokens gets strength / n, so the
summed dose is the strength whatever the span's length. Strengths stay in
residual-norm units, so a span token at strength s receives an offset of norm
s × R. A `fixedTotal` dose applies to additions only, since a removal is a
fraction at each position; put a removal in its own `perPosition` policy.

**Resolution.** One declaration applies to the whole study. The run resolves
it on every task prompt, against the prompt exactly as that agent renders it
(prompt mode, system prompt, and reasoning settings included), once per
distinct rendered prompt, not once per response. Every item is checked twice
before anything is generated: before the model loads, on the user text alone,
and after it loads, with its tokenizer. A missing or ambiguous anchor in any
item refuses the whole run (`studyDeclaration`), naming every offending item
with the resolver's own code (`spanAnchorMissing`, for example) and its
repair. A run never fails halfway because a span did not match. Per-item
declarations are later work.

**Where span policies run.** Phase 1 runs them in single-prompt sampled
studies on the Python engine. Scripted transcripts, panels, agent chat, and the
cost instrument refuse them by name. The capability battery does not run them,
as with every policy agent (see Evidence and current limits).

**Evidence.** Decision rows exist only at span positions. Requested and
applied strengths are the declared ones; for an addition, `appliedScales` is
the multiplier applied at that position, which for `fixedTotal` is
strength / n × R / ‖v‖. Each response's `interventionDecisions.spans` has one
entry per span policy: `tokenRanges` and `tokenCount` (n), `doseMode`, the
`renderedTextSHA256` and `promptTokensSHA256` of the prompt it was resolved
on, and per action its `strengthUnits`, `appliedPositions`,
`perPositionStrength` (null when it varied), and `summedDose` (the sum over
the span, in the action's units). Both evidence validators check these against
the retained prompt tokens and the declaration. The resolver's full evidence
(the matched characters, boundary tokens, and how the user text was located)
is stamped once per distinct resolution in the run's `span-resolutions.json`,
keyed by `spanEvidenceSHA256`, which each response's entry names.
`intervention-scope.json` adds a `policySpanAddition` row for the policy
([INTERVENTION-SCOPE.md](INTERVENTION-SCOPE.md) §10).

**Claim limit.** *A span policy changes how these prompt tokens are
represented while the prompt is read. A behavioral change shows that reading
this span differently moved the output, not that the concept lives there.*

### Steering while reading versus while responding

To compare the two, give each its own agent arm, with the same vector, layer,
and strength:

- **Reading:** the span policy alone. Attach it with
  `"staticSteering": "remove"` in the attachment settings, so the new agent
  version keeps the source agent's model, rendering, and system prompt but
  none of its static vectors or adapters.
- **Responding:** a policy with stages `["prefill", "decode"]` at
  `lastPosition`, attached the same way. At a `residualPost` site and the
  same strength it adds what a static agent adds at that layer, at the same
  positions and to activation precision, so it is the static-equivalent arm.
  The static agent itself also serves.
- **Both**, if the question needs it: one agent version with both policies.
- The baseline, as always.

Read each arm against the baseline, then compare them on the same items. A
span arm needs its own controls, since a random direction or another sentence
of the prompt can change the output too (next section).

### Matched controls for a span policy

A span arm needs controls of its own: static steering's control cells
(`randomMatchedNorm`, `randomDirectionAblation`) cover concept conditions, not
agent arms. Two matched controls are policies too. `science policy-review`
derives one from a published span policy when the settings name it in
`controlOf`, and `science policy-publish` publishes it with the review's hash,
so each control is reviewed, immutable, and inspectable like any policy:

```json
{"controlOf": "runs/<run>/intervention.policy.json", "control": "randomDirection",
 "name": "reading-random-direction", "seed": "1"}
{"controlOf": "runs/<run>/intervention.policy.json", "control": "location",
 "name": "reading-location", "span": {"kind": "substring", "text": "The room was quiet."}}
```

- **Random direction.** Each direction becomes a seeded random unit direction
  of the same width (`gaussian-isotropic-v1`, the construction static
  steering's random controls use), at the same layer and site, with the same
  span, rules, bounds, dose mode, and R. In residual-norm units the added norm
  is strength × R whatever the direction, so only the direction differs. The
  seed is recorded, and the same seed always gives the same control.
- **Location.** The same directions, strengths, and dose over a different
  declared span, typically a neutral sentence of the same prompt. Its token
  count must equal the controlled span's on every prompt, so the run's span
  preflight resolves both and refuses before generating anything when they
  differ, naming each item and both counts. A span of another length receives
  a different summed dose (`perPosition`) or a different strength at each
  token (`fixedTotal`), so a mismatch would confound place with dose. The
  repair is to edit the neutral sentence until the counts agree.

Each control's document records what it is in `control`: its kind, the
controlled policy's SHA-256 (`of`), and the seed and algorithm, or the span it
controls (`sourceSpan`). The hash covers it, so every response's evidence and
the run's scope stamp name the control and what it controls. A control cannot
be derived from another control or from a policy that does not act on a span.

**Seating the controls.** Attach each control to its own new agent version,
exactly as the span policy was attached, with `science policy-attach-review`
and `science policy-attach`, and `"staticSteering": "remove"`:

```json
{"agentPath": "runs/<source-run>/agent.json", "name": "reading-random-direction",
 "policyPaths": ["runs/<control-run>/intervention.policy.json"], "staticSteering": "remove"}
```

Then attach every agent version to the study with `experiment attach-agent`.
A complete design has the baseline, the span arm, its two controls, and, for
the reading-versus-responding question, the `lastPosition` arm. The span
arm's effect is the part of its difference from the baseline that the random
direction and the other location do not also produce.

## Authoring and attachment

The app opens **Intervention policies…** from both Probes and Agents. The guided
starter asks for a portable probe, a fixed/threshold/adaptive rule, an action,
and strength bounds. It uses the probe's pinned model, rendering, and site;
expert settings can define more complex policies. An added vector starts at
strength 0.1 within bounds of 0 and 0.2 residual-norm units; removal and token
actions start at 1 within bounds of 0 and 1. It presents a review before
saving and separately reviews a new agent version. The current Playground
control editor cannot express policies; it explains the Python study/API route
instead of silently dropping them.

Both CLIs expose `science policy-list`, `policy-inspect`, `policy-review`,
`policy-publish`, `policy-attach-review`, and `policy-attach`. Review/publish and
attachment verbs take one settings-file positional. The two publishing verbs
also require `--plan-sha256` from the corresponding review. Add `--root` and
`--json` on either CLI. No model runs during authoring. Workbench HTTP uses
`POST /api/science/workspace/<verb>`: `workspaceRoot` plus `settingsText` for
reviews, plus `planSHA256` for publication. Inspection takes `path`; listing
only takes the root. Runner service-role restrictions remain unchanged.

The complete settings format, worked example, and trusted-provider ABI are
maintained in the [packaged readers guide](../WorkspaceSeed/prompts/method-guides/readers.md#complete-policy-format-and-provider-abi).
Both installed clients expose that guide through `science guide readers`; the
app's copy-instructions action carries it too. No source checkout is needed to
learn the advanced schema.

## Evidence and current limits

Ordinary sampled studies and panel turns retain `interventionDecisions` beside
responses. Panel flattening preserves it. Each decision row records requested
and applied strengths in the action's declared units, those units
(`strengthUnits`), and, for an addition, `appliedScales`: the multiplier applied
to the stored direction, which for residual-norm units is the raw scalar a
static agent at the same α applies. Scope reports summarize policy hashes,
bindings, timing, and each action's bounds and units. The app's Results view exposes those
records. A synchronous Python agent-chat response includes decision records;
streaming chat executes policies but is an exploratory text stream, not durable
study evidence. Policy-failure sidecars preserve partial observations and never
masquerade as completed responses or resume keys.

P5 does **not** implement native MLX policy execution, padded or multi-sequence
batches, arbitrary execution-flow changes, auxiliary model execution, gradient
training through policies, direct choice scoring, or policy-aware capability
battery qualification. Unsupported static-scoring paths refuse instead of
silently omitting the policy. Use sampled-response comparisons for the current
slice; record the missing battery qualification explicitly. Freeze records it
for you (`capabilityBatteryNotApplied`), and when a policy agent also carries
vectors or adapters, validation reads that static steering alone, labelled
"*agent* without its policy", and freeze requires that reading
(`CONDUCTING-A-STUDY.md` §5.2). P6 adds richer analysis; P7 retains the live qualification/remote acceptance walk. Live
GPU qualification and manual app interaction have not been established by the
synthetic/unit tests.

P6 implements remote support admission, acknowledged action evidence, structural
collection checks, and shared offline summaries. See the
[P1–P6 review handoff](PROBES-INTERVENTIONS-P1-P6-REVIEW-HANDOFF.md). P7 retains
live deployment, numerical/performance qualification, and manual app acceptance.
