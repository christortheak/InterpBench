# P4: shared residual runtime and legacy actions

P4 builds on P3 study measurements. It changes execution ownership, not an
agent's scientific settings or the arithmetic of an existing intervention.
The [phase-1 plan](PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md) remains authoritative.

## Researcher outcome and surfaces

A study's pre-action and post-action probe readings now surround the same action
dispatcher that executes its steering. The ordering is explicit, rather than a
consequence of the order in which Python forward hooks were installed. Reading
at a block input means before that block computes. Reading at its output means
after the block computes, before or after this site's action chain as requested.
Neither pre-action reading means the model is globally unsteered.
Within a phase, readings follow declaration order (and subscription order when
several response observers are supplied). This deliberately replaces the reverse
registration order of several prepended P3 Python hooks. A shared recording
budget can therefore favor different readings at its boundary; inspect omission
counts and allocate enough budget for the intended comparison. Existing evidence
is not rewritten.

Existing forms, both CLI clients, and the workbench HTTP owner keep their P3
measurement settings and review/save workflow. Existing steering and ablation
settings keep their identities and scope descriptions. No new manifest field,
policy verb, provider-code loader, or app-only authoring route is introduced.
The portable probe runs through Python Compute locally or remotely. The native
model runtime gains the same residual boundary vocabulary and explicit legacy
ordering; this does not implement portable probe scoring on MLX.

## Owners and adapters

- Python `steering/runtime.py` owns `Site`, `Context`, `Reading`, response-local
  `Subscription` state, ordered observation phases, and `apply_legacy`.
- `HookedModel` owns one runtime per arming handle. Its existing block-output
  dispatcher executes pre-action readings, the unchanged legacy chain, and
  post-action readings. Additional input hooks are registered only on requested
  blocks. Registration order cannot move a pre-action reading after steering.
- P3's `probe_observation.py` subscribes to that runtime. It retains its scoring,
  budget, evidence, and error semantics; position context now comes from the
  same offsets that govern steering. It verifies that its explicit decoder map
  is the actual map armed by the model adapter.
- Native `ResidualRuntime` dispatches the existing chain on `residualPost`
  (`.post` in Swift); the two vendored models expose `residualPre` (`.pre`) too.
  Native code can install readings through `withResidualRuntime`; it validates
  layer indices and clears state on success or thrown errors. A native runtime
  scope is single-use; attempting to share or reinstall it is an explicit error. Model execution
  remains confined to the existing model-container owner.

The legacy chain is an ordered action adapter, not a second implementation of
addition or ablation. Its owner calls remain unchanged: ablation removes the
specified subspace at every position using dimensionless lambda, whereas vector
addition keeps the prompt-end gate and its additive units. Repeated additions
keep their sequential rounding. Existing SAE latent edits, training deltas, and
legacy recorders stay in their original chain positions. There is no new SAE
policy or reinterpretation of a legacy reader as a behavior-changing policy.

## State, positions, and tensor behavior

Each Python subscription carries response identity (including condition, agent
or seat, and sample/turn information supplied by P3), prompt length, and state
keyed by provider ID. Multiple reading IDs may deliberately share a provider's
state within that subscription. Separate subscriptions and responses cannot
share that state. Absolute offsets identify consumed input tokens; prompt length
separates prefill from decode. No forward pass is added for the last sampled token.

Python's named study readings currently require one unpadded sequence, matching
P3's generation owner; an incompatible batch is an explicit error. The unchanged
legacy action ABI still handles its existing tensor batches. Native library
callbacks have separate state per provider and fixed batch row within a response;
rows must retain their identities throughout that scope. Padding masks and
reordered/dynamically packed sequences are not inferred or claimed supported for
study measurement. The native callback context identifies the row and positions;
it is not a portable probe-binding declaration.

A runtime does not make a single model safe for concurrent generations; existing
model-slot/container ownership must serialize execution. Nested Python arming
scopes retain their own subscriptions. Superseding an
abandonable streaming scope closes its subscriptions immediately; a delayed exit
cannot restore them over a newer response. Ending a scope clears provider state
and removes its requested input hooks. Provider failures propagate through P3's
existing partial-evidence path. Cancellation follows the existing generation
thread shutdown and arming lifetime; there is no hidden retry or fallback.

Callbacks are trusted in-process readers: they must not mutate inputs or consume
the generation RNG. Python enforces a None return, and Swift uses a Void callback.
This is not a sandbox for arbitrary code. No per-site HTTP, CLI, or UI calls,
unconditional tensor clones, or host scalar conversions are introduced. The
legacy mathematical path remains tensor-valued and differentiable where its
owners already support it. P3's existing detached CPU scoring remains unchanged.
Bounded study evidence is still P3's responsibility; an expert callback author
must bound any extra retained state.

## Scope and verification

P5 will add policy decisions from the declared pre-action snapshot, typed action
requests and timing checks, independently identified provider RNG streams,
logit/token-selection actions, and pinned custom-provider loading and authoring.
Those are not exposed by this slice. No logits hook, auxiliary model, layer
repetition, scheduling change, or generalized provider checkpoint is added.

Acceptance covers independently calculated addition/ablation, dtype-sensitive
legacy equality, pre/post phases independent of registration order, chunked
positions, same-ID state isolation, provider sharing across sites, failure and
superseded-stream cleanup, and a tensor-gradient fixture. P3's real sampled-model
noninterference and ordinary/panel study journeys continue to execute. Native
fixtures exercise the real vendored model boundaries and preserve legacy output.

The arithmetic audit pins unchanged scientific owners and native model code
outside the explicit dispatch substitution against P3. Hook orchestration and
lifecycle changes are behavioral work, not claimed mechanically unchanged.
Research-model overhead and live app/cluster qualification remain P7 work; the
user has deferred app use until the program is complete. Existing public support
claims must not treat this refactor as measured CUDA/MLX numerical equivalence.
