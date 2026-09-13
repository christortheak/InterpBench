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
the conflict. Addition uses the stored vector and a bounded multiplier; removal
uses a dimensionless fraction of the normalized subspace projection.

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

## Authoring and attachment

The app opens **Intervention policies…** from both Probes and Agents. The guided
starter asks for a portable probe, a fixed/threshold/adaptive rule, an action,
and strength bounds. It uses the probe's pinned model, rendering, and site;
expert settings can define more complex policies. It presents a review before
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
responses. Panel flattening preserves it. Scope reports summarize policy hashes,
bindings, timing, and action bounds. The app's Results view exposes those
records. A synchronous Python agent-chat response includes decision records;
streaming chat executes policies but is an exploratory text stream, not durable
study evidence. Policy-failure sidecars preserve partial observations and never
masquerade as completed responses or resume keys.

P5 does **not** implement native MLX policy execution, padded or multi-sequence
batches, arbitrary execution-flow changes, auxiliary model execution, gradient
training through policies, direct choice scoring, or policy-aware capability
battery qualification. Unsupported static-scoring paths refuse instead of
silently omitting the policy. Use sampled-response comparisons for the current
slice; record the missing battery qualification explicitly. P6 remains responsible
for richer analysis and the full qualification/remote acceptance walk. Live
GPU qualification and manual app interaction have not been established by the
synthetic/unit tests.

P6 release prerequisite: add explicit policy-runtime capability negotiation on
all remote submission and chat paths, including panel agents. An older engine's
agent decoder does not know this additive field and can otherwise ignore it.
The isolated-bundle test here uses the current engine; it does not qualify mixed
client/engine versions. Do not submit policy agents to an older deployed engine
or deploy this feature independently of that admission work. The app is not in
use during these implementation slices, and no live deployment occurs here.
