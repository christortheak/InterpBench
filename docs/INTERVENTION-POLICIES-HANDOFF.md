# P5 intervention policies: reviewer handoff

Branch: `codex/intervention-policies`.
Base: completed P4 `f5f790d` on `codex/intervention-runtime`.
Main at the start of this slice: `3457d89`, already included in P4.
This worktree includes the P3/P4 commits that had not yet landed on main; review
P5 against `f5f790d`, and review the complete branch as appropriate before landing.
No main checkout, installed app, cluster controller, or live study was changed.

## What changed for a researcher

Probes and Agents now open **Intervention policies…**. A researcher can select a
fitted probe, say when an action should occur, bound its strength, and choose a
vector change or exact token action. The probe supplies the scientific input
binding. The review exposes exact inputs and limitations before saving. A
separate attachment review creates a new agent version. A policy-free copy
retains the same static vectors and adapters for comparison.

The native CLI, portable CLI, and workbench HTTP route call the same Python
policy authoring owner. All six verbs are discoverable in generated help and
reference material, and the shared readers guide explains the conceptual
workflow to agents. Advanced settings support weighted combinations of probes
and a trusted Python provider with pinned source and assets. Policy execution
is through Python Compute, using existing sampled-study and panel-turn drivers;
Results exposes the resulting decision records. Python agent chat also executes
policies. The editable Playground controls and native MLX execution give a
specific supported-route explanation instead of dropping policy settings.

See [INTERVENTION-POLICIES.md](INTERVENTION-POLICIES.md) for the complete schema,
provider ABI, action ordering, commands, examples, and implementation limits.

## Owners and invariants to review

- `policy_artifacts.py`: stdlib validation, exact UTF-8 attachments, strict
  scientific bindings, bounds, rule references, and document limits.
- `policy_authoring.py`: bounded ordinary-file reads; resolves probe bytes,
  vector tensors/sidecars, and provider assets into self-contained documents;
  re-reviews before create-only publication. Agent copies retain unknown fields,
  preserve static components, and remove an inherited promotion certificate.
- `policy_execution.py`: current-forward probe scores, response-local state,
  dedicated provider RNG, typed decisions, bounded events/failure summaries,
  and partial failure sidecars. Fitted probe parameters are prepared once per
  device/response. States, caches, and providers close immediately when a stream
  is superseded, not merely when its suspended generator is later collected.
- `steering/policy_actions.py`: new policy arithmetic. All policy ablations at
  one site use a joint orthonormal subspace and must agree on the per-position
  removal fraction. Additions follow in declaration order. Existing legacy
  scientific owners are unchanged. Policy ablation follows the legacy chain,
  so it can remove a component a static injector just added; this is explicit.
- `steering/runtime.py` and `hooks.py`: read-only `Reading` and action-producing
  `DecisionProvider` are distinct declarations. Decisions see isolated copies
  of the same snapshot. Input hooks now return changed residual tensors.
- `generate.py`: appends the policy processor after the finite-logit guard.
  Transformers applies temperature/filtering afterward. Allowed/forced tokens
  intersect, never unmask pre-existing forbidden tokens, and never bypass the
  generator's EOS, cancellation, or maximum-length behavior.
- Optional `interventionPolicies` embeds exact policy text and SHA-256 in each
  new agent. Absent fields retain historical encoding. Existing attachment and
  bundle owners carry the embedded bytes; no manifest revision field or frozen
  file rewrite was introduced. Swift editor copies preserve the new field.
- Ordinary and panel generation retain `interventionDecisions`. The stripped
  baseline excludes policies. Static-only execution paths explicitly decline
  policy agents rather than reporting a measurement of a different agent.

The changes are behavioral, not a claimed mechanical refactor. The historical
lazy-import audit still proves the original move, then exempts the intentionally
extended agent schema and scenario executor while checking every other body
against the exact declared policy-admission changes. The P4 audit continues to
pin the legacy action owners, native model arithmetic, and P3 probe scoring.

## Validation

Final full-suite verification reported **6,559 Python tests passed, 9 skipped,
and 8 warnings**, followed by **295 SteeringKit tests and 4,645 ExperimentKit
tests passing** (`TEST SUCCEEDED`). These passes include the bounded failure
summaries, nonlinear scoring, and malformed-input regression checks. Logs are
`/private/tmp/p5-python-complete.log` and `/private/tmp/p5-native-complete.log`.
Generated-resource, CLI-reference, AST, and bridge checks pass against the newly
built CLI; the gate log is `/private/tmp/p5-gates-complete.log`. The public scan
and whitespace checks are clean. The implementation diff was read before commit.

Focused policy tests exercise an actual in-memory Transformers
Llama generation, independent numeric probe/action expectations, state and RNG
isolation, superseded streams, failed providers, token conflicts, immutable
publication, input drift, HTTP/CLI owners, collected agent execution, ordinary
study resume, and panel baseline stripping. No model download is involved.

Native tests add real Python-owner publication and attachment, exact Swift
round-trips, digest failures, and native execution admission. The full suites run
serially using the existing Python environment and Xcode beta/Metal toolchain.
Derived data stays outside the checkout. No dependency install, app install,
cluster push, or live experiment is part of this work.

## Remaining work and landing process

1. Independent reviewing/integration agents read the complete diff, run both
   suites, and run generated-resource, CLI-reference, historical AST, bridge,
   vocabulary, and whitespace gates. Review/landing remains through the user.
2. P6 must add explicit policy-runtime capability negotiation to every remote
   submission and chat path, including panel agents. **An older engine can ignore
   the additive agent field.** The isolated-bundle fixture uses this engine;
   mixed-version deployment is not qualified. Do not deploy policies separately
   from that admission work. The app remains unused while these slices finish.
3. P6 also completes richer decision analysis, managed remote execution and
   evidence acceptance, and integration with the remaining measurement paths.
   Direct choice scoring and capability-battery qualification do not currently
   execute policies; report that limitation rather than borrowing static-agent
   evidence. Native MLX policy execution and dynamic batching are not claimed.
4. P7 performs live local/remote numerical and overhead qualification, plus a
   manual app walkthrough. Synthetic tests establish implementation behavior,
   not qualification of a particular research checkpoint or GPU.

Fast-forward only after the independent review and integration checks. Do not
install the app, restart a controller, or submit a job as part of landing without
separate authorization for those operational actions.
