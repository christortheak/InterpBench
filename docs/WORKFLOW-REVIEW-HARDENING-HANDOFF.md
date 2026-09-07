# Workflow review follow-up: implementation and audit handoff

Branch: `codex/workflow-review-hardening`. Base main: `7301e6c` (includes the
landed gradient interview fix and the prior custody/managed-method slices).
Main is unchanged by this work. The maintainer's reviewing/integration agents,
through the researcher, own independent diff review, both suites and any landing.

## Changes and acceptance criteria

### N1: bounded cleanup and lock duration

The cleanup policy remains limited to isolated, successful battery and stability
output copies. `plan` and `apply` hold the exclusive workspace switch lock, the
store's RLock and SQLite `BEGIN IMMEDIATE` while re-deriving exports, hashing
output and archive bytes, and inspecting dependency declarations. `apply` also
holds them through quarantine verification, removal and audit publication.
Same-controller bookkeeping waits; other controllers' writes may reach the
30-second SQLite timeout. Small output classes do not guarantee short I/O on
slow filesystems. No lock is removed by this follow-up.

Before admitting larger artifact classes, measure total critical-section time
on representative slow storage and exercise competing controllers. A proposed
shorter critical section must retain fresh dependency admission, exact byte
verification, no new job registration between admission and removal, and the
quarantine recovery rules. Hashing outside the lock without an explicit
revalidation/reservation design is not an acceptable optimization. This is a
scope gate, not a claim that large-output cleanup has been qualified.

### N2: import requires an existing destination

`import_evidence` refuses an absent or non-directory root before it creates
metadata, parents or output directories. Existing plain workspace directories
remain supported; no new workspace marker requirement is introduced.

One correction to the review's CLI attribution: the current Python `steerlab`
entry point already refuses a missing root as `notFound` (66). The defect was in
the shared import owner, reachable through other adapters. Tests retain the CLI
behavior and prove the owner refuses without creating parents. Existing
round-trip fixtures now deliberately create their destination workspace first.

### N3: consistent interview defaults

Absent, empty and whitespace-only optional answers resolve to the interview's
specified default when it has one. Required blank answers still refuse. The
resolved value is present in the request and `effectiveAnswers`; an engine
fallback cannot silently replace it. A regression uses the real fracture
interview's 0.8 threshold against the owner's 0.9 fallback. Explicit form values
and the advanced-settings conflict rule retain their previous behavior.

Integer values are preserved without passing through floating point. Python's
integer syntax, including underscores and Unicode decimal digits, remains
accepted and is tested; the contract no longer claims to preserve spelling.

The F1 test added on main checks interview-shaped keys, but accepts value
refusals and does not call `draft`. This follow-up adds actual fixture-backed
drafts for every one of the thirteen managed operations and passes every config
to `managed_methods.validate` with no mocked parser or swallowed exception. The
case set must match the managed registry. This verifies authoring-to-validation;
it is not GPU execution or proof that fixture artifacts establish scientific
applicability for real studies.

### N4: matching client sources without a checkout

The Mac adapter now reads its `CodeResources.serverPayload` and selects a
lightweight client interpreter independently of a code checkout. The existing
checkout venv remains a development fallback. A compiled source/seed fingerprint
is checked before dispatch; stale or missing identity refuses. Runtime repairs
are surfaced by the Mac adapter instead of being replaced by an archive-only
repair. The subprocess disables bytecode writes and reads the signed payload.

[The runtime guide](PYTHON-CLIENT-RUNTIME.md) specifies client-only setup from an
installed app, interpreter precedence, generation gates and update procedure.
A staged release payload is tested through the real Mac process adapter without
a checkout, with source drift refusing and no bytecode cache writes. Python
exercises the same boundary. The fingerprint pins sources, not dependency
versions. Clean-machine installation, upgrade and interactive GUI qualification
remain required; one-click provisioning and other local Python screens are
outside this follow-up.

### N5: reproduce the observation without guessing its cause

A deterministic test pauses the final durable job update after the live Job has
become cancelled. Both original local submission paths let their test `wait()`
return during that gap, while SQLite still correctly reports a nonterminal row
without the final evidence pointers. After release, the row contains the terminal
status and evidence together. This reproduces the reported missing-pointer
assertion without demonstrating persistent evidence loss or a faulty adoption.

The test helper now waits for a committed terminal row and returns that same
snapshot. Two deterministic regressions fail with the old helper and pass with
the new helper. No production job, cancellation, executor or adoption code is
changed. This establishes an explanation for the observed assertion; it does
not rule out every other cancellation race under live workloads.

### N6: meaningful lens audit controls

The scoped audit requires the declared import and every resolver substitution,
then permits only those changes against the pinned baseline. Negative controls
run through that same gate: removing the migration, changing a resolver's root
argument, and altering a consumer function body must all fail. An unmodified
AST-unparsed positive control ensures formatting alone is not what rejects the
body mutation. The eleven numerical owner AST comparisons remain unchanged.

## Validation and remaining qualification

Final full Python suite: **6,151 passed, 9 skipped, 8 warnings**. Final full
serial Xcode beta suite: **277 SteeringKit + 4,596 ExperimentKit passed**. The
separate app build, source-identity/resource/reference/bridge gates and scoped
scientific-owner audit pass. The actual release payload matches the compiled
source identity. Commands, logs and initial test failures are recorded in
[the validation ledger](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md). Focused tests include the thirteen authoring cases, import-root refusal,
real loopback transport, deterministic cancellation ordering, staged runtime
identity, and copied payload behavior. No model weights or live scheduler are
needed for these tests.

Read the complete diff, including generated resources and the new source-identity
gate. No studies, frozen manifests, existing runs, user environments or deployed
apps were edited. Reviewers should rerun both suites with Xcode beta, the Metal
toolchain, explicit test Python and external scratch as previously documented.
After landing, remaining qualification is live local/cluster researcher journeys,
Linux filesystem behavior, clean-machine runtime setup and interactive app QA.
