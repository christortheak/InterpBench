# Review: `codex/jlens-fitting` @ `c156894` (on main `d42376f`)

Reviewer: the maintainer's integration agent, 2026-09-09. Read against
`docs/JLENS-FITTING-PLAN.md` and `docs/JLENS-FITTING-HANDOFF.md`. Three
commits, 31 files, +1,665/−27. Main has not moved, so a fast-forward is
available.

## 1. Verdict

**Landable by fast-forward as it stands.** The slice adds one managed
operation, `jlens-fit`, that fits a Jacobian lens from a researcher's own
text through the pinned reference kernel, and it stays inside the boundaries
the plan set: no weight changes, no data generation, no downloads, a
four-row timing pilot by default, captured input bytes, float32
accumulation, explicit checkpoint continuation in a fresh run, and a
completed run that registers through the existing reviewed importer rather
than writing into the lens library directly. The estimator the wrapper
claims is the estimator the reference implements, and an independent
analytic fixture proves it rather than asserting it. What the branch cannot
show is the thing the researcher actually wants: a fit on the intended
hybrid-attention 27B checkpoint on a GPU. That is a live pilot after the
release-stage engine deploy, and the handoff says so.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`d42376f`) is an ancestor of `c156894` |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches and every audit passes; the registration audit strips four exact, reviewed dispatch additions before comparing owner bodies to its historical base and carries mutation controls for each |
| Public scan, whitespace, vocabulary in the diff and all three commit messages | clean |
| Estimator semantics | the reference module's own docstring: a one-hot cotangent at every valid target position, so the gradient at a source position is the sum over later targets, then the mean over source positions; the wrapper adds equal-weight averaging over usable prompts and never enters the kernel. The test's expected matrices are derived by hand from a two-layer causal fixture, including the suffix multiplicity the summed-target reduction produces |
| Reference pin | the local engine's `jlens` install records the same commit the loader requires, so the pin check is exercisable here |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | 6,315 passed, 9 skipped |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | `TEST SUCCEEDED`: 290 SteeringKit + 4,613 ExperimentKit |

## 3. What the branch does

**Configuration and inputs** (`jlens_fit.py`). A frozen config with model,
40-hex revision, a pinned `{path, sha256}` corpus, optional source layers,
prompt limit, token limit, skipped leading positions, dimension batch,
checkpoint interval, dtype, device, declared tier and an optional pinned
checkpoint. Corpus rows are UTF-8 JSONL with exactly `id` and `text`, unique
ids, and a 64 MiB bound. Preflight reads and hashes the inputs and, for a
continuation, checks the checkpoint's identity, dtype and progress before any
model is loaded. Authoring imports no GPU package; a test forbids it.

**Model adapter** (`jlens_fit_model.py`). Loads an already-prepared,
unquantized checkpoint offline, choosing the causal or image-text loader by
architecture, with eager attention, caches off, and no forced BOS or chat
template. It refuses a reference package at any other commit than the pinned
one, and records the runtime identity: reference commit, kernel hash, torch
and transformers versions, model config and tokenizer hashes, dtype, device,
matmul precision and TF32 flags, and a hash of the three driver files.

**Execution** (`jlens_fit_execution.py`). Creates the run directory and
reports it before loading the model, captures the corpus bytes into the run,
iterates rows in file order, skips only rows too short for the position mask
and records each skip, calls the reference kernel per prompt, checks every
returned matrix is finite float32 of the right shape, accumulates float32
sums, and checkpoints every N rows to a scratch directory outside `runs/`,
publishing each snapshot atomically before pruning the previous one. A
completed run holds the mean Jacobians, the final checkpoint, the report,
the resource estimate, per-row timings and an `artifact-description.json`
carrying the declared tier and fit dtype. Failures write a failure record
naming the last checkpoint and leave the scratch in place.

**Continuation.** A new request may name a completed checkpoint's
`state.json`; its `sums.safetensors` travels through the input closure. The
identity (estimator, model, revision, corpus hash, layers, geometry, runtime)
must match exactly, the prompt limit may only grow, and the old run is never
touched. A test shows continuation equals an uninterrupted fit to the byte.

**Surfaces.** The operation is a per-operation specification with its
interview and binding, so the catalog, both clients, the HTTP plan and
submit routes, and the app's authoring sheet gain it through the existing
generators. The J-lens panel gets a "Fit a new lens…" button that opens the
shared sheet, with corpus guidance specific to fitting and a copyable
corpus-authoring brief. The importer accepts and records `fitDtype`.

## 4. Findings

**N1 — checkpoints are bound to a hash of the driver source, which a deploy
invalidates.** The continuation identity includes `runtime`, and `runtime`
includes `driverSHA256`, the hash of the three fitting modules, plus the
torch and transformers version strings. Any engine deploy between a failed
job and its continuation, even one that only changes a docstring in the
driver, makes every existing checkpoint unusable with a message about a
runtime mismatch. Binding to the reference commit, kernel hash, dtype,
model config and tokenizer hashes is what protects the numbers; the driver
hash and library versions belong in the receipt as provenance, not in the
equality check. Worth loosening before the first long fit on the cluster.

**N2 — the default settings make the pilot slow, and the guide should say
the arithmetic out loud.** With `dimBatch` at 1, each usable row costs one
forward pass and `hiddenSize` backward passes through the whole model; the
reference's own default is 8. For a model 5,376 wide with 61 source layers,
that is about 5,400 backward passes per row, the accumulated sums are about
7 GiB of CPU memory with another 7 GiB for the per-row matrices, and each
checkpoint writes about 7 GiB. The four-row pilot is the right first step
precisely because these numbers are large; the guide's cost paragraph should
carry one worked example so a researcher can set expectations before
submitting.

**N3 — a continuation hashes the whole checkpoint tensor file at plan time
and again at execution.** Correct, and for a multi-gigabyte checkpoint it
is minutes of controller-side reading during review. Acceptable; noted so
nobody reads it as a hang.

**N4 — a failed fit leaves a partial run directory under `runs/`.** It
carries the failure record and no completion marker, and the branch
promises no cleanup of fitting runs or recovered checkpoints. Consistent
with the immutability rule; the researcher should know these accumulate.

**N5 — the loader chooses the image-text class for any conditional
generation architecture.** Right for the Gemma family; a text-only
conditional-generation architecture would refuse at load with a plain error
rather than fit. Fine as a refusal, and worth remembering when the next
model family arrives.

**N6 — the real target is unverified, by design.** The hybrid-attention 27B
checkpoint the researcher asked for has linear-attention layers whose
backward path through the reference's retained-graph pattern has not been
exercised anywhere; the toy fixtures prove the driver, not the model. The
pilot on the cluster, after the release-stage deploy, is the gate: peak
memory, seconds per row, finite outputs, skipped rows, and whether the
backward runs at all.

## 5. Landing shape

Fast-forward to `c156894`, then this review on its own commit. The branch
changed shipped Python and the compiled identity, so the app and its Python
payload are rebuilt together before the app is used with this main; the
engine on the cluster stays at its September 3 deploy until the release
stage, so no fit can run there yet.

Suite results on `c156894`:

- Python: 6,315 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,613 ExperimentKit tests.
