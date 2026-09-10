# Review: `codex/jlens-fitting-followup` @ `f150f5e` (on main `878bca2`)

Reviewer: the maintainer's integration agent, 2026-09-09. Read against
`docs/JLENS-FITTING-FOLLOWUP.md` and the findings in
`docs/REVIEW-JLENS-FITTING-BRANCH-2026-09-09.md`. Two commits, 23 files,
+656/−70. Main has not moved, so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward as it stands.** The branch answers the two
actionable findings of the fitting review without touching the numerical
loop: checkpoint continuation now compares a named numerical contract
rather than a hash of the driver source, and the cost arithmetic the guide
owed is computed from the exact model's cached configuration, shown in the
shared draft, the HTTP plan, the app sheet and the execution log, with an
honest "geometry unavailable" when nothing is cached. It also turns the
early-failure window (input capture, checkpoint capture, model load,
checkpoint restore) into a recorded failure with a phase, and announces the
minutes-long tensor hash before it starts. An audit pins the fitting loop's
AST to `878bca2` and recomputes the one legacy driver hash from Git, so the
compatibility bypass names exactly one reviewed source. The 27B target is
still unverified, by design, until the cluster pilot.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`878bca2`) is an ancestor of `f150f5e` |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches and every audit passes, including the new `audit-jlens-followup.py` at base `878bca2` (legacy hash recomputed from Git, loop AST identical, mutation control rejected) |
| Public scan, whitespace, vocabulary in the diff and both commit messages | clean |
| Execution-side binding | the engine compares only `inputSHA256` between the reviewed plan and execution (`scientific_execution.py`), so the cost review rides in the plan document and never in what the engine binds |
| Live cost review, no monkeypatching | drafted `jlens-fit` against a cached `Qwen/Qwen3-0.6B` snapshot: 27 source layers, width 1,024, matrix set 113,246,208 bytes, 1,024 backward passes per row; an uncached model id returns `geometryUnavailable` with no download attempted (`HF_HUB_OFFLINE=1`) |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | see §5 |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | see §5 |

## 3. What the branch does

**Compatibility contract** (`jlens_fit_identity.py`, new). The checkpoint
identity gains `fittingContract: jlens-fit-v1`. `verified_identity` checks
the saved identity against its own recorded digest, then `material` strips
only `runtime.driverSHA256` before equality; torch and transformers
versions, the reference commit and kernel hash, dtype, device, TF32 and
matmul settings, and the model and tokenizer hashes all stay bound. A
checkpoint with no contract is accepted only when its driver hash equals
the reviewed `878bca2` driver; any other contract or unknown legacy driver
refuses with a named reason. A mismatch lists the differing keys. The
report records both driver hashes and the decision under
`continuationCompatibility`. The driver hash itself now covers four files.

**Cost review** (`jlens_fit_review.py`, new). `estimate` is pure
arithmetic from width, layer count, dimension batch and prompt limit;
`cached_config` reads at most 1 MiB of the exact revision's `config.json`
from the local hub cache, never downloading, and tolerates a missing hub
package; `review` returns the estimate plus the worked example and the
verification note either way. The draft (`method_authoring.draft`), the
HTTP `input_plan`, and the model loader before weights all call it; the
app's review pane and progress labels render it.

**Failure records** (`jlens_fit_execution.py`). Run creation and
`on_run_created` now precede input capture inside the try, a `phase`
string advances through capture, checkpoint, load, restore, fit and
publish, and the failure record carries the phase and the source
checkpoint; an `OSError` writing that record is logged and the original
exception still propagates. The per-row loop is byte-for-byte the
`878bca2` loop, which the audit proves.

## 4. Findings

**N1 — a cache change between draft and publish refuses with the wrong
message.** The cost review is part of the draft document, so it is inside
`planSHA256`. `publish` re-drafts and compares hashes; if the model's
configuration enters or leaves the local hub cache between the two steps
(a model download in another window is enough), publication refuses with
"The answers or input bytes changed", which is not what happened. Rare,
harmless, and the repair is the same (draft again), but the message could
name the review as a possible cause.

**N2 — library versions stay in the equality check.** The fitting review
suggested moving torch and transformers versions to provenance; the
branch keeps them bound and says why in the handoff (a dependency upgrade
is not automatically compatible). That is the more conservative choice
and it is documented; the cost is that a routine dependency bump on the
cluster still strands every checkpoint made before it. Acceptable for
now; worth revisiting once a real continuation has been exercised.

**N3 — the device string is bound literally.** `runtime.device` comes
from the request (`cuda` by default). A checkpoint fitted with `cuda`
cannot continue under `cuda:0`, and the mismatch message will say
`runtime.device`. Correct in spirit; noting so nobody reads it as a
numerical difference.

**N4 — the legacy driver has no live instances.** The cluster engine is
still the September 3 deploy, which predates fitting entirely, so no
checkpoint carrying the `878bca2` driver hash exists anywhere. The legacy
path is verified by audit and test only.

**N5 — items still open, restated:** the hybrid 27B on CUDA is
unexercised; the dimension-batch default stays at 1 pending measurement;
partial runs and scratch checkpoints accumulate under `runs/` and
`.steerlab/jlens-fitting-state/` with no managed cleanup; the fit → collect
→ register walk is still a live acceptance item after the release-stage
deploy.

## 5. Landing shape

Fast-forward to `f150f5e`, then this review on its own commit. The branch
changed shipped Python and the compiled identity, so the app and its Python
payload are rebuilt together before the app is used with this main; the
engine on the cluster stays at its September 3 deploy until the release
stage.

Suite results on `f150f5e`:

- Python: 6,335 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,613 ExperimentKit tests.
