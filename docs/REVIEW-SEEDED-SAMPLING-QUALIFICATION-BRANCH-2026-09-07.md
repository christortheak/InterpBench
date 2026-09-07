# Review: `codex/seeded-sampling-qualification` @ `0356abb` (on main `d84968c`)

Reviewer: the maintainer's integration agent, 2026-09-07. Read against
`docs/TECHNIQUE-PARITY-COMPLETION-HANDOFF.md` and
`docs/TECHNIQUE-PARITY-QUALIFICATION.md`. Two commits, 87 files,
+4,264/−387. Main has not moved, so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward, with one functional gap to close before the
feature is real for app users (F1).** The engine work is right: the seeded
sampler reproduces the pinned library's filter arithmetic exactly and draws
from a per-record key, seed derivation is bit-identical to the Python
engine, one sampler lives across the reasoning and answer phases, records
carry their policy and pairing index, the Python engine's scoped RNG now
also restores MPS state and serializes overlapping scoped records, and the
registration of managed techniques collapses into one specification per
operation with an audit proving nothing existing moved. What the branch
missed is the app: the Run Study button is still disabled for a warm local
design by a gate that predates this work, and five help strings still tell
the researcher the local engine cannot seed. From the CLI the feature
works; from the app it does not. This is a small, mechanical fix.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`d84968c`) is an ancestor of `0356abb` |
| Unified gates (`check-generated.py --audits`) | every generator matches and every read-only audit passes with its baseline, including the new registration audit with its body-mutation control |
| Worked example | `qualify-technique-example.py` passes on the tip (25 tests), now through the operation-spec route |
| Sampler arithmetic | read `SeededLogitSampler.swift` beside the pinned `TopPSampler` and `CategoricalSampler` in mlx-swift-lm 3.31.3: same bf16 promotion, same log-softmax, same top-p (ascending sort, cumulative mass above `1 − p`), same min-p threshold, same top-k partition, same temperature division, and argmax at temperature zero; only the categorical draw takes the explicit per-record state |
| Seed derivation | `StudySampling.deriveSeed` reproduces Python `derive_seed` for the three test vectors including the non-ASCII one, checked by running the Python function myself; multi-agent turns use the empty condition on both engines |
| Python RNG scope | `seeded_generation` still forks CPU and every CUDA device, seeds inside the fork, and now saves and restores MPS state in `finally` under a process re-entrant lock; the greedy path is untouched; `sampling.py` is outside the eleven audited owners, and the change is deliberate and tested |
| Provenance writes | `RunSamplingProvenance.write` uses `withoutOverwriting`; safe because `makeRunDirectory` always mints a fresh directory (this engine has no resume path), and the multi-agent writer guards on existence |
| Vocabulary in the diff and both commit messages; `git diff --check` | clean |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,180 passed, 9 skipped, 8 warnings, matching the branch's claim (run concurrently with the Xcode suite, no flake) |
| Full Xcode beta suite from the venv-less worktree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild`) | `TEST SUCCEEDED`: 288 SteeringKit + 4,608 ExperimentKit, matching the branch's claim; the opt-in cached-model replay test is skipped without its environment variable, as designed |

## 3. What the branch does

**Seeded sampling.** `SeededLogitSampler` holds a `RandomState(seed:)` per
record; `SeededGeneration` builds the token iterator with that sampler
through the library's public initializer so the same stream spans a
reasoning budget and its answer, and an unseeded call keeps the library
default. `StudySampling` is the Swift twin of the Python seed policy:
`derivedSHA256` when temperature is positive and samples per item exceed
one, otherwise the manifest's literal seeds; the ordinary and saved-agent
loops now iterate the policy's count and pass the seed and the actual
temperature into generation, stamp `seedInert` only at temperature zero,
and record policy, sample index and prompt token count. Multi-agent turns
derive from the empty condition, matching Python. Sampling provenance (a
sidecar with model snapshot, parameter dtypes, quantization, the pinned
dependency revisions from `Package.resolved`, OS and GPU) is written once
per run, and per model in multi-agent runs. The substrate picker no longer
pins stochastic designs to the server; the lifecycle gate for the
greedy-only policy and its refusal registry entry are gone.

**Python RNG.** MPS state restored on every exit; overlapping scoped records
serialized on one lock, which the handoff says can reduce same-process
sampled throughput. Cluster workers are separate processes, so the cluster
path is unaffected.

**Registration.** Every catalog operation is one JSON specification under
`docs/techniques/operations/`, with a registry of shared text and input-role
profiles. `check-operation-specs.py` generates the catalog, the interviews
and `operation_bindings.py`; `managed_methods` and `managed_inputs` read
from that module. `audit-operation-registration.py` proves the original
bindings, roles, catalog rows and dispatch bodies are preserved against
`d84968c`. `check-generated.py` is the one ordered entry point, with
`--audits` for the read-only checks and `--cli` for the reference. CI runs
it and the worked example.

**Qualification.** `scripts/qualification/backend_probe.py` compares CPU
and MPS residuals, extraction directions, interventions and an OptVec
train-then-evaluate journey on a matched small model, with tensor and
protocol identity checks; the local record and machine-readable reports are
committed. The MLX replay test is opt-in and matched 10 of 10 ordinary and
10 of 10 budgeted replays on a cached 4-bit model; the record keeps the
post-test Metal assertion visible. No profile is marked qualified.

## 4. Findings

**F1 — the app still blocks warm local runs and still says MLX cannot seed.**
`StudyRunControlsView.runControls` computes `requiresGreedy` for a local
selection with operative sampling and a non-zero temperature, disables the
Run Study button on it, and prints "Run Study requires saved Temperature = 0
for reproducible measured runs." The engine gate this mirrored is deleted on
this branch, so the app now refuses what the CLI runs. Five researcher-facing
strings carry the old rule: the sampling info popover, the samples-per-item
label in `StudySamplingControls`, the seeds help in `SeedsListRow`, the
warm-rehearsal caption in `MultiAgentPanelView`, and the comment block
above the gate. Fix: drop the gate and rewrite the five strings to say
that local measured runs seed each record, that equal seeds across backends
do not imply equal draws, and that repeatability is a per-backend
measurement. The parity brief asked for exactly this sweep.

**N1 — sampling provenance is written before the first generation.** If
model loading succeeds but the very first generation fails, the run
directory carries provenance for a run with no records. Harmless, and
arguably correct provenance.

**N2 — the throughput cost of the Python lock is process-wide.** Two
sampled records in one server process now serialize their whole
generation, not just their seeding. On the cluster every job is its own
process; on a local Python server with concurrent sampled jobs it will
show. Documented in the handoff; worth a line in the runtime doc.

**N3 — the sampler is unit-tested against the library's own key API, not a
model.** The replay measurement that proves end-to-end repeatability is
opt-in and was run once by the branch on one cached model; nobody else has
run it yet. The recorded Metal assertion at teardown is unresolved and the
record is honest about it.

## 5. Landing shape

Fast-forward, then one landing commit closing F1 (drop `requiresGreedy`,
rewrite the five strings) with the Swift suite rerun, plus this review. No
Python or seed change in that commit, so the compiled identity stays. The
owed app rebuild then makes seeded local sampling available in the app.
