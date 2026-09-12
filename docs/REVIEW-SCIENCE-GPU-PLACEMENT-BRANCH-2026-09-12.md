# Review: `codex/science-gpu-placement` @ `d52b107` (on main `f895430`)

Reviewer: the maintainer's integration agent, 2026-09-12. Read against
`docs/SCIENCE-GPU-TYPE-PER-REQUEST-HANDOFF-2026-09-12.md` (§2 remaining
items, §3 do-not list) and the branch's own
`docs/SCIENCE-GPU-PLACEMENT-IMPLEMENTATION-HANDOFF.md`. Four commits
(three code, one handoff), 24 files, +753/−41. Main has not moved, so a
fast-forward is available.

## 1. Verdict

**Landable by fast-forward with no landing fix.** The branch finishes the
three items the core slice (`c9f4f08`) left open and stays inside the
handoff's boundaries. Placement is chosen at plan time, never written into
the published request or the checkpoint identity; a round's top-up takes a
default type and per-shard overrides, both bound into the round's review
hash and refused for shards already attempted; the controller advertises
its declared vocabulary and capacities so the app never reads a stale local
profile; and the hardware actually seen at runtime is recorded as
provenance in fitting telemetry, benchmark results, and every managed
execution record, including records of failed runs. Nothing scientific
moved: the AST audits still pass, and the only fitting-side change is a
telemetry field, which changes the driver hash the way any telemetry edit
does and which the identity contract already treats as provenance.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`f895430`) is an ancestor of `d52b107`; `d52b107` touches only the handoff document |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches, every audit passes, bridge gates pass |
| Public scan, whitespace, vocabulary in the diff and all four commit messages | clean |
| Placement stays out of scientific content | `plan` adds `requestedGPUType` and `gpuReview` to the plan result only; `inputSHA256` is untouched (the fixture asserts child plans' input hashes are equal across types); no owner, identity, or request schema changed |
| Round hash binds placement | `plan['placement']` is present only when a default or an override was supplied and sits inside the digest; a submit whose placement differs from the reviewed plan refuses with "changed" before any scheduler call, and the fixture drives real child plans and real bundle rendering with only `SlurmExecutor.submit` replaced |
| Attempted shards are immutable | an override naming a shard whose status is not `pending` refuses; an uncertain submission keeps its recorded type from the state file's `placements` and cannot be redirected |
| Default plan shape (corrected after review) | With no placement arguments, the standalone scientific plan has no `gpuReview` and retains its previous shape. A round has no `placement` key, but its shard rows now include `gpuType`, which enters the digest even without overrides. Historical round hashes are not preserved; review the round again before acting. |
| Capability block | `sciencePlacement` is `available` only for a Slurm executor that is not a session worker; `defaultGPUType` is parsed from the controller's own gres; the Mac decodes it from `/api/capabilities` and treats absence as "server default only" |
| Hardware observation | `runtime_hardware.describe` tolerates a failing CUDA query and reports `deviceName: null`; `observe` resolves the device through the engine's own resolver (so an MPS Mac is not mislabelled CPU); `execute_packet` writes the observation to the record before model loading, so a failed load still carries it |
| Names and ordering assumed by the diff | `server_role` exists in `profile.py`; `Measurements.__init__` receives `torch`; `jlens_fit_review.review` emits `pilotMeasurement`; `executor_identity` is defined before the early record write |
| Full Python suite on the tip (main venv, cwd = worktree/Server, `HF_HUB_OFFLINE=1`) | see §5 |
| Full serial Xcode beta suite on the tip | see §5 |

## 3. What the branch does

**Placement owner** (`api/science_placement.py`). `gpu_type` parses the
type out of a gres; `validate` is the declare-or-refuse check the core
slice had inline, now shared with rounds; `capabilities` renders the
`sciencePlacement` block; `review` produces `gpuReview` (declared capacity,
`memoryFit: notChecked`, and, when a pilot measurement is attached, the
pilot's hardware plus a sentence saying the throughput was measured there
and not on the selected GPU).

**Rounds** (`api/jlens_rounds.py`, `api/diagnostic_transport_routes.py`).
`plan`, `submit`, and `cancel` accept `gpuType` and `shardGPUTypes`; status
and both merge actions refuse them. Overrides must name canonical
zero-based indices in the current `submitIndices`, so the researcher
reviews capacity first. Each child plan is produced with its effective
type; the state file gains `placements` written before each attempt so an
uncertain submission remembers what it asked for; status rows carry
`gpuType` from the submitted plan, the recorded placement, or the current
child plan.

**Hardware provenance** (`experiment/runtime_hardware.py`). One
best-effort describer replaces the benchmark's private one; fitting
telemetry includes it under `hardware`; the managed worker records it as
`runtimeHardware` beside the requested type.

**Mac** (`ScientificGPUPlacement.swift`, `ScientificGPUSelection.swift`,
both sheets). A picker over the controller's declared types with the site
default first, hidden for cpu-class operations; the plan invalidates when
the selection changes; review lines show the reviewed gres, the capacity
caveat, the pilot hardware, and per-shard placement. The lifecycle sheet
adds a top-up default and a per-shard picker for the shards in the current
`submitIndices`, with a clear-overrides control.

**Docs.** The J-lens guide gains the round-placement bodies and the
attempted-shard rule; the cluster-profile guide gains a section on where
GPU choices come from. Seed copy, compiled resource, and identity are
regenerated.

**Incidental fixture repair.** `StudyPackSurfaceTests.pack` now encodes its
task records with sorted keys, so two nominally identical fixture packs are
byte-identical; the production importer's byte-exact reuse rule was right
to refuse the unsorted ones.

## 4. Findings

**No landing fixes.**

**N1 — `requestedGPUType` on the default path is the site default.** The
execution record labels the gres the plan carried as "requested" even when
the researcher omitted a type and the controller default applied. Accurate
as a statement of what was submitted; slightly generous as a statement of
what was asked for. Harmless, since the plan's own `requestedGPUType` is
present only on an explicit request.

**N2 — the picker hides for operations the catalog cannot resolve.**
`usesGPU` returns false when the operation is unknown to the compiled
catalog, so a request for a brand-new operation shows no picker until the
app is rebuilt. The CLIs are unaffected.

**N3 — placement lines follow the last shown result.** The lifecycle
sheet's review lines, like the capacity lines noted in the assessment-reuse
review, are recomputed from whatever result was last displayed.

**N4 — live acceptance is still owed and now has its shape.** The two
benchmark legs running today (A100 and H100 on the same staged bundle) are
the cross-hardware numerical comparison; the round-level acceptance is a
small round with more shards than its initial concurrency (for example,
four shards and two initial slots, one submitted shard per GPU type), a later
top-up that submits remaining pending shards and leaves attempted shards alone,
and a check of each shard's recorded
`runtimeHardware` against its requested type.

## 5. Landing shape

Fast-forward to `d52b107`, then one commit carrying this review. Shipped
Python and the compiled identity change, so the app and its payload are
rebuilt together. The cluster push and controller restart wait for the two
benchmark jobs, which the current controller owns.

Suite results on `d52b107`:

- Python: 6,440 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 4,915 passed and 5 skipped of 4,920 (290 SteeringKit and 4,630 ExperimentKit).

## Post-review clarification (2026-09-12)

The default-plan row in §2 is corrected above: absence of a `placement` key
alone does not imply an unchanged round hash. The new `shards[].gpuType`
material is also hashed. The live acceptance in N4 now explicitly requires
more shards than initial concurrency so that the later top-up exercises pending
work, rather than an already exhausted queue. These are corrections to this
report, not changes to the server's hash or submission rules.

N2 and N3 are addressed in the separate
[UI follow-up handoff](SCIENCE-GPU-REVIEW-UI-HANDOFF.md); the original findings
above remain a record of what was reviewed at `d52b107`.
