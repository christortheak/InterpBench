# Review: `codex/science-gpu-review-ui` @ `ef122fc` (on main `0fcc65e`)

Reviewer: the maintainer's integration agent, 2026-09-12. Read against
N2 and N3 of `docs/REVIEW-SCIENCE-GPU-PLACEMENT-BRANCH-2026-09-12.md` and
the branch's `docs/SCIENCE-GPU-REVIEW-UI-HANDOFF.md`. Two commits (one
code, one docs), 8 files, +307/−38, Swift and documentation only. Main has
not moved, so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward with no landing fix.** The branch replaces the
lifecycle sheet's loose round-plan state (three optionals and a toggle
recomputed from whatever result was last shown) with one small value type
that owns the last reviewed round document, its kind, its hash, and its
approval. A review is retained as labelled history; only a hash of the
matching kind can authorize an action; selection changes, attempted
mutations, and errors retire the approval without discarding the review;
switching jobs clears everything. The GPU picker's visibility now comes
from a three-way requirement (unknown, not applicable, GPU) that a server
plan can resolve, and submission reuses exactly the GPU argument the plan
was reviewed with. The engine is untouched.

The branch also corrects my previous review, and the correction is right:
a round's shard rows now carry `gpuType` on every row, rows are inside the
round's digest, so a round planned with no placement arguments hashes
differently from one planned before `0fcc65e`. The standalone scientific
plan's default shape is unchanged as stated. I have verified the claim
against `jlens_rounds.action` and let the edited row and the appended
clarification stand in that document.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`0fcc65e`) is an ancestor of `ef122fc`; `ef122fc` touches only documentation |
| Unified gates and audits, bridge gates | pass; no generator or identity changed |
| Public scan, whitespace, vocabulary in the diff and both commit messages | clean |
| Approval separation | `hash(for:)` returns the hash only when the snapshot's kind matches, so a queue review cannot authorize `merge-submit` and a merge review cannot authorize `submit` or `cancel`; the fixture covers both directions and a document with a missing or empty hash |
| Submission uses the reviewed argument | `plannedGPUType` is captured from the selection used for the plan before the requirement is re-resolved from the plan's `executor`, and submit passes that captured value, including nil for the site default |
| Requirement resolution | server `compute: cpu` or `executor: local` → not applicable; `executor: slurm` → GPU; otherwise the compiled catalog; unknown operations stay unknown until a plan answers |
| Round-hash correction | `plan['shards']` is `rows`, every row now has `gpuType`, and `planSHA256` digests the plan without only `capacity`, so default round hashes changed at `0fcc65e` |
| Full Python suite on the tip (`HF_HUB_OFFLINE=1`) | see §5 |
| Full serial Xcode beta suite on the tip | see §5 |

## 3. What the branch does

`FittingRoundReview` (ExperimentKit): `record` stores the document, kind,
summary lines, and hash and clears approval; `invalidate` clears hash and
approval but keeps the snapshot; `changeJob` resets. `ScientificGPUPlacement
.requirement` replaces the boolean `usesGPU`. The lifecycle sheet shows the
snapshot with the label "Last reviewed … plan for round … — not live
status", an expandable full plan, and a prompt to review again when the
approval is gone; the toggle is disabled without a hash. Six new tests.

## 4. Findings

**No landing fixes.**

**N1 — the standalone execution sheet still hides the picker for
unresolved operations until a plan is reviewed.** By design; the sheet
says so in a caption. A researcher on a brand-new operation reviews once
with the default, sees the executor, and reviews again with a type.

**N2 — N4 of the previous review stands, now with the corrected shape:**
a round with more shards than initial concurrency, one shard per GPU type
in the first top-up, a later top-up for the remaining pending shards, and
the recorded `runtimeHardware` compared with the requested type.

## 5. Landing shape

Fast-forward to `ef122fc`, then one commit carrying this review. Swift
app code changed, so the app is rebuilt and installed; the shipped Python
payload and identity are unchanged, so no cluster push is owed for this
branch.

Suite results on `ef122fc`:

- Python: 6,440 passed, 9 skipped, 8 warnings.
- Swift: `TEST SUCCEEDED`, 4,921 passed and 5 skipped of 4,926 (290 SteeringKit and 4,636 ExperimentKit).
