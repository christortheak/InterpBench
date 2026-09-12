# Review: `codex/jlens-assessment-reuse` @ `ebc9b64` (on main `224de64`)

Reviewer: the maintainer's integration agent, 2026-09-11. Read against
`docs/JLENS-ASSESSMENT-REUSE-HANDOFF.md` and §4 of
`docs/REVIEW-JLENS-SCALING-BRANCH-2026-09-11.md` (N1–N3). Four commits
(three code, one handoff), 18 files, +817/−41. Main has not moved, so a
fast-forward is available.

## 1. Verdict

**Landable by fast-forward with no landing fix.** The branch does what the
previous review's notes asked and nothing more. N1 is closed by restructuring
the assessment into two phases: one forward pass per usable row that saves
only the selected source and final-layer activations to a private temporary
directory, then one pass per source layer that reads and places each lens
matrix once and compares every saved row against it. The per-layer totals
accumulate the same contributions in the same row and chunk order as before,
so the report is unchanged to the byte apart from the new `resources` block;
the branch proves this against a byte-pinned copy of the landed owner, and
an AST audit proves the distance body and the eight-position chunk
arithmetic are unchanged after four declared substitutions. N2 keeps the
exact checkpoint-versus-mean check and names the offending run and layer.
N3 keeps the all-active-scientific-jobs concurrency rule and adds a
`capacity` explanation that is deliberately outside the plan hash. One
statement in the branch corrects a statement in my previous review; see §4.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`224de64`) is an ancestor of `ebc9b64`; `ebc9b64` touches only the handoff document |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches, all thirteen audits pass, including the new assessment audit and its two negative controls; bridge gates pass |
| Public scan, whitespace, vocabulary in the diff and all four commit messages | clean |
| Numerical equivalence | per-layer totals receive the same sequence of additions (rows in corpus order, chunks of eight in position order); `effectiveTopK` takes the same last value; matrix placement is `.to(device, float32)` once per layer instead of once per chunk; activations round-trip through safetensors in their native dtype, and `h` is promoted to float32 exactly where the old code called `.float()` |
| Hook lifetime | hooks are registered once for the distinct set of source and target layers and removed in `capture`'s `finally`, before any matrix is loaded |
| Scratch lifetime | `TemporaryDirectory` under `.steerlab/jlens-assessment-state/` in the execution root, resolved through the symlink-refusing `ordinary` helper; removed on success and on ordinary exceptions |
| Placement of cpu-class operations | `scientific_execution.submit` runs a `local` plan through `LocalExecutor().run([sys.executable, '-m', 'steerlab_server.api.scientific_execution', …])`: a child process of the controller, inside the controller's Slurm allocation |
| Round hash | `planSHA256` is the digest of the plan without `capacity`; `state`, `shards`, `availableSlots`, `submitIndices` and `childPlans` remain inside it, so a change that alters free slots still refuses a stale submit (the new test covers both directions) |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | see §5 |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | see §5 |

## 3. What the branch does

**Assessment** (`jlens_assessment.py`, new `jlens_assessment_inputs.py`).
`capture` forwards each row once with hooks that copy `output[0, positions]`
to the CPU in the activation's own dtype and save one safetensors file per
usable row; short rows keep their `skipped-too-short` status; the device of
each captured layer is recorded. `compare_layer` loads the two matrices for
one source layer, places them on that layer's recorded device as float32,
and calls `compare_row` for every saved row; the function scope releases the
pair before the next layer. `compare_row` is the old inner loop with the
matrix placement hoisted out. `preflight` and the completed report gain a
`resources` block: a float32 upper bound on temporary activation bytes
(`rows × positions × distinct layers × width × 4`), the per-row bound, the
float32 lens-pair size, and, after the run, the actual captured bytes, staged
row count, and the logical read and placement counts (`2 × source layers`,
or zero when every row was short). The block states that these are tensor
payloads, not peak memory.

At the 27B defaults (16 rows, 64 positions, 63 source layers plus the
target, width 5376) the activation bound is about 1.4 GB and the actual
bf16 payload about half that; the lens reads fall from roughly 200 GB to
about 15 GB.

**Merge** (`jlens_merge.py`). The exact `torch.equal(sums / nDone, mean)`
check stays; its message now names the source run directory and layer and
says to select matching completed output rather than edit a run. A new test
perturbs one float32 value by one ULP, re-hashes the file so the hash gate
passes, and confirms the merge still refuses before creating a run.

**Rounds** (`api/jlens_rounds.py`). `capacity_review` returns the limit,
occupied slots, the active scientific jobs sorted by id with kind, status
and whether each belongs to this round, the uncertain shard indices, and a
sentence explaining the count. The concurrency rule is untouched.

**Surfaces.** `FittingReviewSummary` renders the storage bound and the
lens-pair size in GiB with the not-peak-memory caveat; a new
`capacityLines` renders the capacity summary and one line per active job,
shown in `DiagnosticLifecycleSheet` beside the round controls. The method
guide gains three paragraphs (queue capacity, exact merge check and where
merge runs, assessment storage and its limits); the seed copy and compiled
resource are regenerated and the Python identity re-stamped.

**Audit** (`scripts/ci/audit-jlens-assessment.py`, registered in
`check-generated.py`). Confirms the byte-pinned fixture equals the owner at
`224de64`, that `distances` is unchanged, and that the `for start in
range(0, …, 8)` chunk in `compare_row` equals the old chunk after exactly
four substitutions (`len(positions)`→`len(h)`, the per-chunk `matrix.to(…)`
→`matrix`, `totals[str(layer)]`→`totals` twice, `config.topK`→`top_k`).
Negative controls flip the accumulator operator and the divergence scaling.

## 4. Findings

**No landing fixes.**

**C1 — correction to my previous review.** The scaling review's F1 said
cpu-class operations "run inside the controller process". They run in a
child process that the controller spawns and waits on, inside the
controller's Slurm allocation. The memory consequence is the same: the child
shares the controller job's 16 GB, one-core budget on this site, so the
per-layer streaming merge was still necessary. The branch's comment and
guide wording ("a CPU child within the controller's allocation") is the
accurate one.

**N1 — the audit needs `224de64` in the local history.** Like every other
audit in the set, it shells out to `git show`; the byte-pinned fixture makes
the runtime tests portable to a shallow clone, but the audit itself is not.
Consistent with the existing audits; noted so nobody expects otherwise.

**N2 — capacity lines follow the last shown result.** The sheet recomputes
`roundCapacity` from whatever result was last displayed, so showing a
cleanup or evidence result after a round plan clears the capacity lines
until the next round action. Harmless; the full JSON of the round plan is
still in the output view.

**N3 — an interrupted assessment leaves its scratch directory.** Documented
in the guide; the directory is small at the defaults and sits under the
execution root, which the existing custody cleanup already owns. No managed
cleanup is claimed and none is needed yet.

**N4 — live measurements are still owed.** The handoff is explicit: no
throughput or peak-memory claim is made for the CUDA model. The live
acceptance list gains one item: run an assessment on the 27B with the old
and new owners over the same small held-out set and confirm the reports
match, then record elapsed time and memory for the intended budget.

## 5. Landing shape

Fast-forward to `ebc9b64`, then one commit carrying this review. Shipped
Python and the compiled identity change, so the app and its payload are
rebuilt together. The cluster push still waits for the running continuation
job, and the controller restart after it; this branch now rides along with
`224de64` in that deploy.

Suite results on `ebc9b64`:

- Python: 6,426 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 4,910 passed and 5 skipped of 4,915 (290 SteeringKit and 4,625 ExperimentKit).
