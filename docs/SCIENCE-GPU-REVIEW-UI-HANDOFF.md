# Scientific GPU review UI follow-up

Base: `main` at `0fcc65e`. Branch: `codex/science-gpu-review-ui`.
Implementation commit: `daee0b3`; the following commit records this handoff and
the review-document corrections.

This follows N2 and N3 of
[the GPU-placement review](REVIEW-SCIENCE-GPU-PLACEMENT-BRANCH-2026-09-12.md).
It changes the Mac review experience and corrects two documentation details.
No Python engine, request schema, fitting identity, scheduler rule, or stored
research artifact changes.

## What changes for researchers

- A request unknown to the app's operation catalog now says its GPU requirements
  are unknown. A server plan can resolve the choice: Slurm execution enables
  the controller's declared GPU picker; CPU or local execution hides scheduler
  placement. Older servers without the optional capability block still use
  their default. This does not add authoring support for unknown methods: local
  packaging may still need an updated client.
- Submission reuses the exact GPU argument reviewed with the plan, including
  an omitted argument for the site default. Learning the executor from the
  returned plan does not silently change the submission argument. Changing the
  GPU choice requires a new review.
- A round's queue or merge review has its own retained snapshot: originating
  job ID, capacity and placement summary, and expandable full plan. It is
  explicitly labelled as the last review, not live status. Fetching evidence,
  checking receipts, or displaying cleanup output no longer replaces it.
- Changing round GPU selections, attempting a round action, or encountering an
  operation error invalidates approval while retaining the snapshot. Switching
  originating jobs clears it. A queue review cannot approve a merge, and a
  merge review cannot approve a queue action. Server-side hash checks remain
  authoritative.

## Code and regression coverage

`ScientificGPUPlacement.requirement` distinguishes an unknown local catalog
entry from a known CPU operation and defers to a server plan's execution
information. Tests cover unknown, local, CPU, and Slurm cases, including
server information that supersedes a local catalog hint.

`FittingRoundReview` owns the retained document and approval state separately
from the sheet's generic output. Tests cover invalidation without loss of the
snapshot, switching jobs, queue/merge approval separation, fresh confirmation,
and a missing or empty hash retiring the previous approval. The sheet displays
that state and captures the reviewed GPU argument before resolving the returned
plan. This is an intentional UI/state change; no mechanical scientific move is
claimed.

## Documentation corrections and deferred items

The previous review's statement that both default hashes were unchanged was
incorrect. Default standalone scientific plan shape is preserved. Round rows
now include `gpuType`, which is hashed even when no override is supplied;
historical round hashes are not preserved. The review now marks this correction
explicitly. No hash check has been relaxed.

N1 remains informational: `requestedGPUType` describes the scheduler request,
including the inherited site default. No metadata rename is needed here.

N4 remains live acceptance for the running agents after their reviewed deploy.
Use more shards than initial concurrency: for example, four shards, two initial
slots, and one initial shard per GPU type. After capacity becomes available,
submit the remaining pending shards in a later top-up. Confirm attempted shards
retain their placement, compare requested placement with allocation and recorded
`runtimeHardware`, and collect evidence. Cross-hardware numerical agreement is
still a separate measurement, not a consequence of placement tests passing.

## Verification

- Full Python suite: **6,440 passed, 9 skipped, 8 warnings**, 229.56 seconds.
- Full serial Xcode beta suite: **TEST SUCCEEDED**, 290 SteeringKit and 4,636
  ExperimentKit tests; **4,921 passed and 5 skipped** of 4,926 total. The six
  new regression tests passed, and the app target compiled.
- `check-generated.py --audits`: generated declarations, Python identity,
  maintained historical AST audits, and normal/release bridge gates all pass.
- Built CLI reference check: all 17 generated regions match.
- Public-content scan and whitespace checks: clean. Full code and doc diffs read.

The suites ran serially with offline Hub resolution, the existing Python test
venv, Xcode beta, the installed Metal toolchain, and derived data outside iCloud.
No dependency installation or model download was needed. See the commands under
[reproducing verification](SCIENCE-GPU-PLACEMENT-IMPLEMENTATION-HANDOFF.md#reproducing-verification)
for the same setup. Validation used the final source and test bytes in `daee0b3`;
only documentation changed afterward.

## Audit and landing

The maintainer's independent reviewing agent, through the user, reads the full
diff and verifies both suites and the maintained gates before integration.
This worktree does not merge main, install the app, push the cluster engine,
restart its controller, or submit research jobs.

After the app is rebuilt by the integration agents, check the lifecycle sheet:
review a round, fetch evidence or list receipts, and verify its labelled snapshot
and full plan remain visible. Change a GPU selection and confirm approval clears
while the snapshot stays. Change the originating job and confirm the snapshot
clears. Check a default server plan followed by an explicit GPU re-review, and
check that a CPU operation offers no scheduler GPU picker. These interactive
checks complement the state tests; they are not claimed from a package build.
