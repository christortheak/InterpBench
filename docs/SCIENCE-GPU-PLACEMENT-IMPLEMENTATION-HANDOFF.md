# Scientific GPU placement: implementation and audit handoff

Date: 2026-09-12. Branch: `codex/science-gpu-placement`, based on main
`f895430`. This completes the remaining implementation in
[the per-request handoff](SCIENCE-GPU-TYPE-PER-REQUEST-HANDOFF-2026-09-12.md).
Review and integration remain with the maintainer’s agents, through the user.
No app installation, environment change, controller restart, or cluster
submission is part of this branch.

## Researcher-visible changes

- Scientific execution sheets offer the connected controller’s declared GPU
  choices, including its default. CPU operations do not offer GPU placement.
  Choosing another type requires reviewing a new execution plan.
- A materialized fitting round accepts a default GPU for each queue top-up,
  plus overrides for individual eligible shards. The app first reviews capacity,
  then exposes a picker for each shard in that top-up. Both CLI clients reach
  the same owner through their existing `science-call` commands and HTTP body.
- Queue reviews show placement per shard. Already attempted shards keep their
  recorded type. A lost scheduler response cannot be converted into a retry
  by choosing another GPU.
- Selected memory capacity is labeled as a site declaration, not a verified
  workload fit. Pilot throughput is explicitly tied to the pilot’s recorded
  hardware; choosing another GPU does not create a new speed measurement.
- Fitting telemetry and managed execution records report observed hardware,
  including CUDA device name, compute capability, and capacity where available.
  Requested placement remains separately recorded. Hardware observation failure
  is reported as unknown and does not block otherwise valid computation.

## Contracts and implementation

`api/science_placement.py` owns controller vocabulary, selection validation,
capacity context, and throughput guidance. `/api/capabilities` adds the optional
`sciencePlacement` block (`available`, `gpuTypes`, `defaultGPUType`, and
`gpuVRAMGB`). The Mac queries this endpoint on its captured client; it does not
substitute a local profile’s inventory. Older servers that lack the block keep
their default path available. GPU selection leaves the site’s partition,
constraints, GPU count, and other resource settings intact.

`api/jlens_rounds.py` accepts optional `gpu_type` and `shard_gpu_types` on its
reviewed top-up flow. HTTP uses `gpuType` and `shardGPUTypes`. Map keys are
canonical zero-based decimal shard indices in the current `submitIndices`;
values are declared GPU types. The top-up default and overrides are repeated
on submit, and on cancellation against that same review. Status and CPU merge
accept no placement arguments. Changed capacity or placement needs a fresh
review; clear obsolete overrides before reviewing a new set of eligible shards.

The child plan, scheduler bundle, and review hash carry the effective GPU.
The round hash also binds the supplied placement arguments, even if capacity
currently permits no submissions. Before attempting submission, controller
bookkeeping records both the attempted index and its placement. This allows a
status response to retain the requested type even when no durable child response
was received. The implementation does not adopt continuation jobs, redirect
existing allocations, refill automatically, or retry uncertain submissions.

The published scientific request, row partition, and checkpoint compatibility
identity are unchanged. The existing default scientific plan shape remains
unchanged when the GPU override is omitted. Explicit overrides add `gpuReview`
to the reviewed plan, alongside the selected resources. Managed science still
reports `memoryFit: notChecked`; no new numerical fit estimator is claimed.

`experiment/runtime_hardware.py` is the common best-effort observation owner.
Fitting and benchmark telemetry use it, and the managed worker records an
observation after input verification and before execution. Ordinary loading or
execution failures retain that observation in their execution record. It
reports the selected runtime device; it is not an inventory of a distributed
multi-device model. Hardware is provenance, not a new checkpoint equality field.
The fitting source-driver hash changes as normal when telemetry source changes.

`ScientificGPUPlacement` owns decoding, body construction, and readable review
lines; `ScientificGPUSelection` is the shared SwiftUI picker. The execution and
lifecycle sheets keep selection out of scientific authoring, invalidate reviews
on edits, and display effective placement. Shared J-lens guidance is regenerated
into both packaged clients. The cluster-profile guide explains the placement
boundary and the unchanged scheduler constraints.

## Audit focus and qualification

The regression tests exercise real child planning and bundle creation, replacing
only the final scheduler call. Mixed-type top-ups render different GRES headers,
preserve scientific input hashes, and reject stale placement before submission.
Other tests cover invalid maps, attempted-shard refusal, lost-response placement,
HTTP admission, selected capacity, unknown hardware, and failure evidence. Swift
tests cover controller discovery, CPU visibility, selection/body formatting,
review lines, and the existing plan/submit serialization paths.

This is an intentional orchestration and reporting change. No scientific-body
move or new numerical equivalence claim is made. Existing AST audits still guard
the reference fitting, assessment, and historical owner bodies.

Live acceptance after review and deployment:

1. Rebuild the app with its updated Python payload and update the engine at an
   agreed safe point; restart the controller only after active work permits it.
2. Verify the app’s choices against the connected controller. Plan one staged
   request with the default and another declared type, then check the scheduler
   preview and the submitted job record.
3. Use a small two-shard round to submit different GPU types. Verify each actual
   allocation and runtime device against requested placement, then collect the
   evidence. Exercise a later top-up without relabeling already attempted shards.
4. Run the separately approved cross-hardware benchmark. Mixed-GPU merging is
   permitted by the existing identity; numerical agreement still needs measuring.

Live CUDA and interactive app acceptance remain with the running/auditing agents;
this branch does not claim them from fixture tests.

## Verification and incidental fixture repair

- `d34902a`: server placement, round bookkeeping, hardware reporting, and guides.
- `b061e5b`: native controls and parity tests, plus the device-resolution repair.
- `968d80e`: stable byte encoding for the unrelated study-pack test fixture.
- Final full Python suite: **6,440 passed, 9 skipped, 8 warnings** (193.49 seconds).
- Generated resources, compiled Python identity, historical AST audits, and
  normal/release bridge gates pass on the final Python sources.

The first sandboxed Python attempt was interrupted during a Hub network retry.
The complete suite was then run with offline Hub resolution and normal local
process/socket access; no dependency installation or model download was needed.
Final hardware observation uses the engine’s device resolver, including its
`STEERLAB_DEVICE` override and Apple MPS path, rather than assuming CPU when
CUDA is absent.

One native rerun exposed a pre-existing test-fixture issue in
`StudyPackSurfaceTests.pack`: its embedded JSONL records were encoded without
sorted keys, so two nominally identical inputs could have different bytes. The
production importer correctly refused reuse. This branch adds `.sortedKeys` to
that fixture’s encoder, with a comment explaining byte-exact reuse. No production
study-pack owner or integrity predicate changed. Final serial Xcode beta verification with that fixture repair: **TEST SUCCEEDED**,
290 SteeringKit tests and 4,630 ExperimentKit tests (4,920 total, including five
explicit skips). The study-pack reuse test passed. The app target and its new
picker compiled successfully. The built CLI reference matches all 17 generated
regions. Public-content and whitespace checks are clean.

The suites ran serially against the same final Python sources. No code changes
followed the successful native run; this handoff is documentation only.

## Reproducing verification

Set `TEST_PYTHON` to an existing provisioned Python test environment. The
worktree has no venv of its own. Run the suites serially; offline Hub resolution
keeps these fixture tests from waiting on network retries or downloading models.

```sh
cd Server
HF_HUB_OFFLINE=1 PYTHONPATH=. "$TEST_PYTHON" -m pytest -q
cd ..
HF_HUB_OFFLINE=1 \
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON="$TEST_PYTHON" \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-science-gpu-build \
  CLANG_COVERAGE_MAPPING=NO
"$TEST_PYTHON" scripts/ci/check-generated.py --audits \
  --cli /private/tmp/interpbench-science-gpu-build/Build/Products/Debug/steerlab-cli
"$TEST_PYTHON" scripts/ci/public_scan.py
git diff --check
```

The Metal identifier above was verified as installed on this Mac. Use the
installed identifier on another build host. Xcode needs normal access to its
caches and toolchain assets outside a restricted file sandbox.
