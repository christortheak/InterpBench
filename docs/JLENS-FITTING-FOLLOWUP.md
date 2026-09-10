# J-lens fitting review follow-up

Branch: `codex/jlens-fitting-followup`, based on landed main `878bca2`.
Implementation commit: `ea9018f`.
This addresses the actionable findings from the fitting review. The earlier
[implementation handoff](JLENS-FITTING-HANDOFF.md) describes the full workflow;
this document supersedes its source-hash compatibility and early-error details.

## Researcher-visible changes

- A documentation-only fitting-driver deploy no longer prevents continuing a
  checkpoint. The model, corpus, estimator, fitting geometry, numerical runtime
  settings, and dependency versions still match under `jlens-fit-v1`. Reports
  retain both driver hashes and the compatibility decision. Version differences
  are named explicitly in the repair message rather than treated as interchangeable.
- After a run directory is created, ordinary errors during input capture,
  checkpoint capture, model loading, and checkpoint restoration leave a failure
  record with the phase, reason, and source-checkpoint reference. The managed job
  learns the run location immediately. Earlier runs remain untouched. An abrupt
  kill or unavailable storage can still prevent a failure record; incomplete
  runs have no completion marker.
- The app and both clients share a fitting cost review. When the exact model's
  config is cached locally, it reports backward-pass counts, matrix storage,
  sums-plus-row CPU memory, and checkpoint size. Otherwise it says that geometry
  is unknown, provides a labeled example, and directs review to the prepared
  engine. HTTP execution plans include the same review. Estimates also appear
  in execution logs before loading weights.
- Checkpoint hashing announces the file size and verification phase. The app
  displays an explanatory progress label during request and file operations.
  Independent review-time and execution-time verification remain in place.

No automatic data authoring, model downloads, runtime override, run cleanup,
or cluster execution is added. The conservative dimension-batch default stays
at 1 pending actual measurements. The guide explains why a four-row pilot can
still be expensive, and why fewer batched backward passes do not imply a
proportional speedup.

## Compatibility details for reviewers

`jlens_fit_identity.py` separates the numerical contract from `driverSHA256`.
It verifies the saved identity's own digest before comparing normalized
identities. Only driver source provenance is excluded from equality. Torch,
Transformers, reference commit and kernel hash, dtype, tokenizer and model
configuration hashes, device, TF32 settings, and the remaining runtime fields
stay compared. Unknown additional identity fields are not silently dropped.

Old checkpoints without a named fitting contract are accepted only when their
recorded driver hash matches the reviewed driver at `68cb749`/`878bca2`. The
new audit recomputes that historical hash from Git. Unknown legacy drivers and
unknown fitting contracts require assessment; there is no broad legacy bypass.
Old checkpoint bytes are not migrated or rewritten. New output records the
recognition decision and retains the source checkpoint's original metadata.

Future changes to numerical fitting or continuation semantics must change the
contract or introduce an explicitly reviewed migration, with numerical fixtures.
A dependency upgrade is not automatically compatible. The current contract
does not offer an unchecked override or claim cross-runtime numerical equality.

`jlens_fit_review.py` computes advisory costs from a bounded local cached
`config.json` read at the exact model revision. Nested text configurations are
supported. Missing packages, missing metadata, and undecodable geometry produce
an explicit unknown estimate, not a download or a false promise of affordability.
The config hash accompanies known estimates. The estimate is included in the
reviewed plan's hash, outside frozen study manifests. Model execution still uses
the actual model geometry and the existing immutable model revision.

The early-error change extends the exception boundary; it does not alter the
per-row numerical loop. A failure while writing the failure record does not
replace the original model error. Partial runs and scratch checkpoints can
accumulate and remain available for deliberate recovery under existing policies.

## Verification

Focused regression suite: 101 passed. Full Python suite: 6,335 passed, 9 skipped,
and 8 warnings. Full serial Xcode beta suite: 290 SteeringKit tests and 4,613
ExperimentKit tests passed (`TEST SUCCEEDED`), with the app also compiled.
Both real CLIs passed interview → draft → publish → input-plan checks, including
the fitting cost review. The compiled CLI reference matches all 17 generated
regions. Unified generators, compiled Python identity, established audits, the
new fitting audit, bridge gates, public scan, and whitespace checks passed. The
complete source diff was read.

The suites ran serially using the main checkout's existing test Python. Xcode
used the beta developer directory, the installed Metal toolchain, external
derived data, and `CLANG_COVERAGE_MAPPING=NO`. No app was installed. Logs on the
implementation machine are `/private/tmp/jlens-followup-python-full.log`,
`/private/tmp/jlens-followup-swift-full.log`, and
`/private/tmp/jlens-followup-final-gates.log`.

Coverage includes source-only continuation with exact numerical equivalence,
recorded old/new provenance, rejection of changed dependencies and numerical
settings, identity corruption, contract-version changes, recognized and unknown
legacy drivers, early failures and unavailable failure-record storage, status
before tensor hashing, cache-only cost arithmetic, unknown geometry, shared
draft/publication, HTTP planning, and portable operation without GPU or hub
packages. The Swift parity test checks the new cost review through the actual
Python draft and publish owner.

`scripts/ci/audit-jlens-followup.py` is part of the unified audit command. It
verifies the legacy hash and compares the fitting loop's AST against `878bca2`,
with a numerical mutation control. Established generators, numerical-owner
audits, and bridge gates continue unchanged.

## Remaining qualification and landing

The actual requested 27B checkpoint on CUDA remains unverified. Tiny random
hybrid text and conditional-generation models have exercised the real loader,
reference adapter, and backward kernel on CPU; the full target, GPU kernels,
and large-model memory and timing still need the cluster pilot after the agreed
engine deployment. Do not turn an estimate into a measured performance claim.

The maintainer's auditing agents should read the diff, independently run both
suites and gates, and integrate through the researcher. Rebuild the app and its
Python payload together after landing. Live UI acceptance should check readable
cost guidance, visible verification activity, recovery messaging, and the fit →
collect → register workflow. This branch neither updates main nor installs the
app or deploys the cluster engine.
