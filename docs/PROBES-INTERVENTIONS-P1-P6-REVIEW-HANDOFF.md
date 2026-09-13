# Probes and interventions: P1–P6 review handoff

P6 branch: `codex/probe-policy-evidence`, based on P5 `fad2ee2`.
The worktree is `/private/tmp/interpbench-intervention-policies`.
Main at the start of P6 was `3457d89`, already included in the branch.
The branch contains the earlier unlanded P3–P5 work. Review this complete stack
before integration; do not treat P6 alone as a deployable feature.

## What the researcher can do

An agent is the experimental object: a base model with its chosen static vectors,
adapters, and optional intervention policies. A study can separately attach
probes as measurements. The same fitted probe can supply a policy's input, but
reading the model and changing it remain different operations.

P1–P2 provide portable linear and small nonlinear probe artifacts, managed
capture, training, evaluation, shared authoring, and a Probes library. P3 adds
study measurements, recording schedules, and token alignment. P4 provides named
residual sites with explicit ordering around existing steering. P5 adds immutable
policies, bounded fixed/threshold/adaptive decisions, trusted expert providers,
vector actions, and token constraints, attached to new agent versions.

P6 makes the resulting evidence useful and transportable:

- New run bundles declare required instrumentation runtimes outside the hashed
  study document. Both clients inspect support before submission, and continuation
  checks the retained job's bundle requirements. Stored study and chat admission
  inspect agent definitions, including panel seats. The controller also checks
  its actual GPU-session worker before forwarding an instrumented chat. Queued
  instrumented jobs use a guarded entry point that checks the child environment;
  an older installation cannot silently execute an uninstrumented substitute.
- Version-2 policy records distinguish requests from acknowledged actions.
  An action is acknowledged only after the complete site operation succeeds.
  Conflicts and failed operations cannot appear as successful applied strengths.
  This acknowledgement establishes tensor application, not behavioral benefit.
- Records retain absolute consumed/predicted positions and token IDs, policy and
  probe identities, provider/asset hashes, reset/RNG conventions, and ordering.
  Host timing is explicitly distinct from synchronized GPU measurements.
- `science evidence-analyze <run-or-jsonl>` on either CLI and the workbench
  `POST /api/science/workspace/evidence-analyze` use one offline Python owner.
  The app's Results review opens the same analysis, with condition/agent/artifact
  filtering. It reads the complete source, independently of the response preview
  limit, and returns the SHA-256 of the bytes it summarized.
- Summaries separate recorded scores, requested strengths, applied strengths,
  exceptional outcomes, uninstrumented responses, and budget omissions. They
  are descriptive, not confidence intervals over supposedly independent tokens.
  Existing behavioral analysis also writes `instrumentation-summary.json` in
  its new analysis run. Existing statistical calculations remain unchanged.
- Python and native evidence import validate instrumentation structure and token
  alignment after archive hashes pass, before publishing any collected run.
  Partial and legacy evidence remain inspectable; legacy requests are never
  inferred to have been applied. Collection preserves original bytes.

## Review map

Read the [phase-1 plan](PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md),
[study measurements](PROBE-STUDY-MEASUREMENTS.md),
[runtime contract](PROBE-INTERVENTION-RUNTIME.md), and
[policy contract](INTERVENTION-POLICIES.md). The prior
[P5 handoff](INTERVENTION-POLICIES-HANDOFF.md) records its own review point;
this document supersedes its P6 pending list.

P6 owners:

- `instrumentation_contract.py` / `InstrumentationSupport.swift`: versioned
  requirements and client admission. Packaging inspects raw JSON so a typed
  decoder cannot discard an unknown agent field before the check. Requirements,
  archive contents, and hashes derive from the same captured bytes.
- `instrumented_bundle.py`: queued-worker admission before CLI execution;
  ordinary studies retain their existing execution command.
- `policy_execution.py` / `steering/policy_actions.py`: response-owned action
  acknowledgements and evidence version 2. Existing legacy action math is untouched.
- `instrumentation_evidence.py`: streaming validation and descriptive reduction.
  `InstrumentationEvidence.swift` mirrors structural admission, with shared
  fixtures; the scientific summary has only one owner. JSONL processing is
  streaming, with a 64 MiB limit per response row and an explicit error for larger
  rows; source evidence is never truncated or modified.
- `InstrumentationEvidenceView.swift`: Results presentation and filtering.
- Bundle import owners: validate staged files before the first publication.
  No frozen study, completed run, or original agent is rewritten.

The historical AST gates remain active. The managed-owner gate allows exactly
one added descriptive-report block in the analysis workflow and continues to
compare all existing scientific calculations. This is a behavioral feature,
not a claim that the whole slice is a mechanical move.

## Validation and integration

Verification uses the existing Python test environment and Xcode beta with the
installed Metal toolchain. Suites run serially, with network model access disabled.

- Full Python suite: **6,579 passed, 9 skipped, 8 warnings**.
- Full Xcode beta suite: **TEST SUCCEEDED; 295 SteeringKit and 4,649
  ExperimentKit tests passed**. The new native evidence test called the real
  Python analysis owner. Manual UI interaction remains a P7 item.
- Generated resources and CLI reference, scientific AST audits and negative
  controls, bridge gates, whitespace, and public-file hygiene pass.
- The complete diff has been read locally. Independent reviewers must report
  through the user before integration; this is not independent approval.

Reproduce from this worktree, setting `TEST_PYTHON` to the existing test venv's
Python (no environment installation is required by this handoff):

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=Server "$TEST_PYTHON" -m pytest Server/tests -q
HF_HUB_OFFLINE=1 DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON="$TEST_PYTHON" \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-science-gpu-build CLANG_COVERAGE_MAPPING=NO
"$TEST_PYTHON" scripts/ci/check-generated.py --audits \
  --cli /private/tmp/interpbench-science-gpu-build/Build/Products/Debug/steerlab-cli
"$TEST_PYTHON" scripts/ci/public_scan.py
git diff --check
```

Full-suite logs for this review are `/private/tmp/p6-python-complete.log` and
`/private/tmp/p6-native-complete.log`. Generated/audit results are in
`/private/tmp/p6-gates-verified.log`.

No app installation, cluster deployment, controller restart, model download,
or research job is part of this implementation or its landing authorization.

## P7 qualification boundary

Python Compute is the supported policy and study-probe execution route. Native
MLX dynamic policies, padded batching, auxiliary models, execution-flow changes,
gradient training through policies, direct choice scoring, and policy-aware
capability-battery execution are not added here. A sampled-response study is the
current complete-agent comparison. Unsupported execution must explain the
supported route, never silently measure a policy-free substitute.

P7 must exercise real local and remote checkpoints, controller/worker upgrades,
managed submission, cancellation/continuation, collection, Results interaction,
and both app and agent journeys. Measure overhead with no instrumentation,
readings only, fixed policies, and conditional policies. Host timings in ordinary
records do not establish kernel latency, throughput, memory use, or scientific
qualification on a research model. Preserve missing qualification as an explicit
status and retain useful partial evidence.
