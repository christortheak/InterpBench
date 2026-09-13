# P4 residual runtime: reviewer handoff

Branch: `codex/intervention-runtime`, based on completed P3 `bda6d71`.
Main's benchmark-document update (`3457d89`) is integrated by `df14830`.
At implementation start, P3 had not yet landed on main. This branch includes its
commits, and reviewers should distinguish the P3 range from P4's changes.

## Scope

P4 routes existing residual actions and study readings through explicit named
site dispatch. It preserves legacy scientific owners and introduces no policy
settings, agent fields, sampling actions, artifact format, or deployment.
The [runtime contract](PROBE-INTERVENTION-RUNTIME.md) explains the behavior and
boundaries; P5 remains the policy and agent-attachment slice.

Python owns typed residual sites, observation context, and response-local reading
subscriptions and state. The existing HookedModel output hook dispatches phases
around the same action chain; requested input hooks share that scope. P3 probe
scoring and evidence methods remain unchanged apart from using the shared offset.
An explicit check verifies that the probe's declared block mapping is the one
actually armed by the model. Superseding an abandonable stream closes the old
runtime; a delayed cleanup cannot restore it over the new response.

Native Qwen and Gemma adapters expose input/output boundaries and dispatch their
unchanged action chain through `ResidualRuntime`. `withResidualRuntime` validates
sites and closes reader state on scope exit. This is a native library runtime,
not a claim that portable probe scoring has been implemented in MLX. Study
measurements still execute through Python Compute, driven by the existing app,
both CLI clients, and workbench HTTP operations.

## Review carefully

- The adapters call the existing addition, ablation, SAE, and training owners in
  their original order. No folding, new dose conversion, mask, or precision
  change occurs. The current scope descriptors retain their meanings.
- Pre-action and post-action timing is explicit. Within each phase, declaration
  order replaces the former reverse ordering of multiple prepended Python
  hooks. This can affect which readings receive a scarce shared recording
  budget; documentation names that change. Historical evidence is untouched.
- State is subscription/response-local and may be deliberately shared across a
  provider's several named readings. Python named study readings require one
  unpadded sequence. Native library state is isolated per fixed batch row;
  changing row identities or dynamically packed/padded study measurements are
  not implemented. P5 must not silently expand those claims.
- Callbacks are trusted readers, not sandboxed code. They must not mutate inputs
  or consume generation RNG; Python rejects a non-None callback return. There
  are no new tensor clones or host scalar conversions in ordinary dispatch.
  P3 scoring keeps its existing CPU/detached implementation. P5 will introduce
  provider RNG streams, policy decisions, and the typed action-request API.
- New runtime orchestration is tested behavioral work. The audit pins legacy
  owners, native model bodies outside the exact adapter substitution, and P3
  score/evidence bodies. Arithmetic and alignment negative controls are present.
  No historical baseline or scientific gate was relaxed.

## Verification

The numerical fixtures independently compute addition and fractional ablation,
compare legacy dispatch in float32/float16/bfloat16, and check gradients through
the tensor path. Lifecycle coverage includes shared state across sites, isolation
across subscriptions/seats, nested sessions, provider failure, partially installed
hooks, and stale stream cleanup. Existing model-slot/container ownership still
serializes execution; the runtime does not add concurrent generation on one
model. P3's real Transformers sampled-generation tests, ordinary/panel study
journeys, and cross-client tests remain in the full suites.

Native tests run the real Qwen and Gemma model boundaries, preserve read-only
outputs, independently check legacy action arithmetic and gradients, and verify
per-row state and cleanup. Tests use tiny randomly initialized models; no model
was downloaded, and no research job was submitted.

Final verification:

- Full Python suite: **6,532 passed, 9 skipped, 8 warnings** (225.47 seconds).
- Full serial Xcode beta suite: **TEST SUCCEEDED**, with **295 SteeringKit +
  4,643 ExperimentKit tests** (4,938 total, including five skips). The five
  new native runtime tests ran, as did the real Python cross-client tests.
- Shared generators, built CLI reference, historical audits, the new P4
  preservation audit, bridge gates, public scan, and whitespace checks pass.
  The aggregate command was `scripts/ci/check-generated.py --audits --cli
  /private/tmp/interpbench-science-gpu-build/Build/Products/Debug/steerlab-cli`.

The first Python run was sandbox-limited: existing local socket fixtures could
not bind loopback ports. The complete rerun with local-socket access passed;
those failures did not require code changes. Suites run serially. Native builds
use Xcode beta, the installed Metal toolchain identifier, external derived data,
`CLANG_COVERAGE_MAPPING=NO`, and an explicit test Python path. No environment or
dependency was installed or changed.

## Integration and remaining work

The maintainer's reviewing/integration agents review the full diff through the
user, verify both serial suites and audits, and land. Main, the installed app,
the cluster engine, and research evidence were not changed. Rebuild the app and
bundled Python together when deployment is authorized. Live app testing is
intentionally deferred until the larger program is complete, as the user asked.

P5 adds pre-action policy decisions, threshold/adaptive policies, action timing
and conflicts, logit/token-selection actions, and agent identity/authoring. P6
completes decision evidence and remote journeys; P7 measures research-model
behavior and overhead. This branch does not claim those are implemented.
