# J-lens pilot operations follow-up

Branch: `codex/jlens-pilot-operations`, starting from main `f0449d3`.
Scope: the first implementation slice in §8 of
[the pilot handoff](JLENS-FITTING-PILOT-AND-SCALING-HANDOFF-2026-09-11.md):
R7 telemetry, R5 large exports, R6 staging guidance, publication interruption
coverage and recovery (§8.6), and running-versus-deployed controller identity
and direct-transfer guidance (§8.9).

No model fitting, package installation, cluster deployment, controller restart,
or changes to site configuration are authorized by this implementation. The
running agents own the live continuation test and environment repairs. Batching,
optional kernels, sharding, merge, and stopping remain later slices. A zero-norm
running mean makes a relative-change statistic unavailable; it does not justify
aborting a fixed-budget fit.

Preserve the pinned reference kernel and historical accumulation-loop audit.
Telemetry wraps the reference call. Export retains the complete evidence archive
and checkpoint; selective exports require a separate custody/cleanup design.
Use a documented longer idle timeout for preparation of large exports, retaining
streamed file transfer and digest verification. Stage results supply an explicit
runner request, with direct client support for saving that request locally.

Validation includes focused regressions, both full suites run serially with
Xcode beta and its Metal toolchain, generated-resource and AST gates, a diff
review, and the public scan. Auditors decide landing through the user.

## Implemented behavior

- `jlens_fit_telemetry.py` wraps the pinned per-prompt call, retaining the
  original accumulation loop unchanged at AST level. Row timing, batch,
  tokens, CUDA allocated/reserved/free/capacity measurements, and process peak
  RSS are recorded. Reports include output finiteness and compilation status.
  A failure at loading or backward retains the original exception and gains
  measurement context and a repair. Telemetry is part of driver provenance;
  checkpoint numerical compatibility remains unchanged.
- Optional kernel package versions and executed attention-module bindings are
  observations, not package-installation-based claims of acceleration. An
  explicit disabled fast path establishes fallback; otherwise dispatch is
  honestly unverified. Precise fused-kernel qualification belongs to the later
  R4 benchmark slice. No kernels are installed or selected here.
- Staging and export use an hour-long default response-read budget. Ordinary
  requests retain their short budget, and downloads still stream and verify
  the digest before import. Export remains synchronous and idempotent; a
  timeout repair asks for the same export again, never another fitting job.
  The full archive, including its checkpoint, is retained. The existing archive
  size and custody bounds remain in force; this is not a selective-export
  migration or an asynchronous export service.
- Both CLIs and the app save the staged digest request under the local
  workspace's `.steerlab/diagnostic-requests/` and return `localRequestPath`.
  Publication is create-only and repeatable for identical bytes. HTTP callers
  use staging's returned `request` object. Path-specific planning failures
  explain the staged form while ordinary validation errors retain their repair.
- Empty abandoned publication claims receive a recovery message, including at
  repeated HTTP staging. Neither a filename nor emptiness authorizes deletion:
  the researcher must establish publisher exit first. Native publication remains
  the first choice; the filesystem fallback still exposes an incomplete empty
  claim briefly. Tests interleave a reader and simulate interruption in that
  interval without mistaking it for completed output.
- Capabilities expose the controller's startup `runningEngineVersion` separately
  from the current on-disk `deployedBuildCommit`. Cluster status and Cluster
  health display both and give an advisory on disagreement. Push explicitly
  says it does not restart a controller. Older servers retain unknown status;
  no restart, resubmission, or environment mutation is automatic.
- The shipped J-lens guide documents request handoff, telemetry, longer waits,
  direct transfer of a valid evidence archive, local `science import` and custody
  verification, controller restart awareness, and abandoned-claim recovery.

## Validation and hand-back

Focused regressions pass, including local HTTP staging/export/import and the
unchanged reference-estimator/continuation fixtures. The final Python run passed
6,387 tests, with 9 skipped and 8 warnings. Its log is
`/private/tmp/pilot-ops-python-full.log`. Xcode beta reported `TEST SUCCEEDED`:
290 SteeringKit tests and 4,617 ExperimentKit tests. Its log is
`/private/tmp/pilot-ops-xcode-full.log`. Both suites ran serially on the final
implementation at `48c5637`; the final handoff commit only changes this document.

The generated-resource checks, built CLI reference check, historical AST audits
with negative controls, normal and release bridge gates, public scan, and
`git diff --check` pass. The fitting-loop audit remains pinned to `878bca2`.
Final gate output is `/private/tmp/pilot-ops-final-gates.log`. Every source diff
was read; generated copies were checked against their maintained sources.

Reproduce the suites from the branch:

```sh
# Server/; use an existing full test environment.
PYTHONPATH=. <test-python> -m pytest -q

# Repository root; use scratch outside a file-provider-managed directory.
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON=<absolute-test-python> \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath <external-scratch> CLANG_COVERAGE_MAPPING=NO

<test-python> scripts/ci/check-generated.py --audits \
  --cli <external-scratch>/Build/Products/Debug/steerlab-cli
<test-python> scripts/ci/public_scan.py
git diff --check
```

The full Python run caught an overly broad staging repair that also intercepted
scientific config errors. Path failures now have a distinct subclass; ordinary
validation retains its original message and repair. Explicit Python timeouts,
including an explicitly supplied 60 seconds, override the diagnostic default.
Both cases have regression coverage.

Rebuild the app and its bundled Python payload together after review. The running
agents still own live continuation, large-output collection, and deployed-build
drift acceptance on the cluster. No live CUDA fitting or transfer-performance
claim is made by these local tests. Review and landing remain with the maintainer's
auditors through the user; this branch does not deploy or restart anything.
