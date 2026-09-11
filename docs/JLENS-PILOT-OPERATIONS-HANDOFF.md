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

Validation will include focused regressions, both full suites run serially with
Xcode beta and its Metal toolchain, generated-resource and AST gates, a diff
review, and the public scan. Auditors decide landing through the user.
