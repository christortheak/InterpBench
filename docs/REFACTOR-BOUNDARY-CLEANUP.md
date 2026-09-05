# Public Python boundaries and pre-1.0 cleanup

Branch: `codex/maintainability-refactor`
Mechanical baseline: `f6f746f`
Mechanical cleanup commit: `a7536f9`
Sweep-judgment fix commit: `b2d0fd9`
Main baseline: `b6949ff` (0.9.5)

This follow-up removes transitional Python import conventions before landing
and makes the remaining Swift bridge migration an explicit release gate. It
follows [the consolidated refactor overview](MAINTAINABILITY-REFACTOR.md).
The mechanical cleanup and sweep-judgment behavior fix are separate commits.

## Python interfaces

The baseline had 114 `_dep_` module aliases in 20 files, 318 attribute references
through them, and 91 cross-module private targets. The audit also found direct
private imports from the earlier analysis/policy extractions. Including those,
**114 shared symbols across 27 owners** now have public names. Implementation-only
helpers stay private. Public here means the package's intentional module
interface; it does not promise every helper as an external supported API.

Call sites use recognizable owner modules. A normal alias such as
`manifest_module` is retained where an unaliased import would collide with a
local variable. These aliases describe the dependency rather than marking every
moved dependency with a generated `_dep_` prefix. The exact symbol map is
[scripts/ci/python-boundary-renames.json](../scripts/ci/python-boundary-renames.json).

The `tasks.py` facade is **47 lines**, with an explicit 15-operation `__all__`:

- `extract`, `validate`, `sweep`, `run`, `evaluate`, `analyze`, `rescore_style`,
  `pipeline`.
- `complete_sweep_judgment`, `complete_evaluate_judgment`.
- `judge_worker`, `list_awaiting_judgment`,
  `list_awaiting_evaluate_judgment`, `read_judge_fanout_request`,
  `read_judge_worker_artifact`.

Production callers need those worker/query operations as well as lifecycle
stages. Promotion retains its existing `promote` module; pipeline composition
binds that operation alongside the facade's stages. Private helpers and other
incidental imports are no longer re-exported by `tasks.py`. Tests calling
implementation-only functions import their actual owner directly. Tests
replacing shared helpers patch their public spelling at the lookup site;
integration tests still replace public facade stages where appropriate.

Indirect policy references through `experiment_store` have been redirected to
policy owners too. The first full suite exposed these transitive references;
the audit now explicitly resolves them rather than treating direct imports as
the entire dependency graph.

## Evidence of preservation

Run with Python 3.12 from the repository root:

```sh
python scripts/ci/audit-python-boundaries.py --candidate a7536f9
```

The audit reads `f6f746f` from Git and compares executable ASTs against the
candidate. It normalizes only the recorded symbol identities, module imports,
facade/policy re-exports, and a named J-lens source-inspection assertion. It
ignores documentation and import statements while comparing their resolved
uses. It checks module executable statements, function bodies and test
assertions; it does not claim import-time side effects are proven by ASTs.
The import-direction tests and full suite cover that additional concern.

At the mechanical boundary, **3,436 function bodies across 129 changed Python
paths** passed this comparison. A self-test verifies that argument swaps and a
call redirected to a different owner fail equivalence. The bug-fix commit must
NOT pass equivalence against the old buggy body; audit the preceding mechanical
commit instead.

`Server/tests/test_public_experiment_boundaries.py` prevents new `_dep_`
aliases, private imports/calls into the extracted owners, unintended facade
operations, and stale fully qualified private mock paths. The existing offline
import tests still prevent reverse dependencies on the task facade.

The documentation now points to owner symbols rather than obsolete task line
numbers in `EXTRACTION-RECIPES.md`, `INTERVENTION-SCOPE.md`, and
`PORTABILITY-CONTRACTS.md`. The prompt-loader reference also reflects its
existing torch-free extraction.

## Swift retirement gate

See [BRIDGE-RETIREMENT.md](BRIDGE-RETIREMENT.md) for the complete bridge map,
caller inventory, migration sequence and exit criteria. The existing bridge
files remain unchanged in this cleanup; their migration is the next substantive
Swift slice, not part of the Python rename.

The CI command rejects new bridge members/caller occurrences and blocks 1.0
while any of the four files remain. The inventory is a conservative syntactic
scan, not compiler-resolved references; the documented manual review remains
part of retirement. A 1.0 version or release tag enables the strict gate
automatically. Run `--release` explicitly before publishing the codebase linked
from the launch blog, even if the version still says 0.x.

## Validation and landing

The first full Python run identified 19 failures: indirect policy references
and one source-inspection spelling, not changes to scientific calculations.
Those were corrected with the original assertions retained (the source
inspection now names the renamed function). The final Python suite passed:
**5,863 passed, 9 skipped, 8 warnings** in 148.17 seconds, including all four new
boundary/audit tests and the new sweep regression. Log:
`/private/tmp/python-boundary-final-full.log`.

The full serial Swift suite and app build passed with Xcode beta and the Metal
toolchain: **277 SteeringKit + 4,400 ExperimentKit = 4,677 tests**.
Log: `/private/tmp/interpbench-boundary-cleanup-swift.log`.
The normal bridge budget check passed; explicit and automatic 1.0 checks were
verified to reject a replacement bridge in an isolated fixture.

Use the existing Python 3.12 environment from this checkout's `Server/`
directory; `HF_HUB_OFFLINE=1` matches CI and avoids model downloads. Loopback
server tests require the usual local execution permission. Swift validation
uses Xcode beta, the installed Metal toolchain, serial execution and external
scratch as documented in the consolidated overview.

Before landing, require clean worktrees and current main to be an ancestor of
the refactor branch. If it advanced, integrate its fixes and rerun affected
checks. Use `git merge --ff-only codex/maintainability-refactor` from main; never
reset main to force the result. Landing changes source, not installed apps,
running scientific artifacts, or server deployments.

## Sweep-judgment regression

`b2d0fd9` fixes the deferred-completion branch where no cell beats the baseline.
It formerly called `_append_progress`, a closure belonging to the inline sweep
workflow. Completion now logs the selection refusal; its existing phase B writes
the reason into `recommendations.json` in the new judgment run. It does not
append progress to the original immutable sweep.

The new end-to-end regression first failed with the precise NameError, then
passed after the fix. It verifies the persisted negative result, completed
judgment artifacts, cleared awaiting list, byte-identical source sweep,
unchanged study manifest, and idempotent retry without a second run. All 56
sweep-objective tests passed after the fix. No gate threshold or selection rule
was relaxed.
