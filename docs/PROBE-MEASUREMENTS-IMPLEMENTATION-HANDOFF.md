# P3 probe measurements: reviewer handoff

Branch: `codex/probe-study-measurements`, started from main `b1190f7`.
Main advanced by one documentation-only commit (`4321fce`); it is incorporated
by merge `f28b0a1`, so current main remains an ancestor of this branch.
The maintainer's agents retain responsibility for review, landing, and deployment.
The installed app, main checkout, cluster engine, and running studies were not
modified. App interaction and research-model qualification are deliberately pending
until the larger probes/interventions program is complete.

## Commits and scope

- `f855904`: the requested P2 follow-ups. Capture reports explain whole-example
  labels on prefixes, preflight explains full-model memory costs, and the review
  accurately states that discovery and inspection share the run-path restriction.
- `6ffc267` adds P3's read-only study measurement
  journey, with shared authoring, execution, evidence, and Results presentation.

The full behavior and limitations are in [the P3 contract](PROBE-STUDY-MEASUREMENTS.md).
The [phase-1 plan](PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md) remains the program scope.

## What to review

1. `probe_measurements.py`: strict declaration validation, exact probe bytes,
   shared review/save, external manifest-byte preconditions, and draft-only edits.
   App, both CLIs, and workbench HTTP reach this owner. Removal is a reviewed save
   with no selections; absence leaves the historical optional field absent.
2. `probe_observation.py` and the generation adapter: response-scoped named
   block-input/output hooks; pre/post-action ordering around existing steering;
   independent offsets across prefill chunks and decode; CPU float64 scoring;
   scope filters; read/activation budgets; explicit missing and partial evidence.
   No additional forward pass or generation RNG use. Population transfer is
   guidance, while incompatible runtime bindings remain concrete execution errors.
3. `condition_execution.py`, `multi_agent.py`, and `panel_workflow.py`: settings
   reach ordinary conditions and selected seats; generated response/turn context
   accompanies readings; panel flattening preserves them; resume keeps completed
   response bytes. Failed generation evidence is separate from completion keys.
4. Python and Swift pin surfaces: verify, freeze relocation, snapshot, and bundle
   closure include probes. Source bytes are unchanged. Snapshot destinations use
   hashes so equal filenames cannot hide a second external probe.
5. `StudyMeasurementsView`, `ProbeMeasurements`, and Results: guided selection,
   explicit scope/schedule review, shared owner calls off-main, optional manifest
   round-trip, readable score/position/status display, and distinct sample IDs.
   Python records need not contain the native legacy marker-density metric to
   appear in Results.

## Surface and numerical boundaries

- Both workbench implementations use the existing
  `POST /api/science/workspace/{action}` route with `measurements-review` and
  `measurements-save`; runner authority is unchanged.
- Both CLIs use `science measurements-review|measurements-save <experiment>
  --settings <file>`, with the save's `--plan-sha256` and each client's ordinary
  workspace/JSON flags. The native parser/reference census is updated.
- Execution uses the **Python engine** on the Mac or a remote runner. Native MLX
  portable-probe execution is not implemented; native run names Python Compute as
  the repair. Do not claim cross-backend activation equivalence.
- Results keeps its existing bounded response preview. Full JSONL remains the
  authoritative record; this slice does not add a statistical analysis dashboard.
- New study settings change identity. Identity-derived seeds may consequently
  change between distinct frozen studies. Observer noninterference is tested with
  the same actual seed, not with different study identities.
- Live app use, research-size CUDA/MPS runs, and measured overhead remain pending.
  No new model was downloaded, and no cluster job was submitted for this work.

## Tests and audit changes

The new tests independently check pre/post block arithmetic, existing steering and
reader ordering, exact score agreement, chunked positions, per-seat scopes, row and
activation budgets, non-finite policies, teardown and partial evidence, and a tiny
real Transformers sampled generation with identical text, token IDs, and RNG state.

Journey tests exercise actual condition writing and resume, panel turn flattening,
freeze relocation, the real HTTP owner, and package/import into an isolated root.
Native tests invoke the real Python owner, round-trip measurements and historical
absence, display collected records, and preserve two external probes with identical
filenames. CLI smoke testing compares the native and portable review results exactly.

The lazy-import audit proves the original mechanical migration at the landed
`b1190f7`, then checks current bodies outside the deliberately extended
`run_scenario`. It still verifies the lazy wrapper and rejects its negative control.
P3's scenario changes are behavioral work covered by tests, not claimed mechanical.
The old J-lens hook-order source assertion was updated for the named-site dispatch;
real generation tests now also combine an ordinary reader, steering, and a portable
probe. No numerical J-lens owner or prior audit baseline changed.

Use `scripts/ci/check-generated.py --audits --cli <built-helper>`, the public scan,
and both serial suites. Use Xcode beta, the installed Metal toolchain identifier,
external `/private/tmp` build scratch, `CLANG_COVERAGE_MAPPING=NO`, and an explicit
`TEST_RUNNER_STEERLAB_TEST_PYTHON` as in the repository's build instructions.

## Verified checks

- Full Python suite: **6,521 passed, 9 skipped, 8 warnings**.
- Full Xcode beta suite: **TEST SUCCEEDED**, with **290 SteeringKit + 4,643
  ExperimentKit tests** (4,933 total, including five skips).
- Both CLI measurement-review commands returned the exact same owner result.
- Shared generators, both CLI reference regions, historical AST audits and
  negative controls, bridge checks (normal and release), whitespace, and the
  public-content scan pass.
- The new native round-trip, collected-results, and snapshot-collision tests all
  execute and pass; the round-trip invokes the real Python owner.

## Integration

Read the whole diff and obtain the maintainer's independent review through the
researcher before landing. The branch is intended for fast-forward integration
when main remains its ancestor; integrate new main fixes before final review if
it has advanced. Rebuild the app and bundled Python together at deployment.
Do not overwrite a running cluster deployment or install the app as part of review.

Next is P4: the intervention-runtime adapters and sequencing. P5 adds conditional
policies; the remaining phase-1 plan covers decision evidence and qualification.
