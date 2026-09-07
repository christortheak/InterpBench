# Steps 4–6: implementation and qualification handoff

2026-09-07. Branch `codex/seeded-sampling-qualification`, based on main
`d84968c`. Implementation checkpoint: `ccf5045`. The following evidence/test
commit completes this handoff. Main and the installed app have not been changed.
The researcher coordinates independent review and integration.

## What changes for the researcher

Native measured studies can now use nonzero temperature with a seed scoped to
each record or multi-agent turn. Selecting local execution no longer produces
the obsolete greedy-only refusal or silently routes a warm study to the server.
The app and Swift CLI share these owners. New records identify their actual
seed policy, pairing index and prompt token count; sampling provenance records
the model snapshot, dtype, quantization, dependencies and hardware. Historical
records and frozen documents are untouched.

Python sampled execution now restores MPS random state as well as CPU/CUDA state.
Overlapping scoped records serialize around torch's global generators. This
prevents those records from advancing each other's streams; it can reduce
same-process sampled throughput. Unscoped foreign RNG users are not protected.

Adding a managed technique now starts with one small operation specification
under `docs/techniques/operations/`. Catalog text, interview fields, input
reference roles and dispatch bindings derive from those specifications. The
scientific code remains in its existing owner. One command regenerates the shared
client resources and identity; a separate option discovers the read-only audits.
The exercised guide and disposable worked example describe this route.

These changes improve local research usability, reproducibility and extension
maintainability. Supported but unqualified techniques remain available with honest
scope statements; qualification is not a new permission gate.

## Review map

| Area | Principal owners and review focus |
| --- | --- |
| Sampling arithmetic | `SeededLogitSampler.swift`: pinned library filter order, explicit per-record MLX key, no global reseed or dependency bump |
| Record identity | `StudySampling.swift`, `ExperimentTasks.swift`, `MultiAgentRunner.swift`: literal versus derived seeds, per-prompt sample indices, empty condition in multi-agent turn derivation |
| Reasoning | `SeededGeneration.swift`: one iterator/sampler across reasoning and answer; saved-agent thinking remains independent of baseline effort |
| Provenance | `StudyRecordContracts.swift`, `RunSamplingProvenance.swift`: optional historical fields, new sidecars only, multi-agent model-specific provenance |
| Routing | `SubstrateRouting.swift`, `LifecycleGate.swift`: retirement of the obsolete sampling refusal and its vocabulary entry |
| Python RNG | `experiment/sampling.py`: save/restore MPS in `finally`, process RLock, unchanged greedy path |
| Extension | `check-operation-specs.py`, `operation_bindings.py`, `managed_inputs.py`: 24 catalog operations, 13 managed declarations, nine ordinary bindings and four explicit special dispatches |
| Regeneration | `check-generated.py`: ordered resources then Python identity, optional source-built CLI reference, read-only audits with existing baselines |
| Qualification | `scripts/qualification/`: offline matched-model probes, production OptVec owner journey, checked tensor/protocol identities and mismatch tests |

The registration AST audit proves preserved existing bindings, reference roles,
catalog/interview rows and managed dispatch/config/validation bodies against
`d84968c`, including a body mutation control. It does **not** claim the sampling
changes are mechanical. Existing scientific-owner audit baselines remain intact.
The input-role registry describes dependency packaging, not train/selection/test
scientific evidence roles. No developer scaffold was necessary for this slice.

## Qualification evidence and remaining scope

[The qualification record](TECHNIQUE-PARITY-QUALIFICATION.md) contains the
prospective matrix, exact local observations, retained machine-readable reports,
limitations and CUDA commands. CPU/MPS residuals, extraction directions,
interventions, two-step OptVec optimization and separate evaluation passed the
declared diagnostics. MLX matched 10/10 ordinary and 10/10 reasoning-budget
replays on the named cached model. These are small probes, not backend-wide
certification, model efficacy or MLX/PyTorch token equivalence.

**Steps 4 and 6 are implemented. Step 5 is partially measured.** The researcher
has delegated CUDA execution to the cluster coding agents. Full battery and
stability studies, J-lens, J-space, SAE execution, fine-tuning objectives, full
saved-agent/multi-agent transcripts and production long-context chunking remain
live qualification tasks. Independent fresh-agent guide and researcher-journey
trials also remain. No capability profile is upgraded to globally qualified.

The MLX opt-in run logged a beta Metal assertion after its tests passed. The
record retains it as an unresolved teardown observation. Do not hide it behind
Xcode's success banner. Native checkpoint/resume is not introduced: the fixtures
prove a record's stream can be recreated from its address, not a new resume verb.

## Validation and integration

| Final check | Result |
| --- | --- |
| Full Python suite | 6,180 passed, 9 skipped, 8 warnings |
| Full Xcode beta suite | 288 SteeringKit + 4,608 ExperimentKit tests passed (4,896 total) |
| Opt-in MLX measurement and record tests | Four passed; 10/10 ordinary and 10/10 budgeted replays matched; post-test Metal diagnostic noted above |
| Unified generated-resource, CLI-reference, AST and bridge checks | Passed; both normal and release bridge gates |
| Disposable technique example | 25 tests passed |
| Public release scan and diff whitespace check | Clean |

The normal full Swift run leaves the expensive opt-in model probe disabled;
its live result is reported separately. Local logs are under `/private/tmp/`
as `parity-python-final.log`, `parity-swift-final.log`, `parity-mlx-final.log`,
`parity-audits-final.log` and `parity-example-final.log`.
Use Xcode beta, the installed Metal toolchain identifier,
external temporary DerivedData and the main test Python passed through
`TEST_RUNNER_STEERLAB_TEST_PYTHON`. Run Python and Swift suites sequentially.

```sh
python scripts/ci/check-generated.py --audits --cli /path/to/source-built/steerlab-cli
python scripts/ci/qualify-technique-example.py
python scripts/ci/public_scan.py
git diff --check
```

Before landing, the independent reviewer should read the complete diff, rerun
both suites and applicable gates, and review the explicit numerical changes.
Integrate through the researcher under the established process; if main has
advanced, incorporate it and revalidate. Rebuild the app from the integrated
source before using its Python-backed operations with the updated checkout.
Do not alter audit baselines, frozen runs or the installed app as a shortcut.
