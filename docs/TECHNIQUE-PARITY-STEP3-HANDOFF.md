# Step 3: native scientific semantics and provenance

2026-09-07. Branch: `codex/technique-parity-foundation`, worktree
`/private/tmp/interpbench-workflow-handoff`. This extends the foundation at
`d18df26`, based on main `293888e`. The main checkout and installed app were not
modified or deployed. Review and integration remain with the researcher's
independent coding/audit agents, coordinated through the researcher.

## What changes for a researcher

- **Reader evidence is understandable and separated.** A researcher can reserve
  optional final-test rows in the app. The preview shows which rows train,
  select the sign/layer, and evaluate the fitted instrument. Local and server
  requests encode those same splits. Results show final-test accuracy separately,
  and a reader imported from Python keeps its evidence-role stamps.
- **A run says what it changed.** Ordinary and saved-agent measured runs write
  `intervention-scope.json`. It distinguishes last-position additive steering
  from every-position ablation and records actual layers, doses, ranks and
  declared centering. A baseline is explicitly an empty intervention chain.
- **Native extraction has a sensitivity diagnostic.** Agents can call
  `steerlab-cli experiment extract-stability <study> <concept> --json`, with
  `--resamples`, `--fraction`, `--seed` and `--order-shuffles`. It captures each
  input class once, then resamples those activations in memory. The report lives
  under `diagnostics/`, with the full draw values and exact integer seeds.
- **Adapter scale is explicit.** Newly completed native training identifies its
  scale as a direct multiplier, so it cannot be mistaken for PEFT's
  `lora_alpha / rank`. It takes the applied value from the training result,
  rather than rereading the panel's current controls.

These are guidance and provenance improvements. Final evaluation data is optional
for exploration; no backend-qualification allowlist was added. The diagnostic
explains that a stable contrast can still represent a confound, and that it
measures no behavioral outcome.

## Reviewable slices

- `9139e82`: reader final-test separation, role serialization and derived-vector
  propagation, with focused reader and real Python round-trip tests.
- `4cbf59a`: scope vocabulary/types/planner inventory, ordinary/variant run
  stamping, native stability math and cross-Python numerical/behavior fixtures.
- `bb480e2`: native CLI publication, the app's optional split authoring, Unicode
  overlap details, adapter provenance, docs and regression coverage.
- The final test-isolation follow-up gives the preflight fixture both workspace
  overrides, so its concept inputs stay in temporary scratch. Use the branch tip
  as the final review target.

No Python scientific implementation or AST baseline changed. The Python scope
constants are consumed by a new generator; numerical ports are checked with
behavioral and numerical fixtures, not described as AST-equivalent moves.

## Owners and contracts

### Reader

`RepEReader.swift` and `RepEReader+Evidence.swift` implement three case-insensitive
roles: train, held-out (all other spellings except `finalTest`), and final test.
Capture order is train → held-out → final test; only train and held-out reach the
existing fitting/sign/recommendation functions. Final-test rows only score the
fitted classifier. The classifier retains its train-fitted orientation and
threshold; its accuracy is distinct from the pair-agreement statistic used to
choose the reading direction's sign.

Six additive fields remain schema 2: `finalTestAccuracy`, `finalTestPairCount`,
`splitOverlap`, `evidenceRoles`, `evidenceRolesBasis`, `evidenceRoleNote`.
Absent historical fields stay absent on encode; `resolvedEvidenceRoles` reads
existing legacy stamps without writing a new measurement. Derived vectors carry
`readerEvidenceRoles`. No manifest or recipe-identity input field was added.

The exact overlap check mirrors the current Python contract: compare normalized
train stimulus identities against non-train identities. Whitespace/case folding
is checked, including Unicode cases; canonical Unicode composition is not an
extra equality rule. This is **not** a near-duplicate audit or a comparison of
held-out against final-test rows. Do not describe a zero exact-duplicate count as
proof of independent evaluation data.

`ConceptBuilder.readerSplitPreview` drives both row encoders and the app preview.
Final-test rows are reserved first, held-out rows use the remaining tail, and at
least two rows remain for train. Any reduction is displayed before fitting. The
default final-test count is zero, preserving existing request bytes. Both
content-pair and single-stimulus shapes are covered.

### Intervention scope

`InterventionScopeVocabulary.swift` is generated from Python's module-level
constants by `scripts/ci/check-intervention-scopes.py --write`. The regular
science-resource check now checks this vocabulary too. `VectorInjector.scope`
and `SubspaceAblator.scope` describe their own configuration;
`InterventionPlan.scopeInventory` describes the concrete chain in execution order.

`RunInterventionScope` receives the run's existing resolution functions. Centering
is propagated as metadata from each resolved variant injection; numerical
centering/projection remains at the existing owner. Prompt-token count is described
as supplied per item rather than stamped as the placeholder used to construct the
gated chain. Existing sidecars are never replaced. Resolver errors become named
`unresolved` rows; the execution loop still owns the execution error.

Coverage is the ordinary and saved-agent measured-run paths, matching the Python
run-sidecar owner. Playground and native multi-agent runs do not gain this file
here. Native trainable and SAE-latent intervention execution remains unavailable;
use the Python execution path for those mechanisms.

### Extraction stability

`DirectionStability.swift` ports the pure statistic and per-layer driver. The
supported recipes are mean difference, paired-difference PCA (`lat`), and
designated reference. It preserves SplitMix64 streams, partial Fisher–Yates
subsampling without replacement, separate full-order shuffles, half-up sample
size rounding, unpaired-class draws, degenerate-draw accounting, and float64
aggregation of float32 cosine values. Every layer receives the same draw seeds.

`ExtractStability` owns cheap preflight, one capture per class, numerical execution
and fresh report publication. It reads the manifest's reading/rendering and
records live versus pinned stimulus hashes. It does not pin a draft model revision,
rewrite the study, apply neutral-PC projection, or create a lifecycle run.
It uses the existing model-capability resolution before capture.

The report follows the Python shared/per-layer key partition and carries exact
UInt64 seeds, including values above 2^53. The Mac envelope supplies summary
scalars and the report path; full resampling settings/seeds remain in the report
rather than going through the envelope's floating-point JSON value type.
Unsupported methods and invalid draw settings have typed, actionable errors.

The app's existing managed science job sheet continues to run the Python
stability owner. This slice adds the promised native verb; it does not silently
reroute an existing Python job to MLX.

### Adapter provenance

`FineTuneTrainingResult.adapterScale` carries the value passed into the native
training configuration. At successful training completion,
`FineTuneArtifact.recordTrainingScale` records `adapterScaleConvention: direct`,
`effectiveAdapterScale`, `requestedAdapterScale`, and
`requestedAdapterScaleConvention: direct`. Historical/untrained registrations do
not acquire inferred values on decode. Objectives, schedules, data admission,
checkpoint behavior and weight formats remain the distinct implementations
recorded in `TRAINING-RECIPES.md`; this is not a trained-weight equivalence claim.

## Validation

Both full suites ran sequentially against the branch sources:

| Check | Result |
| --- | --- |
| Full Python suite | **6,175 passed, 9 skipped, 8 warnings** |
| Full Xcode beta suite | **286 SteeringKit + 4,609 ExperimentKit passed** (`TEST SUCCEEDED`) |
| Reader fixtures | Final rows cannot change fit/sign/recommendation; exact Unicode overlap; legacy absence; actual Python reader round trip; derived-vector roles |
| Scope fixtures | Python descriptor equality and actual mid-prefill/prompt-end/decode tensor edits; dependent basis rank; immutable run-sidecar publication; unresolved conditions |
| Stability fixtures | Three recipe comparisons at `2e-5` absolute cosine/statistic tolerance; exact seeds including UInt64.max; independent constant contrast; unpaired classes; degenerate draw accounting |
| Owner/surface fixtures | Exact report seeds and draw partition, distinct publication paths, unchanged study bytes, cheap preflight repairs, shared local/server final-test row requests and app previews |
| Managed scientific owner, stability-preflight, design lazy-import and task-parser AST audits | Pass with existing baselines and negative controls |
| Science resources, intervention vocabulary, substrate inventory, study interviews, workspace bootstrap, Python client identity, client assembly reference | Pass |
| Swift bridge gates, normal and release | Pass |
| Source-built CLI reference check | 17 generated regions match |
| Public scan and `git diff --check` | Clean |

The first full Swift run caught the expected verb-count census change (35 → 36);
it was corrected and the full suite rerun successfully. The final cleanliness
check also caught the preflight fixture using the checkout for its two synthetic
concept files; both workspace overrides are now scoped to the temporary root,
the verified fixture files were removed, and the full Swift suite was rerun.
No scientific test failure remains. The diff was read before the final commit. New numerical ports
are supported by the numerical fixtures above; no AST baseline was moved.

Reproduction (from the worktree, with an existing suitable Python environment):

```sh
PYTHONPATH=Server HF_HUB_OFFLINE=1 <client-python> -m pytest Server/tests -q

DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON=<client-python> \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-workflow-xcode \
  CLANG_COVERAGE_MAPPING=NO
```

Session logs: `/private/tmp/step3-python-full.log` and
`/private/tmp/step3-swift-full.log`. No live model training, cluster job, backend
qualification or installed-app deployment is implied by these tests.

## Remaining order

Step 3 completes the promised native semantics/provenance package, pending review.
There is still coding after this step:

1. **Step 4:** implement scoped seeded MLX sampling on every measured record/turn
   path, including resume/interleaving and continuation behavior.
2. **Step 5:** measure the declared MPS/CUDA/MLX configurations and fix observed
   defects. This requires real models/hardware and explicitly available compute.
3. **Step 6:** consolidate regeneration and then simplify technique registration
   and extension using the exercised guide. This is implementation work, not only
   live testing.

See `TECHNIQUE-PARITY-IMPLEMENTATION.md` for the acceptance criteria. No hardware
cell was marked qualified here. Do not remove existing unsupported-path repairs
or manufacture new qualification refusals from this inventory.
