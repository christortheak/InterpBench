# Probe artifacts and the phase-1 runtime contract

P0 decisions, 2026-09-12; implementation baseline `22f08a9`.
The additional J-lens/GPU discussion is resolved. Initial targets are binary
classification. Multiclass and regression are later extensions.

P3 study measurements now execute on Python; see [the measurement contract](PROBE-STUDY-MEASUREMENTS.md) for surfaces, semantics, and qualification limits.

P2 capture, fitting, and evaluation are implemented; see
[the training contract](PROBE-TRAINING-AND-EVALUATION.md). The first-slice account
below records P1; generation measurement and policies remain future work.

## First slice and surface scope

P1 introduces one Python artifact owner and one library owner, consumed through
the existing native Python-client adapter. The Probes section, both clients'
`science probe-list` and `science probe-inspect`, and both workbench HTTP
`POST /api/science/workspace/probe-list|probe-inspect` actions return the same
inspection contract. The root is the local authoring workspace. A runner does
not gain an authoring library through this route. Legacy training and Playground
reading continue using their existing owners and original bytes.

This slice implements library discovery/inspection and a reference scorer for
new artifacts. It does not advertise a new trainer, study attachment, generation
runtime, or conditional policy yet. Those follow P2–P6. No new artifact is offered
in the old Playground picker, whose observation and coordinate contract is weaker.

## Portable activation-probe v1

Files use `.probe.json` under `runs/<run>/`. The identity is SHA-256 of the exact
file bytes. Relabeling or re-encoding produces a new identity. Inspection never
rewrites artifacts. Evaluations are separate outputs pinned to these bytes.
JSON has unique keys and finite numbers. Integers outside the exact binary64
range must be strings (including large seeds in provenance). Executable pickles
and inline executable code are not accepted as this format.

Top-level fields are closed: `artifactType: activation-probe`, `schemaVersion: 1`,
`label`, `createdAt`, `method`, `input`, `preprocessing`, `layers`, `output`, and
`training`. See the shared cross-client fixture for a complete numerical example.

- `input`: `modelID`, nullable `revision`, `substrate` (`mlx` or `pytorch`),
  `coordinateConvention`, `precision`, `hiddenSize`, `site` (`kind` is
  `residualPre` or `residualPost`, and `layer` is zero-based), `reading`
  (`rendering`: `raw` or `chatTemplate`; `position`: `lastNonPadding` or
  `eachNonPadding`; `population`: `prompt`, `generatedPrefix`, or `completeText`),
  nullable `tokenizerSHA256`, and nullable `templateSHA256`. Unknown pins remain
  null and are explained as limitations. Matching names or dimensions does not
  establish model/backend compatibility. Reference scoring requires the caller
  to supply this full binding exactly; later model adapters must derive it from
  the actual computation, never copy it blindly from the artifact.
- `preprocessing`: explicit `center` and strictly positive `scale` vectors,
  fitted using fitting data alone. Compute `z = (x - center) / scale`.
- `layers`: explicit row-major `weights`, `bias`, and `activation` for each
  affine layer. `mean-difference-v1` and `linear-logit-v1` have one identity
  layer with one output. `mlp-relu-logit-v1` has a ReLU hidden layer followed by
  one identity output. Shape checks bind every dimension. The reference scorer
  uses binary64 arithmetic and the interpreter's built-in `sum` for affine
  reductions. Bitwise equality across interpreter versions or tensor backends
  is not implied; tensor-runtime parity is a later measured contract.
- `output`: distinct `negativeLabel` and `positiveLabel`, `scoreKind`
  (`signedMargin` for mean difference, `logit` for classifiers), and a finite
  `threshold`. Positive means score strictly greater than threshold; ties are
  negative. Scores are never labelled calibrated probabilities in this slice.
- `training`: `recipeID`, `data` references (`role`: `fit`, `selection`, or
  `calibration`; `sha256`), and JSON `settings`. At least one fitting reference
  is required. Final-test references cannot appear here. Fitting algorithms,
  optimization, group splits, layer/threshold selection, and calibration must be
  specified and tested in P2; this metadata is provenance, not proof of honest
  data separation. Later evaluation reports record final-test evidence separately.

No scientific compatibility is inferred from the reference scorer accepting a
vector. Generalization to generation prefixes and causal effects require their
own evidence. Unknown provenance is an advisory; malformed numeric parameters
or an incompatible input binding cannot be executed meaningfully.

## Legacy inventory

The Mac's `.probe.json` files contain `ReadingProbeArtifact`: recipe name,
optional hashes, createdAt, and a `ScalarProbe`. The recipe can use different
direction methods; do not label all native readers mean-difference merely from
the filename. Python's `-probe.json` uses `kind: readingProbe`, a mean-difference
reader, layer-selection accuracy, and class counts, without the same recipe/time
fields. The shared library recognizes each structure, preserves its raw bytes,
and reports missing site/rendering/coordinate metadata. Python's heldOutAccuracy
is selection accuracy, never final-test evidence. No legacy record is silently
converted into v1 or admitted to another backend's runtime. Invalid matching files
appear as inspection issues rather than silently disappearing from the library.

## Shared runtime decisions for P3–P5

The measurement configuration belongs to a study. A behavior-changing policy
belongs to an agent and pins its probe dependencies. Both reference the same
semantic sites and observation context; P3 must not build disposable hooks.

- Sites: `residualPre(layer)`, `residualPost(layer)`, and `logitsPreSelection`.
  Block input/output maps must be declared per backend/model. Existing hooks
  supply block outputs; block inputs and logits are not claimed implemented here.
- Context: run, agent instance/seat, sequence, example, stage (`prefill` or
  `decode`), absolute input-token position, mask, site, and provider-local state.
  An activation at token position t predicts t+1; no extra forward pass is added
  to obtain an otherwise unavailable final-token reading.
- Order: requested pre-action readings, all policy decisions from that snapshot,
  ordered action application, and requested post-action readings. Earlier
  interventions may already have affected a pre-action reading.
- State resets at each response/example and is isolated per sequence and seat.
  Providers receive separate RNG streams. Read-only observers cannot mutate the
  tensors or consume generation RNG. Scores are stored by default; retained
  activations are a separate, bounded opt-in.
- Actions: existing addition retains its units, masks, and ordering; ablation
  remains subspace removal with dimensionless lambda. Logit adjustments and
  token constraints run at a declared sampling boundary. No layer repetition,
  backtracking, or hidden extra forward pass. No future information can affect
  an earlier site in the same pass.
- Provider API: in-process tensor inputs, typed observations/actions, and local
  state. No JSON/scalar conversion in the mathematical path. Asset and provider
  hashes join the input closure. An expert extension does not imply an MLX
  implementation of arbitrary Python. Errors produce explicit missing readings
  or recorded fallback/stop outcomes, never synthetic zeros.

These decisions constrain future runtime code; the executable tensor ABI and
sampling adapter types land with their implementations and tests. No empty
callback framework or compatibility bridge is introduced just to reserve names.

## Next slices

P2 adds actual capture, mean-difference/regularized linear/MLP fitting, evaluation,
shared interviews, and the guided training flow. Move the existing training UI
into Probes through shared owners with a mechanical audit where bodies move.
P3 records study measurements using the runtime contract above. P4–P6 add legacy
intervention adapters, policies, decision evidence, and remote/multi-agent paths.
Every slice has concrete surface and numerical acceptance, not just verb counts.
