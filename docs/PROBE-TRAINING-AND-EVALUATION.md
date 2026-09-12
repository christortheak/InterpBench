# Probe training and evaluation

P2 extends the shared library with `probe-capture`, `probe-train`, and
`probe-evaluate`, exposed through the existing managed scientific workflow.
The app's Probes section opens the same maintained interviews as both clients.
No training occurs when a form is opened. Existing Playground readers keep their
original owner and format, under a labelled legacy section in Probes; Data keeps
browsing the editable datasets. No historical reader score or saved byte changes.

## Researcher journey

1. Define a binary label and group related examples (same document, paraphrases,
   or repeated measurements). Use an existing file or copy the author/reviewer
   instructions; no agent or dataset generation starts automatically.
2. Capture one declared residual site from a prepared model. Save activations
   once and reuse them. Capture reads text without generating a response or
   applying steering. Python CUDA, MPS, or CPU execute; native MLX capture is not
   claimed. Unsupported decoder paths produce a concrete mapping repair.
3. Fit a mean-difference reader, a regularized linear classifier, or a small
   nonlinear classifier. Select the fitting activations; optional selection data
   produces a separate comparison. Preprocessing never sees selection data.
4. Evaluate the saved probe on separately chosen matching activations. The new
   report pins the exact instrument and dataset bytes. Known overlap remains
   usable for exploration and is prominently distinguished from independent
   final-test evidence. A lack of detected overlap is not proof of independence.
5. After execution, collect evidence through the existing managed round trip.
   Refresh Probes to inspect the trained instrument. Read `training-report.json`,
   `capture-report.json`, and `evaluation-report.json` in the Results/file viewer.

## Numerical recipes

All three are new `probe-fit-v1/<method>` recipes; they do not reinterpret old
native or Python readers. The binary threshold is fixed at zero before fitting.
Positive means strictly above zero. Scores are signed margins or logits, never
presented as calibrated probabilities.

- Standardization uses the fitting population's per-feature mean and population
  standard deviation (`ddof=0`); constant features use scale one. `none` uses
  zero center and unit scale. Parameters are saved in the probe.
- Mean difference computes the positive minus negative mean in those coordinates,
  normalizes it to unit length, and centers its projection at the midpoint of the
  two means. Coincident means cannot define this direction and receive a repair.
- Linear training initializes weights and bias to zero. It minimizes mean binary
  cross entropy plus `l2/2 * sum(weights²)`; biases are unpenalized.
- MLP training uses one ReLU hidden layer, then one affine scalar output, with
  the same objective. Hidden weights are seeded normal with variance `2/inputWidth`;
  output weights use variance `1/hiddenWidth`; biases start at zero. NumPy PCG64
  is a local RNG; no model RNG is consumed.
- Both classifiers use fixed-step full-batch gradient descent. Learning rate,
  regularization, steps, width, normalization, seed, and NumPy version are recorded.
  No early stopping or automated layer/threshold search consults evaluation data.
- The optional shuffled-label control permutes fitting labels with the same local
  RNG before initialization. Reported metrics retain the original labels. It is
  a separately saved control instrument, not a substitute for the ordinary fit.

Fitting needs at least two rows per class for arithmetic. This is not a claim
of statistical adequacy. Related positions remain grouped; metrics weight rows
and do not treat correlated rows as independent evidence. Reports include class
and group counts, confusion counts, accuracy, balanced accuracy, precision,
recall, specificity, F1, tie-aware ROC AUC, and both constant-class baselines.
Undefined denominators produce nulls. There are no inferential confidence claims.

## Capture and storage contract

Text input is strict UTF-8 JSONL with unique `id`, `group`, nonempty `text`,
Boolean `label`, and `split` in `fit`, `selection`, or `finalTest`. A group or
identical text cannot cross roles. Optional `groupHash` instead omits `split`:
SHA-256 of canonical JSON `[seed, group]`, first 16 hex digits as an integer mod
100, assigns <60 to fitting, <80 to selection, and the rest to final testing.
It never silently rebalances or breaks groups to populate a small split.

Capture takes the first `maxExamples` rows and caps tokenization at `maxSeqLen`.
Raw rendering uses tokenizer-native special tokens. Chat rendering is exactly
one user message plus a generation cue, with no duplicate special tokens.
`prompt` and raw `completeText` are replay populations; live generated-prefix
measurement remains P3. `eachNonPadding` applies the example label to every
observed prefix: the researcher must decide whether that labeling is meaningful.

Hooks read actual decoder block inputs or outputs, with one sequence per forward
pass, evaluation mode, no cache, and no returned tensor replacement. Shape and
execution-count checks prevent silent misalignment. The report records actual
input token IDs and positions. The dataset binding includes actual activation
precision, model revision, explicit decoder path, tokenizer and template hashes,
rendering, and position population. It must match exactly at fitting/evaluation.
Fast tokenizer identity includes the serialized backend and special-token behavior.
A slow tokenizer without that serialization records an unknown tokenizer hash;
a vocabulary alone does not establish its normalization or segmentation identity.

Each nonempty role produces an `activation-dataset` v1 JSON document containing
`artifactType`, `schemaVersion`, `input` (the portable probe input binding),
`rows`, and `provenance`. Rows have `id`, `group`, `sourceSHA256`, Boolean `label`,
and `activation`. Captured row IDs hash `[source id, token position]`; the source
hash identifies the original UTF-8 text. External datasets can use this same
format; unknown provenance stays a limitation, not invented capture history.

Inputs and outputs are bounded to 64 MiB per JSON; capture also bounds total
activation row JSON and `maxRecords`. These are pilot limits, not a streaming
activation warehouse. Model memory is separate and may dominate. Fitting's main
matrix uses approximately `8 * rows * hiddenSize` bytes, with additional working
arrays and classifier parameters. Start small, then choose an adequate budget.
Fresh UUID run directories save capture/fitting requests before computation and
use COMPLETED only after successful publication. In-memory activations or optimizer
state are not checkpointed; cancellation requires a new run. Durable jobs retain
progress logs and the partial run location. Existing job cancellation and export/import
owners handle the lifecycle; no special queue or cleanup owner is added.

## Surface paths and qualification

Both clients use `science interview`, `science draft`, and `science publish`
with these three operation IDs. App forms use those same owners. The existing
HTTP `/api/science/plan` and `/api/science/submit` paths execute reviewed inputs;
input-plan/package, stage, plan, submit, jobs, export/fetch/import, and local
custody work as for other managed operations. Library discovery remains shared.

Numerical tests, toy decoder capture, and managed transport tests establish
implementation behavior. Real-checkpoint CUDA/MPS capture and interactive app
acceptance are qualification work for the running/reviewing agents. A successful
fit does not establish causal use of a feature, live measurement parity, or a
behavior-changing intervention. P3 and later phases remain separate work.
