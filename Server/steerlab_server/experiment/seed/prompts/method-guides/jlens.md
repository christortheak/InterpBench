# Acquire and qualify a J-lens

Inspect a lens under its model, layer and token constraints.

Ask which model/revision, fitted lens, source/target layers and token-level question the researcher wants to inspect. Distinguish acquisition, arithmetic qualification, predictive readout and behavioral evidence.

Use the J-lens pane or the existing jlens family and /api/jlens routes for acquisition/import, token options, direction derivation, qualification, G0, probing, report and support. Read each operation's public help or request schema. Qualification fixtures belong to the instrument; do not replace them with study-specific examples to obtain a pass.

Preserve lens identity, model revision, tokenizer conventions and supported fitted layers. Imported or acquired files are not qualified solely because they load. Unknown applicability requires qualification or an explicit unsupported result.

Outputs include lens metadata, qualification records and readout reports. A token ranking is a model-dependent readout; it does not by itself show that a concept caused a behavioral change. Use a separately designed steering intervention and held-out outcome measurement for that claim.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.

## Import a custom fitted lens

Use an artifact you already have. Place this description beside its tensor file;
replace the miniature example geometry with the actual fitted model and tensors.
A coding agent should read the supplied source documentation and ask about missing
metadata, never infer a model revision or layer mapping from a filename.

```json
{
  "schemaVersion": 1,
  "kind": "jlens",
  "modelID": "example/model",
  "modelRevision": null,
  "hiddenSize": 2,
  "layerCount": 4,
  "tensorFile": "lens.safetensors",
  "lens": {
    "tier": "testing",
    "targetLayer": 3,
    "layers": {"0": "layer_0", "2": "layer_2"},
    "promptsFitted": 100,
    "corpus": "Description of the actual fitting corpus",
    "maxSeqLen": 128
  }
}
```

This example declares two fitted 2×2 matrices; it does not invent the missing
layer 1. It supports readouts at those fitted layers; the current steering-token
vector builder needs every source layer before the final block. Transport is
`J_l @ h` to the final block before normalization. Optional
`configFile` retains original source configuration bytes. Optional `source` can
record a repository, its exact revision, and/or a URL; unknown values are omitted.
Known fit-time `modelRevision` is a 40-character commit; unknown remains null.
Safetensors and numeric NPZ are portable; tensor-only PyTorch checkpoints and
BF16 need the optional PyTorch reader. A saved JacobianLens `J` mapping is read as
`layer_<number>` keys, with checkpoint geometry checked against the description.

Both clients offer `science artifact-plan <description.json> --json`, then
`science artifact-import <description.json> --plan-sha256 <reviewed-hash> --json`.
Select the destination workspace with the client's usual root/workspace option.
The app's J-lens dialog offers **Import my own lens files** and explicitly stages
files to the connected Python workbench before review. Import publishes a fresh
lens ID, retains source copies, and does not qualify it or train anything. Use
that explicit lens ID when several fits exist for the same model.

Choose intended use with the researcher: optional `lens.tier` is `testing`
(the default, for rehearsal) or `evidence` (intended study use). This declaration
applies to every custom lens, including models in the published catalog. It is
not a claim of validity or a passing qualification. Freezing a study with this
readout still needs qualification bound to the exact lens bytes, runtime, and
layers, plus the study's normal pins. To change intended use, update the source
description, review a fresh plan, and import a new lens; keep existing artifacts
unchanged. Published-source imports retain their existing tier policy.

## Fit a new lens from your own text

The managed operation is `jlens-fit`. In the app's J-lens dialog choose
**Fit a new lens…**, or open its interview under Research methods. Both clients
expose `science interview jlens-fit --json`, then the usual
`science draft`/`publish` and execution/collection workflow. The engine uses
the same owner through `POST /api/science/plan` and `/api/science/submit`.
Use an installed engine with the pinned optional `jlens` extra. Authoring is
available without GPU packages; fitting loads only an already-prepared model.

Supply a UTF-8 JSONL corpus with exactly `id` and `text` per row:

```json
{"id":"passage-1","text":"Replace this example with a sufficiently long passage from your documented text source."}
```

Use text representative of the population where you will use the lens. This is
not paired positive/negative concept data. Preserve source and licensing details
in the authoring notes. Keep separate text for assessing readouts; fitting
convergence is not held-out validity. The default considers the first four rows
as a timing pilot. Increase the limit after inspecting cost and coverage.
There is no hidden shuffle or chat template. Tokenization uses the checkpoint's
native special-token settings; the reference position mask omits the first
16 positions and the final token. Short rows are explicitly recorded as skipped.

Configuration fields: `modelID`, immutable 40-hex `revision`, `corpus`
(`{path,sha256}`), optional `sourceLayers` (blank/null means every source
layer), `maxPrompts` (4), `maxSeqLen` (128), `skipFirst` (16),
`dimBatch` (1), `checkpointEvery` (4), `dtype` (bfloat16),
`device` (cuda), `tier` (testing), and optional `checkpoint`
(`{path,sha256}` naming state.json beside sums.safetensors).

The estimator sums output cotangents over valid target positions, averages
gradients over valid source positions, and averages the resulting matrices over
usable prompts. The final target is the last residual block before normalization.
Source layers are explicit; a partial lens can support readouts, while the current
token-vector builder needs every source layer. Weights never change. Accumulation
and final lens storage use float32 even when model execution uses bfloat16.

Cost includes one replicated forward and roughly hidden-size/dimBatch backward
passes per usable passage. More dimensions per pass need more activation memory.
A successful inference run does not prove that a hybrid attention implementation
supports this backward calculation. Measure a pilot on the actual checkpoint,
engine, and hardware before scheduling a large fit. Runtime or numerical errors
are reported; they are not silently counted as short corpus rows.

For a worked example (not a specification for your selected model), width 5,376
and 61 source layers require 5,376 backward passes per usable row at dimension
batch 1. Each float32 matrix set occupies 6.57 GiB; sums and the current row's
matrices alone need 13.14 GiB of CPU memory. Each checkpoint writes about
6.57 GiB, and model weights, activations, temporary matrices, and serialization
need additional space. Batch 8 processes eight dimensions together in 672
passes, using more activation memory; it does not promise eightfold acceleration.
Even four rows can be an expensive pilot. Measure before increasing the limit.
The shared request review estimates these costs from the exact model's cached
configuration when available, without loading or downloading weights. Otherwise
it reports unknown geometry and asks for review on the prepared engine. These
are arithmetic estimates, not measured peak memory or elapsed time.

Completed fits return an immutable run with fit-report.json,
jacobians.safetensors, artifact-description.json, captured input text, and a
checkpoint. Use normal science export/fetch/import and verify custody to bring
the run home. Then review artifact-description.json with artifact-plan and use
artifact-import to register this particular fit in the lens library. In the app,
use **Import my own lens files…** on that returned description. Fitting precision,
model revision, corpus hash, and original report survive registration. Qualify
that registered lens for its intended runtime separately; fitting is not
qualification and supplies no claim about causal behavioral meaning.

Periodic checkpoint snapshots live in the run's named
`.steerlab/jlens-fitting-state/` scratch directory. Each snapshot is published
coherently before this invocation prunes its prior scratch snapshot. Completed
runs retain an immutable final checkpoint and the source checkpoint metadata/hashes;
private working copies are then removed. Cancellation retains the last completed
scratch snapshot. Wait for the original job to stop, use the site's permitted
transfer tools to recover state.json and sums.safetensors together if needed,
and select state.json in a new request. The corpus, model, numerical runtime
settings, dependency versions, and estimator must match under the versioned
fitting compatibility contract. Source hashes remain recorded as provenance, so
a documentation-only change does not prevent continuation. A narrowly identified
checkpoint from the original fitting release is recognized under the same
contract; unknown older formats require review. Numerical changes require a new
contract or an explicitly assessed migration, not silently ignoring differences.
The row limit may increase; progress is never appended to the old run. Review
and execution independently read and verify checkpoint tensors, with a status
message before hashing. Multi-gigabyte files can take several minutes on slow
storage; do not interpret that alone as a hang.
The managed scheduler still reports automatic resume as unavailable: checkpoint
continuation is a new, explicit reviewed job. Failed jobs are not evidence exports.
No general cleanup of fitting runs or recovered checkpoints is authorized.
Failed runs accumulate intentionally. Ordinary errors after run creation,
including model loading and checkpoint restoration, leave a failure record with
the stage, reason, and recovery references when storage permits. An abrupt kill
or full disk can prevent that record; the run has no completion marker, and its
last completed scratch checkpoint remains the recovery point.

### Corpus-authoring prompt to copy

Help me choose or prepare a fitting corpus for a J-lens on my selected model.
First ask what text population the lens should represent and whether I want to
use existing files, author the text myself, or delegate data creation. Do not
generate passages or launch agents without my agreement. If authoring is agreed,
return UTF-8 JSONL with unique id and nonempty text fields only, with enough
tokens after the chosen truncation and skipped-position settings. Keep source
and licensing notes separately. Propose held-out passages separately for later
readout assessment. Explain that this corpus fits an averaged instrument and
does not establish a behavioral concept. Do not invent model commits, execute
fitting, or download weights.

## Preparing an existing fitting corpus

The app's Fit a new lens → Data → Prepare corpus flow and both clients expose
`science corpus-preview <spec.json>` and `science corpus-publish <previewID>
--plan-sha256 <planSHA256> --destination prompts/fitting/<new-name> --json`.
Use `--root <workspace>` on command lines. The workbench API equivalents are
`POST /api/science/workspace/corpus-preview` with `workspaceRoot` and `specText`,
and `corpus-publish` with `workspaceRoot`, `previewID`, `planSHA256`, and
`destination`. All call one portable preparation owner.

A minimal local specification is
`{"source":{"kind":"local","files":["sources/text.jsonl"]},"count":1000,"seed":0}`.
Sources may be text, JSONL, CSV, or Parquet files already in the workspace; the
app can copy selected local originals there. Public dataset sources use
`{"kind":"huggingface","dataset":"owner/dataset","revision":"main","files":["train/*.parquet"]}`.
Choose actual repository paths for the intended configuration and split;
preview resolves and records the exact dataset commit. It can download those
public files, never model weights or dataset scripts. Gated sources need a
local export. Selected files are bounded to 2 GiB, and output to 64 MiB.

Optional fields: `textColumn` (default `text`), `documentColumn` (default null),
`selection` (`seeded` or `first`), `seed` (default 0), `count` (default 1000),
`minChars` (default 1), `passageChars` (0 keeps records), and `scanLimit` (default
100000). Sampling covers only scanned records; inspect the actual counts and
warnings. `first` with minimum 600 characters preserves the reference-style
long-record selection policy; seeded sampling is a different policy. A passage
is one seeded character window, not sentence segmentation. An optional
`tokenizer` object (`modelID`, exact `revision`, `maxSeqLen`, `skipFirst`) previews
lengths with a locally cached tokenizer; absence is guidance, not a data refusal.

Preview captures candidate bytes in `.steerlab/corpus-preparations/` and shows
examples. Publication verifies those bytes, creates a fresh directory, and
returns `fittingInputs.corpus` and `fittingInputs.corpusReceipt`. Use both in the
fit config; the receipt records sources, sampling, selected records, and token
review, and is captured in the fitting run. Existing manually authored corpora
can omit it. Prepare assessment text separately, preferably from another source
split; this action does not establish document-level independence.

Ask the researcher about the text population and sampling choices in plain
language. Do not infer permission to download, generate, or delegate dataset
creation from a model or concept choice. Preparation does not start fitting.

## Pilot measurements and reliable remote collection

A new fit writes `row-resources.jsonl` beside `progress.jsonl`. It records each
reference-kernel call's dimension batch, elapsed seconds, tokens, and available
CUDA memory measurements. `fit-report.json` includes per-layer finiteness,
allocated and reserved device peaks, process-lifetime peak host RSS, compilation
status, optional kernel package versions, and observations from executed
attention modules. An installed package is not proof its fast kernel executed;
unknown dispatch is reported as unverified. Non-CUDA device measurements are
unavailable rather than zero. A failure report names the phase, batch, latest
completed checkpoint, and available memory, including failures before backward.
These measurements describe this runtime and corpus; they do not establish
readout quality or predict proportional speedups from larger batches.

After packaging inputs, transfer the archive beneath the runner's configured
run root using the site's permitted transport. `runner science-stage` (Mac:
`remote science-stage`) receives that server path and the archive SHA-256. Both
clients save `.steerlab/diagnostic-requests/<digest>.json` in the local workspace
and return its absolute `localRequestPath`. Pass that file as the request
argument of `science-plan`, then of `science-submit` with the reviewed plan hash,
on the same controller. Both verbs re-verify the staged files, so a request that
carries a multi-gigabyte checkpoint can take minutes before either answers; both
clients allow an hour for these calls. The app carries this staged request automatically. An API
caller uses the `request` object returned by `POST /api/science/stage` directly.
Do not send the original published request to a runner that cannot see the
local workspace: its file references describe authoring inputs, not staged
execution copies.

Staging and export may hash or compress gigabytes before sending response bytes.
Both clients allow an hour for these requests; the Python client's explicit
request timeout overrides that default. `science-fetch` streams the archive,
checks its digest, and imports it through the existing custody owner. The
complete archive still includes the final checkpoint. No checkpoint is dropped
silently, and a verified transfer is not scientific qualification.

For older deployments or site-required direct transfer, have the controller
finish the idempotent export through its `/api/science/jobs/<job-id>/export`
endpoint with a suitably long client timeout. Obtain the returned `bundlePath`
and `bundleSha256`, transfer that **complete diagnostic evidence archive** using
the permitted transport, then run locally:

```sh
steerlab science import <local-evidence.tar.gz> --sha256 <bundleSha256> --root <workspace> --json
steerlab science verify-custody <receiptSHA256> --root <workspace> --json
```

The Mac equivalents use `steerlab-cli science`. A directory copied by rsync or
an ordinary tar archive is not this evidence format. If export is still running,
wait and request its reference again; do not resubmit the fit. Keep remote
originals until local custody verifies. Register the collected lens through
`science artifact-plan` and `science artifact-import`; qualification and held-out
assessment remain separate decisions.

A code push does not restart an existing controller. `cluster status` reports
running and deployed build identities when the controller supports those fields;
older controllers report them as unknown. Arrange a reviewed restart after
checking active work, then make fresh execution plans. Do not treat a successful
push as proof the running controller loaded the new code.

On filesystems without native create-only rename, an interrupted publication can
leave an empty directory claim. This is incomplete output. Confirm that no
publisher is live before removing only that empty directory by hand and retrying;
otherwise choose a new destination. Never delete populated output as a repair.
