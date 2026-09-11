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


## From a pilot to a measured fitting round

Ask first what readout the researcher needs, which text should represent it, and
what compute budget they approve. Explain that a **fitting round** divides one
fixed total of corpus rows among independent jobs; merging combines their
contributions. More rows or a stable running average alone do not establish
that a lens answers the research question. Keep the declared held-out text
separate, and compare the readouts and the research conclusions.

Four managed operations are available from **Research methods → Author request**
in the app, `science interview/draft/publish` in either client, and the workbench
`/api/science/workspace/{action}` routes. Execution uses the usual package,
stage, plan, submit, inspect, fetch, and custody flow on either client or the
HTTP API. A runner executes; it does not author the researcher's workspace.

| Operation | Researcher choice | Result |
| --- | --- | --- |
| `jlens-fit-benchmark` | A published fresh fitting request of at most eight rows, dimension batches (baseline 1), kernel policy, compilation comparison, and numerical tolerances | Per-prompt gradient and mean-lens agreement, throughput, and memory, with failures visible |
| `jlens-fit-round` | A published fixed-budget fit, shard count, first global corpus row, partition layout, and concurrency cap | Immutable shard requests and a round plan; materializing this plan starts no GPU fitting |
| `jlens-fit-merge` | Completed disjoint fit directories, and whether missing planned rows are acceptable | A weighted merged lens, checkpoint, and explicit coverage/lineage report |
| `jlens-fit-assess` | Two registered lens IDs, pinned held-out text, layers, and token-position budget | Per-layer readout agreement and comparisons with the final residual; no automatic qualification |

### Measure speed and numerical agreement together

Use the same small corpus and exact checkpoint for each benchmark case. Choose
batches such as `1, 8, 16`; larger batches may exhaust memory. `kernelPolicies`
can be `current`, `torch`, or `torch,current`. The explicit Torch policy uses
inspectable model-owned fallbacks; if the installed model has no such fallback,
use the current policy and request a tested model adapter. `compareCompiled`
adds compiled cases. The first case is the comparison baseline. Each case runs
in a fresh subprocess, so a failed CUDA context does not contaminate the next.
The report compares every fitted prompt's matrices and the final mean, not
only a final score. Tolerances remain visible researcher choices.

Timing excludes model loading, includes first-call compilation and temporary
matrix writes, and applies only to the measured rows and environment. Temporary
matrices retain the baseline and current case until comparison, then are
removed; allow disk space for both sets of per-row matrices. A killed worker
can leave scratch under `.steerlab/jlens-benchmark-state/`. A benchmark report
contains measurements, not an importable lens. Failed cases remain in the
report; a `COMPLETED` report is not a claim that every case passed.

A successful report can be selected as the optional **Measured pilot report**
(`benchmarkReport`) in a later fit. The review shows its matching case's rows
per hour and an extrapolation at the requested row cap, with runtime and
limitations. This is not a scheduler walltime guarantee. The default dimension
batch remains 1. No benchmark changes defaults or approves a continuation
across a numerical-runtime mismatch. Kernel policy, compilation, optional
package versions, Torch, and Transformers stay bound in checkpoint identity.

Optional Linux kernel candidates are available in the explicit
`Server[jlens-kernels]` extra. This is separate from normal setup and from the
pinned `jlens` reference extra. It is **not a resolved CUDA environment lock or a
qualified configuration**. After the researcher approves an environment change,
the running agent should first obtain a `pip install --dry-run --report ...`
plan in the intended environment, review dependencies and build prerequisites,
and then install that reviewed selection. Installing `jlens-kernels` neither
acquires weights nor imports lenses. Do not run `bootstrap --with-jlens` merely
to add kernels: that existing option also acquires the curated lenses. Never
change a running job's environment. Compare fallback and accelerated gradients
and lenses before interpreting a speedup; package presence alone does not prove
which kernel ran. Qualification remains a separate instrument check.

### Review and execute a fixed-budget round

The base fitting request's `maxPrompts` is the **global row budget**, including
short rows later skipped. `startRow` is zero-based in the same pinned corpus.
Interleaved partitions distribute successive rows across shards; contiguous
partitions give each shard a consecutive block. The plan lists every global
row and the per-shard count, and shows minimum retained lens/checkpoint storage
when model dimensions are cached. Budget also for staging, exports, and
recovered checkpoints. This is not a peak-memory estimate.

After materialization, use its durable job ID in the app's **Scientific inputs,
evidence and cleanup → J-lens fitting rounds** controls. Review the shard queue,
confirm the displayed plan, and top it up. Review again as capacity becomes
free. The cap counts active scientific jobs on this controller; it does not
replace the site's scheduler policy or account for unrelated controllers.
Uncertain submissions reserve capacity and are not retried. Inspect durable
jobs and scheduler state before reconciling uncertainty; never resubmit merely
because a request timed out. Cancellation requests must be followed by a job
status check. There is no automatic background refill.

Both command lines can invoke these same controls with `science-call`:

```sh
steerlab runner science-call jlens-fit-round --action post-fitting-round --request round-action.json --runner <url> --json
steerlab-cli remote science-call jlens-fit-round --action post-fitting-round --request round-action.json --site <id> --json
```

For review, `round-action.json` is:

```json
{"path":{"job_id":"<materialized-round-job-id>","action":"plan"},"query":{},"body":{}}
```

The HTTP equivalent is `POST /api/science/fitting-round/<job-id>/plan` with `{}`.
`status` and `merge-plan` also take an empty body. `submit`, `cancel`, and
`merge-submit` require `{"planSHA256":"<fresh-plan-hash>","confirmAction":true}`.
Use the hash from `plan` for submission/cancellation and the hash from
`merge-plan` for merging. Mutations recheck inputs and capacity. The round's
shard requests automatically use its verified execution capsule; do not hand
edit their paths. Plan and submit allow time for multi-gigabyte verification.

Each shard can be continued from its own checkpoint using ordinary `jlens-fit`
continuation and its original row selection. A continuation is a new run. The
round helper merges its original completed child jobs; for continued shards,
author `jlens-fit-merge` explicitly with each shard's **latest** completed run.
Never include both a shard and its continuation or an earlier merge covering
the same rows. The owner checks global coverage under the same pinned corpus
and numerical identity and refuses overlaps, including ancestor contributions.

Merge uses raw float32 sums weighted by fitted-row counts, excluding skipped
rows. It records accumulation order (ascending first global row, then checkpoint
hash), missing rows, and source hashes. Grouped float32 addition can differ
from serial addition; compare within declared tolerances. A partial merge stays
labelled partial. Collect and verify its evidence, then register its
`artifact-description.json` with the usual artifact-plan/import workflow.
Registration now carries the fit report hash, reference commit, kernel hash,
and driver hash when the verified source report supplies them. Third-party
artifacts with unknown provenance remain unknown.

### Define stopping and assess readouts

Optional `stopping` on a serial fit has the shape
`{"threshold":0.002,"window":10,"minPrompts":100}`. Those values are an example,
not defaults or a promise that 100 rows suffice. The statistic is the maximum
over source layers of the relative Frobenius change of the running mean.
Stopping requires N consecutive valid values strictly below the threshold,
after the minimum number of **successfully fitted** rows. Skipped rows do not
advance the window. A zero-norm previous layer mean makes the statistic
unavailable, clears the window, and continues to the row budget; it never
counts as convergence. The window travels in checkpoint state. Reports name
whether stability or the row budget ended the fit. Fixed-budget shards refuse
per-shard stopping: independent stopping changes the sample and does not
reconstruct the serial convergence series.

For `jlens-fit-assess`, choose two registered versions and held-out text. On the
same captured source activations, the owner applies each J matrix, then the
model's actual final normalization and unembedding. It reports Jensen–Shannon
divergence in nats and top-k token-set overlap, both between lenses and against
the actual final residual. Aggregation weights assessed token positions equally;
only the first eligible positions up to the declared cap are assessed. Position
and layer choices are recorded. These are distributional readout comparisons,
not causal interventions. A matching fitting-corpus hash is flagged; a different
hash does not establish independence. Neither metric automatically qualifies a
lens or decides whether the research conclusions are stable.

The corpus, checkpoints, and completed runs remain immutable. Hash reuse is
limited to unchanged regular files within one bounded input-review operation;
a new request and the queued worker verify afresh. It does not eliminate every
plan, submit, and worker read. No general fitting-run or checkpoint cleanup has
been added. Keep original remote evidence until verified local custody, and
follow the site's storage policy through an explicitly reviewed cleanup.
