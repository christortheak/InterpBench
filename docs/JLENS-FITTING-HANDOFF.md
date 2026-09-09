# J-lens fitting: review and qualification handoff

Branch: `codex/jlens-fitting`. Base: `d42376f`, including the landed CLI help
fix. Plan commit: `ef5cb8d`; implementation commit: `68cb749`. This branch does
not change main or install the app. Review and integration
remain with the maintainer's auditing agents, through the researcher.

## What changes for a researcher

The J-lens panel now offers **Fit a new lens…**, including when no execution
server is connected. The shared authoring dialog asks for a prepared model,
its exact version, and an existing text corpus. It explains the corpus format,
offers instructions to copy to an agent, and defaults to a four-row timing pilot.
It neither generates data nor downloads weights automatically.

The operation is `jlens-fit`. Both clients discover it with
`science interview jlens-fit --json`; their existing draft, publish, package,
execution, and collection commands use the same owner. HTTP uses the existing
`POST /api/science/plan` and `/api/science/submit` routes. The app reuses this
authoring and execution workflow, rather than implementing another estimator.

Completed evidence contains captured corpus bytes, effective configuration,
timings, skipped-row diagnostics, float32 Jacobians, a final checkpoint, and
`artifact-description.json`. After collecting the run, review and import that
description with the existing artifact importer (the app's **Import my own lens
files…** action). This deliberate registration step keeps completed runs
immutable and puts qualification records in the lens library.

## Implementation and scientific contract

- `jlens_fit.py` owns portable configuration, pinned JSONL input, checkpoint
  companion discovery, and preflight. Authoring does not import GPU packages.
- `jlens_fit_model.py` loads an already-prepared, unquantized Hugging Face
  checkpoint offline. It selects a causal or conditional-generation loader,
  disables generation caches, uses eager attention, freezes weights through
  the pinned reference adapter, and records runtime and source identities.
- `jlens_fit_execution.py` orchestrates the unchanged pinned reference
  `jacobian_for_prompt` kernel. The independent causal fixture verifies its
  orientation and weighting: sum valid target cotangents, average valid source
  positions, then average usable prompts equally. This is not token-weighted
  averaging, model training, OptVec optimization, or SAE training.
- Raw text keeps the tokenizer's native special-token behavior; no chat template
  or forced BOS is inserted. The leading positions and final token are excluded
  by the documented reference mask. Only short rows are skipped; backward errors
  fail visibly. Layer coverage, corpus order, precision, and model revision are
  recorded. Float32 accumulation does not turn lower-precision model derivatives
  into float32 derivatives.
- The managed execution callback records the output directory before model
  loading. The input closure includes checkpoint JSON and its hashed companion
  tensor file. Lens registration now retains optional `fitDtype` metadata.

Fitting completion does not establish readout validity, causal interpretation,
or backend qualification. The default pilot is a cost and compatibility probe.
No large-model performance claim follows from the automated fixtures.

## Continuation, custody, and storage

Periodic state lives under `.steerlab/jlens-fitting-state/<run-id>`. Each
snapshot is published coherently before this invocation removes its preceding
scratch snapshot. Successful runs retain a final checkpoint and remove their
own temporary copies; failed jobs retain their last completed snapshot.
Completed runs and source checkpoints are never rewritten by continuation.

A new reviewed request may name a checkpoint's `state.json`, accompanied by
`sums.safetensors`. The exact corpus, model, estimator, geometry, and recorded
runtime must match. The prompt limit may increase. This is explicit continuation
in a new run, not automatic scheduler resubmission. If a remote job fails, wait
for it to stop, and recover the pair using the site's permitted transfer tools;
failed jobs are not exported as completed evidence.

One float32 matrix set costs `sourceLayers × hiddenSize² × 4` bytes. A completed
run contains the mean and final checkpoint sums; fitting also needs working
matrices, model weights, and backward activations. Checkpoint publication can
temporarily retain two scratch generations. The existing 16 GiB diagnostic
archive limit remains; unusually large fits may require approved external file
transfer. No global cleanup or cluster-policy bypass is introduced.

## Validation and review gates

The Python suite passed **6,315 tests**, with 9 skips and 8 warnings. Fitting
fixtures cover analytic reference agreement, chunk sizes, exact continuation,
immutable earlier runs, malformed and changed inputs, checkpoint corruption,
visible kernel failure, input packaging, evidence return, lens registration,
HTTP planning, and managed child execution. Random tiny hybrid text-only and
conditional-generation models exercise the real offline Hugging Face loader,
reference adapter, and backward kernel on CPU without downloading weights.

The full Xcode beta suite passed: **290 SteeringKit tests and 4,613 ExperimentKit
tests**. The app also compiled in this build. The added Swift authoring test
sends the shared interview through the actual Python draft and publish owner.
Both real client executables passed interview → draft → publish → input-plan
smokes. A separate portable draft/validation smoke forbade imports of torch,
transformers, and jlens and passed. Generated resources, compiled client identity,
CLI reference, established AST audits, bridge gates, public scan, and
`git diff --check` passed. The complete source diff was read before committing.

Validation logs on the implementation machine:

- `/private/tmp/jlens-fit-python-full.log`
- `/private/tmp/jlens-fit-swift-full.log`
- `/private/tmp/jlens-fit-final-gates.log`

The suites ran serially. Xcode used the beta developer directory, the installed
Metal toolchain identifier, an external derived-data directory, coverage mapping
disabled, and the existing test Python interpreter. The researcher's installed
app was not replaced. Live UI acceptance remains below.

No numerical body is claimed as mechanically moved. The historical operation
registration AST audit retains its original baseline and strips only four exact,
reviewed dispatch additions: the module binding for fitting preflight, that
preflight call, the optional execution callback parameter, and its forwarding.
Negative controls reject mutations to those extensions; original owner bodies
and declarations remain protected. Other established AST and bridge gates run
unchanged. Read the full diff and require both suites before integration.

## Live acceptance after review

1. Build the app from the landed source before using shared Python authoring.
   Open J-lens fitting, inspect corpus guidance and tooltips, author a pilot,
   review its inputs, and follow execution and collection to artifact import.
   Confirm the resulting lens appears in its model's library. Automated tests
   do not replace this visual and interaction check.
2. On the chosen cluster, prepare the researcher's exact requested checkpoint
   and supply their chosen corpus. Pin its actual commit; measure a small pilot
   on the intended allocation before increasing corpus size. Record peak memory,
   per-prompt timing, effective precision, finite outputs, and skipped rows.
   The full requested 27B checkpoint and CUDA kernels remain unverified here.
3. Test stop/recovery with a completed checkpoint, continuation in a new job,
   collection, and registration. Estimate runtime from that pilot, not from
   tiny CPU fixtures. Then qualify the registered lens and assess readouts on
   separate text. Record limitations as guidance; do not treat absent scientific
   qualification alone as a reason to block exploratory authoring.
