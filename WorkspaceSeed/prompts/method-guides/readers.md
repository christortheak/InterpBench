# Train and use an activation reader

Measure an activation-based instrument separately from an intervention.

Decide the construct, template, label interpretation, training split, held-out evaluation split and scoring position with the researcher. A reader's score is not automatically an intervention or a behavioral outcome.

Use the existing reader template registry and reader fit/score interfaces. Request the schema and authoring prompt for the selected template; do not substitute ordinary contrastive-pair files for template-mediated reader inputs. Preserve label alignment, model revision, tokenizer/rendering settings and the exact fitted artifact.

Inspect the Readers pane or POST /api/reader/fit and /api/reader/score on a workbench engine. The registry below names supported public paths; these workbench routes are not advertised as batch-runner submissions. Treat training accuracy separately from held-out instrument performance. If importing a reader-derived steering vector, use experiment inspect-artifact and attach-artifact with exact artifact/sidecar/manifest reviews and retain its reader provenance.

Outputs are the fitted reader artifact and scored records. Report the training/evaluation split and applicability to this model and rendering. A successful fit does not establish causal control of the named construct.

## Dataset and handoff

Choose exactly one row shape. `contentPair` compares content under one template:

```json
{"id":"pair-1","concept":"construct","positiveStimulus":"First matched text.","negativeStimulus":"Second matched text.","topic":"shared topic","split":"train","templateID":"chosen-template"}
```

`singleStimulus` compares two template instructions on the same content:

```json
{"id":"row-1","concept":"construct","stimulus":"An ambiguous situation.","topic":"shared topic","split":"train","templateID":"chosen-template-pair"}
```

Emit `authoring prompt reader-pairs --concept <concept> --positive <definition> --negative <definition> --template-id <template> --shape <contentPair|singleStimulus> --json`. Trailing `split:"test"` rows orient the fitted sign in this implementation; reserve an additional independent evaluation set before making a generalization claim. Do not relabel sign-selection data as untouched final evaluation. Templates are real registered inputs, not arbitrary IDs to invent. The fit/score request schema is available in the workbench's `/openapi.json`.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.


## Probe library: shared discovery and inspection

The Probes section is the library for saved activation probes. Inspect it from
an agent with `steerlab science probe-list --root <workspace> --json`, or
`steerlab-cli science probe-list --workspace <workspace> --json`. Use
`science probe-inspect runs/<run>/<name>.probe.json` with the same workspace flag
to inspect one entry. Legacy Python `<name>-probe.json` files are also recognized.
Both workbench HTTP implementations expose
`POST /api/science/workspace/probe-list` with `{"workspaceRoot":"<workspace>"}`,
and `POST /api/science/workspace/probe-inspect` with the same root and a `path`.
These actions inspect the serving workbench's workspace; they do not access a
runner's remote library or copy files automatically. Collect run evidence first.

A list reports `probes`, `issues`, and `count`. Inspection reports the relative
path, exact-byte SHA-256, format, model, layer, method, limitations, and original
JSON document. The portable CLI places this under `result`; the native CLI's
existing workspace adapter places it under `result.response`. HTTP returns it
directly. All operations are read-only; malformed matching files are reported as
issues rather than silently dropped. Inspection is bounded to 64 MiB per JSON
file; activations belong in separate datasets. Hashes identify the original file,
not a re-encoding of the returned document.

Existing Mac readers and Python readers retain their original formats and score
semantics. In particular, the legacy Python reader's `heldOutAccuracy` selected
a layer; it is not final-test accuracy. Missing model, rendering, or coordinate
pins stay unknown. Discovery does not establish cross-backend compatibility.
Native Playground reading still uses its existing supported reader path; the
new portable classifier artifact is not silently inserted into that picker.

The new `activation-probe` v1 format supports explicit binary linear and small
ReLU classifier parameters, preprocessing, input bindings, and score semantics.
Its CPU reference scorer is an implementation/validation API, not a new model
capture or fitting command. Unified training, study measurements, and conditional
interventions are subsequent implementation slices. Do not invent training verbs
or generate data because a researcher has only named a concept. Offer existing
files, pasted data, author/reviewer prompts, or explicitly chosen coworkers.
