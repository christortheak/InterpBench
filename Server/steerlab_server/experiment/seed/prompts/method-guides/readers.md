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
