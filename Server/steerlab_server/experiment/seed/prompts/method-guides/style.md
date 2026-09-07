# Rescore recorded generations

Compute declared style features without rerunning generation.

Choose the taxonomy, source run and interpretation with the researcher. The study must pin the intended taxonomy; the source must satisfy the owner's epoch checks.

Use experiment rescore-style on the Mac or Python engine. There is no dedicated HTTP rescoring route. This is CPU-only post-hoc computation over recorded sampled generations. The source run is immutable: outputs go to a fresh rescore run with reasoning-style.csv and reasoning-style.json.

Keep this explicitly post-hoc when the features were chosen after seeing outputs. A style score is not a new generation, an extraction, a judge verdict or proof of a psychological state. Report taxonomy/source pins, missing outputs and whether the feature definitions were preregistered. Do not bypass an epoch refusal by rewriting the manifest or source metadata.

## Explicit source

On the Mac: `steerlab-cli experiment rescore-style <study> --run <run> --json`.
On the Python engine: `steerlab-server experiment rescore-style <study> --source <run> --json`.
These source flags deliberately retain their existing spellings. Read `result.runDirectory` from the Python response; the output contains the new reasoning-style CSV/JSON. Choose and pin the taxonomy before rescoring. Do not describe a post hoc taxonomy choice as preregistered, or use a legacy-epoch override without the researcher's explicit acceptance of that limitation.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.

## Shared method authoring and managed execution

Use `science interview <operation> --json` to read the exact form fields used by the app. Answer the purpose, proposed claim, controls and selection questions; supply field values as strings (including integer seeds) and put reviewed extra engine settings in `advanced`. `science draft <operation> --answers <answers.json> --json` resolves file hashes and reports all captured inputs without execution. `science publish <operation> --answers <answers.json> --destination requests/<new-name> --plan-sha256 <reviewed-hash> --json` publishes the machine request and rationale together, refusing changed inputs or an existing destination. Both commands are available under each client executable.

Use the resulting request file with `science input-plan` and `science package`, transfer and stage through the declared site policy, then review and submit the engine plan. The shared form is an authoring aid; the engine still checks its complete scientific config before execution. A successful parser or custody check is not scientific qualification. The workbench API exposes the same local owner at `POST /api/science/workspace/{action}`; it takes `workspaceRoot`, `operation`, and for draft/publish `answersText` (the exact JSON text). Publish also requires `destination` and `planSHA256`. Runner services cannot author the local workspace.
