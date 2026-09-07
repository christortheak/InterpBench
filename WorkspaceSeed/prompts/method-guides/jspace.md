# Inspect an intervention in J-space

Separate arithmetic fidelity, readout predictiveness and causal evidence.

Use this exploratory analysis when the researcher wants to inspect the lens readout of a vector's downstream residual change. It does not train an intervention or replace a behavioral study.

Run steerlab-server optvec jspace --config <file>. Required config fields are vectorArtifacts (extension-less artifact paths), lensID and probeItems:{path,sha256}. The probes are pinned choice-row JSONL. Optional observationLayers must be fitted source layers at or above the injection layer. Other supported keys include alphaMultiple, nullSeed, nullDraws, seed, topK, microbatchSize, modelID, revision, device, dtype, promptMode, systemPrompt, qwenThinkingEnabled and name. Unknown keys refuse.

Ask the researcher to choose the vector, lens, probes, observation layers and dose. Obtain hashes from inspected bytes, never fabricate them. alphaMultiple scales the artifact's own norm; it is not the ordinary residual-norm alpha. The seed field is inert here because no generation sampling occurs; nullSeed governs matched-norm random draws.

Report arithmetic fidelity and precision-floor flags separately from predictive readout. Null draws are comparators, not a calibrated test. Behavioral/causal evidence is absent from this operation; collect it through a separate controlled run. The engine CLI is the supported execution path; the app guide exposes that restriction explicitly.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.

## Shared method authoring and managed execution

Use `science interview <operation> --json` to read the exact form fields used by the app. Answer the purpose, proposed claim, controls and selection questions; supply field values as strings (including integer seeds) and put reviewed extra engine settings in `advanced`. `science draft <operation> --answers <answers.json> --json` resolves file hashes and reports all captured inputs without execution. `science publish <operation> --answers <answers.json> --destination requests/<new-name> --plan-sha256 <reviewed-hash> --json` publishes the machine request and rationale together, refusing changed inputs or an existing destination. Both commands are available under each client executable.

Use the resulting request file with `science input-plan` and `science package`, transfer and stage through the declared site policy, then review and submit the engine plan. The shared form is an authoring aid; the engine still checks its complete scientific config before execution. A successful parser or custody check is not scientific qualification. The workbench API exposes the same local owner at `POST /api/science/workspace/{action}`; it takes `workspaceRoot`, `operation`, and for draft/publish `answersText` (the exact JSON text). Publish also requires `destination` and `planSHA256`. Runner services cannot author the local workspace.

J-space is an analysis method, not an OptVec training procedure. Its public managed operation is `jspace`. The historical engine command `optvec jspace` currently accepts OptVec artifacts because it reads their stored layer and absolute dose; that implementation restriction is not a claim that J-space depends scientifically on optimization. Other intervention artifact types require a separate admission change.
