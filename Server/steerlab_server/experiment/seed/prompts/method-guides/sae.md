# Analyze and import SAE features

Preserve feature identity and the import/normalization convention.

Choose the model, release, layer, width, feature IDs and intended use before searching for persuasive examples. Use the existing Gemma Scope pane or /api/gemmascope operations for analysis and import; steerlab-server gemmascope import-id provides a direct named-feature path. The sae family exposes check, pin, family-report and qualification operations.

Inputs are release/model metadata, feature identity and any declared analysis corpus. A feature selected from a report and a feature imported directly by ID have different normalization conventions. Preserve the sidecar's gemmascopeConvention: analyzed-vector-norm-match for report-ranked imports, residual-norm-match for direct feature-ID imports. An unstamped legacy artifact requires the documented re-import/qualification repair before evidence use.

Use experiment inspect-artifact and attach-artifact to attach the actual tensor and sidecar with their reviewed hashes. Never relabel a decoder row as an ordinary contrastive extraction or rewrite an old artifact stamp by hand.

Outputs are analysis/qualification records and immutable vector artifacts with feature provenance. Reconstruction, activation examples and steering effects are separate measurements. Report selection criteria, held-out behavior and capability costs; an interpretable feature name is not independent construct validation.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.

## Shared method authoring and managed execution

Use `science interview <operation> --json` to read the exact form fields used by the app. Answer the purpose, proposed claim, controls and selection questions; supply field values as strings (including integer seeds) and put reviewed extra engine settings in `advanced`. `science draft <operation> --answers <answers.json> --json` resolves file hashes and reports all captured inputs without execution. `science publish <operation> --answers <answers.json> --destination requests/<new-name> --plan-sha256 <reviewed-hash> --json` publishes the machine request and rationale together, refusing changed inputs or an existing destination. Both commands are available under each client executable.

Use the resulting request file with `science input-plan` and `science package`, transfer and stage through the declared site policy, then review and submit the engine plan. The shared form is an authoring aid; the engine still checks its complete scientific config before execution. A successful parser or custody check is not scientific qualification. The workbench API exposes the same local owner at `POST /api/science/workspace/{action}`; it takes `workspaceRoot`, `operation`, and for draft/publish `answersText` (the exact JSON text). Publish also requires `destination` and `planSHA256`. Runner services cannot author the local workspace.

The offline roster paths are `science sae-check <roster>`, `science sae-show <qualification>`, and `science sae-pin-plan <roster> --experiment <draft>`. Pin with `science sae-pin <roster> --experiment <draft> --plan-sha256 <reviewed-hash>`. Paths are workspace-relative. Both CLIs and the app call the same portable owner. The workbench endpoint uses actions `sae-check`/`sae-show` with `workspaceRoot` and `path`, and `sae-pin-plan`/`sae-pin` also take `experiment`; pin adds `planSHA256`. These are workbench operations, never runner authorship.
