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
