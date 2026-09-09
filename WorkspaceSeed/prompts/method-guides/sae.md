# Analyze and import SAE features

Preserve feature identity and the import/normalization convention.

Choose the model, release, layer, width, feature IDs and intended use before searching for persuasive examples. In the app, Data → Concepts & Vectors → Import an SAE feature accepts explicit identifiers or a Neuronpedia feature link. Link lookup uses the installed SAELens directory; it never guesses an unknown mapping or downloads weights. The researcher reviews the selected feature and calibration before import can download weights. Analysis → Compare with Gemma Scope features compares existing vectors with features. Agents use `steerlab-server gemmascope resolve-feature --url <link>`, then `gemmascope import-id`, or the corresponding `/api/gemmascope` operations. The sae family exposes check, pin, family-report and qualification operations.

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

## Import a custom decoder or an exported feature

Supply a fitted decoder you already have, or one already-exported feature.
A coding agent should consult the artifact's documentation for the model, feature
axis, and hook site, and ask you about missing metadata. The miniature example
below uses a two-dimensional residual space only to illustrate the file format.

```json
{
  "schemaVersion": 1,
  "kind": "sae-decoder",
  "modelID": "example/model",
  "modelRevision": null,
  "hiddenSize": 2,
  "layerCount": 4,
  "tensorFile": "decoder.safetensors",
  "calibrationArtifact": "runs/calibration/donor",
  "sae": {
    "layer": 1,
    "feature": 7,
    "decoderKey": "W_dec",
    "featureAxis": "rows",
    "site": "resid_post",
    "label": "Candidate behavior"
  }
}
```

Use actual dimensions and a calibration artifact measured on the same Python
model. `featureAxis` is `rows` or `columns` for a decoder matrix, or `vector` for
one already-exported one-dimensional feature. With `vector`, `feature` records
its original feature number. The vector is scaled to the measured residual norm;
the donor supplies geometry and scale, not a semantic label. Known fit revisions
must match; null stays unknown. Source tensors must be finite and nonzero at the
selected feature. This path applies a residual-post direction; other hook sites
need a matching intervention, and full latent SAE manipulation needs more than
one decoder vector.

Safetensors and numeric NPZ work with the lightweight client. Tensor-only PyTorch
checkpoints and BF16 require the optional PyTorch reader. Optional `configFile`
retains original configuration bytes; optional `source` records a repository,
its exact revision, and/or a discovery URL without claiming a network fetch.

Use `science artifact-plan <description.json> --json`, then
`science artifact-import <description.json> --plan-sha256 <reviewed-hash> --json`
through either client, with its usual destination workspace option. In the app,
choose **Import my own SAE decoder files** in the SAE dialog. The destination
workbench must already have the calibration artifact named in the description.
Stage and review copies files, and Import publishes a new vector with retained
source bytes. It never trains an SAE, invents data, or proves the feature label.
