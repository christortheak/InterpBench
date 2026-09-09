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
