# Import your own fitted lens or SAE decoder

Use existing artifacts from your own work or another source. Importing a J-lens
adds a measuring instrument to the lens library. Importing an SAE feature adds
one decoder direction to the vector library. Neither operation trains anything,
loads a language model, or establishes that a feature has the behavioral meaning
suggested by its label.

In the app, open the J-lens or SAE dialog from Concept Vector Builder, then choose
**Import my own … files**. Choose a JSON description and keep its tensor file and
optional source configuration at the relative paths it names. The dialog can
copy the complete method guide for a coding agent to help prepare the description.
The source guide includes miniature examples; replace their geometry with the
artifact's actual dimensions and layer mapping. No researcher dataset is generated.

The app explicitly stages the selected files to the connected Python workbench,
then displays a plan. After review, Import publishes to a new library location.
Local original files remain in place. A workbench can run on this Mac; no GPU is
needed for import. Staged source files remain under `.steerlab/artifact-inputs/`;
completed artifacts retain their own source copies and an import receipt. This
slice does not add automatic remote cleanup or a runner-managed evidence return.
Use the existing workspace transfer workflow to bring remote library artifacts
home, and retain original files until you have verified their local copies.

## Formats and metadata

The small description is JSON with `schemaVersion: 1`, `kind` (`jlens` or
`sae-decoder`), `modelID`, `modelRevision` (null if unknown), `hiddenSize`,
`layerCount`, and `tensorFile`. Tensor paths and optional `configFile` are relative
to the description's folder, without parent traversal or symbolic links. Optional
`source` contains `repository`, its exact `revision`, and/or a discovery `url`.
These are recorded declarations, not a claim that the importer fetched or
independently authenticated that repository. There is no network discovery in
this importer. Original configuration bytes are retained without reinterpretation;
the description explicitly supplies the conversion contract.

- Safetensors and numeric NumPy `.npz` archives work with the lightweight client.
  F16, F32, and F64 retain their dtype in lens conversion. BF16 requires the
  optional PyTorch reader; this never silently converts the stored lens to F32.
- Tensor-only `.pt`/`.pth` checkpoints require PyTorch. Loading uses
  `weights_only=True`; arbitrary pickle objects are not accepted. A saved
  JacobianLens dictionary (`J`, `source_layers`, `d_model`, and `n_prompts`) is
  recognized. A raw, unfinished fitting checkpoint is a different format.
- A J-lens description supplies `lens.targetLayer` and `lens.layers`, mapping
  source-layer numbers to actual tensor keys. All source layers precede the final
  block, `targetLayer == layerCount - 1`. Missing source layers are permitted when
  explicitly omitted; those lenses support readouts only at the fitted layers.
  Token-vector derivation currently requires every source layer before the final
  block, so the review explains when a partial lens cannot create that vector.
  Each selected matrix is finite, square, and `hiddenSize`
  wide. Optional `promptsFitted`, `corpus`, and `maxSeqLen` retain fitting metadata.
  A saved checkpoint's geometry and prompt count must agree with the description.
- An SAE description supplies `sae.layer`, `feature`, `decoderKey`, `featureAxis`
  (`rows`, `columns`, or `vector` for one exported direction), `site: resid_post`,
  and `label`. A single exported feature retains its declared feature number;
  that number is not a row index into a one-vector file. Matrix orientation is
  explicit even when the decoder is square. Other hook sites require a matching
  intervention path, not a renamed residual vector.
- SAE import also requires `calibrationArtifact`: a workspace-relative vector
  base path without extensions. This supplies measured residual norms for the
  same Python model and geometry. Known model revisions must agree; unknown SAE
  fit revisions remain unknown in the result. Source labels are not borrowed from
  the calibration donor. The decoder row is scaled in F32 to the measured norm.
  A zero row, non-finite values, or a non-positive calibration norm needs repair.

Exact JSON examples are maintained in the packaged method guides:
[custom J-lenses](../WorkspaceSeed/prompts/method-guides/jlens.md) and
[custom SAE decoders](../WorkspaceSeed/prompts/method-guides/sae.md).
The app copies these same guides, and agents can read `science guide jlens` or
`science guide sae` through either client without a checkout.

## CLI and HTTP

Both authoring clients use the same owner. Python:

```sh
steerlab science artifact-plan /path/to/import.json --root /path/to/workspace --json
steerlab science artifact-import /path/to/import.json --plan-sha256 <reviewed-hash> --root /path/to/workspace --json
```

Mac: use `steerlab-cli science artifact-plan` and `artifact-import` with the same
positionals and review hash, selecting the workspace through its existing
`--workspace` option. The Mac's helper environment needs the optional PyTorch
reader only for PyTorch/BF16 sources; ordinary safetensors/NPZ imports use the
packaged lightweight client. The engine distribution also installs `steerlab`,
so these same verbs are available on an engine host without another implementation.

The Python workbench exposes:

- `POST /api/artifact-imports/stage/{source_id}/{file_path}`: stream original
  bytes with `X-Content-SHA256`. Use a fresh 32-character lowercase hex source ID.
  Paths are ordinary relative components. Each file is create-only, and uploads
  exceeding 32 GiB need the existing cluster file-transfer tools instead.
- `POST /api/artifact-imports/plan` with `{"descriptionFile":"<staged-path>"}`.
  Returns geometry, limitations, file hashes, and `planSHA256` without publishing.
- `POST /api/artifact-imports/import` with the same `descriptionFile` and
  `planSHA256`. Returns a durable job ID; success includes `lensID` or `vectorPath`
  and the output directory. A change between review and execution fails that job
  without publishing an artifact.

The plan/import actions are also discoverable under the existing `jlens` and
`gemmascope` catalog entries, callable via the clients' science-call adapters.
Binary staging uses the streaming HTTP endpoint or existing file-transfer tools.
The existing workbench `/api/science/workspace/artifact-plan` and
`artifact-import` actions are synchronous adapters with `workspaceRoot`,
`descriptionFile`, and (for import) `planSHA256`. These also exist on the Mac's
HTTP service. Runner-only deployments reject workbench authoring.

## Retention and downstream behavior

Every successful publication gets a fresh ID, even for a second import of the
same source. No old lens or vector is replaced. Lenses start without qualification
records; model-tier labels are not qualification. Imported lenses use the same
per-layer reader and token-direction/readout machinery as published lenses.
The output receipt binds the reviewed source files and retained copies.
Token directions from custom imports name `custom-jacobian-lens` as their source.
Derivation checks the fitted model identity and any known fit-time revision
against the checkpoint whose token embeddings it uses.

SAE vectors retain the existing `gemmaScopeSAE` wire-method spelling so existing
attachment, identity, and injection paths can read them. That historical spelling
is not source provenance: `gemmascopeSource.importPath` is `custom-sae-decoder`,
with the actual source hashes, feature axis, layer, and calibration recorded.
The app labels the family “SAE feature (decoder row).” This imports a decoder
intervention, not the encoder, thresholds, or a complete latent SAE.
