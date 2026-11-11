# Import your own fitted lens or SAE decoder

Use existing artifacts from your own work or another source. Importing a J-lens
adds a measuring instrument to the workspace's lenses. Importing an SAE feature adds
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
then displays a plan. After review, Import publishes into the workspace.
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
  wide. Optional `promptsFitted`, `corpus`, and `maxSeqLen` retain fitting metadata. Optional `fitDtype` records the known model execution precision (float32, float16, or bfloat16); omit it when unknown.
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
  Paths are ordinary relative components. Each file is create-only. Uploads obey
  `STEERLAB_MAX_UPLOAD_BYTES` (default 4 GiB), with a ceiling of 32 GiB. Larger
  files need the existing external file-transfer tools instead.
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

One lens, one name. A custom lens is named from the files that were reviewed:
`custom-lens-` followed by the first 32 hexadecimal characters of the SHA-256 of
the canonical JSON `{"descriptionSHA256", "kind": "jlens", "tensorSHA256"}`, the
file hashes the plan shows. When the description declares a `configFile`, that
JSON also has `"configSHA256"`: a fit report decides the lens's fitting
provenance, so another report is another lens. The same reviewed import
therefore gives the same lens ID in any workspace and on any machine.

Importing files that a workspace already holds as a lens finds that lens. The
match is on the lens's `import-receipt.json`: the same files, role for role, with
the same SHA-256. The lens's record must agree with it: `source.tensorSHA256` is
the tensor file's hash and `source.configSHA256` the description's. A receipt
that lists these files beside a record of other bytes is not this lens. The match
therefore also finds a lens imported before this rule, which keeps the random ID
it was given. A different description (a different intended use, say) or a
different config file is a different lens with its own name.

One lens has no receipt: a lens that was sent to a site. `lens-push` registers
the lens's record in the engine's lens store, under the lens's own name, and
writes no receipt, because an import is what writes one. Importing the same
files in that engine's workspace still finds that lens. A folder under the name
these files give a lens (the name is a hash over every reviewed file), whose
record names that lens and agrees with the plan as above, is the lens even with
no receipt. Under any other name a lens needs its receipt, and a receipt that
is there must list these files whatever the folder is called. So a lens that
kept an older name (one imported before names followed the files) is not found
this way: importing its files on the engine makes a second lens there, under
the name the files give, with its own copy of the bytes.

The lens that is found is checked before the import answers:

- Its converted file is in place and has the SHA-256 its record names: the import
  returns the lens with `existing: true` and `changed: false`, and writes nothing.
- Its converted file is not in the workspace, and the import runs on an engine
  whose lens library holds that file under the hash the record names: the lens
  is whole. That is where a site keeps a lens's bytes, once, and everything that
  reads the lens on that engine reads them there. The import returns the lens
  with `existing: true` and `changed: false`, puts nothing back, and makes no
  second copy. The import does not hash the library's file. The library checked
  the whole file against that hash when it took it, and a study's readout and a
  queued request each hash what they read, as they do a file in the workspace.
  A machine with no lens library, which is every researcher's own machine
  unless one was declared, never takes this path.

  When the lens that is found is one the site was sent (a record with no
  receipt), the engine does not convert the reviewed files again for this
  answer. It answers on the record that was sent: that the bytes filed under
  the record's `converted.sha256` are what these files convert to rests on the
  sender, whose client verified its own converted file against that record
  before it sent either. A lens that was imported on the engine has a receipt,
  and its record is the engine's own conversion.

  One consequence, on a machine whose environment names a lens library that
  holds a lens's bytes (a site engine's own workspace): a repeated import
  leaves the file in the library, so `lens-push`, `lens-status`, `lens-copy
  --from`, and packaging run from that workspace still report the file
  missing. Work from the researcher's workspace.
- Its converted file is missing, and no lens library on this machine holds it:
  the import converts the reviewed sources again and puts the file back when it
  has the SHA-256 the record names (`existing: true`, `changed: true`,
  `restored: true`). Nothing else in the lens folder changes. This is what
  "import the lens again" does when a lens verb refuses a lens whose file is
  gone.
- Its converted file is there with other bytes, or its record names no converted
  file: the import refuses, names the lens, and says what to move aside. It never
  overwrites a file.
- The disk cannot take a single file into the lens folder (the file is put back
  with a hard link, which ExFAT, for one, does not have): the import refuses in
  the same words, changes nothing, and says to move the lens folder aside and
  import again. A whole lens folder is published another way, so that works on
  the same disk.

Both command lines show a refusal with its repair, and so does the app's import
dialog. The record in the answer is strict JSON: a number in it that is not
finite is null, and `recordNonFiniteNumbers` counts them.

A folder under the lens's name that holds no such lens is refused at once, before
anything is copied or converted. A symbolic link under that name is refused as a
link, even when it leads to the lens's folder: a lens is read only from an
ordinary folder inside the workspace. One unreadable record elsewhere in
`runs/jlens-lenses/` does not stop an import.

An SAE decoder import still gets a fresh ID every time. No old lens or vector is
ever replaced. Lenses start without qualification
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

## The same lens elsewhere

A lens lives in one workspace's `runs/jlens-lenses/<lensID>/`, and the workspace
stays its source. Two verbs take it elsewhere without minting another lens:

- **A second workspace on the same machine.** `science lens-copy <lensID> --from
  <workspace directory>` copies the lens folder into the current workspace under
  the same name and record. It reviews by default and writes nothing; repeat it
  with the `--plan-sha256` the review shows to copy. Every file is hashed before
  and after. On macOS each file is cloned where the filesystem allows, so the
  copy uses no extra disk space; elsewhere, or where a clone fails, it is a full
  copy, and the result's `copyMethod` says which. A lens already present under
  that name is checked, not copied again: intact, nothing changes; its
  converted file missing, that file is put back (`restored: true`); its
  converted file different, the copy is refused and the file is left alone. On
  a disk that cannot take a single file into the lens folder, the copy refuses
  and says to move the lens folder aside and copy the whole lens again. A
  different lens under that name is refused. A study names a lens by its ID, so
  when the workspace holds the same lens under another name (the same reviewed
  files), the copy is still made under the origin's name. The review and the
  copy both list those other names, sorted, in `alsoHeldAs` (an empty list when
  there are none), and the message names them. Each name has its own record,
  and so its own qualifications: qualifying the lens under one name does not
  qualify it under another.
- **A site.** `steerlab-cli remote lens-push <lensID> --site <id>` (or `steerlab
  runner lens-push <lensID> --runner <url>`) sends the converted file to the
  site's lens library once, in parts that follow its layers, each checked, and
  registers the lens's record there. No part is larger than the engine's
  `lensLibrary.partMaxBytes` (256 MiB at most), so the upload limit does not
  stand in the way of a large lens. Run again after an interruption (a lost
  connection, or a failure on the engine, at any step), it sends only what is
  missing. A number in the record that is not finite (a
  qualification's value that could not be scored) is sent as null, and the
  result counts them (`recordNonFiniteNumbers`). `lens-status
  <lensID>` says what the site holds; it only reads, and when the engine does
  not answer one of its reads it says so and to run it again. It says whether
  the site has the lens's record on a runner as on a workbench, and it says
  when the site's library has an entry under the lens's hash that a newer
  SteerLab wrote, which this engine cannot use (`newerEntry`). A request
  packaged with `science package
  --lens-reference` then refers to the site's copy and leaves the converted file
  out. A lens can be referred to only when its record names its converted file
  in its own lens folder and names the lens as its folder does; otherwise
  packaging refuses (`lensNotReferable`), and the repair is to import the lens
  again or restore its record. When the site does not hold a lens that a
  request refers to, staging is refused as `lensNotInLibrary`, and both clients
  report the engine's own code, sentence, and repair. On a site that forbids
  HTTP transfer, copy the converted file beneath the
  engine's run root with the site's own tool and run `lens-push` with `--adopt
  <the path of that copy on the site>`: the site checks the copy against the
  lens's hash, keeps it, and leaves the copy as it is.

### The site's lens library

The site's lens library is the engine's own store of converted lens files,
outside any workspace: one folder per file, named by the file's SHA-256 (the
`converted.sha256` of a lens record). It holds bytes and nothing else. A lens's
record and its qualifications live in workspaces (and, once `lens-push` has
registered the lens, in the engine's own lens store), and results live in
workspaces. The workspace a lens was imported in stays its source: anything the
library holds can be sent again with `lens-push`. An entry is created once and
never edited, and no verb deletes one. A workspace's own `runs/jlens-lenses/` is
not a lens library; it is where that workspace's lenses live.

Where the library is:

- **A cluster site** keeps it in the folder its profile names as the storage
  role `constraints.storageRoots.lensLibrary`, beside the other declared roles
  (`archive`, `asset`, and `nodeCache`). The role is rendered into the site's
  environment as `STEERLAB_LENS_LIBRARY_ROOT`, and the engine passes it to the
  jobs it starts. Name permanent storage that the engine's account can write,
  with room for several gigabytes per lens (about 6.6 GB for a 27B model's).
  Do not name scratch storage that the site purges: a purged entry costs
  nothing but another `lens-push`, yet every request that refers to the lens
  is refused until it is sent again. `profile validate` says whether the
  declared folder can be written.
- **Nothing is guessed.** A profile with no `lensLibrary` role renders no
  variable, and a deployment that runs its work through Slurm, or an engine
  inside a scheduler job, then has no library. An engine never picks a folder
  on cluster storage for itself, because only the site knows that storage's
  quotas and purge rules.
- **A runner on its own machine** (`steerlab runner serve`, or any engine in the
  runner role that runs its work locally) keeps a library beside its own
  records, at `<metadata root>/lens-library`, with nothing declared.
- **A workbench engine** (the default role, which serves a researcher's own
  workspace) has none unless `STEERLAB_LENS_LIBRARY_ROOT` names a folder. Its
  lenses already live in the workspace it serves.
- **An engine from before the lens library** has none.

Both clients read the engine's `lensLibrary.available` capability first. Where
it is not true, they refuse `lens-push`, and the upload of a request that refers
to its lens, before sending anything. Package such a request without
`--lens-reference`, and it carries its lens as before.

A lens that is fitted on a site stays there. When a `jlens-fit` or
`jlens-fit-merge` job ends on an engine with a library, the engine converts the
fit as an import would and keeps those bytes (a fitting round keeps its merged
lens, not its shards). It keeps bytes only: importing the fit is still your
reviewed step, and the import is what gives the lens its record. After that
import, `lens-push` looks for the lens on the site by the hash of its converted
file. It finds the bytes the site kept, uploads nothing, and registers the
record. If your machine converted the fit to different bytes, the hashes differ
and the lens is sent like any other.

## Intended use and large files

For custom lenses, `lens.tier` declares `testing` (default) or `evidence`.
The app shows this choice during review, and both clients return it in the plan.
Ask the researcher which use they intend. Custom declarations take precedence
over the published-model table; published-source imports keep their existing
policy. Evidence intent does not establish validity: qualification for the exact
lens bytes, runtime, and layers remains a separate step before ordinary freeze.
Changing the choice creates a new reviewed import, never a rewrite of an old lens.

Workspace folders and explicitly selected source folders can be reached through
normal filesystem aliases. Relative files inside the selected source folder must
remain ordinary files, without symlink redirects or parent traversal. HTTP
imports remain confined to the served workspace. A missing workspace is refused.

Safetensors and NPZ readers materialize only the declared tensor keys. Lens
validation checks finite values in stored precision, avoiding a widened float64
matrix copy. The optional BF16 reader is needed only when a selected tensor is
BF16. SAE conversion retains its existing arithmetic. PyTorch checkpoints may
still load the whole checkpoint, and publication still materializes the selected
output tensors: this is not a streaming, constant-memory importer. Original
source copies, output tensors, and repeated integrity hashing still consume
disk space and time. Plan for at least the source copy plus converted output
on disk, in addition to any staged upload. Measure a representative large fit
before promising interactive import times.
