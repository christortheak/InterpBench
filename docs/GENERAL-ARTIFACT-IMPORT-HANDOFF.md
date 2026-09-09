# General artifact import: review and integration handoff

Branch: `codex/general-artifact-importers`.
Base: `95ee982`, the landed researcher-experience work and its review.
Worktree: `/private/tmp/interpbench-general-artifact-importers`.

## Researcher experience

The J-lens and SAE dialogs now offer **Import my own … files**. A researcher
chooses a small import description prepared from their existing artifact and its
documentation, optionally with a coding agent's help. The dialog supplies that
agent with the packaged method guide and an exact example. It identifies the
model and files, explains the destination and transfer, reviews the conversion,
and publishes only after the researcher selects the import action. It starts no
training, dataset generation, or model download.

Custom models are not restricted to the curated published-source list. Imported
lenses join the existing model-aware library. Imported SAE decoder features join
the existing steering-vector library. Importing an artifact does not add model
architecture support to the execution engine or establish behavioral validity.
The original published-source import paths remain available.

See [the user and agent guide](GENERAL-ARTIFACT-IMPORTS.md) for exact JSON,
formats, commands, API routes, retained files, and current limits. The
[pre-implementation plan](GENERAL-ARTIFACT-IMPORT-PLAN.md) states the contracts.

## Implementation map

- `artifact_sources.py` owns description admission, source hashing, explicit
  tensor readers, and preservation of lens precision. PyTorch loading is
  tensor-only; ordinary safetensors imports work without importing torch.
- `artifact_imports.py` owns read-only plans, reviewed input capture, lens and
  SAE conversion, source receipts, and atomic publication to fresh directories.
- `artifact_import_routes.py` exposes workbench-only staging, plan, and import.
  Uploads stream to disk, obey the deployment's transfer policy, verify hashes,
  respect the configured upload-size limit, and never replace a source file. Import registers a durable job under the
  shared workspace submission lock. Numerical work does not hold that lock.
- Both authoring CLIs expose `science artifact-plan` and `artifact-import`.
  Existing science-call adapters discover the plan/import HTTP actions from the
  operation specifications. The Mac HTTP/process adapters call the same owner.
- `ArtifactImport.swift` handles Mac file selection and request types;
  `ArtifactImportButton.swift` presents the guided dialog. Bulk uploads use
  URLSession's file upload API rather than a multi-gigabyte Data buffer.
- Method-guide and catalog changes originate in the maintained sources. The
  unified generator updates packaged copies, compiled Swift resources, and
  Python payload identity. The built helper generates the Swift CLI reference.

## Scientific and provenance boundaries to review

A lens declares source layers explicitly and maps to the final pre-normalization
block with `J_l @ h`. Import preserves dtype and never manufactures missing
layers. Partial lenses are useful for readouts at their fitted layers; the
existing full-depth token-vector guard remains, and the import review explains
that limitation. Token-vector derivation checks model identity and any known
fit revision. Custom-derived vectors no longer claim Neuronpedia provenance.

An SAE import selects one residual-post decoder feature, explicitly identifying
rows, columns, or a previously exported single vector. It uses calibration from
an existing Python vector for the same declared model and geometry, with matching
known fit revision. It retains the donor's norm convention, rendering, and corpus
identity, while recording the artifact's own fit revision as unknown when absent.
It imports no encoder, threshold, complete latent SAE, or non-residual hook.

The `gemmaScopeSAE` wire-method spelling remains for the existing generic vector
attachment and injection contract. It is labeled “SAE feature (decoder row)” in
the app. The custom source is explicitly recorded in `gemmascopeSource`; the
historical method name does not assert a Gemma Scope origin. No manifest schema
or old artifact bytes change, and no qualification transfers from another fit.
Source declarations are recorded, not independently authenticated against a
remote repository. The JSON description is the explicit conversion contract;
optional original configuration is retained without interpreting arbitrary
third-party configuration formats.

## Verification

Verified on the final implementation (2026-09-09), with the suites run serially:

| Check | Result |
| --- | --- |
| Full Python suite | 6,243 passed, 9 skipped, 8 warnings |
| Full Xcode beta suite | TEST SUCCEEDED: 290 SteeringKit + 4,612 ExperimentKit tests |
| Unified generated-resource and source-identity check | Passed |
| CLI reference checked with the branch-built helper | Passed |
| Established AST audits and negative controls | Passed; existing baselines retained |
| Swift bridge gates, normal and release | Passed |
| Complete diff read and whitespace check | Completed; clean |

Logs on the build host: `/private/tmp/interpbench-artifact-python-full.log`,
`/private/tmp/interpbench-artifact-swift-full.log`, and
`/private/tmp/interpbench-artifact-final-gates.log`.
Xcode used `/Applications/Xcode-beta.app/Contents/Developer`,
`TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1`, external derived data at
`/private/tmp/interpbench-artifact-build`, `CLANG_COVERAGE_MAPPING=NO`, serial
execution, and `TEST_RUNNER_STEERLAB_TEST_PYTHON` set to the existing test venv.
No app was installed or substituted for the researcher's running app.

Fixtures independently exercise nonsymmetric Jacobians and their transposes,
row/column/single-feature SAE layouts, measured scaling, sparse coverage, known
and unknown revisions, dtype preservation, duplicate imports, stale input bytes,
source copying, corrupt containers, CLI envelopes, HTTP authority, and transfer
policy. The Swift parity test invokes the actual portable owner and decodes its
new lens record through the app's existing model.

No existing numerical bodies are claimed as mechanical moves. The established
AST audits still run against their existing baselines; no baseline was advanced.
The J-lens derive change is an explicit identity/provenance correction, covered
by an independently calculated direction fixture.

## Live acceptance after review

1. Build the app and Python payload from the same reviewed tip. Use a disposable
   workspace and a Python workbench. Leave existing studies and artifacts alone.
2. Open both import dialogs, copy the agent instructions, and prepare a small
   valid description from known source files. Verify the padding, readable labels,
   destination, optional technical details, and plain status text.
3. Stage and import a lens; select its model and fresh lens ID in the library.
   For a full-depth fixture, derive a token vector with its actual model. For a
   partial lens, verify that readouts remain available at fitted layers and the
   full-depth vector limitation is explained.
4. Import an SAE feature using calibration already measured for that model.
   Confirm it appears in Data and Playground, retains its feature and source
   identity, and can be attached through the existing vector path.
5. Change the source after planning and verify the import requests a new review.
   Repeat an unchanged import and verify that it receives a new destination.
6. On a deployment requiring external transfer, verify that the app/API explain
   that route and do not upload. Stage through the site's existing transfer tools
   before using plan/import. Check large-file disk and memory behavior on the
   actual source format intended for research.

Automated fixtures are not live scientific qualification or full GUI acceptance.
Managed return and automatic cleanup for this new artifact class are not added:
remote workbench files use the existing workspace transfer workflow, and original
files remain until the researcher verifies the local copies. Interrupted staging
can leave source files under `.steerlab/artifact-inputs/`; no automatic deletion is
claimed. Large imports can require substantial CPU memory as well as disk space.

## Landing

The maintainer's coding/audit agents review the complete diff and verification
results through the researcher, then decide integration. This work does not merge
main or replace the installed app. Check that the intended main is still an
ancestor; integrate any intervening changes and rerun affected checks before
landing. Rebuild the installed app and its matching Python payload afterward.
