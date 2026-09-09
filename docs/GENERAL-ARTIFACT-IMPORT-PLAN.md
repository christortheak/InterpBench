# General instrument artifact imports

Base: `95ee982`. Branch: `codex/general-artifact-importers`.

The researcher supplies an existing fitted lens or SAE decoder and describes
its model, tensor layout, and layer mapping. Import creates a reusable instrument
or one steering vector; it does not train a model, fit a lens, train an SAE, or
establish that a feature controls the behavior suggested by its label.

## Contracts before implementation

- Inputs are a JSON description plus the original tensor file. Optional source
  configuration is retained as evidence. File hashes are captured during review
  and checked again before publication. A local file is identified by its bytes;
  repository provenance is optional and never invented. Model revisions that
  are absent remain unknown. Known incompatible identities require correction.
- A J-lens maps a source-layer residual to the final block's pre-normalization
  residual: `J_l @ h`. Matrices must be square, finite, dimension-matched, and
  mapped to explicit source layers below the final target. Sparse source-layer
  coverage is valid when declared. Multiple fits must coexist without replacing
  earlier lenses or inheriting their qualification records. Dtype is preserved.
- An SAE decoder import selects one explicitly identified feature from a decoder
  matrix, with the feature axis stated. Only residual-post directions are
  represented by this vector path. An encoder, threshold, and full latent SAE
  operation are distinct capabilities; an imported decoder does not supply them.
  The existing measured residual-norm convention supplies vector scaling. A
  calibration donor supplies model geometry and scale, not a semantic label.
- Original bytes, effective metadata, and the conversion description accompany
  the result. Publication uses a fresh destination. Existing scientific records
  and published-source imports remain unchanged.
- Authors can prepare the description with their coding agent from supplied
  artifact documentation. Missing scientific metadata is asked about, not guessed.
  The dialogs explain these choices, offer file selection, and show the reviewed
  model, layer coverage, source, output location, and limitations.
- Shared Python owners serve both CLIs, HTTP, and the app. CPU import does not
  require loading model weights. Safetensors is the portable interchange format;
  tensor-only PyTorch checkpoints require the optional PyTorch reader and never
  execute arbitrary pickle objects. Known format adapters must state their
  accepted layout rather than promise every artifact format.

## Verification and landing

Exercise independently calculable matrix/vector fixtures, invalid geometry,
non-finite values, explicit layer mapping, model mismatch, stale input bytes,
unknown revisions, repeated imports, and source retention. Prove imported lenses
load through the existing layer reader and imported vectors through the existing
vector library. Check all declared client and API entry points, shared resources,
identity generation, both full suites, and the complete diff. Live cluster and
scientific qualification remain separately recorded acceptance work. Integration
is by the maintainer's agents through the researcher.
