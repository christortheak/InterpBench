# Review: `codex/general-artifact-importers` @ `b5d64b3`, landed at `e884bd4` (on main `95ee982`)

Reviewer: the maintainer's integration agent, 2026-09-09. Read against
`docs/GENERAL-ARTIFACT-IMPORT-HANDOFF.md`, `docs/GENERAL-ARTIFACT-IMPORT-PLAN.md`
and `docs/GENERAL-ARTIFACT-IMPORTS.md`. Two commits, 39 files, +1,940/−25 at
`b5d64b3`; §§1–5 review that tip. The refactoring agents then answered the
findings on the branch in `e884bd4`, which §6 reviews and which is the commit
landed. Main has not moved, so a fast-forward is available.

## 1. Verdict

**Landable by fast-forward as it stands.** The slice does what the previous
review asked for and no more: a researcher's own fitted J-lens or one SAE
decoder feature can be described in a small JSON file, staged to a Python
workbench, reviewed as a hashed plan, and published into the existing lens
or vector library under a fresh identity, with the original bytes retained
and unknown provenance kept unknown. The owners are offline, the routes are
workbench-only and privileged, uploads stream to disk and are create-only,
and the plan hash binds every input byte between review and publication. The
one design gap worth a decision (N1) is that a custom lens is always
testing tier and therefore can never enter a frozen study, which the
researcher who fitted it on their own evidence model will not expect.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry | `main` (`95ee982`) is an ancestor of `b5d64b3` |
| Unified gates and audits (`check-generated.py --audits`) | every generator matches and every audit passes; the two catalog rows changed additively, so the registration audit passes at its historical base |
| Public scan, whitespace, vocabulary in the diff and both commit messages | clean |
| Request shapes | the Swift client's stage, plan and import calls match the routes' required bodies and headers; the workspace-action adapter carries `descriptionFile` and `planSHA256` exactly as the Python owner expects |
| Route authority | the three routes are workbench-only in the census and sit under a privileged prefix, so a runner deployment and an unauthenticated caller both refuse |
| Lightweight client can read safetensors and NPZ without torch | `numpy` and `safetensors` are base dependencies and pinned in the client runtime lock |
| The cluster's workspace path | contains no symbolic links, so the importer's ancestor-symlink refusal does not bite there |
| Full Python suite on the tip (main venv, cwd = worktree/Server) | 6,243 passed, 9 skipped |
| Full serial Xcode beta suite on the tip (`TEST_RUNNER_STEERLAB_TEST_PYTHON` set, external derived data, coverage mapping off) | `TEST SUCCEEDED`: 290 SteeringKit + 4,612 ExperimentKit |

## 3. What the branch does

**Owners.** `artifact_sources.py` admits a description (schema 1, `jlens` or
`sae-decoder`, model identity, optional 40-hex revision or null, hidden size
and layer count, relative tensor and optional config paths with no traversal
or symlinks), hashes files in 1 MiB chunks, and reads tensors from
safetensors, numeric NPZ, or tensor-only PyTorch checkpoints with
`weights_only=True`; BF16 needs torch and is never silently widened.
`artifact_imports.py` produces a read-only plan whose hash covers the
description, tensors, config and, for SAE, the calibration donor pair; it
hashes before and after inspection and re-reads the description so the plan
describes the bytes it pins. Publication re-captures every input into a
staged directory, re-validates the copies, converts, writes a receipt and
publishes atomically to a fresh `custom-lens-*` or `sae-import-*` directory.

**Lens conversion.** Explicit source-layer to tensor-key mapping, target
must be the final block, every matrix square and finite, checkpoint metadata
(`d_model`, `source_layers`, `n_prompts`) must agree with the description,
dtype preserved on save, record written through the existing `JLensRecord`
schema with `tierSource: custom-artifact`. A partial lens is allowed and the
plan says it supports readouts only, since token-vector derivation still
needs every layer before the final block.

**SAE conversion.** One residual-post decoder feature selected by explicit
`rows`, `columns` or `vector` orientation, scaled in float32 to the measured
residual norm of a Python-engine calibration vector for the same model
geometry; the donor's norm convention, rendering and corpus identity are
retained and its label is not. The result keeps the `gemmaScopeSAE` wire
method so attachment and injection read it, records
`gemmascopeSource.importPath: custom-sae-decoder` with the source hashes, and
the app label becomes "SAE feature (decoder row)".

**Routes and clients.** A streaming stage endpoint with `X-Content-SHA256`,
a per-deployment size cap (default 4 GiB, ceiling 32 GiB) and the transfer
policy check; plan and import endpoints; the import runs as a durable job
after a cheap admission under the workspace submission lock. Both clients
gain `science artifact-plan` and `artifact-import`; the Mac dialogs gain
"Import my own … files" with a copyable agent brief that carries the
packaged method guide. J-lens derivation now refuses a lens fitted on
another model or a known other revision, and custom-derived vectors carry
`custom-jacobian-lens` as their source rather than a Neuronpedia claim.

## 4. Findings

**N1 — a custom lens is always testing tier, so it can never be frozen
into a study.** `publish_lens` hard-codes `tier='testing'`, and the freeze
gate refuses a `jlensReadout` whose lens is not evidence tier. The
uncurated published-import path added last slice lets the researcher declare
the tier; the custom path does not, and the description schema has no field
for it. A researcher who fitted a lens on their own evidence model can
import it, qualify it, and still be refused at freeze with a repair that
names a verb (`jlens import --tier`) that does not apply to their artifact.
Recommend accepting a declared tier in the description or on
`artifact-import`, stamped with `tierSource: custom-artifact` as now, and
saying in the plan that the declaration is intent, not qualification.

**N2 — existing published lenses gain a new refusal at derivation.** The
fit-revision check in `derive_direction` applies to every lens record with a
known fit revision, not only custom ones. A published lens whose record
carries a fit revision, derived against a cached model at a different
commit, now refuses where it used to proceed. Scientifically right, and the
message names the repair; worth a changelog phrase so nobody reads it as a
regression.

**N3 — the ancestor-symlink rule refuses whole workspaces on some hosts.**
`artifact_sources.ordinary` requires every ancestor of the workspace root to
be a real directory. A site whose home or scratch root is a symlink refuses
every import with a message about source files. The cluster in use has none,
and macOS temporary paths are normalized, so nothing breaks today; resolving
the root once and checking containment of the sources against it would keep
the safety property without the false refusal.

**N4 — memory and time on large lenses.** The safetensors reader loads
every tensor of the file into memory, and each selected matrix is widened to
float64 for its finiteness check, so a 30-layer 4096-wide lens costs about
1 GiB resident plus a transient 134 MiB per layer, and each file is hashed
twice per plan. The guide says imports can need substantial memory; fine,
and the numbers are for the record.

**N5 — items the handoff itself leaves open, restated:** no managed return
or cleanup for staged sources (`.steerlab/artifact-inputs/` grows until the
researcher deletes), original files stay until verified, an interrupted
stage can leave partial temporaries, and the live acceptance walk in the
handoff's final section is still ahead, including the SAE layout and the
uncurated path against a real workbench.

## 5. Landing shape as proposed at `b5d64b3`

Fast-forward, then this review on its own commit. Suite results on that tip:

- Python: 6,243 passed, 9 skipped, 8 warnings (matching the handoff's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,612 ExperimentKit tests.

## 6. The follow-up, `e884bd4`, and what actually landed

One commit, 21 files, +462/−46, read in full.

- **N1.** The description gains optional `lens.tier`, `testing` by default or
  `evidence` for intended study use; it is part of the plan hash, shown in
  the dialog's review and the library badge, and stamped on the record with
  `tierSource: custom-artifact`. The shared tier resolver now lets a custom
  record carry its own declaration even for a model in the published table,
  which corrects a precedence the first slice got wrong; published-source
  imports keep the table policy. Freeze refuses a testing-tier custom lens
  with a repair that names the artifact-plan and artifact-import workflow,
  and a new freeze-pin test proves an evidence-tier custom lens still needs a
  passing qualification bound to the exact runtime and layers. Swift decodes
  and re-encodes both tier fields and mirrors the precedence.
- **N2.** The changelog now says the fit model and revision checks apply to
  existing published lenses too.
- **N3.** The workspace root and the description's own folder are resolved
  once, so home or scratch aliases work; components below those anchors
  still refuse symlinks, HTTP references still stay inside the served
  workspace, and a test covers an alias and a nested redirect.
- **N4.** Readers load only the declared tensor keys from safetensors and
  NPZ, lens validation checks finiteness and shape in stored precision with
  no float64 copy, and an unselected BF16 tensor no longer requires torch.
  SAE conversion arithmetic is unchanged. Their synthetic measurement shows
  lower peak memory; it is not a large-fit acceptance result.
- **N5.** Unchanged and still open.

Verified on `e884bd4` in a clean worktree: the unified gates and audits, the
public scan, whitespace, vocabulary in the diff and the commit message, the
full Python suite and the full serial Xcode beta suite (counts below). The
branch changed shipped Python and the compiled identity, so the app and its
Python payload are rebuilt together before the app is used with this main;
the engine on the cluster is unchanged until the release-stage deploy.

Suite results on `e884bd4`:

- Python: 6,263 passed, 9 skipped, 8 warnings (matching the follow-up's claim).
- Swift: `TEST SUCCEEDED`, 290 SteeringKit and 4,612 ExperimentKit tests.
