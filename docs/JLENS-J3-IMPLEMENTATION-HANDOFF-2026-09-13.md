# J3: matched readout assessment

Date: 2026-09-13
Branch: `codex/jlens-assessment-readout`; initial baseline `168aee8`, rebased onto
main `4a87703` after its documentation-only J1 correction landed.
Scope: J3 implementation and local acceptance. Live GPU qualification is pending.

## Researcher outcome

`jlens-fit-assess` now includes a plain-residual (identity-transport/logit-lens)
baseline at every assessed source layer. Researchers can also choose a paired
float32 readout, without refitting either lens or forwarding the corpus again.
Matrix differences and readout differences are shown separately. Neither becomes
an automatic scientific pass/fail threshold.

The app's shared request form has a readout-precision picker. The generated
interview exposes `readoutDtype` to both clients and API authoring. The execution
API accepts it in the existing managed `parameters.config`. Results and collected
evidence retain the complete JSON report through the existing viewer/import paths.

## Numerical and compatibility contract

- The absent-key default preserves old effective configs and preflight material.
  `native` explicitly selects the same arithmetic; `float32` adds comparisons.
  The updated interview has a new hash, so unpublished drafts need review again.
- The old `layers` report groups and eight-position arithmetic are unchanged.
  The historical AST audit and byte-pinned runtime comparison remain active.
- New `readoutComparison` data names its own schema version, actual observed
  tensor dtypes, readout convention, extra parameter bytes, baseline, matrix
  metrics, and optional paired float32 metrics. See the packaged J-lens guide
  for field names and interpretation.
- Transport multiplication was already float32 and remains so. Float32 readout
  uses the pinned HF adapter's actual final norm, head, and softcap with float32
  state through `torch.func.functional_call`. It does not permanently change
  model parameters, tied embeddings, or the forward activation dtype. The small
  compatibility adapter is isolated in `jlens_assessment_readout.py`.
- Both modes use the same native final-residual readout as their target. Module
  internals and backend matmul/TF32 settings still apply. A dtype choice does not
  establish exact arithmetic or native-logit parity for every architecture.
- Matrix comparisons use bounded float64 CPU reductions of the loaded float32
  transport matrices. Zero norms yield null relative/cosine values. There is no
  invented ratio of JS divergence to matrix error.
- One usable-row forward and one lens-pair placement per source layer remain.
  Extra unembedding and matrix-statistic work does add time. Float32 head/norm
  state adds roughly four bytes per parameter when conversion is needed, beyond
  the existing lens and staging budgets. Temporary activations are cleaned on
  success and ordinary failure; killed-process cleanup remains unchanged.

No fitting, continuation identity, benchmark tolerance, merge policy, stopping
rule, corpus sampling, or old artifact bytes change in this slice. J1 document
corrections are included from main; J2 direct tensor comparisons remain separate
qualification work. Retained
intermediate checkpoints and baseline-only execution are not added here.

## Local evidence

The new tests cover independent RMSNorm arithmetic on real tiny Llama and Gemma
models, Gemma's offset gain and softcap, native HF logit reconstruction, unchanged
weights/tied embeddings, self-comparison, independent baseline statistics,
zero-norm matrix metrics, unchanged old config bytes, one-forward execution,
failure cleanup, and draft/publication through the actual managed owner.

The transport test packages and stages the full registered-lens closure, plans
through HTTP, rejects a stale submission hash, executes the queued packet owner
with a small numerical model fixture, exports the result, and imports it through
the portable CLI. Collected report bytes must match exactly. This is local
integration evidence, not a live Slurm job or a production-model measurement.

Native tests check the generated field and the owner's precision/resource review
presentation. Manual app interaction and a real model run still belong to the
running agents. Final suite and audit results are recorded below before handoff.

## Running-agent acceptance

After independent review and authorized deployment, verify matching app/payload,
controller, and worker identities. A code push alone does not restart a running
controller. Use existing registered local/published Gemma lenses and pinned
held-out text. Do not refit or download a model merely to test this feature.

1. Draft a small assessment with `readoutDtype: float32`. Check that the app/CLI
   review includes additional norm/head storage, beyond the lens-pair estimate.
2. Plan and submit on the approved device with enough headroom. The nearly full
   27B batch benchmark does not establish room for a float32 vocabulary head.
3. Collect/import, open `assessment-report.json` in Results, and inspect each
   comparison mode, row/position counts, dtypes, baseline, and matrix statistics.
4. Include layers 3–6 and the previous middle-layer comparisons. Compare both
   readout modes against the same native final target and inspect the paired
   readout changes; do not treat a final-residual precision check as an error
   bound for transported residuals.
5. Record execution identities, inputs, measured memory/time, custody receipts,
   and any remaining scientific limitations. Preserve original evidence.

This acceptance needs no full production 27B fit. It also does not establish
mixed-GPU fitting equivalence or causal usefulness of transported directions.

## Verification status

- Full Python suite: **6,588 passed, 9 skipped, 8 warnings** (235.85 seconds).
- Full serial Xcode beta suite: **TEST SUCCEEDED**; 295 SteeringKit and 4,650
  ExperimentKit tests reported, including 5 skipped tests (4,940 passed overall).
- After the final interview-label refinement, 38 targeted authoring/numerical
  tests passed. Both CLI executables returned identical complete interview
  payloads and hashes, in ready envelopes.
- Generated resources and the source-built native CLI reference, all existing
  scientific AST audits and negative controls, bridge gates, public-content
  scan, and whitespace checks pass. The full change was reviewed locally;
  independent review through the user is still required before landing.
- Verification used Xcode beta and Metal toolchain
  `com.apple.dt.toolchain.Metal.32023.920.1`, dedicated scratch under
  `/private/tmp/interpbench-j3-build`, and the existing test Python environment.
  No dependencies, app installation, model download, or cluster deployment
  were added by this implementation.

Local logs: `/private/tmp/j3-python-full.log`, `/private/tmp/j3-native-full.log`,
`/private/tmp/j3-gates-final.log`, and `/private/tmp/j3-label-validation.log`.
These local fixture and suite results do not replace the running-agent acceptance
above. Rebasing onto the J1 documentation correction changed no tested code.
Main was not changed by this branch.
