# Review: codex/jlens-assessment-readout (J3 matched readout assessment)

Date: 2026-09-13. Reviewed at branch head 25df56b (two commits over main
4a87703, fast-forwardable). Handoff: `docs/JLENS-J3-IMPLEMENTATION-HANDOFF-2026-09-13.md`.
Motivating work order: item 2 of `docs/JLENS-NUMERICS-PROGRAM-RESULTS-AND-HANDOFF-2026-09-13.md`
§6 and J3 of the agents' closure handoff.

## 1. What the branch does

`jlens-fit-assess` gains, at every assessed source layer and on the same staged
positions as the two lenses: an identity-transport (plain residual, logit-lens)
readout against the native final distribution; a float64 matrix comparison of
the two loaded float32 transport matrices (reference-denominated relative
Frobenius, norms, max-abs, cosine, null on zero norms); and, when the new
optional `readoutDtype: float32` is chosen, a paired float32 readout (float32
inputs and parameters through the pinned adapter's own final norm, head, and
softcap via `torch.func.functional_call`) with eight further comparison groups,
including each lens's float32-vs-native readout and the final residual's own
float32-vs-native readout. Everything lands under a new report key
`readoutComparison` with its own schema version and a `precision` record
(requested mode, native head and norm dtypes, additional parameter bytes,
limitations). The historical `layers` groups and their arithmetic are
untouched. The option reaches the operation declaration, the generated
interview, both CLIs, the API's `parameters.config`, the app's request form
(a picker), and the review summary; the J-lens guide gains a section.

## 2. What I checked

- Read the full diff: 18 files, +660/−26. Engine: `jlens_assessment.py`
  (+37/−?) and the new `jlens_assessment_readout.py` (142 lines); tests
  (+208 new, 2 lines adjusted); operation JSON; seed guide and workflows
  (+ mirrored WorkspaceSeed); Swift review summary, sheet picker, generated
  identity and resource text, one Swift test; changelog; handoff.
- Compatibility: `to_dict` drops `readoutDtype` when absent, so historical
  effective-request bytes and hashes are unchanged; explicit `native` or
  `float32` changes identity by design. The reuse tests' byte-equality
  assertions were adjusted only to exclude the new top-level key.
- Arithmetic: the baseline calls the pinned wrapper's `unembed` on the
  float32 source residual exactly as the T1 hand diagnostic did, so the
  built-in baseline is the same measurement the program used. The float32
  path converts norm and head state without touching model parameters or
  tied embeddings (asserted on real tiny Llama and Gemma-2 models against an
  independent RMSNorm formula, offset gain and softcap included). Matrix
  statistics are float64 CPU reductions in 64-row chunks.
- Execution shape: one forward per usable row and one lens-pair placement per
  layer are preserved (asserted by the one-forward test); the extra path
  re-opens the staged activation file per layer and adds unembeds, not
  forwards. Scratch is cleaned on success and on the float32 refusal path.
- Managed chain: draft/publish through the real owner, `input_plan`
  equality with the draft review, HTTP plan, stale-hash 409, packet
  execution, export, and portable-CLI import with byte-identical report.
- Gates on the branch tree: `check-generated.py --audits` PASS (all
  scientific audits and negative controls), `public_scan.py` clean,
  `git diff --check` clean, identifying-vocabulary grep clean.
- Suites on the branch tree: see §4.

## 3. Findings

No landing fix. Notes for follow-up, none blocking:

- **N1 Client-side baseline line.** `FittingReviewSummary` appends "Includes
  a plain-residual (logit-lens) baseline" unconditionally for
  `jlens-fit-assess`. Against an engine older than this branch the line is
  untrue; keying it on a server-provided review field (as `readoutReview`
  already is) would make it exact. Mixed-version windows are short here.
- **N2 Picker tags.** The app picker's "Native readout" tag is `""` unless
  the field already holds `native`, so the app never emits an explicit
  `native` unless typed; that keeps historical request bytes by default,
  which is the intended behaviour, but the guide's "native explicitly
  selects the same arithmetic" is reachable only from the CLIs.
- **N3 Double read of staged activations.** `compare` re-opens the staged
  safetensors file per layer that `compare_row` has just read; on large
  populations this doubles activation I/O. Acceptable for assessment sizes
  used so far; fold into one pass if assessments grow.
- **N4 Adapter coupling.** `Float32Readout` and `precision` read the pinned
  jlens wrapper's private `_final_norm`, `_lm_head`, `_logit_softcap`; the
  kernel is pinned by commit and the failure is a typed refusal, so this is
  contained, but a kernel bump must re-check these names.
- **N5 Live acceptance owed** (running agents, per the handoff): a float32
  assessment on the registered 4B lenses with layers 3–6 included, memory
  headroom for the float32 vocabulary head (about 4 × vocabulary × hidden
  bytes), and a check that the built-in baseline reproduces the T1 hand
  numbers at layers 0, 8, 17, 25, 32 on the same 64 WikiText rows.

## 4. Suite results on the branch tree

Python (`Server/`, main venv, `HF_HUB_OFFLINE=1`, `pytest -q -p no:cacheprovider`):
6,588 passed, 9 skipped, 8 warnings in 230 s. Matches the handoff's count
(6,580 on main before the branch, +8 new tests).

Swift (Xcode beta, Metal toolchain 32023.920.1, serial, coverage mapping off,
test Python = main venv): TEST SUCCEEDED, 4,945 tests, 4,940 passed, 5 skipped,
0 failed. Matches the handoff.

## 5. Landing

Fast-forward of main to 25df56b, this review committed on top, app rebuilt
and installed (shipped Python sources changed, Python client identity
regenerated on the branch), engine pushed to the cluster (controller was
stopped; no running controller retained old code).
