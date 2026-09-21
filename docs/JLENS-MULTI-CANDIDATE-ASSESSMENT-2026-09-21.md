# J-lens assessment: several candidates and corpora in one request (2026-09-21)

Closes item 7 of `docs/JLENS-AND-P7-STATUS-HANDOFF-2026-09-21.md`, first
option. One `jlens-fit-assess` request now names one reference lens, one or
more candidate lenses, and one or more held-out corpora. Every distinct lens
travels once, activations are captured once per corpus, candidates are
compared one after another, and every candidate–corpus comparison is
identified and digested on its own.

Owner: `Server/steerlab_server/experiment/jlens_assessment.py`
(`AssessmentConfig`, `preflight`, `assess`). Managed inputs:
`managed_inputs.inventory` with the new `lenses` role. Interview:
`docs/techniques/operations/jlens-fit-assess.json` (generated twins under
`WorkspaceSeed/` and `Server/.../seed/`). Tests:
`Server/tests/test_jlens_multi_candidate_assessment.py`.

## Request shape

Two fields gain a list twin. A request names one form per axis, never both:

| axis | single form (unchanged) | list form |
| --- | --- | --- |
| candidates | `candidateLensID: "<lens ID>"` | `candidateLensIDs: ["<lens ID>", …]` |
| corpora | `corpus: {path, sha256}` | `corpora: [{path, sha256}, …]` |

The forms may be mixed across axes (one candidate on several corpora, or
several candidates on one corpus). Refusals, all typed `FitError` before any
model or lens is touched: both forms on one axis; an empty list; a candidate
listed twice; the same corpus path listed twice; the reference among the
candidates; a list entry that is not a registered lens ID or a pinned file.
The single form keeps its historical behaviour and request bytes exactly:
`to_dict()` never emits the unused form, so an existing request hashes as
before and the frozen baseline owner still parses it.

Source layers are the intersection over every selected lens when
`sourceLayers` is omitted; an explicit list must be present in every lens.
All comparisons in one request therefore share one layer set.

Interview fields `candidateLensIDs` (kind `texts`: one ID per line or
comma-separated) and `corpora` (kind `fileRefs`: one workspace-relative file
per line, each hashed and recorded as a source document) sit beside the
single fields, which are now optional. Both kinds are new to
`method_authoring.value`; the Mac authoring sheet shows them as multi-line
text and lists `fileRefs` on its Data step.

## What travels once

`managed_inputs.inventory` walks `candidateLensIDs` under the `lenses` role
and adds each lens's `lens.json`, import receipt, and converted tensor
exactly as the `lens` role does; the inventory is a set, so a lens named
once is shipped once regardless of how many comparisons use it. Corpora are
hash-verified by the generic walker and appear once each. The plan hash
still binds every lens's converted-tensor hash and every corpus hash; a
changed corpus or converted tensor refuses at planning as before.

The transport archive bound (16 GiB) is unchanged: it applies when the
closure is packaged for a remote runner, so two 27B lenses at about 6.6 GB
each travel in one archive while a request naming four is refused at
packaging with the existing transport refusal.

## Execution order and memory bound

```
for corpus in corpora (request order):
    stage activations once (two-phase capture; one TemporaryDirectory)
    for candidate in candidates (request order):
        for layer in layers:
            load [reference layer, candidate layer]      # one pair resident
            compare rows (native readout, plain-residual baseline,
                          optional float32 readout)
        finalize this comparison; nothing per-candidate stays resident
    remove this corpus's staging before the next corpus
```

Peak tensor residency is the same as for a single pair: one corpus's staged
activations on disk, one lens layer pair in memory, and (when requested) one
float32 copy of the norm/head, created once and reused. Candidates add reads,
not residency: the reference layer is re-read for every candidate, which the
per-comparison `lensLayerReads` records. The staging directory of a corpus is
released before the next corpus is captured, so temporary bytes are bounded
by the largest corpus rather than by their sum.

`preflight` and the report's plan-level `resources` reflect this: the
single-pair budget evaluated at the maximum row count over corpora, plus
`candidates`, `corpora`, `candidateStrategy`, `corpusStrategy`, and
`budgetBasis`. `rows` in the review is that maximum. The executed report adds
`capturedActivationBytesMaximum` and the total `lensLayerReads` and
`lensLayerPlacements` over all comparisons.

## Report layout and digests

One run directory, `runs/jlens-assessment-<id>/`, holds
`assessment-report.json`, `comparisons/`, and `COMPLETED`.

Every comparison entry has exactly these keys: `referenceLensID`,
`candidateLensID`, `corpus` `{path, sha256}`, `layers`, `readoutComparison`,
`rows`, `resources`, `heldOutStatus`, and `comparisonSHA256`, the SHA-256 of
the canonical JSON (sorted keys, compact separators) of the entry without
its own digest. The same bytes are written alone as
`comparisons/<candidateLensID>--<first 8 hex of corpus sha256>.json`, so one
comparison can be taken on its own and re-verified.

Single form (one candidate and one corpus through the single fields):
`schemaVersion` 1, every historical top-level key unchanged
(`config`, `runtime`, `readoutComparison`, `lenses`, `layers`, `rows`,
`resources`, `heldOutStatus`, `qualification`, `aggregation`,
`limitations`), plus `comparisons` with one entry. The frozen-baseline
equality tests exclude only that added key.

List form (either list field present): `schemaVersion` 2 with the shared
`config`, `runtime`, `lenses` (reference first, then candidates in request
order), `candidateLensIDs`, `corpora`, plan-level `resources`,
`qualification`, `aggregation`, `limitations`, and `comparisons` in
corpus-major, candidate-minor order. The per-pair top-level mirror is absent
by design: a reader of a schema 2 report takes its pairs from `comparisons`.
The version number is the signal an old reader needs; it is not a
migration.

A single pair through the list form produces a comparison entry whose
canonical bytes, digest included, equal the single form's entry for the same
candidate and corpus. `precision` in `readoutComparison` reports the float32
readout only when that corpus staged at least one row, exactly as the single
form does, so the equality holds for all-short corpora too.

`managed_methods.execute` returns `comparisonReports` (the standalone file
paths) beside the unchanged `runDirectory`, `reportPath`, and
`qualification`.

## Readers checked

- Python: `jlens_fit_review.operation_review` and
  `scientific_execution.input_plan` call `preflight`, whose single-form
  output is unchanged. The transport, custody, and import paths copy the
  report bytes without parsing them.
- Swift: no Codable model decodes the assessment report. The review summary
  (`FittingReviewSummary`) adds one line when the preflight review carries
  `comparisons`; single-form reviews render as before.
- The frozen baseline owner and the AST audit
  (`scripts/ci/audit-jlens-assessment.py`) still hold: `distances` and the
  token-chunk loop in `compare_row` are untouched.

## Out of scope

- Cross-request lens reuse (a runner holding a lens by converted-tensor hash
  so it uploads once per runner). Every request still carries its lenses;
  the second option in the handoff item stays open.
- Per-pair layer sets. Comparisons in one request share the intersection of
  source layers; request separate assessments for lenses with different
  layer coverage.
- Any qualification claim. `qualification` stays `notPerformed`.
