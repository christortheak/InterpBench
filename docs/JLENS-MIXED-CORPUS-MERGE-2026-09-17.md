# Mixed-corpus merge — summing fits from different corpora (2026-09-17)

## The request

A completed `jlens-fit-merge` of a general-text corpus (hundreds of rows,
already merged from shards) should be reused together with a new, smaller fit
of domain text, producing one lens without refitting the general rows.
`jlens-fit-merge` refused: `reviewed()` compares every input's
`numerical(identity)`, and `jlens_fit_identity.material` includes
`corpusSHA256`, so two corpora answered "Merge inputs differ in model, corpus,
estimator, layers, or numerical runtime." That refusal is an identity rule,
not a numerical limit.

## Why summing across corpora is a fit

A fitted lens is, per source layer, the mean over fitted rows of one Jacobian
per row. Every completed fit keeps the raw per-layer sums in
`checkpoint/sums.safetensors` with its fitted-row count `nDone`; merging adds
the sums and the counts and divides once, so the result is the mean over the
union of rows at equal row weight. Nothing in that arithmetic knows which
corpus a row came from. A merge over rows from two corpora is therefore
exactly the fit that would have been obtained from one corpus formed by
concatenating them, with the same estimator, model, layers, position policy
and numerical runtime — a mixed-corpus fit whose weight per corpus is its
fitted-row count. What has to stay identical is everything else the identity
binds: `modelID`, `revision`, `estimator`, `sourceLayers`, `targetLayer`,
`hiddenSize`, `maxSeqLen`, `skipFirst`, `dimBatch`, `fittingContract`, and
every numerical `runtime` field other than the driver hash.

## The opt-in

`MergeConfig` gains `allowMixedCorpora: bool = False`. Off, the behaviour and
the refusal text are unchanged. On, inputs may differ only in `corpusSHA256`
(and in `rowIndices`, which are per corpus); any other identity difference is
refused with the same message. The interview for `jlens-fit-merge` exposes the
field (boolean, default `false`). The fitting-round `merge-plan` /
`merge-submit` actions do not set it: a round is one corpus by construction.

## Rows are `(corpusSHA256, rowIndex)`

`source()`, `reviewed()`, `preflight()` and `merge()` identify every row by
the pair `(corpusSHA256, rowIndex)`. Two inputs from different corpora may
share row indices without overlapping; overlap within one corpus is still
refused, including a mixed lens beside a fit on one of its own corpora. The
planned-row (`expectedGlobalRows`) and partial-merge logic applies within each
corpus. Inputs are accumulated in ascending `(corpus digest, first row,
checkpoint SHA-256)` order, which reduces to the previous "first global row,
then checkpoint SHA-256" order when there is one corpus, so single-corpus
merges produce the same identity and the same bytes as before.

## Identity and provenance of a mixed lens

- `identity.corpora` — one entry per corpus, ascending by digest:
  `{corpusSHA256, promptsFitted, rowIndices}` with `rowIndices` the rows
  considered in that corpus, ascending. Present only for a mixed merge.
- `identity.corpusSHA256` — the composite digest
  `jlens_merge.composite_corpus(corpora)`: SHA-256 of the canonical JSON
  (sorted keys, separators `,` and `:`, no whitespace, `allow_nan=False`) of
  the contribution list sorted by `corpusSHA256`. It depends on which rows of
  which corpora were fitted and not on the order the inputs were listed.
- `identity.rowIndices` — absent for a mixed merge. Indices from different
  corpora are never unioned into one list.
- Report: `mixedCorpora: true`; `corpora` with per-corpus `rowIndices`,
  `rowsConsidered`, `promptsFitted`, `skippedIndices`, `globalSkippedIndices`,
  `expectedGlobalRows`, `missingGlobalRows`, and `sources` (run paths);
  the flat `globalRowIndices`, `globalSkippedIndices`, `expectedGlobalRows`
  and `missingGlobalRows` are `null`; each `sources[]` entry carries
  `corpora: [{corpusSHA256, globalRows}]` and `globalRows: null` when the
  source itself was mixed. Single-corpus merge reports gain the same
  `corpora` (one entry) and `mixedCorpora: false` and are otherwise unchanged.
- `artifact-description.json`: `lens.corpus` is `sha256:<composite>` and
  `lens.corpora` lists `{corpusSHA256, promptsFitted, rowsConsidered}`.
  The importer validates the list (at least two, distinct, ascending, counts
  summing to `promptsFitted`) and writes it to the lens record as
  `fit.corpora`; `fit_artifact_provenance` refuses a description that hides a
  mixture the report declares, or declares one the report does not.
  `JLensFitProvenance` on the Mac gains the optional `corpora` field.
- `jlens-fit-assess` accepts a mixed lens as candidate or reference; a
  held-out corpus that is one of the lens's components reports
  `heldOutStatus: sameCorpusAsFit`.

A mixed merged run is itself a valid merge input: `source()` reads its
contributions from `identity.corpora`, checks them against the checkpoint
counts and the report, and hands back per-corpus rows. Merging it further
needs `allowMixedCorpora` again (its composite digest differs from any corpus),
and `merge(merge(a, b), c)` records the same identity as `merge(a, b, c)`.
Continuing a fit from a mixed checkpoint is refused by the existing
compatibility review, since no `jlens-fit` request has that corpus digest.

## What stays refused

- Different corpora without the opt-in (unchanged message).
- Any identity difference other than the corpus with the opt-in.
- Overlap within a corpus, with or without the opt-in.
- A mixed lens together with an ancestor or an earlier merge of its rows.
- A run whose `identity.corpora`, checkpoint counts, report `corpora`, or
  composite digest disagree.

## Tests

`Server/tests/test_jlens_mixed_corpus_merge.py`, on the analytic `Tiny`
fixture: default refusal and config validation; opt-in merge whose sums equal
the elementwise sum of both inputs' sums, with the composite identity
recomputed from the report and equal across input order; single-corpus output
unchanged by the opt-in; same-corpus overlap still refused; `skipFirst`,
`dimBatch`, `maxSeqLen` and `sourceLayers` differences still refused; import
of a mixed lens with `fit.corpora`, refusal of a description that hides the
mixture, and assessment with the mixed lens on a component corpus and on a
held-out one; re-merging a mixed lens only with the opt-in, equal to the
direct three-way merge; the interview field and the rounds path leaving it
unset.
