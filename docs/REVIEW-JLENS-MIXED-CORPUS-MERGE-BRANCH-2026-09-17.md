# Review: J-lens mixed-corpus merge branch (2026-09-17)

Branch `claude/jlens-mixed-corpus-merge` at 0a7bf79, one commit on main
01fd4e5, landed by fast-forward. Companion note:
`docs/JLENS-MIXED-CORPUS-MERGE-2026-09-17.md`.

## What prompted it

The full 828-row general-text lens for the hybrid 27B model beats the plain
residual only from layer 21 on general text and only from layer 32 on legal
prose. The lens is a linearization taken at its corpus, so the plan is to add
a few hundred rows of appellate opinion text. Refitting the 828 rows would cost
eight GPU-days; the merge already sums per-row Jacobian sums, so summing a
new domain fit into the existing sums is the same mixed-corpus fit at row
weight. Only an identity rule stood in the way.

## What the branch does

- `MergeConfig.allowMixedCorpora` (default false; interview field). Off,
  behaviour and refusal text are unchanged. On, inputs may differ only in the
  corpus digest; every other identity field still has to match.
- Rows are `(corpusSHA256, rowIndex)` throughout review and merge, so overlap
  and coverage are per corpus. Accumulation order is ascending corpus digest,
  first row, checkpoint hash, which collapses to the previous order for one
  corpus.
- A mixed merge records one contribution per corpus in the identity and a
  composite corpus digest (SHA-256 of the canonical JSON of the sorted
  contribution list); `rowIndices` is never unioned across corpora. The
  report, the artifact description, and the imported lens record carry
  `corpora`; the importer refuses a description that hides a mixture the
  report declares; assessment treats a component corpus as not held out.
- A mixed lens can be merged again only with the opt-in; a plain merge of a
  mixed lens with a same-corpus fit is refused. Fitting rounds stay
  single-corpus.
- Eleven new tests (`Server/tests/test_jlens_mixed_corpus_merge.py`): default
  refusal; opted-in merge with sums equal to the elementwise sum of the inputs
  and an order-independent composite digest; same-corpus overlap and any other
  identity difference still refused; import of a mixed lens; re-merge rules;
  round actions carry no opt-in.

## Review

I read the whole diff. Single-corpus merges produce the same identity and
bytes as before (the sort key and accumulation order reduce to the old ones),
which is the invariant that protects the existing 27B lens. The composite
digest is recomputable from the report alone and independent of input order.
The importer and provenance reader cross-check the mixture entry for entry, so
a description cannot understate what was merged.

Suites on the exact branch tree, run by me:

| suite | result |
|---|---|
| Python (`HF_HUB_OFFLINE=1 pytest`) | 6635 passed, 9 skipped, 0 failed |
| Swift (`xcodebuild test`, serial) | 4954 passed, 5 skipped, 0 failed |

Gates: `check-generated.py --audits` PASS, `public_scan.py` clean,
`git diff --check` clean, vocabulary scan clean.

## Deployment

The compiled client identity moved, so the app is rebuilt after this landing
and the cluster payload pushed before the next controller starts.
