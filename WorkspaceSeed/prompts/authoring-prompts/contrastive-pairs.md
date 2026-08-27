# Authoring prompt: contrastive stimuli for `{{concept}}`

You are writing the paired stimulus set a steering direction for `{{concept}}`
will be extracted from, plus the held-out probe that tests whether the
direction generalizes past its own vocabulary.

## The two poles

**Positive pole** — {{positive}}

**Negative pole** — {{negative}}

Both poles are considered positions. The negative pole is not the absence of
the positive one, not naivety, and not a strawman: it is a second stance a
thoughtful person holds. If one pole reads as the sensible view and the other
as its failure mode, the direction you extract will be a direction for
"sensible", not for `{{concept}}`.

## Pair discipline (this is the whole game)

Each pair shares ONE concrete scenario, and the two members diverge ONLY in the
pole expressed about it. Hold constant within every pair:

1. **The scenario** — same situation, same setting, same parties; ideally a
   shared opening clause, verbatim or near-verbatim.
2. **Length** — within ±{{lengthDeltaWords}} words of each other; target
   {{minWords}}–{{maxWords}} words per row.
3. **Register** — both analytical, neither academic-jargon nor casual.
4. **Emotional intensity** — both speakers equally confident and composed. Audit
   this hard: under pressure one pole drifts toward indignation and the other
   toward complacency, and that drift is a second axis riding along inside your
   corpus.
5. **Concreteness** — both grounded in the scenario's specifics, not slogans.
6. **Syntactic frame** — row *i* of both files uses the SAME final-sentence
   frame (both cleft, both "not X but Y", both gerund-subject, …), the same
   mood, the same tense, the same person.

What MAY differ, because it is constitutive rather than a confound: each pole's
own conceptual vocabulary. Do not artificially suppress it — and do not let
either pole's conceptual vocabulary appear in the other pole's file.

Vary ACROSS pairs: the domain (at least five distinct ones), the syntactic
frame, and which pole's phrasing comes more naturally to the scenario. Include
scenarios where the facts pull toward each pole and scenarios where they pull
toward neither, at roughly a third each — if every scenario of one kind reads
positive and every scenario of the other kind reads negative, you have
re-confounded the axis with the facts, and the extraction will find the facts.

## The files

**`prompts/concepts/{{concept}}/positive.jsonl`** — {{count}} rows, one JSON
object per line, exactly `{"text": "…"}`. The positive-pole member of each
pair, in pair order. No other key is read; a line without `text` is refused at
pin time.

**`prompts/concepts/{{concept}}/negative.jsonl`** — {{count}} rows, same shape,
the negative-pole member, in THE SAME pair order. Pairing is POSITIONAL: row
*i* of each file is one pair, and a reordering silently re-pairs the corpus.

**`prompts/concepts/{{concept}}/validation.jsonl`** — {{validationCount}} rows,
`{"text": "…", "expresses": true|false}`. `expresses` must be a real JSON
boolean; the string `"false"` reads as true and would invert the row.

Validation rows are **held out and vocabulary-free**. They EVOKE one pole or
the other WITHOUT using either pole's conceptual vocabulary: a person acts, or
reasons, in a way that manifests the stance, and the row is labelled `true` for
the positive pole and `false` for the negative one. Half each, shuffled. Reuse
no extraction scenario. These rows are the independent evidence that the
direction tracks the stance rather than the words — a validation row that names
the concept tests a word detector and passes it.

{{discipline}}

## The audit battery — compute these and report them

| # | Measure | Threshold |
|---|---|---|
| 1 | Max intra-pair word-count delta | ≤ {{lengthDeltaWords}} |
| 2 | Top-20 content stems per file, as a share of rows | none above {{stemCapPercent}}% |
| 3 | Final-sentence frame distribution per file | no frame above {{frameCapPercent}}% of pairs, and every frame present in one file present in the other |
| 4 | Per-pair frame identity | 100% of pairs share their frame, mood, tense and person |
| 5 | Reversal/concessive constructions ("presents as … operates as", "what looks like … is") | counts within {{parityPercent}}% across files, none one-sided within a pair |
| 6 | Persistence adverbs (systematically, routinely, continually, reliably, …) | counts within {{parityPercent}}% across files |
| 7 | Loaded-affect tokens, EXCLUDING the two poles' own domain vocabulary | counts within {{parityPercent}}% across files, and no pair loaded on one side only |
| 8 | Sentence-2 opener distribution per file | no single opener above {{frameCapPercent}}% of rows in either file |
| 9 | Vocabulary bleed: each pole's conceptual vocabulary in the other pole's file | zero, with borderline calls listed |
| 10 | Scenario valence: facts favouring the positive pole / the negative pole / neither | each group ≥ 25% of pairs |
| 11 | Validation vocabulary: either pole's conceptual vocabulary in `validation.jsonl` | zero |
| 12 | Validation label balance | {{validationCount}} rows, half `true`, shuffled |

Report each as an actual count. Measures 5, 6 and 7 are the ones deliveries
fail most often, and they fail them after fixing the ones above — the asymmetry
re-expresses itself one layer down each time. Rule 1 of the discipline is the
general form of all twelve; when your counts pass but the poles are still
readable from surface form, you have found a thirteenth measure, and you should
report it.

{{delivery}}
