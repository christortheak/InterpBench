# Authoring prompt: held-out probe for `{{concept}}`

You are writing the **held-out probe** — the rows that decide whether an
extracted direction for `{{concept}}` found the thing, or found the words the
extraction corpus happened to use.

These rows are never used in extraction. They are read once, by a projection
onto the extracted direction, and scored against their labels. A study whose
concepts have no scored probe cannot be frozen: the evidence gate names them.

## The two poles

**`expresses: true`** — {{positive}}

**`expresses: false`** — {{negative}}

## The one rule that matters

**Never name the thing.** A validation row must EVOKE its pole without using
that pole's conceptual vocabulary, or any obvious synonym of it. A person acts,
decides, notices, or reasons in a way that manifests the stance; a reader
recognises it; the words for it never appear.

This is not a stylistic preference. The extraction corpus is full of each
pole's vocabulary, so a direction extracted from it will partly be a detector
for that vocabulary. A validation row that reuses the vocabulary tests the
detector against itself and passes — which is exactly the failure the probe
exists to catch. A vocabulary-free probe is the only kind that can distinguish
"the model represents this" from "the model recognises these words".

Reuse **no** scenario from the extraction corpus, and no situation close enough
to one that a reader would call it the same case.

## The row shape

**`{{path}}`** — {{count}} rows, one JSON object per line:

```json
{"text": "…", "expresses": true}
```

| Field | Required | Rule |
|---|---|---|
| `text` | yes | A non-empty JSON string. |
| `expresses` | yes in practice | A real JSON **boolean**. The string `"false"` reads as true and would invert the row; the loader refuses a string rather than coercing it. A file with the key absent everywhere is treated as unlabelled, scores no accuracy, and leaves the concept on the vacuity ledger — which blocks freeze. |

Half the rows `true`, half `false`, shuffled — a probe whose labels arrive in
blocks scores the same but reads as if it were assembled rather than composed,
and any residual position effect lands on one label.

## Balance, so the score means something

The probe is scored as a classification, so anything that correlates with the
label and is not `{{concept}}` inflates the number. Match across the two
labels: length, register, domain, the presence of dialogue, the presence of
numbers, first- versus third-person, and how sympathetic the actor is. Vary all
of those ACROSS rows.

Aim for pairs of situations rather than independent rows where you can: the
same setting, once with an actor manifesting each pole. Paired situations make
measure 6 below easy to satisfy honestly and hard to fake.

{{discipline}}

## The audit battery — compute these and report them

| # | Measure | Threshold |
|---|---|---|
| 1 | Row count and label balance | {{count}} rows, within one row of an even split |
| 2 | Rows whose `expresses` is not a JSON boolean | zero |
| 3 | Either pole's conceptual vocabulary anywhere in the file | zero, with borderline calls listed |
| 4 | Scenarios shared with the extraction corpus | zero |
| 5 | Mean and max word count per label | means within {{parityPercent}}%, no max more than 1.5× the other label's |
| 6 | Distinct domains, and the label split within each | ≥ 5 domains, each carrying both labels |
| 7 | Top-20 content stems, as a share of rows, per label | none above {{stemCapPercent}}% |
| 8 | Stems that appear under one label only | listed, with a judgement on each |
| 9 | First-person rows, dialogue rows, numeric rows | each within {{parityPercent}}% across labels |
| 10 | Label run length after shuffling | no run above 3 |

Measure 8 is the one that catches a probe that will score 0.95 and mean
nothing. A stem appearing only under one label is a shortcut; report every one
you find, and say why you kept or removed it.

{{delivery}}
