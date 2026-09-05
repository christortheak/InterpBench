# Authoring prompt: reader pairs for `{{concept}}` ({{shape}})

You are writing the dataset a **reader** for `{{concept}}` is fitted on. A
reader is a measurement instrument, not a steering vector: it is a direction
found by contrasting activations, and it is then used to SCORE text. Its
dataset therefore decides what the score means, and a confound here does not
push a run in a direction — it silently redefines the quantity being reported.

Every row names the same concept and the same task template. The template is
`{{templateID}}`, and every row must declare it: a row fitted under one
template and labelled with another is refused, because the template is half of
what produced the activation.

## The two poles

**Positive** — {{positive}}

**Negative** — {{negative}}

## The shape: {{shape}}

{{shapeBlock}}

The two shapes may not be mixed in one file. Which one you are writing decides
what the contrast is *carried by*, and a file containing both would fit a
direction that is partly one and partly the other.

## The file

**`{{path}}`** — {{count}} rows, one JSON object per line. Every row carries
`concept` and `templateID`; all rows share one concept.

| Field | Required | Rule |
|---|---|---|
| `concept` | yes, every row | Exactly `{{concept}}`. A file with two concepts is refused. |
| `templateID` | yes, every row | Exactly `{{templateID}}`. |
| `split` | no | Lower-cased; defaults to `train`. **Anything other than `train` or `finalTest` is held out** — write `test` for held-out rows (`heldOut` also works). |
| `id` | no | A string; stable ids make a review diff readable. |
| `topic` | no | Free text, for your own grouping. |

Write the last {{heldOut}} rows with `"split": "test"` and the rest with
`"split": "train"`. The held-out rows are what decide the direction's SIGN: the
fit checks which orientation agrees with them, and falls back to a train-set
majority — recording that it did so — when too few held-out pairs decide. Too
few honest held-out rows is therefore not a small loss; it is a coin flip
stamped into the artifact.

`"finalTest"` is reserved for an optional final-evaluation set that no fitting
or selection step reads; leave it out unless the study reserves one.

At least 2 non-degenerate training rows are required or the fit refuses. That
is a floor, not a target: {{count}} rows with {{heldOut}} held out is the shape
this prompt is written for.

{{discipline}}

## The audit battery — compute these and report them

| # | Measure | Threshold |
|---|---|---|
| 1 | Row count, and rows with `split: "test"` | {{count}} total, {{heldOut}} held out |
| 2 | Rows whose `concept` is not `{{concept}}` | zero |
| 3 | Rows whose `templateID` is not `{{templateID}}` | zero |
| 4 | Rows mixing the two shapes | zero |
| 5 | Duplicate `id`s | zero |
| 6 | Word-count delta within each contrast | ≤ {{lengthDeltaWords}} |
| 7 | Top-20 content stems per side, as a share of rows | none above {{stemCapPercent}}% |
| 8 | Syntactic frame distribution per side | no frame above {{frameCapPercent}}%, every frame present on both sides |
| 9 | Per-contrast frame identity (same frame, mood, tense, person) | 100% |
| 10 | Loaded-affect tokens per side, excluding the poles' own vocabulary | within {{parityPercent}}% |
| 11 | Distinct domains | ≥ 5 |
| 12 | Held-out rows: domains and stems shared with the train split | listed; a held-out split that reuses the train split's domains decides the sign on the train split by another route |

{{delivery}}
