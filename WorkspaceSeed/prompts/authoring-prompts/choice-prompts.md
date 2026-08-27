# Authoring prompt: choice instrument for `{{concept}}`

You are writing the **choice instrument** a sweep selects a dose on. Each row
puts a model in a situation and offers it a small closed set of answers; the
sweep reads the model's log-probability over those answers, with and without
the intervention, and the shift is what decides which layer × α cell wins.

That makes this file unusually load-bearing. It does not measure prose, it
decides a setting — and a flaw in it does not produce a bad number, it produces
a confidently wrong setting that every later run inherits.

## The decision

{{decision}}

Every row is one instance of that decision, in a fresh concrete situation. The
row must be answerable by a reader who has never heard of `{{concept}}`: the
situation carries the pull, the options carry the answer, and nothing names the
thing being measured.

## The row shape

**`{{path}}`** — {{count}} rows, one JSON object per line:

```json
{"id": "…", "prompt": "…", "options": ["…", "…"], "target": "…"}
```

| Field | Required | Rule |
|---|---|---|
| `prompt` | yes | A JSON string. The situation and the question. (`text` is accepted as an alias; use `prompt`.) |
| `options` | yes | A JSON **array of strings**, at least 2. A bare string is refused — it would be split into characters. |
| `target` | no | Must be a JSON string and must be one of `options`. Absent means `options[0]`, so declare it rather than relying on order. |
| `id` | no | A JSON string, unique across the file. Duplicates are refused: per-row log-probabilities are keyed by id, and two rows sharing one would overwrite each other. |

Every one of those is a refusal at sweep start, not a warning — the instrument
is resolved before the model loads, so a malformed row costs you nothing but a
re-edit.

## What makes a choice row measure the thing

**Options must be surface-matched.** The scorer reads the joint
log-probability of an option's tokens, which means a longer option is
penalised for being longer. Keep every option in a row within one or two tokens
of the others, in the same grammatical form (all noun phrases, or all clauses —
never one of each), and never make one option a prefix of another.

**Options must not be labels for the poles.** If one option is recognisably
"the {{concept}} answer", you are measuring whether the model can read a label.
Both options must be defensible answers to the situation as stated.

**Balance the target across the file.** The share of rows whose `target` is the
first-listed option must sit between {{balanceLowPercent}}% and
{{balanceHighPercent}}%. Outside that band, position alone predicts the answer
and the instrument scores a position bias.

**Vary the situations.** At least five distinct domains, and no situation
template reused. Vary which option is listed first independently of which is
the target.

**No format instructions in the prompt.** Do not write "Answer with A or B" or
"Respond with one word": the scorer never samples, so the instruction changes
the distribution without changing what is measured.

{{discipline}}

## The audit battery — compute these and report them

| # | Measure | Threshold |
|---|---|---|
| 1 | Row count | {{count}} |
| 2 | Duplicate `id`s | zero |
| 3 | Rows whose `target` is not in `options` | zero |
| 4 | Rows with fewer than 2 options | zero |
| 5 | First-option target share | {{balanceLowPercent}}–{{balanceHighPercent}}% |
| 6 | Max option-length ratio within a row (longest ÷ shortest, in words) | ≤ {{optionLengthRatio}} |
| 7 | Options that are a prefix of another option in the same row | zero |
| 8 | Distinct domains represented | ≥ 5 |
| 9 | Top-20 content stems across prompts, as a share of rows | none above {{stemCapPercent}}% |
| 10 | Prompts containing a format instruction ("answer with", "respond with", "reply only") | zero |
| 11 | Prompts naming `{{concept}}` or an obvious synonym | zero |
| 12 | Option-order/target independence: share of rows where the target is first, split by domain | each domain within {{parityPercent}}% of the file-wide share |

{{delivery}}
