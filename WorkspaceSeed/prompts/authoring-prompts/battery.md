# Authoring prompt: capability battery

You are writing a **capability battery** — the fixed set of probes that says
whether an intervention has damaged the model's general competence. It is the
brake on the sweep: a cell that expresses the concept beautifully and fails the
battery is not selected, and the battery is the only thing standing between a
study and a setting that produced the effect by degrading the model.

That job makes the battery deliberately boring. It has nothing to do with the
concept under study, and it must be answerable by a plain model with no
context. Its rows are graded by log-probability over a closed option set, so
nothing is generated and nothing is parsed.

## The file

**`{{path}}`** — one JSON object per line. The FIRST non-empty line is a
header, and every line after it is an item.

**Header** (exactly this shape; do not add keys you were not asked for):

```json
{"batteryFormat": 2, "scoring": "choiceProbability", "promptMode": "chatAssistant", "maxTokens": {{maxTokens}}, "description": "…"}
```

`batteryFormat` must be `2`. Declaring `1` is refused: format 1 is the
headerless legacy shape, and a file that declares it changes what an existing
pinned hash means. Do **not** declare `systemPrompt` — the header's arming is
applied identically to the baseline arm and to every steered arm, which is what
makes the battery a fair comparison, and a system prompt in the header is a
frame the study never asked for.

**Items** — {{count}} of them:

```json
{"id": "…", "prompt": "…", "answer": "…", "options": ["…", "…", "…"]}
```

| Field | Required | Rule |
|---|---|---|
| `prompt` | yes | A non-empty string: one self-contained question. |
| `answer` | yes | A string, and it must be one of `options`. |
| `options` | yes | At least 2, at least {{minOptions}} to score as discriminative; all non-empty strings, no repeats. |
| `id` | no | A stable, non-empty string. Absent falls back to an ordinal, which makes a diff across two batteries unreadable. |
| `grading` | **must be absent** | Nothing is generated, so there is nothing to grade. |

## What makes a battery item usable

**One right answer, unambiguously.** A capability probe with a defensible
second answer measures the model's taste, and taste moves under steering. If
you would argue about it, cut it.

**Distractors must be wrong but plausible.** An item whose distractors are
absurd is passed by a badly damaged model and reports no damage. Three or four
options; each a real candidate.

**Options must be surface-matched.** Same grammatical form, within a token or
two of each other in length, none a prefix of another, and none normalising to
the same string as another (case and punctuation are stripped before
comparison, so "42" and "42." collide).

**No format instructions.** "Answer with the letter", "respond in one word" —
the scorer never samples, so these change the distribution and measure nothing.

**Spread the abilities.** Arithmetic, factual recall, short multi-step
reasoning, instruction following, and simple language manipulation, at roughly
equal counts. A battery that is all arithmetic reports arithmetic damage.

**Nothing about the concept under study.** The battery is a control. Any item
that touches the concept turns the brake into a second measurement of the thing
being measured.

{{discipline}}

## The audit battery — compute these and report them

| # | Measure | Threshold |
|---|---|---|
| 1 | Item count | {{count}} (at least {{minItems}}, below which the reading is noise) |
| 2 | Header present, `batteryFormat` 2, no `systemPrompt` | pass |
| 3 | Items with `grading` or with fewer than 2 options | zero |
| 4 | Items whose `answer` is not in `options` | zero |
| 5 | Items with fewer than {{minOptions}} options | zero |
| 6 | Options colliding after normalisation (case, punctuation, whitespace) | zero |
| 7 | Options that are a prefix of another option in the same item | zero |
| 8 | Max option-length ratio within an item (longest ÷ shortest, in characters) | ≤ {{optionLengthRatio}} |
| 9 | Duplicate `id`s, duplicate prompts | zero |
| 10 | Prompts containing a format instruction | zero |
| 11 | Ability categories, and items per category | ≥ 5 categories, none above 30% of items |
| 12 | Correct-answer position distribution | no position above {{frameCapPercent}}% of items |
| 13 | Items referring to the concept under study | zero |

{{delivery}}
