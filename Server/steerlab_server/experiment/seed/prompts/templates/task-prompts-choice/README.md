# task-prompts-choice-template.jsonl — choice-instrument task prompts

One JSON object per line:

```json
{"text": "<the task>", "options": ["optionA", "optionB"], "target": "optionA"}
```

- `text` (or `prompt`, server style) — the task the model performs. End
  it with an explicit instruction to answer with one of the options, so
  the answer tokens are well-defined.
- `options` — the closed set of answer strings the answer-token logprob /
  choice-probability instrument scores. Required on every item when the
  manifest declares `answerTokenLogprob` or `choiceProbability` in
  `outcomeInstruments`. Prefer options that tokenize to equal lengths
  (joint logprobs favor shorter options; unequal sets require the
  explicit manifest acknowledgement).
- `target` (optional) — the option treated as the "target" for shift
  metrics (e.g. the sweep's `logprobShift` objective); omitted, the
  first option is used.

Other per-item keys (`id`, `arm`, `caseID`, …) are preserved by the
editor and carried into run records.

## Where the file lives in a workspace

Anywhere under `prompts/` works — `prompts/tasks/<study>-prompts.jsonl`
is the convention. The study pins the file by hash (`taskPromptsFile` +
`taskPromptsHash`); frozen studies refuse drifted or unpinned prompts.
