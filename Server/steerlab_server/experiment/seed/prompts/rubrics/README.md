# Judge rubrics

Git-versioned, hash-pinned rubric files for judge evaluations. Two modes
share this directory and the same `judgeRubricFile`/`judgeRubricHash` pin:

- **Paired preference** (the default): the judge sees a blinded A/B pair —
  one baseline response, one condition response, in randomized order — and
  returns a winner or a tie. Any rubric file with no frontmatter is a paired
  rubric.
- **Per-response coding**: the judge sees ONE blinded response and returns
  declared, typed codes — no comparison, no winner. A rubric opts in by
  starting with a frontmatter block:

  ```
  ---
  mode: perResponseCoding
  field: statesAReason boolean
  field: severity integer optional
  field: stance enum(rule|balance|mixed)
  ---
  <markdown body the coder reads>
  ```

  Field types: `boolean`, `integer`, `number`, `string`, `enum(a|b|…)`
  (values contain no spaces); fields are required unless marked `optional`
  (null allowed). Malformed frontmatter REFUSES — it is never silently
  skipped. Evaluate detects the mode from the pinned rubric and writes
  `codings.jsonl` + `coding-report.json` (per-condition per-field
  aggregates, engine-computed word counts, per-field inter-judge percent
  agreement + Cohen's κ). Coding rubrics cannot drive paired-only
  machinery (sweep `judgeScore`, structured comparison prompts,
  `humanValidation`, deferred judging packets) — each refuses loudly.

## Convention

- One rubric per file: `prompts/rubrics/<name>.md`. The **entire raw file** is
  the rubric text handed to the judge (both engines strip leading/trailing
  whitespace before insertion into the judge prompt, so a trailing newline does
  not change judging behavior — but it DOES change the file's SHA-256, which is
  the pin).
- Rubrics are **immutable once referenced by a frozen study**. To iterate,
  create a new file with a bumped version suffix (`…-v2.md`) — never edit a
  rubric a frozen manifest pins. Drift in a pinned rubric surfaces as a
  freeze/verify violation, exactly like task-prompt drift.
- Manifests pin a rubric with (exact JSON keys, both engines):

  ```json
  {
    "judgeRubricFile": "prompts/rubrics/default-paired-v1.md",
    "judgeRubricHash": "<sha256 of the raw file bytes>",
    "judges": [
      {"name": "remote-1", "kind": "claude", "model": "<api-model-id>"},
      {"name": "local-1", "kind": "local", "model": "<open-weight-model-id>"}
    ]
  }
  ```

  `kind` is `"claude"` (Anthropic API judge) or `"local"` (a locally served
  open-weight judge). A judge's `name` is a LABEL, never a model id. Judged
  studies must pin **at least one judge** before freeze; **two or more**
  distinct judges are what make inter-judge agreement (percent agreement +
  Cohen's kappa) reportable, and a single-coder design freezes with an
  advisory saying none will exist. Each judgment record is stamped with the
  judge's `name`.
- Optional human anchor: `"humanValidation": {"path": …, "hash": …}` pins a
  labeled subset (same id-keyed row shape as judge outputs: `promptID`, `seed`,
  `condition`, `outcome` with `outcome` ∈ `baseline|variant|tie`); evaluation
  then reports judge-vs-human agreement per judge.

## Files

- `default-paired-v1.md` — the historical default criterion both engines fell
  back to when a manifest carried no rubric text (preserved verbatim so
  pre-versioning behavior is reproducible by pinning this file). It is a
  STARTING POINT: a one-line general-quality criterion measures general
  quality, not your study's outcome.

Write your own rubric beside it. `prompts/templates/rubrics/rubric-template.md`
is a fuller domain-neutral starting point with named criteria, and
`steerlab-cli experiment pin-rubric <name> prompts/rubrics/<file>.md` records
the pin.
