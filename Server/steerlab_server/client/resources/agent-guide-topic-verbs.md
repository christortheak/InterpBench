# Verbs: the quick card

One line per task: the command, and the rule that applies. Add `--json` to
every command. When a line does not answer the question, load the topic in
its heading (`steerlab workspace guide <topic>`) or ask the verb itself
(`<family> <verb> --help`).

## Every command (`contract`)

- **Reply:** one JSON document on stdout. Read `state` first. On `refused`,
  carry out `error.repairAction`, then retry. Read `advisories[]` even when
  the command succeeded.
- **Guarded writes:** read the digest first, as `manifestFileSHA256` from
  `steerlab experiment inspect <name>`, and pass it as `--manifest-sha256`.
  A changed file refuses: read it again.
- **Review, then write:** a write takes the digest of what was reviewed
  (rename and delete preview by default and also need `--yes`). Write only
  after the researcher agrees.
- **Never** use `--force` unless the researcher accepts what it skips, and
  never edit a frozen study or anything under `runs/`.

## Start (`workspace`, `methods`)

- Readiness and a workspace: `steerlab setup start <dir> --create`
- A finished example: `steerlab workspace init <new-dir> --demo <mlx|mps|cuda>`
- The study interview: `steerlab authoring study <intent>`, with
  `conceptStudy`, `agentComparison`, or `multiAgent`. Discuss its questions
  with the researcher before writing anything.
- Methods: `steerlab science list --brief`, then `steerlab science guide <method>`.
- Existing studies: `steerlab experiment list`

## Author a draft (`lifecycle`, `settings`)

- Create: `steerlab experiment create <name> --model <id> --revision <commit>`
- Concept examples: `steerlab concept import <concept> --file <path> --side
  positive` (and `negative`). `steerlab authoring prompt <kind>` drafts them
  for review.
- Attach concepts: `steerlab experiment attach <name> <concept>…` pins each
  file's hash.
- Task prompts: `steerlab experiment import-prompts <name> --file <jsonl>
  --manifest-sha256 <digest>`
- Arms: `steerlab experiment declare-condition <name> <condition>
  --alpha-units norm --slots <concept>:<layer>:<alpha>`, and `--baseline` for
  the control arm.
- Generation and instruments: `steerlab experiment set-protocol <name> --set
  <key>=<json>`. The keys are in the `settings` topic.
- Primary outcome: `steerlab experiment set-primary-outcome <name>
  <outcome>`. Every summary then leads with it.

## Judging (`evaluation`)

- Rubric and judges: `steerlab experiment pin-rubric <name>
  <prompts/rubrics/file.md> --judges <judge>:<kind>[:<model>]
  --manifest-sha256 <digest>`
- A single judge is allowed; the report then carries no agreement statistics.
  A local judge on another model needs `--judge-pin <judge>=<revision>:<dtype>`.

## Derive and check (`lifecycle`, `sweep`): loads a model

- Validate a draft on a runner: `steerlab run <name> --runner <url> --verb
  validate` (also `--verb extract`). `--force` is not the route.
- Sweep a frozen study: `steerlab run <name> --runner <url> --verb sweep`

## Freeze (`freeze`)

- Check every pin: `steerlab experiment verify <name>`. Then `steerlab
  experiment freeze <name>`: one-way, and a refusal names the gate and its
  repair.
- Change a frozen study: `steerlab experiment duplicate <name> <new-name>`. The
  copy is a draft.

## Run (`lifecycle`, `remote`)

- On a runner: `steerlab run <name> --runner <url>` packages, submits, waits,
  and imports. A timeout detaches and never cancels.
- `submitOutcomeUnknown` means look with `steerlab runner jobs --runner
  <url>`. Never submit again blindly.
- A local runner, on macOS or Linux: `steerlab runner serve`

## Results (`evaluation`)

- Effect sizes and judging: `steerlab run <name> --runner <url> --verb
  analyze`, and `--verb evaluate`.
- Lead a summary with `result.headline`, and say how it was chosen.
- Tables, transcripts, and a methods summary: `steerlab results export
  <study>`. One page to open or send: `steerlab results report <study>`.

## Reuse (`templates`, `assembly`, `panels`)

- Templates: `steerlab design list`, then `steerlab design instantiate <name>
  --casting <file.json> --file-sha256 <digest>`. Save one with `steerlab design
  save <study> --manifest-sha256 <digest>`.
- Agents: `steerlab agent list`, `steerlab agent inspect <path>`, then `steerlab
  experiment attach-agent <study> --artifact <path> --artifact-sha256 <digest>
  --manifest-sha256 <digest>`.
- Study packs: `steerlab pack preview <file>`, then `steerlab pack apply <file>
  --review-sha256 <digest>`.
- Panels: `steerlab panel check <file>`, then `steerlab panel compile` (the
  `panels` topic).

## Housekeeping (`assembly`, `models`)

- Rename or delete a draft: `steerlab experiment rename <name> <new-name>` or
  `steerlab experiment delete <name>`. Each previews; repeat with
  `--manifest-sha256 <digest> --yes`. A delete moves the study into
  `experiments/.trash-<time>/`.
- Custom code: `steerlab experiment acknowledge-custom-code <name>` shows it.
  Only the researcher decides to trust it, with `--sha256`.
- Models on a runner: `steerlab model plan <modelID> --runner <url>`, then
  `steerlab model install <modelID> --plan-sha256 <digest> --runner <url>`.
