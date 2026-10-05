# Verbs: the quick card

<!-- The first stop for "which command does this?": the command and the one rule that applies. -->
<!-- Depth belongs in the topic each heading names. Both test suites check every command and -->
<!-- flag here against the client's own verb table, and the rendered card has a size budget. -->
<!-- client: all -->

One line per task: the command, and the rule that applies. Add `--json` to
every command. When a line does not answer the question, load the topic in
its heading (`{{cli}} workspace guide <topic>`) or ask the verb itself
(`<family> <verb> --help`).

## Every command (`contract`)

- **Reply:** one JSON document on stdout. Read `state` first. On `refused`,
  carry out `error.repairAction`, then retry. Read `advisories[]` even when
  the command succeeded.
<!-- client: mac -->
- **Guarded writes:** read the digest first, as `manifestFileSHA256` from
  `steerlab-cli experiment manifest <name>`, and pass it as
  `--manifest-sha256`. A changed file refuses: read it again.
<!-- client: python -->
- **Guarded writes:** read the digest first, as `manifestFileSHA256` from
  `steerlab experiment inspect <name>`, and pass it as `--manifest-sha256`.
  A changed file refuses: read it again.
<!-- client: all -->
- **Review, then write:** a write takes the digest of what was reviewed
  (rename and delete preview by default and also need `--yes`). Write only
  after the researcher agrees.
- **Never** use `--force` unless the researcher accepts what it skips, and
  never edit a frozen study or anything under `runs/`.

## Start (`workspace`, `methods`)

- Readiness and a workspace: `{{cli}} setup start <dir> --create`
- A finished example: `{{cli}} workspace init <new-dir> --demo <mlx|mps|cuda>`
- The study interview: `{{cli}} authoring study <intent>`, with
  `conceptStudy`, `agentComparison`, or `multiAgent`. Discuss its questions
  with the researcher before writing anything.
- Methods: `{{cli}} science list --brief`, then `{{cli}} science guide <method>`.
- Existing studies: `{{cli}} experiment list`

## Author a draft (`lifecycle`, `settings`)

<!-- client: mac -->
- Create: `steerlab-cli experiment create <name> --model <id> --revision <commit>`
- Concept examples: write `prompts/concepts/<concept>/positive.jsonl` and
  `negative.jsonl` (the `workspace` topic). `steerlab-cli authoring prompt
  <kind>` drafts them for review.
- Attach concepts: `steerlab-cli experiment attach <name> <concept>…` pins
  each file's hash.
- Task prompts: `steerlab-cli experiment pin-prompts <name> <prompts/…/file.jsonl>`
- Arms: `steerlab-cli experiment declare-condition <name> <condition>
  --alpha-units norm --slots <concept>:<layer>:<alpha>`, and `--baseline` for
  the control arm.
- Generation: `steerlab-cli experiment set-sampling <name> --max-tokens <n>`.
  Instruments: `steerlab-cli experiment set-instruments <name> <instrument>[,…]`
- Primary outcome: `steerlab-cli experiment set-primary-outcome <name>
  <outcome>`. Every summary then leads with it.
- Readiness: `steerlab-cli data check <name>`
<!-- client: python -->
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
<!-- client: all -->

## Judging (`evaluation`)

<!-- client: mac -->
- Rubric and judges: `steerlab-cli experiment pin-rubric <name>
  <prompts/rubrics/file.md> --judges <judge>:<kind>[:<model>]`
<!-- client: python -->
- Rubric and judges: `steerlab experiment pin-rubric <name>
  <prompts/rubrics/file.md> --judges <judge>:<kind>[:<model>]
  --manifest-sha256 <digest>`
<!-- client: all -->
- A single judge is allowed; the report then carries no agreement statistics.
  A local judge on another model needs `--judge-pin <judge>=<revision>:<dtype>`.

## Derive and check (`lifecycle`, `sweep`): loads a model

<!-- client: mac -->
- Extract, then validate on held-out examples: `steerlab-cli experiment
  extract <name>`, then `steerlab-cli experiment validate <name>`
- Choose a layer and strength: `steerlab-cli experiment set-sweep-grid <name>
  --layers <L1,L2,…> --alphas <a1,a2,…>`, `steerlab-cli experiment sweep
  <name>`, then `steerlab-cli experiment promote <name> <concept>`
<!-- client: python -->
- Validate a draft on a runner: `steerlab run <name> --runner <url> --verb
  validate` (also `--verb extract`). `--force` is not the route.
- Sweep a frozen study: `steerlab run <name> --runner <url> --verb sweep`
<!-- client: all -->

## Freeze (`freeze`)

- Check every pin: `{{cli}} experiment verify <name>`. Then `{{cli}}
  experiment freeze <name>`: one-way, and a refusal names the gate and its
  repair.
- Change a frozen study: `{{cli}} experiment duplicate <name> <new-name>`. The
  copy is a draft.

## Run (`lifecycle`, `remote`)

<!-- client: mac -->
- On this Mac: `steerlab-cli experiment run <name>` writes an immutable run
  under `runs/`.
- On a cluster: `steerlab-cli remote package <name> --site <id>`,
  `steerlab-cli remote upload <bundle> --site <id>`, then `steerlab-cli remote
  submit-bundle <server-path> --site <id> --verb run` with the path `upload`
  printed. Watch with `steerlab-cli remote
  jobs --site <id>`. Bring evidence home with `steerlab-cli cluster import
  --site <id>`.
- `submit-bundle` is not idempotent: look with `remote jobs` before
  submitting again.
<!-- client: python -->
- On a runner: `steerlab run <name> --runner <url>` packages, submits, waits,
  and imports. A timeout detaches and never cancels.
- `submitOutcomeUnknown` means look with `steerlab runner jobs --runner
  <url>`. Never submit again blindly.
- A local runner, on macOS or Linux: `steerlab runner serve`
<!-- client: all -->

## Results (`evaluation`)

<!-- client: mac -->
- Effect sizes: `steerlab-cli experiment analyze <name>`. Judging:
  `steerlab-cli experiment evaluate <name>`.
<!-- client: python -->
- Effect sizes and judging: `steerlab run <name> --runner <url> --verb
  analyze`, and `--verb evaluate`.
<!-- client: all -->
- Lead a summary with `result.headline`, and say how it was chosen.
- Tables, transcripts, and a methods summary: `{{cli}} results export
  <study>`. One page to open or send: `{{cli}} results report <study>`.

## Reuse (`templates`, `assembly`, `panels`)

- Templates: `{{cli}} design list`, then `{{cli}} design instantiate <name>
  --casting <file.json> --file-sha256 <digest>`. Save one with `{{cli}} design
  save <study> --manifest-sha256 <digest>`.
- Agents: `{{cli}} agent list`, `{{cli}} agent inspect <path>`, then `{{cli}}
  experiment attach-agent <study> --artifact <path> --artifact-sha256 <digest>
  --manifest-sha256 <digest>`.
- Study packs: `{{cli}} pack preview <file>`, then `{{cli}} pack apply <file>
  --review-sha256 <digest>`.
- Panels: `{{cli}} panel check <file>`, then `{{cli}} panel compile` (the
  `panels` topic).

## Housekeeping (`assembly`, `models`)

- Rename or delete a draft: `{{cli}} experiment rename <name> <new-name>` or
  `{{cli}} experiment delete <name>`. Each previews; repeat with
  `--manifest-sha256 <digest> --yes`. A delete moves the study into
  `experiments/.trash-<time>/`.
- Custom code: `{{cli}} experiment acknowledge-custom-code <name>` shows it.
  Only the researcher decides to trust it, with `--sha256`.
<!-- client: mac -->
- Models: `steerlab-cli model plan <modelID>`, then `steerlab-cli model install
  <modelID>`.
<!-- client: python -->
- Models on a runner: `steerlab model plan <modelID> --runner <url>`, then
  `steerlab model install <modelID> --plan-sha256 <digest> --runner <url>`.
