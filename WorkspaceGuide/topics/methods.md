# Methods and managed scientific operations

<!-- client: all -->

## Find the method before authoring its data

Both installed clients expose the same shipped references. Use
`{{cli}} science list --json`. Then `science guide <method> --json` returns the
method-specific dataset schemas, coworker author prompt and independent
review prompt; `science operation <operation> --json` names exact supported
CLI/API paths, outputs and restrictions. These commands only read guidance.

<!-- client: mac -->

The app's Studies → Research methods and guides displays the same text.

<!-- client: all -->

`GET /api/science/catalog`, `/api/science/guide/<method>` and
`/api/science/operation/<operation>` return it from either workbench HTTP
implementation; the engine also serves these reads in runner role.

Method IDs: `extraction`, `readers`, `optimization`, `finetuning`, `jlens`,
`jspace`, `sae`, `stability`, `batteries`, `judging`, `style`, `multi-agent`.
Choose the method, comparison, split roles, model/revision and claim with the
researcher. Ask about unresolved scientific decisions; obtain pins through
inspection/import operations. When delegation is approved, give independent coworkers the method guide and
author/reviewer prompts. A valid file is not scientific validation.

## Reviewed remote submission

The catalog distinguishes engine-only execution from an absent interface:
standalone `steerlab-server experiment extract-stability` and `battery run`
also have reviewed remote submission via `{{remote}} science-plan` and
`{{remote}} science-submit`.

Use a request file with `operation` and `parameters` from the method guide,
then submit its exact `--plan-sha256` within the same controller session. A
controller restart needs a new plan. For portable inputs, use `science input-plan`
and `science package`, the permitted upload or external transfer, then
`{{remote}} science-stage`. Stage returns the isolated
request to plan and submit. Local workspace files remain authoritative.

For completed evidence, use `{{remote}} science-fetch`,
or `science import` after approved external transfer and `science-export`.

`science custody` and `science verify-custody` re-read retained archives and
expanded output files offline. Custody proves bytes, not scientific quality.
Only isolated successful diagnostic output copies can be removed using
`cleanup-plan` then `cleanup-apply --confirm-removal`; an explicit server retention
policy, unchanged plan and freshly verified local receipt are required. Inputs,
export archives, job records and local evidence remain. Partial and resumable
outputs stay protected. Never infer permission to remove from a job's completion.

The job uses the declared local/Slurm executor and keeps its scientific output
type; these standalone diagnostics have no checkpoint resume. Existing HTTP
scientific actions are listed with method, route and service role by `science
operation`. Use `{{remote}} science-call` with a method-guide request; the same
server gates apply.

<!-- client: mac -->

The app's Scientific workflows actions make the same calls.

<!-- client: all -->

J-space has its own public operation `jspace`, separate from OptVec
training. Its current batch owner requires OptVec artifact metadata for layer
and dose; the historical `optvec jspace` command is an implementation namespace.

Use `science interview <operation>` for the shared method form and decisions,
then `science draft <operation> --answers <answers.json>` to review resolved input
hashes. Publish with `science publish <operation> --answers <answers.json>
--destination requests/<new-name> --plan-sha256 <reviewed-hash>`. Add `--json`
to every command. Keep numeric answer values as strings, and preserve original
advanced JSON; the owner writes exact integer seeds. The app authors through
this same owner. Publication creates a request plus its review, not a study or
execution authorization. Package and stage it before reviewing the engine plan.

Managed operations include style rescoring, SAE family/qualification reports,
OptVec training/evaluation/geometry/fracture/interpretation/family/gradient/mint,
and separate J-space analysis. A campaign job first materializes its cells;
`science-call optvec-campaign --action post-science-campaign` reaches status/plan
or explicitly confirmed submit/cancel actions. Inspect uncertain scheduler
outcomes before another top-up. Successful campaign export requires every cell
complete and scheduler termination established. Non-diagnostic scientific outputs
remain on the runner: the battery/stability cleanup policy does not cover them.
For offline SAE roster work use `science sae-check`, `sae-show`, `sae-pin-plan`
and `sae-pin`; the pin needs the reviewed external hash and an unchanged draft.
Discovery or a passing source test does not qualify a numerical claim.

## Other `science` verbs

The `science` family also carries local, reviewed operations for instruments
beyond steering vectors. Each takes `--help`, and every write takes the
`--plan-sha256` its review step returned, refusing if anything changed in
between.

- **Probes and measurements:** `science probe-list` and `science probe-inspect
  <path>` read the workspace's probes; `science measurements-review
  <experiment> --settings <file>` reviews probe measurement settings for a
  draft study, and `science measurements-save <experiment> --settings <file>
  --plan-sha256 <digest>` saves them to the unchanged draft. `science
  evidence-analyze <path>` compares retained probe readings and requested or
  applied policy actions in a run.
- **Intervention policies:** `science policy-list`, `science policy-inspect
  <path>`, `science policy-review <settings.json>`, and `science
  policy-publish <settings.json> --plan-sha256 <digest>`, which publishes the
  reviewed policy as an immutable artifact. `science policy-attach-review
  <settings.json>` then `science policy-attach <settings.json> --plan-sha256
  <digest>` create a new agent version without editing its source.
- **Fitting corpora:** `science corpus-preview <spec.json>` reads the chosen
  data sources and captures a preview. Public dataset files may download, so
  ask the researcher first. `science corpus-publish <preview-id> --plan-sha256
  <digest> --destination <dir>` saves the reviewed corpus and its provenance
  in a new directory.
- **Custom instruments:** `science artifact-plan <description.json>` inspects
  a custom lens or SAE decoder and hashes its source files without publishing;
  `science artifact-import <description.json> --plan-sha256 <digest>` imports
  the reviewed instrument into a fresh library destination.
- **Readable reports:** `science report <run-folder-or-report.json>` turns a
  stored J-lens assessment report into one self-contained HTML page: a plain
  summary first, then every comparison with its hashes, a chart and a table
  by layer for each text, the texts each lens was fitted on, and a list of
  anything omitted. A new assessment run already holds the page as
  `assessment-report.html`. For an older run the page is written to
  `reports/<run-name>/` in the workspace, or to `--out <file>`, and never into
  the run folder. The page shows only what the report stores: it adds no
  average, interval, or test. Give the researcher the `htmlPath` from the
  result to open in a browser or send to a colleague.
