# Reporting and troubleshooting

This page covers writing up a study, citing SteerLab, handling data
responsibly, and what to do when SteerLab declines a request. The terms are in
[concepts and glossary](CONCEPTS-AND-GLOSSARY.md).

## Writing up a study

### Export the results

Exporting copies a completed run's stored results into a new folder that you
can open in R, Stata, SPSS, or a spreadsheet, and share.

- **In the app:** on the Studies page, choose the study, then **Export
  Results…** under "Runs & Results", and choose a new or empty folder.
- **On a command line:**

  ```sh
  steerlab-cli results export <study>   # the Mac command line
  steerlab results export <study>       # the cross-platform client
  ```

  Without options, it exports the newest completed run into a new folder under
  `exports/` in your workspace. `--run <run-folder>` picks another run, and
  `--out <folder>` names a new or empty folder instead.

Exporting never changes a run folder, loads no model, and recalculates nothing:
every number is copied from what the engine stored. Where something was not
stored, the export says it is not available rather than guessing.

What the folder holds, depending on what the study measured:

| File | What it is |
|---|---|
| `responses.csv` | One row for each response the model generated, with its condition and prompt |
| `choice-readouts.csv` | The model's probabilities for each answer option, read without sampling |
| `effects.csv` | Each outcome compared between each condition and the baseline: estimate, interval, test, and correction |
| `effects-by-stratum.csv` | The same comparisons within subgroups of the prompts |
| `judgments.csv`, `codings.csv` | What each judge decided, row by row |
| `transcripts/` | One text file for each conversation in a multi-agent study, and `turns.csv` with the same turns as a table |
| `methods.md` | A plain-language methods summary, built from the run's stored facts |
| `codebook.md` | Every column of every table, in a sentence each |
| `manifest.json` | Every file the export read, with its SHA-256 hash, and every file it wrote |

The tables are UTF-8 text with one header row. An empty cell means the run did
not record a value, never zero. Yes and no are written as 1 and 0. A line break
inside a response is written as ` ¶ ` so each row stays on one line.

### What the methods summary gives you

`methods.md` opens with a **Read this first** notice when the study was frozen
with force, was not frozen when the run was made, or had a condition the
capability battery could not test. Then it covers: what the export was built
from; the model, its exact revision, the engine, and the software versions;
the design (prompts, number of responses, sampling settings, system prompt,
and responses cut off at the length limit); the conditions and the capability
battery's scores; the outcomes, including the primary outcome if you declared
one; the judges, the rubric, and their agreement; exclusions; the statistics;
the freeze status and identifying hashes; and a final list of what was not
available. Adapt its wording for your paper, and check it against your own
records.

To have every summary lead with the outcome your study is about, declare it
before the run: choose it under "Primary outcome" in the study's evaluation
settings on the Studies page, or use
`experiment set-primary-outcome <name> <outcome>` on either command line.

### What to add yourself

The methods summary describes the run. These belong to your study, and you
write them:

- **The question and the hypotheses**, and why your example texts are a fair
  operationalization of the concept. Say how they were written: by you, taken
  from a source, or generated, and if generated, by which model and from which
  prompt.
- **How the direction was validated.** The validation run's folder holds its
  accuracy on the held-out texts and, with two or more concepts, how similar
  their directions are.
- **How the layer and strength were chosen**, and the rule you declared for
  choosing them. A sweep's run folder records the winning setting and the rule.
- **Why this many prompts.** SteerLab has no power calculation.
- **The bootstrap settings.** The summary says that the number of resamples,
  the seed, and the confidence level are not stored with the run; they are
  fixed by the SteerLab version it names.
- **Parse failures by condition**, if an outcome was read from free text.
- **Limits.** Which engine and precision you used, that results from other
  engines are not interchangeable, and what has and has not been measured; see
  [models, hardware, and limits](MODELS-HARDWARE-AND-LIMITS.md).
- **Ethics**, as below.

The frozen settings summary beside the study
(`experiments/<name>/preregistration.md`, or
`preregistration-frozen-settings.md` when you wrote your own
`preregistration.md`) is the record of what you fixed before measuring. Cite it,
or attach it.

To share the study design itself, `pack export <study>` on either command line
writes a study pack: the study's text inputs and a list of anything else it
depends on.

## Citing SteerLab

Cite the software and the exact version you used. The citation details are in
`CITATION.cff` at the root of the repository, and GitHub offers them under
"Cite this repository". The version that made a run is in `methods.md`, under
"Software that made the run"; `steerlab-cli --version` and `steerlab --version`
print the installed one. Also cite the model and its exact revision, and the
published sources of the method you used, which are listed in
[METHODS.md](METHODS.md). Say whether the study was frozen before it was run,
and say plainly if it was frozen with force.

## Data management and ethics

SteerLab is a measuring instrument, not a data-protection system. These notes
are where it touches research ethics.

- **Human baselines.** The human-effect table holds published summary numbers
  (an estimate and its interval for each outcome), not participant data. Copy
  each number from the source table itself, never from an abstract or a
  secondary citation, record the citation in the table, and do not put the
  paper's text in your workspace.
- **Your own participants.** If your study collects data from people, consent,
  ethics review, and storage are governed by your institution, as for any
  study. SteerLab has no participant records. Keep identifiable data out of the
  workspace, and enter only summary effects in the human-effect table.
- **Prompts from real people's writing.** If your task prompts or example
  texts come from interviews, archives, or social media, the usual terms of
  consent, licensing, and anonymization apply. The workspace is a git
  repository, freezing normally commits the study's inputs to it, and its
  history keeps every committed version.
- **Where your text goes.** It is sent to another machine when you run there,
  and to an outside service when you use a paid judge or ask an online model to
  generate examples. Check that this is allowed for your material before you
  do it.
- **Transcripts.** Responses can contain offensive, false, or disturbing text,
  especially at strong steering. Read them before sharing, and consider a
  content note. Some model licences also set terms on what you do with
  outputs. The workspace does not exclude `exports/` from its git history, so
  decide deliberately before committing an export.
- **Sharing for reproducibility.** The export folder (with `manifest.json`),
  the frozen study (`experiments/<name>/`, including its `pinned/` snapshot of
  every input), and the run folder are what another person needs to check your
  result. `runs/` is excluded from the workspace's git history, so share run
  folders separately.
- **Secrets.** Passwords and tokens never go in the workspace, in a command,
  or in anything you share.

## When SteerLab declines a request

Most problems arrive as a **refusal**: SteerLab declining a request that would
give an unreliable or unreadable result. Every refusal names what is wrong and
gives a repair you can carry out on the client you are using. On a command
line with `--json`, they are `error.reason`, `error.code`, and
`error.repairAction`, and the exit code is `65`. In the app, the message gives
the reason.

Three habits solve most of them:

1. **Do the repair, then try again.** Retrying without it gives the same
   refusal.
2. **Check the workspace.** Every answer names the workspace it used.
3. **Ask the client.** `--help` works on every command, runs nothing, and lists
   what the command accepts.

### Freezing

A freeze refusal has the code `freezeGateFailed`, and `error.gate` names the
check. `error.gates` lists every check that failed, not just the first.

| Check | What it means | What to do |
|---|---|---|
| `revision` | The model is not pinned to an exact published version | Mac: extracting or validating pins the version of the model on this Mac, or create the study with `--revision <commit>`. Client: `steerlab experiment pin-revision <name> <commit>`, or validate on a runner, which pins the version it used |
| `validateEvidence` | No validation run matches this study's exact inputs, or validation had no held-out texts to score | Write the concept's `validation.jsonl`, attach the concept again, then validate: `steerlab-cli experiment validate <name>`, or `steerlab run <name> --runner <url> --verb validate` |
| `batteryEvidence` | A condition has no capability-battery result | Validate again; it scores every condition the battery can run. A condition whose agent uses an intervention policy is exempt, and the frozen study records that |
| `judgeValidity` | The study has no rubric file, no judge, or two judges that are really the same one | Mac: `steerlab-cli experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md --judges a:local`, or your own rubric. Client: see `steerlab workspace guide evaluation` |
| `measurementPins` | A setting that decides what is measured is not usable, such as a precision the model cannot load | Point that setting at a usable value |
| `variantValidity` | An attached agent's adapter lacks a fingerprint of its weights or a record of its training data | Save the agent again so its weights are fingerprinted, and attach it again |
| `gitClean` | An input the study pins is not committed in the workspace's history | Commit the named files |

Do not freeze "with force" to get past a check. It is recorded permanently,
and the study is then not citable.

### Changing and running a study

On these refusals, `error.code` names the problem directly.

| Code | What it means | What to do |
|---|---|---|
| `emptyStudy` | Nothing is attached yet, so there is nothing to measure | Attach a concept or an agent, or start from a template. If the study is not planned yet, start with the study interview: `authoring study <intent>` |
| `statusImmutable` | The study is frozen, so it cannot change | `experiment duplicate <name> <new-name>`, and change the copy |
| `pinDrift` | A file the study pinned has changed, is missing, or has appeared | `experiment verify <name>` names each file. Restore it, or duplicate the study and pin the new version |
| `studyDeclaration` | The study's own settings are incomplete or contradict each other | Correct each setting the refusal lists, then `experiment verify <name>` |
| `missingPrerequisite` | The step needs something that is not there, such as task prompts, a completed run, or a file the study names | Write or add the named input, then try again |
| `dataReadiness` | `data check` found required data still missing | Write the files it names |
| `inertConditions` | The study would measure no condition, only the baseline | Declare a condition (an arm) the study actually runs |
| `responseFormat` | A declared outcome cannot read the prompts it points at | Change the outcome, or the prompts, as the refusal says |
| `thinkingModeConflict` | An answer-probability outcome is declared with the model's reasoning mode on | Turn reasoning off for the study, or measure the written text instead |
| `lengthStopped` | Too many responses in one condition and prompt hit the length limit | Raise the limit (`experiment set-sampling <name> --max-tokens <n>` on the Mac, `experiment set-protocol <name> --set maxTokens=<n>` on the client); duplicate first if the study is frozen |
| `conceptInUse` | A concept cannot be removed while a condition still uses it | Remove or change those conditions first |
| `manifestEpoch` | The run was made under different settings from the study's current ones | Analyze the run on the engine that produced it, or run the study again. To judge an old run with a new rubric, duplicate the study and evaluate the original run |
| `staleManifest` | The study changed since it was last read, for example in the app | Read it again (`experiment manifest <name>` on the Mac, `experiment inspect <name>` on the client), review the change, and resubmit |
| `promotionEvidence` | An agent was promoted with no sweep to choose it | Run the sweep first |
| `noCompletedRun` | `results export` found no completed run in this workspace | Run the study, bring the results home if it ran elsewhere, then export |

### Setup, workspaces, and connections

| Code or state | What it means | What to do |
|---|---|---|
| `noWorkspace` (Mac) | No workspace is chosen | `steerlab-cli workspace init <folder>`, then name it with `--workspace <folder>` or `STEERLAB_WORKSPACE` |
| `workspaceNotSet` (client) | No workspace was named | Add `--root <folder>`, or set `STEERLAB_WORKSPACE` |
| `demoNotCarried` | This copy of SteerLab carries no Demo Workspace for the backend you asked for | Create an ordinary workspace with `workspace init <folder>`, or ask for a backend the message says this copy carries |
| exit `64` | A command or option was mistyped, or is not on this client | Ask `--help`. Mac commands do not work under `steerlab`, and the reverse |
| `runnerUnreachable` | The runner did not answer | Check that it is running and reachable; a dropped SSH tunnel looks like this. `steerlab runner capabilities --runner <url>` is the smallest test |
| `runnerUnauthorized` | The runner wants a token the client did not send | Put the token in a file and pass `--token-file <path>`, or set `STEERLAB_RUNNER_TOKEN`; never type it on the command line |
| `notARunner` | The address answered, but it is not a SteerLab engine | Check the address and port |
| `needsHumanAuthentication` (exit `10`) | A person must sign in at their own terminal | Sign in yourself; your assistant cannot do it for you |
| `pending` (exit `12`) or `degraded` (exit `13`) | Work is still in progress, or something could not be read for now | Repeat the command later |
| `verbFailed` | Not a typed refusal: something failed | Read the reason. It usually names a file to fix |

### Advisories worth knowing

An advisory never stops a command, but each one matters for your write-up:
`judgePanelTooSmall` (one judge, so no agreement statistics),
`vacuousValidation` (a concept had no held-out texts to score, so freezing will
refuse), `freezeGateSkipped` (a check was skipped by a forced freeze),
`emptyAnalysis` (the run had no condition besides the baseline), and
`choiceItemsWithoutInstrument` (prompts carry answer options, but no outcome
reads them, so only the text is recorded).

### Reporting a problem

Include the versions (`steerlab-cli --version` or `steerlab --version`), what
you ran, and the refusal's code and repair. Never include tokens, logs that
show your folders or user name, or study data. For a problem with a cluster,
`steerlab-cli cluster diagnose --site <id> --redact` gives a status report with
user names and home folders removed, so it can be shared.
