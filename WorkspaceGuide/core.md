# AGENTS.md
<!-- Source of the core workspace guide. scripts/ci/check-workspace-bootstrap.py generates both clients' copies from this directory. Keep this file client-neutral and within its size budget; depth belongs in topics/. Raise "Guide version" whenever this text changes, so a refresh only ever upgrades. -->

Guide version: 5

You are a coding assistant working inside a **SteerLab data workspace**. This
file is the core contract: read it before running anything. It is short on
purpose. The installed client serves the depth on demand, one topic at a time
(see "Topics on demand" at the end).

This is *data*, not code. The SteerLab source tree lives elsewhere. Nothing
here is built or compiled. Study data stays in this folder: prompts,
experiments, and runs are written here and nowhere else.

---

## Collaborate at the researcher's level

The central experimental object is an **agent**: a base model with its chosen
steering vectors and/or adapters and their settings. The unmodified base model
is the baseline agent. Vector strengths combine under the declared injection
convention; adapters have their own application convention. Explain the research
comparison using these objects before discussing files or commands.

Adapt explanations to the researcher's experience. Introduce a concept as the
behavior being investigated, extraction as estimating a direction from examples,
steering strength as how strongly it is applied, and a sweep as trying a planned
range of settings. Explain concept data (construct the direction), validation
(separate examples to check generalization) and capability batteries (check other
abilities we want to preserve) when each becomes relevant. Validation is not by
itself evidence that steering changes behavior. Use the same terms as the app,
and use the Oxford comma. Call reusable study settings templates; the current `design` CLI family operates on those same
templates. Explain this mapping when showing commands.

Choosing a model and concept is NOT permission to generate datasets, call paid
models or launch coworkers. Before producing data, offer existing files, pasted
content, a copyable authoring prompt, or explicitly commissioned generation.
For generation/delegation agree the scope, authoring model/tool, worker roles and
expected resource use first; say when cost cannot be estimated. Respect a chosen
external or less expensive author. After approval, proceed within that batch
without repeatedly asking. Previewing schemas and drafting a plan need not wait.
Never invent examples to fill missing scientific inputs without that agreement.

Lead progress and problems with their research impact and next action. For
example: "The dataset needs revision: two checks did not measure the examples,
and several responses exceed the agreed length. Here are the affected rows."
Keep formulas, hashes, engine logs and detailed audit reasoning available on
request or in a technical report. Avoid dramatic audit narration and unexplained
jargon. Preserve scientific uncertainty; guidance is not an automatic veto.

## Ask before you spend

Ask the researcher, and wait for a clear yes, before any of these:

- **Generating data.** Concept examples, validation sets, task prompts, and
  batteries are scientific inputs. The section above says what to offer first.
- **A run that loads a model or spends compute.** Extraction, validation,
  sweeps, measured runs, evaluation by a local judge, and every remote or
  cluster submission. Say what will run, where, and roughly how long. Say so
  when you cannot estimate it.
- **A model download.** Show the plan first. A model can be tens of gigabytes.
- **Paid judges.** A judge that calls a paid service spends the researcher's
  money on every response it codes.

Reading, listing, inspecting, verifying, previewing, and drafting a plan need
no permission. After a yes, proceed within what was approved without asking
again at every step. Approval for one run is not approval for the next.

## Choose the installed client

Two command-line clients can work in this workspace. Use the one the
researcher or the setup handoff names. They are two products with different
verbs: never type one client's command under the other's name.

| Client | Executable | Names the workspace with | Runs studies |
|---|---|---|---|
| The Mac app's command line | `steerlab-cli` | `--workspace <dir>` | on this Mac, or on a cluster site |
| The cross-platform Python client | `steerlab` | `--root <dir>` | through a runner, local or remote |

The Python client is a preview: it authors studies and submits them to a runner
someone has set up. It does not run models itself. On a Mac, the app is the
supported route to running studies and reading their results.

Either client also reads `STEERLAB_WORKSPACE`. Name the workspace in every
session: on the Mac command line an unnamed workspace falls back to the app's
last choice, which may be a different study. Every JSON answer carries a
top-level `workspace` field; check it on your first command.

Both clients can initialize a new workspace with `workspace init <directory>
--json` and return an agent handoff with `workspace handoff --json`.
A source checkout is needed only to develop the product, not to conduct studies.
Keep the authoring workspace local; a runner's filesystem is execution storage.

`setup start <directory> --create --json` returns client readiness, a new workspace
and an agent handoff; omit `--create` to use an existing workspace. `setup inspect
--json` separates authoring prerequisites from execution readiness; select the
workspace using that client’s global directory flag. Inspect `setup plan --json` before provisioning; run `setup apply` or
`setup repair` with its `--expect` hash and `--yes` only after the researcher
approves the displayed installation. Do not install a GPU stack to author a study.

## Discover what the client can do

Never guess a verb or a flag. Ask the installed client, in this order:

1. **The verb card.** `workspace guide verbs` gives one line per task: the
   command and the rule that applies. Read it first, and load a full topic
   only when it does not answer.
2. **Help.** `--help` works at three levels — the client, a verb family, and
   one verb. It runs nothing and exits 0, and with `--json` the page comes back
   as data.
3. **The study interview.** `authoring study <intent> --json`, with the intent
   `conceptStudy`, `agentComparison`, or `multiAgent`, returns the same
   questions the app asks. Discuss the substantive choices with the researcher
   before authoring anything.
4. **The method catalog.** `science list --brief --json` names the methods and
   operations, and says where each operation runs (drop `--brief` for the full
   catalog).
   `science guide <method> --json` returns one method's dataset shapes, author
   prompt, and independent review prompt. These commands only read guidance.
5. **The topic guides.** `workspace guide` lists them, and `workspace guide
   <topic>` prints one. The list is at the end of this file.

To show a researcher a finished study first, open a copy of a Demo
Workspace: `workspace init <new-folder> --demo <backend>` (the `workspace`
topic).

Method IDs: `extraction`, `readers`, `optimization`, `finetuning`, `jlens`,
`jspace`, `sae`, `stability`, `batteries`, `judging`, `style`, `multi-agent`.
Choose the method, comparison, split roles, model/revision and claim with the
researcher. Ask about unresolved scientific decisions; obtain pins through
inspection/import operations. When delegation is approved, give independent coworkers the method guide and
author/reviewer prompts. A valid file is not scientific validation.

## What this folder is

```
prompts/          git-versioned inputs
experiments/      experiment manifests — freezable recipes
runs/             immutable run outputs (gitignored)
catalog/          GENERATED navigation over runs/ (symlinks; gitignored)
adapters/         per-adapter training data and outputs
WORKSPACE.md      the marker file that makes this a workspace
.gitignore        runs/, catalog/, adapters/**/*.safetensors, .DS_Store
```

An experiment is a **recipe**, not results: it pins inputs by SHA-256 plus the
options used to derive vectors from them, and runs re-derive deterministically.
That pinning is the firewall — settings are chosen and frozen *before* behavior
is measured, so a result cannot be reverse-fitted to the settings that produced
it.

A new workspace carries generic instruments only, and no concepts. None of the
seeded content is study material: adapt it before any run you intend to keep.
Where each file lives, and its exact shape, is the `workspace` topic.

## The study lifecycle

| Step | What it is for | Topic |
|---|---|---|
| Choose the question and the method | Agree the comparison, the model, and the claim with the researcher | `methods` |
| Author or import the data | Concept examples, a held-out validation set, task prompts, and a rubric | `workspace`, `sweep` |
| Create a draft and attach concepts | Pin each input by its hash, so the recipe is fixed | `lifecycle` |
| Declare what is measured | Task prompts, conditions (the arms), and generation settings | `lifecycle`, `settings` |
| Declare how it is judged | The rubric file and the judges | `evaluation` |
| Extract and validate | Derive each direction and check it on held-out examples; loads a model. On the cross-platform client, validate a draft on a runner with `run <name> --runner <url> --verb validate`; `--force` is not the route | `lifecycle` |
| Sweep and promote (optional) | Try a planned range of layers and strengths, then choose one | `sweep` |
| Verify and freeze | Fix the settings before behavior is measured; one-way | `freeze` |
| Run | Generate under every condition into an immutable run directory | `lifecycle`, `remote` |
| Evaluate and analyze | Judge the responses and compute effect sizes. Lead a summary with the outcome the study is about, and say how it was chosen; the `verbs` card names the export and report commands | `evaluation` |
| Keep the evidence | Verify what came home, and what it proves | `custody` |

Study packs and attaching existing agents or vectors are the `assembly` topic.
Templates, multi-agent panels, and model preparation have their own topics.

## The machine contract, in brief

- **Pass `--json` on every command.** Exactly one JSON document arrives on
  stdout; every diagnostic goes to stderr.
- **`state` is authoritative; the exit code is a convenience.** `0` succeeded
  (`ready`, `planned`, `running`, or `okWithAdvisories`), `10` a person must
  authenticate, `11` an explicit permission flag is missing, `12` work is in
  flight (`pending`), `13` retry later (`degraded`), `64` malformed
  (`blocked`), `65` `refused`, `66` `notFound`, and `70` `failed`.
- **A refusal is the instrument working.** `refused` means a gate declined a
  well-formed request; it is not a transient error. Read `error.code`,
  `error.gate`, and `error.repairAction`, carry out the repair, then retry.
  Never retry unrepaired, and never edit the state a gate protects to get past
  it. If a repair spells the other client's command, do the same act with this
  client's verb.
- **Read `advisories[]`.** An advisory never changes the exit code. It is not a
  failure, and it is not ignorable.
- **Never use `--force` unless the researcher explicitly accepts what it
  skips.** A forced freeze is stamped `freezeForced`, with every skipped gate
  in `forcedGatesSkipped`, permanently, and it is not citable. If the
  researcher asks for one, report exactly which gates were skipped.
- **Flags are strict.** An undeclared flag is exit `64` before the verb does
  any work, so a typo cannot silently change what a study means.

The full envelope, every state, and the refusal vocabulary are the `contract`
topic.

## Immutability

- **`runs/` is append-only.** Never edit, overwrite, or delete a run directory.
  A run carries enough to rebuild its tables without rerunning the model, and
  that is only true if nobody touches it. `runs/` is gitignored.
- **Three subtrees under `runs/` are deliberately mutable libraries**:
  `runs/model-variants/`, `runs/neutral-pcs/`, `runs/jlens-lenses/`. A promoted
  agent's artifact is editable in place there; frozen studies are protected by
  the manifest snapshot and the artifact hash, not by the directory.
- **Frozen manifests are read-only.** Every verb that writes a manifest refuses
  on a frozen or complete study. There is no unfreeze. Iterate with
  `experiment duplicate <name> <new-name>`, then edit the copy.
- **`experiments/<name>/pinned/`** is the freeze-time snapshot of every pinned
  input — the reproducibility floor when git is unavailable. Do not edit it.
- Do not hand-edit `experiment.json`. Its bytes *are* the content hash; an edit
  that bypasses the verbs surfaces as a verify violation, which is the good
  outcome, or as a silently different study, which is not.

## What not to do

- **Do not parse prose.** Use `--json` and read `state`, `error.code`,
  `error.gate`, `error.gates[]`, `advisories[].code`, and `result`.
- **Do not retry a `refused` (65) without performing the repair.** It will
  refuse identically. Read `error.repairAction`.
- **Do not `--force` a freeze to get past a gate.** It is stamped, loud, and
  permanently non-citable. Fix the gate instead. If a human explicitly asks for
  a forced freeze, report every id in `forcedGatesSkipped`.
- **Do not write into `runs/`**, and do not edit a frozen manifest or a
  `pinned/` snapshot.
- **Do not edit a manifest to iterate.** `duplicate`, then edit the copy.
- **Do not move or delete studies, templates, or agents by hand.** Use the
  `rename` and `delete` verbs, which preview first. The researcher sees your
  changes when they switch back to the app or press Refresh.
- **Never acknowledge custom code on the researcher's behalf.** A study can
  carry an intervention policy's custom code; show them the notice and the
  code (`experiment acknowledge-custom-code <study>`) and let them decide.
- **Do not treat an advisory as a failure**, and do not ignore one.
- **Do not skip `validation.jsonl`.** A study whose vectors were never probed
  on held-out material measures its own stimulus vocabulary.
- **Do not select a steering cell on marker density** for any study whose
  outcome is a decision rather than prose. It is a manipulation check.
- **Do not cite seeded or sample content.** It is there to be modified.
- **Do not guess at a file shape.** The shapes in the `workspace` topic are the
  ones the loaders parse; a wrong key is refused with the expected key named.
- **Do not guess a verb**, and do not type one client's command under the
  other's name. Ask `--help`.
- **Do not put a secret in a file.** Tokens and passwords never go in this
  workspace, in a command's arguments, or in anything you write.

## Topics on demand

`workspace guide` lists the topics, and `workspace guide <topic>` prints one,
with the commands of the client that printed it. Type it under the client you
were handed: `steerlab-cli workspace guide lifecycle` on the Mac command line,
or `steerlab workspace guide lifecycle` with the Python client. With `--json`,
`result` carries `topic`, `text`, and `topics`. The topics ship inside the
installed client, so they always describe the client you are using.

- `verbs` — The quick card: one line per task, with the command and the rule that applies. Start here.
- `workspace` — The folder layout, where each file lives and its shape, and how the client finds the workspace.
- `lifecycle` — The study lifecycle step by step, from a new draft to a measured run.
- `settings` — Generation settings, exclusions, the system prompt, parsers, and other declarations on a draft.
- `evaluation` — Rubrics and judges, evaluation, subsamples, effect sizes, and re-measuring a run.
- `sweep` — Sweeps, choosing a layer and strength, promotion, and the missing-data rule.
- `freeze` — What freezing checks, the seven gates and their repairs, and what stays immutable.
- `assembly` — Study packs, agents, vector artifacts, and prompt imports, each written from reviewed bytes.
- `templates` — Reusable study settings: listing, saving, instantiating, and batch casting.
- `panels` — Multi-agent studies: panels, seats, casting, and pipelines.
- `methods` — The method catalog, and reviewed submission of managed scientific operations.
- `models` — Preparing a model, locally or on other hardware, and its chat-template capabilities.
- `remote` — Running on other hardware: submission, sharding, resuming, and recovering jobs.
- `custody` — Verifying imported evidence, and what a custody receipt proves.
- `engines` — Which engine runs what, seeded sampling, and why vectors do not transfer between engines.
- `contract` — The full machine contract: the envelope, states, exit codes, advisories, and refusals.
