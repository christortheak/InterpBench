# Working with a coding assistant

SteerLab is built so that an AI coding assistant can do the typing while you
make the scientific decisions. Any assistant that can read files and run
commands in a folder you choose will do. This page covers how to hand it a
workspace, what to ask first, what it should and should not do without asking
you, how to read what it reports, and how it shares a workspace with the app.

For the terms used here, see [concepts and glossary](CONCEPTS-AND-GLOSSARY.md).

## What you need

- A **workspace**: the folder that holds your study. Research Setup in the app
  creates one, and so does `workspace init <folder>` on either command line.
- A **command line** for the assistant to use. On a Mac with the app, that is
  `steerlab-cli`, which ships inside the app at
  `SteerLab.app/Contents/Helpers/steerlab-cli`. Without the app, it is the
  cross-platform `steerlab` client. They are two different programs with
  different commands. [Getting started](CLIENT-FIRST-RUN.md) covers installing
  either.

## Hand over the workspace

**From the app.** Open **Research Setup…** from the Workspace menu. Its last
step has a button, **Copy Instructions for Your Coding Assistant**. Open the
workspace folder in your coding assistant, paste what you copied, and then add
your question.

**From a command line.** Ask the client for the same instructions:

```sh
steerlab-cli workspace handoff --workspace <folder> --json   # the Mac command line
steerlab workspace handoff --root <folder> --json            # the cross-platform client
```

Either way, the instructions name the command line to use, the workspace
folder, and the first commands to run, and they point the assistant to the
workspace's own `AGENTS.md`. That file is the assistant's guide. It is short,
and the assistant fetches more detail on demand with `workspace guide <topic>`.
You do not need to explain SteerLab to the assistant.

## What to ask first

Start with your research question, not with a method or a model. For example:

> Read this workspace's AGENTS.md. I want to know whether a model's answers
> change when it is nudged toward a more formal register. Walk me through the
> study interview, explain each choice in plain words, and do not generate data
> or run anything yet.

A good assistant then runs the **study interview** (`authoring study <intent>`,
with the intent `conceptStudy`, `agentComparison`, or `multiAgent`). It asks
the same questions the app asks. For a concept study, these include which
concepts, what example texts already exist, the held-out validation texts, the
task prompts, and how the layer and strength will be chosen. It also asks how
you want to provide any data before anything is written. Other useful first
questions:

- "Which methods fit this question, and what does each need?" The assistant
  reads the method catalog (`science list --brief`) and one method's guide
  (`science guide <method>`).
- "Can you show me a finished example first?" If your release carries a Demo
  Workspace, the assistant can open a copy of it with
  `workspace init <new-folder> --demo <backend>`, where the backend is `mlx`,
  `mps`, or `cuda`. A release that carries none says so plainly.
- "What will this cost in time, disk, and money before we see a result?" See
  [models, hardware, and limits](MODELS-HARDWARE-AND-LIMITS.md) for what it can
  and cannot estimate.

## What it does without asking, and what it asks about

The workspace guide gives your assistant these rules.

**It may do these without asking:** read, list, inspect, verify, and preview
anything, and draft a plan for you to review.

**It discusses these with you before writing anything:** the method, the
comparison, the model and its version, and the claim the study is meant to
support.

**It asks, and waits for a clear yes, before any of these:**

- **Generating data.** Concept examples, validation sets, task prompts, and
  capability batteries are scientific inputs. It first offers your own files,
  pasted content, or a prompt you can give to a writer of your choice. If you
  want it to generate them, it agrees the scope, the tool or model that writes
  them, and the expected cost with you first.
- **Running anything that loads a model or uses compute.** That includes
  extraction, validation, sweeps, measured runs, judging with a local model,
  and every submission to another machine. It should say what will run, where,
  and roughly how long, or say that it cannot estimate it.
- **Downloading a model.** It shows you the plan first. A model can be tens of
  gigabytes.
- **Using a paid judge.** A judge that calls a paid service spends your money
  on every response it codes, and sends each response to that service.

After a yes, it works within what you approved without asking at every step.
Approval for one run is not approval for the next.

**It does not do these at all, unless you explicitly accept the consequence:**
freeze a study "with force" (skipping the evidence checks), which is recorded
permanently and makes the study not citable; and invent examples to fill a
missing scientific input. It never edits a run folder or a frozen study, and
never writes a password or token into a file or a command.

SteerLab enforces some of these rules itself: a frozen study cannot be edited,
a run folder is never overwritten, and installing the client helper needs the
hash of the plan you approved. Others depend on your assistant following its
guide, so it is worth watching for them.

**Never paste a password or a token into the conversation.** On a Mac,
SteerLab asks for credentials itself and keeps them in the Keychain. The
cross-platform client reads a runner's token from a file you name or from an
environment variable.

## Reading what it reports

Ask for the research meaning first. The guide tells your assistant to lead
with what a result means for your study and what to do next, and to keep
hashes, logs, and formulas for when you ask. If a report is hard to follow, say
so; that is a reasonable request.

Behind every report is an answer from SteerLab in a fixed form. Three parts of
it are worth knowing:

- **The state.** `ready` or `okWithAdvisories` means the step succeeded.
  `refused` means a check declined the request, and `failed` means something
  went wrong. Ask your assistant which one it got.
- **Refusals and their repairs.** A refusal is SteerLab working as intended:
  it declined something that would give an unreliable or unreadable result. It
  always says what is wrong and how to repair it. The assistant should carry
  out the repair, or ask you when the repair needs a decision or spends
  compute, and then try again. It should never get past a refusal by editing
  protected files or by forcing. The common refusals are listed in
  [reporting and troubleshooting](REPORTING-AND-TROUBLESHOOTING.md).
- **Advisories.** Things you should know that did not stop the step, such as
  a study with only one judge, or a validation that had nothing to score. Ask
  your assistant to tell you every advisory in plain words.

Good questions to ask along the way:

- "Which workspace did that command use?" Every answer names its workspace,
  and it should be yours.
- "Show me what the study will measure before we freeze it." The app's Studies
  page shows the same study, and `experiment verify <name>` checks every input.
- "Which outcome is the headline, and why?" Results summaries say which rule
  chose it.
- "Where did that number come from?" The answer should name a run folder or an
  exported file, never a figure the assistant worked out itself.

## The app and the assistant share one workspace

The app and your coding assistant read and write the same folder, in the same
file formats, so a study written in one is the same study in the other. You
can design in the app and ask the assistant to check it, or the other way
around.

A few habits keep the two in step:

- **Name the workspace in every command.** The Mac command line, given no
  workspace, uses the one the app last opened, which may be a different study.
  The handoff includes the workspace in every command it suggests.
- **Do not edit the same draft in both at the same moment.** Some commands
  check that a study has not changed since the assistant last read it, and
  refuse with `staleManifest` if it has. The repair is to read it again.
- **Let the app catch up.** The app shows what is in the folder when it loads
  a list. If something your assistant just made is not showing, move to
  another section and back. The Results section has a Refresh button.
- **Bring results home first.** A run on another machine appears in the app
  once it is in the workspace. The cross-platform client's `run` command brings
  it home when the run finishes; on a Mac, the assistant uses
  `cluster import --site <id>`.
- **Use one engine for one study.** A workspace is set to one place to run (see
  [models, hardware, and limits](MODELS-HARDWARE-AND-LIMITS.md)), and directions
  and results made by one engine do not count for the other. The app warns you
  when the workspace and the engine in use disagree.

The full machine contract your assistant follows is in its own guide: ask it
to run `workspace guide contract`, or read
[ONBOARDING.md](ONBOARDING.md) §9.
