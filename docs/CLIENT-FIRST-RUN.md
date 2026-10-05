# Getting started

SteerLab keeps each project in a **workspace**: one folder that holds your
prompts, study designs, and results. You can work on that folder in the Mac
app, through a coding assistant (an AI tool that can read files and run
commands for you), or both at once. This page takes you from a download to a
workspace where you can design a study, and then to your first results.

There are two routes:

- **The Mac app** is the supported route from a study design to results. It
  needs an Apple Silicon Mac with macOS 26.4 or later. No Xcode, no copy of
  the source code, and no Python of your own are needed.
- **The app-free client**, a command line named `steerlab`, is a preview. It
  designs a study and sends it to a runner that someone has already set up.
  It runs on an Apple Silicon Mac or on x86_64 Linux with glibc. Windows is
  not supported.

You can start designing a study before you download a model or decide where
it will run. Both routes make the same kind of workspace, so you can move
between them, and between the app and your coding assistant, at any time.

## What you will need

| Step | How long | Disk space |
|---|---|---|
| Download SteerLab | a minute or two | the app is a download of about 50 MB; the client is a few MB |
| Create a workspace | under a minute | about 1 MB to begin with; it grows with your studies and results |
| Set up the study-design helper | a few minutes, with an internet connection | a few hundred MB |
| Download a model, when you are ready to run a study | depends on your connection | typically 3 to 35 GB for each model |
| Set up the Python engine on this Mac, only for "This Mac, full capabilities" | depends on your connection | about 2 GB to download, and several GB once installed |

As a guide to download times, 2 GB takes about 3 minutes at 100 megabits per
second, and about half an hour at 10 megabits per second.

Models are stored in your Hugging Face cache (`~/.cache/huggingface` unless
you have changed it). No model is distributed with SteerLab. You choose and
download each model under its own license, and some licenses restrict what
you may do with things you derive from the model; see [NOTICE](../NOTICE).

## Route 1: the Mac app

### 1. Download and open the app

Download `SteerLab-<version>+<revision>.zip` from the
[Releases page](https://github.com/christortheak/InterpBench/releases/latest).
A `.sha256` file beside it holds its checksum. Unzip it and open SteerLab. You
can keep the app wherever you like, such as `/Applications` or a `SteerLab`
folder in your home folder.

### 2. Follow Research Setup

Research Setup opens when the app starts and you have no workspace yet. You
can return to it at any time: open the Workspace menu in the toolbar and
choose **Research Setup…**. It has four steps.

**Choose where your study lives.** Choose **New Workspace…** to create a
folder, **Open Workspace…** to use one you already have, or **Open Demo
Workspace…** to open a worked example. A Demo Workspace holds a finished study
you can read before you download a model, and a draft you can run yourself.
SteerLab copies it into a folder you choose and opens the copy, so the
original never changes. A release may not include a Demo Workspace; if it
does not, the app says so.

**Choose where studies run.** There are three choices. Choosing one installs
nothing by itself, and anything that needs a download shows you what it will
fetch and waits for your approval.

| Choice | What it runs | What it needs |
|---|---|---|
| This Mac, quick start | core steering studies on small models, with the engine built into the app | a model, and nothing else |
| This Mac, full capabilities | every method, with the Python engine on this Mac's own graphics processor | a one-time setup of several gigabytes |
| Another machine | every method, on a workstation or a cluster you connect to; your study files and results stay on this Mac | a machine to connect to |

A Mac on its own, including a laptop, is enough for either of the first two
choices. A new workspace starts on the quick start. **What Runs Where…** shows
which methods each choice covers. A workspace belongs to one engine: vectors
and results made by one engine do not carry over to the other, so if you
switch, build your vectors and run your study again. You can change the
choice later from the Workspace menu.

**Set up the study-design helper.** The helper is a small Python program
that guides study design, brings results back into your workspace, and
prepares text collections. It does not download a model and does not set up
a server. Choose **Review Setup Plan** to see what will be installed and
where, then choose **Approve and Install**. Earlier installs are kept, and
none of your study files change.

**Begin with your question.** Choose **Copy Instructions for Your Coding
Assistant**. Open the workspace folder in your coding assistant, paste the
instructions, and then describe what you want to understand. For example:

> Read this workspace's AGENTS.md and help me design a study. I want to test
> whether an intervention changes a model's response style on new prompts.
> Help me choose the method and controls, prepare the data, and review a
> draft before we run anything.

The instructions ask the assistant to read the workspace's own `AGENTS.md`,
to start with a short interview about your study, and to ask you before
anything that spends compute or money, such as downloading or running a
model, submitting a job, or calling a paid judge. To work in the app instead,
close Research Setup and open Studies or Templates.

### 3. When you are ready to run a study

Prepare a model in the Playground or in Compute. A download keeps running
while you work, and a cancelled download resumes. If you chose "This Mac,
full capabilities", the app offers the Python engine's one-time setup, which
lists what it will download before anything is fetched. If you chose
"Another machine", open the Compute menu in the toolbar and choose **Add a
Machine by Address…** or **Set Up a Cluster…**.

A study is frozen before its measured run. Freezing fixes every setting
before any outcome is measured, and it cannot be undone; to change a frozen
study, you duplicate it. The Studies page lists what a study still needs
before it can be frozen, and why.

### 4. Read and export your results

A study's results lead with the outcome the study is about. That is the
**primary outcome** you declare on the Studies page, if you declare one;
otherwise SteerLab chooses by a fixed order that puts judged outcomes first
and surface measures, such as word count, last. The results say which rule
chose the outcome.

**Export Results…** writes a completed run's stored results to a new folder:
tables that open in R, Stata, SPSS, or a spreadsheet, transcripts for coding
by hand, a plain-language methods summary, and a codebook that describes every
column. Nothing is recalculated, and the run itself is never changed. For a
J-lens assessment, **Open Report** shows its results as one self-contained
web page.

### The Mac command line, if you want it

The app carries its own command line, `steerlab-cli`. To type it by name,
link it into a folder on your `PATH`:

```sh
mkdir -p ~/.local/bin
ln -s "<path-to>/SteerLab.app/Contents/Helpers/steerlab-cli" ~/.local/bin/steerlab-cli
export PATH="$HOME/.local/bin:$PATH"
steerlab-cli --version
```

`--version` reports how many of the app's resource families it found; all of
them should resolve. Do not name the link `steerlab`, which is the app-free
client's name. Always tell `steerlab-cli` which workspace to use, with
`--workspace <folder>` before the command or with the `STEERLAB_WORKSPACE`
environment variable. For example, `steerlab-cli workspace init <folder>`
creates a workspace, `steerlab-cli workspace init <folder> --demo <backend>`
opens a copy of a Demo Workspace, and `steerlab-cli --workspace <folder>
results export <study>` exports results.

## Route 2: the app-free client (preview)

The app-free client designs a study, checks it, freezes it, and sends it to a
runner: a machine with the Python engine that someone has set up, which gives
you its address and a token file. The client from the release does not run a
model itself. The Mac app remains the supported route to results.

### 1. Download and extract

Download `steerlab-client-<version>+<revision>.tar.gz` and its `.sha256` file
from the [Releases page](https://github.com/christortheak/InterpBench/releases/latest),
and extract the archive anywhere. The folder it makes holds the installer, a
`README.md`, a `SHA256SUMS` file for its contents, and an `AGENTS.md` written
for a coding assistant. The folder is not a workspace; your studies will live
in a separate folder.

### 2. Install the client, or let your coding assistant do it

To let your coding assistant do it, open the extracted folder in your coding
assistant. It needs permission to read local files and run commands. Paste:

> Read AGENTS.md in this folder and help me set up SteerLab. Show me the
> installation plan before installing anything. Then create a new study
> workspace outside this folder, give me its handoff, and follow the
> workspace's AGENTS.md from there. I want to start by discussing my research
> question; we can choose models and compute later.

To do it yourself, run these from the extracted folder:

```sh
sh install-client.sh plan
```

This changes nothing. It shows where the client will be installed, what the
installer will do, and a plan hash. If you agree with the plan, install it
with that hash:

```sh
sh install-client.sh install --expect <planSHA256> --yes
```

This takes a few minutes and a few hundred MB. It downloads a verified copy
of `uv`, a Python installer, and a managed copy of Python, then installs the
client and its locked dependencies into a separate folder:
`~/Library/Application Support/SteerLab/client-runtime` on a Mac, or
`~/.local/share/SteerLab/client-runtime` on Linux. Add `--runtime
<absolute-path>` to both commands to choose another folder. It needs internet
access to GitHub, Astral's Python distribution, and PyPI, and the `curl`,
`tar`, and SHA-256 tools found on most systems. It needs no administrator
rights, and it does not edit your shell's startup files.

The result names the full path of the `steerlab` command. Use that path, or
add its folder to your `PATH`. `<executable> --version` should print
`steerlab <version> (client)`.

### 3. Create a workspace and hand it to your coding assistant

```sh
<executable> setup start ~/steerlab-studies/first-study --create --json
```

This creates a workspace outside the download folder, checks that it is ready
for designing a study, and returns the instructions for your coding assistant.
Leave out `--create` to open a workspace you already have. To print the
instructions again:

```sh
<executable> workspace handoff --root ~/steerlab-studies/first-study --json
```

Open the workspace folder in your coding assistant and give it those
instructions with your question. The workspace's own `AGENTS.md` guides the
assistant from there, and `workspace guide` lists further topics that it can
read when it needs them.

### 4. Run on a runner, and export the results

With the runner's address and the path of its token file,

```sh
<executable> run <study> --root <workspace> --runner <url> --token-file <path>
```

takes the study to the runner and brings its verified evidence home. Before a
study is frozen, `--verb validate` and `--verb extract` run the two steps that
come before freezing; the measured run itself needs a frozen study. A token is
always read from a file, never typed on the command line, where other
programs on the machine could read it.

Then export what the run stored:

```sh
<executable> results export <study> --root <workspace>
```

This writes the same tables, transcripts, methods summary, and codebook as the
app's **Export Results…**, into a new folder under `exports/` in the
workspace. The client has no results view of its own; open the tables in R,
Stata, SPSS, or a spreadsheet.

## Where to go next

- Your workspace's `AGENTS.md`, and `workspace guide <topic>` on either
  command line, are what your coding assistant reads.
- [Onboarding](ONBOARDING.md) explains activation steering and walks a first
  study on the Mac command line.
- [Conducting a study](CONDUCTING-A-STUDY.md) covers what makes a result
  defensible.
- [A general introduction](GENERAL-INTRODUCTION.md) describes the method for
  a reader new to the field.
- [Capabilities and measured limits](SUBSTRATES.md) records which methods run
  on which hardware, and what has been measured.

## Updating and repairing

In the app, choose **Check for Updates…** in the SteerLab menu. Research Setup
offers **Review Update Plan** when the helper is missing newer tools, and
**Review Repair Plan** when it is already up to date; either way it shows a
plan first and installs nothing until you approve.

For the app-free client, download the new release and run `plan` and
`install` again from its folder, or use `repair` in place of `install` to
build a fresh copy of the current one. Each install builds a new environment,
checks it, and only then switches to it. If a download or check fails, the
previous environment stays in use. Earlier environments are kept, and setup
never deletes study data or models.

## If something goes wrong

Every refusal says what is wrong and how to repair it; follow the repair. If
that does not help, see [reporting a problem](../CONTRIBUTING.md#reporting-a-problem).
Never paste a token, a password, a log that shows folder paths, or study data
into a public report.

## For maintainers: how the installer works and how a release is qualified

This section is for people who build and check releases. A researcher does
not need it.

**Installer guarantees.** The installer never replaces an ordinary existing
folder; an environment you made yourself stays usable when you select its
interpreter explicitly. Managed upgrades and `repair` need a new plan and its
approval. Each creates a new environment, verifies imports and the source
identity, and switches the public runtime link to it in one step. A failed
download or verification leaves the prior runtime in place. A concurrent setup
refuses until the other one releases its lock. The installer does not start
services, download models, choose a cluster, or supply credentials. Paths stay
local to the authoring workspace; cluster execution receives copies through
managed submission and verified import.

**Readiness.** `setup inspect --root <workspace> --json` reports client and
workspace readiness without treating a missing model as a setup failure. It
reports basic authoring, Parquet, public dataset downloads, and offline token
previews separately. `clientReady` means every capability is available, while
`basicClientReady` and `authoringReady` report the usable authoring path. An
older environment can continue basic authoring while Research Setup offers
**Review Update Plan** for missing corpus tools; see
[corpus preparation](FITTING-CORPUS-PREPARATION.md) for capability repairs.

**Building.** `python scripts/build-client-release.py --output
<new-directory>` builds a release, with uv 0.12.5 available; add `--archive`
to also write the `.tar.gz` and its checksum. The builder checks generated
resources and the compiled Python identity, and builds from a disposable copy
of the source. The app packager includes this same release inside its
`ServerPayload` before signing. The command builds artifacts only; publishing
and signing are separate release steps.

**Qualification.** `scripts/ci/qualify-client-release.py <release> --repair`
installs a built release in scratch space and exercises installation, workspace
creation, the agent handoff, method discovery, source identity, and the
absence of engine-only dependencies, outside the checkout. The Linux workflow
runs it on an independent runner. A real fresh-Mac pass through the app,
network-failure behavior, and updates remain release gates: a test on a
developer's machine cannot prove that experience by itself.
