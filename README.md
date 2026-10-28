# SteerLab

**Change how a language model behaves from the inside, then measure what
changed, fairly.** Works through a Mac app, your coding assistant, or the
command line.

One tool covers the whole loop. Try open-weight models on your Mac, then
steer them with directions you extract from your own examples or with
sparse-autoencoder features you found on Neuronpedia. Read what the model
represents while it writes, with probes and J-lenses. Run the same study on
your Mac or on a Slurm cluster, and let your coding assistant drive any of
it.

A language model's behavior can be nudged while it writes by adding a
direction to its internal activations. SteerLab lets you make those
directions from examples you write, or import them from published
interpretability work. You then apply them at a chosen strength and compare
the result against a fair baseline. Every number is tied to the exact inputs
and settings that produced it.

You make the scientific choices. SteerLab keeps the work organized, records
what ran and which checks applied, and leaves the judgment of what the
results support to you.

You can study one model, compare several configured versions of it, or watch
what happens when they talk to each other. In SteerLab these configured
models are called **agents**. Your **coding assistant** (such as Claude Code
or Codex) is the separate tool that can help you drive SteerLab.

**[Download from the Releases page](https://github.com/christortheak/InterpBench/releases/latest)**
· **[Getting started](docs/CLIENT-FIRST-RUN.md)**
· **[Report a problem](CONTRIBUTING.md#reporting-a-problem)**

*Pre-release research software. It has so far run in earnest on a small
number of installations. Expect some rough edges; setup guidance and
diagnostic messages explain what needs attention. Which methods run on which
hardware, and what has actually been measured, is recorded in
[capabilities and measured limits](docs/SUBSTRATES.md).*

## Choose how to start

| You have | Start with | What you get |
|---|---|---|
| An Apple Silicon Mac (macOS 26.4 or later) | [The Mac app](#use-the-mac-app), or [one link for your coding assistant](#hand-one-link-to-your-coding-assistant) | Everything: design, run, and read studies on the Mac, or run them on another machine |
| x86_64 Linux, or a Mac without the app | [The app-free client](#the-app-free-client-preview) (preview) | Design studies and send them to a runner someone has set up |
| A development or GPU machine, and want the source | [Build from source](#build-from-source) | The command lines and the Python engine |

Windows is not supported. You can start designing a study before you
download a model or decide where it will run. The app and your coding
assistant work on the same workspace folder, so you can switch between them
at any time.

### Hand one link to your coding assistant

Give your coding assistant this sentence:

> Set up SteerLab for me by following
> https://github.com/christortheak/InterpBench/blob/main/docs/AGENT-SETUP.md
> and then help me design my first study.

[That page](docs/AGENT-SETUP.md) is written for the assistant. It shows you
a plan and asks before downloading anything. On an Apple Silicon Mac it
installs and checks the app, creates a workspace for your study, and offers
to open the app on it. On x86_64 Linux it sets up the app-free client. The
assistant needs permission to run commands and to reach the internet; if it
has neither, the page ends with the same steps for you to do by hand.

### Use the Mac app

1. **Download `SteerLab-<version>+<revision>.zip`** from the Releases page,
   unzip it, and open SteerLab. Keep the app wherever you like, such as
   `/Applications` or a `SteerLab` folder in your home directory.
2. **Follow Research Setup.** It opens at first launch and stays in the
   Workspace menu. Create or open a workspace folder, or open a copy of a
   Demo Workspace, a worked example. Choose where studies run. Then review
   the plan for the small Python helper that study design uses, and approve
   it.
3. **Work in the app, or hand the workspace to your coding assistant.**
   Click **Copy Instructions for Your Coding Assistant** in Research Setup,
   open the workspace folder in your coding assistant, and paste the
   instructions followed by your question. For example:

> Read this workspace's AGENTS.md and help me design a study. I want to test
> whether an intervention changes a model's response style on new prompts.
> Help me choose the method and controls, prepare the data, and review a
> draft before we run anything.

No Xcode, repository checkout, or separate Python is needed. The one-time
helper setup needs internet access. Choosing where studies run installs
nothing by itself; downloading a model, and any setup the place you chose
needs, are separate steps you take when you are ready. When a newer release
is out, SteerLab says so; choose **Install Update…**
([details](docs/CLIENT-FIRST-RUN.md#updating-and-repairing)).

<details>
<summary>Prefer the command line on the Mac?</summary>

The app carries its own command line, `steerlab-cli`. Put it on your `PATH`
with:

```sh
"<path-to>/SteerLab.app/Contents/Helpers/steerlab-cli" install link
```

This makes `~/.local/bin/steerlab-cli` point into the app, and updates keep
it working. It never replaces a file that is not SteerLab's, and it tells you
when `~/.local/bin` is not on your `PATH` and the line to add. Then
`steerlab-cli --version` reports 7/7 resource families resolved. Do not name
the link `steerlab`: that name belongs to the app-free client, a different
product.

</details>

### The app-free client (preview)

The app-free client creates workspaces and designs studies, then sends them
to a runner that someone has set up: a workstation, a cluster, or a local
engine installed separately. It does not run models itself.
`results export` writes a completed run's tables, transcripts, and methods
summary to files. On a Mac, the app is the supported route to running
studies and reading their results.

1. **Download `steerlab-client-<version>+<revision>.tar.gz`** from the
   Releases page and extract it into a folder.
2. **Open that folder in your coding assistant.** It needs permission to
   read local files and run commands.
3. **Paste this:**

> Read AGENTS.md in this folder and help me set up SteerLab. Show me the
> installation plan before installing anything. Then create a new study
> workspace outside this folder, give me its handoff, and follow the
> workspace's AGENTS.md from there. I want to start by discussing my research
> question; we can choose models and compute later.

The installer provisions Python for you; no repository, preinstalled Python,
or GPU is needed. It needs internet access, plus curl, tar, and a SHA-256
tool, and its plan checks for them and states the download size before
anything is downloaded. Supported: Apple Silicon macOS, and x86_64 Linux
with glibc 2.28 or newer.
[Getting started](docs/CLIENT-FIRST-RUN.md#route-2-the-app-free-client-preview)
has the steps for running the installer yourself.

### Build from source

For development, or to run the Python engine on your own GPU machine, start
from a checkout. [AGENTS.md](AGENTS.md) is the contract for a coding
assistant pointed at this repository, and
[ONBOARDING.md](docs/ONBOARDING.md) §4.3 is the walk for a person. Install
the Python engine from the committed dependency lock for your platform
([Server/README.md](Server/README.md), "Dependency locks"), so that two
machines resolve the same `torch` and `transformers`. A source install gives
you the command lines, not a workspace; create one with `workspace init` on
either command line.

## How a study goes

1. **Ask the question.** The study interview helps you state what you want
   to learn and choose controls, before any model is chosen.
2. **Make a direction**, from pairs of example texts you write or by
   importing one, and check that it separates held-out examples.
3. **Choose a strength.** A sweep tries several layers and strengths on
   prompts kept apart from the study, and you pick one by reading the
   results.
4. **Freeze the design.** Freezing locks the study's settings and inputs
   before any behavior is measured, after checks that the inputs are pinned
   and the model still answers unrelated questions sensibly.
5. **Run and read the results**, on this Mac or another machine, against a
   paired baseline and matched controls, with every result traceable to its
   inputs.

[Onboarding](docs/ONBOARDING.md) walks through each step, and
[Conducting a study](docs/CONDUCTING-A-STUDY.md) explains what makes a result
defensible.

## What you can do with it

- **Make and apply directions.** Extract a direction from contrastive
  examples with one of five extraction methods, or import one from published
  work, such as sparse-autoencoder features. Apply it at a chosen layer and
  strength during generation, or remove it.
- **Measure fairly.** Compare against paired baselines and matched random
  controls, with seeded sampling, judged outcomes, and a capability battery
  (a short set of unrelated questions that shows whether steering broke the
  model).
- **Compare agents, alone and together.** Compare a base model with steered
  or fine-tuned versions, or seat several agents in a multi-agent scenario,
  vary their interventions, and examine how individual and group behavior
  change.
- **Read the model while it writes.** Train a probe (a small classifier that
  reads the model's internal state), record its readings while a study runs,
  and give an agent an intervention policy that changes its steering based
  on those readings.
- **Try models by hand.** The Mac app downloads open-weight models into your
  local cache and lets you chat with them, with or without an intervention,
  before you design anything measured.
- **Read and export results.** Results lead with the outcome your study is
  about. `results export`, or **Export Results…** in the app, writes a
  completed run's results to a new folder: tables for R, Stata, SPSS, or a
  spreadsheet, transcripts, a methods summary, and a codebook. Nothing is
  recalculated.
- **Go further.** Optimize a direction against a stated objective (OptVec),
  fine-tune an adapter and compare it with steering on the same footing, or
  inspect an intervention with a J-lens, which reads what each layer
  represents in the model's own vocabulary. The
  [glossary](docs/CONCEPTS-AND-GLOSSARY.md) explains these terms.

Your coding assistant can read the shipped method guides (`science list` and
`science guide <method>` on either command line).

### Where studies run

To run a study you need a model and somewhere to run it. Each workspace runs
its studies in one of three places:

- **This Mac, quick start**: the engine built into the app, for core steering
  studies on small models, with nothing to install beyond a model.
- **This Mac, full capabilities**: the Python engine on the Mac's graphics
  processor, which runs every method after a one-time setup of several
  gigabytes.
- **Another machine**: the Python engine on a workstation or a Slurm cluster
  you connect to. You keep designing and reviewing locally; results come back
  to your workspace with their evidence verified, and your coding assistant
  can drive the cluster steps for you.

A Mac laptop is enough for small models and core steering. Larger models,
long multi-agent conversations, and some methods need a machine with more
memory; [models, hardware, and limits](docs/MODELS-HARDWARE-AND-LIMITS.md)
says what fits where. A Demo Workspace, a finished study to read and a draft
to run, opens from **Open Demo Workspace…** in Research Setup, or with
`workspace init <folder> --demo <backend>` on either command line (a
release may carry none, and then says so).

## Where the depth is

- [Getting started](docs/CLIENT-FIRST-RUN.md): both routes, step by step,
  with time and disk space.
- [Onboarding](docs/ONBOARDING.md): from "what is activation steering" to a
  first frozen study, for a person at the keyboard.
- [A general introduction](docs/GENERAL-INTRODUCTION.md): what steering is
  and how a study is built, for a reader new to the field.
- [Conducting a study](docs/CONDUCTING-A-STUDY.md): what makes a result
  defensible.
- [Concepts and glossary](docs/CONCEPTS-AND-GLOSSARY.md): the terms, and how
  they map to research methods you know.
- [Working with a coding assistant](docs/WORKING-WITH-A-CODING-ASSISTANT.md):
  handing it a workspace, and what it asks before acting.
- [Models, hardware, and limits](docs/MODELS-HARDWARE-AND-LIMITS.md): where
  studies run, disk and time, and what has been measured.
- [Reporting and troubleshooting](docs/REPORTING-AND-TROUBLESHOOTING.md):
  writing up, citing, data ethics, and what each refusal means.
- [Technical overview](docs/TECHNICAL-OVERVIEW.md): the engines, the artifact
  model, and the design philosophy.
- [Capabilities and measured limits](docs/SUBSTRATES.md), the
  [CLI reference](docs/CLI-REFERENCE.md), the [changelog](CHANGELOG.md), and
  [SECURITY.md](SECURITY.md).

No model weights are distributed here. You download the models you choose
under their own licenses, some of which carry use restrictions that can
extend to artifacts you derive; see [NOTICE](NOTICE).

## Report a problem

Open an issue on the
[issue tracker](https://github.com/christortheak/InterpBench/issues/new/choose)
and follow the form; [CONTRIBUTING.md](CONTRIBUTING.md) says what to include
and how to propose a change. Never paste a token, a password, a log that
shows folder paths, or study data into an issue. Report a security
vulnerability privately, as [SECURITY.md](SECURITY.md) describes.

Apache License 2.0; see [LICENSE](LICENSE).
