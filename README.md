# SteerLab

**Change how a language model behaves from the inside, then measure what
changed, fairly.** Works through a Mac app, your coding agent, or the
command line.

A language model's behavior can be nudged while it writes by adding a
direction to its internal activations. SteerLab lets you make those
directions from examples you write or import them from published
interpretability work, apply them at a chosen strength, and compare the
result against a fair baseline, with every number tied to the exact inputs
and settings that produced it. You make the scientific choices; SteerLab
keeps results traceable to their inputs and settings, and keeps the work
organized.

**[Download from the Releases page](https://github.com/christortheak/InterpBench/releases/latest)**
(the Mac app, or the app-free client archive) · **[Getting started](docs/CLIENT-FIRST-RUN.md)**

*Pre-release research software. It has so far run in earnest on a small
number of installations; expect rough edges away from the paved path, and
read a refusal before working around it. Which methods run on which
hardware, and what has actually been measured, is recorded in
[capabilities and measured limits](docs/SUBSTRATES.md).*

## Start here

Start designing a study before downloading a model or connecting to
compute. Pick either route; you can move between the app and your agent at
any time, because both work on the same workspace folder.

Release files carry the version and a short source revision in their names,
for example `SteerLab-0.9.6+416d0675.zip` and
`steerlab-client-0.9.6+416d0675.tar.gz`. The release notes give the client
archive's SHA-256, and the archive carries a `SHA256SUMS` for its contents.

### Use the Mac app

1. **Download `SteerLab-<version>+<revision>.zip`** from the Releases page,
   unzip it, and open SteerLab. Keep the app wherever you like, such as
   `/Applications` or a `SteerLab` folder in your home directory.
2. **Follow Research Setup.** It opens on first launch and stays in the
   Workspace menu. It creates or opens a workspace folder, and shows you a
   plan for the small Python helper that study authoring uses before
   installing it with your approval.
3. **Work in the app, or hand the workspace to your agent.** Click
   **Copy Agent Handoff** in Research Setup, open the workspace folder in
   your coding-agent tool, and paste the handoff followed by your question.
   For example:

> Read this workspace's AGENTS.md and help me design a study. I want to test
> whether an intervention changes a model's response style on new prompts.
> Help me choose the method and controls, prepare the data, and review a
> draft before we run anything.

Requires an Apple Silicon Mac on macOS 26.4 or later. No Xcode, no
repository checkout, no separate Python. The one-time helper setup needs
internet access. Models and where they run are separate choices you make
when you are ready.

<details>
<summary>Prefer the command line on the Mac?</summary>

The app carries its own command line, `steerlab-cli`, inside the bundle.
Run it from where it is, or put it on your `PATH`:

```sh
mkdir -p ~/.local/bin
ln -s "<path-to>/SteerLab.app/Contents/Helpers/steerlab-cli" ~/.local/bin/steerlab-cli
export PATH="$HOME/.local/bin:$PATH"
steerlab-cli --version      # reports 6/6 resource families resolved
```

Do not name the link `steerlab`; that name belongs to the app-free client
below, a different product with a different verb surface.

</details>

### Let your coding agent set it up, without the Mac app

1. **Download `steerlab-client-<version>+<revision>.tar.gz`** from the
   Releases page and extract it into a folder.
2. **Open that folder in your coding-agent tool.** It needs permission to
   read local files and run commands.
3. **Paste this:**

> Read AGENTS.md in this folder and help me set up SteerLab. Show me the
> installation plan before installing anything. Then create a new study
> workspace outside this folder, give me its handoff, and follow the
> workspace's AGENTS.md from there. I want to start by discussing my research
> question; we can choose models and compute later.

The folder's own AGENTS.md walks the agent through it: plan first, install
only with the approved hash, use the returned executable, keep the research
in a separate workspace. The installer provisions Python for you; no
repository, preinstalled Python or GPU is needed for authoring. It needs
internet access and the usual shell tools. Supported: Apple Silicon macOS
and x86_64 Linux with glibc. Windows is not supported.

<details>
<summary>Prefer to run the installer yourself?</summary>

From the extracted folder:

```sh
sh install-client.sh plan
```

Review the destination and actions, then install with the hash it printed:

```sh
sh install-client.sh install --expect <planSHA256> --yes
```

It returns the absolute path of the `steerlab` executable. Create a workspace
and get its handoff:

```sh
<executable> setup start ~/steerlab-studies/first-study --create --json
<executable> workspace handoff --root ~/steerlab-studies/first-study --json
```

</details>

### Build from source

For development, or to run the Python engine on your own GPU machine, start
from a checkout: [AGENTS.md](AGENTS.md) is the contract for a coding agent
pointed at this repository, and [ONBOARDING.md](docs/ONBOARDING.md) §4.3 is
the walk for a person. Install the Python engine from the committed
dependency lock for your platform rather than from the version floors, so
that two machines resolve the same `torch` and `transformers`; the
maintained instructions are in [Server/README.md](Server/README.md) under
"Dependency locks". A source install gives you the command lines, not a
workspace; create one with `workspace init` on either client.

## What you can do with it

- **Make a direction** from contrastive examples you write, with five
  extraction recipes, or import one from published interpretability work
  such as sparse-autoencoder features and J-lens vectors.
- **Optimize a direction** against a stated objective (OptVec training,
  evaluation, geometry and family campaigns), or **fine-tune an adapter**
  and compare it with steering on the same footing.
- **Apply it** at a chosen layer and dose, or remove it, during generation;
  inspect what an intervention does in J-space.
- **Measure** against paired baselines and matched random controls, with
  seeded sampling, judged outcomes, trained activation readers and
  capability batteries.
- **Compare agents**: a model plus its interventions is an agent, and a study
  is a comparison between agents, including agents talking to each other.
- **Run where the work fits.** To run a study, you need a model and suitable
  compute. Supported studies can run locally on an Apple Silicon Mac or
  through the Python engine on a workstation or cluster. Choose according to
  the method, model size and available memory. Remote results return to your
  local workspace with their evidence verified.

The methods, their inputs and their limits are described in the shipped
method guides your agent can read (`science list`, `science guide <method>`
on either command line).

## Where the depth is

- [Getting started](docs/CLIENT-FIRST-RUN.md): both routes, step by step.
- [Onboarding](docs/ONBOARDING.md): from "what is activation steering" to
  a first frozen study, for a person at the keyboard.
- [A general introduction](docs/GENERAL-INTRODUCTION.md): what steering is
  and how a study is built, for a reader new to the field.
- [Conducting a study](docs/CONDUCTING-A-STUDY.md): what makes a result
  defensible.
- [Technical overview](docs/TECHNICAL-OVERVIEW.md): the engines, the
  artifact model and the design philosophy, in full.
- [Capabilities and measured limits](docs/SUBSTRATES.md), the
  [CLI reference](docs/CLI-REFERENCE.md), the [changelog](CHANGELOG.md), and
  [SECURITY.md](SECURITY.md).

No model weights are distributed here. You download the models you choose
under their own licenses, some of which carry use restrictions that can
extend to artifacts you derive; see [NOTICE](NOTICE).

Apache License 2.0; see [LICENSE](LICENSE).
