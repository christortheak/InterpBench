# SteerLab

> **Pre-release research software.** SteerLab is under active development and
> has so far run in earnest on a small number of installations — the authors'
> Macs and one Slurm cluster. It is still being adapted to other machines and
> sites: expect rough edges away from the paved path, read refusals before
> working around them, and pin your own dependency versions if you compare
> results across installations. Interfaces on the study path are versioned
> and change additively, but nothing here is 1.0.

SteerLab is a workbench for building and studying **steered agents**:
open-weight language models given deliberately induced dispositions, then
measured with the discipline of a controlled experiment.

An agent here is a base model plus a chosen combination of interventions —
activation vectors injected during generation and fine-tuned adapters —
though the workhorse case is a single vector at one layer and strength. The
vectors can be **extracted** in-workbench from contrastive stimulus sets
(contrastive activation addition, a grand-mean contrast against a reference
corpus, paired-difference PCA, the template-mediated RepE reader of Zou et al.,
linear probes) or **imported** from external interpretability work, including
sparse-autoencoder feature directions (e.g. Gemma Scope) and Jacobian-lens
vectors. However an agent is built, it lands in the same artifact model with
the same provenance, and a study is a **comparison between agents**: an agent
against its paired unsteered baseline, agents against each other under
identical conditions, or agents interacting in multi-agent scenarios to see
how induced dispositions propagate.

Underneath, the engine does three things, and everything else in the
repository exists to make them trustworthy:

1. **Source** a concept direction — extract it from stimuli you author, or
   import and rescale one derived elsewhere. The position in the sequence a
   direction is read at, and the template it is read through, are declared
   rather than assumed, and a contrastive direction's opposite pole is minted
   as its own provenance-stamped artifact rather than left as a negative α.
2. **Inject** it during generation, at a chosen layer and a strength measured
   in units of the residual-stream norm at that layer, on every decode step
   rather than only during prefill.
3. **Measure** what moved — with paired baselines, matched-norm random
   controls, capability batteries, answer-token log-probabilities for
   categorical outcomes, and effect sizes with bootstrap confidence
   intervals.

The steering core is concept-agnostic. Concepts, stimulus sets, task prompts,
rubrics, and taxonomies are *data* you author in a workspace; no concept is
named in the engine. If a change to the core would not work equally for an
arbitrary concept, it is a bug.

## Guiding philosophy

Three commitments shape everything else in the design:

- **You control it locally — by hand or by LLM.** Your machine owns the
  workspace: the study definitions, the pinned inputs, the accepted evidence,
  the git history. Compute may happen elsewhere — a local GPU, a Python server,
  a Slurm cluster — but a compute site is a place runs execute, never the
  authority on what the study *is*. And "you" includes an agent acting for
  you: the macOS app and the command lines drive the same engine, every verb
  on the study path speaks `--json` with typed refusals and stable exit codes,
  and each workspace is born with an `AGENTS.md` written for exactly that
  hand-off. Point a coding agent at it and the instrument is as drivable by an
  LLM as by a person at the keyboard.

- **The agent is the unit of study.** As the opening says: configured agents —
  base model plus interventions — are what get built, compared, and measured.
  The instrument enforces it structurally: in the app, configurations
  accumulate in a **variant library** and are promoted to named agents, and
  new capability routes through that library rather than around it, so
  anything you can build is automatically something you can compare.

- **Rigor is structural, not aspirational.** The discipline is built into the
  artifact lifecycle rather than left to good intentions: inputs pinned by
  hash, gates that refuse rather than warn, a one-way freeze that fixes every
  setting *before* behavior is measured, held-out validation for every vector,
  matched-norm random controls, and doses reported in comparable units. The
  section below on why a stranger can trust a result is the concrete form of
  this commitment.

## Two engines, one artifact model

SteerLab has two independent compute engines that read and write the same
durable artifacts. `steerlab-cli` is a Swift/MLX engine for Apple silicon: a
native macOS instrument for authoring, fast iteration, and local runs.
`steerlab-server` is a Python/PyTorch/Hugging Face engine (FastAPI, with a
Slurm and SSH deployment path) for CUDA hardware, where larger models and
stochastic multi-sample runs actually execute. They are not wrappers around
each other — each implements extraction, injection, and the freeze lifecycle
natively — so activations do not transfer between them and **vectors are
re-extracted and re-validated on whichever engine a study runs on**. What is
identical across engines is the structure: the SHA-256 stimulus and corpus
hashes, the manifest and run-directory schemas, the JSON output contract, and
the committed golden fixtures that the `vectors compare` verb checks on both
sides.

A third command line, **`steerlab`**, is not an engine and must not be confused
with either: it is the cross-platform Python client that authors a workspace
locally and hands hash-pinned work to an engine (the Quickstart's third seat,
and `docs/CLI-REFERENCE.md` §1.4). Two names, two products —
`steerlab-cli` is the Mac instrument, `steerlab` is the client, and neither
answers the other's verbs.

## Requirements

**Swift engine and app (macOS):**

- Apple silicon Mac. Intel is not supported.
- macOS 26.4 or later (the deployment floor; there is no fallback path).
- Xcode 27 **only to build from source**. SteerLab.app carries the command
  line it was built with, so installing the app needs no developer tools at
  all. If you do build, it has to be `xcodebuild` rather than `swift build`:
  SwiftPM alone cannot build the Metal shader library MLX needs, and nothing
  that touches the GPU works without it.
- Memory sized to the model tier you intend to run: roughly 3 GB for a 4-bit
  4B model, 13–16 GB at the 12–14B tier. Leave headroom for the KV cache and
  activations beyond the weights themselves.

**Python engine (cluster or workstation):**

- Linux with an NVIDIA CUDA GPU for real work. The engine also installs and
  runs on macOS for parity checks and tests.
- Python 3.10 or newer; 3.12 is what the project is developed and tested
  against.
- Disk for the Hugging Face model cache, which lives in the usual
  `~/.cache/huggingface` location and never inside this repository.

**Cross-platform `steerlab` client (any machine):**

- Python 3.10 or newer, and nothing else — no GPU, no torch, no Xcode.
  macOS, Linux, and Windows all author, freeze, package, and submit.
- Executing locally (`steerlab runner serve`, the `[runner]` extra) needs
  macOS or Linux. **Windows is client-only** and the serve verb refuses there;
  submit to a runner elsewhere instead.

No model weights ship here. You download the models you want, and their
licenses are your own to read — see NOTICE.

## Quickstart

```bash
git clone <this repository>
```

Then take either seat, or both:

- **Point a coding agent at the checkout.** [AGENTS.md](AGENTS.md) at the
  repository root is the cold-start contract: it tells the agent how to see
  what your machine already has, install the right command line for it — the
  Mac instrument `steerlab-cli` (from the app's bundle when there is no Xcode)
  or the cross-platform `steerlab` client — create the `~/SteerLab/` home layout
  and your first workspace, and hand off to that workspace's own `AGENTS.md`
  for study work. *"Read AGENTS.md and set me up"* is a complete instruction.
- **Open the Mac app.** Download SteerLab.app from this repository's
  Releases, unzip it into `~/SteerLab/`, and launch. The app carries the same
  CLI inside its bundle (`Contents/Helpers/steerlab-cli`), so this path needs
  no developer tools at all.
- **Not on a Mac?** `pip install -e "Server"` from the checkout installs the
  cross-platform `steerlab` client — ~30 MB, no GPU, no Xcode — which
  authors, freezes, and drives studies against any runner
  (`steerlab run <experiment> --runner <url>`); add `[runner]` to execute
  locally through `steerlab runner serve`, on macOS and Linux. Windows is
  **client-only**: authoring and remote submission work, `runner serve`
  refuses. A correct install answers `steerlab --version` with
  `steerlab <ver> (client)` — it is a different product from `steerlab-cli`
  and does not report resource families. The contract is
  [docs/PORTABILITY-CONTRACTS.md](docs/PORTABILITY-CONTRACTS.md).

Everything converges on one home folder that moves and backs up as a unit:

```text
~/SteerLab/
├── Workspaces/     study workspaces, one folder each, each its own git repo
├── Sites/          your PRIVATE cluster-site registry — one JSON file per
│                   site under cluster-sites/, read and written by the app
│                   AND steerlab-cli; keep it in a private git repo to sync
│                   between machines. Tokens and passwords never go here —
│                   they live in each Mac's Keychain.
├── SteerLab.app/   the app, and the CLI it carries
└── <checkout>/     this repository — any folder name; detected by content
```

Preferring to do it by hand — installing the CLI, `steerlab-cli init`, creating
a workspace — is [docs/ONBOARDING.md](docs/ONBOARDING.md) §4–5, and building
from source (the one path that needs Xcode 27) is covered there too.

## The study lifecycle

A workspace is a plain folder holding `prompts/`, `experiments/`, and
`runs/` — its own git repository, seeded with templates and its own
`AGENTS.md`, which is written to hand a coding agent. That contract keeps
itself current: its header carries a hash of the body it wrote, so an
untouched one is refreshed to the shipped text when SteerLab updates, and one
you have edited is yours and is left alone. Study data lives in a workspace,
never in this checkout. Drive the lifecycle through the app, through your
agent, or yourself.

The full lifecycle below is the **Mac instrument's**, so every line types
`steerlab-cli` — the Swift command line, whether installed from the app bundle
or built from source. (The cross-platform `steerlab` client authors, freezes,
and drives a runner; it has no `workspace init` and does not execute these
verbs. Typing one of them under `steerlab` exits `64`.)

```bash
steerlab-cli workspace init ~/SteerLab/Workspaces/my-study
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/my-study

steerlab-cli experiment --help                # the study lifecycle, one line each
steerlab-cli experiment create demo --model <model-id>
steerlab-cli experiment attach demo <concept> # pins stimulus hashes + options
steerlab-cli experiment detach demo <concept> # attach's inverse, and gated
steerlab-cli experiment extract demo
steerlab-cli experiment validate demo
steerlab-cli experiment set-sweep-grid demo   # the layer x alpha axes
steerlab-cli experiment sweep demo            # layer x alpha on the dev split
steerlab-cli experiment promote demo <concept>
steerlab-cli experiment freeze demo           # one-way, and gated
steerlab-cli experiment run demo
steerlab-cli experiment analyze demo
steerlab-cli data check demo                  # data-readiness checklist
```

A study is blocked by missing *data* more often than by a missing verb, so
`steerlab-cli authoring prompt <kind>` emits the generation prompt for each
kind — contrastive pairs, a validation set, reader pairs, choice prompts, a
capability battery — carrying that kind's audit checks as numbers. The templates are
workspace data: your copy wins over the shipped one, and every emission stamps
the hash of the wording it used.

`<family> --help` lists a family's verbs; `<family> <verb> --help` prints one
verb's arguments and exit codes. The full surface is in
[docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md), which is generated from the
parser (`steerlab-cli docs cli-reference --check`) rather than maintained by
hand.

To run the test suite or the macOS app from the checkout:

```bash
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO
./scripts/run-app.sh
```

Serialized testing is required, not a preference: the suite serializes a
process-global data-root override, and running it in parallel starves the
cooperative thread pool and wedges.

## Built to be driven by an agent

Every verb on the study path accepts `--json`. In JSON mode stdout carries
exactly one envelope document and nothing else, all human diagnostics go to
stderr, there are no ANSI sequences, keys are sorted, and timestamps are
ISO-8601. Refusals are typed: a gate that declines a well-formed request
names itself with a stable code and states the repair. Exit codes separate
the cases an agent must distinguish — `0` ok, `64` malformed invocation, `65`
refused by a gate, `66` not found, `70` failed — and when `--json` is in
effect the envelope's `state` is authoritative and the exit code is a
convenience. Advisories are reported without ever changing the exit code, so
a non-blocking warning cannot masquerade as a failure.

A refusal, verbatim:

```json
{
  "changed" : false,
  "engine" : "swift-mlx",
  "error" : {
    "code" : "freezeGateFailed",
    "gate" : "validateEvidence",
    "gates" : [ "validateEvidence" ],
    "reason" : "cannot freeze 'demo': no validate run matches its exact pins (model+revision, concepts, neutral corpus) on the run substrate swift-mlx. Run 'steerlab-cli experiment validate demo' first, or freeze --force to record an unvalidated experiment",
    "repairAction" : "Run 'steerlab-cli experiment validate demo' first, or freeze --force to record an unvalidated experiment"
  },
  "observedAt" : "2026-08-18T17:33:12Z",
  "schemaVersion" : 1,
  "state" : "refused",
  "verb" : "experiment freeze",
  "workspace" : "/Users/you/SteerLab/Workspaces/my-study"
}
```

Nothing in the envelope, or in any type it nests, can hold a credential:
secrets appear only as presence booleans and provenance labels. Both engines
emit the same envelope with the same closed key set, asserted by tests on each
side.

## Why a stranger can trust a result

The instrument is built around one claim: the settings that produce a result
were chosen and frozen *before* the behavior was measured, and anyone can
check it after the fact.

- **An experiment is a recipe, not a result.** `experiment.json` pins its
  inputs by SHA-256 — stimulus sets, neutral corpus, task prompts, rubric,
  markers, dev split, battery — together with the options used to derive
  vectors from them. Vector bytes are never the artifact of record; runs
  re-derive them.
- **Freeze is one-way and gated.** It re-verifies every pin against the file
  bytes on disk, requires a pinned model revision and validation evidence
  matching the exact scope, and stamps a content hash and data commit into the
  manifest. Iterating means duplicating, never editing.
- **`--force` is loud and permanent.** A forced freeze records which gates were
  skipped, by name, in the manifest itself. It stays checkable rather than
  remembered.
- **Runs are immutable.** Each run is its own timestamped directory carrying a
  manifest snapshot and content hash, raw generations, metrics, judge outputs,
  and substrate metadata — enough to rebuild the tables without rerunning the
  model. A run stamped with one manifest epoch will not be silently analyzed
  against a later one.
- **Drift is an error, not a surprise.** If a pinned file changes after a
  freeze, `verify` fails. There is no path by which a pinned input quietly
  becomes a different pinned input.

Vectors are validated before use — each must move a held-out probe for its own
concept, and cross-concept similarities are reported so distinct concepts
cannot collapse into one direction — and strength is reported in units of the
residual-stream norm on a pinned reference corpus, so a dose is comparable
across concepts and layers.

## The Python engine

At this stage the Python engine is a documented editable install from this
checkout; a packaged, versioned engine is a later stage.

```bash
python3.12 -m venv Server/.venv.nosync
Server/.venv.nosync/bin/pip install -e "Server[all]"

# Serve from the repository root, or pass --root explicitly. The artifact root
# defaults to STEERLAB_ROOT or the current directory, so serving from inside
# Server/ would point prompts, experiments, and runs at the wrong tree.
Server/.venv.nosync/bin/python -m steerlab_server.cli serve --root <workspace>

# Its tests:
cd Server && .venv.nosync/bin/python -m pytest -q
```

`./scripts/start-local-server.sh` does the venv creation, install, and launch
in one step, and writes the pidfile the macOS app uses to adopt a running
server. The server binds `127.0.0.1` by default. Reach a remote one over an
SSH tunnel — see SECURITY.md before exposing it to anything else.

The extras are deliberate: `lora` (adapter training, PDF stimulus ingestion),
`gemmascope` (sparse-autoencoder feature analysis), `test`, and `all`. The
`jlens` extra is excluded from `all` on purpose — it raises the effective
`transformers` floor and installs from a git URL — so opt into it explicitly
when you need it.

## Documentation

- [docs/GENERAL-INTRODUCTION.md](docs/GENERAL-INTRODUCTION.md) — the
  one-document synthesis for a new reader: the method, the instrument, the
  firewall, the study lifecycle.
- [docs/METHODS.md](docs/METHODS.md) — extraction, injection, and validation,
  with the math and the source lineage.
- [docs/REPE-IMPLEMENTATION-BRIEF.md](docs/REPE-IMPLEMENTATION-BRIEF.md) —
  what is actually implemented from Zou et al.'s Representation Engineering,
  schema by schema, with the faithful-vs-departure table.
- [docs/CONDUCTING-A-STUDY.md](docs/CONDUCTING-A-STUDY.md) — how to run a
  defensible study end to end.
- [docs/RESULTS-ARCHITECTURE.md](docs/RESULTS-ARCHITECTURE.md) — what each
  result layer can claim, and what gates it.
- [docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md) — every verb, flag, default,
  and refusal on both engines' command lines (`steerlab-cli` and
  `steerlab-server`), with §1.4 covering the cross-platform `steerlab` client.
- Your workspace's own `AGENTS.md`, written at workspace creation — the
  contract to hand a coding agent.

## Status

This is a source release, an early one. What that means concretely:

- The app is the distribution on macOS: `scripts/build-app.sh` assembles,
  signs, and packages SteerLab.app, and the CLI ships inside it, so an app
  install needs no Xcode; building from this checkout still needs Xcode 27.
  Off the Mac, the cross-platform `steerlab` client installs from this
  checkout (`pip install -e "Server"` — light; `[runner]` to execute
  locally); PyPI publication is a later stage.
- From a checkout, the macOS app runs through a developer launcher
  (`./scripts/run-app.sh`) rather than the assembled bundle.
- The Python engine is an editable install, and its dependencies declare
  version floors rather than a lockfile. Two sites can therefore resolve
  different `torch` and `transformers` versions; pin them yourself if you are
  comparing across machines.
- Supported user interfaces are the macOS app, the two engines' command lines
  (`steerlab-cli`, `steerlab-server`) and the `steerlab` client, plus the
  Python server's own browser workbench for remote use.
- A **cross-platform `steerlab` client is available as a preview** for agents
  and developers: a third command line that authors a local workspace on any
  platform, installed by `pip install -e Server` with no torch. The install is
  self-contained — the authoring-prompt registry travels inside the package, so
  a machine with no checkout beside it still renders those prompts. See
  `docs/CLI-REFERENCE.md` §1.4 and `docs/PORTABILITY-CONTRACTS.md` §7. The name
  `steerlab` is the client's; the source installer writes its shim as
  `steerlab-cli` and never takes the short name unless you pass
  `--short-name`. If an older install left a `~/.local/bin/steerlab` symlink or shim
  pointing at the Swift CLI, **drop it** and type `steerlab-cli` for Mac
  verbs, per `AGENTS.md` step 1.
- Cluster deployment works against a generic Slurm site through a versioned
  site profile, but the profile schema does not yet represent every field a
  site might need.
- See SECURITY.md for the server's threat model and one known limitation that
  is being closed.

Interfaces on the agent path — the JSON envelope schema, run and manifest
schemas, exit codes — are versioned, and additive change is the default. A
workspace created by one release loads in the next, and a frozen manifest
keeps verifying across upgrades; breaking changes get a new schema number, a
reader for the old form, and a changelog entry, never a migration that
rewrites frozen bytes.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party attributions for
vendored source are in [NOTICE](NOTICE). Model weights are not distributed
here and carry their own licenses, some with use restrictions that can extend
to artifacts you derive from them.
