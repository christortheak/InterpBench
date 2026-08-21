# SteerLab

SteerLab is a concept-steering workbench for open-weight language models. It
does three things, and everything else in the repository exists to make those
three trustworthy:

1. **Extract** a named concept direction from a model — a vector in the
   residual stream derived from a contrastive stimulus set, by contrastive
   activation addition, a PCA-based reading of difference vectors, a
   grand-mean contrast against a reference corpus, or a linear probe.
2. **Inject** that direction during generation, at a chosen layer and a
   strength measured in units of the residual-stream norm at that layer, on
   every decode step rather than only during prefill.
3. **Measure** what moved — with paired baselines, matched-norm random
   controls, capability batteries, answer-token log-probabilities for
   categorical outcomes, and effect sizes with bootstrap confidence
   intervals.

The steering core is concept-agnostic. Concepts, stimulus sets, task prompts,
rubrics, and taxonomies are *data* you author in a workspace; no concept is
named in the engine. If a change to the core would not work equally for an
arbitrary concept, it is a bug.

## Two engines, one artifact model

SteerLab has two independent compute engines that read and write the same
durable artifacts. `steerlab` is a Swift/MLX engine for Apple silicon: a
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

No model weights ship here. You download the models you want, and their
licenses are your own to read — see NOTICE.

## Quickstart

**If you have SteerLab.app, you already have the command line.** The app ships
the CLI it was built from, inside the bundle at
`Contents/Helpers/steerlab-cli`: the same binary the app itself calls into,
signed with the bundle, carrying its own copy of the Metal shader library. So
there is nothing to build and no Xcode involved — just give it a name on your
`PATH`:

```bash
mkdir -p ~/.local/bin
ln -s ~/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli ~/.local/bin/steerlab-cli

# `steerlab` is the shorter name the rest of these docs use, and the name
# install-cli.sh's shim takes. TEMPORARY alias: the `steerlab` name is
# reserved for a future cross-platform client, which will take it over —
# `steerlab-cli` is this binary's durable name.
ln -s ~/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli ~/.local/bin/steerlab

export PATH="$HOME/.local/bin:$PATH"
steerlab --version    # the build, and where every shipped resource resolved
steerlab --help
```

A symlink is enough: the binary resolves its shaders and the app's bundled
resources against its real location, not the link's. Invoking it in place by
full path works identically, and the path inside the bundle is the same
wherever the app lives — `~/SteerLab`, `/Applications`, anywhere.

**Building from source** is the developer path, and the only one that needs
Xcode 27:

```bash
git clone <this repository>
cd <checkout>

# Build the Swift CLI with xcodebuild and install it under ~/.local.
# Re-runnable; a failed install leaves the previous one untouched.
./scripts/install-cli.sh

# Put it on PATH if it isn't already, then confirm what you installed.
export PATH="$HOME/.local/bin:$PATH"
steerlab --version
steerlab --help
```

If `xcode-select` points at an older Xcode, set `DEVELOPER_DIR` to your
Xcode 27 installation before running the installer. `scripts/build-app.sh`
assembles the signed app from the same checkout, CLI included.

Either way, create the home layout — a `SteerLab/` folder holding your
workspaces, your private site library, and (if you have one) the checkout as
siblings, so one directory moves and backs up as a unit:

```bash
steerlab init                      # or: steerlab init --home /path/to/SteerLab
```

```text
~/SteerLab/
├── Workspaces/     study workspaces, one folder each, each its own git repo
├── Sites/          your PRIVATE site library (cluster-site profiles, presets)
├── SteerLab.app/   the app, and the CLI it carries
└── <checkout>/     this repository — any folder name; detected by content
```

`init` creates the two directories if they are absent and reports what was
already there; run it as often as you like. It never deletes or overwrites
anything, creates no workspace, and does not change how a workspace root is
resolved. `Sites/` is deliberately left empty so you can `git clone` your own
private site repository into it — keeping site configuration out of workspaces
you share.

Then create a workspace — a plain folder holding `prompts/`, `experiments/`, and
`runs/`, its own git repository, seeded with templates and its own
`AGENTS.md`. Study data lives in a workspace, never in this checkout:

```bash
steerlab workspace init ~/SteerLab/Workspaces/my-study
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/my-study
```

Any location works: the resolution order is `STEERLAB_WORKSPACE`, then
`--workspace`, then the choice the app has persisted.

Then either point your coding agent at the workspace's `AGENTS.md` — it is
written for exactly that — or drive the lifecycle yourself:

```bash
steerlab experiment --help                    # the study lifecycle, one line each
steerlab experiment create demo --model <model-id>
steerlab experiment attach demo <concept>     # pins stimulus hashes + options
steerlab experiment extract demo
steerlab experiment validate demo
steerlab experiment sweep demo                # layer x alpha on the dev split
steerlab experiment promote demo <concept>
steerlab experiment freeze demo               # one-way, and gated
steerlab experiment run demo
steerlab experiment analyze demo
steerlab data check demo                      # data-readiness checklist
```

`<family> --help` lists a family's verbs; `<family> <verb> --help` prints one
verb's arguments and exit codes. The full surface is in
[docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md), which is generated from the
parser (`steerlab docs cli-reference --check`) rather than maintained by hand.

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
- [docs/CONDUCTING-A-STUDY.md](docs/CONDUCTING-A-STUDY.md) — how to run a
  defensible study end to end.
- [docs/RESULTS-ARCHITECTURE.md](docs/RESULTS-ARCHITECTURE.md) — what each
  result layer can claim, and what gates it.
- [docs/CLI-REFERENCE.md](docs/CLI-REFERENCE.md) — every verb, flag, default,
  and refusal on both command lines.
- Your workspace's own `AGENTS.md`, written at workspace creation — the
  contract to hand a coding agent.

## Status

This is a source release, an early one. What that means concretely:

- The app is the distribution: `scripts/build-app.sh` assembles, signs, and
  packages SteerLab.app, and the CLI ships inside it, so an app install needs
  no Xcode. Building either from this checkout still needs Xcode 27, and there
  is no downloadable Python engine yet.
- From a checkout, the macOS app runs through a developer launcher
  (`./scripts/run-app.sh`) rather than the assembled bundle.
- The Python engine is an editable install, and its dependencies declare
  version floors rather than a lockfile. Two sites can therefore resolve
  different `torch` and `transformers` versions; pin them yourself if you are
  comparing across machines.
- Supported user interfaces are the macOS app and the two command lines, plus
  the Python server's own browser workbench for remote use.
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
