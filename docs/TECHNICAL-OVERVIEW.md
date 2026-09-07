# SteerLab: technical overview

*The engines, the artifact model and the design philosophy, in full. This is
the long-form orientation that used to open the repository README. Install
and first-run instructions live in the [README](../README.md),
[CLIENT-FIRST-RUN.md](CLIENT-FIRST-RUN.md) and [ONBOARDING.md](ONBOARDING.md)
§4 and are not repeated here.*

SteerLab is a workbench for building and studying **steered agents**:
open-weight language models given deliberately induced dispositions, then
measured with the discipline of a controlled experiment.

An agent here is a base model plus a chosen combination of interventions,
activation vectors injected during generation and fine-tuned adapters,
though the workhorse case is a single vector at one layer and strength. The
vectors can be **extracted** in-workbench from contrastive stimulus sets
(contrastive activation addition, a grand-mean contrast against a reference
corpus, paired-difference PCA, the template-mediated RepE reader of Zou et al.,
linear probes), **optimized** against a stated objective (OptVec), or
**imported** from external interpretability work, including
sparse-autoencoder feature directions (e.g. Gemma Scope) and J-lens vectors.
However an agent is built, it lands in the same artifact model with the same
provenance, and a study is a **comparison between agents**: an agent against
its paired unsteered baseline, agents against each other under identical
conditions, or agents interacting in multi-agent scenarios to see how induced
dispositions propagate.

Underneath, the engine does three things, and everything else in the
repository exists to make them trustworthy:

1. **Source** a concept direction: extract it from stimuli you author, train
   it, or import and rescale one derived elsewhere. The position in the
   sequence a direction is read at, and the template it is read through, are
   declared rather than assumed, and a contrastive direction's opposite pole
   is minted as its own provenance-stamped artifact rather than left as a
   negative α.
2. **Inject** it during generation, at a chosen layer and a strength measured
   in units of the residual-stream norm at that layer, on every decode step
   rather than only during prefill; or ablate it, removing the direction's
   component from the stream.
3. **Measure** what moved, with paired baselines, matched-norm random
   controls, capability batteries, answer-token log-probabilities for
   categorical outcomes, trained activation readers, judged outcomes, and
   effect sizes with bootstrap confidence intervals.

The steering core is concept-agnostic. Concepts, stimulus sets, task prompts,
rubrics, and taxonomies are *data* you author in a workspace; no concept is
named in the engine. If a change to the core would not work equally for an
arbitrary concept, it is a bug.

## Guiding philosophy

Three commitments shape everything else in the design:

- **You control it locally, by hand or by LLM.** Your machine owns the
  workspace: the study definitions, the pinned inputs, the accepted evidence,
  the git history. Compute may happen elsewhere, on a local GPU, a Python
  server, or a Slurm cluster, but a compute site is a place runs execute,
  never the authority on what the study *is*. And "you" includes an agent
  acting for you: the macOS app and the command lines drive the same owners,
  every verb on the study path speaks `--json` with typed refusals and stable
  exit codes, and each workspace is born with an `AGENTS.md` written for
  exactly that hand-off. Point a coding agent at it and the instrument is as
  drivable by an LLM as by a person at the keyboard.

- **The agent is the unit of study.** Configured agents, base model plus
  interventions, are what get built, compared, and measured. The instrument
  enforces it structurally: in the app, configurations accumulate in a
  **variant library** and are promoted to named agents, and new capability
  routes through that library rather than around it, so anything you can
  build is automatically something you can compare.

- **Rigor is structural, not aspirational.** The discipline is built into the
  artifact lifecycle rather than left to good intentions: inputs pinned by
  hash, a one-way freeze that fixes every setting *before* behavior is
  measured, held-out validation for every extracted vector, matched-norm
  random controls, and doses reported in comparable units. Integrity gates
  refuse rather than warn, and a refusal names its repair. Beyond integrity,
  the instrument guides rather than forbids: a configuration that is
  implemented but not yet qualified on your hardware is available for
  exploration with its limitations and effective settings recorded, and the
  measured evidence for each configuration is inventoried in
  [SUBSTRATES.md](SUBSTRATES.md) rather than assumed. The section below on
  why a stranger can trust a result is the concrete form of this commitment.

## Two engines, three command lines, one artifact model

SteerLab has two independent compute engines that read and write the same
durable artifacts. `steerlab-cli` is a Swift/MLX engine for Apple silicon: a
native macOS instrument for authoring, fast iteration, and local runs, with
the full core lifecycle (extract, validate, sweep, promote, freeze, run,
analyze), MLX adapter training, and seeded sampling for measured runs.
`steerlab-server` is a Python/PyTorch/Hugging Face engine (FastAPI, with a
Slurm and SSH deployment path) for CUDA hardware and, on a Mac, for PyTorch
on MPS. It carries the same core lifecycle plus the managed scientific
operations that exist only there: OptVec optimization, J-lens acquisition
and qualification, J-space inspection, SAE analysis and import, and the
standalone capability battery. The Mac engine reads those artifacts without
executing their mathematics; reader fitting and scoring and the extraction
stability diagnostic have native twins on both engines. Which engine
produces which operation, and what has been measured on which backend, is
the subject of [SUBSTRATES.md](SUBSTRATES.md).

The engines are not wrappers around each other. Each implements extraction,
injection, and the freeze lifecycle natively, so activations do not transfer
between them: a concept extracted from pinned stimuli is **re-derived on
whichever engine a study's measured runs execute on**, and validation
evidence counts on the substrate it was produced on. Artifacts that are not
derived from stimuli, such as an imported SAE feature, a J-lens vector, an
OptVec-trained vector, or reviewed vector bytes attached through
`attach-artifact`, are pinned by digest with their provenance and are never
relabeled as belonging to a different backend. What is identical across
engines is the structure: the SHA-256 stimulus and corpus hashes, the
manifest and run-directory schemas, the JSON output contract, and the
committed golden fixtures that the `vectors compare` verb checks on both
sides.

A third command line, **`steerlab`**, is not an engine: it is the
cross-platform Python client that creates and authors a workspace locally
(`workspace init`, `setup start`, the design and pack verbs, the shared
study interviews) and hands hash-pinned work to an engine, either a managed
local runner or a remote one over `--runner <url>`, with evidence returning
verified. Two names, two products: `steerlab-cli` is the Mac instrument,
`steerlab` is the client, and neither answers the other's verbs. The
contract is [PORTABILITY-CONTRACTS.md](PORTABILITY-CONTRACTS.md), and
[CLI-REFERENCE.md](CLI-REFERENCE.md) §1.4 is the client's verb-by-verb
reference.

Everything converges on one workspace folder that moves and backs up as a
unit. On a Mac, a `SteerLab/` home holds `Workspaces/`, a private `Sites/`
registry for cluster profiles, the app and, if you have one, the checkout as
siblings ([ONBOARDING.md](ONBOARDING.md) §4.4). The app-free client needs no
prescribed home layout; a workspace can live anywhere.

## The study lifecycle

A workspace is a plain folder holding `prompts/`, `experiments/`, and
`runs/`: its own git repository, seeded with templates, the shipped method
guides, and its own `AGENTS.md`, which is written to hand a coding agent.
That contract keeps itself current: its header carries a hash of the body it
wrote, so an untouched one is refreshed to the shipped text when SteerLab
updates, and one you have edited is yours and is left alone. Study data
lives in a workspace, never in a code checkout. Either client creates one
(`steerlab-cli workspace init`, `steerlab workspace init` or
`steerlab setup start --create`). Drive the lifecycle through the app,
through your agent, or yourself.

The full local lifecycle below is the **Mac instrument's**, so every line
types `steerlab-cli`. The `steerlab` client authors the same manifest
(create, attach, declare conditions, sampling, parsers, freeze) and runs
studies through a runner, but does not execute extraction, sweep or analysis
itself; typing one of those verbs under `steerlab` exits `64`.

```bash
steerlab-cli workspace init ~/SteerLab/Workspaces/my-study
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/my-study

steerlab-cli experiment --help                # the study lifecycle, one line each
steerlab-cli experiment create demo --model <model-id>
steerlab-cli experiment attach demo <concept> # pins stimulus hashes + options
steerlab-cli experiment detach demo <concept> # attach's inverse, and gated
steerlab-cli experiment extract demo
steerlab-cli experiment validate demo
steerlab-cli experiment set-sampling demo     # the generation protocol
steerlab-cli experiment set-parser demo <p>   # the endpoint parser, registry-pinned
steerlab-cli experiment set-sweep-grid demo   # the layer x alpha axes
steerlab-cli experiment sweep demo            # layer x alpha on the dev split
steerlab-cli experiment promote demo <concept>
steerlab-cli experiment freeze demo           # one-way, and gated
steerlab-cli experiment run demo
steerlab-cli experiment analyze demo
steerlab-cli data check demo                  # data-readiness checklist
```

Everything a run measures is **declared** rather than inferred, and each
declaration has a headless writer: `set-instruments` and `set-instrument-scope`
for which instruments read which rows, `set-parser` for the endpoint grammar
(named from a workspace registry whose hash is pinned for you, never typed),
`set-sampling` and `set-exclusions` for the generation protocol and the
analysis-time exclusion rules, `pin-rubric --judges … --judge-pin` for the
judging instrument down to each local judge's weights, and
`set-sweep-selection` for the rule that picks a dose. A study is authorable
end to end from a terminal, which is what makes it authorable by an agent.

A study is blocked by missing *data* more often than by a missing verb, so
`authoring prompt <kind>` on either client emits the generation prompt for
each kind (contrastive pairs, a validation set, reader pairs, choice prompts,
a capability battery) carrying that kind's audit checks as numbers, and
`authoring study <intent>` emits the shared researcher interview that walks
from a question to a reviewed study pack. The templates are workspace data:
your copy wins over the shipped one, and every emission stamps the hash of
the wording it used.

The managed scientific operations (OptVec, J-lens, J-space, SAE, readers,
batteries, stability, style rescoring, deferred judgment) share one
interview source: `science list`, `science guide <method>` and
`science operation <id>` discover them on either client, `science interview`,
`science draft` and `science publish` turn conceptual answers into an exact
request with pinned input bytes, and the runner's plan and submit path
executes them on the Python engine.

`<family> --help` lists a family's verbs; `<family> <verb> --help` prints one
verb's arguments and exit codes. The full surface is in
[CLI-REFERENCE.md](CLI-REFERENCE.md), which is generated from the parsers
(`steerlab-cli docs cli-reference --check`) rather than maintained by hand.

## Built to be driven by an agent

Every verb on the study path accepts `--json`. In JSON mode stdout carries
exactly one envelope document and nothing else, all human diagnostics go to
stderr, there are no ANSI sequences, keys are sorted, and timestamps are
ISO-8601. Refusals are typed: a gate that declines a well-formed request
names itself with a stable code and states the repair. Exit codes separate
the cases an agent must distinguish (`0` ok, `64` malformed invocation, `65`
refused by a gate, `66` not found, `70` failed), and when `--json` is in
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
and the client emit the same envelope with the same closed key set, asserted
by tests on each side.

## Why a stranger can trust a result

The instrument is built around one claim: the settings that produce a result
were chosen and frozen *before* the behavior was measured, and anyone can
check it after the fact.

- **An experiment is a recipe, not a result.** `experiment.json` pins its
  inputs by SHA-256 (stimulus sets, neutral corpus, task prompts, rubric,
  markers, dev split, battery) together with the options used to derive
  vectors from them. For an extracted concept, vector bytes are never the
  artifact of record; runs re-derive them from the pinned recipe. For an
  imported or trained artifact, the reviewed bytes are pinned by digest with
  the manifest and sidecar that produced them.
- **Freeze is one-way and gated.** It re-verifies every pin against the file
  bytes on disk, requires a pinned model revision and validation evidence
  matching the exact scope, and stamps a content hash and data commit into the
  manifest. Iterating means duplicating, never editing.
- **`--force` is loud and permanent.** A forced freeze records which gates were
  skipped, by name, in the manifest itself. It stays checkable rather than
  remembered.
- **Runs are immutable.** Each run is its own timestamped directory carrying a
  manifest snapshot and content hash, raw generations, metrics, judge outputs,
  sampling provenance and substrate metadata: enough to rebuild the tables
  without rerunning the model. A run stamped with one manifest epoch will not
  be silently analyzed against a later one.
- **Drift is an error, not a surprise.** If a pinned file changes after a
  freeze, `verify` fails. There is no path by which a pinned input quietly
  becomes a different pinned input.
- **Sampling is seeded per record on both engines.** Each generated record
  carries its effective seed and sample index, derived the same way on both
  engines. A seed fixes the draw, not the arithmetic: equal seeds across
  backends do not promise equal tokens, and repeatability on a given backend
  is a measurement, recorded in
  [TECHNIQUE-PARITY-QUALIFICATION.md](TECHNIQUE-PARITY-QUALIFICATION.md).

Extracted vectors are validated before use: each must move a held-out probe
for its own concept, and cross-concept similarities are reported so distinct
concepts cannot collapse into one direction. Strength is reported in units
of the residual-stream norm on a pinned reference corpus, so a dose is
comparable across concepts and layers.

## The Python engine

The Python engine is installed from this checkout as an editable package,
from the committed dependency lock for your platform rather than from the
version floors alone, so that two sites resolve the same `torch` and
`transformers`. The maintained instructions, the lock regeneration recipe
and the cluster's site-owned-torch exception are in
[Server/README.md](../Server/README.md) under "Dependency locks";
[ONBOARDING.md](ONBOARDING.md) §4.3 has the short form. Serve it with an
explicit `--root <workspace>`; the artifact root must be the workspace, not
the `Server/` directory. The server requires a bearer token by default and
binds loopback; reach a remote one over an SSH tunnel and read
[SECURITY.md](../SECURITY.md) before exposing it to anything else.

The extras are deliberate: `lora` (adapter training, PDF stimulus ingestion),
`gemmascope` (sparse-autoencoder feature analysis), `test`, and `all`. The
`jlens` extra is excluded from `all` on purpose, because it raises the
effective `transformers` floor and installs from a git URL, so opt into it
explicitly when you need it. The achieved resolution is stamped into every
run's `config.json`, and a run whose installed versions differ from the lock
says so in an advisory rather than dying.

## Documentation

- [GENERAL-INTRODUCTION.md](GENERAL-INTRODUCTION.md): the one-document
  synthesis for a new reader: the method, the instrument, the firewall, the
  study lifecycle.
- [METHODS.md](METHODS.md): extraction, injection, and validation, with the
  math and the source lineage.
- [REPE-IMPLEMENTATION-BRIEF.md](REPE-IMPLEMENTATION-BRIEF.md): what is
  actually implemented from Zou et al.'s Representation Engineering, schema
  by schema, with the faithful-vs-departure table.
- [CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md): how to run a defensible
  study end to end.
- [RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md): what each result layer
  can claim, and what gates it.
- [SUBSTRATES.md](SUBSTRATES.md): where each technique runs and what has
  been measured; [ADDING-A-TECHNIQUE.md](ADDING-A-TECHNIQUE.md) for
  extending the catalog.
- [CLI-REFERENCE.md](CLI-REFERENCE.md): every verb, flag, default, and
  refusal on both engines' command lines (`steerlab-cli` and
  `steerlab-server`), with §1.4 covering the cross-platform `steerlab` client.
- Your workspace's own `AGENTS.md`, written at workspace creation: the
  contract to hand a coding agent.

## Status

Pre-release research software. What that means concretely:

- On macOS the packaged app is the distribution, and the CLI ships inside
  it, so an app install needs no Xcode; building from this checkout needs
  Xcode 27 and `xcodebuild` (SwiftPM alone cannot build the Metal shader
  library MLX needs). The app-free client ships as a release archive with a
  pre-Python installer; PyPI publication is a later stage.
- From a checkout, the macOS app runs through a developer launcher
  (`./scripts/run-app.sh`) rather than the assembled bundle.
- The Python engine is an editable install with committed platform locks.
  A site that installs from the floors instead can resolve different
  `torch` and `transformers` versions and produce different numbers; every
  run stamps what it actually ran.
- Supported user interfaces are the macOS app, the two engines' command lines
  (`steerlab-cli`, `steerlab-server`) and the `steerlab` client, plus the
  Python server's own browser workbench for remote use. Windows is out of
  scope: untested and unsupported.
- Cluster deployment works against a generic Slurm site through a versioned
  site profile that the app and the Mac CLI can review and accept, but the
  profile schema does not yet represent every field a site might need.
- See [SECURITY.md](../SECURITY.md) for the server's threat model and its
  known limitations.

Interfaces on the agent path (the JSON envelope schema, run and manifest
schemas, exit codes) are versioned, and additive change is the default. A
workspace created by one release loads in the next, and a frozen manifest
keeps verifying across upgrades; breaking changes get a new schema number, a
reader for the old form, and a changelog entry, never a migration that
rewrites frozen bytes.

## License

Apache License 2.0; see [LICENSE](../LICENSE). Third-party attributions for
vendored source are in [NOTICE](../NOTICE). Model weights are not distributed
here and carry their own licenses, some with use restrictions that can extend
to artifacts you derive from them.
