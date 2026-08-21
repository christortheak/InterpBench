# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Schema compatibility is a promise, not a convention: workspaces and frozen
manifests created by one release keep loading and keep verifying in the next.
A breaking change to a run, manifest, or JSON-envelope schema gets a new
schema number, a reader for the old form, and an entry in this file — never a
migration that rewrites frozen bytes.

## [Unreleased] — first source release

The first public form of SteerLab: a buildable source tree for a
concept-steering workbench, complete enough for an outside researcher, or
their coding agent, to run a defensible study end to end.

### Added

- **The steering core.** Concept-agnostic extraction (contrastive activation
  addition, PCA over difference vectors, grand-mean contrast against a
  reference corpus, linear reading probes, and import of sparse-autoencoder
  features for cross-checking), residual-stream injection that fires on every
  decode step rather than only during prefill, and activation capture — all
  through vendored, hook-capable model implementations, since upstream model
  code exposes no residual-stream hooks. Steering strength is reported in
  units of the residual-stream norm at the layer, measured on a pinned
  reference corpus.
- **The experiment lifecycle and its firewall.** An experiment is a recipe:
  every input is pinned by SHA-256 alongside the options used to derive
  vectors from it, and runs re-derive rather than reuse stored bytes. Freeze
  is one-way and gated on a pinned model revision, scope-matched validation
  evidence, judge and variant validity, and a clean data repository; forcing
  it is loud and stamps which gates were skipped. Runs are immutable
  directories carrying a manifest snapshot, content hash, generations,
  metrics, and substrate metadata; an epoch guard refuses to analyze a run
  against a manifest it was not produced under.
- **The measurement layer.** Layer × alpha sweeps that select by a declared,
  manifest-level criterion; promotion of a sweep-selected cell into a variant
  artifact carrying its birth certificate; a screen → promote → confirm funnel
  with disjoint prompt pools; paired judging against a pinned rubric and judge
  panel; answer-token log-probability instruments for categorical outcomes;
  per-condition capability batteries; matched-norm random controls; and
  headless statistics with paired bootstrap confidence intervals, Wilcoxon
  tests, and multiplicity correction.
- **Two engines.** A Swift/MLX engine for Apple silicon and an independent
  Python/PyTorch engine for CUDA, sharing one artifact model and checked
  against each other by committed golden fixtures and a `vectors compare`
  parity verb. The Python engine adds durable jobs, seeded multi-sample
  stochastic runs with per-record RNG isolation, and a Slurm and SSH
  deployment path driven by versioned site profiles.
- **An agent-driveable command surface.** A `steerlab` binary installed by
  `scripts/install-cli.sh` that runs with no environment and no checkout on
  its PATH; `--json` across the study path emitting exactly one versioned
  envelope on stdout with all diagnostics on stderr; typed refusals carrying
  a stable gate id and a concrete repair; distinct exit codes for malformed
  invocation, gate refusal, not-found, and failure; and an `AGENTS.md`
  written into every workspace as the contract to hand a coding agent.
- **Workspaces.** Study data lives in a plain folder — its own git repository
  with `prompts/`, `experiments/`, and `runs/` — created by
  `steerlab workspace init`, never inside the code checkout. Freeze commits
  the workspace and snapshots every pinned input beside the manifest.
- **Documentation and legal files.** A generated CLI reference, a methods
  note, an end-to-end study guide, a results-architecture note, this
  changelog, plus `LICENSE` (Apache-2.0), `NOTICE`, `SECURITY.md`, and
  `CITATION.cff`.

### Known limitations

- No signed or notarized macOS application, and no packaged Python engine:
  the app runs from a developer launcher and the engine is an editable
  install from this checkout.
- Python dependencies declare version floors rather than a lockfile, so two
  sites can resolve different `torch` and `transformers` versions.
- Outside token mode, several mutating routes on the Python server remain
  reachable on loopback without authentication. See `SECURITY.md`; hardening
  is in progress.
- Local Swift runs are greedy-only — the run loop requires `temperature == 0`
  and a single seed, and stamps every generation record accordingly. Studies
  that need sampling run on the Python engine.
- Cluster site profiles do not yet represent every field a site may need, and
  parts of the remote environment are still generated from bootstrap
  constants rather than from the profile.
