# Brief: MLX and MPS parity program

**Implementation update (2026-09-07):** [TECHNIQUE-PARITY-IMPLEMENTATION.md](TECHNIQUE-PARITY-IMPLEMENTATION.md)
is the active refinement. In particular, use its exact per-path seed policies and
guidance-first qualification policy instead of the simplified derivation and
model-family refusal proposal below. [SUBSTRATES.md](SUBSTRATES.md) inventories
the current implementation without claiming new hardware measurements.

Written 2026-09-07 by the maintainer's integration agent for the refactoring
agents, on main `7284463`. This is a work program, not a design ruling: the
researcher decides which work packages run and in what order. Each package
names its acceptance criteria and the gates it must leave green. Nothing here
authorizes weakening a refusal, editing a frozen artifact, or claiming a
numerical result that has not been measured.

## 0. The situation, stated once

Three substrates exist. CUDA on the cluster produced every reported number
and is the reference. The Python engine on Apple silicon (PyTorch on MPS) has
the full feature set in code and no qualification: nobody has run the shipped
recipes end to end on MPS and compared. The Swift engine on MLX runs the core
loop only: five extraction recipes, additive injection and ablation, sweep,
run, analyze, and MLX LoRA. It is greedy-only for measured runs, and every
method added since August is Python-first with Swift reading the artifacts.

The greedy-only rule is ours, not MLX's. The pinned libraries
(mlx-swift 0.31.4, mlx-swift-lm 3.31.3) expose `MLXRandom.RandomState(seed:)`
and `withRandomState(_:body:)`, and the library samplers already draw inside a
per-sampler `RandomState`; their public initializers simply seed it from the
clock. The token iterator has a public initializer that accepts any
`LogitSampler`. Our runner never passes one; it builds `GenerateParameters`
with a temperature and takes the library default.

## 1. Work packages

### WP-P1 — Seeded sampling on MLX (removes the greedy-only rule)

**Change.** Add a Swift sampler type in SteeringKit that holds a
`MLXRandom.RandomState(seed:)` and samples inside `withRandomState`, with the
same top-p, top-k and min-p filter order as the library sampler so the two are
interchangeable at temperature. Pass it through the token-iterator initializer
that takes a sampler, from every local measured-run call site:
`ExperimentTasks.swift` (the study runner's `generateTokens` path and the
reasoning-budget path), the variant/saved-agent path, and `MultiAgentRunner`.
The Playground and Chat paths may keep the library default; say so in code.

**Seed contract.** Mirror the Python engine exactly: one seed per record,
derived the same way the Python runner derives it from the study's declared
seeds and the (condition, prompt, sample index) triple, so a record's seed
reads the same on both engines. Stamp the seed as live, not inert. Find every
place that currently stamps or explains inertness and update it in the same
change: `ExperimentTasks.swift` (the local sampling gate, its refusal text and
the `inert` stamping), `SubstrateRouting.swift`, `StudySamplingControls.swift`,
`InfoPopover.swift`, `CLIHelp.swift` (the temperature help), the CLI runner's
"server substrate only" sampling line, `AgentContract.swift` and its generated
`docs/AGENTS-WORKSPACE-DRAFT.md`, `docs/GENERAL-INTRODUCTION.md` ("one sampling
rule surprises people"), and `docs/CONDUCTING-A-STUDY.md`. Leave no sentence
that says MLX cannot pin a seed.

**Determinism is a separate claim.** A seed fixes the draw, not the logits.
Before the rule is lifted, run the same prompt, model and seed ten times at
temperature 0.7 with the study's prefill step size and require identical token
sequences; run it at two prefill step sizes and record whether they differ.
If Metal is not bit-stable for a model family, the rule is lifted only for
families that pass, and the runner refuses the others by name with the
measurement cited. Store the measurement under `docs/` with the model
revision, the MLX versions and the machine.

**Gates.** Swift suite green; the sampling-policy tests in ExperimentKit
rewritten to assert the new contract, with one test proving two runs with the
same seed produce the same records and two seeds do not; the CLI reference
regenerated; the Swift parser census and envelope lists unchanged unless a
flag is added.

### WP-P2 — MPS qualification of the Python engine

**Change.** No product code unless a defect is found. Produce a qualification
record: for each shipped extraction recipe, additive injection, ablation, one
battery, one stability diagnostic, one OptVec training run, one J-lens import
and qualification, and one Gemma Scope import, run the smallest model that
exercises the path on an Apple silicon Mac through the Python engine, and
compare against the same run on CUDA at matching revision, dtype and seeds.
Report per path: identical, within a tolerance the existing tolerance ruling
already defines, or divergent with the first divergent tensor named.

**Known hazards to test on purpose, not discover.** The attention-kernel
selection already forces `eager` on MPS in `model_loader.py` because SDPA on
MPS ratchets memory with context shape; measure the long-context memory curve
and confirm the fallback holds at the study's largest context. The J-lens
reference-agreement step promotes the reference head to fp32; confirm MPS
honors that and that bf16 matmul on MPS does not change the qualification
verdict. Count the test-suite `mps` branches (there are dozens) and make sure
each is exercised on a real MPS device at least once, since today they run
only on CUDA or CPU.

**Gates.** Python suite green on the Mac with the engine on MPS, not only on
CPU; the qualification record committed under `docs/` with a table per path;
any defect fixed on its own commit with a regression test, never inside the
qualification commit.

### WP-P3 — Swift twins owed by existing contracts

These are already promised by the documents that own them. Each is its own
commit with the contract document updated to say "landed" and the AST or
parity test that proves the port.

- **Extraction stability** (`experiment extract-stability`, direction
  stability by layer): port `vector_math.direction_stability` and
  `stability_by_layer` verbatim into `SteeringVectorMath.swift`, which is
  already documented as its 1:1 twin, and add the CLI verb on the Mac.
  `docs/EXTRACTION-RECIPES.md` §Swift parity names this as owed.
- **Intervention scope descriptor and sidecar**: port the module-level
  constants in `steering/intervention.py` (`PATHS`, the centering vocabulary)
  verbatim, add a Swift `InterventionScope`, and have Mac runs write
  `intervention-scope.json`. `docs/INTERVENTION-SCOPE.md` "Owed on the Swift
  side" is the contract. The trainable path stays server-only.
- **RepE reader `finalTest` role**: the Swift reader must stop treating every
  non-train row as held out. `docs/REPE-IMPLEMENTATION-BRIEF.md` names it.
- **Adapter-scale wire key and the evidence-role sidecar stamps** from the
  science follow-ups: stamp the same keys from the MLX LoRA path so artifacts
  from both engines carry the same identity fields.

### WP-P4 — Substrate boundary statement (documentation only)

One page, `docs/SUBSTRATES.md`, that says for every operation in the science
catalog which substrate produces it, which merely reads it, and why. J-lens
and SAE are "invalid on MLX by substrate, runnable on MPS pending WP-P2";
managed methods are "Python only, Swift reads". Generate the table from the
catalog rather than hand-writing it, and gate it with the existing science
resources check so it cannot drift.

### WP-P5 — Mac runner executor (only if Macs are a target)

Today a second Mac can serve one job at a time over HTTP with no queue, no
fan-out and no site profile type. If a Mac cluster is wanted, this is a new
executor beside `local` and `slurm`, a site profile kind for it, and a queue.
Do not start this without the researcher's explicit go; it is the largest
item here and changes the workspace-executor binding.

## 2. Order and dependencies

P1 and P2 are independent and can run in parallel. P3 items are independent
of each other and of P1. P4 follows P2 so its table states measured facts.
P5 waits for a ruling. Recommended order for one team: P1, then P3's stability
and scope twins, then P2, then P4.

## 3. Rules that apply to every package

- Verify against the pinned library versions in `Package.resolved`; do not
  bump MLX to get an API, and say so if a bump is the only way.
- Any change to shipped Python or seed files: run
  `scripts/ci/check-python-client-identity.py --write` and commit the
  regenerated Swift constant with the change.
- Any AST-pinned owner you must legitimately change: re-pin the audit's base
  in the same commit and say why in the message; never edit an owner and leave
  the audit pointing at the old base.
- Public repository hygiene: no site, host, person or username words, no
  home-directory paths, in any file or commit message.
- Every claim in a handoff must be something a reviewer can rerun: name the
  command, the tree, and the counts.
