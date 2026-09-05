# SteerLab — A General Introduction

*The one-document synthesis: what this instrument does, the method behind it,
the discipline wrapped around it, and how a study runs end to end. Written for a
researcher deciding whether SteerLab fits their question; no machine-learning
background assumed. See [README.md](../README.md) for install and orientation
and [ONBOARDING.md](ONBOARDING.md) for the hands-on tour; each section below
ends with a pointer to the deeper document.*

---

## 1. What this is

SteerLab is a **workbench for concept steering** in open-weight language
models. Rather than prompting a model and reading its answers, it reaches
*inside* the model and adjusts its internal state directly — a measured dose of
some named concept added to the model's working state while it performs a task,
without one word of the prompt changing — and then measures what moved.

Three verbs, and nothing else in the repository is more important than making
them trustworthy:

1. **Extract** a direction inside the model corresponding to a concept you
   name, derived from text you write.
2. **Inject** that direction during generation, at a layer you choose and a
   strength you can report in comparable units.
3. **Measure** what changed — against a paired unsteered baseline, with random
   controls, capability checks, and effect sizes.

The steering core is **concept-agnostic**. It has no idea what your concept
means: concepts, stimulus texts, task prompts, rubrics, and scoring vocabularies
are *data you author in a workspace*, and no concept is named anywhere in the
engine. If a change to the core would not work equally for an arbitrary concept,
it is a bug. You bring the question; SteerLab supplies the apparatus and the
discipline.

Everything else — two compute engines, a native Mac application, a provenance
system, statistics machinery — exists to make that one manipulation *checkable
by someone who was not there*: measured in known units, validated before use,
controlled against placebos, and frozen before outcomes are observed, so a
skeptical reader cannot say the effect was built in by the experimenter.

---

## 2. The question it is built to answer

The instrument-level question, stated once:

> When you push a named concept direction into a model's internal state while
> it performs a task, does the **decision** move — or only the **prose**?

That question is worth an instrument because the two can come apart, and the
difference is the whole point. A model can produce output saturated with a
concept while the choice it makes is untouched; it can also make a different
choice while its language stays unremarkable. The second case is the
interesting one, and the harder to catch, because every surface indicator says
nothing happened. The class of questions this apparatus answers, generally:

- **Does surface behavior reflect internal state?** If the output register
  shifts while the answers do not — or the reverse — the relation between what a
  model says and what it does becomes measurable rather than assumed.
- **Are a model's task dispositions manipulable along interpretable
  directions?** Not "can it be prompted into a persona", but: is there a
  direction in the model's own representation space that, dosed, moves what it
  decides — coherently, dose-responsively, reversibly under a negative dose?
- **How does an intervention compare across concepts, layers, and model
  families?** With a fixed strength denominator and identical recipes, "concept
  A moves this endpoint twice as much as B" and "layer 18 steers where layer 30
  does not" become comparisons rather than anecdotes.
- **What survives the controls?** A random direction of the same magnitude, a
  capability battery under every condition, and a paired unsteered baseline are
  the difference between a finding and a perturbation.

**Three layers of result stand on their own**, and it is worth knowing which
one you are producing:

1. **Model-internal variation.** Does the model have coherent, dose-responsive
   internal states at all? Does an induced concept move the outcome endpoint
   *and* the reasoning style *and* the surface vocabulary as one stance, or is
   it noise? No external data required.
2. **A human-anchored comparison, optionally.** Where a credible published human
   effect size exists for the same manipulation, a human-baseline table can be
   pinned by hash like any other input and the analysis reports the model's
   shift against it — a residual saying whether the model's response is
   human-shaped, muted, amplified, or unrelated.
3. **Propagation through groups.** Seat one conditioned agent in a multi-agent
   panel and measure whether deliberation corrects it or the disposition
   spreads — error correction and covert influence in multi-agent systems.

*Deeper: [RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md) — what each layer
can claim, and what gates it.*

---

## 3. The method, gently

### 3.1 What "inside the model" means

A language model is a stack of a few dozen similar processing layers. As it
reads and writes text, each layer passes forward a long list of numbers — a
*vector*, thousands of numbers wide — representing everything the model
currently has in mind. That flowing vector is the **residual stream**: the
model's working state, updated token by token. The empirical discovery
underlying this whole research family is that human-meaningful concepts often
correspond to *directions* in that space. There is, approximately and
measurably, a formality direction, a French-language direction, a fear
direction.

### 3.2 Finding a direction (extraction)

The baseline recipe, **Contrastive Activation Addition (CAA)**, is disarmingly
simple. Run texts that express the concept and matched texts that do not through
the model, record the internal state for each, and take

```text
vector = mean(positive activations) − mean(negative activations)
```

per layer. The worked example that ships is **formality** — the register
difference between "I regret to inform you" and "bad news, sorry" —
domain-neutral, easy to judge by eye, cheap to validate: a good first concept
and a bad final one.

Other recipes are selected as *data*, not code: a **grand-mean or
designated-reference** contrast (concept mean minus the mean over a deliberately
authored reference corpus), **paired-difference PCA (RepE-inspired)** over
difference vectors, trained linear **probes**, and **imports of independently
discovered features** (Gemma Scope). Running two recipes over the same stimuli
is a cross-check: no result should depend on one recipe's quirks.

Two design rules dominate extraction, and both are enforced as process:

- **Content matching.** The concept texts must differ from the comparison texts
  *only* in the concept — same topics, lengths, register otherwise — or your
  "formality vector" is really a topic-or-length vector.
- **Independence from the outcome (the circularity constraint).** The stimuli
  must not contain the vocabulary or the situations you intend to measure. If
  the texts already discuss the behavior you will test, a shift proves only
  that the model noticed. This is the firewall at the level of raw text, and no
  downstream statistic repairs a violation of it.

### 3.3 Injecting it (steering)

During generation, at a chosen layer, the instrument adds a small multiple of
the concept vector into the residual stream — at the final position of the
prompt and again at **every generated token**. Injecting only while the model
reads the prompt, and not while it writes, is the classic silent implementation
failure: it produces near-null results that look like findings. Off-the-shelf
model implementations expose no hook into the residual stream at all, so
SteerLab **vendors** the model files and adds one intervention point between
transformer blocks, on both engines; the per-token guarantee is permanently
unit-tested and re-asserted by a smoke test before study runs.

The dose is the scalar **α**. Raw α is meaningless across models — one model's
internal numbers can run a hundred times larger than another's — so doses are
expressed in **norm units**: fractions of the typical size of the model's own
activity at that layer, measured on a pinned neutral corpus. That fixed
denominator lets α ≈ 0.3 mean roughly the same push across concepts, layers, and
model families. Dose matters scientifically: a real effect should grow with α,
reverse when α is negative, and appear *below* the dose at which coherence
degrades — otherwise you have measured damage, not disposition.

A scope note worth stating early: in the default design the model reads the task
input *unsteered*, and the intervention runs while it generates. That is the
cleaner covert design, but it narrows what a null result proves — "steering did
not move the endpoint" is then evidence about the generation phase only, and a
prompt-span steering arm is the natural robustness check. The inverse operation
ships too: **ablation**, removing a direction by projection ("the model cannot
represent this here") rather than adding to it.

### 3.4 Knowing the vector is real (validation)

Before any behavioral use, a direction must earn its place:

- **Convergent validity.** It must correctly classify *held-out* scenarios that
  evoke the concept **without ever naming it**. A vector that merely memorized a
  keyword fails here.
- **Discriminant validity.** Its similarity to every other concept's vector is
  computed and reported. Distinct concepts must not collapse into one direction,
  or every "separate concept" result is one axis wearing different labels. With
  a single concept attached, the tool reports that the measurement was *not
  made* rather than a comforting number.
- **Placebo floor.** A **random direction of identical magnitude** at the same
  layer is carried through the design as a first-class arm. If noise moves the
  endpoint as much as your concept does, the finding is about perturbation.

### 3.5 Measuring what moved (instruments)

The output side has three tiers of instrument:

- **Answer-token probabilities** — the workhorse for categorical outcomes. For a
  task with fixed options the instrument reads the model's *probability
  distribution* over them directly: no sampling, no temperature, no parser, and
  sensitive enough to register a shift in leaning long before the discrete
  answer flips.
- **Deterministic parsers** for structured outputs: categorical choices, numeric
  quantities, derived endpoints — with parse failures preserved as data, because
  a parser failing at a rate correlated with the intervention is a confound.
- **Judge models** for qualities parsers cannot reach. Every steered output is
  paired with its same-item baseline; a judge sees the two side by side — order
  randomized, condition never revealed — and scores them under a version-pinned
  rubric, with at least two genuinely distinct judges and agreement statistics.
  A second mode codes each response individually against a declared field
  schema, reporting per-field agreement (Cohen's κ for categorical fields).

Alongside those, two surface measures that are diagnostics rather than outcomes:
**marker density** (how often the concept's own vocabulary appears) and a pinned,
hashed **reasoning-style taxonomy** (deterministic text features scored at
metrics time, no model involved). Both sit *beside* the outcome endpoint, never
instead of it — which is what makes "the decision moved and the prose did not" a
measured contrast rather than an impression.

The statistics follow: every comparison is **paired** to the same item's own
baseline, effect sizes carry bootstrap confidence intervals rather than bare
p-values, and multiplicity is corrected by phase — false-discovery-rate for a
broad screen, Holm for confirmation on held-out items.

### 3.6 The phenomenon to watch for

The Anthropic emotion-concepts paper found that injecting *desperation* raised a
model's rate of reward-hacking on a task roughly fourteenfold **with no visible
emotional language in its output**. Behavior moved; prose did not.

That result sets the measurement problem this instrument is built around. The
deepest finding it can produce is a decision that shifts under a covert concept
vector while the output's language stays unremarkable — which is why behavior,
reasoning style, and vocabulary are measured *separately on the same
generations*, and why concept vocabulary is never the criterion for tuning doses
in a study whose claim is about outcomes.

*Deeper: [METHODS.md](METHODS.md) — the math, the source lineage (Zou et al. on
representation engineering; Rimsky et al. on CAA; the Anthropic
emotion-concepts paper), and every divergence from those sources, logged.*

---

## 4. The instrument itself

**Two engines, one artifact model.** SteerLab is implemented twice, on purpose:
a **native macOS application and CLI** (Swift, Apple's MLX framework) — the
interactive research cockpit for authoring concepts, watching a dose-response
sweep by hand, chatting under steering, and iterating fast — and a **Python
engine** (PyTorch / Hugging Face) for Linux/CUDA hardware, with an HTTP API and
browser workbench, built for unattended batch campaigns: scheduler submission,
durable jobs, evidence bundles.

They are not wrappers around each other. They share one *artifact model* — the
same stimulus files, manifests, vector formats, and run directories, pinned by
identical SHA-256 hashes — but each implements extraction, injection, and the
freeze lifecycle natively. Activations from Metal and CUDA never byte-match, so
**vectors are re-extracted and re-validated on whichever engine a study runs
on**. What is identical is the *structure*: input hashes, manifest and run
schemas, the JSON output contract, prompt-render and tokenization goldens, and a
`vectors compare` verb on both sides that checks committed parity fixtures.

**Headless first, and everything is an artifact.** Every operation that could
appear in a paper runs from the command line; the app calls into the engine,
never the reverse. Concepts, corpora, vectors, model variants, experiments, and
runs are durable objects with provenance sidecars — who made them, from what
inputs (by hash), on which engine, when — and every operation writes one
**immutable run directory**, never overwritten, carrying enough to rebuild the
analysis without re-running the model. Study data lives in a **workspace**: a
self-contained, git-versioned folder, separate from the software checkout.

**Conditions and agents.** A named steering configuration is a *condition*
(several slots in one literally *are* the linear mix `h + Σ αᵢ·vᵢ`). A tuned
intervention can be packaged as a durable **agent**: base model, prompt
settings, the vector at its chosen layer and dose. Agents are minted by a formal
*promotion* step from a dose-selection sweep and carry a birth certificate
recording how they were selected; hand-made agents remain legal for exploration
and surface as advisories in any study that uses them.

**Three intervention depths.** Vector injection is the lead modality, but the
same nominal concept can be instantiated as a **fine-tuned adapter** (LoRA on
both substrates) or as a **system prompt** — three depths of the same
manipulation, all first-class conditions. Whether they produce the same
behavioral fingerprint is itself a research question, with a safety edge:
prompts are auditable by inspection; injections and adapters are not.

**A reading channel, alongside the writing channel.** The **RepE reader** of Zou
et al. is implemented in full on both engines, and it is a *reading* instrument
rather than a fourth spelling of extraction: a hashed task template, the LAT
token at the rendered scaffold's final position, mean-centred PCA over paired
differences, both of the paper's contrast constructions, and a sign and a layer
chosen on a held-out split — the layer stamped as a recommendation, never as a
silent selection. A reader-derived direction can be converted into a steering
vector through an explicit, provenance-stamped conversion, and refuses α until
its norm denominator is backfilled.
[REPE-IMPLEMENTATION-BRIEF.md](REPE-IMPLEMENTATION-BRIEF.md) itemises what is
faithful to the paper and what departs from it.

The Python engine also
carries a family of **lens instruments** (`jlens` — server-only, the imported
lens artifacts being PyTorch-native; any model with a published lens, some forty
at the time of writing): acquire and convert a lens,
decompose a steering vector into the vocabulary it is made of, probe a prompt
for per-layer readouts of what the model is poised to verbalize, derive a
steering direction from the lens, report across conditions. Useful as an
injection-took-hold check and a mechanism-level cross-check — and, like every
instrument here, qualified per model and substrate before its output is
evidence.

*Deeper: [Server/README.md](../Server/README.md) — the Python engine's design
and the cross-engine artifact contract.*

---

## 5. The circularity firewall — preregistration as mechanism

The methodological heart of the project: the lifecycle is built so that
**analytic choices are settled before outcomes are observed**, the way a
preregistered protocol is supposed to work — except enforced by software rather
than promised. An **experiment** is a manifest that pins every input *by
cryptographic hash*: the concept stimuli, the extraction recipe, the neutral
corpus, the task prompts, the judge rubric, the marker lists, the sweep dev
split, the battery, any human-baseline table. It holds a **recipe, not a
result** — vector bytes are never the artifact of record; runs re-derive them
deterministically from the pins.

**Freezing** is a one-way act. It re-verifies every pin against the bytes on
disk, snapshots all pinned inputs into the experiment folder, stamps the
manifest's content hash and the workspace's git commit, and makes the manifest
read-only. After freeze you do not edit — you **duplicate**, visibly. If a
pinned file is altered afterward, every subsequent operation refuses: drift
surfaces as a *violation*, never as a quiet change.

Pin verification always runs and is never skippable. On top of it, seven
evidence gates apply:

| Gate | What it demands |
|---|---|
| `revision` | a pinned, immutable model commit — "the model" is not a moving target |
| `measurementPins` | the inputs that determine what is measured are present and valid |
| `validateEvidence` | a `validate` run matching the exact pins, on this engine, that is not vacuous |
| `variantValidity` | attached variants carry hashed weights and a pinnable dataset manifest |
| `batteryEvidence` | each variant condition has scope-matched capability-battery evidence |
| `judgeValidity` | a rubric *file* and at least two genuinely distinct judges |
| `gitClean` | every pinned input is committed in the workspace repository |

A refusal is typed and names its repair. `--force` skips the seven gates — never
pin verification — and is **loud and permanent**: every skipped-and-failing gate
is stamped into the frozen manifest by name, so a forced freeze stays
non-citable by stamp rather than by memory.

Downstream the same discipline continues. Runs stamp the manifest epoch they
came from, and judging and statistics refuse to interpret a run against a
different epoch (per-engine, so analyze a run on the engine that produced it). A
screening-then-confirmation design must prove its confirmation items are
disjoint from those used during screening.

*Deeper: [CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) — the firewall in
practice, with the current limitations and their workarounds.*

---

## 6. How a study actually runs

A program runs as a staged funnel — cheap and broad first, expensive and
confirmatory later:

| Phase | What happens | Where |
|---|---|---|
| 0 · Shakedown | Prove the instruments end to end on small development models; run the standing smoke tests | Mac |
| 1 · Author data | Concept corpora, never-named validation sets, the independence screen, runnable task items, batteries | (research work) |
| 2 · Screen | Many concepts, coarse doses, one sensitive endpoint; FDR correction; promote the movers | Mac or server |
| 3 · Confirm | Promoted movers only, held-out items, fine dose grids, the full control matrix, a second model family | server for anything sampled |
| 4 · Triangulate | The same concept as vector / adapter / prompt; alternative extraction recipes; feature cross-checks; human-anchored residuals where a baseline exists | mixed |
| 5 · Panels | Multi-agent deliberation with 0/1/2/3 conditioned seats; spillover and transmission | server |

One pass through the middle of the funnel, concretely:

1. **Create** an experiment against a pinned model *revision*, then **attach**
   concepts — pinning the current hash of every stimulus file and the extraction
   options.
2. **Pin** the measurement side: task prompts, rubric and judges, outcome
   instruments, the sweep's selection objective.
3. **Extract** — derive the vectors; deterministic, so anyone with the pins
   re-derives the same bytes on the same substrate.
4. **Validate** — the never-named scenario test, the cross-concept similarity
   matrix, and a capability check. This is the evidence freeze will demand.
5. **Sweep** — a layer × dose grid on *development* prompts, never the real task
   items, scored under a **declared selection objective** that lives in the
   manifest as data. Where the claim is about outcomes rather than prose, that
   objective is behavioral (a judge-scored or probability-shift criterion),
   explicitly never vocabulary presence.
6. **Promote** — mint the agent from the winning cell, birth certificate
   attached. A manual override is permitted, loud, stamped, and still requires
   evidence that a sweep ran.
7. **Freeze** — the preregistration moment (§5).
8. **Run** — generate the full condition matrix: baseline, each treatment at two
   or more positive doses and a negative dose, the matched-norm random placebo,
   any positive control, and the capability battery under every condition.
   Deterministic runs execute locally; anything needing repeated stochastic
   samples runs on the Python engine.
9. **Evaluate** — blinded paired judging or per-response coding, where free text
   is an endpoint.
10. **Analyze** — paired effect sizes with bootstrap confidence intervals, FDR
    or Holm correction by phase, dose-monotonicity, the human-anchored residual
    table where a baseline is pinned, and the promoted-movers artifact that
    feeds the next phase.

Then duplicate, expand to a confirmation design on held-out items, and repeat
under stricter correction — with a second model family as the robustness check.

**One sampling rule surprises people.** Local Swift/MLX measured runs are
**greedy-only**: the runner requires `temperature == 0` and a single seed,
because the MLX generator cannot yet pin a per-run sampling seed. A local run
records its nominal seeds but stamps them inert — never read one as causally
meaningful. Stochastic studies belong on the Python engine, which seeds per
record and writes one record per condition, prompt, and sample index.

*Deeper: [CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) (the working rulebook);
[ONBOARDING.md](ONBOARDING.md) (commands, a first session).*

---

## 7. What is in the box

- **Two engines over one artifact model.** `steerlab-cli` (Swift/MLX, Apple
  silicon) for authoring and local runs; `steerlab-server` (Python/PyTorch/
  Hugging Face) for CUDA hardware, larger models, and stochastic multi-sample
  work. Both read and write the same manifests, vectors, and run directories.
  The bare name `steerlab` is neither of them — it is the cross-platform Python
  client, a third product with its own smaller verb surface.
- **CLI-first, headless by contract.** Every paper-relevant operation is a
  command-line verb — the whole study path on both engines. The app and the
  browser workbench are clients of the same machinery.
- **An agent surface.** Every workspace is created with an `AGENTS.md` — the
  lifecycle in order, the file shapes, the freeze gates with their repairs, what
  not to do — and it stays current: its header carries a hash of the body, so an
  untouched contract is refreshed to the shipped text on workspace open and on
  every CLI verb, while one you have edited is yours and is left alone. Every
  study-path verb accepts `--json`: one envelope on stdout,
  diagnostics on stderr, sorted keys, complete hashes, a typed `error` with a
  `repairAction`, and exit codes separating malformed invocation (64), gate
  refusal (65), not found (66), failure (70). Advisories never change one.
- **Workspaces.** `workspace init` creates a plain folder with `prompts/`,
  `experiments/`, `runs/`, and `adapters/`, its own git repository, and a
  **neutral instrument seed**: capability batteries, a neutral corpus, sweep dev
  prompts, judging rubrics, a parser registry, and templates for every study-data
  file you will need. It is deliberately concept-empty — the concept is what you
  are studying and arrives by your decision. `SampleWorkspace/` is a recipe-only
  worked example (the formality concept, task prompts, a battery, a neutral
  corpus, a default rubric) with no vectors, no runs, no frozen manifest.
- **The freeze firewall,** with hash-pinned inputs, a one-way gated freeze,
  stamped forcing, immutable runs, epoch guards on analysis, and a
  data-readiness checklist (`data check`) that names the file you still owe and
  refuses to call a study ready without it.
- **Measurement instruments:** sampled text with deterministic parsers,
  answer-token log-probability and choice probability for categorical outcomes,
  blinded paired judging and per-response coding under hash-pinned rubrics with
  inter-judge agreement, per-condition capability batteries, marker density,
  pinned reasoning-style taxonomies.
- **Statistics,** headless on both engines: paired bootstrap confidence
  intervals, Wilcoxon tests, BH-FDR for screens and Holm for confirmations,
  dose-monotonicity, control margins, and human-anchored residuals where a
  baseline is pinned. `analyze` writes `effect-sizes.csv` and folds the numbers
  into `report.json`.
- **A cluster path.** Saved site profiles, an SSH bootstrap that installs from
  the committed platform dependency locks, durable jobs with checkpoint/resume
  and fan-out, submission preflight, evidence bundles for engines that cannot
  see your workspace, and `site qualify` — nine structural checks asking whether
  a freshly provisioned node reproduces the contracts a result depends on.
- **Security posture in one sentence:** both servers are single-researcher
  instruments that bind loopback and reject cross-origin requests, and the
  Python engine requires a bearer token by default on every platform — read
  [SECURITY.md](../SECURITY.md), limitations included, before exposing anything.

---

## 8. If you want to see it with your own eyes

The visceral demonstration takes about half an hour on a Mac and needs no study
apparatus. Load a small 4-billion-parameter development model and run the
bundled toy configuration (`prompts/configs/toy-french.json`): in minutes it
extracts a French-language concept vector and verifies both that it moves
generation into French and that an equal-magnitude random vector does nothing.
Then open the app's steering surface, enable the vector, and slide the dose:
watch ordinary English answers acquire French, become French, then — past the
coherence cliff — dissolve. Slide it back. That dose-response curve, felt by
hand, is the entire method in miniature; everything else here is the discipline
required to turn it into evidence. *([ONBOARDING.md](ONBOARDING.md) §5 walks it
through.)*

---

## 9. Reading map

| Read | For |
|---|---|
| [README.md](../README.md) | Orientation, requirements, install, quickstart |
| [ONBOARDING.md](ONBOARDING.md) | **Start here to use it.** The hands-on tour: first vector, first experiment, first freeze |
| [METHODS.md](METHODS.md) | The math, the source lineage, and every logged divergence from it |
| [REPE-IMPLEMENTATION-BRIEF.md](REPE-IMPLEMENTATION-BRIEF.md) | What is actually implemented from Zou et al.'s Representation Engineering: pipeline, schemas, and the faithful-vs-departure table |
| [CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) | The working rulebook: lifecycle, artifacts, current limitations and their workarounds |
| [RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md) | What each layer of result can claim, and what gates it |
| [CLI-REFERENCE.md](CLI-REFERENCE.md) | Every verb, flag, default, and refusal on both command lines |
| [SECURITY.md](../SECURITY.md) | Threat model, auth posture, known limitations, disclosure |
| [Server/README.md](../Server/README.md) | The Python engine: design, cross-engine artifact contract, deployment |
| [SampleWorkspace/README.md](../SampleWorkspace/README.md) | The worked example workspace, and what it deliberately omits |
| Your workspace's `AGENTS.md` | The contract to hand a coding agent (generated at workspace creation, and refreshed while it stays unedited) |

### A note on models

SteerLab vendors two model families so any finding can be checked against a
second architecture. **Qwen3** is Apache-2.0; keep the reasoning effort `off`
(`set-sampling --reasoning-effort off`, the default) for measured runs unless
you are studying it, since reasoning traces change output structure and break
parsers — and when you do study it, declare the reasoning block's own token
cap (`--reasoning-max-tokens`) so it cannot eat the answer's budget. **Gemma 3** has no system role in its chat template (system
text is prepended to the first user turn, which affects prompt design and
prompt-set hashing) and needs care about double-BOS when templates meet manual
tokenization; its license is custom rather than Apache, with use restrictions
that plausibly pass through to derivatives, steering vectors you publish
included. Gemma's compensating asset is Gemma Scope, whose per-layer
sparse-autoencoder features let a derived direction be cross-checked against
independently identified ones. No weights ship here; read the license of what
you download.

---

## 10. Glossary

- **Open-weight model** — a model whose parameters are published, so its
  internals can be read and modified; closed API models expose nothing to steer.
  **Residual stream / activation** — the per-layer internal working state, where
  SteerLab reads and writes. **Concept vector** — a direction in that space
  corresponding to a nameable concept, derived from curated texts.
- **Steering / injection** — adding a dosed multiple of a concept vector at every
  generated token. **Ablation** — removing a direction by projection instead.
  **α (alpha), norm units** — the dose, relative to the layer's typical residual
  norm on a pinned neutral corpus, so doses compare across concepts and models.
- **CAA / grand-mean / designated reference / paired-difference PCA** — the
  extraction recipes: paired mean difference; concept mean minus a corpus mean
  or an authored reference class; the first principal component of per-pair
  activation differences (called `repeLAT` until the 2026-08-27 naming ruling,
  and still spelled that way in artifact bytes). **RepE reader** — the separate
  template-mediated *reading* instrument, not an extraction recipe.
- **Validation (never-named)** — held-out scenarios that evoke a concept without
  naming it; a vector must classify them to be used. **Matched-norm random
  control** — a random direction of identical magnitude: the placebo arm.
  **Capability battery** — unrelated probes under every condition; if they
  degrade, an apparent effect is impairment.
- **Experiment / manifest / pin** — the preregistration object: every input named
  by cryptographic hash, with the options used to derive from it. **Freeze** —
  the one-way, gated act that makes it read-only before outcomes are measured.
  **Run directory** — one immutable `runs/<timestamp>-<slug>/` per operation; the
  unit of reproducibility. **Advisory** — something you should know that did not
  stop the verb; never changes an exit code.
- **Condition (arm)** — a named, complete steering configuration; several slots
  in one *are* the linear mix `h + Σ αᵢ·vᵢ`. **Sweep → promote → agent** — dose
  selection on development data under a declared criterion, minting the winning
  cell as a reusable agent with a birth certificate.
- **Answer-token logprob instrument** — reads the probability over fixed answer
  options directly; deterministic and temperature-free. **Paired judge** —
  blinded side-by-side scoring of steered versus baseline output by two or more
  judge models under a hash-pinned rubric.
- **Marker density / reasoning-style taxonomy** — surface diagnostics measured
  beside the outcome endpoint, never as a substitute for it. **FDR / Holm** —
  multiple-comparison corrections (screening / confirmation).
- **Adapter (LoRA)** — a small trained weight modification: disposition by
  training, contrasted with disposition by injection. **Multi-agent panel** —
  several agents deliberating under a scripted protocol. **Workspace** — the
  folder with one project's prompts, experiments, and runs, carrying its own git
  history, separate from the code.
