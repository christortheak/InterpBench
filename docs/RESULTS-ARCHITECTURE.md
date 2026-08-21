# Results Architecture — what each result layer can claim

Revised 2026-08-18. A study run on SteerLab produces results at up to **three
layers**, and they are not the same kind of claim. This document says what each
layer asserts, what it needs first, and — the part easiest to skip and most
expensive to get wrong — what it can never license however clean the numbers
look. It describes claim shapes, not any particular question; where an example
helps it uses the register/formality family that ships with `SampleWorkspace/`.

Companions: [ONBOARDING.md](ONBOARDING.md) (first pass through the instrument),
[METHODS.md](METHODS.md) (the math and its sources),
[CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) (the order to do things in),
[CLI-REFERENCE.md](CLI-REFERENCE.md) (every verb and flag).

---

## 0. The three layers at a glance

| Layer | Headline question | Extra data needed | Reads from |
|---|---|---|---|
| 1. Model-internal variation | Does this model hold a coherent, dose-responsive internal state for the named concept, and what does it do to the model's decisions? | none beyond the study itself | `effect-sizes.csv`, `fingerprints.csv`, sweep dose grids, cosine matrices, `battery.jsonl` |
| 2. Human-anchored comparison | How does the model's response to the intervention compare to a *measured human* response on the same task family? | a transcribed human-effect table | `alien-residuals.csv` against the hash-pinned baseline CSV |
| 3. Propagation | How does an induced disposition in one agent move through a group of interacting agents? | a multi-agent scenario and its stripped baseline | `panel-effects.csv`, `panel-voice-lint.csv`, `turns.jsonl` |

**Layer 1 is the layer every study has.** Layers 2 and 3 are additions with real
costs: published human numbers you can defend transcribing, and many generations
per data point. One pipeline serves all three — the same frozen manifests and
the same immutable run records, differing only in analysis and framing.

### Intervention modality is a design axis, not an implementation detail

"Intervention" means several things at different depths, each a first-class
hashed condition. `analyze` tags every condition with its modality, derived from
the manifest and never guessed, so cross-modality comparison is a table rather
than an argument:

| Modality | Depth | Mechanism | What validates it |
|---|---|---|---|
| `injection` | activations | `α·v` added to the residual stream on every decode step | held-out probe validation, norm-unit α, matched-norm random floor |
| `saeLatent` | activations | a sparse-autoencoder latent edited in latent units | its own dose units — deliberately **not** pooled with `injection`, because the doses are not comparable |
| `adapter` | weights | a fine-tuned adapter trained on concept data | training-set hash + capability battery (there is no probe analogue) |
| `systemPrompt` | context | an instruction in the prompt | prompt-text hash; the auditable modality |
| `stacked` | mixed | two or more of the above in one variant | the union of the above, and the interaction is yours to argue |

`none` is the baseline arm. Whether the *same* nominal concept produces the same
behavioral fingerprint across modalities is itself a layer-1 claim (§1), and a
consequential one: prompting is the only modality an outside auditor can see.

---

## 1. Layer 1 — model-internal variation

**Claim shape:** "Under intervention X at dose α, this model's behavior on task T
moves in way Y — reliably, dose-responsively, and without capability collapse."
A claim about one model, on your task, in your conditions. The sub-questions,
and the endpoint that carries each:

- **Mean shifts** — the outcome endpoint's average movement against each item's
  own baseline. (`effect-sizes.csv`; paired bootstrap CI + Wilcoxon; BH-FDR
  across a screen, Holm within a confirmation family.)
- **Variance shifts** — does the intervention make the model *inconsistent*
  rather than biased? Per-item spread across `samplesPerItem` is an endpoint in
  its own right: leaving the mean alone while doubling the spread is induced
  unreliability, invisible to mean endpoints. Python engine only — local runs are
  greedy-only (ONBOARDING §7).
- **Dose shape** — monotone drift, threshold, or collapse. Promotion rules
  usually *require* dose-monotonicity, but a non-monotone shape is a result to
  report, not merely a gate to fail; collapse shows up as parse-failure rate and
  distinct-2 degradation before it reaches the outcome.
- **Cross-task coherence** — does one concept move several independent task
  families as a coherent stance, or as unrelated local perturbations? Coherence
  across tasks sharing no surface vocabulary is what licenses talking about an
  internal *state* rather than a prompt-shaped reflex.
- **Geometry ↔ behavior** — do geometrically similar directions (the
  cross-concept cosine matrix, produced at validation) produce similar behavioral
  fingerprints? `fingerprints.csv` (Python engine `analyze`) is the behavioral
  side, one row per condition × endpoint; correlating the two similarity
  structures is analysis you write, not a verb the engine ships.
- **Modality convergence** — same concept, different modality, same fingerprint?
  Convergence argues the concept is a substrate-independent state; divergence —
  prompting moving prose while injection moves the decision — is the
  depth-versus-surface distinction made measurable, and the more interesting
  outcome.

**What this layer requires** is nothing exotic: a frozen manifest, a validated
vector, a paired baseline arm, a matched-norm random control (`--control
randomMatchedNorm`), a per-condition capability battery, statistics paired to
each item's own baseline.

**What it can never license.** A layer-1 result is about *this model, at this
revision, on this task, under these prompts* — not about language models
generally, about the concept as humans use the word, or about what the model
would do unprompted. Two guards matter most. **Marker density is a manipulation
check, never an outcome**: selecting doses on how often the concept's own
vocabulary appears optimizes for surface style, the exact confound most steering
work falls into, so declare a behavioral sweep objective (`judgeScore`,
`logprobShift`) whenever the outcome is a decision rather than prose. And **a
concept that does not beat its matched-norm random control** has measured the
size of your nudge, not the meaning of your concept.

---

## 2. Layer 2 — the human-anchored comparison

**Claim shape:** "On endpoint E, humans move Δ_human and this model moves
Δ_model; the residual `R = Δ_model − Δ_human` places this concept in region
*r*." The Python engine's `analyze` writes one row per (condition, endpoint) into
`alien-residuals.csv` (the Mac app renders it but does not produce it): both
effects with their intervals, R with a conservative interval
`[Δm_low − Δh_high, Δm_high − Δh_low]` — exact when the two estimates are
independent, and they are — and a region label from a decision flow over those
intervals, not from a judgment call:

| Region | Meaning |
|---|---|
| `humanAligned` | both effects real, R's interval covers zero |
| `hyperHuman` | same direction as humans, credibly larger |
| `hypoHuman` | same direction as humans, credibly smaller |
| `alien` | humans do not move, the model credibly does |
| `inverted` | the model credibly moves against the human direction |
| `inertBoth` | neither moves |

**The scarce resource is the baseline, not the stimulus.** Producing model
effects is cheap; producing a *defensible* Δ_human is the whole cost of this
layer. Each concept should declare its inferential role before anything is
pinned — honest labeling here is what protects the eventual paper. **Anchored**:
a measured human effect exists for a task that genuinely corresponds to yours,
so R is quantitative. **Plausibly-null**: theory predicts Δ_human ≈ 0, making
the cell a detector for the `alien` region where any credible model movement is
the finding — but the null must be *argued in the preregistration*, never
assumed silently. **Directional-only**: a sign prediction with no defensible
magnitude, so R degrades to a sign/region check and should be reported as one.

### What an acceptable human baseline looks like

The engine does not know or care which literature your numbers come from. It
requires a *shape*, and refuses a file that lacks it:

| Element | What it must say | Why it matters |
|---|---|---|
| Direction | which way the effect goes on your endpoint's scale | the sign convention has to be written down and match the model side |
| Magnitude + interval | point estimate and published bounds | a number without an interval cannot enter the conservative R arithmetic — treat it as directional-only |
| Population | who was measured | a convenience sample and a professional sample are different anchors; the difference belongs in the write-up |
| Task correspondence | how close the human instrument is to the prompts you run | same instrument/same cells is the strong case; "a related paradigm on a related population" is a weak anchor to label, not quietly average in |
| Provenance | citation, table or figure, n, transcription date | so the next reader can re-check it without you |

Mechanically that is a CSV whose required columns are exactly what `analyze`
reads — `endpoint`, `deltaHuman`, `ciLower`, `ciUpper`, one row per
endpoint, `endpoint` matching the name your study measures — plus recommended
`source`, `n`, `notes`. Extra columns travel with the file; a missing required
one makes the pin refuse. Every workspace is seeded with the template and its
discipline note at `prompts/templates/human-baseline/`.

**The transcription rule (hard).** Every number entering an R computation comes
from a CSV pinned into the manifest as `humanBaseline` (path + hash), verified
by opening the source table — never from memory, an abstract, or a secondary
citation, and never written into analysis code. Drift in the pinned bytes after
freeze is a verify violation, exactly like drift in a stimulus file. Do not
commit the source paper's full text to your workspace; cite it in `source`.

**What this layer can never license.** R compares two estimates from two
populations measured by two instruments. It supports "the model's response is
larger / smaller / absent / inverted relative to the published human effect" —
not "the model is more or less *human*", not a claim about human decision
making, and not a claim about any endpoint the baseline table does not name. A
weak anchor produces a weak claim however tight the model side's interval is.

---

## 3. Layer 3 — propagation through multiple agents

**Claim shape:** "A disposition induced inside one agent propagates through a
group of deliberating agents in way Z." The broadest-reach layer — error
correction, capture, and covert influence in multi-agent systems — where the
group setting is the *setting*, not the claim.

**Implemented estimands** (`panel-effects.csv`, one row per endpoint), computed
from paired configured/baseline scenario runs where the baseline is the same
scenario with the interventions stripped:

| Estimand | What it is |
|---|---|
| `direct` | treated seats' shift against their own baseline turns |
| `spillover` | untreated seats' shift when sharing a panel with treated seats |
| `unexposed` | untreated seats speaking *before* any treated output could reach them — a placebo channel that should sit near zero |
| `group` | the shift of the designated group-outcome turn |
| `transmissionRatio` | spillover / direct — how much of the induced stance leaks through deliberation |
| `amplification` | group / direct — does the group amplify or damp the treated seat's shift? |

`unexposed` separates transmission from a common-cause artifact: seats that move
before they could have read anything a treated seat wrote make the spillover
number mean something else. Turns whose endpoint fails to parse in either
condition are dropped and counted, never coerced; `panel-voice-lint.csv` ships
beside the decomposition for every panel run.

**What the harness supports today.** Composition sweeps (all-baseline, each seat
solo-treated, denser cells) compile from a protocol template into ordinary
scenario files, so group-level dose-response — "deliberation corrects one
compromised member but is captured at two" — is a batch of castings rather than
hand-authored panels. A protocol template holds seats, turn script, routing, and
endpoints fixed while the shared materials change, and is refused by the
scenario decoder on both engines so an empty-record template can never run by
accident.

**Extensions worth the engineering, roughly in order:** aggregation-rule
comparison (private-vote-then-reveal vs deliberate-then-vote vs consensus), which
makes institutional design a manipulable variable; persistence, a per-turn series
on untreated seats after the treated seat stops contributing; detectability, a
judge or peer agent asked to identify the treated seat from the transcript (high
transmission with low detectability is the crisp result); and covert-influence
variants that constrain the treated seat's output channel.

**What this layer can never license.** Panels are expensive and the headline
estimands are *ratios of small effects*. Run layer 3 only on concepts that
survived a single-agent confirmation with the largest stable effects — otherwise
a transmission ratio is noise over noise, and it prints to three decimals
anyway. A ratio whose denominator's interval covers zero is not a number; report
the components instead.

---

## 4. Concepts come in clusters, and the cluster is the unit

A study rarely learns much from one concept. The design pattern is a **cluster**
spanning a relevance ladder *relative to your own task*: concepts your task's
norms say ought to move the outcome, concepts that plausibly should not,
concepts that certainly should not, and a matched-norm random direction as the
floor. The ladder turns "the model moved" into "the model moved for reasons the
task's own theory does or does not endorse" — a property of the pairing between
concept and task, not of the concept alone. In the register family that ships as
the worked example, a cluster might run formality, hedging, and verbosity against
a task whose outcome is a decision rather than prose: all three plainly move the
*style*, and whether any moves the *decision* is the question.

Each concept carries the same contract — content-matched stimulus set,
**never-named** held-out validation set, independence screen against the outcome
vocabulary, declared inferential role if layer 2 is in play. Each cluster runs its
screen phase as its own frozen experiment and carries its own floor and
plausibly-null cells, which are what make a positive result readable; the
deliverable is the same triple every time: a layer-1 internal map, layer-2
residuals for whatever cells are genuinely anchored, a layer-3 arm for survivors.

---

## 5. What gates a claim

Every layer rests on the same machinery, and that machinery is why a result is
checkable by someone who was not there.

- **Freeze is preregistration as mechanism.** Pin verification always runs and is
  never skippable; then seven evidence gates apply (`revision`,
  `measurementPins`, `validateEvidence`, `variantValidity`, `batteryEvidence`,
  `judgeValidity`, `gitClean`). `--force` skips the seven, warns for each
  skipped-and-failing one, and stamps `freezeForced` plus the gate ids into
  `forcedGatesSkipped`. A forced freeze stays non-citable — but **by stamp**,
  checkable years later rather than by your memory of why it seemed fine.
- **Measurement-side inputs are pinned too** — validation sets, marker files,
  neutral principal-component bases, rubrics, sweep dev splits and batteries, and
  the human-baseline CSV, all hash-enforced against the bytes on disk. Drift
  after freeze is a verify violation, never a silent change.
- **Epoch guards.** `analyze` and `evaluate` refuse a run whose stamped
  experiment hash differs from the live manifest's; `--allow-unverified-epoch`
  covers only unstamped legacy runs and stamps `epochUnverified`. The guard is
  per-engine — analyze a run on the engine that produced it.
- **Arms should trace to a rule.** A sweep selects a cell by a criterion declared
  in the manifest as data; `promote` mints an arm carrying a birth certificate
  recording how it was selected. Manual overrides are permitted, loud, stamped,
  and still require evidence a sweep ran; hand-created variants remain permitted
  but surface as freeze advisories.
- **Confounds are measured, not assumed away.** Capability batteries run per
  condition inside `run` (`conditions[].capabilityBattery`, `battery.jsonl`),
  because "the concept moved the decision" and "the model got worse at
  everything" otherwise produce the same table. Judges are pinned instruments:
  hashed rubric file, paired against the same item's baseline, at least two
  genuinely distinct judges — identity resolves to (kind, model, provider), so
  two local judges with blank model fields are one judge agreeing with itself.
- **The viewer never invents a statistic.** In the Results Explorer — the one
  canonical results surface — every number is **stored** (read from a run
  artifact, rendered plain), **derived** (computed by the viewer from stored
  records; badged, formula in the tooltip), or **heuristic** (derived *and*
  resting on a convention the data does not declare; amber badge naming the
  assumption). κ, confidence intervals, p-values, and multiplicity corrections
  render exclusively from engine artifacts, and a derived number never appears
  unbadged in a column beside a stored one.

---

## 6. Choosing your layers

Start at layer 1 and earn the others. A study reporting a clean model-internal
map — dose-responsive, above its random floor, capability intact, frozen before
it was measured — is a complete result. Layer 2 commits you to defending
somebody else's numbers; layer 3 to many generations for ratios of small
effects. Both earn their cost when the question needs them and are expensive
theater when it does not. Whatever layer you claim at, the claim should be
checkable from the run directory alone, by someone who does not have you
available to ask.
