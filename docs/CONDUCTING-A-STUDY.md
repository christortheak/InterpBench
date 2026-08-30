# Conducting a Study

The operating manual for a study whose result has to survive review — one level
deeper than [ONBOARDING.md](ONBOARDING.md), which gets you to a first frozen
experiment, and beside [METHODS.md](METHODS.md), which carries the math and the
validity rationale for each recipe. The subject here is the sequence of
**decisions** a study forces, in the order the lifecycle forces them, and what
each costs if you make it late. The command line is the source of truth: a
result you cannot reproduce headlessly is a result you cannot hand to anyone,
and the app drives the same engine through the same store (§8).

The running example is **formality** — the register difference between "I
regret to inform you" and "bad news, sorry" — the concept that ships in
[SampleWorkspace](../SampleWorkspace/README.md). Substitute your own; nothing
in the engine knows the difference.

Companions: [GENERAL-INTRODUCTION.md](GENERAL-INTRODUCTION.md),
[RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md) (what each layer of result
licenses you to claim), [CLI-REFERENCE.md](CLI-REFERENCE.md) (every verb, flag
and refusal), [Server/README.md](../Server/README.md) (the Python engine and
the cluster path), [SECURITY.md](../SECURITY.md).

Every Mac lifecycle command below is typed `steerlab-cli` — the Swift CLI's own
name. `steerlab` and `steerlab-cli` are **two products, not two spellings**:
the short name is the cross-platform Python client, with a smaller verb surface
(CLI-REFERENCE §1.4), and `steerlab-server` is the Python engine (§7 here).

---

## 1. The decisions, and where each is made

| # | Decision | Made at | Cost of making it late |
|---|---|---|---|
| 1 | concept family, task, endpoint | before any file exists (§2) | a confound no statistic repairs |
| 2 | stimulus texts, and their independence from the outcome | authoring (§2.2) | the whole result |
| 3 | which instrument measures the endpoint | `set-instruments` (§2.4) | prose parsing, and a parser confound |
| 4 | extraction recipe and reading position | `attach` (§4.1) | re-extract, re-validate, re-freeze |
| 5 | dose (layer × α), and how it was chosen | `set-sweep-grid` → `set-sweep-selection` → `sweep` → `promote` (§4.4) | selection on the confound you meant to rule out |
| 6 | the control set | `declare-condition` (§5.1) | an effect indistinguishable from perturbation |
| 7 | all of the above, jointly | `freeze` (§4.6) | the firewall itself |
| 8 | the substrate the measured run executes on | before `validate` (§6) | evidence that does not satisfy its own gate |

Everything from #1 to #7 happens **before any behavioral outcome is measured**.
That ordering is the discipline:

> You choose and freeze every setting — stimuli, recipe, layer, α, model
> revision, instrument, controls, judges — *before* you measure the behavior
> you will report.

Mechanically an experiment is a **recipe, not a vector**:
`experiments/<name>/experiment.json` pins its inputs by SHA-256 and its options
by value, and runs re-derive vectors deterministically into immutable
directories that stamp the manifest hash they ran against. Drift in any pinned
byte is a `verify()` violation, never a silent change. **Freeze before you run,
and iterate by `duplicate`, never by editing.**

---

## 2. Design before data

### 2.1 Concept family, task, endpoint

**The concept family.** A single direction is hard to interpret: you can show
it moves an endpoint, but not that the *direction* rather than any perturbation
of that size did it. Design at least the concept of interest, one or more
**control concepts** that should not move the endpoint, and — where the claim
admits one — a **positive-control concept** that plainly should, without which
a null on the concept of interest is uninformative rather than interesting.

**The task.** The measured task-prompt set is a pinned file. Write items a
person could score without knowing which condition produced them, keep their
subject matter away from the stimulus texts, and give every item a stable `id`:
the engine keys baselines by id, so a renumbered file is a different
measurement.

**The endpoint.** Not "does the output change" but a specific quantity — a
categorical choice, a magnitude on an ordinal scale, a judged preference
against the item's own baseline, a scored style feature. Decide it before
authoring the task, because the endpoint decides the item shape (§2.4).

### 2.2 Outcome-independence: the firewall as a design discipline

The freeze lifecycle is a mechanism, and a mechanism cannot save you from a
stimulus set that already contains the outcome. **The texts a direction is
extracted from must be independent of the outcomes it will be tested on.** If
the formality stimuli discuss the same situations the task prompts pose, a
shift in the endpoint proves only that the model noticed a topic; no pairing,
bootstrap, or correction repairs that, and a reviewer who can say "the vector
already knew about the task" has ended the study. Keep the concept's subject
matter, vocabulary, and scenario space disjoint from the task's.

Two corollaries. **Held-out validation scenarios must never name the concept**,
because they exist to show the direction detects the concept *without* its
vocabulary — a scenario saying "formally" tests a keyword. And **nothing
measured may be tuned on the measurement**: dose selection happens on a
separate dev split (§4.4), confirmation on prompts disjoint from the screen's
(§5.5).

A third rule governs the stimulus file itself: **content matching.** Positives
and negatives must differ in the concept and nothing else — same topics,
lengths, register otherwise — because a length or subject asymmetry *becomes*
the direction. `validate`'s per-stimulus margins show the mislabeled and
off-axis items; cut them and re-attach.

### 2.3 Spanning the relevance ladder

Where the question is whether behavior tracks *task-relevant* considerations or
is also moved by *task-irrelevant* ones, choose concepts to span that ladder
deliberately and record where each sits: concepts a competent procedure should
weigh, concepts plausibly persuasive but with no business bearing on the
endpoint, and concepts with no defensible bearing at all. Nothing in the code
knows a concept's rung — what the engine gives you is that each rung is an
ordinary attached concept with its own held-out probe, sweep, arm and paired
statistic, so comparing rungs compares like with like.

### 2.4 Declaring the instrument

Outcome instruments are **declared, never inferred**.

| instrument | what it measures | reach for it when |
|---|---|---|
| `sampledText` | generated prose, scored by metrics, judges, style taxonomies | the task is open-ended, or a rubric must read it |
| `answerTokenLogprob` | log-probability of each declared option's tokens | the endpoint is categorical — the default |
| `choiceProbability` | normalized probability across the option set | a categorical endpoint reported as probabilities |
| `ordinalScale` | a graded readout over an ordered option set | magnitude endpoints (`--ordinal-aggregation expectedValue\|argmax`) |
| `repeReaderScore` | a fitted reader's score on the generation | reading-instrument studies (METHODS) |

`repeReaderScore` is a **relative** endpoint: report it as a difference between
conditions, not against an absolute threshold. A template-pair reader
(`unsupervisedTemplatePair`) scores new text under its T+ instruction while its
probe was centered between T+ and T−, so its scores carry a constant positive
offset — one that cancels in a between-condition comparison and does not cancel
in the claim "this score is above zero, so the concept is present"
(`docs/REPE-IMPLEMENTATION-BRIEF.md` §1).

```bash
steerlab-cli experiment set-instruments formality-pilot answerTokenLogprob
```

**Categorical endpoints should prefer the log-probability instruments over
sampled prose**, for three reasons that decide the study's sensitivity: they
are *deterministic* (temperature-free, no sampling variance, no seed policy to
defend); they need *no parser*, and a parser failing at a rate correlated with
the intervention is a confound rather than a nuisance (§5.4); and they are
*more sensitive*, because a margin between options moves continuously while a
word count moves only when the model changes its mind. Items carry the options,
but the instrument remains an explicit declaration:

```jsonl
{"id": "item-01", "prompt": "…Answer with exactly one word: road or rail.", "options": ["road", "rail"], "target": "rail"}
```

`pin-prompts` raises a `choiceItemsWithoutInstrument` advisory when a file
carries options and no direct-scoring instrument is declared — the run will
then record prose and nothing else. Thinking modes must be off for
log-probability arms, and a manifest asking for both is refused.

**Two declarations that scope what an instrument reads.** Both are draft-only,
both exist on the cross-platform client under the same spelling, and both
*derive* their pin from a workspace file rather than taking it as an argument —
which is what makes them preregistration facts rather than settings. (A third
declaration of the same shape, `set-evaluation-sampling`, declares *how many*
records the judged coding reads; §4.7.)

```bash
steerlab-cli experiment set-parser formality-pilot <registry-entry-name>
steerlab-cli experiment set-instrument-scope formality-pilot label,json
```

`set-parser` names an entry in `prompts/parsers/parser-registry.json` and pins
that registry's SHA-256 alongside it. **The hash is never typed**: the
registry file is the authority on which parser *version* the study
preregistered, so what the study cites is a version, not a name, and
re-declaring the same name is also the drift repair. Declare it on any study
with a free-text numeric endpoint — §5.4 is why. Leaving it undeclared drops
the study onto a deprecated implicit selection that now announces itself
(`deprecatedImplicitSelection`), and an announced fallback is still a fallback.

`set-instrument-scope` declares which `responseFormat` rows the
option-consuming instruments read (`label`, `json`, `freeText`), pinning the
row set they select. It exists for the mixed prompt file: without it, a file
carrying both json and label rows meets a run-start refusal whose only other
repair is to drop the instrument for `sampledText` — a **lossy** repair that
quietly changes what the study measures. Declaring the scope keeps
`answerTokenLogprob`/`ordinalScale` on the rows that can carry them. A scope
selecting **zero** rows is refused at the declaration, hours before it would
have produced zero records.

**Judged semantic classification, not marker density, is the primary
instrument for a reasoning-style question.** The temptation with "did the
model reason differently?" is a word list: count how often the concept's own
vocabulary appears and call the difference an effect. That number is a
**manipulation check** — evidence the injection did *something* to the surface
— and it is diagnostic-only, for the same reason §4.4 refuses it as a
promotion objective: it optimizes for and measures surface style, which is
precisely the confound a claim about reasoning has to rule out. The primary
instrument is **judged classification against a hashed rubric**: a
per-response coding rubric (§3.2) declaring the categories, pinned by
`judgeRubricFile` + `judgeRubricHash`, coded blinded, with the agreement
statistics of §5.3 saying whether the coding is a measurement at all. Report
marker density beside it as the manipulation check it is, never in its place.

---

## 3. Data readiness

### 3.1 The workspace and its files

Study data lives in a workspace — a plain folder with its own git history,
never in the code checkout — born with **instruments only** (batteries, a
neutral corpus, sweep dev prompts, rubrics, a parser registry, a template for
every study-data file you will author, an `AGENTS.md` for a coding agent) and
deliberately no concepts.

```bash
steerlab-cli workspace init ~/SteerLab/Workspaces/formality-pilot
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/formality-pilot
```

| What | Where | Required for |
|---|---|---|
| concept stimuli | `prompts/concepts/formality/{positive,negative}.jsonl` | extraction |
| held-out probe | `prompts/concepts/formality/validation.jsonl` | `validate`, therefore `freeze` |
| marker vocabulary | `prompts/concepts/formality/markers.json` | the manipulation check |
| measured task | `prompts/tasks/<file>.jsonl` | `run` |
| neutral corpus | `prompts/neutral/corpus.jsonl` | norm-unit α (the denominator) |
| sweep dev split | `prompts/dev/dev-prompts.jsonl` | `sweep` |
| capability battery | `prompts/batteries/<file>.jsonl` | the degradation control |
| judge rubric | `prompts/rubrics/<file>.md` | `evaluate`, `judgeScore` sweeps |
| style taxonomy | `prompts/taxonomies/<file>.json` | reasoning-style endpoints |
| human-effect table | `prompts/baselines/<file>.csv` | the human-anchored layer only |

```jsonl
positive.jsonl    {"text": "…"}                            expresses the concept
negative.jsonl    {"text": "…"}                            matched, does not
validation.jsonl  {"text": "…", "expresses": true|false}   never names the concept
markers.json      {"words": [...]}                         diagnostic vocabulary
```

**Author `validation.jsonl` before you `attach`.** `attach` pins the file's hash
— or an explicit "absent" when there is none — so a probe set appearing
afterwards is a verify violation that blocks `validate` and `freeze` alike. The
order is **author → attach → validate → freeze**; found out late, author the
file, re-run `attach` for those concepts, then validate. On a frozen manifest
there is no repair — duplicate first.

### 3.2 Judging inputs, and the human-effect table

Two rubric modes, distinguished by the file itself. **Paired preference** (the
default) presents each steered generation beside its *same-item baseline*,
blinded and position-randomized, and the judge picks a winner;
`prompts/rubrics/default-paired-v1.md` ships as a generic starting point.
**Per-response coding** — a rubric opening with a `mode: perResponseCoding`
frontmatter block plus one `field: <name> <type>` line per declared field —
codes every sampled record individually and blinded, baseline included, with no
pairing and no winner, reporting per-field aggregates and inter-judge agreement
(percent, and Cohen's κ for categorical fields). That is the shape for "what
does each response *contain*" as against "which is preferred". Either way the
schema rides inside the rubric file, so the single `judgeRubricFile` +
`judgeRubricHash` pin covers it and drift refuses like any other pin.

A study anchoring model effects against measured human effects also pins a
transcribed CSV whose first four columns are required, read by the analysis
loader:

```csv
endpoint,deltaHuman,ciLower,ciUpper,source,n,notes
```

`endpoint` must match an endpoint your study measures. Transcription discipline
is part of the method: verify every number, CI, and n against the cited table
before committing, record the transcription date, never transcribe from an
abstract or a secondary citation, and cite third-party texts rather than
committing them. Pinning the table unlocks the human-anchored residual
computation; a study without it remains entirely valid at the model-internal
layer.

### 3.3 `data check` is the gate

```bash
steerlab-cli data check formality-pilot
```

One line per requirement, blockers first, each naming **the path you must
author** and why the requirement exists. It is manifest-driven, so it answers
for *this* study rather than in general.

```text
✓ [present]  stimuli / validation set / markers — formality
✗ [missing]  task prompts       prompts/tasks/formality-prompts.jsonl
· [optional] human-baseline CSV, reasoning-style taxonomy …
```

Blockers are a **refusal**: `state: "refused"`, exit **65** in both human and
`--json` mode. Run it early and often while authoring; the alternative is
meeting the same list at a failing freeze gate, after the authoring context is
gone.

For each row it names, `steerlab-cli authoring prompt <kind>` renders the
generation prompt that produces that file, with the kind's audit battery as
numbers: `contrastive-pairs` and `validation-set` for §3.1's stimuli and
held-out probe, `choice-prompts` for the `logprobShift` instrument,
`reader-pairs` for `prompts/readers/<c>/pairs.jsonl`, and `battery` for the
capability battery. The templates are workspace data in
`prompts/authoring-prompts/` — your copy wins over the shipped one, and each
emission stamps the hash of the wording it used, so a study can cite the prompt
its data came from. Counts and shapes have defaults; nothing that describes the
study does, because a plausible default there would be a study nobody declared.
The emitter is deliberately not the acceptor: what an LLM returns still has to
pass the audit, and §3.2's independent review still applies.

---

## 4. The lifecycle as preregistration

```bash
steerlab-cli experiment create formality-pilot --model <model-id> --revision <commit>
steerlab-cli experiment attach formality-pilot formality
steerlab-cli experiment pin-prompts formality-pilot prompts/tasks/formality-prompts.jsonl
steerlab-cli experiment pin-rubric  formality-pilot prompts/rubrics/default-paired-v1.md \
    --judges a:local,b:claude
steerlab-cli experiment set-instruments formality-pilot answerTokenLogprob
steerlab-cli experiment set-sweep-selection formality-pilot --objective logprobShift \
    --choice-prompts prompts/dev/dev-choices.jsonl

steerlab-cli experiment extract formality-pilot     # derive the vectors
steerlab-cli experiment validate formality-pilot    # held-out probe + cosines
steerlab-cli experiment sweep formality-pilot       # layer × α on the dev split
steerlab-cli experiment promote formality-pilot formality
steerlab-cli experiment freeze formality-pilot      # ONE-WAY, and gated
steerlab-cli experiment run formality-pilot
steerlab-cli experiment evaluate formality-pilot    # paired judging
steerlab-cli experiment analyze formality-pilot     # effect sizes, CIs, corrections
```

Not every study needs every verb. `steerlab-cli experiment --help` lists the
family, `steerlab-cli experiment <verb> --help` prints one verb's arguments and
runs nothing, and `--json` belongs on anything you script.

### 4.1 `create` and `attach` — pinning the recipe

`create` names the study and pins the model; `--revision` pins the model commit
(unset, it is auto-pinned from the local cache at first extract, and freeze
demands it either way). `attach` pins each concept's **current stimulus hash**
plus the extraction options — recipe
(`--method meanDifference|lat|emotionGrandMean|designatedReference`), reading
position (`--reading-position '<label>'`), the template the stimulus is rendered
through (`--extraction-rendering`), the reference concept (`--reference`),
grand-mean corpus members (`--corpus a,b,c`) — and pins the **neutral corpus**
whenever
one exists, because that corpus is the denominator that makes α comparable
across concepts, layers and model families. It is a pinned measurement input,
not a convenience.

```text
pinned formality @ 3a9c2c048f5e… (24+24 stimuli, meanDifference, last token)
```

Recipe choice is a methods decision — METHODS' reference-and-centering policy
says when paired mean-difference is right, when a deliberately authored
reference class is, and why a small family's own centroid is not usable as a
zero. `--project-neutral K` is legacy pooled projection: draft-only, and
verification-blocked.

Reading position and rendering are declarations, not defaults you inherit.
`--reading-position` takes one of eight named labels and is strict-parsed — a
writer never falls back, so an unknown label, or a template role asked for under
raw rendering, refuses at declaration rather than hours later on a GPU. The
older `--pool-from K` is the legacy spelling of one of those eight; it produces
byte-identical pins and is mutually exclusive with the new flag. Leaving both
absent keeps today's bytes exactly, but an explicit position **moves the recipe
identity**, because the position a direction was read at is part of what the
direction is. The same is true of `--extraction-rendering`, whose unknown keys
are typed refusals on both engines, and both axes travel: `attach` copies a
pinned artifact's rendering as it copies its position, and `verify` refuses a
contradiction in either.

### 4.2 `extract` — deterministic derivation

Re-derives the per-layer vectors into a fresh immutable run directory with full
provenance: model and revision, method, reading position, per-layer norms and
the corpus they were measured on, stimulus hash, date. Same pins → same bytes,
and nothing here looks at an outcome.

### 4.3 `validate` — the evidence freeze will demand

**Convergent validity** is accuracy classifying the never-named held-out
scenarios by projection against the training-class midpoint — out-of-sample by
construction, and a direction that cannot beat chance is not measuring your
concept, so stop and fix the stimuli. **Discriminant validity** is the
cross-concept cosine matrix (`cosine-matrix.csv`), computed at each concept's
own steering layer rather than a blanket mid-layer: two "different" concepts at
cosine ≈ 1 have collapsed into one axis, and any contrast between them is
meaningless. With one concept attached there is no pair, and the tool says the
measurement was *not made* rather than reporting a comforting number.

Two mechanics decide whether the evidence counts at all. **Vacuity**: a concept
with no probe leaves the run vacuous — it exits 0 and looks normal, but carries
the stamp, reports `result.vacuous` with one `vacuousValidation` advisory per
concept, and `freeze` refuses it naming the missing files, so read the
advisories rather than the exit code. **Scope**: evidence is keyed by the
*pins* — model + revision, concepts and their options, neutral corpus, run
substrate — and the experiment name is not among them, so a duplicate, or any
fresh experiment with matching pins, can freeze on validation it never ran. A
passing gate is not by itself proof that *this* experiment produced the
evidence; if that matters to your record, say which run did.

### 4.4 `sweep` — dose-response on a dev split, by a declared rule

`sweep` walks a layer × α grid on the **dev prompts** — never the measured task
set — scoring each cell for concept expression, degeneration (distinct-bigram
ratio; repetition collapse → 0), and capability-battery accuracy, and records a
recommendation per concept in its own run directory.

**The grid is manifest data too**, and it is a cost as well as a
preregistration — cells × concepts generations, every one of them paid for:

```bash
steerlab-cli experiment set-sweep-grid formality-pilot \
    --layer-fractions 0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13
```

Layers are stored as depth FRACTIONS, so one declaration names the
proportionally same sites in a 26-block model and a 62-block one; `--layers
13,18,28` is the absolute spelling, converted against the pinned model's depth
as read from an already-extracted vector. α is in residual-norm units, and 0 is
the baseline cell every sweep runs anyway. The defaults above are the engine's,
recalibrated on live testing — a starting grid, not a finding.

**Selection is manifest data**, declared before sweeping:

```bash
steerlab-cli experiment set-sweep-selection formality-pilot \
    --objective judgeScore --capability-tolerance 0.15 \
    --coherence-ratio 0.85 --coherence-backstop 0.6 \
    --control-margin 0.05 --control-apply-to topK --control-top-k 3
```

The flags are independent axes and the verb **merges**: re-declaring one of them
later keeps the rest, and the success line names whatever it carried over. Only
`--objective ""` clears the whole declaration.

**The coherence floor is relative to the α=0 baseline.** A cell passes only when
its distinct-2 holds at least `--coherence-ratio` of the baseline cell's *and*
clears `--coherence-backstop` absolutely. This replaced a fixed floor because a
fixed number cannot know what the model's own prose looks like: a sweep admitted
a cell at distinct-2 0.535 against a baseline of 0.989 — half the coherence the
unsteered model produced, on output 65% longer — whose `logprobShift` turned out
to be repetition rather than steering. It cleared the old absolute 0.45 and was
recommended. Declaring `--coherence-floor` instead still gives you the fixed
number, and every criterion pinned before this change keeps the absolute
semantics it ran under, permanently.

Three objectives exist on both engines: `markerDensity` (expression of the
concept's marker vocabulary), `judgeScore` (paired judging of each cell against
the baseline through the pinned rubric and judges, baseline pinned at the 0.5
tie), and `logprobShift` (mean Δ log-probability of the declared target option
over a hashed choice-prompts file, baseline 0). **The general rule: marker
density is a manipulation check, never the promotion objective when the
endpoint is prose-confoundable.** Selecting a dose on how often the concept's
own vocabulary appears optimizes for surface style — the exact confound a study
about decisions has to rule out — so any study whose claim is about an outcome
rather than about prose selects on `judgeScore` or `logprobShift`. With no
declared rule the sweep falls back to `markerDensity` and says so with a
`sweepSelectionDefaulted` advisory; on a choice-task study, treat that advisory
as a stop sign.

**Read the whole grid.** The recommendation is one cell; `sweep.csv` is the
dose-response curve and the coherence cliff. Pick a dose on the rising part of
the curve, comfortably below collapse — not the single most expressive cell an
optimizer found. Two columns are there for exactly this reading:
`distinct2Ratio` is each cell's coherence as a fraction of the baseline's (the
number the floor gates on), and `lengthInflated` marks a cell whose mean output
ran more than 1.5× the baseline's. The length flag is **reported, not gated** —
but a flagged cell with a strong objective is the shape a repetition artifact
takes, and it should be read before it is believed. Expression usually appears well below α ≈ 0.5 on small models
and coherence starts failing above ≈ 1, but that is a bracket to sweep, never a
setting to adopt. And note that **the matched-norm control here is a screen,
not specificity evidence**: `--control-apply-to winner` controls the argmax
cell alone while `topK` walks the top K promotable cells and promotes the first
that beats its own control, but both compare each candidate against a *single*
deterministic random draw, so the chance that some candidate passes a noisy
control grows with K. Keep K small, read a pass as "survived the screen", and
leave direction specificity to the confirmation stage's controls on held-out
data with paired statistics. The sweep's dev prompts and battery file are
themselves manifest pins (`sweep.devPromptsHash`, `sweep.batteryHash`).

### 4.5 `promote` — arms are born from evidence

```bash
steerlab-cli experiment promote formality-pilot formality --agent-name formality-mid

steerlab-cli experiment promote formality-pilot formality --cell 18:0.35 \
    --reason "coherence cliff at 0.45; chose the last stable cell"
```

The first form mints a variant artifact (an "agent") from the sweep-selected
cell carrying a **birth certificate**: `promotedBy: "criterion"`, the resolved
criterion, the dev-split hash, the winning cell, its metrics, the control
outcome. An arm should trace to the rule that produced it, not to someone's
memory of a good afternoon. The second is the manual override — permitted,
loud, stamped `promotedBy: "manualOverride"` — and it **still requires evidence
that a sweep ran** for that concept: the recommendation, or the newest sweep
run's recommendations entry (failure entries count, and are stamped as the
selection outcome). An override with no sweep at all is refused. `--reason` is
optional but recommended, and omitting it permanently records that no reason
was given. Hand-created and override-promoted variants stay legal but surface
as non-blocking freeze advisories.

### 4.6 `freeze` — the preregistration moment

```bash
steerlab-cli experiment freeze formality-pilot [--run-substrate local|server] [--force]
```

Freeze re-verifies every pinned input against the bytes on disk, stamps the
manifest's content hash and the workspace git commit, snapshots every pinned
input into `experiments/<name>/pinned/` (the no-git reproducibility floor),
writes the generated settings summary beside the manifest, and makes the
manifest read-only. The summary lands at `preregistration.md` when that path
is free (or holds a previous freeze's own generated file, recognized by its
"*Generated at freeze*" marker line); a researcher-authored preregistration at
that path — analysis commitments written before any data existed — is
preserved untouched, and the summary lands beside it as
`preregistration-frozen-settings.md` instead. There is no unfreeze. **Pin verification** always runs and is
**never skippable** — `--force` does not reach it — while the **seven evidence
gates**, each with a stable cross-engine id, are force-skippable:

| Gate id | What it demands |
|---|---|
| `revision` | a pinned, immutable model commit — not absent, not symbolic |
| `measurementPins` | the pins that determine *what is measured* are present and valid |
| `validateEvidence` | a `validate` run matching the exact pins on the run substrate, and not vacuous |
| `variantValidity` | attached variants carry hashed weights and a pinnable dataset manifest |
| `batteryEvidence` | each variant condition has scope-matched capability-battery evidence |
| `judgeValidity` | a rubric **file** and at least one judge (a panel of two or more must be genuinely distinct) |
| `gitClean` | every pinned input is committed in the workspace git repo |

A refusal is typed and names its repair: under `--json`, `error.code` is
`freezeGateFailed`, `error.gate` and `error.gates[]` name **every** failing
gate, and `error.repairAction` is an executable command sequence — so a refusal
is not an error to retry, it is a repair to perform. Two gate details worth
meeting in advance: judge identity resolves to `(kind, model, provider)`, so
two local judges with blank model fields are one judge agreeing with itself and
are refused; and `validateEvidence` looks for a real completed validation run
with actual results, so an empty planted directory will not satisfy it.

**`--force` is loud, stamped, and permanent.** Every skipped-and-failing gate
prints a warning, emits a `freezeGateSkipped` advisory, and is recorded in the
frozen manifest as `freezeForced: true` plus the ids in `forcedGatesSkipped`. A
forced freeze remains **non-citable — but checkably so, by stamp**, years
later, by someone who was not there: the discipline is enforced by the artifact
rather than by memory. Do not force a freeze to get past a gate; fix the gate.

After freeze the manifest is read-only and the writing verbs (`attach`,
`detach`, `pin-prompts`, `pin-rubric`, `declare-condition`, `set-sweep-grid`,
`set-style-taxonomy`, `confirm`) refuse. Two read-only verbs stay legal and
surprise people: **`sweep`**, which records recommendations in its own run
directory rather than the manifest, and **`promote`**, which mints into the
mutable variant library.

Iterate with `steerlab-cli experiment duplicate formality-pilot
formality-pilot-2`. A duplicate brings the donor's pinned concepts with it, so
when you are re-pointing a study at a different concept the full move is
**duplicate → clear the declarations that name the old concept → `detach` it →
`attach` the new one**. `detach` is all-or-nothing and refuses (`conceptInUse`)
while any condition, sweep-selection instrument, variant condition, or
perturbation policy still names the concept, which is what forces that ordering.
Without it a donor's concepts ride along swept but uncitable.

**A duplicate inherits its donor's validation evidence, and this saves real
GPU time.** The `validateEvidence` gate matches on a **validation scope
hash** — the model id and pinned revision, the attached concepts with their
pins, the neutral-corpus hash, the grand-mean corpus, the capability-battery
hash where variant conditions exist, and any declared validation depths. The
experiment **name is deliberately not in it**. So a duplicate that changes
only measurement-side declarations — a new rubric, a different judge panel,
new exclusion rules — freezes against the evidence its donor already produced,
with no re-extraction and no re-validation. Change anything the scope covers
and the inheritance correctly stops: a re-attached concept, a re-pinned
revision, a different battery each mint a new scope and demand their own
`validate` run. The rule to carry: **evidence is keyed by pins, not by
names**, so the cost of iterating is the cost of what you actually changed.

### 4.7 `run`, `evaluate`, `analyze`

```bash
steerlab-cli experiment run      formality-pilot [--prompts prompts/tasks/<override>.jsonl]
steerlab-cli experiment evaluate formality-pilot [--run <run-dir>]
steerlab-cli experiment analyze  formality-pilot [--allow-unverified-epoch]
```

**`run`** loads the model at the pinned revision, re-derives vectors from the
pins, and generates under **every declared condition** into one immutable run
directory: `generations.jsonl` (one stamped record per sampled output and/or
deterministic instrument readout), `battery.jsonl` (the capability battery,
scored per condition, kept separate), `metrics.csv` (surface metrics, including
reasoning-style `rs_*` columns when a taxonomy is pinned), `report.json`
(per-condition summaries, effect sizes, `conditions[].capabilityBattery`), and
the canonical `config.json`. `--prompts` overrides the pinned task file, and on
a frozen manifest that override is itself pin-checked. `run` refuses, before
the model loads, a concept-bearing manifest with no injection, variant, or
feature arm — the firewall telling you the study would measure nothing.

**`evaluate`** pairs every steered generation with its **same-item baseline**,
presents the two to each pinned judge blinded and position-randomized, and
tallies condition-wins / baseline-wins / ties into a new directory beside the
source run, which is never mutated. The judge prompt never sees the condition,
the α, or which side is steered; that blinding is load-bearing, so do not
undermine it by leaking labels into a custom rubric.

**`analyze`** is pure CPU: effect sizes **paired to each item's own baseline**,
with bootstrap confidence intervals and Wilcoxon, and the multiplicity
correction the phase calls for — **BH-FDR for screens, Holm for
confirmations**. It writes `analysis.json` and `effect-sizes.csv` into a fresh
run directory (plus `choice-deltas.csv`/`.json` for answer-token readouts), and
measured runs additionally carry `effectSizes` in their own `report.json`; the
server engine also writes `alien-residuals.csv` against a pinned human-effect
table and `promoted-movers.json` under the pinned promotion rule. Report
**effect sizes with confidence intervals, paired per item** — not bare win
counts, not p-values alone. Zero entries is an `emptyAnalysis` advisory rather
than a failure: the source run had no non-baseline condition.

**The epoch guard** covers `analyze`, `evaluate` and `rescore-style`: the
source run's stamped experiment hash must equal the live manifest's content
hash, or the verb refuses. `--allow-unverified-epoch` bypasses only *unstamped
legacy* runs and stamps `epochUnverified`. The guard is **per-engine**, since
canonicalization differs, so the rule is simply **analyze and evaluate a run on
the engine that produced it** — and through the bundle path (§7.2) it cannot be
bypassed at all, since `bundle execute --verb evaluate|analyze` accepts
`--source` but not `--allow-unverified-epoch`.

**Re-measuring a finished run with a new instrument.** Sooner or later a study
wants a different rubric, a different judge panel, or a human-validation
anchor applied to generations that already exist — and the instinct is to edit
the study. Do not: the epoch guard exists to stop exactly that, and editing
the source study would destroy the record of what its own run was measured
under. The guard instead tolerates the drift that **cannot have moved a byte
of the source run's generations** — six manifest fields (`judges`,
`evaluation`, `pipeline`, `judgeRubricFile`, `judgeRubricHash`,
`humanValidation`), plus the study's `name`, which is identity rather than a
measurement setting and which a duplication must change. So the sanctioned
path is a duplicate that never touches the source:

```bash
steerlab-cli experiment duplicate formality-pilot formality-pilot-recoded
steerlab-cli experiment pin-rubric formality-pilot-recoded \
    prompts/rubrics/<new-rubric>.md --judges a:local:<judge-model>,b:claude \
    --judge-pin a=<commit-hash>:bfloat16
steerlab-cli experiment evaluate formality-pilot-recoded \
    --run runs/<the-original-run-directory>
```

Executing remotely, the source run must be named the same way — the server's
run *discovery* is scoped by experiment name, and a duplicate has no runs
under its own name, so a submission without the override dies at discovery
before the tolerance can rule:

```bash
steerlab-cli remote submit-bundle <bundle> --site <id> --verb evaluate \
    --executor slurm --source runs/<the-original-run-directory>
```

(`--source` is the one spelling every surface uses — `bundle execute
--source`, `study submit --source` — and the server refuses an unreadable
directory at submit time, not on the allocation.)

### Coding a preregistered subsample

A power computation produces a per-condition record count, and it is routinely
smaller than the corpus. Coding a 7,200-record run when the design called for
2,400 is not a conservative choice — it is a different design, judged at three
times the cost — and hand-building a run directory holding the chosen records
is worse: `runs/` is immutable, and a directory nothing generated is a break
in the evidence chain. **Declare the design on the study**, and the rest
follows:

```bash
steerlab-cli experiment set-evaluation-sampling formality-pilot-recoded \
    800 0x5eed0a5e5eed0a5e
steerlab-cli experiment evaluate formality-pilot-recoded \
    --run runs/<the-original-run-directory>
```

**Choose the seed before you look at anything and declare it** — that is what
makes the subsample a design rather than a selection, and the same seed always
draws the same records. Declaring is what makes "preregistered" a *fact* rather
than a claim: the design lands in the manifest, every run stamps the manifest
snapshot into its own `experiment.json`, and a reader holding the run holds the
design. A plan document is pre-registration; the snapshot is provenance. Both
halves or neither, and the draw rule is derived by the engine rather than
typed. `set-evaluation-sampling <name> ""` clears the declaration.

Declaring a coding design is measurement-side, so it does not invalidate the
run being coded — which is exactly the house flow: duplicate the frozen study,
declare the coding design on the duplicate, and evaluate against the original's
run.

The two flags still exist (`--sample-per-condition` / `--sample-seed`, both or
neither) for an undeclared study. On a study that DOES declare a design they
become a cross-check: a value differing from the declaration refuses, naming
both, and never overrides. Prefer the declaration — it is the spelling that
travels with the evidence.

The draw is stratified across promptIDs within each condition and is identical
on both engines, so a corpus coded on the cluster and re-checked on the Mac is
one subsample rather than two of the same size. An `n` larger than a
condition's population refuses instead of quietly shrinking to fit — checked
at `evaluate`, where the records are, not at the declaration, where no run
exists yet; per-response
coding only, because the paired judge's unit is a pair rather than a record.

That last constraint is enforced where it can still be repaired. A study that
declares a design AND pins a **paired** rubric can never execute the design —
so `verify`, and therefore `freeze`, refuses the combination, naming both
facts and both repairs (clear the declaration, or pin a `perResponseCoding`
rubric). A freeze is permanent and a frozen declaration cannot be cleared, so
catching it at the desk is the difference between a five-second fix and a
study duplicated to escape it. The declaration verb refuses the same way when
the rubric is already pinned. Declaring **before** choosing a rubric stays
legal — that is the ordinary authoring order, and the gate simply waits.

Everything the reader needs is stamped: a `sampling` block in
`coding-report.json` and the run's `config.json` carrying the size, the seed,
the counts and the derivation rule (plus `declared: true` when the draw came
from the declaration), and every line reading `coded N of M
(seeded subsample)`. When you report the result, report it as a subsample —
and when a report has **no** `sampling` block, it covered the whole corpus.

Remotely, a declared design needs nothing on the submission at all, and the
walltime estimate prices the sampled count rather than the full matrix. On an
undeclared study the two flags ride the submission instead:

```bash
steerlab-cli remote submit-bundle <bundle> --site <id> --verb evaluate \
    --executor slurm --source runs/<the-original-run-directory> \
    --sample-per-condition 800 --sample-seed 0x5eed0a5e5eed0a5e
```

Three things make this defensible rather than a loophole. The tolerance is
**by field, not by intent**: change a generation-side pin — the model, a
concept, the task prompts, the sampling protocol — and the guard refuses,
because those runs *would* have been different. The tolerated drift is
**stamped** into the output (`measurementDrift`, naming the fields that
moved), with a warning on stderr, so the re-measurement is never mistaken for
the original measurement. And **`promote` never tolerates**: a promotion binds
a judged sweep's evidence, and a judge swap changes what that evidence means,
so a renamed or re-judged manifest still refuses there. Report a
re-measurement as what it is — the same generations read by a second
instrument — and cite both the run and the recoding study.

---

## 5. Controls and confounds

### 5.1 The control matrix

Declare each arm explicitly; `--alpha-units norm` (the default) is what makes α
comparable across concepts.

```bash
steerlab-cli experiment declare-condition formality-pilot baseline --baseline
steerlab-cli experiment declare-condition formality-pilot formality-lo  --slots formality:18:0.25
steerlab-cli experiment declare-condition formality-pilot formality-hi  --slots formality:18:0.45
steerlab-cli experiment declare-condition formality-pilot formality-neg --slots formality:18:-0.35
steerlab-cli experiment declare-condition formality-pilot formality-rand \
    --slots formality:18:0.35 --control randomMatchedNorm
```

| Arm | What it rules out |
|---|---|
| **baseline** | the zero every paired statistic is computed against; added implicitly if undefined, so define it when you want it named |
| **two or more positive doses** | a monotone dose-response is evidence; a single dose is an anecdote at one α |
| **negative α** | the *sign* control — if pushing the concept out moves the endpoint the other way, the effect is signed and real rather than noise |
| **matched-norm random** | the *magnitude* control, most often skipped: if a random direction of equal norm at the same layer moves or degrades output as much, you measured the size of your nudge and nothing else |
| **a positive-control concept** | that the instrument can detect anything at all; without it a null is uninterpretable |
| **the capability battery** | that a "bias" finding is not plain degradation (§5.2) |

Under a designated-reference recipe, −α steers toward the reference class
rather than a concept's opposite; phrase reversal claims accordingly, and use
`--control randomDirectionAblation` as the ablation analogue of the random arm.
Multi-slot conditions are legal and hash as one condition — several slots *are*
the linear mix `h + Σ αᵢ·vᵢ` — so treat each mix as a first-class arm, keeping
in mind that mixes multiply the matrix fast.

### 5.2 Capability is a gate, not a footnote

The battery runs **per condition inside `run`** on both engines, writing
`battery.jsonl` and `conditions[].capabilityBattery`. Its job is to separate
"the concept moved the decision" from "the model got worse at everything". If
accuracy drops under a condition, report the degradation alongside the effect
and treat the effect as confounded — extreme doses break tasks rather than
biasing them, and the battery is what tells you which happened.

#### The floor battery, and why it is not this one

There are **two** capability artifacts, and conflating them is the mistake to
avoid. The one above is the *pinned per-condition control*: it belongs to a
study, it is frozen into that study's manifest, and it is scored inside that
study's run matrix. The other is the **floor battery** — `steerlab-server
battery run <battery-file> --agent <ref>…`, which reads one or more agents
against a battery *before any study uses them* and writes its own pinned run.

The floor battery is governed by a charter (`docs/CLI-REFERENCE.md` §6.10):
ex ante justified, study-blind and fixed; both operating regimes (greedy short
answers **and** long-form generation at a positive temperature, because agents
are used generatively); sensitivity validated — never defined — by a known
degraded positive control. Its boundary is worth stating in the negative,
because it is the line researchers cross: if your study measures relative
performance on very hard, lengthy real-analysis proofs, that capability must
**not** be probed by the battery, much less gated on it. The battery asks
whether the model still works; whether it is good at your hard task is settled
by performance in your study.

The second regime exists because the first could not see the failure. A short
greedy battery scored accuracy **1.0** at a dose that three independent
instruments had already confirmed degraded. Twenty-four greedy tokens have no
room for length inflation, variance collapse, or incoherence to appear in — so
a floor reading also reports mean word count, distinct-2, their spreads, and
completion rate, each against the baseline agent's.

#### Matched capability: the comparison that needs it first

The floor battery earns its keep on comparisons *between different kinds of
agent*. Suppose the claim is that an injected agent and a prompted persona
produce different legal behaviour. Before that comparison means anything, the
two have to be shown **capability-equivalent**: if the injected agent is
quietly degraded — writing longer, repeating itself, running past its token
budget — then any behavioural difference you measure is a difference in
*competence*, not in the disposition you named, and the study cannot tell them
apart afterwards. The same applies to a fine-tuned variant against a stock
model, and to two doses of the same direction against each other.

So run the floor battery over every arm you intend to compare, at the doses you
intend to use, before the study is frozen — one `battery run` naming each arm
as an `--agent`. Cite the report's per-agent accuracy *and* its generation
health in the methods note, and say which arms matched and which did not. A
comparison whose arms were never shown equivalent is not a comparison of
dispositions; it is a comparison of two capabilities that happen to have
different names. Because the report is keyed by pins (battery digest, model
revision, vector-artifact hashes, dose, protocol), a later study using the same
agent at the same dose can cite the same reading rather than buying its own.

Nothing in the floor reading refuses anything: whether an arm's floor is
acceptable is your ruling against your study's stakes. A battery that refused
on a number would have become a difficulty target, which is exactly what the
charter's first clause forbids.

### 5.3 Judge validity

- **At least one judge**, enforced at freeze; a panel of two or more must be
  **genuinely distinct**, where distinctness is `(kind, model, provider)`. A
  judge's **name is a label, never a model id** — an empty local-judge model
  resolves to the study model at its pinned revision and generates through the
  already-resident model, while a different-model local judge refuses at start
  wherever a second resident model is impossible, rather than dying partway
  through a panel. Zero judges is the state `judgeValidity` refuses: a judged
  instrument with no judge codes nothing.
- **A single-coder design is legal, and it carries no inter-rater
  statistics.** The gate stopped demanding two judges on 2026-08-28: how many
  coders a study declares is a methodological choice, not something an
  instrument should forbid. What the choice costs is said rather than hidden.
  A one-judge study freezes cleanly with a loud, non-blocking
  `judgePanelTooSmall` advisory, and its coding report records
  `fieldAgreement` as **absent with a reason** —
  `fieldAgreementAbsentReason: "single-coder design: 1 judge coded this run,
  so no inter-rater agreement statistics exist"` — rather than as an empty
  list, which would read as "agreement was measured and there was none".
  **The recommended shape when cost pushes you toward one coder: calibrate
  once with a two-judge run, then run production single-judge, and report the
  calibration's κ alongside the single-coder results.** That gives a reader
  the one thing a single-coder design cannot otherwise supply — evidence that
  the rubric is reproducible between coders at all — without paying for a
  second coder on every record. Cite the calibration run by directory, and say
  plainly that the production codings are single-coder.
- **A cross-substrate local judge is the economical instrument for a large
  batch.** A local judge naming a model other than the study model is a
  genuinely independent coder that costs no per-token API spend, which is what
  makes a many-thousand-record coding run affordable. It needs three things:
  the judge model installed (a judging run never downloads weights on your
  behalf), `STEERLAB_MAX_LOADED_MODELS ≥ 2` on the engine host so a second
  model may be resident beside the study model, and the judge's exact bytes
  pinned — `pin-rubric … --judge-pin <judge>=<commit-hash>[:<dtype>]`, or
  `judgeValidity` refuses at freeze. The revision must be a commit hash: a
  branch is re-pointed by definition and cannot identify the weights that
  coded your data.
- **A judging run never downloads weights on your behalf.** A local judge whose
  declared model is not installed refuses before any generation and names the
  install; it does not reach the hub loader. The judge picker likewise vets a
  candidate repository from the snapshot's own files — a text-generation
  architecture in `config.json` and a chat template — so a cache entry that
  cannot answer a question is never offered as a judge. OpenRouter judges are
  available on the robustness path too, spelled
  `openrouter:<model>:<provider>`, and are offered only when a key is present.
- **The rubric is a pinned file**, hashed into the manifest; inline rubric text
  is draft-only and cannot freeze.
- **Noncompliant judge answers become recorded rows** rather than crashes: a
  judgment failing the declared schema is retried once and then refused as data
  — invented values are never recorded — and the run completes with the
  noncompliance visible in the report. A panel producing many of them is a
  finding about the panel, not a reason to hand-fix rows.
- For per-response coding, read the **inter-judge agreement** block before the
  aggregates: κ across judges is what says whether the coding instrument is a
  measurement at all. Each categorical `fieldAgreement` entry now carries a
  **`confusion` block** beside its κ — `confusion[a][b]` is how many shared
  cells judgeA coded `a` while judgeB coded `b`, computed over the very label
  pairs κ was computed over, so the counts sum to the entry's `n`. Read it,
  because κ alone cannot tell two very different findings apart: a κ of 0.55
  concentrated in **one** systematically confused label pair is a
  **rubric-anchor** problem — two categories your rubric does not separate,
  fixable by re-anchoring them and re-coding — while the same κ spread evenly
  across the matrix is a **noisy-field** problem, and says the field is not
  measurable as written. Numeric fields carry `meanAbsoluteDifference` and no
  confusion block, having no κ to explain.

### 5.4 Parser failure is a confound

Any endpoint requiring free-text parsing carries this risk, and it is not
hypothetical: a duration parser that accepted digits but not number words once
left a small fraction of records unparsed **dose-dependently**, because
steering pushed generations toward a register where number words are more
common, and unparsed records clustering in treated conditions bias the treated
mean relative to baseline. **An endpoint parser that fails at a rate correlated
with the intervention is a confound and must be checked as one** — report
parse-failure rates per condition, always. The structural fix is to prefer the
log-probability instruments (§2.4), which have no parser. Where prose parsing
is unavoidable, the server's `analyze` re-parses **null-only** run-time parses
under the pinned grammar and stamps what it rescued, so a parser fix reaches
finished runs without regenerating them; stored parses are never overwritten.

### 5.5 Screen → confirm, on disjoint pools

Screening and confirming are two studies, not two phases of one manifest.
**Screen** broadly, over many concepts, with BH-FDR correction, selecting by
the declared criterion and promoting the survivors. Then **confirm**:
`duplicate` the frozen screen into a fresh draft, `detach` the screened concepts
the confirmation does not carry (§4.6 — clear their declarations first, detach
last), and

```bash
steerlab-cli experiment confirm formality-pilot-2 --agent formality-mid --deltas 0.2
```

declares a perturbation policy around the promoted arm's anchor cell (α and
α ± δ), with a matched-norm control unless `--no-control`, expanding
mechanically into ordinary hashed conditions. Re-test on **held-out** task
prompts — `verify` enforces that the confirmation pool is disjoint from the
screen's — and analyze with Holm correction.

---

## 6. Measured-run policy: which substrate supports which claim

**Local (Swift/MLX) measured runs are greedy-only.** The study runner requires
`temperature == 0` and rejects more than one seed, because the local generator
cannot pin a per-run sampling seed; `manifest.seeds` is recorded for provenance
but does *not* affect local generation, and every local record stamps
`seedInert: true`. A local measured run is an honest, reproducible **N = 1
deterministic cell** per condition × item, so a "20-seed" local run is one
sample counted twenty times and must never masquerade as an N. **Stochastic
studies belong on the Python engine**, which seeds per record and writes one
record per (condition, prompt, sampleIndex) plus per-item distribution
summaries — while **categorical endpoints sidestep the question**, the
log-probability instruments being temperature-free on either engine.

**Vectors do not transfer between engines.** CUDA/HF activations do not
byte-match MLX/Metal, so a study's vectors must be **re-extracted and
re-validated on whichever substrate its measured runs execute on**. What *is*
identical across engines is the structure: SHA-256 input hashes, the manifest
and run schemas, the JSON envelope, and committed golden fixtures both sides
check (`steerlab-cli vectors compare <a.safetensors> <b.safetensors>` is the parity
harness, key-identical on both CLIs). The consequence is an ordering rule:
**freeze where you validated.** Validation and battery evidence count on the
substrate the gates are matched against, so a study the server will run should
be extracted, validated, and frozen with
`steerlab-cli experiment freeze formality-pilot --run-substrate server`, and a
fully local study validates and freezes locally; foreign-substrate evidence
surfaces as a cross-substrate advisory rather than silently passing. Adapters
are substrate-specific for the same reason, so report adapter format, base
model, revision, and training-data hash in any study using one.

---

## 7. Scaling out to a GPU

### 7.1 Qualify the node before the first study

```bash
steerlab-server site qualify [--skip-model-fixtures]
```

Run it on the node itself — after provisioning, after any deploy that changes
the engine, and before the first study on a site you have not used; no GPU is
needed for the core checks. Nine checks ask whether this machine reproduces the
contracts a result depends on (build identity, the measurement-stack
fingerprint, dependency-lock agreement, the stimulus-hash convention,
prompt-render and tokenization goldens, vector-parity arithmetic, profile
validation, CUDA visibility), and none aborts the rest, so a cold node always
gets a complete report. Read the **summary line** (`5 passed, 1 warning, 0
failed, 3 skipped of 9 checks`), not the exit code alone: skips never change
the verdict but are always counted, and a node with six skips passed almost
nothing. `vectorParity` uses same-engine synthetic fixtures — it qualifies the
parity *arithmetic* on this node, not cross-substrate agreement, which is why
§6's re-extraction rule stands regardless of a clean report.

### 7.2 Two ways to reach the engine

**Paired**: the server serves this workspace's tree
(`steerlab-server serve --root <workspace>`) and the direct verbs
(`steerlab-server experiment run|validate|sweep|evaluate|analyze <name>`)
execute server-resident studies. **Bundle submit**: the portable path, and the
default whenever the engine cannot see your workspace — the study travels as a
hash-pinned bundle exactly when needed, and the evidence bundle imports back.
`bundle import` re-verifies pins and refuses to overwrite a frozen manifest.

```bash
steerlab-server study submit formality-pilot --verb validate --executor slurm \
    --gres A100 --walltime 04:00:00
```

> **`--verb` defaults to `run` on both submit paths.** Omitting it submits a
> full measured run, not the sweep or validate you meant — a GPU allocation you
> will then have to cancel. Always pass `--verb` explicitly.

GPU *type* is chosen by `--gres` (`--gres A100` expands to `gpu:A100:1`), not
by any `--gpu` flag; and `--parallel N` fans a `run` across sibling jobs whose
partials are merged by a **running** `steerlab-server serve`, not by the
submitting process. Sharding is execution logistics — it never enters the
manifest or its content hash, so a sharded run and a single-job run of the same
frozen study are the same measurement.

**Sharding in practice, and the two ways it bites.** The fan-out is reachable
from the Mac as well: `steerlab-cli remote submit-bundle <path> --site <id>
--verb run --executor slurm --parallel 4`. It applies to Slurm submissions of
the `run` verb only (or a pipeline whose declared chain starts with `run`), and
the envelope echoes what actually went on the wire —
`parallelJobsRequested`, `parallelJobsEncoded`, and
`parallelJobsSuppressedBecause` — so a request the rule suppressed reads as
suppressed rather than as honored.

Then two disciplines, both learned the expensive way:

1. **Verify the shard jobs landed; do not trust the exit code.** K shards are
   K independent scheduler submissions by design, and a fan-out can
   **partially fail while the submit still exits 0** — the abort is reported
   through the *parent* job record, not the submitting process. After any
   sharded submit, look at the queue (`steerlab-cli remote jobs`, or the
   scheduler's own queue command at the site) and count the jobs. A run that
   silently lost half its shards does not merge, and the missing cells are
   found much later, at the merge's completeness check.
2. **Stagger submissions where the site caps queued jobs per user.** Most
   Slurm sites enforce a per-user submit and run limit through the QOS; a
   fan-out that crosses it has its later shards refused by the scheduler while
   the earlier ones run. Find your site's real limit
   (`sacctmgr show qos format=Name,MaxTRESPerUser`), record it in the site
   profile as `maxParallelGPUJobs`, and keep total queued work under it —
   submitting several sharded studies back to back is the usual way to
   discover the cap.

The mechanics, the merge rules, and the completeness check are
[CLI-REFERENCE.md](CLI-REFERENCE.md) §5.3.

### 7.3 Deploying, reaching a site, and auth

Site profiles are workspace data, not repository data: keep your site's profile
JSON beside the study data and import it.

```bash
steerlab-cli cluster sites import <profile.json>
steerlab-cli cluster preview  --site <id>      # read exactly what will run, offline
steerlab-cli cluster ensure   --site <id> --target connected --allow-push --allow-bootstrap
```

`ensure` is the idempotent principal command, and its refusals are the design:
with no `--allow-…` flag it returns `needsApproval` and names the flag it
wants, because every remote side effect is authorized separately. The human
boundary is absolute — no verb accepts a password, a passcode, or a second
factor in any form. Expired SSH auth is not a broken site and has its own verb
family: `cluster auth open` spawns *your* Terminal for the password and second
factor and persists the resulting master connection, `cluster auth status`
confirms it, and `cluster auth close` ends it. An agent that hits `Permission
denied, keyboard-interactive` should say so and wait for you, never retry the
refusal. Two deploy facts that have bitten real runs: a successful
`push` **re-stamps the deployed build identity** (an rsync `--delete` would
otherwise erase it, and every later run would stamp the previous deploy's
commit), and bootstrap installs the **committed platform lock** before the
editable install, so a node's torch and transformers are the ones every other
node resolved rather than whatever the index published that morning.

**`payload: current` means deployed == last-pushed *intent*, not deployed ==
your local code.** This is the anti-rollback design and it is right: `status`
compares the deployed engine against the revision *this machine last pushed*,
so a site deployed from another machine, or from a payload built outside the
app bundle, is not accused of being stale and no push is offered that would
silently roll it back. What that comparison cannot say on its own is that
**your build has moved on since that push** — and it once did not say it for a
whole day, while a deployed engine trailing the app bundle by eight commits of
engine-side selection semantics ran six GPU sweeps under a clean status line.

So the payload line now carries the missing half, as an advisory and never a
state change: when the deployed revision differs from this build's payload it
appends *"server-side changes since that push are NOT running; push a fresh
payload if the study needs them"*. The same warning reaches the moment it
matters — `remote submit-bundle --site` prints it once on stderr before
submitting, computed from local records only, so a `--url` invocation or a
never-pushed site stays honestly silent rather than guessing. Ancestry between
the two revisions is deliberately not claimed: the deployed commit may not
exist in any repository on this Mac, so the sentence states inequality, never
"older than".

**The operational rule that follows: after landing engine-side changes a study
depends on, push a fresh payload — do not read `current` as "my code is
running."** Build the payload, push it, then cycle the controller, since a
push replaces files on disk and the running controller keeps the code it
loaded. And re-run `site qualify` (§7.1) after any deploy that changes the
engine.

The Python engine **requires a bearer token by default, on every platform**:
with no configuration, `serve` resolves token mode, hydrates the token from
`~/.steerlab-token`, and mints that file (0600) when absent, printing the path
and never the value. The historical open-on-loopback tier is an explicit opt-in
(`serve --dev-open-loopback`) that *refuses to start* on a non-loopback bind or
beside a Slurm executor, and every mutating `/api/` route is privileged by
default. Reach a remote engine over an SSH tunnel; there is no TLS. Read
[SECURITY.md](../SECURITY.md) before exposing anything.

### 7.4 Provenance a remote run must carry home

Every run directory stamps a canonical `config.json` (closed key set,
cross-engine; both engines refuse to overwrite it) recording the run id,
substrate, platform, dtype, sampling policy, scheduler job id, and — on the
Python engine — a `pythonEnvironment` block naming the interpreter and the
science-relevant package versions actually imported, torch's local CUDA version
segment included. The lock says what was *intended*; the stamp says what *ran*,
and disagreement between them is an **advisory** written to the run's
`advisories.txt`, never a refusal: a queued job must not die because an index
moved, and a submitted study must not change meaning mid-flight. The policy
throughout is **continue loudly and stamp**, so read `advisories.txt` when you
read the results.

---

## 8. The app as a view of the same state

The macOS app is a rich surface over the same store: authoring concepts,
watching a sweep stream, freeze readiness with every unmet gate listed,
submitting and monitoring server jobs, browsing runs, chatting under steering.
It calls into the engine — never the reverse — and a manifest authored in the
app is byte-identical to one authored on the CLI, because both go through the
same store setters. Two consequences are the only ones to remember: **anything
you will cite must be reproducible headlessly**, and **there is deliberately no
force-freeze button** — forcing requires the CLI, which is the right amount of
friction. For results, the embedded Results Explorer is the canonical viewing
surface; it may show viewer-derived numbers only under its badge vocabulary
(stored plain, derived badged, heuristic badged), while κ, confidence
intervals, and p-values render exclusively from engine artifacts.

---

## 9. What you can claim, and what carries the claim

Every operation writes one immutable `runs/<timestamp>-<slug>/`, never
overwritten and never reused, carrying enough to rebuild your tables without
rerunning a model. Three subtrees under `runs/` are deliberate mutable
*libraries* rather than outputs — promoted variants, neutral
principal-component bases, imported lens artifacts — where frozen studies are
protected by the manifest snapshot and the artifact hash rather than by the
directory. Do not hand-edit `experiment.json`: its bytes *are* the content
hash, so an edit bypassing the verbs surfaces either as a verify violation,
which is the good outcome, or as a silently different study, which is not.

| File | Produced by | What "good" looks like |
|---|---|---|
| `cosine-matrix.csv` | `validate` | low off-diagonals — visibly distinct directions |
| `validation-evidence.json` | `validate` | the stamp `freeze` checks for; not vacuous |
| validate `report.json` | `validate` | convergent accuracy ≫ chance, per concept |
| `sweep.csv` | `sweep` | smooth dose-response, a clear coherence cliff, the recommended cell below it |
| `dev-generations.jsonl` | `sweep` | the prose behind every cell — baseline, grid, and control dev texts, readable after the fact |
| `recommendations.json` + `selection` | `sweep` | resolved criterion, dev-split hash, winning cell, controls evaluated |
| `generations.jsonl` | `run` | one stamped record per output or instrument readout, full provenance |
| `battery.jsonl` | `run` | per-condition capability records, separate from generations |
| `metrics.csv` | `run` | surface metrics, `rs_*` style features when a taxonomy is pinned |
| run `report.json` | `run` | per-condition means, `effectSizes`, `conditions[].capabilityBattery` |
| `judge-report.json` | `evaluate` | blinded per-condition wins against baseline |
| `coding-report.json` | `evaluate` (coding rubric) | per-field aggregates plus inter-judge κ |
| `effect-sizes.csv` | `analyze` / `run` | paired bootstrap CIs, Wilcoxon, BH-FDR or Holm by phase |
| `alien-residuals.csv` | server `analyze` | model-minus-human residuals against the pinned table |
| `config.json` | every writer | substrate, platform, dtype, sampling policy, environment stamp |
| `advisories.txt` | every writer | everything that did not stop the verb — read it |
| `preregistration.md` | `freeze` (or the researcher, first) | the frozen design, exported at the freeze instant — unless the researcher authored this path before freezing, which freeze preserves, exporting to `preregistration-frozen-settings.md` instead |

[RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md) is the authority on what
each layer of result licenses: in outline, a model-internal layer needing no
external anchor, a human-anchored layer needing a transcribed and pinned effect
table, and a multi-agent propagation layer. The layers gate independently, and
a study reaching only the first is a complete study, not a partial one.

Three claims a study *cannot* make, whatever the numbers say. A **forced
freeze** is not evidence — it is a pilot with a permanent label. A result whose
**capability battery dropped** under the condition is confounded until the
degradation is reported beside it. And a result that did not beat its
**matched-norm random** arm is a statement about the size of the nudge, not
about the direction. Each is visible in the artifacts above, which is the
point: the reader does not have to take your word for any of it.

---

## 10. In one paragraph

Author a concept and a task that cannot have known about each other; declare
the endpoint and the instrument that measures it; derive the direction and
prove on held-out material that it detects its own concept and is not its
neighbours; choose the dose by a rule you wrote down first, on a dev split that
is not the measurement; declare the controls that let a reader distinguish your
effect from a nudge of that size; freeze, which pins all of it and makes the
manifest read-only; run; judge blinded and paired; and report effect sizes with
intervals, corrected for the phase, alongside the capability numbers and parse
rates that would undermine them. Every one of those steps leaves a hash, a
stamp, or an advisory — so the claim can be checked by someone who was not
there, which is the only kind of claim worth making.
