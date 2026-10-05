# Judging, evaluation, and analysis

## Declare the judging instrument

This client has no dedicated rubric verb. The rubric and the judge panel are
protocol fields — `judgeRubricFile`, `judgeRubricHash`, `judges`, and
`evaluation` — written with `steerlab experiment set-protocol <name> --set
<key>=<json>`, or delivered already pinned by a study pack
(`steerlab workspace guide assembly`). Prefer the pack: it pins the rubric
file's real bytes, so no hash is typed by hand. The rules are the same on
every client:

- The judge **name is a label, never a model id.** Kinds are `claude`,
  `local`, `openrouter`. A blank model field is *absent*, not empty: a local
  judge then resolves to the study model at its pinned revision; a `claude`
  judge to the default judge model. A serving provider is only legal on
  `openrouter`.
- Declare **any number of judges, including exactly one**: a single-coder
  design is a legal methodology and freezes cleanly. What it costs is said
  rather than forbidden — a `judgePanelTooSmall` advisory at freeze, saying
  that no inter-rater agreement statistics will exist for the study's codings,
  and the coding report then records `fieldAgreement` as **absent with that
  reason** rather than as an empty list. Zero judges is the state
  `judgeValidity` refuses: a judged instrument with no judge codes nothing.
- Inline rubric text is draft-only and cannot freeze — pin a file.
- A panel of two or more must be **distinct**: identity resolves to (kind,
  model, provider), so two `local` judges with blank model fields resolve twice
  to the study model at temperature 0 — one judge agreeing with itself by
  construction — and refuse at freeze under `judgeValidity`. Vary the kind,
  the model, or the provider.
- A **local judge naming a model other than the study model** must pin the
  exact bytes that will judge — `judges[].revision` and `judges[].dtype` — or
  freeze refuses under `judgeValidity`. Dtypes are `bfloat16`, `float16`,
  `float32`. The revision must be a commit hash: a branch or tag is re-pointed
  by definition, so it cannot identify the weights a run used.
- A paid judge (`claude`, `openrouter`) spends the researcher's money per
  response. Ask before a judged evaluation runs.

## `analyze`

```bash
steerlab run <name> --runner <url> --verb analyze
```

Pure CPU, no model load. Paired-to-baseline effect sizes — bootstrap CIs and
Wilcoxon — over the newest completed run the runner holds for the study;
writes `effect-sizes.csv` and folds `effectSizes` into `report.json`. `n` counts items: an item sampled several
times contributes one difference, its responses averaged within each
condition. Rows marked `withinItemSamples` describe one item's samples and
are not tests.

Guarded by the **epoch guard**: the run's stamped experiment hash must equal
the manifest's content hash, or the verb refuses. The guard is per-engine —
analyze a run on the runner that produced it.

**Which outcome leads.** When you summarize an analysis from the files that
came home, lead with the outcome the study is about, by the order below, and
say which rule chose it: "declared by the researcher" or "chosen by default
order". The declared outcome is `primaryOutcome` in the analysis run's
`experiment.json`. The engine's own analysis envelope states the same answer
as `result.headline`, for a caller who has it.

Do not lead with word count because it is the first row of the table. The
order: the outcome the study declared (`steerlab experiment
set-primary-outcome`, `workspace guide settings`), when the run has it; else a
judged outcome, which lives in the evaluation report; else a choice or numeric
outcome; else a reader or probe score; else reasoning style; else marker
density; else surface measures. When the study declared an outcome the run
does not have, lead by that order and say the declared outcome is missing.
Every row is still in `effect-sizes.csv`, in the engine's own order.

Say an effect "survives correction" only when the outcome was compared across
more than one condition. With one treatment condition the adjusted p equals
the raw one, so report it as the one test it is. A row with fewer than three
paired items is too few pairs for an interval: describe it, and do not report
its interval or its p-value as a finding.

Zero effect-size entries is reported as an `emptyAnalysis` advisory, not a
failure. It means the source run had no non-baseline condition. Check for it.

## `evaluate`

```bash
steerlab run <name> --runner <url> --verb evaluate
```

Paired-judge evaluation of a completed run through the manifest's pinned
rubric and judges, writing a new evaluation directory beside the source run,
which is never mutated. Same epoch guard as `analyze`; the runner evaluates
the newest completed run it holds for the study.

**To code a preregistered SUBSAMPLE rather than the whole run**, declare the
design on the draft before freezing:
`steerlab experiment set-evaluation-sampling <name> <n> <seed>`
(`workspace guide settings`). The evaluation then draws it with no flags. The
draw is stratified — within each condition, `floor(n / P)` records per
promptID with the remainder handed out in seeded order, and records inside
each cell chosen over `sampleIndex` — and it is the same draw on both engines
for the same seed. An `n` above a condition's population REFUSES; it never
clamps, because a clamped design is a different design than the one that was
preregistered. Per-response coding only: a paired rubric refuses, since a pair
is not a record. The result is stamped loudly — a `sampling` block in
`coding-report.json` and in the run's `config.json` carrying
`samplePerCondition`, `sampleSeed`, `sampledRecords`, `sourceRecords` and the
derivation `rule`. **No `sampling` block means the full corpus was coded**;
never report a sampled coding as a census.

Under a per-response coding rubric it writes `coding-report.json`. Read its
`fieldAgreement` entries before the aggregates: each categorical entry carries
`percentAgreement`, `kappa`, and a `confusion` block where `confusion[a][b]`
is how many shared cells judgeA coded `a` while judgeB coded `b`, summing to
the entry's `n` — so you can say WHERE two coders part ways without
re-deriving anything. A single-coder run has **no** `fieldAgreement` key at
all; it carries `fieldAgreementAbsentReason` instead. Do not report that as
"agreement was measured and was zero" — report it as what it says.

**To re-measure an existing run with a NEW instrument, duplicate — never edit
the source study.** The epoch guard tolerates exactly the drift that cannot
have moved a byte of the source run's generations: `judges`, `evaluation`,
`pipeline`, `judgeRubricFile`, `judgeRubricHash`, `humanValidation`, and the
study's own `name`. Duplicate with `steerlab experiment duplicate <name>
<name>-recoded` and declare the new instrument on the copy. This client's
submission verbs carry no flag that names a source run, so evaluating the
duplicate against the ORIGINAL run cannot be submitted from here yet. Say so
to the researcher as a gap; do not edit the source study to get around it.

## Reasoning-style rescoring and CPU completion

CPU completion is available through
`steerlab-server experiment complete-sweep-judgment <study> --awaiting-run
<run> --judgments <file> --json` or `steerlab-server experiment complete-judgment` for an
evaluation. Preserve the original packet, judge and epoch requirements.
The sweep may project a recommendation into a draft; repeated evaluation
completion reuses evidence. Read `result.runDirectory`, `reused` and `changed`.

For style rescoring use `steerlab-server experiment rescore-style <study>
--source <run>` on the engine, with `--json`. It recomputes reasoning-style
features for a completed run through the pinned taxonomy into a **new** run
directory, never touching the source. New reports preserve the source run.

The engine's three CPU verbs return typed envelopes and 64/65/66/70 for
usage/refusal/missing input/failure; do not depend on the old catch-all exit 1.
