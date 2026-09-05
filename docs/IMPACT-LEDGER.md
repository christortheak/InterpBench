# `ledger impact` — what the 2026-09-05 fixes mean for artifacts that already exist

**What this document is.** The reader's guide to `steerlab-server ledger
impact`: what it reads, how it decides `exposed` / `unaffected` / `unknown` for
each of the four science fixes, what the reassessed promotions file contains,
and what is left for the researcher. The implementation is
`Server/steerlab_server/experiment/impact_ledger.py`; its module docstring says
the same things in the same words.

**Why it exists.** Four defects were fixed on 2026-09-05 after an external
review. Each fix landed with a regression suite. **Those suites establish that
the behaviour is covered from now on. They establish nothing whatever about the
artifacts already sitting in a workspace** — a passing test in this checkout is
not evidence about a J-lens readout taken three weeks ago. This verb answers
that second question, artifact by artifact, from the artifacts' own bytes.

It is a **screening instrument**, not a verdict. It says which artifacts a fix
could have reached and what was read to decide; it never says how much a number
moved, and several of its rules say so in the entry text itself.

---

## §1 — Running it

```sh
steerlab-server --root <workspace> ledger impact [--code-checkout <path>] \
    [--json] [--out <file>]
```

- `--root DIR` (the global engine flag) or `$STEERLAB_ROOT` names the
  workspace. This is the ENGINE's convention; `$STEERLAB_WORKSPACE` belongs to
  the cross-platform `steerlab` client and is not read here.
- `--code-checkout <path>` is a git checkout of this repository. With it, a
  stamped build commit is dated against the fix commits by
  `git merge-base --is-ancestor`. Without it, every revision-dependent finding
  (SCI-01, SCI-03) answers `unknown`, and the ledger says so.
- `--json` prints one envelope on stdout with every diagnostic on stderr;
  `--out <file>` writes the same envelope to a file in either mode.

Exit codes are the shared vocabulary: **0** the ledger was written (an
`exposed` artifact is the PRODUCT, not a failure), **64** a malformed
invocation or an unusable `--code-checkout`, **66** the root is not a
workspace (the source checkout is refused by name), **70** the scan broke.

Output lands in a fresh `diagnostics/impact-ledger-<UTC stamp>/`:

| file | what it is |
| --- | --- |
| `impact-ledger.json` | the machine document — sorted keys, one entry per candidate artifact |
| `impact-ledger.md` | the same thing readable: one section per finding, counts, and a table an owner can fill in |
| `reassessed/<analysis run id>/promoted-movers.reassessed.json` | one per screen whose funnel could be recomputed (§4) |

**Nothing under `runs/`, `experiments/`, or any analysis directory is ever
written, moved, or edited.** Frozen artifacts and run directories are
immutable; a correction is a NEW artifact with a provenance link back to the
original, which is exactly what the reassessed file is.

---

## §2 — What it reads

The scan walks `runs/` and `adapters/` (depth-capped) and recognises candidates
by their own stamps, never by guesswork:

| candidate | recognised by | `artifactType` |
| --- | --- | --- |
| J-lens probe run | `config.json` `runType: jlens-probe` | `jlensProbeRun` |
| J-lens G0 run | `config.json` `runType: jlens-g0` | `jlensG0Run` |
| J-space run | `config.json` `runType: optvec-jspace` | `optvecJSpaceRun` |
| study run with a J-lens trace | `jlens-readout.jsonl` present | `jlensTraceRun` |
| Python LoRA adapter | sidecar `substrate: python-hf-transformers` | `loraAdapterSidecar` |
| MLX adapter | sidecar `substrate: swift-mlx` (`fine-tune.json`) | `mlxAdapterSidecar` |
| screen funnel | `promoted-movers.json` | `promotedMovers` |
| SAE feature qualification | `sae-feature-qualification.json` | `saeFeatureQualification` |

Each entry carries a **closed** key set:

```
artifactID           the run id, or "<run id>/<file>" for a file inside one
artifactType         from the table above
path                 workspace-relative, POSIX-spelled
producingRevision    the build stamp, verbatim (config.json appVersion,
                     or the sidecar's buildIdentity) — null when absent
pins                 what the artifact says it was measured against
                     (model id, revision, content hashes); absent pins are
                     OMITTED, never nulled
finding              SCI-01 | SCI-02 | SCI-03 | SCI-04
exposure             exposed | unaffected | unknown
evidence             the concrete facts read, one string each
assessment           what the exposure means, and what it does NOT mean
requiredAction       none | reassess | recompute | rerun | resolveProvenance
replacementArtifact  null — the researcher's to fill in
owner                null — the researcher's to fill in
disposition          unresolved, or unaffected where the evidence settles it
```

`rerun` and `recompute` are deliberately different verbs. A readout whose
producer did not retain its transported residuals can only be produced again by
RUNNING the model; a promotion decision is a pure function of an effect table
that is still on disk.

---

## §3 — The classification rules, finding by finding

### SCI-01 — the J-lens full-vocabulary readout (fix `ff4c47a`)

`LensReadout.logits()` projected the transported residual through the model's
output head without the final-norm gain `g = 1 + norm.weight`. Because `g`
varies by coordinate this **reordered** tokens rather than rescaling them, so
top-k identities and ranks off that path are not trustworthy. The watchlist
path already folded `g` into its token rows and is unaffected; energies, null
ratios, and the linearity residual go through `transported`, not `logits`, and
are unaffected too.

There is **no stamp** distinguishing a pre-fix readout from a post-fix one, so
the rule is two-step:

1. *Was the full-vocabulary path armed?* Read from the artifact's own files —
   `probe-topk.csv` for a probe run, `jlens-g0-report.json` for a G0 run, a
   non-empty `topKDelta`/`topKEmergent` table in `jspace.json` for a J-space
   run, a trace row carrying `topKIDs` for a study trace. **Not armed →
   `unaffected`, `none`.** That is a positive finding, not an absence: those
   tables are written only when top-k is armed.
2. *Otherwise, date the producing build.* `config.json` `appVersion` carries
   `<name> <version>+<sha8>`; with `--code-checkout` the sha is tested for
   ancestry against `ff4c47a`. Fix is an ancestor → `unaffected`. Fix is not an
   ancestor → **`exposed`, `rerun`**. No stamp, no checkout, or a commit the
   checkout does not contain → **`unknown`, `resolveProvenance`**.

### SCI-02 — Python LoRA gradient accumulation (fix `d280762`)

The split-mode trainer took HF's per-micro-batch MEAN loss and divided it by
the nominal accumulation factor: a mean of means weighted by the micro-batch
cut, and an incomplete final group scaled down by a factor it never filled. The
corrected objective is `L = sum(token losses in the optimizer-step group) /
count(supervised targets in the group)`.

Read from the adapter sidecar (and its `training-history.json`):

| what the artifact says | exposure | action |
| --- | --- | --- |
| `trainingMode: legacy_inline` | `unaffected` | `none` — that path steps once per iteration and never groups micro-batches |
| `schedule.objective == "tokenMeanPerOptimizerStep"` (or the same key in the history) | `unaffected` | `none` — only the corrected trainer writes it |
| no objective stamp, accumulation **== 1** | `unaffected` | `none` — one micro-batch per group makes the two objectives the SAME number |
| no objective stamp, accumulation **> 1** | **`exposed`** | `rerun` |
| accumulation not recoverable | **`unknown`** | `resolveProvenance` |

Accumulation is read from `schedule.gradientAccumulation`, or derived from
`schedule.effectiveBatchSize / schedule.batchSize`.

**`exposed` here is a SCREEN.** Accumulation > 1 says the wrong objective was
reachable, not that these weights moved by any particular amount: the size of
the difference depends on how uneven the supervised-target counts actually
were, which the sidecar does not record. The entry's `assessment` says this in
so many words, and a reader must not quote the exposure without it. Note also
that a pre-fix checkpoint correctly REFUSES to resume under the current code —
the objective is in the config fingerprint — so retraining is the only route to
weights under the stated objective.

### SCI-03 — the MLX instruction trainer's generation prompt (fix `c692877`)

The Swift/MLX instruction trainer rendered the completed answer with a
generation prompt appended, so the supervised target carried template tokens no
answer contains.

**The sidecar cannot show whether the template appended a suffix.** The
decisive evidence is the chat template's render on the app build that trained
the adapter, and no artifact retains it. Every entry for this finding says so,
including the clean ones. What the sidecar does carry is the training mode, and
the run directory beside it carries the app's build identity:

- `trainingMode` present and not `instruction_chat` → `unaffected`, `none`
  (document-mode targets are raw text and never reach the corrected render);
- `trainingMode: instruction_chat` **and** the producing app build predates
  `c692877` → **`exposed`, `rerun`**;
- `trainingMode: instruction_chat` and the build already contains it →
  `unaffected`, `none`;
- anything else — no mode, no app build stamp, no `--code-checkout`, or a
  commit the checkout cannot resolve → **`unknown`, `resolveProvenance`**, with
  the missing fact named.

The `fine-tune.json` sidecar itself carries no build stamp — the app build
comes from the `config.json` `RunMetadata` writes into the same
`runs/fine-tunes/<stamp>-fine-tune-<slug>/` directory. So an MLX adapter is
datable only when that file is present AND `--code-checkout` was given;
without either, `unknown` is the correct answer and the entry names which half
is missing.

### SCI-04 — flat and single-dose ladders (fixes `00275b1`, `3cb8a59`, `5309b5f`)

Two artifact kinds, both decided from the artifact's own bytes — no build stamp
is needed, so `--code-checkout` changes nothing here.

**`promoted-movers.json`.** A decision with `doseMonotone: true` beside a
null/absent `doseSpearmanRho` is the pre-fix signature and nothing else: under
the current helper a monotone verdict requires two distinct doses AND a nonzero
effect range, and those are exactly the conditions under which Spearman's rho
is defined. Such a document is **`exposed`, `recompute`** — and the verb does
the recompute (§4). Otherwise `unaffected`.

**`sae-feature-qualification.json`.** Loaded with `sae_qualification.load`, then
checked with the promote verb's own `dose_response_violations` (which is built
on `dose_response_geometry`, itself built on `study_stats.dose_monotonicity`) —
reused, never restated. Violations → **`exposed`, `reassess`**; none →
`unaffected`. The record is immutable, so the repair is a NEW qualification
whose declaration matches its rows, linked back to this one, plus a
re-examination of every promotion that cited it. The decision itself stays the
researcher's: the check runs one way, and a record may always claim LESS than
its rows show.

---

## §4 — The reassessed promotions file

For every analysis directory that carries `promoted-movers.json`,
`effect-sizes.csv`, and a `source-run.txt` whose run holds an `experiment.json`
snapshot, the verb recomputes the funnel under the current rule and writes
`reassessed/<analysis run id>/promoted-movers.reassessed.json`.

It recomputes through the **analysis path itself** — the effect rows are
rebuilt from `effect-sizes.csv`, the manifest is loaded from the source run's
own snapshot, and `tasks._promotion_decisions` applies the pinned promotion
rule — so the reassessment differs from the original in exactly one thing: the
code. Re-deriving the funnel by hand would make the comparison meaningless.

Contents:

- `original` — the source path, the sha256 of its exact bytes, and the whole
  original document copied verbatim;
- `reassessed` — `promoted`, `rejected`, and the promotion rule, in the shape
  `promotion.write_promoted_movers` produces;
- `changedVerdicts` — per concept, which of `promoted`, `doseMonotone`,
  `doseSpearmanRho` changed, with the old and new reasons; concepts present on
  only one side are listed too;
- `provenance` — source analysis, source run, manifest content hash, the
  engine version and build commit of the REASSESSING build, the fix commits,
  the timestamp, and a `reconstruction` block.

Two honesty notes live in `reconstruction`:

- **stratified rows are skipped**, exactly as `analyze` skipped them: the
  analysis writes pooled rows first and passes only those to the promotion
  rule, so a stratified row (including a within-item `diagnostic` row, which
  carries no adjusted p at all) is not promotion evidence. The count of skipped
  rows is recorded rather than assumed;
- **`BootstrapCI.replicates` and `.seed` are not columns of
  `effect-sizes.csv`.** They are recorded as `0` and the notes say so. No
  promotion criterion reads them, and inventing plausible-looking values would
  be a provenance forgery.

When reconstruction is impossible — no snapshot, no promotion rule in it, a
CSV missing a required column, an unreadable source run — **nothing is
guessed**. The reason is appended to the ledger entry's `evidence` and the
entry's `requiredAction` becomes `reassess`, which is the researcher's job
rather than the tool's.

---

## §5 — What `unknown` means, and how to resolve it

`unknown` is **not** a soft `unaffected`. It means the artifact cannot say
which build produced it, or cannot say something else the rule needs. Missing
metadata is never evidence of non-exposure — a ledger that quietly rounded
absence down to "fine" would be worse than no ledger.

Resolving one:

1. **No `--code-checkout`.** Re-run with a git checkout of this repository.
   This alone converts most SCI-01 and SCI-03 `unknown`s.
2. **A commit the checkout does not contain.** The producing build was never
   pushed, or the checkout is shallow. Fetch, unshallow, or point at a
   checkout that has it.
3. **No build stamp at all.** The run predates `config.json`'s `appVersion`
   build identity, or the artifact was written by a path that does not stamp
   one. Recover the build from the job record (`jobId`, the Slurm log, the
   deploy's `BUILD_COMMIT`) and record it beside the artifact — as a NEW note,
   never by editing the run.
4. **No recoverable accumulation (SCI-02) or training mode (SCI-03).** Recover
   the training job's configuration. If it cannot be recovered, the adapter
   cannot be cited as evidence about the objective it was trained under, and
   `unknown` is the correct permanent state.

---

## §6 — The researcher's remaining job

The tool classifies; it does not dispose. Three fields are deliberately left
null or `unresolved` for a person:

- **`disposition`** — the tool writes `unaffected` only where the artifact's own
  bytes establish it. Everything else is `unresolved` until a researcher closes
  it, and closing it is a scientific judgement: "this run is superseded", "this
  result did not depend on the top-k table", "this adapter was never used in a
  reported condition".
- **`replacementArtifact`** — the path of the re-run or recomputed artifact
  that supersedes this one. Filling it is what turns the ledger into a
  provenance chain rather than a list of complaints.
- **`owner`** — who is doing it.

A ledger is a snapshot, not a database: re-run the verb whenever the answer
should change (a new `--code-checkout`, a re-run artifact) and the new
directory stands beside the old one. Neither is edited.

Finally, the sentence that belongs at the top of any report built from this
document: **the fixes' regression suites establish covered behaviour going
forward; they establish nothing about the artifacts listed here.** That is the
whole reason this verb exists.
