# AGENTS.md

You are working inside a **SteerLab data workspace**. This file is the
contract: read it before running anything. It describes the folder, the
command surface, the machine-readable output contract, and the refusals you
will hit. Every command below is real; every file shape below is the one the
loaders actually parse.

This is *data*, not code. The SteerLab source tree lives elsewhere. Nothing
here is built or compiled.

---

## 1. What this folder is

A workspace is a plain folder that is its own git repository, created with an
initial commit. Layout:

```
prompts/          git-versioned inputs (see §3)
experiments/      experiment manifests — freezable recipes
runs/             immutable run outputs (gitignored)
catalog/          GENERATED navigation over runs/ (symlinks; gitignored)
adapters/         per-adapter training data and outputs
WORKSPACE.md      the marker file that makes this a workspace
.gitignore        runs/, catalog/, adapters/**/*.safetensors, .DS_Store
```

A folder counts as a workspace if it carries `WORKSPACE.md` or at least a
`prompts/` directory.

An experiment is a **recipe**, not results: it pins inputs by SHA-256 plus the
options used to derive vectors from them, and runs re-derive deterministically.
That pinning is the firewall — settings are chosen and frozen *before* behavior
is measured, so a result cannot be reverse-fitted to the settings that produced
it.

---

## 2. Pointing the CLI at this workspace

Resolution order, exactly: (1) `STEERLAB_WORKSPACE`; (2) `--workspace <dir>`,
or the app's in-process override; (3) the app's persisted choice, honored only
while that directory still exists; (4) a compiled-in development fallback.

**Set the environment variable once at the start of your session** rather than
passing `--workspace` on every call:

```bash
export STEERLAB_WORKSPACE=/abs/path/to/this/workspace
```

Every JSON response carries a top-level `workspace` field — a sibling of
`state`, never something under `result` — naming the root that answered, so you
can tell a wrong-workspace answer from a wrong answer. Check it on your first
command, and compare **resolved** paths: it echoes the path you configured with
symlinks intact, so `/tmp/ws` and `/private/tmp/ws` are one directory
disagreeing on paper. `realpath` both sides before concluding they differ.

If this folder is not a workspace yet — no `WORKSPACE.md`, no `prompts/` — run
`steerlab-cli workspace init <path>` first. `experiment create` will *not* do
this for you: it will build a half-workspace and say nothing.

---

## 3. Where things live

| Path | What goes there | Shape |
|---|---|---|
| `prompts/concepts/<name>/positive.jsonl` | contrastive stimuli, concept-present | `{"text": "…"}` per line |
| `prompts/concepts/<name>/negative.jsonl` | contrastive stimuli, matched control | `{"text": "…"}` per line |
| `prompts/concepts/<name>/validation.jsonl` | held-out probe (§4.4) | `{"text": "…", "expresses": true\|false}` per line |
| `prompts/concepts/<name>/markers.json` | optional marker word list | `{"words": ["…"]}` |
| `prompts/tasks/*.jsonl` | the measured task prompts | `{"id", "prompt", ["options", "target"]}` per line; ids unique |
| `prompts/rubrics/*.md` | judging instruments (Markdown) | prose rubric; pinned by file, not inline text |
| `prompts/batteries/*.jsonl` | capability probes | `{"prompt", "answer", "grading"}`; grading ∈ `exact_number`, `yes_no`, `token_exact`, `exact_normalized`, `regex` |
| `prompts/neutral/corpus.jsonl` | neutral corpus that denominates norm-unit α | `{"text": "…"}` per line |
| `prompts/templates/`, `probes/`, `readers/`, `dev/`, `panels/`, `parsers/`, `generation/`, `emotions/` | other pinned inputs | per their loaders |
| `experiments/<name>/experiment.json` | the manifest | written by the CLI; edit through verbs, not by hand |
| `experiments/<name>/pinned/` | freeze-time snapshot of every pinned input | written by `freeze`; read-only |
| `runs/<timestamp>-<slug>/` | one immutable run | see §7 |

A new workspace is born with generic INSTRUMENTS only — batteries, the neutral
corpus, dev prompts, the parser registry, judge-rubric and data templates, and
the dataset-generation prompts. It carries **no concepts**: `prompts/concepts/`
exists and is empty, and authoring or importing a concept is the first real
step. A worked example (stimuli, task prompts, a battery — a recipe, never
vectors or runs) ships separately as a sample workspace you open on purpose.
**None of the seeded content is study material — adapt it before any run you
intend to keep**, and every concept needs its own `validation.jsonl` or freeze
refuses (§4.4).

---

## 4. The lifecycle, in order

Every command takes `--json`. Use it (§5).

### 4.1 `workspace init`

```bash
steerlab-cli workspace init /abs/path/to/workspace
```

Creates directories, copies seed data, `git init`s, commits. Idempotence is
*not* offered: it refuses if the path is already a workspace.

### 4.2 `experiment create`

```bash
steerlab-cli experiment create <name> --model <model-id> \
  [--revision <commit>] [--description "…"]
```

`--model` is required. `--revision` pins the model commit; without it, freeze
demands one (or auto-pins from the local model cache). The manifest starts in
status **draft**, which is the only status any authoring verb accepts.

### 4.3 `experiment attach`

```bash
steerlab-cli experiment attach <name> <concept>… \
  [--method meanDifference|lat|emotionGrandMean|designatedReference] \
  [--pool-from K] [--reference <concept>] [--corpus a,b,c]
```

Pins each named concept's **current** stimulus hash plus its extraction
options. It also pins the neutral corpus when one exists — that corpus
denominates norm-unit α, so it is a pinned input, not a convenience.

Author the concept directory first. Both stimulus files must exist; if either
is missing, the refusal names *both* missing paths and the row shape in one
message. The row shape is:

```jsonl
{"text": "A sentence that expresses the concept."}
{"text": "Another one, same topic, same length, same register."}
```

`positive.jsonl` and `negative.jsonl` should be content-matched pair-for-pair:
the difference between the files should be the concept and nothing else.
Everything the extraction reads is these two files.

`--project-neutral K` exists and is **legacy, draft-only** — verified and
frozen manifests reject it. Do not use it.

### 4.4 Author `validation.jsonl` — do this before you validate

`prompts/concepts/<name>/validation.jsonl` is the **held-out probe**: scenarios
that evoke the concept (or deliberately do not) *without using its
vocabulary*. It plays no role in extraction. It is the only evidence that the
extracted direction moves anything other than the words it was built from.

```jsonl
{"text": "A never-named scenario that should elicit the concept.", "expresses": true}
{"text": "A matched scenario that should not.", "expresses": false}
```

**Author it before `attach`.** `attach` pins this file's hash — or an explicit
"absent" when there is none — so a set that appears afterwards is a `verify()`
violation ("appeared after attach (pinned as absent) — re-attach to pin it")
that blocks `validate` and `freeze` alike. Working order: **author → attach →
validate → freeze**. Found out late: author the file, re-run `experiment attach
<name> <concept>…` for those concepts, then `validate`, then `freeze`. On a
frozen manifest there is no repair — duplicate first.

`validate` builds a per-concept **vacuity ledger**: every pinned concept owes a
scored held-out probe and is struck off only when one is actually scored. A
concept with no probe leaves the run **vacuous** — it exits 0 and looks
identical on the surface, but it carries the stamp, and `freeze` refuses it
under the `validateEvidence` gate, naming the missing file paths. So: no
`validation.jsonl` → `validate` "succeeds" → `freeze` refuses. In `--json` mode
`validate` reports `result.vacuous`, `result.vacuousConcepts[]`, and one
`vacuousValidation` advisory per concept. Read those, not the exit code.
Deleting the set after `validate` is a `verify()` pin violation, not a way out.

### 4.5 `pin-prompts` — the measured task

```bash
steerlab-cli experiment pin-prompts <name> prompts/tasks/<file>.jsonl
```

Pins `taskPromptsFile` + `taskPromptsHash` (SHA-256 of the raw bytes) and
parses the file with the run loop's own parser, so a file the run would refuse
is refused here instead of at generation time. Rows:

```jsonl
{"id": "item-01", "prompt": "…", "options": ["a", "b"], "target": "b"}
```

`options` + `target` are optional; when present the answer-token/logprob
instrument can score the item deterministically, which is the preferred
instrument for categorical outcomes — but the instrument is an explicit
declaration, never inferred from the items: run
**`set-instruments <name> answerTokenLogprob`** (draft-only) or the run
records prose and `parsedChoice` only. `pin-prompts` warns with a
`choiceItemsWithoutInstrument` advisory when items carry `options` and no
direct-scoring instrument is declared. Ids must be unique. `""` clears the
pin.

**`responseFormat` is optional, and absence is fine.** The instrument reads
any item whose `options` is non-empty; `target` is not consulted at dispatch
on either engine. The field only ever *subtracts*: an option-bearing item
that explicitly declares `"responseFormat": "json"` or `"freeText"` is
refused at run start under the `responseFormat` gate, and when the manifest
declares an `outcomeInstrumentScope` only rows whose declared format the
scope lists are measured — so in a mixed file `"label"` becomes required on
the rows you want scored, and only then. Do not add `"responseFormat":
"label"` to a file that already runs; nothing asks for it.

### 4.6 `pin-rubric` — the judging instrument

```bash
steerlab-cli experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md \
  --judges <name>:<kind>[:<model>[:<provider>]][,…]
```

Pins `judgeRubricFile` + `judgeRubricHash`, optionally replaces the judge
panel, and writes the explicit `evaluation` declaration the pin pair implies.
Kinds are `claude`, `local`, `openrouter`. A blank model field is *absent*, not
empty: a local judge then resolves to the study model at its pinned revision; a
`claude` judge to the default judge model. The fourth field pins a serving
provider and is only legal on `openrouter`.

The judge **name is a label, never a model id.** Fewer than two judges is a
non-blocking `judgePanelTooSmall` advisory here and a `judgeValidity` freeze
gate later. Inline rubric text is draft-only and cannot freeze — pin a file.

The two must also be **distinct**: identity resolves to (kind, model,
provider), so `--judges a:local,b:local` with both model fields blank resolves
twice to the study model at temperature 0 — one judge agreeing with itself by
construction — and refuses at freeze under `judgeValidity`. Vary the kind, the
model, or the provider.

### 4.7 `declare-condition` — the arm

```bash
steerlab-cli experiment declare-condition <name> <condition> \
  --slots <concept>:<layer>:<alpha>[:add|ablate][,…] \
  [--band-width K] [--alpha-units norm|raw] \
  [--control randomMatchedNorm|randomDirectionAblation]

steerlab-cli experiment declare-condition <name> <condition> --baseline
```

**Without at least one condition, a concept study runs the implicit baseline
alone and measures nothing.** A multi-slot condition *is* the linear mix
`h + Σ αᵢ·vᵢ` and hashes as a single condition. Every named concept must
already be attached. `--baseline` and `--slots` are exclusive.

`--alpha-units norm` (the default) denominates α by the residual-stream norm at
that layer on the pinned neutral corpus — that is what makes α comparable
across concepts. Use `raw` only when you know why.

`--control` substitutes a deterministic random direction into the same slots,
giving you the matched-norm control arm.

### 4.8 `validate`

```bash
steerlab-cli experiment validate <name>
```

Extracts (or reuses) the vectors and scores the held-out probes; writes a run
directory. This is the evidence `freeze` looks for, and it must match the
manifest's *exact* pins — model + revision, concepts and their options, neutral
corpus, and the run substrate. Change any pin and the evidence stops matching;
re-validate.

Those pins *are* the evidence's key; the experiment's name is not among them,
so evidence is shared across the workspace — a `duplicate`, or any fresh
experiment with matching model, revision, concepts, options and neutral corpus,
freezes on validation it never ran. A passing `validateEvidence` gate is not by
itself proof that *this* experiment produced the evidence.

`extract` runs the derivation alone if you want it separately. Both load the
model.

### 4.9 `freeze` — one-way

```bash
steerlab-cli experiment freeze <name> [--force] [--run-substrate local|server]
```

Freeze verifies every pin, stamps the manifest's content hash and the
workspace git commit, snapshots every pinned input into
`experiments/<name>/pinned/` — concept stimulus directories, task prompts,
judge rubric, capability battery, reasoning-style taxonomy, human tables, and
the neutral corpus that denominates norm-unit α — writes
`experiments/<name>/preregistration.md` beside the manifest, and makes the
manifest read-only. There is no unfreeze. **Iterate by `experiment duplicate
<name> <new-name>`, never by editing.**

The snapshot is taken at freeze time only: it is the no-git reproducibility
floor, not a live mirror. A study frozen before a pin joined the snapshot
keeps whatever its own freeze wrote.

`--run-substrate server` matches the evidence gates against evidence produced
on the *server* engine — the substrate the measured runs will execute on —
instead of this one.

Two classes of check, and the difference matters:

- **`verify()` pin integrity** — always runs, **never skippable**, owns no gate
  id. Drift in any pinned file's bytes is a violation here. `--force` does not
  reach it.
- **The seven gates below** — force-skippable, each with a stable id.

#### The seven freeze gates

| Gate id | What it demands | Repair |
|---|---|---|
| `revision` | a pinned, immutable model commit — not absent, not symbolic | `create --revision <commit>`, load the model once so it is cached, or pin the commit the symbolic ref resolves to |
| `measurementPins` | pins that determine *what is measured* are present and valid (e.g. a loadable study dtype) | repoint the invalid pin at a loadable value |
| `validateEvidence` | a `validate` run matching the exact pins on the run substrate, **and** that evidence is not vacuous (§4.4) | author the named `validation.jsonl` files, **re-attach** their concepts, then `experiment validate <name>` (§4.4; the refusal's own `repairAction` names this full sequence) |
| `variantValidity` | attached variants carry hashed adapter weights and a pinnable dataset manifest | re-save the variant with hashed weights and re-attach it |
| `batteryEvidence` | each variant condition has scope-matched capability-battery evidence | re-run `experiment validate <name>` (each variant condition runs the pinned battery) |
| `judgeValidity` | a rubric **file** and ≥ 2 distinct judges the pipeline can actually run | `experiment pin-rubric <name> prompts/rubrics/default-paired-v1.md --judges a:local,b:claude` |
| `gitClean` | every pinned input is committed in the workspace git repo | commit the pinned inputs (freeze auto-commits in most workspaces; this gate speaks when it could not) |

In `--json` mode a refusal gives you `error.gate` (the gate whose message is in
`error.reason`), `error.gates[]` (**every** gate that failed, not just the
first), and `error.repairAction`. Fix the gate; do not retry the same command.

**`--force` skips the seven gates and stamps the manifest** `freezeForced: true`
plus `forcedGatesSkipped: [<gate ids>]`, and emits one `freezeGateSkipped`
advisory per gate. A forced freeze is permanently non-citable — but checkably
so, by stamp. **Do not `--force` to get past a gate.** If a human explicitly
asks for it, do it and report exactly which gates were skipped.

### 4.10 `run`

```bash
steerlab-cli experiment run <name> [--prompts prompts/tasks/<file>.jsonl]
```

Generates under every declared condition and writes an immutable run directory
containing the manifest snapshot + content hash, `generations.jsonl`,
`battery.jsonl`, computed metrics, and a canonical `config.json`.

`run` refuses, before the model loads, a concept-bearing manifest with no
injection, variant, or SAE arm. That refusal is not an obstacle: it is the
firewall telling you the study would have measured nothing. Declare a condition
(§4.7) or promote an agent (§4.12), or declare the baseline-only study
explicitly if that is genuinely what you want.

### 4.11 `analyze`

```bash
steerlab-cli experiment analyze <name> [--allow-unverified-epoch]
```

Pure CPU, no model load. Paired-to-baseline effect sizes — bootstrap CIs and
Wilcoxon — over the newest completed run; writes `effect-sizes.csv` and folds
`effectSizes` into `report.json`.

Guarded by the **epoch guard**: the run's stamped experiment hash must equal
the live manifest's content hash, or the verb refuses. `--allow-unverified-epoch`
bypasses only *unstamped legacy* runs and stamps `epochUnverified` on the
result. The guard is per-engine — analyze a run on the engine that produced it.

Zero effect-size entries is reported as an `emptyAnalysis` advisory, not a
failure. It means the source run had no non-baseline condition. Check for it.

### 4.12 `evaluate`, `sweep`, `promote`, `confirm`

**`evaluate <name> [--run <dir>] [--allow-unverified-epoch]`** — paired-judge
evaluation of a completed run through the manifest's pinned rubric and judges,
writing a new evaluation directory beside the source run, which is never
mutated. Same epoch guard as `analyze`; defaults to the newest completed run.

**`sweep <name>`** — walks layer × α on a dev split and records a
recommendation per concept, selecting by the manifest's declared criterion
(`sweep.selection`: an objective, capability/coherence constraints, an optional
matched-norm-random control margin). Objectives are `markerDensity`,
`judgeScore`, `logprobShift`. Marker density measures surface vocabulary — it
is a manipulation check, not a selection objective for any study whose outcome
is a decision rather than prose. Declare the rule headlessly with
**`set-sweep-selection <name> --objective judgeScore|logprobShift|markerDensity …`**
(draft-only); with no declared rule the sweep defaults to `markerDensity` and
says so with a `sweepSelectionDefaulted` advisory — on a choice-task prompt
set, treat that advisory as a stop sign. The sweep's `--json` result carries
the run directory and each concept's winning cell, criterion, and metrics;
on a frozen manifest it records recommendations only
(`sweepRecommendationsOnly` advisory). Loads the model.

**`promote <name> <concept> [--agent-name N] [--cell L:α --reason "why"]`** —
mints a variant artifact (an "agent") from the sweep-selected cell with a
`promotion` birth certificate (`promotedBy: "criterion"`). `--cell` is the loud
manual override: it stamps `promotedBy: "manualOverride"`, warns, and still
**requires evidence that a sweep ran for the concept** — promotion with no
sweep at all is refused. Hand-created variants stay legal but surface as freeze
advisories. Pure CPU.

**`confirm <name> --agent <A> [--deltas 0.2,0.5] [--no-control]`** — declares a
perturbation policy around a promoted agent's anchor cell, which expands
mechanically into ordinary hashed conditions on the draft manifest. Pure CPU.

**`data check <name>`**, at any point, returns the full classified readiness
list: every requirement, its status, the **path you must author**, and the
rationale. Fastest way to find what is missing. Blockers are a refusal:
`state: "refused"`, exit **65 in both modes** (the one verb whose human exit
has migrated — §5).

### 4.13 The rest of the surface, and how to find it

<!-- Draft-only: this document is the human source of truth; the shipping copy is the AgentContract constant, held byte-equal to it by test. -->

**`--help` is how you discover the surface**, at three levels, on both CLIs:

```bash
steerlab-cli --help                      # every family
steerlab-cli experiment --help           # that family's verbs, one line each
steerlab-cli experiment attach --help    # one verb's positionals and flags
```

It is a declared flag on **every** verb, it runs **nothing**, and it exits 0
in both modes — so it is always safe to ask, including on a verb that would
otherwise write a manifest. With `--json` the same page comes back as data in
`result`, so you never have to parse the columns. Each page ends with the
exit-code line, and a verb page names every flag it accepts with its purpose
and its argument's shape, including closed vocabularies (`set-instruments`
prints the legal instrument names).

Secondary: running a family with no verb (`steerlab-cli experiment`) refuses
and lists every verb it accepts. That roster answers "which verbs exist";
`--help` answers "and what do they take", so reach for `--help` first.

**`list`** — every experiment here with its status, model, concepts, condition
count and `freezeHash`. The cheapest orientation command; run it first.

**`verify <name>`** — re-hashes every pinned input against the manifest and
refuses (`pinDrift`, one line per drifted pin) if a byte moved. What the other
verbs run for you, available alone, safe at any status.

**`set-style-taxonomy <name> prompts/taxonomies/<file>.json`** — pins a
reasoning-style taxonomy (path + hash) on a draft. No pin, no reasoning-style
scoring; drift after pinning is a verify violation like any other.

**`rescore-style <name> [--run <dir>]`** — recomputes reasoning-style features
for a completed run through that taxonomy into a **new** run directory, never
touching the source. Epoch-guarded like `analyze`. Pure CPU.

### 4.14 Multi-agent studies: casting a panel

A **panel** is a scenario under `prompts/panels/` — roles, turns, visibility,
case materials. A *semantic* panel binds no model to any seat, which makes it
deliberately unrunnable: it has to be **cast** first, and casting is the step
that binds the study's model and sampling settings to one seat assignment.

```bash
steerlab-cli panel list
steerlab-cli panel check <path-or-name>
steerlab-cli panel compile <path-or-name> --experiment <name> \
  [--seat <seat>=<agent-artifact-path>]… \
  [--model <id>] [--temperature <t>] [--max-tokens <n>] [--file-slug <slug>]
```

`compile` writes the bound scenario to `prompts/panels/compiled/` and pins it
into the **draft** in one step — both the scenario pins and the provenance pair
recording which semantic panel it came from. It also declares the study
multi-agent, because a panel scenario is read only by that run path.

**Seats are keyed by the scenario's agent `id`**, not its display name — `list`
reports the ids, and an id the panel does not have refuses with the list of the
ones it has. Seats you do not name stay **baseline**, and an all-baseline
casting is the control composition, not an absence. A `--seat` value is an
agent artifact path under `runs/model-variants/`; its hash is read from the
file.

`--model`, `--temperature` and `--max-tokens` **default from the manifest** and,
when given, are written to it before the compile: the manifest stays the one
place those three are decided. `check` validates a *bound* panel and reports
its advisories; a semantic panel fails that check by design, and `compile` is
the answer.

---

## 5. The machine contract

**Pass `--json` on every command.** In JSON mode:

- **Exactly one JSON document on stdout.** Every diagnostic, progress line,
  warning, and human report goes to stderr. No ANSI.
- Keys are sorted, dates are ISO-8601, there is exactly one trailing newline.
- `--json` is honored even when argument parsing itself fails.
- `--out <path>` also writes the document to a file. (`--json <path>` is the
  deprecated spelling on the one verb that had it; it warns on stderr.)
- **Hashes are full.** The human lines elide them; the document never does.
  `freezeHash`, `taskPromptsHash`, `judgeRubricHash` and friends are complete
  in `result`.

Document shape:

```jsonc
{
  "schemaVersion": 1,          // the ENVELOPE's version, never the payload's
  "verb": "experiment freeze",
  "engine": "…",               // which engine answered
  "state": "refused",          // AUTHORITATIVE
  "changed": false,            // did this mutate durable state
  "observedAt": "2026-01-01T00:00:00Z",
  "message": "…",              // one sentence for a human
  "workspace": "/abs/path",    // which data root answered
  "advisories": [ { "code": "…", "detail": "…" } ],   // omitted when empty
  "nextAction": { "verb": "experiment validate demo", "requiresHuman": false,
                  "missingPermissionFlags": [], "detail": null },  // successes
  "error": { "code": "freezeGateFailed", "gate": "validateEvidence",
             "gates": ["validateEvidence", "judgeValidity"],
             "reason": "…", "repairAction": "…" },
  "result": { /* per-verb payload */ }
}
```

`schemaVersion`, `verb`, `engine`, `state`, `changed`, `observedAt`, and
`message` are always present. `workspace`, `advisories`, `nextAction`, `error`,
and `result` appear only when they have something to say — a missing key is a
straight answer, not a null.

### State and exit codes

**The JSON `state` is authoritative; the exit code is a convenience.**

| `state` | Exit | Meaning |
|---|---:|---|
| `ready` | 0 | the requested target is reached |
| `planned` / `running` | 0 | work remains / in progress |
| `okWithAdvisories` | 0 | succeeded, and `advisories[]` is non-empty |
| `needsHumanAuthentication` | 10 | a person must authenticate at their own terminal |
| `needsApproval` | 11 | a mutation needs its explicit `--allow-…` flag |
| `pending` | 12 | valid asynchronous work is in flight; repeat the command |
| `degraded` | 13 | retryable: a layer could not be read |
| `blocked` | 64 | malformed invocation or unusable configuration |
| `refused` | 65 | a gate declined a well-formed request against a healthy system |
| `notFound` | 66 | the named experiment, run, or panel does not exist — or a named artifact could not be read at all |
| `failed` | 70 | non-retryable operational failure |

Two live caveats. **These codes are the `--json`-mode codes.** Without
`--json`, most failures still exit `1`; two verbs differ, and both differ in
both modes — `data check` exits `65` for blockers, and `vectors compare`
exits `1` when it compared and diverged but `2` when it could not compare at
all. One more reason to always pass `--json`. And an undeclared flag is exit
`64` in **both** modes, before the verb does any work: flags are parsed
strictly against a per-verb table, so a typo cannot silently change what a
study means.

Refusals are typed everywhere, not only at freeze: a lifecycle refusal
carries `error.code == error.gate` from a second closed vocabulary
(`statusImmutable`, `pinDrift`, `missingPrerequisite`,
`promotionEvidence`, …), while freeze refusals keep
`code: "freezeGateFailed"` with the gate id in `error.gate`. Either way,
`error.repairAction` is an executable command sequence — run it, then retry.

### Advisories

`advisories[]` never changes the exit code. An advisory is something you should
know that did not stop the verb: a skipped freeze gate, a vacuous validation, a
one-judge panel, an empty analysis. **Read them.** Treating them as failures
will make you refuse to walk a legitimate lifecycle; ignoring them will make
you produce results that are stamped as not citable.

### Refusals

`state: "refused"` means a gate declined a well-formed request. It is not a
transient error. **`error.repairAction` is the field to read**: today
`nextAction` is emitted on *successes*, where it names the next lifecycle step,
and a refusal carries `error` without one. **Never retry a refusal without
performing the repair first.**
---

## 6. Immutability

- **`runs/` is append-only.** Never edit, overwrite, or delete a run directory.
  A run carries enough to rebuild its tables without rerunning the model, and
  that is only true if nobody touches it. `runs/` is gitignored.
- **Three subtrees under `runs/` are deliberately mutable libraries**:
  `runs/model-variants/`, `runs/neutral-pcs/`, `runs/jlens-lenses/`. A promoted
  agent's artifact is editable in place there; frozen studies are protected by
  the manifest snapshot and the artifact hash, not by the directory.
- **Frozen manifests are read-only** — the verbs that WRITE the manifest
  (`attach`, `pin-prompts`, `pin-rubric`, `declare-condition`,
  `set-style-taxonomy`, `confirm`) refuse on a frozen or complete one;
  `duplicate`, then edit the copy. The verbs that only read it stay legal,
  including two that surprise people: **`sweep`**, which records its
  recommendations in its own run directory rather than in the manifest, and
  **`promote`**, which mints its agent into the mutable `runs/model-variants/`
  library. Both are confirmation-stage gestures a frozen study must still be
  able to make, and neither can change what the study means.
- **`experiments/<name>/pinned/`** is the freeze-time snapshot of every pinned
  input — the reproducibility floor when git is unavailable. Do not edit it.
- Do not hand-edit `experiment.json`. Its bytes *are* the content hash; an edit
  that bypasses the verbs surfaces as a verify violation, which is the good
  outcome, or as a silently different study, which is not.

---

## 7. Which engine runs what

Two compute engines read and write the same artifacts. Measured runs on the
**local** engine are **greedy only**: the study runner requires
`temperature == 0` and rejects more than one seed, because the local generator
cannot pin a per-run sampling seed. `seeds` is recorded for provenance and does
not affect local generation — every local generation record stamps
`seedInert: true`. A manifest needing `temperature > 0` or
`samplesPerItem > 1` belongs on the **server** engine, which seeds per record
and writes one record per (condition, prompt, sampleIndex). For categorical
outcomes prefer the answer-token/logprob instrument over sampled prose on
either engine: deterministic and temperature-free.

Activations do not transfer between engines. Vectors must be **re-extracted and
re-validated on whichever engine a study runs on** — the parity claim is
structural, not byte identity. Stimulus and corpus SHA-256 hashes *are*
identical across engines, which is what makes cross-engine comparison possible
at all. `freeze --run-substrate` and the evidence gates enforce this: evidence
from the other engine will not satisfy them.

**Authoring is Mac-authority by design.** The workspace on the Mac is the
source of truth; the server is a runner and cache. `create`, `attach`, the
`pin-*`/`declare-*`/`set-*` verbs, `panel compile`, and `freeze` run on the
local CLI;
execution and analysis (`run`, `evaluate`, `analyze`, `sweep`) answer
identically-shaped envelopes on either engine, and the epoch guard keeps
`analyze` on the engine that produced the run. A server-side refusal whose
repair is an authoring act names the local verb on purpose — go author
there, then submit. Some verbs exist on one engine only; asking the server
for one of them is refused with `error.code: "macAuthorityVerb"` (no
`error.gate` — it describes the engine, not the study) and an
`error.repairAction` spelling the local command. Do not emulate the verb;
run the repair where it belongs.

---

## 8. Remote execution, and where the depth is

When this workspace's study runs on a cluster, the same contract holds — and
one rule outranks convenience: **never write a bare sbatch script.** Submit
through the engine's rendered path (`steerlab-server study submit …`, or the
app), which requests node-scratch via the site's gres and arms the cleanup
trap; a hand-rolled script silently gets neither, and stale node-scratch is
how clusters come to email their operators about you. If a submission need
seems to force a hand-roll (a dependency chain, a resume), that is a missing
verb to report, not a reason to bypass the renderer.

The cluster lifecycle has first-class verbs — prefer them to raw `ssh`:
`steerlab-cli cluster push` (deploys the engine AND re-stamps its build
identity), `cluster ensure`, `cluster tunnel open`, `cluster remote --site
<id> …`, and `cluster import --site <id>` (verified, never-purging run
import). Site profiles live in the SteerLab home's `Sites/cluster-sites/`
registry — never invent one; ask the researcher for theirs.

This file is deliberately self-contained for study work, but it is not the
whole reference. The code checkout (normally a sibling of this workspace's
SteerLab home, e.g. `~/SteerLab/<checkout>/`) carries the depth:
`docs/CLI-REFERENCE.md` — every verb, flag, envelope, and refusal on BOTH
command lines, generated from the parsers, so it is never stale —
plus `docs/ONBOARDING.md` (§9 is specifically about driving SteerLab as an
agent), `docs/CONDUCTING-A-STUDY.md`, and `SECURITY.md`. When a verb here
seems to lack a flag you need, check CLI-REFERENCE before improvising.

---

## 9. What not to do

- **Do not parse prose.** Use `--json` and read `state`, `error.code`,
  `error.gate`, `error.gates[]`, `advisories[].code`, and `result`.
- **Do not retry a `refused` (65) without performing the repair.** It will
  refuse identically. Read `error.repairAction`.
- **Do not `--force` a freeze to get past a gate.** It is stamped, loud, and
  permanently non-citable. Fix the gate instead. If a human explicitly asks for
  a forced freeze, report every id in `forcedGatesSkipped`.
- **Do not write into `runs/`**, and do not edit a frozen manifest or a
  `pinned/` snapshot.
- **Do not edit a manifest to iterate.** `duplicate`, then edit the copy.
- **Do not treat an advisory as a failure**, and do not ignore one.
- **Do not skip `validation.jsonl`.** A study whose vectors were never probed
  on held-out material measures its own stimulus vocabulary.
- **Do not select a steering cell on marker density** for any study whose
  outcome is a decision rather than prose. It is a manipulation check.
- **Do not cite seeded or sample content.** It is there to be modified.
- **Do not guess at a file shape.** The shapes in §3 are the ones the loaders
  parse; a wrong key is refused with the expected key named.
