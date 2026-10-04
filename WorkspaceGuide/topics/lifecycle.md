# The study lifecycle, step by step

<!-- client: all -->

Every command takes `--json`. Use it (`{{cli}} workspace guide contract`).
The steps are in working order. A step that loads a model or spends compute
needs the researcher's approval first.

## 1. `workspace init`

```bash
{{cli}} workspace init /abs/path/to/workspace
```

<!-- client: mac -->

Creates directories, copies seed data, `git init`s, commits. Idempotence is
*not* offered: it refuses if the path is already a workspace.

<!-- client: python -->

Creates directories, copies seed data and, when git is available, initializes
a repository and commits (`--no-git` skips that). It refuses unless the
destination is new or empty.

<!-- client: all -->

## 2. `experiment create`

```bash
{{cli}} experiment create <name> --model <model-id> \
  [--revision <commit>] [--description "…"]
```

<!-- client: mac -->

`--model` is required. `--revision` pins the model commit; without it, freeze
demands one (or auto-pins from the local model cache). The manifest starts in
status **draft**, which is the only status any authoring verb accepts.

<!-- client: python -->

`--model` is required. `--revision` pins the model commit. This client never
guesses a revision from a model cache: pin it here, or later with
`steerlab experiment pin-revision <name> <revision>`, or freeze refuses under
the `revision` gate. The manifest starts in status **draft**, which is the
only status any authoring verb accepts.

<!-- client: all -->

## 3. `experiment attach`

```bash
{{cli}} experiment attach <name> <concept>… \
  [--method meanDifference|lat|emotionGrandMean|designatedReference] \
  [--pool-from K] [--reading-position '<label>'] \
  [--extraction-rendering '<json>'] \
  [--reference <concept>] [--corpus a,b,c]
```

`--reading-position` pins WHERE the residual stream is read, by name (`last
content token`, `content offset 2`, `mean content from token 0`, `offset from
end 3`, …). `--pool-from K` is the legacy spelling of one of them (`mean from
token K`); declaring both is refused, and a template-aware role declared under
a raw rendering is refused here rather than hours later on a GPU.

`--extraction-rendering` pins HOW those stimuli reach the model before they
are read: a JSON object (`{"mode": "raw"|"chatTemplate", "voice":
"user"|"assistant", "addGenerationPrompt": true|false}`), or the bare mode
word. Raw and chat-template renderings of the same stimuli give different
directions, so the rendering is recipe identity exactly as the reading
position is. **Absent ≡ raw ≡ today's bytes**, and an explicit `raw` — or an
explicit `voice: "user"` — canonicalizes away, so a silent legacy declaration
and a newly explicit one compare equal. `voice: "assistant"` renders the
stimulus as the model's OWN output, extracted by template subtraction; under
it `addGenerationPrompt` and `systemPrompt` reach nothing and are typed
refusals at declaration time.
The template-aware reading positions above are the other side of this pairing:
under a raw rendering there is no template for them to find, which is the
refusal `attach` fires.

<!-- client: mac -->

The Mac's local engine refuses the assistant voice — a typed engine asymmetry
— so run assistant-voice cells on the server engine.

<!-- client: all -->

Pins each named concept's **current** stimulus hash plus its extraction
options. It also pins the neutral corpus when one exists — that corpus
denominates norm-unit α, so it is a pinned input, not a convenience.

Author inputs for the chosen method before attaching:

- `meanDifference` and `lat`: `prompts/concepts/<concept>/positive.jsonl`
  and `negative.jsonl`. Both files are required. Each row is a JSON object
  with a `text` string, for example `{"text":"A concept-relevant statement."}`.
  Match topics, length and register to reduce confounds. LAT uses aligned
  pairs; mean difference compares class means and does not require equal counts.
- `emotionGrandMean`: `prompts/emotions/<concept>/stories.jsonl` for every
  member of the declared `--corpus a,b,c`. Rows contain `text`. The comparison
  population is the pooled corpus, including the target concept; membership
  and file hashes are part of the recipe.
- `designatedReference`: the target and `--reference <concept>` each supply
  `prompts/emotions/<concept>/stories.jsonl`. Both story corpora are pinned;
  the reference must be chosen for the research question, not copied by default.

Rendering, reading position, model revision and applicable neutral-norm inputs
also affect extraction. Record and review those declarations alongside the data.
Matched inputs and random controls support interpretation; they do not prove
that a direction isolates the intended construct.

<!-- client: mac -->

`--project-neutral K` exists and is **legacy, draft-only** — verified and
frozen manifests reject it. Do not use it.

<!-- client: python -->

`steerlab concept import <name> --file <path> [--side positive|negative]`
reads stimuli out of a JSONL, CSV, or plain-text file and saves them into the
concept's datasets. Single texts need `--side`: a stimulus filed on the wrong
pole inverts the direction the vector points, and nothing downstream would
say so.

<!-- client: all -->

`{{cli}} experiment detach <name> <concept>…` is the inverse: it removes
each named concept's pin from a draft, all-or-nothing, and refuses
(`conceptInUse`) while any declaration still names one — an injection
condition's slot, a per-concept sweep-selection instrument, a variant
condition's `fromPromotion`, a perturbation policy. Remove or re-declare those
first. Re-pointing a draft at a different concept is `detach` then `attach`.

## 4. Author `validation.jsonl` — do this before you validate

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
under the `validateEvidence` gate, naming the missing file paths.

<!-- client: mac -->

So: no `validation.jsonl` → `validate` "succeeds" → `freeze` refuses. In
`--json` mode `validate` reports `result.vacuous`, `result.vacuousConcepts[]`,
and one `vacuousValidation` advisory per concept. Read those, not the exit
code.

<!-- client: python -->

So: no `validation.jsonl` → validation "succeeds" on the runner → `freeze`
refuses. The validation run that comes home carries the same vacuous stamp;
read it before you freeze.

<!-- client: all -->

Deleting the set after `validate` is a `verify()` pin violation, not a way out.

## 5. Pin the measured task prompts

<!-- client: mac -->

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

<!-- client: python -->

```bash
steerlab experiment inspect <name> --json     # returns manifestFileSHA256
steerlab experiment import-prompts <name> --file <file>.jsonl \
  --manifest-sha256 <digest>
```

Imports full JSONL records as an immutable input version and pins it to the
reviewed draft; a changed manifest refuses, so inspect again and review before
retrying. Rows:

```jsonl
{"id": "item-01", "prompt": "…", "options": ["a", "b"], "target": "b"}
```

`options` + `target` are optional; when present the answer-token/logprob
instrument can score the item deterministically, which is the preferred
instrument for categorical outcomes — but the instrument is an explicit
declaration, never inferred from the items: run
`steerlab experiment set-protocol <name> --set outcomeInstruments='["answerTokenLogprob"]'`
(draft-only) or the run records prose and `parsedChoice` only. Ids must be
unique.

<!-- client: all -->

**`responseFormat` is optional, and absence is fine.** The instrument reads
any item whose `options` is non-empty; `target` is not consulted at dispatch
on either engine. The field only ever *subtracts*: an option-bearing item
that explicitly declares `"responseFormat": "json"` or `"freeText"` is
refused at run start under the `responseFormat` gate, and when the manifest
declares an `outcomeInstrumentScope` only rows whose declared format the
scope lists are measured — so in a mixed file `"label"` becomes required on
the rows you want scored, and only then. Do not add `"responseFormat":
"label"` to a file that already runs; nothing asks for it.

## 6. `declare-condition` — the arm

<!-- client: mac -->

```bash
steerlab-cli experiment declare-condition <name> <condition> \
  --slots <concept>:<layer>:<alpha>[:add|ablate][,…] \
  --alpha-units norm|raw [--band-width K] \
  [--control randomMatchedNorm|randomDirectionAblation]

steerlab-cli experiment declare-condition <name> <condition> --baseline \
  --alpha-units norm|raw
```

<!-- client: python -->

```bash
steerlab experiment declare-condition <name> <condition> \
  --slots <concept>:<layer>:<alpha>[,…] --alpha-units norm|raw [--band-width K]

steerlab experiment declare-condition <name> <condition> --baseline \
  --alpha-units norm|raw

steerlab experiment remove-condition <name> <condition>
```

<!-- client: all -->

**Without at least one condition, a concept study runs the implicit baseline
alone and measures nothing.** A multi-slot condition *is* the linear mix
`h + Σ αᵢ·vᵢ` and hashes as a single condition. Every named concept must
already be attached. `--baseline` and `--slots` are exclusive.

`--alpha-units` is required on every arm, the baseline included; there is no
default. `norm` denominates α by the residual-stream norm at that layer on the
pinned neutral corpus — that is what makes α comparable across concepts. Use
`raw` only when you know why.

<!-- client: mac -->

`--control` substitutes a deterministic random direction into the same slots,
giving you the matched-norm control arm.

<!-- client: all -->

## 7. `validate`

<!-- client: mac -->

```bash
steerlab-cli experiment validate <name>
```

Extracts (or reuses) the vectors and scores the held-out probes; writes a run
directory. This is the evidence `freeze` looks for, and it must match the
manifest's *exact* pins — model + revision, concepts and their options, neutral
corpus, and the run substrate. Change any pin and the evidence stops matching;
re-validate.

<!-- client: python -->

This client loads no model, so validation runs on a runner, from a packaged
bundle. `steerlab run` needs a frozen study, so a draft takes the steps one at
a time:

```bash
steerlab bundle package <name>
steerlab runner upload <bundle.tar.gz> --runner <url>
steerlab runner submit --runner <url> --bundle-path <printed-path> \
  --bundle-sha <printed-digest> --verb validate
steerlab runner jobs <job-id> --runner <url>
steerlab runner evidence <job-id> --out <file.tar.gz> --runner <url>
steerlab bundle import <file.tar.gz> --sha256 <digest>
```

The runner extracts (or reuses) the vectors and scores the held-out probes; the
run directory comes home in the evidence bundle. This is the evidence `freeze`
looks for, and it must match the manifest's *exact* pins — model + revision,
concepts and their options, neutral corpus, and the run substrate. Change any
pin and the evidence stops matching; validate again.

<!-- client: all -->

Those pins *are* the evidence's key; the experiment's name is not among them,
so evidence is shared across the workspace — a `duplicate`, or any fresh
experiment with matching model, revision, concepts, options and neutral corpus,
freezes on validation it never ran. A passing `validateEvidence` gate is not by
itself proof that *this* experiment produced the evidence.

<!-- client: mac -->

`extract` runs the derivation alone if you want it separately. Both load the
model.

<!-- client: python -->

`--verb extract` runs the derivation alone if you want it separately. Both
load the model on the runner. `runner submit` is not idempotent: after a
timeout, look with `steerlab runner jobs --runner <url>` before submitting
again.

<!-- client: all -->

## 8. Verify and freeze

`{{cli}} experiment verify <name>` re-hashes every pinned input. Freezing is
one-way and has its own topic: `{{cli}} workspace guide freeze`.

## 9. `run`

<!-- client: mac -->

```bash
steerlab-cli experiment run <name> [--prompts prompts/tasks/<file>.jsonl]
```

Generates under every declared condition and writes an immutable run directory
containing the manifest snapshot + content hash, `generations.jsonl`,
`battery.jsonl`, computed metrics, and a canonical `config.json`.

<!-- client: python -->

```bash
steerlab run <name> --runner <url>
```

Takes a **frozen** study to a runner and brings its verified evidence home:
it packages the bundle, uploads it, submits, waits, downloads the evidence,
verifies it, and imports it into this workspace. The runner generates under
every declared condition and writes an immutable run directory containing the
manifest snapshot + content hash, `generations.jsonl`, `battery.jsonl`,
computed metrics, and a canonical `config.json`. A local runner is
`steerlab runner serve`; see `steerlab workspace guide remote` for both.

<!-- client: all -->

`run` refuses, before the model loads, a concept-bearing manifest with no
injection, variant, or SAE arm. That refusal is not an obstacle: it is the
firewall telling you the study would have measured nothing. Declare a condition
(step 6) or promote an agent (`workspace guide sweep`), or declare the baseline-only study
explicitly if that is genuinely what you want.

## 10. Evaluate and analyze

Judging, evaluation, and effect sizes: `{{cli}} workspace guide evaluation`.

## Find what is missing

<!-- client: mac -->

**`data check <name>`**, at any point, returns the full classified readiness
list: every requirement, its status, the **path you must author**, and the
rationale. Fastest way to find what is missing. Blockers are a refusal:
`state: "refused"`, exit **65 in both modes** (the one verb whose human exit
has migrated — `workspace guide contract`).

<!-- client: python -->

This client has no data-readiness verb. `steerlab experiment verify <name> --json`
re-checks every pinned input against the bytes on disk and names each one that
is missing or has drifted, and `steerlab science guide <method> --json` lists
the files a method needs and their shapes.
