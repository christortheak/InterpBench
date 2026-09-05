# SteerLab — Onboarding

For a researcher who has never seen this tool: from "what is activation
steering" to a frozen, reproducible study you could hand to a stranger. Nothing
assumes you are an ML engineer; jargon is explained where it first appears.

Companions in this repository: [README.md](../README.md) (orientation and
install), [GENERAL-INTRODUCTION.md](GENERAL-INTRODUCTION.md) (the one-document
synthesis), [METHODS.md](METHODS.md) (the math and its sources),
[CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) (running a defensible study),
[RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md) (what each result layer can
claim), [CLI-REFERENCE.md](CLI-REFERENCE.md) (every verb and flag), and
[SECURITY.md](../SECURITY.md) (the servers' threat model).

---

## 1. What SteerLab is

A **workbench for concept steering** in open-weight language models. It does
three things:

1. **Extract** a direction inside the model corresponding to a concept you name
   — derived from text you write, not from anything built into the tool.
2. **Inject** that direction into the model's internal state while it generates,
   at a layer you choose and a strength you can report in comparable units.
3. **Measure** what moved — against a paired unsteered baseline, with random
   controls, capability checks, and effect sizes.

The engine is **concept-agnostic**: it has no idea what your concept means.
Concepts, stimulus texts, task prompts, rubrics, and scoring vocabularies are
*data you author in a workspace*, and no concept is named anywhere in the
engine. If a change to the core would not work equally for an arbitrary concept,
it is a bug. You bring the question; SteerLab supplies the apparatus and the
discipline.

The discipline is the other half. Everything beyond those three verbs exists to
make a result checkable by someone who was not there: inputs pinned by hash,
settings **frozen before behavior is measured**, immutable runs, and any
shortcut you take stamped into the artifact rather than remembered.

---

## 2. What "activation steering" means

At every layer a language model carries a large vector of numbers representing
its current internal state — an **activation**. The sequence of them flowing
through the layers is the **residual stream**.

Steering is three steps: find a direction in that space corresponding to a
concept; add a small amount of it to the internal state during generation; see
whether — and how — the output changes.

The worked example that ships here is **formality**: the register difference
between "I regret to inform you" and "bad news, sorry". It is domain-neutral,
easy to judge by eye, and cheap to validate, which makes it a good first concept
and a bad final one.

The baseline recipe, **Contrastive Activation Addition (CAA)**, is disarmingly
simple. Run texts that express the concept and matched texts that do not through
the model, record the activation for each, and take

```text
vector = mean(positive activations) − mean(negative activations)
```

per layer. Injection is equally simple: `h ← h + α·v` at a chosen layer, on
**every generated token** — not merely while the prompt is being read.

Two lines of arithmetic. Everything else — provenance, validation, controls,
freezing, statistics — exists to make those two lines mean something. Other
recipes are selected as data, not code: paired-difference PCA (RepE-inspired), a
grand-mean or designated-reference contrast against a reference corpus, the
template-mediated RepE reader, and linear probes. [METHODS.md](METHODS.md) has
the math and the literature behind each, and
[REPE-IMPLEMENTATION-BRIEF.md](REPE-IMPLEMENTATION-BRIEF.md) separates the
reader from the PCA family that used to borrow its name.

---

## 3. What you need

**Swift engine and macOS app:** an Apple silicon Mac (Intel is not supported),
macOS 26.4+. Xcode 27 only if you build from source — SteerLab.app carries the
command line it was built with, so an app install needs no developer tools. If
you do build, it has to be `xcodebuild` rather than `swift build`: SwiftPM
alone cannot build the Metal shader library MLX needs, and nothing touching the
GPU works without it (one time: `xcodebuild -downloadComponent MetalToolchain`).
Size memory to your model tier: roughly 3 GB for a 4-bit 4B model, 13–16 GB at
the 12–14B tier, plus headroom for the KV cache.

**Python engine:** Linux with an NVIDIA CUDA GPU for real work; it also installs
and runs on macOS for parity checks. Python 3.10+, 3.12 preferred.

**Both:** disk for the Hugging Face cache at `~/.cache/huggingface`. No weights
are distributed here — you download the models you choose, under their own
licenses. Some of those licenses carry use restrictions that plausibly extend to
artifacts you derive, including steering vectors you publish. Read the license
of the model you use; see [NOTICE](../NOTICE).

---

## 4. Install

**From the app (no Xcode).** SteerLab.app ships the CLI it was built from, at
`Contents/Helpers/steerlab-cli` — the same binary the app calls into, signed
with the bundle and carrying its own copy of the Metal shader library. Give it
a name on your `PATH` and you are done:

```bash
mkdir -p ~/.local/bin
ln -s ~/SteerLab/SteerLab.app/Contents/Helpers/steerlab-cli ~/.local/bin/steerlab-cli
export PATH="$HOME/.local/bin:$PATH"
steerlab-cli --version
```

`steerlab-cli` is the binary's own name, the one refusals print, and the one
these docs type for every Mac verb. Do **not** also link it as `steerlab`:
that shorter spelling belongs to the cross-platform Python **client** (its
console script; see CLI-REFERENCE §1.4), which is a different product with a
different verb surface, and one name for two things is how a lifecycle command
ends up exiting `64` for someone who copied it faithfully. A symlink is enough:
the binary resolves its shaders and the app's bundled resources against its real
location inside the bundle, not the link's, and invoking it in place by full
path is identical. `steerlab-cli --version` prints where each shipped resource
family resolved — **6/6 resolved** is the quick answer to "is this install
intact"; the app's code signature is the integrity guarantee, and there is
deliberately no writable manifest inside a signed bundle.

**From source (the developer path).** Optional, and the only one that needs
Xcode 27:

```bash
git clone <this repository> && cd <checkout>
./scripts/install-cli.sh          # builds with xcodebuild, installs under ~/.local
export PATH="$HOME/.local/bin:$PATH"
steerlab-cli --version
```

Re-runnable, no `sudo`, and it stages everything before swapping the live tree,
so a failed install leaves the previous one untouched. The binary lands in
`~/.local/libexec/steerlab/` beside the Metal shader library it needs, with a
shim at `~/.local/bin/steerlab-cli` — which is why the installed CLI needs no
environment variables, unlike running out of a build directory. Set
`DEVELOPER_DIR` first if `xcode-select` points at an older Xcode.

**The short name `steerlab` is the Python client's**, so the installer never
writes it as an alias unless you ask: pass `--short-name`, on a machine that
will never install the client. (Even then, an existing `~/.local/bin/steerlab`
the installer did not write itself — a symlink you made to the app's bundled
binary, say — is left alone.) The rule used to be conditional, "write it when
nothing else answers to the name", and that condition is evaluated the day you
install: the client arriving a week later would find its own console script
shadowed by PATH order. An install from before this policy may have left a
Swift-CLI shim at `~/.local/bin/steerlab`; if this machine also gets the
client, delete it so the name has one owner. `steerlab-cli install verify`
re-hashes the installed tree against its own manifest and answers "is what I
am running what was installed".

One macOS wrinkle the installer warns about: keychain access is granted per
binary identity, so the first verb that actually *uses* a stored credential may
prompt once for your Mac password. Run one interactively before pointing an
unattended agent at the install.

`scripts/build-app.sh` assembles the signed app — CLI included — from the same
checkout; `./scripts/run-app.sh` builds and launches it as a plain developer
binary instead, without the bundle. The app is the richest surface for authoring
concepts, watching a dose-response sweep, and chatting under steering, but
everything that matters for a paper is reproducible through `steerlab-cli` —
the app calls into the same engine, never the reverse.

**The Python engine:**

```bash
python3.12 -m venv Server/.venv.nosync
Server/.venv.nosync/bin/pip install -r Server/requirements-macos-arm64.lock
Server/.venv.nosync/bin/pip install -e "Server[all]"
```

Install **from the committed lock** (`requirements-macos-arm64.lock` or
`requirements-linux-x86_64.lock`), not from the floors in `pyproject.toml`: two
sites satisfying the same floors can resolve different `torch` and
`transformers` and produce different numbers, on the substrate where the
reproducibility claims live. The lock is the *intended* resolution; the
*achieved* one is stamped into every run (`config.json`'s `pythonEnvironment`),
and a run whose installed versions differ says so in an advisory rather than
dying. Regeneration and the cluster's site-owned-torch exception are in
[Server/README.md](../Server/README.md). Once the venv exists, `install-cli.sh`
also drops a `steerlab-server` shim on your PATH.

**The cross-platform `steerlab` client.** A third command line, and a different
product from the two engines: `pip install -e Server` (no extras, no torch,
~30 MB, any OS with Python 3.10+) installs `steerlab`, which authors a
workspace locally, freezes it, packages a hash-pinned bundle, and hands it to
an engine over `--runner <url>` — `steerlab run <experiment> --runner <url>` is
that whole round trip, evidence verified on the way home. A correct install
says so: `steerlab --version` prints `steerlab <version> (client)`. It does not
report resource families, and it does not answer the Mac lifecycle verbs this
document types — no `workspace init`, no `extract`, no `sweep` — so a command
from §5 or §6 typed under `steerlab` exits `64`, correctly. Add the `[runner]`
extra to execute locally through `steerlab runner serve` (macOS and Linux;
Windows is out of scope — untested and unsupported, and `runner serve` refuses
that verb there). The verb-by-verb reference is
[CLI-REFERENCE.md](CLI-REFERENCE.md) §1.4, and the contracts behind it are
[PORTABILITY-CONTRACTS.md](PORTABILITY-CONTRACTS.md) §7–§10.

**Where to put things.** A `SteerLab/` folder in your home directory holds your
workspaces, your private site library, the app, and — if you have one — the
checkout as siblings, so one directory moves, backs up, and is handed to an
agent as a unit. One command materializes it:

```bash
steerlab-cli init              # or: steerlab-cli init --home /path/to/SteerLab
```

```text
~/SteerLab/
├── Workspaces/          study workspaces, one folder each, each its own git repo
│   └── register-pilot/  one study's data
├── Sites/               your PRIVATE site registry
│   └── cluster-sites/   one JSON file per site — app AND steerlab-cli read it
├── SteerLab.app/        the app, and the CLI it carries in Contents/Helpers/
└── <checkout>/          the code checkout (this repository), under any name
```

`init` creates `Workspaces/` and `Sites/` when they are absent and reports what
was already there, so running it twice is a no-op that says so. It never deletes
or overwrites anything, and it deliberately does nothing else: it does not move
the app, does not clone or create the checkout, does not create a workspace, and
does not change how a workspace root is resolved. The checkout's folder name is not fixed — a clone
lands under whatever name you gave it, and `init` finds it by content rather
than by name.

`Sites/` is yours and is left **empty** by `init`, so
`git clone <your-private-repo> Sites` works into it. Keeping site configuration
there rather than inside a study workspace is what stops it travelling with a
workspace you share.

Inside it, `Sites/cluster-sites/` is the **canonical cluster-site registry** —
one pretty-printed JSON file per site, named by the site's id, and the single
store the app and `steerlab-cli` share. The app edits those files;
`steerlab-cli cluster sites import <profile.json>` writes into the same
directory; `steerlab-cli cluster sites list` reads it. **You sync it**:
`Sites/` is normally a private git repository, and git is how your sites reach
your other machines. SteerLab never runs git —
its writes leave the tree dirty and committing/pushing is always your act. The
directory also works perfectly well as a plain folder with no git at all.

Three things stay out of it, deliberately:

* **Secrets** — bearer tokens, the Hugging Face token — live in this Mac's
  Keychain, never in the registry. That is why a new machine prompts you once
  after a pull, and it is a feature: a credential committed to a repository is
  a credential in every clone of it, forever.
* **Connection state** — the endpoint a tunnel landed on, the last server build
  — is per machine, in
  `~/Library/Application Support/SteerLab/site-runtime.json`. Connecting and
  disconnecting therefore never dirties your registry.
* **Study data** — workspaces still never contain site profiles.

The first time a new build runs it absorbs the two old stores (the app's
saved-servers preference and
`~/Library/Application Support/SteerLab/cluster-sites.json`) into
`Sites/cluster-sites/` and says what it moved. A file already in the registry
always wins; nothing is overwritten. After that the old stores are read-only
history that nothing writes to.

Any layout works; the resolution order for "which workspace am I in" is
`STEERLAB_WORKSPACE`, then `--workspace`, then the choice the app has persisted.

---

## 5. Your first hour

`SampleWorkspace/` is a complete, **recipe-only** worked example built around
the formality concept — see [its README](../SampleWorkspace/README.md) for what
is deliberately left out: no vectors, no runs, no frozen manifest. Activations
do not transfer between substrates, so a shipped vector would be useless at best
and misleading at worst. You derive it on your own machine, which is exactly
what the firewall asks of every study.

Every command from here through §7 is **`steerlab-cli`**, the Mac instrument of
§4 — not the cross-platform `steerlab` client, which authors and submits but
does not extract, sweep, or measure.

```bash
cp -R SampleWorkspace ~/SteerLab/Workspaces/first-hour
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/first-hour

steerlab-cli experiment create demo --model Qwen/Qwen3-4B-MLX-4bit
steerlab-cli experiment attach demo formality
```

`create` writes a **draft** manifest pinned to a model. `attach` pins the
concept's stimulus files *by SHA-256* along with the extraction options:

```text
pinned formality @ 3a9c2c048f5e… (24+24 stimuli, meanDifference, last token)
```

Read that line carefully — it is the whole idea in one sentence. The experiment
does not contain a vector. It contains a **recipe**: these exact bytes, this
method, this reading position. Runs re-derive the vector from the recipe, and if
a stimulus file changes by one character afterwards the hash no longer matches
and `verify` fails loudly instead of quietly measuring something else.

### Extract and validate

```bash
steerlab-cli experiment extract demo
steerlab-cli experiment validate demo
```

Extraction writes a vector into a new immutable run directory and pins the
**model revision** — the exact snapshot commit — from your local cache if you
did not pass `--revision`:

```text
pinned model revision 52a5ab34fa60… (local HF cache)
formality: 36 layers, hidden 2560, norm @ mid 13.386816
```

Validation separates a direction from a decoration. It scores the vector on the
concept's **never-named** held-out scenarios — texts that exhibit the concept
without using its vocabulary, so a vector that merely memorized a keyword fails:

```text
formality: validation accuracy 100% over 14 never-named scenarios @ layer 18 of 36
worst cross-concept |cosine|: not measured — the matrix holds 1 direction, so
there is no cross-concept PAIR. Declare validation controls or attach a second
concept to measure discriminant validity
```

Both halves matter: convergent validity (the direction detects its own concept
out of sample) and discriminant validity (distinct concepts must not collapse
into one generic direction), the latter visible only with two directions to
compare. With one concept attached, the tool reports that the measurement was
*not made* rather than a comforting number.

### Ask what is still missing

`steerlab-cli data check demo` is the manifest-driven readiness checklist, and
the fastest way to learn what a study needs:

```text
✓ [present] stimuli / validation set / markers — formality
✗ [missing] task prompts       prompts/tasks/demo-prompts.jsonl
✗ [missing] neutral corpus     prompts/neutral/corpus.jsonl
· [optional] human-baseline CSV, reasoning-style taxonomy, capability battery …
3 present · 0 partial · 2 missing · 4 optional
```

Every row names the **path you must author** and why the requirement exists.
Blockers are a refusal, not a warning: `data check` exits 65 when any blocker is
present. For the row you owe, `steerlab-cli authoring prompt <kind>` renders the
generation prompt that produces it — `contrastive-pairs`, `validation-set`,
`choice-prompts`, `reader-pairs`, `battery` — with that kind's audit battery as
numbers. The templates are workspace data in `prompts/authoring-prompts/`, your
copy wins over the shipped one, and each emission stamps the hash of the wording
it used. The emitter is deliberately not the acceptor: what comes back still has
to pass the audit.

The task-prompt file is the measured task, and the sample ships one:
`steerlab-cli experiment pin-prompts demo prompts/tasks/starter-prompts.jsonl`.

The neutral corpus is the denominator that makes steering strength comparable,
and the sample carries one (`prompts/neutral/corpus.jsonl`), along with the
default paired-judging rubric — so extraction, validation, norm-unit α, and a
first judged comparison all work inside the sample. What the sample does NOT
carry is the wider instrument library (sweep dev split, robustness sets,
authoring templates); those come with a workspace created by
`steerlab-cli workspace init`. Past your first extraction, create a real
workspace and copy the formality concept into it rather than growing the sample
in place.

### See a dose-response

Steering strength, α, is reported in **units of the residual-stream norm** at
the injected layer, measured on the pinned neutral corpus. That fixed
denominator is what lets α ≈ 0.3 mean roughly the same push across concepts,
layers, and model families, where a raw α would not. Declare an arm and you have
two conditions — the implicit baseline and one steered condition:

```bash
steerlab-cli experiment declare-condition demo formality-mid --slots formality:18:0.3
```

The intuition to build, in the app's Steering tab or across a sweep, is the
dose-response curve: too little α and nothing moves, too much and the model
stops making sense. On small models expression usually appears well below α ≈
0.5 and coherence starts failing above ≈ 1 — a bracket to sweep, never a setting
to adopt. Mid-network layers usually steer best. Two self-tests ship for exactly
this, and they are what to run after touching the engine:

```bash
steerlab-cli --config prompts/configs/smoke-test.json   # steered ≠ baseline; α=0 == baseline
steerlab-cli --config prompts/configs/toy-french.json   # + concept beats a matched-norm random control
```

---

## 6. Your first real study

```bash
steerlab-cli workspace init ~/SteerLab/Workspaces/register-pilot
export STEERLAB_WORKSPACE=~/SteerLab/Workspaces/register-pilot
```

A workspace is a plain folder with `prompts/`, `experiments/`, `runs/`, and
`adapters/`, its own git repository, a `WORKSPACE.md` marker, and an `AGENTS.md`
written for a coding agent (§9). Study data lives here — never in the checkout.

A new workspace is born with **instruments only**: capability batteries, the
neutral corpus, sweep dev prompts, judging rubrics, a parser registry, and
templates for every study-data file you will need. It is deliberately
**concept-empty** — `prompts/concepts/` is created and left empty — because the
concept is the thing you are studying and it should arrive by your decision. The
seeded set is an explicit allowlist in the code, not a directory sweep, so
nothing reaches your workspace by accident.

### Author a concept

```text
prompts/concepts/<name>/positive.jsonl   {"text": "…"}   expresses the concept
prompts/concepts/<name>/negative.jsonl   {"text": "…"}   matched, does not
prompts/concepts/<name>/validation.jsonl {"text": "…", "expresses": true|false}
prompts/concepts/<name>/markers.json     {"words": [...]}  diagnostic vocabulary
```

`SampleWorkspace/prompts/concepts/formality/` carries all four with a README
explaining each choice. Two rules matter more than anything else you will do:

- **Content matching.** Positives and negatives should differ in the concept and
  nothing else — same topics, lengths, register otherwise. If your positives are
  all about one subject and your negatives about another, the vector encodes the
  subject.
- **Independence from the outcome.** The stimuli must not contain the vocabulary
  or the situations you intend to measure. If the texts already discuss the
  behavior you will test, a shift proves only that the model noticed. This is
  the circularity firewall at the level of raw text, and no downstream statistic
  repairs a violation of it.

Validation scenarios have their own rule: they must **never name the concept**.
That is what makes the validation gate a real out-of-sample test. The app's
Concepts tab and the Python engine's browser workbench both do this
interactively, including LLM-generated candidate pairs you accept or reject one
at a time; templates live under your workspace's `prompts/templates/`.

### The lifecycle, in order

```bash
steerlab-cli experiment create <name> --model <model-id> [--revision <commit>]
steerlab-cli experiment attach <name> <concept>…          # pin stimuli + options
steerlab-cli experiment detach <name> <concept>…          # attach's inverse, all-or-nothing
steerlab-cli experiment pin-prompts <name> prompts/tasks/<file>.jsonl
steerlab-cli experiment pin-rubric  <name> prompts/rubrics/<file>.md --judges a:local,b:claude
steerlab-cli experiment set-instruments <name> answerTokenLogprob
steerlab-cli experiment set-sampling  <name> --temperature 0.7 --max-tokens 1024
steerlab-cli experiment set-parser    <name> <parser-registry-entry>
steerlab-cli experiment set-instrument-scope <name> label,json
steerlab-cli experiment set-exclusions <name> unparseableEndpoint
steerlab-cli experiment set-sweep-grid <name> --layer-fractions 0.5,0.7,0.85
steerlab-cli experiment set-sweep-selection <name> --objective judgeScore
steerlab-cli experiment declare-condition <name> <arm> --slots <concept>:<layer>:<alpha>

steerlab-cli experiment extract  <name>                   # derive vectors
steerlab-cli experiment validate <name>                   # held-out probe + cosines
steerlab-cli experiment sweep    <name>                   # layer × α on the dev split
steerlab-cli experiment promote  <name> <concept>         # mint an arm from the winning cell

steerlab-cli experiment freeze   <name>                   # ONE-WAY, and gated
steerlab-cli experiment run      <name>                   # generate every condition
steerlab-cli experiment evaluate <name>                   # paired judging
steerlab-cli experiment analyze  <name>                   # effect sizes, CIs, corrections
steerlab-cli experiment duplicate <name> <name>-2         # the only way to iterate after freeze
```

Not every study needs every verb. `<family> --help` lists a family's verbs,
`<family> <verb> --help` prints one verb's arguments, and
[CLI-REFERENCE.md](CLI-REFERENCE.md) is the complete surface. Order matters in
places — a `judgeScore` sweep objective is refused until a rubric is pinned — and
the shape is always the same: **declare and pin everything, gather evidence
about your settings on a development split, freeze, and only then measure.**

The grid and the rule that picks from it are both manifest data with their own
writers: `set-sweep-grid` declares the layer × α axes (the layer axis as depth
fractions, so one declaration names the proportionally same site in a 26-block
model and a 62-block one) along with the dev split, the battery, and the
per-cell token budget, and refuses a grid no sweep could run — empty,
non-ascending, repeated, or out of range. `set-sweep-selection` declares how the
winner is chosen. `detach` is attach's inverse and is what lets a duplicated
study shed the donor's concepts, which would otherwise ride along swept but
uncitable; it refuses while any declaration still names the concept.

The four `set-*` verbs between the instrument and the grid are the
**measurement declarations**, and each exists because the alternative was a
silent default. `set-sampling` owns the generation protocol — temperature,
token budget, prompt mode, and the stochastic replication pair
(`--samples-per-item` × `--seed-policy`) — merge-style, so only the flags you
give move and the protocol can be built one flag at a time (the joint rule
that many samples need a non-zero temperature and a derived seed policy
surfaces at `verify`, not at the write). `set-exclusions` declares the
record-exclusion rules analysis applies, which are stamped honestly: no record
ever leaves `generations.jsonl`. `set-parser` names an entry in
`prompts/parsers/parser-registry.json` and pins that registry's hash alongside
it — you never type the hash, because the registry file is the authority on
which parser *version* the study preregistered. And `set-instrument-scope`
declares which `responseFormat` rows the option-consuming instruments read, so
a prompt file mixing json and label rows keeps its log-probability instrument
on the label rows instead of dropping it; a scope selecting zero rows is
refused at the declaration rather than producing an empty run.

`sweep` walks that grid and selects a cell by the criterion **declared in
the manifest as data**; `promote` mints an arm from the winning cell carrying a
birth certificate recording how it was selected. A manual override is permitted,
loud, and stamped — and still requires evidence that a sweep ran. An arm should
trace to the rule that produced it, not to someone's memory of a good afternoon.

That certificate also records when the promoted direction is a **mirrored
pole**. `steerlab-cli vectors mirror-poles` mints the opposite pole of a
contrastive direction as its own provenance-stamped artifact — a bit-exact
sign flip under a required new concept name — rather than leaving it as a
negative α, which every dose surface reads as "less of the concept" and no
artifact records. Promote from one and the certificate carries a
`poleProvenance` block naming the source concept and its hashes, because an
arm that injects the negation of a concept must not look identical to one that
injects the concept.

### Freeze is the point, not the paperwork

`freeze` is one-way. It re-verifies every pinned input against the bytes on
disk, stamps a content hash and the workspace's git commit into the manifest,
snapshots all pinned inputs into `experiments/<name>/pinned/`, and makes the
manifest read-only. This is preregistration implemented as a mechanism instead
of a promise: the settings that produced a result were chosen *before* the
behavior was measured, and anyone can check that afterwards without trusting
you.

Pin verification always runs and is never skippable. Then seven evidence gates
apply:

| Gate | What it demands |
|---|---|
| `revision` | a pinned, immutable model commit |
| `measurementPins` | the pins that determine what is measured are present and valid |
| `validateEvidence` | a `validate` run matching the exact pins, on this engine, that is not vacuous |
| `variantValidity` | attached variants carry hashed weights and a pinnable dataset manifest |
| `batteryEvidence` | each variant condition has scope-matched capability-battery evidence |
| `judgeValidity` | a rubric *file* and at least one judge (a panel of two or more must be genuinely distinct) |
| `gitClean` | every pinned input is committed in the workspace repository |

A refusal is typed and names its repair — in `--json` mode, `error.code` is
`freezeGateFailed`, `error.gate` and `error.gates[]` name every gate that
failed, and `error.repairAction` is an executable command sequence:

```text
cannot freeze 'demo': no validate run matches its exact pins (model+revision,
concepts, neutral corpus) on the run substrate swift-mlx. Run 'steerlab-cli
experiment validate demo' first, or freeze --force to record an unvalidated
experiment
```

`--force` skips the seven gates. It does not skip pin verification, and it is
**loud and permanent**: every skipped-and-failing gate prints a warning and is
stamped into the frozen manifest as `freezeForced` plus the gate ids in
`forcedGatesSkipped`. A forced freeze stays non-citable — but checkably so, by
stamp, years later. Do not reach for it to get past a gate; fix the gate.

After freezing the manifest is read-only; `duplicate` it into a fresh draft to
change anything. That is not an inconvenience, it is what keeps the record of
what you preregistered intact. Runs stamp the manifest epoch they came from, and
`analyze`/`evaluate` refuse to interpret a run against a different epoch.

Two consequences that save real time once you know them. A **duplicate
inherits its donor's validation evidence**, because the `validateEvidence`
gate matches on the study's *pins* — model, revision, concepts, corpora,
battery — and not on its name: a duplicate that changes only measurement-side
declarations freezes with no re-extraction and no re-validation. And the epoch
guard tolerates measurement-side drift on `analyze`/`evaluate`/`rescore-style`
(the judges, the evaluation block, the rubric pin, the human-validation pin,
and the study's own name), which is what makes the sanctioned re-measurement
path work: **duplicate the study, pin the new rubric and panel on the
duplicate, and evaluate against the ORIGINAL run directory.** The source study
is never edited, the tolerated drift is stamped into the output, and `promote`
still refuses. [CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) §4.6–4.7 has
both in full.

---

## 7. Measuring what moved

Generating steered text is easy; turning it into a defensible claim is what the
rest of the instrument is for. This is a map —
[CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) is the operating manual and
[RESULTS-ARCHITECTURE.md](RESULTS-ARCHITECTURE.md) says what each layer of
result licenses you to claim.

- **Outcome instruments are declared, never inferred.** `set-instruments` names
  what a run measures. For categorical outcomes prefer the **answer-token
  log-probability / choice-probability** instrument over sampled prose: it is
  deterministic, temperature-free, needs no parser, and is far more sensitive
  than counting which word came out. Give task-prompt rows `options` and a
  `target`, then declare `answerTokenLogprob`.
- **Judges** score free text. The rubric is pinned as a *file* with its hash,
  and judging is paired against the same item's baseline. A study needs **at
  least one** judge; a panel of two or more must be genuinely distinct —
  identity resolves to (kind, model, provider), so two local judges with blank
  model fields are one judge agreeing with itself, and are refused. A
  **single-coder** design is legal and freezes cleanly, but it buys no
  inter-rater statistics: it carries a loud `judgePanelTooSmall` advisory and
  its coding report records `fieldAgreement` as *absent with that reason*
  rather than empty. The usual answer when cost pushes you there is to
  calibrate once with two judges and report that κ beside single-judge
  production results. A local judge naming a model other than the study model
  must pin its exact bytes — `pin-rubric … --judge-pin
  <judge>=<commit-hash>[:<dtype>]` — and needs
  `STEERLAB_MAX_LOADED_MODELS ≥ 2` on the engine host.
- **Capability batteries** run per condition inside `run`. If capability drops
  under steering, an apparent behavioral finding is confounded by degradation
  and both must be reported — the difference between "the concept moved the
  decision" and "the model got worse at everything". **Controls** matter for the
  same reason: a matched-norm random direction is a first-class arm
  (`--control randomMatchedNorm`), and a concept that does not beat it has
  measured the size of your nudge and nothing else.
- **Statistics** are paired to each item's own baseline, with bootstrap
  confidence intervals rather than bare p-values and multiplicity correction
  appropriate to the phase (FDR for broad screens, Holm for confirmations).
  `analyze` writes `effect-sizes.csv` and folds the numbers into `report.json`.
- **Marker density** — how often your concept's vocabulary appears — is a
  *manipulation check*, not an outcome. Selecting doses on it optimizes for
  surface style, the precise confound most steering work falls into. Declare a
  behavioral objective (`judgeScore`, `logprobShift`) for anything whose outcome
  is a decision rather than prose. The same rule governs *reporting*, not just
  dose selection: for a question about reasoning style, the primary instrument
  is **judged classification against a pinned rubric**, with marker density
  reported beside it as the diagnostic it is.

**Sampling policy, which surprises people.** Local Swift/MLX measured runs are
**greedy-only**: the runner requires `temperature == 0` and one seed, because
the MLX generator cannot yet pin a per-run sampling seed. A manifest's seed is
recorded for provenance and stamped `seedInert: true` — never read a local run's
seed as causally meaningful. **Stochastic studies belong on the Python engine**,
which seeds per record and writes one record per (condition, prompt, sample
index).

---

## 8. Scaling out to a GPU

The Python engine is a peer, not a wrapper: it re-implements extraction,
injection, and the freeze lifecycle in PyTorch/Hugging Face over the same
artifacts. Because CUDA activations never byte-match MLX-on-Metal, **vectors are
re-extracted and re-validated on whichever engine a study runs on**. What is
identical across engines is the structure: SHA-256 input hashes, the manifest
and run schemas, the JSON envelope, and committed golden fixtures both sides
check.

```bash
Server/.venv.nosync/bin/python -m steerlab_server.cli serve --root <workspace>
```

**It requires a bearer token by default, on every platform.** With no
configuration at all, `serve` resolves token mode, hydrates the token from
`~/.steerlab-token` (`STEERLAB_AUTH_TOKEN_FILE`), and mints that file — 0600 —
when absent, printing the path and never the value. The historical
open-on-loopback tier is an explicit opt-in, `serve --dev-open-loopback`, and it
*refuses to start* on a non-loopback bind or next to a Slurm executor. Reach a
remote engine over an SSH tunnel; there is no TLS. Read
[SECURITY.md](../SECURITY.md) before exposing anything, including its known
limitations.

### Qualify the node before you trust it

```bash
steerlab-server site qualify [--json] [--skip-model-fixtures]
```

Run it right after provisioning a node, after any deploy that changes the
engine, and before the first study on a site you have not used. It is cheap and
needs no GPU. Nine checks ask whether this machine reproduces the contracts a
result depends on — build identity, the measurement-stack fingerprint,
dependency-lock agreement, the stimulus hash convention, prompt-render and
tokenization goldens, vector-parity arithmetic, profile validation, GPU
visibility — and none aborts the others, so a cold node always gets a complete
report. Exit 0 clean or with warnings, 70 if anything failed.

Read the summary line rather than the exit code alone, because **skipped checks
never change the exit code but are always counted**:

```text
5 passed, 1 warning, 0 failed, 3 skipped of 9 checks
```

A node with six skips passed almost nothing; it is unverified, not healthy.
Cluster deployment, Slurm submission, checkpoint/resume, and the bundle path for
engines that cannot see your workspace are in
[Server/README.md](../Server/README.md) and [CLI-REFERENCE.md](CLI-REFERENCE.md).

**Sharding a long run across GPUs.** A measured `run` partitions cleanly —
every generation record is independent — so a Slurm submission can fan out
across N sibling jobs whose partials the server merges back into one ordinary
run directory. Both command lines reach it: `steerlab-server study submit
<name> --verb run --executor slurm --parallel 4` on the engine side, and
`steerlab-cli remote submit-bundle <path> --site <id> --verb run --executor
slurm --parallel 4` from the Mac. Sharding is execution logistics and never
enters the manifest or its content hash, so a sharded run and a single-job run
of the same frozen study are the same measurement. Two field rules before you
use it: the merge is performed by a **running** `steerlab-server serve`, not
by the submitting process; and a fan-out can **partially fail while the submit
exits 0**, so check the scheduler queue rather than the exit code, and stagger
submissions where the site caps queued jobs per user.
[CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md) §7.2 has the discipline;
[CLI-REFERENCE.md](CLI-REFERENCE.md) §5.3 has the mechanics.

### Reading what the model is poised to say: J-lens

A J-lens is a published measuring instrument, not a steering vector. For the
model it was fitted on, it maps a residual at any layer into the model's
final-layer basis, so the ordinary output head can read it out: which tokens
the model is poised to verbalize at that layer, and how a steering vector or a
condition changes that. SteerLab uses it three ways — as a check that an
injection took hold, as a mechanism-level cross-check on a vector (what
vocabulary is it made of), and as a readout recorded alongside generation in
a study.

It is server-only and lives entirely under `steerlab-server jlens`. The Mac
app renders lens artifacts and carries a study's readout block, and never
produces either; the Swift CLI has no lens verbs. The reference package the
engine checks itself against is an optional extra, deliberately outside
`[all]` because it raises the transformers floor and installs from GitHub:
`bootstrap.sh --with-jlens` on a cluster node, or
`pip install -e "Server[jlens]"` in a venv you have decided to change.

The path, in order:

```bash
steerlab-server jlens supported --published   # what upstream carries (~40 models; needs egress)
steerlab-server jlens acquire <model-id>      # bytes into the HF cache (needs egress)
steerlab-server jlens import  <model-id> --tier evidence|testing   # convert into runs/jlens-lenses/ (offline)
steerlab-server jlens qualify <lens-id> <model-id> [--token-id N] [--battery P] [--alpha-range …]
```

**Any model with a published lens can be imported.** Three of them — the
Gemma-3 27B, 12B and 4B instruction models — have curated rows that fix
their evidence tier for this project, and `--tier` is refused where it would
contradict one. Every other model's tier is yours to declare at import,
because nothing upstream can say whether a study treats a model as evidence
or as rehearsal; the declaration is stamped on the lens record with its
provenance, and freeze refuses a testing-tier lens and a lens with no tier
at all.

`qualify` is what makes a lens citable. It loads the model (a GPU job) and
runs seven named checks — geometry, runtime numerics, finite Jacobians,
agreement with the pinned reference implementation, stable readouts, a
causal smoke test, and the capability battery at the top of the intended
dose range — then appends the verdict to the lens record whether or not it
passed. Exit 3 means it did not qualify; the record is still written,
because "tested and failed" is evidence and its absence is not. Two defaults
are worth overriding on a real model: the smoke-test token (` the` unless you
pass `--token-id`) and the dose ladder, since the capability guard can
collapse at the top of the default ladder on a token you never intended to
steer.

One thing the engine does for you that you should know about: the readout
folds the model's final-norm gain into the token rows, and two
parameterizations of that gain exist (Gemma and Qwen3.5 store it as an
offset from one; Llama, Qwen3, OLMo and most others store it directly). The
engine **observes** which one the model's own norm computes, on every build,
and refuses a norm it cannot fold — a LayerNorm with a bias, for instance —
rather than reading a wrong gain. The observed convention is stamped on
every derived direction and qualification.

After that, the instruments: `support` decomposes a vector into the
vocabulary it is made of; `token-options` and `derive` mint a steering
direction indexed by an exact token; `probe` reads one prompt by position and
layer; `g0` is the feasibility gate for a model, with two separately
verdicted arms; `report` rolls readouts up across conditions. A study opts
in through the manifest's `jlensReadout` block, which freeze checks against
the lens's qualification for the study's exact model, revision and dtype.
[CLI-REFERENCE.md](CLI-REFERENCE.md) §6.3 has every flag.

---

## 9. Driving SteerLab with a coding agent

Every workspace gets an `AGENTS.md`, generated at creation: the lifecycle in
order, the file shapes, the freeze gates with their repairs, the machine
contract, and an explicit list of what not to do. Point an agent at the
workspace and it has what it needs; you do not have to explain SteerLab to it.

It also keeps itself current. The generated header carries a SHA-256 of the body
it wrote, so a file whose hash still matches is provably the machine's and is
refreshed to the shipped contract — atomically, with one info notice — at every
workspace open and every CLI verb. A hash that no longer matches means you edited
it, and the file is then yours and is left alone. Contracts written before the
hashed header keep the older advisory-only behaviour until one manual
regeneration graduates them.

Tell it which command line it has. On a Mac that is `steerlab-cli`, the
instrument this document types throughout, verified by `steerlab-cli --version`
reporting **6/6 resource families resolved**. Elsewhere it is the `steerlab`
client (§4), verified by `steerlab --version` printing `steerlab <version>
(client)` — a smaller verb surface reached the same way, since everything below
about `--json`, refusals, and exit codes holds identically on it. An agent that
types a Mac lifecycle verb under the client's name gets `64`, which is the
instrument being clear rather than the agent being unlucky. The repository
root's `AGENTS.md` is the cold-start contract that picks between them.

Every verb on the study path accepts `--json`. In JSON mode stdout carries
exactly one envelope and nothing else, every diagnostic goes to stderr, there
are no ANSI sequences, keys are sorted, timestamps are ISO-8601, and hashes are
complete rather than elided. Always present: `schemaVersion`, `verb`, `engine`,
`state`, `changed`, `observedAt`, `message`. Present when they have something to
say: `workspace`, `advisories[]`, `error` (with `code`, `gate`, `gates[]`,
`repairAction`), `result`, and `nextAction` on successes. A missing key is a
straight answer, not a null.

`state` is authoritative and the exit code is a convenience: `0` ok, `64`
malformed invocation, `65` refused by a gate, `66` not found, `70` failed.
Without `--json` most failures still exit `1` — one more reason to always pass
it. (`data check` is the one verb whose human-mode exit has migrated; an
undeclared flag is `64` in both modes, refused before the verb does any work, so
a typo cannot silently change what a study means.)

Two behaviors to internalize. **A refusal is not an error to retry**: it means a
gate declined a well-formed request against a healthy system, and
`error.repairAction` is an executable repair — perform it, then retry. And
**advisories never change the exit code**: they are things you should know that
did not stop the verb, such as a skipped gate, a vacuous validation, a one-judge
panel, an empty analysis. Ignoring them produces results stamped as not citable;
treating them as failures makes you refuse a legitimate lifecycle.

---

## 10. Where things live

```text
<workspace>/
├── AGENTS.md              the agent contract (generated)
├── WORKSPACE.md           the marker that makes this a workspace
├── prompts/
│   ├── concepts/<name>/   positive/negative/validation/markers — yours to author
│   ├── tasks/             the measured task-prompt sets
│   ├── neutral/           corpus.jsonl — the α denominator
│   ├── dev/               sweep dev split and robustness prompts
│   ├── batteries/         capability probes
│   ├── rubrics/           judge rubrics (pinned by hash)
│   ├── templates/         the shape of every file you must author
│   ├── authoring-prompts/ the registry `authoring prompt <kind>` renders from
│   └── generation/        prompt texts for LLM-assisted authoring
├── experiments/<name>/    experiment.json (the manifest) + pinned/ (freeze snapshot)
├── runs/                  immutable outputs
└── adapters/              adapter training data and outputs
```

**Runs are immutable.** Every operation that produces artifacts writes one
directory `runs/<timestamp>-<slug>/`, never overwritten and never reused.
Inside, depending on the task: the manifest snapshot and its content hash,
vectors as `.safetensors` with a JSON sidecar recording every extraction choice,
raw generations as JSONL, the capability battery separately, metrics as CSV,
judge outputs, and a canonical `config.json` with substrate and version metadata
— enough to rebuild your tables without rerunning a model. A few subtrees under
`runs/` are deliberate mutable *libraries* rather than outputs (promoted
variants, neutral principal-component bases, imported lens artifacts); frozen
studies are protected there by their manifest snapshot and artifact hashes.

The provenance chain, end to end:

```text
stimuli (hashed) → vectors (sidecar: model+revision, layer, method, position,
norms) → conditions (a named steering configuration) → manifest (content hash +
git commit, stamped at freeze) → run directory (manifest snapshot + hash +
substrate metadata)
```

Every arrow is checkable after the fact. Drift anywhere is a verification
failure, never a silent change.

---

## 11. Building and testing from source

**SwiftPM cannot build Metal shaders**, so anything that executes on the GPU
must be built by Xcode.

```bash
swift build --scratch-path .build.nosync                            # compile check only
swift test  --scratch-path .build.nosync --filter VectorMathTests   # pure-CPU tests
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO        # everything GPU-bound
cd Server && .venv.nosync/bin/python -m pytest -q                   # the Python engine
```

`Failed to load the default metallib` from a plain `swift test` is expected, not
a regression. `-parallel-testing-enabled NO` is required, not a preference: the
suite serializes a process-global data-root override, and parallel execution
starves the cooperative thread pool and wedges. `-skipMacroValidation` is
required under command-line `xcodebuild`. Keep build products in `*.nosync`
directories if your checkout lives in a synced folder.

Run the smoke test after touching the steering engine: it asserts that hooks
fire on every forward pass, that steered output differs from baseline, and that
α = 0 reproduces baseline exactly. Off-the-shelf model implementations expose no
hook into the residual stream, so SteerLab vendors the model files and adds one
intervention point between transformer blocks — and injection must fire on
**every decode step**, not only while the prompt is read. Steering during
prefill alone silently produces near-null results; permanent unit tests on both
engines guard against it.

---

## 12. Glossary

- **Activation / residual stream** — the per-layer hidden vector the model
  carries through the network; where SteerLab reads and writes.
- **CAA** — Contrastive Activation Addition: mean(positive) − mean(negative),
  per layer. **Paired-difference PCA (RepE-inspired)** — a direction taken as
  the first principal component of per-pair activation differences; called
  `repeLAT` until the 2026-08-27 naming ruling, and its raw values on disk stay
  `lat`/`repeLAT` forever. **RepE reader** — the separate, template-mediated
  reading instrument of Zou et al.: a task template, the LAT token at the
  rendered scaffold's final position, and a sign and layer chosen on a held-out
  split. See [REPE-IMPLEMENTATION-BRIEF.md](REPE-IMPLEMENTATION-BRIEF.md).
- **Alpha (α)** — steering strength. In residual-norm units the injected
  perturbation's L2 norm is α × the layer's typical residual norm on the pinned
  neutral corpus, which is what makes doses comparable.
- **Stimulus set** — the texts a concept is extracted from; its hash is the
  concept's identity in every pin. **Sidecar** — the JSON file beside each
  `.safetensors` vector recording full provenance.
- **Condition (arm)** — a named, complete steering configuration in a manifest;
  several slots in one condition *are* the linear mix `h + Σ αᵢ·vᵢ`.
- **Sweep** — the layer × α dose-response grid, scored for expression,
  degeneration, and capability. **Capability battery** — short unrelated probes
  that must survive steering, or a finding is confounded.
- **Freeze** — the one-way transition that verifies every pin, applies the
  evidence gates, and makes a manifest read-only.
- **Run directory** — one immutable `runs/<timestamp>-<slug>/` per operation;
  the unit of reproducibility. **Advisory** — something you should know that did
  not stop the verb; never changes an exit code.
- **Workspace** — the folder holding one project's prompts, experiments, and
  runs, with its own git history, separate from the code.

---

## 13. In one paragraph

SteerLab finds directions inside open-weight language models, injects them
during generation at doses reportable in comparable units, and measures whether
behavior moves — against paired baselines, random controls, and capability
checks, with settings frozen and hash-pinned before the behavior was measured,
and every shortcut stamped into the artifact rather than remembered. The concept
is yours; the discipline is the instrument's.
