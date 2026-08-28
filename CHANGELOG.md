# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Schema compatibility is a promise, not a convention: workspaces and frozen
manifests created by one release keep loading and keep verifying in the next.
A breaking change to a run, manifest, or JSON-envelope schema gets a new
schema number, a reader for the old form, and an entry in this file — never a
migration that rewrites frozen bytes.

## [Unreleased]

### Added

- **The last two measurement declarations gained headless writers.** The
  manifest's `numericParser` + `parserRegistryHash` and its
  `outcomeInstrumentScope` were writable only from the app's pickers, which
  blocked headless authoring of three replication studies. `experiment
  set-parser <name> <parser>` resolves a name against
  `prompts/parsers/parser-registry.json`, shape-checks the entry, and pins
  the registry's current SHA-256; the hash is never an argument, because the
  registry file is the authority on which parser VERSION a study
  preregistered. Without a declaration a numeric study fell back to the
  deprecated implicit selection (`caseFamily: "sentencing"` → the built-in
  duration parser), and clearing the declaration now says so as a
  `deprecatedImplicitSelection` advisory. `experiment set-instrument-scope
  <name> <responseFormat>[,…]` declares which formats the option-consuming
  instruments read and pins the row set they select — the NON-LOSSY repair
  the run-start `responseFormat` refusal names, where the only other repair
  (`set-instruments … sampledText`) drops the instrument. Both refuse an
  out-of-vocabulary value at 64, and a scope selecting zero rows is refused
  at the declaration rather than producing zero records at the run. Both are
  Mac-authority and redirect on the Python client: neither is a field
  assignment (each derives its pin from a workspace file), and both pins are
  preregistration facts, the reason `set-sweep-selection` stays Mac-authority
  too.

- **`remote submit-bundle --parallel <n>`** surfaces the multi-GPU fan-out
  the client and server have supported since 2026-07-22 and only the app's
  stepper could reach. The value is encoded by the existing rule (sent only
  when `n > 1`, the executor is `slurm`, and the verb shards), and the
  envelope echoes `parallelJobsRequested`, `parallelJobsEncoded` and
  `parallelJobsSuppressedBecause` so a request the rule suppressed never
  looks honored. The flag's help carries the field caveat: a fan-out can
  partially fail while the submit still exits 0, so verify the shard jobs
  landed.

- **A cross-substrate judge panel is authorable headlessly.** A local judge
  naming a model other than the study model must pin `judges[].revision` and
  `judges[].dtype` or freeze refuses under `judgeValidity` — and the
  `--judges` grammar (`<name>:<kind>[:<model>[:<provider>]]`) had no room for
  either, so the panel could only be built in the app. `pin-rubric
  --judge-pin <judge-name>=<revision>[:<dtype>]` declares them, repeated per
  judge and keyed by name (the `panel compile --seat` shape, chosen over a
  fifth colon field because position 4 is OpenRouter's provider and the two
  new fields are local-only). A symbolic revision and an out-of-vocabulary
  dtype are refused at the declaration, in the loader's own words.

- **The generation protocol and the exclusion rules gained headless
  writers, on both engines.** Six manifest protocol fields — `temperature`,
  `maxTokens`, `promptMode`, `samplesPerItem`, `seedPolicy`,
  `exclusionRules` — had no writer on either CLI (field-discovered: a
  stochastic replication arm of 25 samples × T=0.7 × 1024 tokens could not
  be authored headlessly and was cut from a study design). The Mac grows two
  verbs on the `set-sweep-grid` ownership pattern: `experiment set-sampling`
  owns the top-level sampling fields with merge semantics (only the flags
  given move; the joint stochastic rules stay `verify()` violations so the
  fields can be declared one flag at a time), and `experiment
  set-exclusions` owns the declared record-exclusion rules, refusing with
  the exclusion engine's own violation sentences. The Python client's
  `set-protocol` vocabulary gains `samplesPerItem` and `seedPolicy`, and all
  six fields now carry declaration-time value gates with cross-engine twin
  sentences — an out-of-vocabulary `promptMode`/`seedPolicy` is read by
  equality tests downstream and silently behaves as the default, and a
  non-numeric `temperature`/`maxTokens`/`samplesPerItem` bricks the manifest
  at the next decode, so both engines refuse at the write. The Mac's
  `POST /api/experiment/protocol` body accepts the three keys it was
  missing (`samplesPerItem`, `seedPolicy`, `exclusionRules`) with the same
  gates. **Behaviour break, deliberate:** a `set-protocol` call that used to
  write a malformed value for one of these six fields verbatim now refuses
  at `65` with nothing written.

- **Agreement rows carry their confusion counts, on both engines.** Each
  categorical `fieldAgreement` entry now emits `confusion` — `confusion[a][b]`
  is how many shared cells judgeA coded `a` while judgeB coded `b`, computed
  over the very label pairs Cohen's kappa was computed over, so the counts
  always sum to the entry's `n`. An analysis layer no longer re-derives the
  cell key, the intersection, or the label normalization to see WHERE two
  coders part ways — a kappa of 0.55 from one systematically-confused label
  pair is a rubric-anchor finding; the same kappa spread evenly is a
  noisy-field finding, and until now the report could not tell them apart.
  Numeric fields (which carry no kappa) carry no confusion block.

### Changed

- **A renamed duplicate can be measured against its source run.** The
  measurement-drift tolerance compared whole manifests minus the tolerated
  fields — and `name` is inside the content hash, so the sanctioned
  duplicate-never-edit path (duplicate a study, pin a new rubric and panel
  onto the duplicate, evaluate against the ORIGINAL run) refused on the one
  field duplication itself must change, and the only working path was
  mutating the source study in place. The epoch comparison now blanks `name`
  on both sides, by the tolerance's own rule: a rename cannot have affected
  a byte of the source run's generations. The rename is named in the
  tolerated-drift stamp like any other tolerated field, and `promote` —
  which never tolerates — still refuses a renamed manifest. Both engines.

- **`judgeValidity` accepts a single-judge panel, on both engines.** The gate
  required ≥ 2 distinct judges so the report could carry inter-rater
  agreement; the maintainer's ruling is that a researcher may declare any
  number of judges including exactly one. A single-coder design now freezes
  cleanly and carries a `judgePanelTooSmall` advisory — loud, never blocking —
  saying that no agreement statistics will exist for its codings, and the
  coding report records `fieldAgreement` as **absent with that reason**
  (`fieldAgreementAbsentReason`) rather than as an empty list, which would
  read as "agreement was measured and there was none". Zero judges is the
  state the gate now refuses: a judged instrument with no judge codes
  nothing. Multi-judge behaviour, including the distinctness rule, is
  unchanged. `pin-rubric`'s advisory was reworded off the old gate
  requirement, which it would otherwise still be asserting.

### Fixed

- **A materialized default is not manifest drift.** `experiment duplicate`
  decodes and re-encodes, so every non-optional defaulted field appears in
  the copy even when the donor's bytes never carried it — and the epoch
  comparison, over raw JSON where absent ≠ false, refused a duplicate's
  `recordTokenIDs: false` against its donor's absent key as generation-side
  drift: a refusal over a difference with no meaning. The comparison now
  canonicalizes default-valued keys to absent on both sides
  (`DEFAULT_VALUED_KEYS`, extensible); any non-default value still refuses,
  pinned in both directions. Python-only — the Swift comparison decodes both
  sides through the same struct and materializes symmetrically. And the
  stamped-mismatch epoch refusal stops offering `allowUnverifiedEpoch`, a
  flag consulted only in the unstamped branch — the offer sent an agent back
  to spend a queue slot learning it does nothing; it now names the real
  alternative (a measurement verb tolerates measurement-side drift and
  stamps what it tolerated).

- **`remote submit-bundle` can name the measurement verb's source run.** The
  duplicate re-measurement path (duplicate → pin rubric+panel → evaluate
  against the ORIGINAL run) worked locally (`experiment evaluate --run`) and
  in the node child (`bundle execute --source`), but the remote submission
  verb had no way to hand the child the flag its own help documents — and
  run discovery is scoped by experiment name, so a renamed duplicate's
  submission died at discovery ("no prior run with generations found") one
  statement before the epoch tolerance built to accept it. `remote
  submit-bundle --source <run-dir>` passes the override through (`sourcePath`
  on the wire, encoded only when given so older servers never see it), the
  envelope echoes `sourceRunRequested`, and the server keeps refusing an
  unreadable directory at submit time rather than on the allocation. Spelled
  `--source` — the one spelling every surface already uses.

- **`pin-rubric --judges` no longer wipes the judge pins it cannot see.**
  Re-declaring the panel replaced every row with nil `revision`/`dtype`, so a
  headless re-declaration silently destroyed pins written in the app and the
  study then refused at freeze for want of pins it used to have — the
  silent-drop class the sweep-selection merge exists to kill, one level down.
  The roster still replaces, but the pins merge field by field beneath it: a
  judge whose name, kind and model all survive keeps them, a judge whose
  model changed drops them (they identify the old bytes), and either way the
  echo says which under `result.inheritedFromExistingDeclaration` — the same
  key the selection merge uses. A `--judge-pin` naming no declared judge, or
  aimed at a judge kind that carries no pins, is refused at 64 rather than
  written and silently normalized away.

- **One artifact, one dose: the residual-norm denominator table now answers
  the same way in every verb, on both engines.** A `residualNormPerLayer`
  shorter than the artifact's depth used to produce four different outcomes
  from the same bytes — the server's condition path substituted 0.0 and
  refused as `degenerateData`, its sweep and variant paths clamped to the
  last entry and dosed the deepest layers with a shallower layer's number,
  the Mac condition path clamped too, and an EMPTY table indexed `[-1]` on
  the Mac and crashed outright. Three of the four were silent, which meant a
  sweep comparing layers could be comparing a denominator it shared. Two
  gates close it. At LOAD, both engines tie the table's length to the
  artifact's layer count and refuse a short one by name, with both numbers in
  the sentence: no writer produces a short table (extraction measures one
  norm per layer, `backfill-norms` writes exactly `layerCount`, the SAE
  by-id import slices the donor), so a short one is malformed rather than
  legacy. Absent stays legal and untouched — OptVec, J-lens and Gemma Scope
  report imports are BORN with no norms and acquire them through the
  backfill, and refusing them would strand three whole families. At USE, the
  four injection-building sites read the table through one accessor that
  refuses an uncovered layer in a sentence byte-identical across the engines.

- **A condition slot naming a concept that was never extracted refuses on the
  server, and is caught at verify.** The Mac has always thrown here; the
  server's `_condition_injections` did `continue`, so the slot vanished and
  the condition executed weaker — or, with one slot, as an unlabelled
  baseline — under a steered arm's name, with nothing in the run record
  saying so. The refusal is now the Mac's sentence verbatim, and
  `Manifest.verify` gained the condition-slot check the Mac's `verify()`
  already had, so the state is caught before a run is scheduled instead of
  after a model has loaded. SAE latent arms keep their carve-out: they live
  in their own collection, name features rather than attached concepts, and
  have their own declaration check.

- **SAE latent arms refuse to run on the Mac instead of quietly not
  happening.** This engine carries latent conditions faithfully and executes
  none of them — no local run path reads them — while their presence
  suppresses the baseline-only refusal, because on the engine that DOES run
  them they are perfectly good arms. A latent-only study therefore ran
  BASELINE ALONE and looked complete: the 2026-08-11 declared-`studyType`
  incident through a different door. `experiment run` now refuses before the
  model loads, naming latent execution as server-only and carrying the
  package-and-submit repair. Mixed manifests refuse too, deliberately — a
  partial run whose record says nothing about the arms it skipped is the
  failure this whole family of refusals exists to stop. `verify` and `freeze`
  stay open: a latent study authored on a Mac and submitted to a server is
  entirely legal.

- **Ad-hoc generate cells bound their injection layer instead of silently
  doing nothing.** `/api/generate` and `/api/generate/stream` clamped the
  layer they looked the vector row up with and then built the cell with the
  RAW request value, and nothing downstream revalidated: a layer past the
  model's depth dispatched to no hook at all and returned UNSTEERED output
  with HTTP 200, while a layer between the artifact's depth and the model's
  injected the artifact's LAST row where the artifact describes nothing. Both
  now refuse with a 400 naming the valid range and the depth it came from.
  Refusal rather than a shared clamp: the frozen-run paths resolve the layer
  once and use that value for both purposes, but this is the open playground,
  and a cell that quietly moves to a layer nobody asked for is a wrong number
  wearing a 200.

- **PCA in count mode stops at the data's rank.** `n` centred rows span at
  most `n − 1` dimensions, and the residual after that many deflations is
  float round-off — whose Gram trace is tiny but positive, which is exactly
  what the power iteration's relative degenerate-start floor accepts. Asking
  for more components than the data has therefore returned rounding noise
  normalised into unit "components", indistinguishable in the result from
  real directions: four rows at `count: 6` returned six, the last three
  carrying explained variances around 1e-15 — and the neutral-PC projection
  then removed the concept vector's component along those three arbitrary
  directions. Both engines now cap at `min(count, rows − 1)`, the bound the
  variance-target branch and the neutral-bank path always applied. Clamped,
  not refused, with an advisory when the request is trimmed: the PC count is
  a study-level knob applied to whatever neutral corpus each concept has, so
  over-asking is an honest declaration rather than an error.

## [0.9.3] — 2026-08-27

Everything new since the internal `v0.9.2` cut, which was never
published. The section below it stays the cumulative description of what
a first source release contains; 0.9.3 is the tag that description will be
published under, and the two are read together.

The theme is fidelity. The axes an extraction is read along became declarable,
transmissible, and provable rather than implied; four verbs the headless study
path had been missing arrived; and four rounds of external review were spent
turning quiet paths loud.

### Added

- **Four verbs the headless study path was missing.** `experiment detach` is
  attach's inverse — all-or-nothing, refused (`conceptInUse`) while any
  condition, sweep-selection instrument, variant condition, or perturbation
  policy still names the concept, because a detach that orphaned a declaration
  would be exactly the silent drop this instrument exists to prevent; the
  grand-mean corpus follows its targets out. `experiment set-sweep-grid` owns
  the sweep's layer × alpha axes, its instrument files, and its per-cell token
  budget, which until now were reachable only from the app's Optimizations
  panel — so the only headless way to obtain a grid was `duplicate`, which
  brings the donor study's concepts along as passengers that get swept but
  cannot be cited. Both axes must ascend with no repeats (`sweepGridRule`), the
  layer axis is stored as depth fractions and echoed both ways, and one rule
  answers the CLI, the HTTP route, and the panel. `authoring prompt <kind>`
  renders the generation prompt for each of five kinds of missing study data
  from a hashed template registry in `prompts/authoring-prompts/`: a
  workspace's own copy wins, each emission stamps both the wording's hash and
  the emission's, counts and shapes default while nothing that describes the
  study ever does, and the emitter is never the acceptor of what it asks for.
  `vectors mirror-poles` mints the opposite pole of a contrastive direction as
  an artifact with provenance instead of a hand-flipped mystery — a bit-exact
  IEEE sign-bit flip under a required new concept name, with `negatedFrom` and
  `polesSwappedFromSource` stamps, the sign-invariant statistics (norms and the
  whole residual-norm denominator family) preserved so α means the same dose,
  the neutral-mean tensors left alone, and the recipe identity hash dropped so
  promotion can never mistake a mirror for its source.
- **Representation Engineering, implemented in full.** The template-mediated
  RepE reader described in the block below reached that description in this cut:
  sign fixed by held-out classification per the paper
  (train-majority survives only as a stamped, reasoned fallback), the best
  layer stamped as a recommendation and never a silent selection, T+/T−
  instruction-pair templates for the paper's unsupervised construction with a
  seeded orientation draw the reference implementation lacks, and a declarable
  chat-template rendering that puts the LAT token where `rep_token=-1`
  actually reads on an instruct model. Reader-derived directions apply the
  probe's orientation, carry full provenance, and can attach — refusing α
  until their norm denominator is backfilled. Explained variance is now of the
  difference cloud, the power iteration's degenerate-start guard is real, the
  PCA path gained a cross-engine numeric fixture, and the normalization
  previously credited to RepE Appendix C.1 is attributed as ours at all four
  sites. `docs/REPE-IMPLEMENTATION-BRIEF.md` exists for those citations to
  land on.
- **The designated-reference recipe, on every surface that had promised it.**
  `ExtractionMethod.designatedReference` and `experiment attach --method
  designatedReference --reference <concept>` had existed since July, but the
  SwiftUI concept builder could not author it, the recipe enum could not name
  it, and the server's extract route would have accepted the method string and
  silently extracted from the *paired* files with no reference pinned. All
  three are closed: a builder family that routes the dataset pane to story
  rows, refuses a self-reference with repair text, gates the save on both
  corpora existing, and restores both its recipe and its reference pin when an
  existing story concept is selected; a local build that mirrors the CLI
  lifecycle exactly; a server build that pushes both corpora and verifies the
  pin through a response echo, with its own typed refusal
  (`ReferenceNotApplied`) for a server too old to honour it; and the browser
  workbench's own reference picker. The Python route refuses a missing or
  unknown reference at declaration time, in attach's words.
- **Reading position and extraction rendering as declared axes.**
  `experiment attach --reading-position '<label>'` on both command lines and
  the authoring route: eight positions, strict-parsed — writers never fall
  back — mutually exclusive with the legacy `--pool-from`, and refusing an
  unknown label or a template role under raw rendering at declaration time,
  hours before a GPU would have found out. Two of the eight are new:
  `contentOffset(k)` counts back into what the stimulus itself said rather
  than into the template's trailing scaffolding, and `meanContentFromToken(n)`
  pools content tokens only. `--extraction-rendering` gains a `voice` of
  `user` or `assistant`, where the assistant render is obtained by subtracting
  two template renders rather than by hand-written markers, and the
  combinations that mean nothing under it refuse by type. Absent stays
  byte-identical everywhere, but an explicit position moves the recipe
  identity, because position *is* identity. Both axes travel: `attach` copies
  a pinned artifact's rendering as it copies its position, `verify` gained the
  rendering-contradiction twin of the position check, and server-side
  extraction accepts the whole declaration with a response echo the client
  verifies, so an older server can no longer drop a declared axis in silence.
  Extraction diagnostics also gained per-direction logit-lens vocabulary — the
  top promoted and suppressed tokens at the declared validation depths — as an
  instrument, never a gate.
- **The app grows the surfaces the engine already had.** A Reader Lab that
  shows the faithful RepE pipeline and renders the engine's refusals verbatim:
  a template picker that marks T+/T− pairs rather than hiding mismatches,
  single-stimulus dataset authoring, a held-out split control that says when
  the sign rule will stand down, and per-layer results carrying sign
  convention, fallback reason, and variance basis. Reading position and
  extraction rendering get one shared control pair in the study-attach row and
  the concept builder, leading with a "method default" that declares nothing.
  Selecting in Data › Inventory now renders in the display pane, and a
  selected study shows its manifest there. The judge picker asks whether a
  candidate repository can answer a question at all — a text-generation
  architecture in `config.json` and a chat template — instead of offering
  every snapshot in the Hugging Face cache; OpenRouter judges, a complete
  backend the robustness path had never wired, route through one dispatch
  shared with the local and server paths, offered only when a key is present.
  The Prompts row gained a numeric field beside its stepper and states the
  coherence file's actual supply instead of truncating silently.
- **A workspace contract that can prove it is the machine's.** The generated
  `AGENTS.md` header now declares a SHA-256 of the body it wrote. A hash that
  matches the body is proof of machine ownership, so an app update refreshes
  such a file to the shipped contract — atomic replace, one info notice — at
  every workspace open and every CLI verb, while a hash that no longer matches
  is an edit and the file is the researcher's, untouched. Legacy hashless
  headers keep the advisory-only behaviour until one manual regeneration
  graduates them. The contract itself gained §4.3's `--reading-position` and
  `--extraction-rendering`, §4.15's sweep workflow (the concept-swap chain
  with `detach` last, the grid dialog, and the missing-data rule ending at
  independent review), and §8's `cluster auth open` — the documented move for
  an expired SSH master, and one only a human can answer.

### Changed

- **`repeLAT` is now `pairedDifferencePCA`.** What that family computes was
  never LAT, and the label now says so: **paired-difference PCA
  (RepE-inspired)** in prose, `pairedDifferencePCA` in code. No artifact
  moves — the sidecar `recipeMethod` value stays `"repeLAT"` and the manifest
  method stays `"lat"`, permanently, because they are written bytes.
- **`set-protocol` refuses an unknown key instead of dropping it.** On all
  three surfaces — the Python store, the HTTP authoring route, and the Mac's
  `POST /api/experiment/protocol`, where a default `JSONDecoder` had been
  discarding out-of-vocabulary keys with no warning at all — an unknown key
  now refuses before anything is written, naming the key and listing the
  closed vocabulary; valid keys in the same invocation do not land either.
  `outcomeInstruments` and `sweep` join that vocabulary, which
  `docs/PORTABILITY-CONTRACTS.md` had claimed for them while neither was
  actually accepted. **Behaviour break, deliberate:** a script relying on the
  client's old tolerate-and-warn (exit `0` plus an `ignored` list) now stops
  at `64`.
- **A judging run never downloads weights on your behalf.** Selecting an
  uninstalled curated judge tier used to reach the hub loader and pull up to
  35 GB outside the named, cancellable Install flow. Uninstalled tiers are now
  listed flagged and refused before a single generation — at the picker, at
  study evaluate, and in the coding loops alike — and local judges a server
  workspace would skip are flagged up front, quoting the route's own sentence.
  One `JudgeReadiness` rule (provider pin, key presence, installation,
  capability, in execution's order) answers the picker, the Run gate, and
  execution.
- **Panels stop paying for their data before they may draw.** Switching to the
  Agents tab used to run two full `runs/` walks, a config decode per run
  directory, and a SHA-256 of every saved agent artifact — synchronously, on
  the main actor, inside `onAppear`. A summary-index layer loads row-ready
  entries off the main actor and defers hashing and robustness-report reads to
  a background phase after the list is visible, latest-wins, with the previous
  rows on screen while a rescan runs; the dashboard, the Studies panel, and the
  optimization-runs view moved onto the same pattern. No split-view geometry
  changed.
- **The Optimizations panel saves through the sweep-grid gate.** It validated
  with its own copy of the checks, minus the two ascent rules, so the app could
  save a grid the CLI refuses. It now writes through the same fully-gated store
  verbs the CLI and the route use, with the selection validated before the grid
  write so a refusal on either half leaves the manifest untouched, and a live
  caption resolving typed fractions to absolute layers at the pinned model's
  depth.

### Fixed

Four rounds of external review — the fourth through the seventh — account for
most of what follows. Their findings are grouped by class rather than listed.

- **Cleanup that proves ownership.** Every unlink in the runner goes through
  one primitive that proves the name still refers to the reserved inode before
  removing it, so a foreign file moved onto a staging path survives cleanup and
  the refusal says so. Evidence staging lives on one file descriptor from
  reservation to commit, with the published inode proven to be the verified one
  (`stagingPathHijacked`, `stagedBytesChanged`), and the portable ledger rides
  the same transactional no-replace commit.
- **Reading-position and rendering fidelity.** Python artifact-attach carries
  all eight reading positions in the Swift twin's codable shape instead of
  collapsing six of them to `lastToken`. Unknown keys under a `chatTemplate`
  rendering are typed refusals on both engines, at declaration and on sidecar
  reads alike, so a misspelled `addGenerationPrompt` can no longer quietly
  become the default. Swift's recorded-raw decode refuses parameters exactly as
  declaring does.
- **Held-out sign provenance.** Derivation is convention-aware: under held-out
  pair agreement the fitted direction ships unflipped, and a train/held-out
  disagreement is stamped rather than silently resolved. The companion claim
  that the same interaction distorts `heldOutAccuracy` was audited and refuted
  — the probe is sign-invariant by construction, now pinned by test.
- **Depth witnesses.** Cached model depth is collected only from artifacts that
  state they cover it (a new `coversModelDepth` stamp), requires revision
  agreement, and refuses on conflict rather than choosing a network; a pinned
  copy of a partial source can no longer launder itself full. Reader-derived
  artifacts are partial by construction, and the depth-display paths pass the
  pinned revision the resolver already accepted.
- **Emission and packaging.** The authoring-prompt registry ships inside the
  wheel, so a `pip install` with no checkout beside it renders prompts instead
  of refusing with a prerequisite no repair could satisfy; a byte-identity gate
  keeps the packaged copies equal to `WorkspaceSeed/`. Custom reader templates
  dry-run the job's own render before any canonical byte moves,
  `promptInstanceHash` identifies the emission beside the wording's hash, and
  authoring counts are typed, ceilinged, and cross-checked.
- **Smaller corrections.** J-lens fp32 promotion unwinds itself on a mid-loop
  exception; a whitespace-only composition level stamps null instead of
  claiming a contribution, and two of them compose to nothing whichever side
  the blank is on; chat-template sidecars must be non-empty regular files whose
  JSON parses; evidence refreshes carry their own generation counter and
  snapshot verification; and client evidence downloads are gitignored, because
  study data never belongs in this checkout.

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
- **Representation Engineering, implemented and labelled as such.** The
  template-mediated RepE reader (Zou et al., arXiv:2310.01405): a hashed task
  template registry, the LAT token at the rendered scaffold's final position,
  mean-centred PCA over paired differences, both of the paper's contrast
  constructions (a supervised content contrast and the unsupervised T+/T−
  instruction pair), held-out selection of sign and layer, persisted fit
  parameters reused at inference, and either rendering — a raw scaffold or the
  family chat template. Its reading direction converts to a steering vector
  through an explicit, provenance-stamped conversion that is never described
  as reproducing the paper's control experiments. The separate
  PCA-over-difference-vectors family is called **paired-difference PCA
  (RepE-inspired)**, because it borrows the paper's arithmetic and none of its
  pipeline; the per-pair normalization and norm-matching it adds are ours and
  are attributed as ours. `docs/REPE-IMPLEMENTATION-BRIEF.md` carries the
  schemas and the itemised faithful-vs-departure table, including what is not
  implemented (LoRRA control training, strided control bands, prompt-span
  steering).
- **Documentation and legal files.** A generated CLI reference, a methods
  note, a RepE implementation brief, an end-to-end study guide, a
  results-architecture note, this changelog, plus `LICENSE` (Apache-2.0),
  `NOTICE`, `SECURITY.md`, and `CITATION.cff`.

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
