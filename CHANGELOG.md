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

- **`steerlab-cli remote resubmit <job-id> [--walltime hh:mm:ss]` — the
  managed resume of a checkpointed job.** Field incident 2026-08-29: a
  sharded run's shard hit its walltime, checkpointed cleanly (the trap
  flushed its records, wrote the resume pointer, exited with the checkpoint
  code 85, record status `checkpointed`) — and the designed resume,
  re-executing the shard's own rendered `run.sbatch` so the resume pointer
  beside its child record is consulted, had **no managed spelling**: neither
  the `remote` nor the `cluster` family could run it, and the fix was a
  hand-rolled scheduler command. The workspace operating contract calls that
  a missing verb; this is the verb.

  The CLI calls the server's existing manual-resume implementation (the
  Resume button's `POST /api/jobs/{id}/resubmit` — the same single core
  auto-resubmit uses), which locates the submission's rendered sbatch script
  from the job record and re-submits it **byte-for-byte**. `--walltime`
  (new, both surfaces) raises the scheduler limit for the continuation on
  sbatch's **command line** — flag beats `#SBATCH` header — so a shard that
  parked AT its limit gets a longer one while the identical-script contract
  (gres, cleanup trap, record path) holds untouched. The override is encoded
  only when given; a server that understood it echoes `walltime` back, the
  envelope carries `walltimeRequested` beside the response, and the CLI
  warns when the echo is missing (an older server ignores unknown fields
  silently). Refusals are typed (`resubmitRefused`, 65; `notFound`, 66) with
  runnable repairs; a resume that races an existing continuation reports
  `alreadyResumed` and submits nothing; a sharded parent fans the resume out
  to its checkpointed shards and needs nothing else — it merges when its
  shards finish. The gate also re-asks the scheduler before refusing a stale
  record: a job recorded `running` or `failed` whose sacct ExitCode is the
  checkpoint code is corrected to `checkpointed` and resumed, while
  succeeded and cancelled records are never re-asked (cancelled beats
  checkpointed). An idempotent re-run of a continuation whose run already
  completed remains the child's own answer (`alreadyComplete`), per the
  bundle-execute reliability contract.

- **The standalone capability battery: `steerlab-server battery run
  <battery-file> --agent <ref>…`, with a charter, a second operating regime,
  and a pinned evidence run.**

  **The motivating insensitivity.** The per-cell battery a sweep already runs
  is short and greedy — 24 tokens, temperature 0 — and it scored accuracy
  **1.0** at a dose that three independent instruments had separately
  confirmed degraded. Nothing was wrong with the scoring. A short greedy
  answer simply has no room for the failure modes an agent actually fails in:
  length inflation, variance collapse, incoherence. An instrument that cannot
  move where the thing it measures moves is not a lenient control; it is a
  control that reports a pass by construction.

  **The charter (maintainer ruling, 2026-08-29)**, recorded in the code
  (`battery.py`), in the authoring brief (`battery generation-prompt`), in
  `docs/CLI-REFERENCE.md` §6.10 and in `docs/CONDUCTING-A-STUDY.md` §5.2:

  1. A battery is **ex ante justified, study-blind, and fixed**. An agent
     precedes any study it appears in, so the battery inherits no study's
     protocol, domain, or difficulty target. It is a **floor** —
     instruction-following, basic multi-step reasoning, factual recall, fluent
     extended generation, moderate difficulty — never a frontier
     differentiator. The recorded boundary example: a study measuring relative
     performance on very hard, lengthy real-analysis proofs must **not** find
     that capability probed, much less gated, by the battery. Study-specific
     capability is the study's own business, settled by performance in the
     study itself.
  2. The **operating regimes are generic and owned by the battery spec**: both
     greedy short-answer items *and* long-form generative items at a standard
     positive temperature with a generous token budget — justified because
     agents are used generatively, never because some study samples.
  3. **Sensitivity is validated, never defined, by known positives.** A
     battery version proves itself by *failing* a known-degraded state; if it
     cannot, it is revised on ex ante grounds until it can. The known positive
     is a check on the instrument, not a tuning target — a battery tuned until
     one dose fails has become the difficulty target clause 1 forbids.

  **`batteryFormat: 3`**, additive: format 2 plus a `generativeProtocol`
  header block (`temperature` 0.7, `maxTokens` 512, `samplesPerItem` 3 by
  default, all the battery's own) and a third scoring mode
  `generationHealth` — items with no answer and no grading, generated under
  that protocol and read for word count, distinct-2 and completion rate, per
  sample, with means and population spreads per agent. A format-3 battery is
  deliberately **not pinnable** into a study on either engine
  (`battery.PINNABLE_FORMATS`, `PinShapeValidation`): scored per condition
  inside a run matrix, a sampled multi-sample regime would be a second outcome
  measure wearing a control's name. Existing format-1 and format-2 files are
  byte-identical in meaning and keep their pinned hashes.

  **Agent references** (repeatable `--agent`, the `panel compile --seat`
  shape, plus `--agents a,b,c`): `baseline`, a condition spec
  `<concept>:<layer>:<alpha>` resolved against the workspace's vector
  catalogue, a promoted agent artifact by name or path, and `<name>=<ref>` for
  any of them. Doses are built through the same `variant_injections` path a
  study condition uses, so a battery dose and a study dose are one arithmetic.

  **Sequential agent custody.** Agents group by `(modelID, revision, dtype)` in
  declaration order, and every container the remainder of the run will not use
  is released before the next model loads — the agent twin of the judge-column
  release seam, with the same guarantee: peak device memory is the **max** of
  any one still-needed model, never the **sum**. Order is never reshuffled;
  `--dry-run` prices what the order costs first.

  **Honest pricing.** Records are `agents × (graded + health × samplesPerItem)`
  — a sampled long-form item is one record *per sample* — divided by the
  sampled family's own observed rate where there is one, times the standard
  margin, plus the fixed startup cost **per model load**, because sequential
  custody means a two-model run pays two cold loads. No throughput history
  gives no estimate rather than an invented one.

  **Evidence.** An immutable `runs/<stamp>-battery-run/` with `battery.jsonl`
  and `battery-report.json`, keyed by pins throughout (battery digest, model
  revision, vector-artifact hashes, dose, protocol) so two studies using the
  same agent at the same dose cite one floor reading instead of each buying
  their own. Health comparisons speak the sweep's own coherence vocabulary
  (`distinct2Ratio`, `lengthInflated`), and everything is **reported, never
  gated**. Sampled generations are seeded per record under the house
  `derivedSHA256` discipline keyed on the battery digest with an empty
  condition field — common random numbers across agents, so a health
  difference is the intervention and not the dice.

  **Refusals**, all upstream of the run directory so a refused invocation
  writes nothing: `notFound`, `malformedBattery`, `batteryLintBlocked` (the
  battery must lint clean before a number from it is worth having),
  `unknownAgent`, `ambiguousAgent` (a concept naming two extractions refuses
  and lists both — which extraction a dose was read at is a provenance fact),
  `agentModelConflict`, `agentReference`, `agentNameCollision`, `noAgents`,
  `modelRequired`. New closed-vocabulary advisory
  `singleRegimeCapabilityReading` (both engines) for a floor reading taken
  against a one-regime battery.

  The `battery` family joins the agent path for this verb; `battery lint` and
  `battery generation-prompt` keep their historical human output and exit
  codes byte for byte. The verb exists on the **engine alone** — it loads
  models, so it cannot live on an authoring client, and unlike every other
  model-loading verb it is not manifest-shaped, so no bundle route carries it
  either (`docs/PORTABILITY-CONTRACTS.md` §10.9). The battery CONTENT itself
  is not authored in this change: `battery generation-prompt` now emits the
  charter, and the first floor battery written against it will be validated
  against a positive control before it ships.

- **An in-flight model load is cancellable, and a load that would download
  says so first.** Field incident 2026-08-29: selecting an uncached 27B on
  the app-managed local engine silently began a ~55 GB hub download inside
  the load, which held the engine's only resident-model slot —
  `POST /api/models/unload` answered `unloaded: 0` (a loading placeholder is
  never evictable), no cancel existed anywhere, the busy-slot refusal said
  "cancel the work holding it" with no such cancel in existence, and the only
  way out was SIGTERMing the engine process by hand.

  Engine: `POST /api/models/load/cancel` interrupts in-flight load(s)
  cooperatively — a hub download stops within seconds (cold-cache downloads
  now run through the install verb's cancellable child process instead of
  inline in `from_pretrained`, and carry that verb's semantics: the FULL
  snapshot, which the announced size prices honestly — on a legacy
  multi-format repo that is more than `from_pretrained` used to fetch, and
  it is what the install-complete marker has always asserted), a weight
  copy stops at its next phase boundary — and the slot frees through the
  ordinary failed-load path. The
  load stream's FIRST event announces an impending download with its
  hub-metadata size (`willDownload` / `downloadBytes`), heartbeats carry
  live download progress, `GET /api/models/preflight` answers
  cached/downloadBytes before a load commits the slot, unload's answer names
  the cancel verb when a loading slot blocked it, and the busy-slot refusal
  names a repair that exists. Capabilities: `chat.loadCancel`,
  `chat.loadPreflight`.

  App: a server-workspace load of an uncached model asks first
  ("Download ~N GB and load?") — the same rule the local tier always
  enforced with its separate Download… verb; Cancel Load now also cancels on
  the engine (capability-gated); and the Local Engine Details pane gained a
  running-engine section: resident models with live load phase (download
  progress included), Unload Models, Cancel Load, and Restart Engine…
  (identity-gated stop, then re-provision).

- **`evaluate` gained a seeded, stratified subsample:
  `--sample-per-condition <n>` with `--sample-seed <hex-or-int>`.**
  Field-discovered 2026-08-29, and the discovery is the shape of the gap: a
  7,200-record corpus needed judged classification, the preregistered design
  was a stratified 2,400-record subsample, and the instrument had no honest
  spelling for it. `evaluate` codes a whole source run on both engines, so the
  only routes available were coding all 7,200 — a different design, judged at
  three times the cost the power computation asked for — or hand-building a
  run directory holding the chosen records, which the reviewing agent
  correctly refused as evidence-chain corruption under immutable `runs/`. The
  operation was legitimate and preregistered; nothing could say it.

  **Both flags or neither.** A sample with no seed is a subsample nobody can
  redraw; a seed with no sample size would be stamped on a coding it did not
  shape. Either half alone refuses at 64, and there is deliberately no
  fraction spelling — an absolute per-condition `n` is what a power
  computation produces, and "40% of whatever happened to be there" is not a
  design.

  **The draw** (`EvaluateSubsample` / `evaluate_subsample.py`, twins): within
  each condition, `floor(n / P)` records per promptID over that condition's
  `P` promptIDs, the `n mod P` remainder handed out one at a time in seeded
  promptID order — a promptID already at its population is skipped and its
  quota passes on, so exactly `n` records are drawn even from a ragged source
  run; within each `(condition, promptID)` cell the records are drawn over
  `sampleIndex` ascending. Every choice is a partial Fisher–Yates over
  SplitMix64 seeded from SHA-256 of the seed and the stratum's length-prefixed
  names — the neutral token bank's primitive, adopted for its reason (2026-08-28
  audit, convention note 9): a subsample whose membership can move under a
  runtime upgrade while its seed stamp stays identical is not reproducible
  evidence. promptIDs sort by UTF-8 bytes, because Swift's `String` ordering
  normalizes and Python's does not. The same `(source run, n, seed)` therefore
  selects **byte-identical records on both engines**, pinned by a shared
  fixture of `(condition, promptID, sampleIndex)` triples and by the
  derivation string's SHA-256 in both suites.

  **It never clamps.** An `n` above any condition's codeable population
  refuses, naming the binding condition and its count: clamping would code a
  smaller design than the preregistered one while every stamp still said
  `samplePerCondition: n`.

  **Per-response coding only.** The paired judge's unit of analysis is a
  `(baseline, variant)` PAIR rather than a record — "baseline" there is not a
  sampled condition but the other half of every comparison — so a
  per-condition record count does not name a set of pairs. A paired rubric
  refuses rather than half-executing a correct-looking command line, the rule
  `--shard` already follows; so does the flag on any non-`evaluate` verb.

  **Loud stamping, non-negotiable.** `coding-report.json` and the evaluate
  run's `config.json` both gain a `sampling` block —
  `{samplePerCondition, sampleSeed, sampledRecords, sourceRecords, rule}`,
  where `sampleSeed` is the canonical `0x` + 16 hex digits (JSON has no
  unsigned 64-bit integer, and a decimal a parser rounds is a seed that no
  longer redraws its own sample) and `rule` is the derivation verbatim. Every
  human line reads `coded N of M (seeded subsample)`, and so does the
  envelope's success message. The block is additive: its ABSENCE means the
  full corpus, so every report written before this reads back byte-identically.

  **Every surface.** `steerlab-cli experiment evaluate`, `steerlab-server
  experiment evaluate`, `bundle execute --verb evaluate`, `study submit --verb
  evaluate`, and `remote submit-bundle` (wire fields `samplePerCondition` /
  `sampleSeed`, encoded only when given so older servers see the body they
  always saw, echoed at `result.samplePerConditionRequested` /
  `result.sampleSeedRequested`). Submit-time refusal, not compute-node
  refusal, for the same reason `--source` gets one. The walltime preflight now
  prices a sampled evaluate at the SAMPLED record count rather than the full
  manifest matrix — an evaluate coding a third of a corpus was asking for
  three times the walltime it needs.

- **The study's system prompt gained a headless writer on both surfaces:
  `experiment set-system-prompt <name> "<text>"`.** Field-discovered
  2026-08-28, and the discovery is the shape of the gap: a persona-carrying
  replication study — one whose donor manifest arms every arm with a
  judge-persona frame — could not be authored headlessly. On the Mac there was
  no writer at all; the Study Setup panel's "Baseline system prompt" field was
  the only one. On the cross-platform client the field *was* reachable as a
  `set-protocol` key, but nothing on either surface **named** it, so an
  authoring agent looking for the verb found none. Running the study without
  the persona would have been a different study, so it correctly stopped
  rather than improvising one. Reachability is not authorability.

  Both spellings now exist and behave identically — `steerlab-cli experiment
  set-system-prompt` and `steerlab experiment set-system-prompt`, over
  `ExperimentStore.setSystemPrompt` and `experiment_store.set_system_prompt` —
  and the engine redirects the verb naming both, as its siblings do. Inline
  **text**, because that is what the manifest stores: no path field and no
  hash beside it, so there is deliberately no `--file` (a file the manifest
  does not pin would read as provenance the study does not have) and no hash
  flag (a record's `systemPromptHash` stamps the *effective* prompt, computed
  at run time). The value is trimmed and `""` clears — byte-identical to the
  panel's own save path, so the two writers cannot disagree about the bytes a
  study was armed with. Draft-only through the standard gate. `systemPrompt`
  **stays** a `set-protocol` key: it is a plain field assignment, unlike the
  two derived-pin declarations beside it.

  **What is written reaches the model, and the verb says by which route.** The
  maintainer's condition on this writer was that the text be *either prepended
  to the initial prompt or inserted as a genuine system prompt depending on
  model capability* — which is what the renderer already does, and is now
  asserted end to end rather than assumed: a family whose chat template has a
  system role gets a genuine system turn, Gemma (which has none) gets the same
  text prepended to the first user turn, and `rawCompletion` prepends it to the
  prompt text. Every route delivers it, so there is no prompt-mode gate. The
  capability itself is now named on both engines
  (`PromptRendering.hasSystemRole` ↔ `prompt_render.has_system_role`) instead
  of being re-derived from the family at each branch, `result.delivery` echoes
  `systemTurn` or `prependedToFirstUserTurn`, and the help text states the rule
  so a researcher arming a persona knows what the model sees before running
  anything. Composition is unchanged: an agent arm reads under *persona*, blank
  line, *this frame*.

  **The one path a declared frame does not reach is no longer silent.** A
  pinned task item whose scripted `transcript` opens with its own `system` turn
  replaces the study frame for that item — the transcript is the more specific
  declaration, one rule on both engines — so a persona typed into the frame
  simply does not arm those items. That is now the closed-vocabulary advisory
  `systemPromptNotApplied` (non-blocking, exit code unchanged), raised at the
  moment the frame is declared and naming the real counts: *N of M pinned items
  … the frame declared here reaches the other K only.* Its detail sentence is a
  twin literal like every other cross-engine sentence.

- **A judged run releases each finished judge's model before the next one
  loads, so a panel needed sequentially costs the MAX of its models, not the
  SUM.** The maintainer's ruling, verbatim: *any runs that require two models
  will need to unload and load models in order not to OOM. We need to ensure
  this happens.* The incident behind it (2026-08-28, the two-judge
  calibration): a per-response-coding evaluate with the study model
  (~55 GiB bf16) as reference coder and a second local judge (~24 GiB) codes
  judge-column-outer — coder A finishes all 1,200 records, then B starts.
  When B came to load, A's container was still resident (the registry does
  not evict while a slot is free, and there were two), and the loader's
  capacity gate refused on headroom: ~22.7 GiB needed against 23.3 GiB free
  on an 80 GiB A100. A's container was dead weight; nothing in the run would
  ever touch it again.

  The guarantee now: at every model-boundary seam the engine computes the
  set of models the REMAINDER of the run still needs and releases every
  resident container outside it before loading the next one. The rule is
  `tasks.judge_models_still_needed` — the resolved model of every local
  judge from this column onward, plus the study model when a later stage
  still generates (`_pipeline_needs_model` over the stages after evaluate;
  analyze and rescore are CPU-side). Both roster loops are instrumented, the
  per-response-coding one and the paired one; candidates are the run's own
  models only, so an unrelated resident (chat, variant generate) is left to
  the cache policy, which is deliberately unchanged. A single-judge run
  releases nothing, and two consecutive same-model columns stay warm rather
  than reloading the weights they are about to use.

  The mechanics reuse the one eviction implementation rather than growing a
  second: `ModelRegistry.release_models` drops the named containers through
  `_evict` (skipping busy slots exactly as `unload`/`_make_room` do — the
  pipeline's chain-held model is locked for the whole chain and can never be
  taken away mid-chain), and `model_loader.free_device_memory` — factored
  out of `_evict` — is what actually returns blocks to the driver, because
  a dropped reference alone leaves them in torch's caching allocator, which
  is what `cuda.mem_get_info` and therefore the capacity gate read. The
  CLI/bundle path has no registry, so the column's private copy is dropped
  by an `ExitStack` callback (previously it stayed pinned by the judge
  closure until the NEXT column's callable replaced it — after that model
  had already loaded) and the same trim runs at the boundary. Every release
  is loud in the run log: `released 'X' (~55.0 GiB) from cuda:0 — column 'A'
  complete, next judge 'B' needs 'Y'`.

  What this restores: a two-local-judge panel in ONE run on a single 80 GiB
  device, which is what the engine's own `fieldAgreement`/`confusion` blocks
  (multi-judge `codings.jsonl`) need to exist at all.

  The two engines answer this problem differently ON PURPOSE, and both are
  right for their memory model: the Mac REFUSES up front
  (`SweepObjectives.localJudgeSlotProblem` declines a two-model sweep before
  any work starts, because an MLX unified-memory container is not cheaply
  reclaimable mid-run), while CUDA can hand the weights back and so releases
  between columns. No Swift changed.

- **The cross-platform client gained `experiment set-parser` and `experiment
  set-instrument-scope`.** Both were Mac-only, on the reading that authoring
  is "Mac-authority". The maintainer's ruling replaced that reading: *it is
  not the Mac that matters, it is the client machine — the separation is
  between the client and the running hardware.* An engine on compute
  hardware never authors (its workspace is a cache, which is why the engine
  still redirects both verbs); a client authoring a LOCAL workspace is as
  legitimate on Linux or Windows as on a Mac, exactly as `attach` already
  was, deriving `stimulusSetHash` from workspace bytes on any platform. So
  these two joined it, spelled and refusing identically:
  `experiment_store.set_numeric_parser` and
  `declare_outcome_instrument_scope` mirror `ExperimentStore.setNumericParser`
  and `declareOutcomeInstrumentScope` sentence for sentence — the registry's
  own undefined-parser and malformed-entry refusals, the twin
  unknown-`responseFormat` and zero-selection literals, `""` clearing both
  (and, for the scope, clearing *before* the prompts-file guard, so a stale
  declaration is removable in the dropped-pin, moved-file and drifted-bytes
  states that make clearing necessary), and an out-of-vocabulary value
  malformed at `64` rather than refused at `65`.

  What the carve-out was actually protecting is untouched: **the pin is
  computed from workspace bytes, never typed by a caller.** There is no
  `--registry-hash` and no `--item-ids-hash` on either engine, the keys stay
  out of `PROTOCOL_FIELDS` so `--set parserRegistryHash=…` still refuses, and
  the scope's row set is read by a torch-free reader
  (`experiment_store.scope_items`) held by test to the same items the run
  path's loader selects. The engine's redirect now names the client's
  spelling alongside the Mac's, read from the client's own verb table — a
  Linux caller cannot run `steerlab-cli`, and a repair they cannot run is not
  one. `test_client_cli.py` now compares the two surfaces exhaustively from
  the real tables, with every deliberate exclusion named and its
  `set-protocol` route asserted, so the next redirected verb cannot silently
  regress the promise.

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
  at the declaration rather than producing zero records at the run. Neither
  is a field assignment (each derives its pin from a workspace file), and
  both pins are preregistration facts — the reason `set-sweep-selection` is
  not a protocol field either, and the reason no surface accepts a pin as
  input. The engine redirects both; the client implements both (see above).

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

- **The evaluate subsample's flags-only spelling is superseded by a
  declaration: `experiment set-evaluation-sampling <name> <n> <seed>`, on both
  authoring surfaces.** External review round 12, finding 4, maintainer-ruled.
  The seeded subsample shipped hours earlier as CLI flags plus run stamps, and
  the review's point is accepted: **a stamp records what HAPPENED, and
  "preregistered" is a claim about what was decided BEFORE anything ran.** A
  claim like that has to be substantiated in the artifact chain or it is not
  evidence — a `sampling` block that appears only because someone typed two
  flags proves the draw was seeded, not that it was planned.

  The reviewer's remedy was a frozen, hashed design document. That does not fit
  this house's flow, because judged re-measurement deliberately runs on
  **never-frozen duplicates**. The adapted remedy gets the substance instead:
  the sampling design becomes a DRAFT MANIFEST DECLARATION —
  `evaluationSampling: {samplePerCondition, sampleSeed, rule}` — and every run
  stamps the manifest snapshot into its own `experiment.json`. **A plan
  document is pre-registration; the snapshot is provenance.** The design now
  travels with the evidence rather than with the command line that produced it,
  and a reader holding the run holds the design.

  **The rule is derived, never typed.** The block's third field is
  `EvaluateSubsample.rule` / `evaluate_subsample.RULE` verbatim, taken at the
  write — the same guarantee `parserRegistryHash` carries, for the same reason:
  a caller-supplied rule would let a study claim a derivation nothing
  performed. There is no `--rule` on any surface and there will not be one.
  Both-or-neither applies inside the declaration exactly as at the flags
  (`n` with no seed, a seed with no `n`: 64, nothing written), and `<name> ""`
  clears it. The writer exists on **both** authoring surfaces per the
  authoring-client doctrine (`docs/PORTABILITY-CONTRACTS.md` §7) — twin store
  setters, twin refusal sentences, the same flat `result` keys — and the engine
  redirects it like its two siblings, because it executes and does not author.

  **The flags become a cross-check, never an override.** With a design
  declared, `evaluate` draws it with no flags at all. `--sample-per-condition`
  / `--sample-seed` may still be given, and any inequality with the declaration
  refuses at 64 naming both values; the repair is to drop the flag or to
  declare the design you actually want, never to force. A flag that won would
  code one design while the run's snapshot recorded another — the exact silent
  substitution the refusal vocabulary exists to prevent. Reconciliation lives
  in the evaluate TASK on both engines, the first point that holds the
  manifest, so the local CLI, `bundle execute`, a submitted argv and any
  library caller are all checked against the same bytes; submit-time paths run
  the check early against the bundle's own manifest, so a contradiction never
  costs a queue wait, and the walltime preflight prices a declared design with
  no wire fields at all.

  **Undeclared studies are untouched.** The flags-only path behaves exactly as
  it did and its stamps still say SUBSAMPLE on every line — the ad-hoc route
  stays honest, it simply cannot claim the declared route's provenance. The
  declared route's stamps carry one additional key, `declared: true`; its
  absence is the older spelling, not a finding.

  **Validation splits where knowledge does**, and the split is documented
  (§7 of the contracts, §3.5 of the reference). Declare time and `verify()`
  check what a desk can check — a whole `n` of at least 1, a seed that parses
  as 64-bit unsigned, and a `rule` this build actually derives (an older rule
  version would not redraw the same records, which is what the version marker
  is for). The POPULATION check cannot run at either: the source run the design
  will be drawn from need not exist yet, and usually does not, since declaring
  before running is the point. It stays at `evaluate`, and still refuses rather
  than clamping. `evaluationSampling` also joins the epoch guard's
  `MEASUREMENT_FIELDS` on both engines — it chooses which of a completed run's
  records get judged, so it cannot have moved a byte of that run's
  generations, and the duplicate-and-declare flow depends on that.

  Recorded, not built (`docs/PORTABILITY-CONTRACTS.md` §10.9): the client's
  `runner submit` and composite `run` still do not parameterize measurement
  verbs at all — no sample flags and no `--source` — so an *undeclared*,
  unnamed-source subsample still needs another surface. The reviewer's
  observation stands; it is one flag-pair away, and the phase's rule is that
  the client invents no surface the runner protocol does not already expose.

- **"Mac-authority" is retired as a principle: the rule is CLIENT-authority.**
  External review round 12, finding 8, maintainer-ruled verbatim —
  *"Mac-source-of-truth was always my sloppy way of saying client (not
  cluster) is the source of truth."* The principle, stated once and now
  reflected wherever the topic appears: **the authoring client's workspace is
  the source of truth; the engine on compute hardware never authors, and its
  workspace is a cache.** The boundary is the client versus the running
  hardware — never macOS versus everything else. Round 11 had already moved
  the *behaviour* (the cross-platform client authors, and the engine's
  redirects name its spelling); this closes the gap between that behaviour and
  the words around it.

  Swept: the workspace contract (`AgentContract.swift` and its byte-identical
  mirror `docs/AGENTS-WORKSPACE-DRAFT.md`), `README.md`,
  `docs/CLI-REFERENCE.md`, `docs/PORTABILITY-CONTRACTS.md`,
  `docs/CONDUCTING-A-STUDY.md`, the engine's own principle docstrings
  (`cli_envelope.py`, `client_cli.py`), and the four twinned refusal/comment
  sentences that said "the Mac workspace is the source of truth" where they
  meant the client's.

  **`macAuthorityVerb` keeps its name and its meaning, as a legacy MACHINE
  CODE only.** It is a stable identifier agents and the Swift CLI already
  switch on, so it does not move; every place it is documented now carries a
  one-line note that **the name is historical** and that what it reports is
  "this engine executes, it does not author", never "author on a Mac". No
  behaviour, exit code, or envelope field changed.

- **Windows is stated as out of scope rather than promised as supported.**
  The maintainer's ruling, verbatim: *"I am not interested in supporting
  windows - as I have no way to test it. Maybe someone else who clones the
  codebase could add it on their own. But we shouldn't expend efforts in that
  direction and shouldn't promise it."* `AGENTS.md` path 4 had said Windows
  is "client-only: authoring, freezing, packaging and remote submission all
  work there" — a promise nothing in this project tests. Every mention
  (`AGENTS.md`, `README.md`, `docs/CLI-REFERENCE.md` §1.4 and §1.5,
  `docs/ONBOARDING.md`, `docs/PORTABILITY-CONTRACTS.md` §9.2) now says the
  same smaller true thing: **untested, unsupported, out of scope**, with the
  honest remainder kept — the client is pure Python and imports nothing
  platform-specific on its authoring path, so authoring is the half one would
  *expect* to work, and a downstream user who wants Windows is welcome to
  pursue it on their own clone. Documentation only: **`runner serve`'s typed
  Windows refusal (`runnerPlatformUnsupported`) is unchanged**, because
  promises change and behaviour does not.

- **The loader's co-residency refusal says who can still reach it.** With the
  judge-column release seam in place, a sequential panel never arrives at
  `_assert_gpu_capacity`'s "another model is already resident" refusal, so
  advice that pointed only at manual unloading would misdescribe who is
  standing there. The message now names the callers that genuinely need two
  models AT ONCE — a judgeScore sweep panel (every judge's model is held for
  the whole grid, because judging is interleaved with the selection), an
  interactive/API load beside a resident model, or another process on the
  device — and keeps the unload/use-the-study-model/pin-an-external-judge
  remedies for exactly those. The refusal's typed shape, its
  `advice_complete` marking, and the margin (`_GPU_LOAD_HEADROOM_BYTES`,
  reviewed and left alone) are unchanged.

- **The residual-norm denominator's two averaging rules stopped sharing one
  stamp.** `residualNormConvention: "wholeCorpusMean-v1"` was defined by the
  shared convention module as the **per-position** mean over every measured
  corpus position — and was also written by every vector-sidecar writer, which
  measures something else: `extract`, `extract_grand_mean` and `vectors
  backfill-norms` (both engines) pool through the activations path, where each
  capture already holds the mean norm over *its own* reading window and those
  per-text numbers are then averaged with equal weight per text. Two rules,
  one string, in direct violation of that string's own "bump the version when
  the averaging RULE changes" contract. The Mac's `extractGrandMean` emitted
  both from one function — the token bank's per-position tally for a
  `neutral-token-bank` denominator, the per-text average otherwise — under the
  single stamp. The second rule now has its own: **`perTextMean-v1`**, defined
  beside `wholeCorpusMean-v1` with the same rigor, and every writer stamps the
  rule it applied (extraction, backfill and the pooled grand-mean branches →
  `perTextMean-v1`; the tally → `wholeCorpusMean-v1`). The rules coincide at
  any single-position reading (`last token` and friends) or where every window
  is the same length, and diverge at a pooled reading (`mean from token k`,
  `mean content from token n`) over variable-length texts — the case no test
  discriminated, now pinned on both engines by a fixture whose two texts are
  read over 2 and 6 positions and whose two rules give 6.0 and 4.0.
  **Nothing on disk is rewritten and nothing refuses.** A `wholeCorpusMean-v1`
  stamp on an existing sidecar was written by a per-text writer (no tally
  number has ever reached a sidecar on either engine), so it is grandfathered
  as exactly that, documented in the convention module and in METHODS; and
  because the string is read only by display and provenance surfaces —
  `display_label`, the α-default convention note, the OptVec packaging
  advisory — never by a gate, an old-stamp and a new-stamp artifact cannot
  refuse against each other. What changed is that a reader of a
  pooled-reading denominator can now tell which weighting produced it.

- **The neutral token-bank draw no longer leans on `random.sample`.** The
  server chose which token positions to bank with `random.sample` seeded from
  `int(corpusHash[:16], 16) % 2**63`. CPython guarantees reproducibility
  across versions only for `Random.random()` — `sample`'s algorithm is
  explicitly allowed to change — so an interpreter upgrade could have silently
  rebuilt a neutral-PC basis from different positions while every stamp said
  the recipe was identical, and a basis is not the kind of artifact whose
  bytes anyone notices drifting. The draw is now the Mac's, exactly:
  SHA-256-derived seed (first 8 bytes, big-endian), SplitMix64, seeded partial
  Fisher–Yates, in a new `steering/token_bank_downsampling.py` that is the
  line-for-line twin of `TokenBankDownsampling.swift` — **so the two engines
  select byte-identical rows for the same corpus and cap**, where before the
  seed, the generator and the cap all differed. Twin literals pin the seed,
  the RNG stream and the selected indices on both sides, so no future runtime
  can move the draw undetected, and the contract is written down in
  PORTABILITY-CONTRACTS §12, which said nothing about the bank before. The
  default per-layer CAP deliberately did not converge (4096 on the Mac behind
  a memory preflight, 2048 on the server, which has none): a cap is a memory
  bound rather than a rule, and doubling an unguarded transient on the engine
  that runs on shared cluster GPUs buys nothing. Pass the same `maxTokenRows`
  on both engines for a bit-identical bank. **No pinned artifact moved** — a
  basis travels by `neutralPCBasisHash`, the SHA-256 of its own bytes, so
  every pinned basis keeps verifying; the draw only decides what a *new* basis
  is built from, and a rebuild of the same corpus now banks different
  positions than it did before this release. One edge case converged with the
  algorithm: a cap of `0` or less now selects nothing, where the server read
  it as "uncapped" and would have banked the whole corpus.

- **Swift's element-wise mean accumulates in `Double`.**
  `SteeringVectorMath.mean` — and with it `meanDifference`,
  `grandMeanDifference` and the PCA centring that calls through it — summed
  float32 rows sequentially, an error of order N·ε against numpy's pairwise
  ~log₂(N)·ε. That difference, not any recipe divergence, was the ceiling on
  cross-engine agreement for a re-derived grand mean (~1e-4..1e-3 relative in
  unlucky cases). Rows stay float32 (the deliberate cross-engine storage
  dtype) and the result stays `[Float]`; only the accumulator is `Double`,
  cast once at the end, which moves this engine's value *toward* the exact
  mean and therefore toward numpy's. `ConceptExtractor.neutralMeanPerLayer`
  has always accumulated this way. Values may move in the last float32 ulp.

- **Concept screening splits are drawn from the stimulus CONTENT, and the
  reported numbers move.** *Read this before comparing a concept's held-out
  accuracy or split-half cosine against a number recorded before this
  release: they are not the same statistic.* The two screening diagnostics
  used to be split positionally, and the two engines did not even split the
  same way — the server held out the last ~20% of each class **in file order**
  and halved on even/odd rows, while the Mac held out `index % 5 == 4`. So the
  number depended on how the stimulus file happened to be ordered (a file
  authored in topic runs handed its last topic block over whole as the
  "held-out" set — pessimistic; a parity split put adjacent near-duplicates on
  both sides of the halves — optimistic), it showed no variance because the
  split was deterministic, and the same data scored differently on the two
  engines. Both now sort each class's rows ascending by the lowercase SHA-256
  hex of the row's UTF-8 text and take every 5th of that order as the test set
  (halves on its parity) — the rule the reading-probe validation split has
  used since 2026-07-13, extended to these. Content-derived means shuffling
  the file changes nothing, both engines select the same rows byte for byte,
  and no RNG is involved to diverge. Sorted-order rather than `hash % 5` keeps
  the exact ~20% / 50% proportions the positional rules had, so `testCount`
  does not wander between concepts. The floor is now one rule as well — six
  stimuli per class for held-out, four for split-half — where the Mac
  previously reported a held-out "accuracy" over a one-row test set. Scope is
  screening only: `scenario_accuracy` over `validation.jsonl`, the actual
  circularity firewall, is untouched, no promote gate reads these numbers, and
  nothing frozen embeds them. Pinned by a committed cross-engine fixture whose
  topic-blocked case appears twice, in two different row orders, selecting the
  same rows both times (2026-08-28 audit, F3).

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

- **A declared evaluation-sampling design can no longer be frozen into a
  study that could never execute it.** The `evaluationSampling` declaration
  applies to the per-response coding instrument only — the paired-rubric
  evaluation path refuses every sampling request unconditionally, because a
  per-condition record count does not name a set of (baseline, variant)
  pairs. Verification, however, checked only the declaration's own shape and
  never the pinned rubric's MODE, so a study carrying both a declaration and
  a paired rubric verified, froze — permanently, and a frozen declaration can
  never be cleared — and then refused at evaluate every single time it ran.
  Both engines now report a typed violation at verify/freeze naming both
  facts and both repairs, and `experiment set-evaluation-sampling` refuses
  with the same sentence when the study ALREADY pins an incompatible rubric.
  The mode is read by the evaluate path's own predicate (`parse_rubric` /
  `ResponseCoding.parseRubric`), never by a second reading that could drift
  from it. Legacy tolerance is deliberate: a declaration with no rubric
  pinned yet stays clean (the declaration may precede the rubric), a
  per-response rubric plus a declaration stays clean, and a manifest with no
  declaration — every frozen study written before this — gains nothing.
  Pair-level sampling is explicitly NOT implemented: defining a seeded draw
  over pairs is a design decision nobody has taken, and the gate is the fix.

- **The Local Engine Details pane's `site qualify` no longer reports a
  profile nobody is running.** The qualify subprocess inherited the app's
  working directory (`/` when launched from Finder) and no `STEERLAB_ROOT`,
  so its serverProfile check derived the artifact root from `getcwd()` and
  failed on `metadataRoot: /.steerlab does not exist`. The argv now pins
  `--root` to the same workspace root the serve step passes. Engine-side,
  env-supplied profile paths (`STEERLAB_ROOT`, `STEERLAB_METADATA_ROOT`, and
  the optional asset/run/archive/node-cache roots) expand `~` and `$VAR` at
  the one place they become profile fields — launchers that never ran a
  shell (managed processes, launchd, cron) pass them verbatim, and the
  literal spelling was probed, reported, and created as-is.

- **The MPS cache trim could kill the process — a crash, not a warning.**
  `free_device_memory` guarded its `torch.mps.empty_cache()` on the MODULE
  existing (`hasattr(torch, "mps")`) and wrapped the call in a `try`. Neither
  guard is the right question. On a torch build whose `torch.mps` module is
  present while `torch.backends.mps.is_available()` is False, the call reaches
  a backend that was never initialized and takes the whole process down with
  SIGSEGV (exit 139) — there is no Python exception, so the `except` never
  runs and every model release, every registry eviction, and every judge
  column boundary is a place the server can simply die. Reproduced by an
  external reviewer on a Mac sandbox with MPS unavailable; this project's own
  suites never saw it because the machines they run on have MPS available.
  Availability is now asked FIRST (itself `hasattr`-guarded for a torch old
  enough to predate the backend) and gates the call, so on an unavailable
  backend the trim does not HAPPEN rather than being caught. The CUDA trim and
  the `gc.collect()` are unchanged.

- **A judged evaluate on a one-slot server can finally judge sequentially.**
  The column release seam landed so a panel needing its models one at a time
  would pay the MAX of their weights instead of the sum — but `evaluate`'s
  capacity guard still refused any foreign local judge whenever
  `STEERLAB_MAX_LOADED_MODELS < 2`, decided per judge inside the resolution
  loop, BEFORE the seam it was supposed to make room for could ever run. The
  capability was unreachable on exactly the registry it was built for, and the
  integration test missed it by passing `max_loaded=2` into `evaluate` while
  its fake residency held one slot. The guard now runs ONCE over the whole
  panel and asks `judge_slots_required` for the run's largest single moment —
  columns loaded so far intersected with columns still needed, plus the study
  model when a later stage still generates — so a five-judge panel of five
  distinct models runs in one slot, while a panel that genuinely wants a model
  back after a second one has loaded still refuses, saying which moment costs
  two. A caller with no release seam is priced at the whole panel and the
  refusal says so. The sweep's rule is deliberately untouched:
  `_judge_preflight` requires SIMULTANEOUS residency because a judgeScore
  sweep interleaves judging with selection and holds every judge's model for
  the whole grid. Two instruments, two honest arithmetics. The integration
  tests now drive the production guard with one capacity number.

- **An evicted model is released by every owner, not just the registry.**
  `ModelRegistry._evict` dropped the slot's reference and trimmed the
  allocator, but `ServiceState.model` — the model `/api/load` put there —
  still pointed at the same container, so the weights stayed alive and the
  trim reclaimed nothing. A release that reports GiB freed and frees none is
  worse than no release. The registry now carries an eviction listener called
  after the slot is unregistered and BEFORE the trim; the service clears its
  own reference by identity, so only the exact container that was released is
  forgotten and an unrelated eviction never blanks the active model. Explicit
  clearing, matching how the registry hands its own reference back — not a
  weakref.

- **Model release speaks identities, not slugs.** `judge_models_still_needed`
  and `ModelRegistry.release_models` named bare `modelID`s while judges pin
  `(model, revision, dtype)` — and `--judge-pin` makes a same-slug,
  two-revision panel expressible. Slug vocabulary read those two containers as
  one model: the finished OLD-revision container matched the next column's
  still-needed name, nothing was released, and it survived as dead weight
  beside the one that had just loaded — recreating the co-residency OOM the
  seam exists to prevent. `(modelID, revision, canonical dtype)` triples now
  flow through the still-needed computation, the release call, and the release
  log line, which prints the revision prefix it released. The registry keys
  containers by `(modelID, revision, requested dtype, device)`, so dtype is
  genuinely part of the container key and belongs in the identity; `device` is
  the registry's own choice, not something a judge pins, and rides only in the
  release record. A bare slug is refused rather than silently widened into
  "every revision of this model".

- **`set-system-prompt` stops naming a delivery route the run does not
  take.** Both engines derived the echoed `delivery` from the model family
  alone, so a `rawCompletion` study on a system-role family reported
  `systemTurn` — while the renderer, correctly, prepends the frame to the raw
  prompt, because rawCompletion renders no chat template and no family has a
  system turn to offer there. Execution was always right; only the echo lied.
  `promptMode` is now asked first and a rawCompletion study reports the third
  honest value, `promptPrepend`, spelled identically on both engines, with the
  human line and `docs/CLI-REFERENCE.md`'s enumeration saying the same thing.
  `chatAssistant` keeps the family split unchanged.

- **A Gemma Scope report row must NAME the feature it is matched against.**
  The requested feature id got the exact-integer predicate in the previous
  round, but the row side of the match still ran through `int()` — so a report
  row whose `feature` is `7.5` truncated to 7 and satisfied a request for
  feature 7, and `true` became 1. The decoder values of a different dictionary
  entry would then have been imported under the requested feature's name.
  A row whose feature fails the same predicate is no longer a match; it is
  skipped, and when no valid row matches, the existing not-found refusal fires
  and nothing is written. `7.0` still names seven — the predicate refuses
  fractions and booleans, not JSON's habit of writing whole numbers as floats.

- **A sharded submission's walltime prices the largest shard AND its startup.**
  `_check_walltime` divided the record estimate by K exactly and stopped
  there. Two things were missing, both under-prices. The largest shard runs
  `ceil(records ÷ K)` — an uneven split hands someone the extra record and the
  wall has to fit THAT job — and every child pays a fixed startup, the model
  load above all, whatever slice of the matrix it draws: exact division priced
  a 1-record shard of a `--parallel 4` submission at essentially nothing, when
  it is a model load with one record after it. The per-shard estimate is now
  `ceil(records ÷ K) ÷ rate × margin + PREFLIGHT_JOB_STARTUP_HOURS`, and the
  basis string shows both terms. The startup constant is 15 minutes, rounded
  UP from this repository's own recorded cold-load observations (the live 10+
  minute load off `/work` that made `/api/load/stream` an SSE route, and the
  ~12 MB/s mmap-fault measurement behind node-local staging, both 2026-07-17)
  — a preflight that under-states startup approves a wall the job cannot
  finish in, and a warm cache simply comes in under the estimate. Unsharded
  jobs pay it too: one job is still a job, and it loads the model before its
  first record.

- **A judge-load failure names its real cause.** The local-judge load wrapper
  replaced every loader error with "install the model on the server" — good
  advice for the raw hub dump it was written against ("check your internet
  connection" is misleading on an air-gapped node), and precisely wrong for
  the engine's own capacity refusal, whose text already named the right
  remedies. The observed cost: a co-residency refusal (another model
  resident, headroom short) surfaced as an install instruction for a model
  that was already installed. The loader's typed refusals now carry an
  `advice_complete` marker and the wrapper carries their prose through
  verbatim, adding only the judge's name; raw hub dumps stay summarized.
  The Swift twin is deliberately unchanged: it decides on a presence check
  before the loader is asked, so "not installed" is the one possible cause
  there.

- **PCA deflation stops at the data's EFFECTIVE rank, not its theoretical
  one.** The rank cap added earlier in this cycle bounded the component count
  by `rows − 1`, which is what the SHAPE could hold, not what the cloud
  contains — so mathematically rank-one, non-axis-aligned rows asked for three
  components returned three, with explained-variance shares `[1.0, 1.9e-15,
  1.5e-16]`: normalized float32 residue handed back as if it were a direction,
  the exact failure the cap was written to prevent, arriving through the other
  door. `extract()` then projected the concept vector's component along those
  arbitrary directions OUT of the science vector. The `min_variance` path had
  its own version — a target of 1.0 against a first share of 0.9999999 kept
  deflating in pursuit of a gap that is itself round-off. Both engines now cap
  at `min(count, rows − 1, columns)` (the column count is a real bound the old
  cap omitted) and stop deflating once the residual trace falls below a
  scale-relative floor of the ORIGINAL total variance:
  `deflationResidualTraceFloorEpsFactor · (rows + columns) · eps²`, a named
  twin constant whose factor of 8 was calibrated against 120 measured clouds —
  round-off residues topped out at 0.0093 of that unit, genuine ones bottomed
  at 2.2e+03, and 8 sits between. Components computed before the stop are
  bit-identical: the cross-engine parity fixture passes unmodified, and 60
  random full-rank clouds compare exactly against the old loop (review round
  11, F2).

- **`ComputeChoiceAccessory` states its main-actor isolation.** The type
  builds `NSTextField`/`NSSegmentedControl`/Auto Layout constraints and took
  the segmented control's callback through a `nonisolated` relay, which under
  the strict toolchain produced a 29-warning isolation cascade in one 83-line
  file. `@MainActor` on the type and on the relay records a fact rather than
  imposing a rule — the panel is assembled on the main thread and AppKit
  delivers control actions there — and the single caller (`WorkspaceSelector`,
  a SwiftUI `View`) was already isolated, so no hop was needed anywhere.
  Twenty-nine warnings gone, none added (review round 11, F6).

- **`sweep: null` clears like every other protocol field.** `sweep` was the
  one key in the vocabulary whose shape gate spelled `"sweep" in fields`
  instead of `fields.get("sweep") is not None`, so an explicit null was
  refused — *sweep must be an object, got NoneType* — before it could reach
  the null-clears loop below it, whose whole promise is that every field in
  this vocabulary clears that way. A declared grid was therefore removable
  only by hand-editing the manifest, which is the one repair this store
  exists to make unnecessary. The non-dict, non-null refusal is unchanged: a
  string still cannot be a sweep block, and a refusal still writes nothing
  (review round 11, F4).

- **A Gemma Scope layer must be a real integer.** Both the route
  (`POST /api/gemmascope/run`, and its `import-id` sibling next door) and the
  importer took `int(body["layer"])`, so `2.5` truncated to `2` and `true`
  became `1` — and a truncated layer can then PASS the SAE-agreement check
  against the WRONG layer, which is a silent scientific error, not a typing
  nit. One predicate now guards every entry point (`gemma_scope.coerce_layer`):
  a JSON integer, or a finite integral float (`2.0` is layer 2 — JSON has one
  number type), with `bool` excluded explicitly; anything else refuses under
  the existing *layer must be an integer* contract, at the route's own 400 and
  in the importer's own `ValueError` style, before anything is written. This
  is also cross-engine parity: the Swift decoder (`GemmaScopeReportVector`)
  has always refused fractionals, and the Python engine was the one out of
  step (review round 11, F5). The **feature** field had the identical shape
  one column over — `2.5` truncated to *feature 2* and imported the wrong
  dictionary entry outright, and the report importer's row match coerced both
  sides so even a string `"7"` selected a row — so the same predicate now
  guards every feature entry point too: both import routes, the report
  importer, and both by-id importers (found by the F5 fix's own agent while
  in the code, verified and closed in the same landing).

- **A `layerCount` that no integer can hold refuses instead of crashing.**
  Both engines had a representability hole in the pole-mirror sidecar read.
  Swift's gate was `value <= Double(Int.max)`, and `Double(Int.max)` rounds
  UP to 2^63 — so exactly 2^63 was admitted and the following `Int(_:)`
  TRAPPED, a crash rather than a refusal on a hostile or corrupt sidecar; the
  gate is now `Int(exactly:)` and the trapping conversion is gone, not merely
  unreachable. Python's guard called `float()` on the raw value, so a
  400-digit integer raised a bare `OverflowError` past every typed refusal;
  integers are now range-checked as integers against a named bound and floats
  are handled on their own branch, so nothing converts before it is
  validated. The twin refusal sentence and its repair are unchanged on both
  engines, and every shape that refused before still refuses (review round 11,
  F3).

- **An explicit JSON null CLEARS a protocol field instead of bricking the
  manifest.** The client's `set-protocol` gates every sampling field with
  `is not None`, so a null passed them all ungated — and the persistence loop
  then WROTE it. `Manifest.from_dict` raises `TypeError` on a null
  `temperature`, so every later verb died before it could name the problem,
  `verify` included: one client call left a draft that no verb could read.
  The loop now pops the key, which is what the site's own comment always
  claimed ("a JSON null clears like an absent key on decode") and what makes
  the gates above it sound. It is also the symmetric affordance — the Swift
  writers clear with `""`, the client clears with `--set temperature=null`,
  and both land on a key that is simply absent (review round 10, F1).

- **A local sweep judge cannot pin a dtype the study never loads.** `--judge-pin
  <name>=<revision>[:<dtype>]` made a judge's precision declarable, but the
  one-model-slot rule compared only model and revision. A same-model judge
  pinned to a different dtype judged through the container the sweep already
  held — at the STUDY's precision — while `resolvedJudges` carried its own
  dtype into the provenance: the stamp-says-what-never-ran shape the revision
  check was added to close, one field over. The rule now refuses at sweep
  start, names both dtypes, and offers the two repairs (drop the judge's dtype,
  or use a claude judge). Comparison is canonical, so `bf16` and `bfloat16` are
  one dtype; a judge that declares NO dtype inherits the study's and never
  trips it (review round 10, F2).

- **The Mac's HTTP protocol route gates `temperature` and `maxTokens`.** The
  route checked `samplesPerItem` and `seedPolicy` and assigned their two
  neighbours unchecked, so a negative temperature or a zero `maxTokens`
  reached the panel fields while the note `saveProtocol` would have raised
  went nowhere — the route answering ok over a declaration the store refuses.
  All four now share one decision, in the store setter's own sentences, so the
  route and `experiment set-sampling` say the same thing to the same body
  (review round 10, F3).

- **The submit-time engine-lag advisory can see a development checkout.** The
  advisory read the payload's identity from `deployment-manifest.json` alone
  — and in developer mode the payload root IS the checkout, which has no
  manifest. The identity came back nil and the advisory never fired: it was
  structurally silent for exactly the payload shape of the 2026-08-27
  stale-engine incident it was written for. It now falls back to the dev
  checkout's git stamp through the same `devPayloadStamp` the status path
  uses — no second mechanism — and unknown still stays nil, because an
  advisory is never invented (review round 10, F4).

- **A value flag with no value is a malformed invocation, not a silent
  default.** The shared preprocessor kept a declared value flag that arrived
  without one, on the assumption that the verb would refuse it. The verbs
  that read flags through the strict reader do; the ones that read through
  the tolerant helper see nil and fall back to a DEFAULT — `experiment
  set-sampling <name> --temperature` wrote the sampling protocol at defaults
  and reported success. Both shapes now refuse at 64 before any verb runs:
  a value flag at end-of-args, and one whose next token is another of this
  verb's declared flags (`--temperature --json`, which ate the `--json`). The
  refusal names the metavar and carries the verb's synopsis as the repair.
  An explicit empty token is still a VALUE and still reaches the verb, because
  several verbs use `""` as their clear affordance; so is a flag-shaped token
  that is not one of this verb's flags (review round 10, F5).

- **Clearing the matched-norm control and describing it in one breath is
  refused.** `--control-margin ""` REMOVES the control block, and the clear
  fired before anything else was looked at — so `--control-margin ""
  --control-top-k 3` removed the control and discarded the width without a
  word, the flag-that-exits-0-having-done-nothing class this verb refuses
  everywhere else. Naming a sibling control flag alongside the clear now
  refuses, in the style of the winner-scope-plus-width contradiction beside
  it, and for the same reason: either half could be the one meant. A clear
  alone still clears (review round 10, F6).

- **Already-relative artifact references are normalized instead of passed
  through.** `workspaceRelative` relativized absolute paths and returned
  anything else verbatim, so `prompts/x`, `./prompts/x` and `prompts/a/../x`
  named one file and compared as three — and a re-declaration that spelled the
  path the other way read as a different file, which can drop the hash pin
  standing beside it. Relative references are now normalized LEXICALLY:
  leading `./` stripped, empty segments collapsed, `..` resolved against the
  components to its left, and no filesystem touched. A reference that escapes
  the root (`../outside`) is returned verbatim, for the same reason an
  outside-the-workspace absolute path is (review round 10, F7).

- **A commit that cannot finish takes its destination back out.**
  `_commit_no_replace` lands the destination and THEN drops the staging name;
  a failure in that last step propagated with the destination in place, and
  the callers' cleanup owns the temporaries, not the destination — a
  half-final artifact wearing the final name, which is the state
  `destinationOccupied` then refuses to repair. In `experiment.bundles` it was
  sharper still: the failure happens inside `_commit_one`, before the member
  reaches `landed`, so `_rollback` never knew the file was there. The
  staged-removal is now guarded in all three mirrors of the primitive
  (`steering.pole_mirror`, `experiment.bundles`, `client.runner` — the
  docstrings' own rule that a change to any belongs in all), and the pole
  mirror's caller keeps its both-or-neither: a promotion whose sidecar commit
  fails takes back the vectors file it already landed. The Swift twin commits
  with `moveItem`, an atomic rename with no separate cleanup step, so it has
  no such window (review round 10, F8).

- **A sidecar's `layerCount` must be a whole number of layers.** Both engines
  checked that the value was a NUMBER and then converted it: `2.5` truncated
  and stamped a mirror claiming a depth its source never had, `0` and `-3`
  stamped an impossible one, and a non-finite value reached the conversion —
  a bare `ValueError`/`OverflowError` past every typed refusal on the server,
  and a TRAP on the Mac. Finiteness, integrality and `≥ 1` are now checked
  before any conversion, on both engines, with the offending value named in
  the refusal. No upper bound was invented: no other sidecar reader on either
  engine bounds this key above (review round 10, F9).

- **`set-instrument-scope ""` clears a stale scope even when the prompts pin
  is broken.** The task-prompts guard and the file read ran before the
  empty-formats branch, so clearing refused in exactly the states that make
  clearing necessary: no pin at all, a pin whose file has moved, a pin that
  drifted. A stale scope was then unremovable except by hand-editing the
  manifest — the one repair this store exists to make unnecessary. Clearing
  derives nothing from the prompts and now runs first; DECLARING a non-empty
  scope still requires the pin and still reads it, because a scope is a
  selection over those rows (review round 10, F10).

- **A template-pair reader's score is a relative endpoint, and the docstring
  no longer claims otherwise.** `score_texts` / `scoreTexts` documented T+ as
  "the rendering the probe's center and scale were calibrated on". For an
  `unsupervisedTemplatePair` reader that is affirmatively wrong: the probe is
  fitted over BOTH renderings of the same stimuli, so its `projectionCenter`
  is the MIDPOINT of the T+ and T− train projections, while inference renders
  new text under T+ alone. Every such score therefore carries a systematic
  positive offset of about `|posMean − negMean| / (2·projectionScale)`, and
  `score > 0` is not concept presence — neutral T+-rendered text scores
  positive by construction. The prose now says so on both engines, at
  `classifiesPositive` / `classifies_positive` (valid only where the scored
  activation comes from the distribution the center was fitted on), in the
  RepE brief §1, and in the instrument table in CONDUCTING-A-STUDY. **No
  number changed**: the `repeReaderScore` instrument already reports a
  continuous endpoint compared across conditions, where the constant offset
  cancels, and the artifact's own train/held-out accuracies are computed over
  both renderings where the midpoint IS the right threshold. Supervised-content
  readers are unaffected in every respect. A new test pins the arithmetic — a
  T+ class-mean activation scores at exactly the documented offset — so the
  corrected prose cannot quietly drift back (2026-08-28 audit, F4).

- **The PC1 power iteration stamps whether it converged.** ≤200 float32
  iterations against a max-abs-delta tolerance, with no residual check and no
  convergence flag, meant a near-degenerate spectrum returned a wrong PC1
  silently: the audit reproduced |cos| = 0.148 against the TRUE second
  eigenvector at an eigenvalue ratio of 0.9945, deterministically and with no
  warning, and nothing in the artifact could say the direction was
  ill-defined (the explained variance looks perfectly normal — a wrong
  direction inside a near-tied 2-plane explains almost as much as the right
  one). Both engines now compute the relative Rayleigh residual
  ‖Gw − λw‖/λ after the iteration, return it beside the component, warn once
  above 1e-4, and stamp it into the reader artifact as `pc1PowerIteration`
  (`{converged, illConditioned, iterations, maxIterations, relativeResidual}`)
  next to the `pc1ExplainedVariance*` fields. The threshold is calibrated
  against the audit's own geometry and documented with its table at the
  constant: a 5% sample eigengap sits at ~2e-6 and does not warn — even though
  it legitimately uses all 200 iterations without meeting a 1e-7 float32 delta,
  which is why the residual and not `converged` is what gets thresholded —
  while the audit's failure sits at ~6e-3. **Warn and stamp, never refuse**:
  the result is deterministic and mirrored across engines, so a near-tied
  spectrum is a fact about the data to record, not malformed input to reject.
  Additive throughout — **the component is bit-identical** (the committed
  cross-engine PCA fixture passes unmodified), the stamp is absent on every
  reader written earlier, and no `recipeIdentityHash` can move, since recipe
  identity reads a closed list of sidecar keys and no reader field is in it
  (2026-08-28 audit, F5).

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
