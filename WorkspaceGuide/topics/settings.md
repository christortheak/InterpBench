# Study settings on a draft

<!-- client: all -->

Every verb here writes a **draft**; a frozen study refuses, so duplicate it
first. Each takes `{{cli}} experiment <verb> …`, and `--help` on any of them
prints its exact flags.

**`list`** — every experiment here with its status, model, concepts, condition
count and `freezeHash`. The cheapest orientation command; run it first.

**`verify <name>`** — re-hashes every pinned input against the manifest and
refuses (`pinDrift`, one line per drifted pin) if a byte moved. What the other
verbs run for you, available alone, safe at any status.

<!-- client: mac -->

**`set-sampling <name> [--temperature <t>] [--max-tokens <n>]
[--prompt-mode chatAssistant|rawCompletion] [--samples-per-item <n>]
[--seed-policy manifestSeeds|derivedSHA256] [--seeds <n>[,<n>…]]
[--reasoning-effort off|on|low|medium|high|xhigh] [--reasoning-max-tokens <n>]`** —
declares the generation protocol on a draft, merge-style: only the flags given
move, `""` clears `--prompt-mode`/`--seed-policy`, and `--samples-per-item 1`
clears to the default of one. The reasoning protocol replaced the
Qwen-specific `qwenThinkingEnabled` boolean: a non-off effort renders the chat
template in thinking mode and REQUIRES `--reasoning-max-tokens` (the reasoning
block's own cap, counted up to `</think>`; `--max-tokens` is then the answer
budget); `on` is thinking at the template's default effort, a LEVEL is refused
unless the model's chat-template capability record (`prompts/models/`, probed
from the pinned template — `steerlab-cli model capabilities <modelID>`) says
the template accepts it (Qwen3-14B/-32B ignore the level: declare `on`), and
`--reasoning-effort off` retires the budget. A manifest frozen under the old
boolean still reads (`true` meant the template's default effort, xhigh) and is
never rewritten. A stochastic replication arm is `--samples-per-item 25
--temperature 0.7 --max-tokens 1024 --seed-policy derivedSHA256` — legal to
declare and run on either engine with scoped per-record seeds (`workspace guide engines`). The joint rules (`samplesPerItem > 1` needs `temperature > 0` and
seedPolicy `derivedSHA256`) surface at `verify`, not here, so the fields can
be declared one flag at a time. They apply where the run reads the count, a
study that generates sampled text (or a panel): a study of answer-token
readouts alone reads each item once, and is told its `samplesPerItem` is
inert instead. Above temperature 0, `derivedSHA256` runs each
item `samplesPerItem` times, including once, with a seed per record; the fixed
list (`manifestSeeds`, the default) runs each item once per listed seed.
`--seeds` replaces that list: whole numbers, order kept, each seed once. At
temperature 0 every seed gives the same response, so a study that generates
sampled text lists one seed: `verify` refuses a draft with more (so `freeze`
does), with `--seeds <n>` as the repair. A study frozen with more still
verifies, so its runs stay readable, and says so with a `refusedWhenRun`
advisory; this command line refuses to submit it, and the engine refuses it
at submission and at run start (duplicate it first). Answer-token readouts
read each item once, so a readout-only study is never refused for its list.
A readout-only study with several seeds or samples per item runs on an engine
from this release; one from before it refused such a study after the model
load, so this command line refuses to send it to an engine whose capabilities
do not list `remoteStudy.greedyRulesSampledTextOnly` (code
`missingPrerequisite`; update the engine, or set one seed and one sample per
item). A
`seedSettingNotUsed` advisory, here, at `set-instruments`, and at `verify`,
says when a declared seed setting will not be read: `derivedSHA256` at
temperature 0 with one seed (greedy, no seed), `derivedSHA256` above
temperature 0 beside a seed list of more than one entry, or a readout-only
study's list of more than one seed.

**`set-exclusions <name> <rule>[,…] [--endpoint <key>] [--min <x>]
[--max <x>]`** — declares the record-exclusion rules analysis applies
(`failedAttentionCheck`, `unparseableEndpoint`, `outOfRange`) on a draft;
`""` clears the declaration. `--min`/`--max` bound the `outOfRange`
keep-window and `--endpoint` names the parsed-value key the endpoint rules
read (default `parsedMonths`). Exclusions apply at analysis time only and
are stamped honestly — records never leave `generations.jsonl`.

<!-- client: python -->

**`set-protocol <name> --set <key>=<json> [--set <key>=<json>]…`** — sets
declared protocol fields on a draft, merge-style: only the keys given move. A
JSON `null` clears a field, a bare word is read as a string, and a key outside
the vocabulary refuses before anything is written, listing the vocabulary. The
fields: `experimentDescription`, `taskDescription`, `outcomeMeasures`,
`promptMode`, `systemPrompt`, `reasoningEffort`, `temperature`, `maxTokens`,
`reasoningMaxTokens`, `seeds`, `samplesPerItem`, `seedPolicy`,
`taskPromptsFile`, `taskPromptsHash`, `studyKind`, `studyType`,
`multiAgentScenarioPath`, `multiAgentScenarioHash`,
`multiAgentIncludeBaseline`, `evaluation`, `judgeRubricFile`,
`judgeRubricHash`, `judges`, `humanValidation`, `capabilityBatteryFile`,
`capabilityBatteryHash`, `reasoningStyleTaxonomyPath`,
`reasoningStyleTaxonomyHash`, `exclusionRules`, `outcomeInstruments`, and
`sweep`. Prefer a dedicated verb, a study pack, or `import-prompts` wherever
one exists: they compute hashes from real bytes, and a hash typed by hand is a
claim nothing checked. `--set studyType=agentComparison` on a study with
nothing attached declares a study of the baseline alone (`workspace guide
lifecycle`).

The generation protocol is set here. `promptMode` is `chatAssistant` or
`rawCompletion`; `seedPolicy` is `manifestSeeds` or `derivedSHA256`. A non-off
`reasoningEffort` renders the chat template in thinking mode and REQUIRES
`reasoningMaxTokens` (the reasoning block's own cap, counted up to `</think>`;
`maxTokens` is then the answer budget); `on` is thinking at the template's
default effort, a LEVEL is refused unless the model's chat-template capability
record (`prompts/models/` — `steerlab model capabilities <modelID>`) says the
template accepts it, and `reasoningEffort=off` retires the budget. A
stochastic replication arm is `--set samplesPerItem=25 --set temperature=0.7
--set maxTokens=1024 --set seedPolicy=derivedSHA256` — legal to declare and
run with scoped per-record seeds (`workspace guide engines`). The joint rules
(`samplesPerItem > 1` needs `temperature > 0` and seedPolicy `derivedSHA256`)
surface at `verify`, not here, so the fields can be declared one at a time.
They apply where the run reads the count, a study that generates sampled text
(or a panel): a study of answer-token readouts alone reads each item once, and
is told its `samplesPerItem` is inert instead.
Above temperature 0, `derivedSHA256` runs each item `samplesPerItem` times,
including once, with a seed per record; the fixed list (`manifestSeeds`, the
default) runs each item once per listed seed. `--set seeds='[<n>, …]'`
replaces that list: whole numbers, order kept, each seed once. At temperature
0 every seed gives the same response, so a study that generates sampled text
lists one seed: `verify` refuses a draft with more (so `freeze` does), with
`--set seeds='[<n>]'` as the repair. A study frozen with more still verifies,
so its runs stay readable, and says so with a `refusedWhenRun` advisory; `run`
refuses to send it, and a runner refuses it before anything is queued
(duplicate it first). Answer-token readouts read each item once, so a
readout-only study is never refused for its list. A readout-only study with
several seeds or samples per item runs on an engine from this release; one
from before it refused such a study after the model load, so `run` refuses to
send it to an engine whose capabilities do not list
`remoteStudy.greedyRulesSampledTextOnly` (code `greedyRulesUnsupported`;
update the engine, or set one seed and one sample per item). A
`seedSettingNotUsed`
advisory, here and at `verify`, says when a declared seed setting will not be
read: `derivedSHA256` at temperature 0 with one seed (greedy, no seed),
`derivedSHA256` above temperature 0 beside a seed list of more than one entry,
or a readout-only study's list of more than one seed.

Record-exclusion rules are the `exclusionRules` field (`failedAttentionCheck`,
`unparseableEndpoint`, `outOfRange`); a malformed value refuses and names the
accepted shape. Exclusions apply at analysis time only and are stamped
honestly — records never leave `generations.jsonl`.

**`pin-revision <name> <revision>`** — pins the model commit a draft resolves
to. It must be a commit hash: a branch or tag is re-pointed by definition, so
it cannot identify the weights a run used.

**`pin-sae-candidates <name> <path>`** — pins an SAE candidate roster into a
draft study (`workspace guide methods` covers the roster checks).

<!-- client: all -->

**What an agent keeps and what the study sets.** How hard an agent thinks
belongs to the agent: its file's `reasoningEffort` (set with `agent create
--reasoning-effort`, in the app's agent editor, or by promotion from a sweep, which
gives the agent the level the sweep's cells ran at, read from the sweep run's
own `experiment.json`) is the level it runs at, in an agent arm of an
ordinary study and in a panel seat alike. An
agent file that spells only the older `qwenThinkingEnabled` reads as before
(true is the template's default effort, false is off); a file whose two fields
disagree is refused. The study keeps the reasoning budget
(`reasoningMaxTokens`), which applies to every arm that thinks and to no arm
that does not, and the temperature, which applies to every arm: an agent
file's `temperature` is a Playground setting, stamped on records as
`agentPlaygroundTemperature` and never used for a measured run. A study whose
own effort is off may still declare a reasoning budget when an attached agent
arm, or an agent cast in one of its panel's seats, thinks. Each response of an
agent arm stamps `agentReasoningEffort`, and each panel turn its
`reasoningEffort`: the level that reached the chat template (a level the
template ignores is stamped `on`). `verify` names each attached agent arm that
breaks a rule, so freeze refuses it: a level the template does not accept, a
file whose fields disagree, more than one adapter, an agent that thinks beside
an answer-token instrument, a commit other than the study's, or a vector
without that layer or its residual norm. It names each agent a panel's seats
are cast with by the same rules, less the answer-token and revision ones (a
panel reads its seats' generated turns, and loads each seat at its own
revision), and the Mac command line's `panel check` refuses such a seat in a
bound panel too; to fix one, make the corrected agent and put it in that seat
(the refusal's repair names the commands). A revision is refused only when both the agent and the
study name full commits that differ (or a shortened hash the study's commit
does not begin with); a branch, a tag, a shortened hash the other begins with,
or an agent that records no revision gets an `agentRevisionNotChecked` note
instead. To fix an arm, make the corrected agent and put it in the arm's place
with `attach-agent <name> --replace <arm> …`, which replaces that arm on a
draft and gives it the new agent's name (it refuses when another arm already
has that name or file). For a thinking agent beside an answer-token
instrument, either declare instruments that read the answer after the
reasoning, or replace the arm with the agent made with `--reasoning-effort
off`. A study frozen
before these rules verifies with a `refusedWhenRun` note and is refused when
its run is submitted or started; duplicate it, then repair the duplicate the
same way. Each refusal's repair names the commands.

**`set-system-prompt <name> "<text>"`** — declares the study's system prompt
on a draft: the deployment frame every arm is read under. `""` clears it. The
text is stored inline (there is no file and no hash beside it), and what the
model receives is capability-dependent, decided by the renderer: a family
whose chat template has a system role gets a **genuine system turn**; a family
without one — Gemma — gets the **same text prepended to the first user turn**
(`system + "\n\n" + user`); `rawCompletion` prepends it to the prompt text.
Every route delivers it, so there is no prompt-mode gate. An arm carrying an
agent persona reads under *persona*, blank line, *this frame* — declaring one
never displaces an agent's identity. The one place a declared frame does not
apply is a pinned item whose scripted transcript opens with its own `system`
turn, which replaces it for that item; the verb counts those and says so
through the `systemPromptNotApplied` advisory rather than letting the
substitution be silent.

**`set-parser <name> <parser>`** — declares the numeric-endpoint parser from
the workspace registry (`prompts/parsers/parser-registry.json`) on a draft and
pins that registry's SHA-256 as `parserRegistryHash`; `""` clears both. The
hash is never an argument: the registry file is the authority on which parser
VERSION the study preregistered, so re-declaring the same name is also the
drift repair. Without a declaration a numeric study falls back to the
DEPRECATED implicit selection (`caseFamily: "sentencing"` → the built-in
duration parser), which every firing site now announces.

**`set-instrument-scope <name> <responseFormat>[,…]`** — declares which
response formats (`label`, `json`, `freeText`) the option-consuming
instruments apply to on a draft, pinning the row set they select
(`itemCount` + `itemIDsHash` computed from the study's own task prompts);
`""` clears the declaration. This is the NON-LOSSY repair the run-start
`responseFormat` refusal names: a mixed json+label file keeps
`answerTokenLogprob`/`ordinalScale` on its label rows instead of dropping the
instrument for `sampledText`. Declaring formats no pinned item carries is
refused — a scope selecting zero rows would produce zero records.

**`set-jlens-readout <name> <lens-id> --layers <l,…> [--top-k <k>]
[--top-k-layers <l,…>] [--watchlist <token-id,…>] [--no-companion]
[--qualification <id>] [--tokenizer-hash <sha256>] [--max-armed-layers <n>]
[--max-observations <n>] [--max-trace-bytes <bytes>]
[--max-full-vocab-projections <n>] [--keep-tokens] [--dry-run]`** —
declares the study's J-lens run readout on a draft (`jlensReadout`): the
top-k tokens and the watched tokens' scores at the chosen source layers, at
the final prompt position and every generated token, read during the study's
own generations. `""` as the lens id clears it. Layers take ranges (`20-24`);
`--top-k` is 0 to 50 and the watchlist holds at most 64 token ids. The
logit-lens companion (the same readout without the lens) rides along unless
`--no-companion`. `--keep-tokens` also keeps the exact sampled tokens, so
the run can be replayed exactly. Only the Python engine records a readout.

The verb prices one response at the study's token cap, as the engine does at
run start, and says how many responses the declared budget covers. At a
2,048-token cap, top-50 at 8 layers with the companion costs 32,768
full-vocabulary projections and about 41 MB of trace per response, so the
default budget covers 6 responses. Run start refuses a study with more
responses than that (conditions × items × samples), and the verb refuses a
readout that cannot fit even one. The `--max-…` flags declare a different
ceiling; each declared ceiling is pinned with the readout.

Every pin is checked: the lens must be imported and fitted on the study's
model, every layer must be one the lens fitted, and the pins include the
tokenizer and configuration hashes run start recomputes. Freeze also needs a
pinned passing qualification, a pinned model revision and dtype, and an
evidence-tier lens.

<!-- client: mac -->

This command line never reads a lens (J-lens work is server-only), so the
lens's pins are flags: `--lens-sha256 <sha256> --lens-model
<model-id>[@<revision>] --fitted-layers <l,…>`, plus `--tokenizer-hash` and
`--qualification`. Where the lens is imported, the Python client's dry run of
this same verb (`--dry-run --json`) prints all of them, with this command's
exact spelling in `result.macCommand`; the lens's own `lens.json` holds the
first three as `source.tensorSHA256`, `fit.modelID`, and `sourceLayers`.
Without `--tokenizer-hash` the tokenizer and
configuration hashes are not pinned, and freeze refuses. Without
`--qualification` a run is exploratory and freeze refuses; the block stays
valid if the lens is qualified later, and running this verb again with
`--qualification` pins it.

<!-- client: python -->

This client reads the pins itself: the lens's bytes and fitted layers from
`runs/jlens-lenses/<lens-id>/lens.json`, the tokenizer hash from the study
model's files in this machine's model cache (or `--tokenizer-hash` when the
model is not installed here), and the newest passing qualification for the
study's declared revision and dtype when `--qualification` is not given (a
named one is checked against its own revision and dtype when the study
declares none). `--dry-run` writes
nothing and reports the pins in `result.pins`, with the Mac command line's
spelling of the same declaration in `result.macCommand`.

<!-- client: all -->

**`set-prompt-reading <name> (--single-pass | --chunk-tokens <n>
--above-tokens <m> | --clear) [--dry-run]`** — declares how the Python engine
reads this study's prompts (`promptReading`). `--single-pass` reads every
prompt in one pass. `--chunk-tokens N --above-tokens M` reads a prompt longer
than the larger of the two numbers in chunks of N tokens, and a shorter one in
one pass. Reading in chunks lets a long prompt fit on one GPU, because the
pass no longer holds working memory for the whole prompt at once. Every token
still attends to every earlier token, and only the order of the arithmetic
changes. Seeded texts can therefore differ, at near ties, from a run that read
the same prompt another way. The declaration is frozen with the study, so
every run of it reads its prompts the same way. `--clear` removes it.

Without a declaration, each submission chooses a reading and sends it to
every job it starts, shards included. On CUDA that is the engine's default:
chunks of 4,096 tokens for prompts over 8,192 tokens. When the memory check
estimates the default will not fit the requested card, the submission uses a
smaller chunk and says so, because those texts will not match runs that used
the default token for token. Declare a reading when a study must match runs
made another way, or when twin studies must read their prompts alike. The
default is the same on every card, however much memory it has, so one study
reads alike on every job. A card with memory to spare gains nothing that
matters from a single pass: the arithmetic is the same and chunks cost well
under a second for a 200,000-token prompt. So declare `--single-pass` (or
larger chunks) when the researcher wants that reading, not to speed a run
up; the memory check then confirms it fits the requested card. A
server too old to honour the declaration is refused at submit, and the Mac's
own engine runs the study with a note that the declaration applies to the
Python engine.

The memory check sizes the job by the longest prompt, measured with the
run's own token count, plus `maxTokens` and `reasoningMaxTokens`. It refuses
only when the weights and that cache cannot fit the card. The prompt pass's
working memory is an estimate: when it exceeds the card, the check advises
and does not refuse. A model always loads onto one GPU.

In a study run, answer-token readouts (`answerTokenLogprob`,
`choiceProbability`, and `ordinalScale`) read their prompts by the same rule
as generation, so a study that can generate at a length can also be read out
at it. The run records this as `answerTokenReadoutsFollowRule` in
`config.json`'s `notes.promptReading`. A run started before this release
keeps reading its readouts in one pass when resumed, and its log says so. A
long prompt read that way may not fit the card, so start a new run to read
them in chunks.
The Mac's own engine reads every readout's prompt in one pass.

**`set-evaluation-sampling <name> <n> <seed>`** — declares the study's
EVALUATION SAMPLING DESIGN on a draft: how many records per condition the
judged coding preregistered, and the seed that draws them
(`evaluationSampling`); `<name> ""` clears it. Both halves or neither. The
draw RULE is the third field and is derived from the engine at the write —
never an argument, for the same reason `parserRegistryHash` is not one.

Declare it rather than typing the flags. A stamp records what HAPPENED;
"preregistered" is a claim about what was decided BEFORE anything ran, and
only the declaration puts that claim in the artifact chain — every run writes
the manifest snapshot into its own `experiment.json`, so the design travels
with the evidence. `evaluate` then draws it with no flags at all;
`--sample-per-condition`/`--sample-seed` may still be typed and become a
CROSS-CHECK, refusing on any inequality rather than overriding. Declaring is
measurement-side, so it never invalidates the run being coded — which is what
lets you duplicate a frozen study, declare the coding design on the duplicate,
and evaluate against the original's run. What the desk checks is a whole `n`
of at least 1, a seed that parses, and a rule this build derives; the
POPULATION check stays at `evaluate`, because at declaration time the source
run need not exist yet.

The design must also be one this study's INSTRUMENT can run: the draw is over
per-response coding records, and a paired rubric refuses every sampling
request. So a declaration plus a PAIRED pinned rubric is a verify (and
therefore freeze) violation naming both facts and both repairs — clear the
declaration, or pin a `perResponseCoding` rubric — and this verb refuses the
same way when the rubric is already pinned. Declaring before choosing a rubric
stays legal; the gate fires only when both are present.

**`set-primary-outcome <name> <outcome>`** — declares the outcome the study is
ABOUT (`primaryOutcome`); `<name> ""` clears it. Every results summary of the
study then leads with that outcome and says it was "declared by the
researcher". Ask the researcher which outcome their question turns on, and
declare it before freezing: the declaration is frozen with the study, appears
in the generated settings summary, and travels in every run's manifest
snapshot.

`<outcome>` is an outcome name: `judged` (the judges' verdict, read from the
evaluation report), a choice or numeric outcome (`choiceLogOdds`,
`ordinalPosition`, `parsedValueMean`, `meanMonths`, `choiceRate`), a reader
score (`readerScore:<concept>`), a reasoning-style feature
(`rs_<feature>`), a concept's marker density (`<concept>MarkerDensity`), or a
surface measure (`wordCount`, `distinct2`). An outcome the study's settings
cannot produce is refused (`blocked`, exit 64), and `error.repairAction` lists
the ones they can — for example `choiceLogOdds` needs the
`answerTokenLogprob` instrument, and `judged` needs judges and a rubric. The
result echoes the same list as `result.producibleOutcomes`.

Declaring nothing is fine. Summaries then lead by the default order — a judged
outcome, then a choice or numeric outcome, then a reader or probe score, then
reasoning style, then marker density, then surface measures — and say "chosen
by default order". Declaring is measurement-side, so it never invalidates a
run that already exists.

**`set-style-taxonomy <name> prompts/taxonomies/<file>.json`** — pins a
reasoning-style taxonomy (path + hash) on a draft. No pin, no reasoning-style
scoring; drift after pinning is a verify violation like any other.
