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
[--seed-policy manifestSeeds|derivedSHA256]
[--reasoning-effort off|on|low|medium|high|xhigh] [--reasoning-max-tokens <n>]`** —
declares the generation protocol on a draft, merge-style: only the flags given
move, `""` clears `--prompt-mode`/`--seed-policy`, and `--samples-per-item 1`
clears to the deterministic default. The reasoning protocol replaced the
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
be declared one flag at a time.

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
claim nothing checked.

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
