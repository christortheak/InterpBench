# Multi-agent studies: casting a panel

<!-- client: all -->

A **panel** is a scenario under `prompts/panels/` — roles, turns, visibility,
case materials. A *semantic* panel binds no model to any seat, which makes it
deliberately unrunnable: it has to be **cast** first, and casting is the step
that binds the study's model and sampling settings to one seat assignment.

<!-- client: mac -->

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

<!-- client: python -->

```bash
steerlab panel list
steerlab panel inspect <path>
steerlab panel check <file>
steerlab panel import <file> --file-sha256 <digest>
steerlab panel compile <path> --experiment <study> --casting <file> \
  --file-sha256 <panel-digest> --manifest-sha256 <study-digest>
```

`panel check <file>` checks proposed semantic JSON before it is imported.
`compile` writes the bound scenario and pins it into the **draft**; it also
declares the study multi-agent, because a panel scenario is read only by that
run path. Seats you leave as null stay **baseline**, and an all-baseline
casting is the control composition, not an absence.

<!-- client: all -->

**Every seat and every turn needs its own `id`.** A seat is an entry in the
panel file's `agents`. `panel check` refuses a repeated seat or turn ID by
name (`missingPrerequisite`, exit 65): give one a different ID, check again,
and pin the corrected file. To attribute a turn in `generations.jsonl`, read
`speakerAgentID`; `speakerName` is a display name and may be shared. Runs
made before that key existed lack it; join `turns.jsonl` for those.

## What a seat passes on

Whether a seat thinks belongs to whoever sits in it: a cast agent's own
setting, or, for a baseline seat, the study's `reasoningEffort` (so a
baseline seat in a thinking study thinks, as an ordinary study's baseline
does). A seat that thinks gets the study's `reasoningMaxTokens` for its
reasoning, and its turn's `maxTokens` is then the budget for its answer. To
cast a thinking judge beside a study that does not think, cast an agent with
no steering and thinking on.

**Other seats read a thinking seat's answer, never its reasoning.** Each such
turn is split into the two: `output` on its record is the answer, and
`reasoning` holds what it thought. A template's `outputs.<label>` placeholder
and a contract input read the answer. An endpoint is read from the answer, and its stamp carries
`markerInReasoning: true` when the reasoning also wrote the marker. A turn
whose reasoning was cut off before it finished has no answer: nothing from
it is passed on, its `routedAgentIDs` is empty, and its `reasoningStatus` is
`cutOff`. A seat that does not think passes on its whole text, as before.

A seat reads its **own** earlier reasoning only when the panel declares
`"includeOwnReasoning": true` (off by default, as chat templates drop
earlier turns' reasoning). It then sees its routed turns and its own
contract inputs with their reasoning, and its colleagues still see answers
only.

**Temperature is the study's**, for every seat that declares none. A panel
may give a seat its own with `"temperature": <t>` on that seat; a seat at 0 is
greedy and one above it is seeded per turn. An agent file's temperature is
never used. Every turn records `temperature`, `temperatureSource` (`study`,
`panelSeat`, or `panel` when no study runs it), `reasoningEffort`, and
`reasoningMaxTokens`. `panel check` refuses a negative seat temperature, and
`experiment verify` names one that equals the study's
(`seatTemperatureChangesNothing`), because it changes nothing.

An engine from before this rule passed reasoning on. Both clients refuse to
send it a panel that has a seat that thinks, a seat temperature, or
`includeOwnReasoning` (`remoteStudy.panelAnswerRouting` missing from its
capabilities). A run such an engine started is not resumed or merged under
the new rule when that changes what the seats read; start a new run.

## Reviewed casting, on either client

**Complete local authoring.** Both clients offer `agent list`, `agent inspect`,
and `experiment attach-agent` with the reviewed artifact and manifest digests.
`panel inspect` reads an input and its `fileSHA256`; `panel import <file>
--file-sha256 <digest>` publishes a new semantic panel version, leaving earlier
pins intact. For
reviewed casting on either client, use `panel compile <workspace-relative-path>
--experiment <study> --casting <file> --file-sha256 <panel-digest>
--manifest-sha256 <study-digest>`. The casting names every seat, with null for
baseline and artifactPath/artifactFileSHA256 for each agent. Change the study's
model/sampling fields through its authoring operations before casting.
`experiment set-pipeline <study> --file <file> --manifest-sha256 <digest>` writes
only the reviewed pipeline declaration: an object, or JSON null to clear it.
It does not run the chain.

<!-- client: mac -->

Mac HTTP equivalents are the explicit pipeline and panel authoring operations;
a cluster runner does not become an authoring client.
