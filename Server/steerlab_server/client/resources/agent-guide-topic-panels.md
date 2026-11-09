# Multi-agent studies: casting a panel

A **panel** is a scenario under `prompts/panels/` — roles, turns, visibility,
case materials. A *semantic* panel binds no model to any seat, which makes it
deliberately unrunnable: it has to be **cast** first, and casting is the step
that binds the study's model and sampling settings to one seat assignment.

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

**Every seat and every turn needs its own `id`.** A seat is an entry in the
panel file's `agents`. `panel check` refuses a repeated seat or turn ID by
name (`missingPrerequisite`, exit 65): give one a different ID, check again,
and pin the corrected file. To attribute a turn in `generations.jsonl`, read
`speakerAgentID`; `speakerName` is a display name and may be shared. Runs
made before that key existed lack it; join `turns.jsonl` for those.

## What a seat passes on

Whether a seat thinks belongs to whoever sits in it: a cast agent's own
level (its file's `reasoningEffort`, else its older thinking switch, as in an
agent arm: `workspace guide settings`), or, for a baseline seat, the study's
`reasoningEffort` (so a baseline seat in a thinking study thinks, as an
ordinary study's baseline does). A seat thinks only when its model's chat
template can. A level the older switch cannot say (`on`, `low`, `medium`,
`high`) reaches the template even when the study declares no budget, and each
turn's `reasoningEffort` is the level the template rendered (a level it
ignores is recorded `on`). A turn played as thinking never records `off`: a
seat that asks to think and that no probe of its template has cleared is
split, and budgeted when the study declares a budget, and records `on` when
no level reached a thinking switch. The reasoning
budget is the study's `reasoningMaxTokens`: a thinking seat gets it when the
study declares one, and its turn's `maxTokens` is then the budget for its
answer. A study whose own effort is off may still declare
`reasoningMaxTokens` when its panel casts an agent that thinks; with nothing
in the panel thinking, a budget is refused. To cast a thinking judge beside a
study that does not think, cast an agent with no steering and a thinking level
(`agent create <name> --model <id> --revision <sha> --reasoning-effort on`).

**A seat's agent meets the agent rules.** On either client, `experiment
verify` (a violation on a draft, a `refusedWhenRun` note on a frozen study),
submission, and run start refuse a seat whose agent file says two things
(fields that disagree, or both a promotion and a handmade record), names a
level its model's chat template does not accept, lists more than one adapter,
or steers at a layer its vector does not have or without that layer's
residual norm. The Mac command line's `panel check` refuses such a seat in a
bound panel too; the cross-platform client's `panel check` checks only
unbound semantic panels, which cast no agents. The answer-token and revision
rules of an agent arm do not apply to a seat. A seat whose agent file says
two things also refuses when it plays, on either engine. The checks read a
seat's agent file as it is now, because that is the file the run loads: an
agent saved in place after the panel was compiled is judged as saved. To fix
a seat, make its agent again and put it in the seat: compile the panel again
(in the Mac app's study editor, the Seats section), or, for a hand-bound
panel, pin a copy of it that names the new agent. The refusal's repair names
the commands, and the seats by their IDs.

**Other seats read a thinking seat's answer, never its reasoning.** Each such
turn is split into the two: `output` on its record is the answer, and
`reasoning` holds what it thought. A template's `outputs.<label>` placeholder
and a contract input read the answer. An endpoint is read from the answer;
on a `complete` turn its stamp carries `markerInReasoning: true` when the
reasoning also wrote the marker. **When in doubt, nothing is passed on.**
Whether the reasoning block opened in the prompt is read from the prompt each
turn rendered, and stamped as `reasoningOpenedInPrompt`. A turn's
`reasoningStatus` says how it was read: `complete` (the reasoning finished
and the answer followed), `notFound` (the prompt did not open the block and
the text has no reasoning tag, so it is all answer), or one of three under
which the turn has no answer, nothing from it reaches any seat, and its
`routedAgentIDs` is empty: `cutOff` (a token limit ended it before its
reasoning finished) and `unfinished` (the seat stopped first), whose
`reasoning` holds all it wrote, and `unclear` (a reasoning tag appeared where
the answer should be), whose whole text is kept as `withheldText`. Any other
status passes nothing on either. A routed answer never holds a reasoning tag.
A seat that does not think passes on its whole text, as before.

A seat reads its **own** earlier reasoning only when the panel declares
`"includeOwnReasoning": true` (off by default, as chat templates drop
earlier turns' reasoning). It then sees its routed turns and its own
contract inputs with their reasoning, and its colleagues still see answers
only.

**Temperature is the study's**, for every seat that declares none. A panel
may give a seat its own with `"temperature": <t>` on that seat; a seat at 0 is
greedy and one above it is seeded per turn. A study at temperature 0 with a
warm seat may run several samples per item, and its records and settings
name the seed of each turn. An agent file's temperature is never used.
Every turn records `temperature`, `temperatureSource` (`study`, `panelSeat`,
or `panel` when no study runs it), `reasoningEffort`, and
`reasoningMaxTokens`. `panel check` refuses a negative seat temperature, and
`experiment verify` names one that equals the study's
(`seatTemperatureChangesNothing`), because it changes nothing.

An engine from before this rule passed reasoning on, never let a baseline
seat think, and ignored seat temperatures. Both clients refuse to send it a
panel where that changes something (`remoteStudy.panelAnswerRouting` missing
from its capabilities), and say what would change. An agent that asks to
think counts as thinking here unless a probed capability record for its exact
model revision says the template cannot. A bundle packaged to run such a panel
requires `panel-answer-routing-v1`, which current engines support. A run such an engine started is not
resumed or merged under the new rule when that changes what its seats read;
start a new run.

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
