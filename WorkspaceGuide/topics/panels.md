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
