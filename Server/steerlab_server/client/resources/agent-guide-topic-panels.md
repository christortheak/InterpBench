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
