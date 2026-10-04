# Preparing models

A model download can be tens of gigabytes. Show the researcher the plan and
get a clear yes before any `install`.

**Model preparation on a runner.** This client offers `model plan`,
`model install`, `model status`, and `model cancel` with an explicit
`--runner <url>`; a local runner (`steerlab runner serve`) uses the same
workflow.

Review the plan's model, revision, cache location and policy, then supply its
`--plan-sha256` to installation. The returned job ID and original endpoint
identify later observation/cancellation. Use `runner logs` to follow
progress; never retry an uncertain submission without inspecting jobs. A known
no-egress policy refuses downloads at the service boundary; stage through the
permitted transfer host instead. Unknown size, credentials or memory fit
remain unknown. Cached files are not numerical qualification.

**Chat-template capabilities.** `steerlab model capabilities <modelID>
[--revision <commit>]` shows the model's capability record: whether its chat
template has a system role, whether it has a thinking switch, and which
reasoning-effort levels it accepts. `steerlab model set-capability <modelID>
<field> <value> --reason <text>` overrides one detected field, with a reason
that is displayed beside the detected value and stamped into runs; `""` clears
the override.

This client holds no tokenizer, so it shows a record and cannot probe one.
Probing is the engine's `steerlab-server model capabilities --probe`.
