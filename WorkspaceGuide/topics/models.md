# Preparing models

<!-- client: all -->

A model download can be tens of gigabytes. Show the researcher the plan and
get a clear yes before any `install`.

<!-- client: mac -->

**Local model preparation.** Use `steerlab-cli model plan <owner/repo> --revision <commit-or-ref> --json`
to inspect this Mac's cached file set without a download, credential lookup or
weight load. Omit revision to inspect `main`. Missing or partial cache files are
not an installed revision. A present file set is not memory-fit or scientific
qualification; those remain separate checks. Then explicitly use
`steerlab-cli model install <owner/repo> --revision <commit-or-ref> --json`
to fetch through the app's installer and wait for completion. This foreground
command is local to the Mac; it creates no server job. Failed or interrupted
fetches can leave partial cache files for a later install. Inspect the returned
status and run `model plan` again before loading or conducting a study. The app's
local-model HTTP operations expose plan, install, status and cancellation of its
own installer. A cancellation names the observed request ID; it cannot follow a
newer install. Server installations remain separate and subject to server policy;
never use the Mac cache result as evidence of server readiness.

**Remote model preparation.** The Mac offers `remote model-plan`,
`remote model-install`, `remote model-status`, and `remote model-cancel` with
`--site <id>` or the explicit endpoint.

<!-- client: python -->

**Model preparation on a runner.** This client offers `model plan`,
`model install`, `model status`, and `model cancel` with an explicit
`--runner <url>`; a local runner (`steerlab runner serve`) uses the same
workflow.

<!-- client: all -->

Review the plan's model, revision, cache location and policy, then supply its
`--plan-sha256` to installation. The returned job ID and original endpoint
identify later observation/cancellation. Use `{{remote}} logs` to follow
progress; never retry an uncertain submission without inspecting jobs. A known
no-egress policy refuses downloads at the service boundary; stage through the
permitted transfer host instead. Unknown size, credentials or memory fit
remain unknown. Cached files are not numerical qualification.

<!-- client: mac -->

The app's Plan and Install controls use these same services.

<!-- client: all -->

**Chat-template capabilities.** `{{cli}} model capabilities <modelID>
[--revision <commit>]` shows the model's capability record: whether its chat
template has a system role, whether it has a thinking switch, and which
reasoning-effort levels it accepts. `{{cli}} model set-capability <modelID>
<field> <value> --reason <text>` overrides one detected field, with a reason
that is displayed beside the detected value and stamped into runs; `""` clears
the override.

<!-- client: mac -->

`steerlab-cli model capabilities <modelID> --probe` derives the record from
the pinned template and writes it into the workspace under `prompts/models/`.

<!-- client: python -->

This client holds no tokenizer, so it shows a record and cannot probe one.
Probing is the engine's `steerlab-server model capabilities --probe`.
