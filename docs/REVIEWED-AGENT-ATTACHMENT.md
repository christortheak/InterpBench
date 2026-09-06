# Attach a reviewed agent to a study

An agent is a saved intervention artifact. Attaching one adds its condition to a
draft; it does not run the study, change the artifact or establish that the
intervention is scientifically effective. The app, Mac CLI and Swift workbench
HTTP adapter use `StudyAgentAuthoring` for this operation.

## Researcher and agent workflow

The researcher chooses the intervention and target study. A collaborating agent
can inspect the local library, review the artifact and attach it through supported
commands:

```sh
steerlab-cli agent list --json
steerlab-cli agent inspect <path> --json
steerlab-cli experiment manifest <study> --json
steerlab-cli experiment attach-agent <study> --artifact <path> --artifact-sha256 <digest> --manifest-sha256 <digest> --json
```

Inspection returns `artifactFileSHA256`, the decoded `artifact`, and the exact
stored JSON `document`. The library result contains `agents` with those same
fields and an `issues` list for discovered entries that failed inspection. It
uses the existing native-library and imported `variant-save` discovery rules;
this is not a complete diagnostic census of malformed or unrecognized files.
A named inspection can diagnose a file that discovery did not recognize.

Use the artifact's reviewed file digest and the study's `manifestFileSHA256`.
These preconditions travel outside the documents. The shared command reads the
artifact once for its embedded condition and pin, verifies the reviewed versions,
and publishes against the study's file digest under its transaction lock. It
never combines an old picker artifact with a hash from a new file.

A changed study refuses with `staleManifest`; a changed, unreadable or mismatched
agent refuses with `artifactPin`. Frozen studies refuse with `statusImmutable`.
In JSON mode these are exit 65, malformed requests are 64, and missing named
files are 66. Read the repair, inspect and review the changed inputs, then
reconstruct the intended request. Do not refresh digests merely to retry an old
edit. The resulting condition is pinned by the service; nobody fabricates a hash.

## App

Selecting an agent in the study's Add agent picker captures its artifact review.
Add uses that review and the study editor's retained review. A stale file refuses;
failed attachment retains the selection. **Reload agent selection** deliberately
captures a new artifact version. Changing study or workspace clears that local
selection. Adding from the Agent Library checks that its displayed record still
matches the artifact before invoking the same command.

The study picker can explicitly carry the researcher's unsaved base-model choice
into the attachment, preserving the existing model-change rule. CLI/HTTP attach
to the reviewed study's saved model; author a model change separately there.
An incompatible agent is refused rather than changing the model implicitly.

## Swift workbench HTTP

All operations require an explicit absolute `workspaceRoot`, matched against the
workspace served by the workbench. Agent paths are relative files under `runs/`,
including the native `runs/model-variants/` library and imported evidence.
Traversal and links in a named artifact path are refused.

| Operation | Additional body fields | Response |
|---|---|---|
| `POST /api/agent/list` | None | `agents`, `issues` |
| `POST /api/agent/inspect` | `artifactPath` | `path`, `artifactFileSHA256`, `artifact`, `document` |
| `POST /api/experiment/attach-agent` | `name`, `artifactPath`, `artifactFileSHA256`, `manifestFileSHA256` | Authoritative saved study document and new manifest file digest |

Missing attachment preconditions return 428; a stale study returns 412; artifact,
model, frozen-status or wrong-workspace refusals return 409. Unknown fields or
malformed values return 400; missing files return 404. Refusals include a repair.
No operation changes selection, connects compute, rewrites an artifact, or runs a
study. Imported evidence remains immutable.

## Remaining work

This completes the Mac/Swift attachment path, not all artifact preparation.
Python parity mapping, generic vector/reader attachment, design batch authorship,
remaining bridge retirement, and interactive app qualification remain tracked.
Artifact discovery retains the existing store's recognition limits. This
checkpoint does not claim a filesystem transaction spanning mutable artifacts;
artifacts in `runs/` remain immutable, and downstream verification still checks
that the pinned bytes remain available.
