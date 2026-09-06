# From research question to a reviewed study draft

This workflow joins conceptual study authorship to the existing design, model,
verification and execution operations. The app, Mac `steerlab-cli` and Swift
workbench HTTP API share the new pack, prompt-import and vector-attachment
owners. They use the same saved workspace; switching interfaces does not require
copying a second study or reconstructing its hashes.

This is the Mac authoring path. The Python `steerlab` client retains its own
supported authoring/bundle/runner path; it does not yet implement these pack or
design commands. An imported draft is not a qualified experiment. Scientific
review, prerequisites, freeze and execution remain distinct decisions.

## 1. Start with the researcher's question

In an existing workspace, ask the installed Mac CLI for the study interview:

```sh
steerlab-cli authoring study conceptStudy --json
```

The other intents are `agentComparison` and `multiAgent`. This returns the same
interview prompt as the app's **Copy LLM Prompt**. The agent should establish the
hypothesis, comparison, measured outcomes, relevant controls and available
compute before proposing methods. The researcher can make those decisions in
ordinary language. The agent translates them into declarations and checks them
with the instrument.

Use `steerlab-cli authoring prompt --help`, then `authoring prompt <kind> --json`
for the required dataset format and the generation/review prompts for that kind.
Do not manufacture input hashes, norm measurements, model revisions or scientific
qualification. Ask about unresolved research choices; use the actual engine's
refusals to identify missing technical prerequisites.

An existing design may replace pack assembly: `design list`, `design inspect`,
then `design instantiate` or `design batch` with reviewed design bytes and
explicit casting. These retain their existing lineage and per-row result rules.

## 2. Review and import the proposed pack

A pack is JSON with `study` containing a complete manifest and optional `files`
mapping workspace-relative paths under `prompts/` to UTF-8 text. A plain manifest
is also accepted. This is authored input, distinct from an execution bundle.

```sh
steerlab-cli pack preview <pack.json> --json
steerlab-cli pack apply <pack.json> --review-sha256 <reviewSHA256> --json
```

Preview does not write. It shows the new draft's name, the file plan, referenced
inputs and advisories. Apply must receive that exact preview's digest. Changes
to pack bytes, destination workspace or observed inputs require another preview
and review. Existing differing files and study-name collisions refuse.

Apply creates a draft, strips freeze metadata, pins supported named input bytes
and reports `verificationIssues` and `filesWritten`. Even a successful import may
need scientific declarations, datasets, a scenario, agents or vectors. Read that
result before proceeding. An invalid or missing named input remains an explicit
verification issue; the importer does not fabricate its pin.

In SwiftUI, **Paste Study JSON… → Preview → Import as Draft** uses the same
operation and opens the created draft. Review the reported issues there. A
refused import keeps the dialog available for repair.

## 3. Continue with the same saved study

Inspect its current version before another agent-driven edit:

```sh
steerlab-cli experiment manifest <study> --json
steerlab-cli experiment import-prompts <study> --file <records.jsonl> --manifest-sha256 <manifestFileSHA256> --json
```

Prompt import preserves full records, including identifiers, response options,
targets and additional fields. It publishes `prompts/tasks/versions/<sha256>.jsonl`
and pins those bytes. It never overwrites an earlier input. The app's JSONL and
mapped-table imports use this same publication owner. A stale study review
refuses before preparing a version. A later publication failure can leave an
unreferenced prepared input; previous inputs and pins remain intact.

When an agent edits a study while its native editor is open, the app requires
an explicit reload before old form fields may be saved. A catalog refresh alone
cannot authorize overwriting the newer study. No revision field is inserted
into content-hashed manifest bytes.

Attach stored agents through `agent inspect` and `experiment attach-agent`.
For a reader-derived, optimized or other existing vector pair:

```sh
steerlab-cli experiment inspect-artifact <relative-vector-path> --json
steerlab-cli experiment attach-artifact <study> <concept> --artifact <relative-vector-path> --artifact-sha256 <artifactSHA256> --sidecar-sha256 <sidecarSHA256> --manifest-sha256 <manifestFileSHA256> --json
```

Use the digests returned by inspection. The path names the `.safetensors`/`.json`
pair and can omit its extension. The store still enforces provenance, model,
substrate and residual-norm requirements. Supply optional `--source-concept` or
`--eval-run` only where appropriate to the artifact. Attachability is decided by
these checks, not by the existence of files. Run and vector bytes remain read-only.

The native **Attach vector artifact…** sheet captures its study when opened,
then offers inspection and attachment of those reviewed bytes. It reports the
same scientific refusals. It attaches to that named study even if the selection
later changes, provided the workspace and reviewed files still match.

`pack export <study> --json` returns a `pack` object and `externalDependencies`.
Save the `pack` object itself as a future import document. Text under `prompts/`
is included; model/vector and other external dependencies are named separately.
It is not a replacement for `remote package` or the bundle custody workflow.

## 4. Resolve readiness and choose execution explicitly

Use `model plan <model-id> --json` to inspect local installation requirements;
`model install` is a separate action. The app's installer and the workbench
model routes use the existing shared model-preparation owner. Installation is
not scientific qualification or evidence that a particular method is supported.

Run `experiment verify <study> --json` and `data check <study> --json`, repair
missing declarations and files, and review the scientific design. Freeze only
when its prerequisites are met. Do not bypass a gate to make the demonstration
look complete.

Choose the intended execution path: local lifecycle verbs, a named
server-resident study, or explicit bundle submission. A pack import does not
submit, run a pipeline, download models or clean remote storage. Monitor the
resulting job at its recorded origin and bring evidence home through the existing
verified import workflow. The local workspace remains the authoring source.

## HTTP mapping

These are Swift workbench routes. Each stateful request supplies an absolute
`workspaceRoot`, matching the workspace the workbench actually serves. Unknown
request fields refuse. Responses carry authoritative saved versions or typed
refusals; missing write preconditions return 428 and changed versions return 412.

| Operation | POST route | Request fields |
|---|---|---|
| Study interview | `/api/authoring/study` | `intent` |
| Pack preview | `/api/pack/preview` | `workspaceRoot`, `text` (the exact JSON string) |
| Pack apply | `/api/pack/apply` | preview fields plus `reviewSHA256` |
| Pack export | `/api/pack/export` | `workspaceRoot`, `name` |
| JSONL import | `/api/experiment/prompts/import` | `workspaceRoot`, `name`, `text`, `manifestFileSHA256` |
| Vector inspection | `/api/experiment/artifact/inspect` | `workspaceRoot`, `artifact` |
| Vector attachment | `/api/experiment/attach-artifact` | inspection fields plus `name`, `concept`, `manifestFileSHA256`, `artifactSHA256`, `sidecarSHA256`; optional `sourceConcept`, `evalRun` |

## Qualification boundary

Fixture tests cover interface handover, unchanged/stale reviews, immutable prompt
versions and artifact admission. They do not constitute a GPU study, an interactive
app acceptance session, or live cluster qualification. The maintained
[operation matrix](RESEARCH-OPERATION-MATRIX.md) keeps Python parity, advanced
method guides and complete journey qualification visible as remaining work.
