# Diagnostic transport, custody and reviewed cleanup

Branch `codex/remote-custody-cleanup`, based on main `6a94a8f`.
This extends the [cluster execution slice](CLUSTER-EXECUTION-WORKFLOW-HANDOFF.md).
The maintainer's reviewing agent, through the user, owns diff review and landing.
No live site, run, deployed app or main checkout is changed by this work.

## What changes for the researcher

A researcher can ask an agent to prepare a standalone battery or stability request
using the shared scientific guide, review the input inventory, and send an isolated
copy to a runner. After execution, the agent or app can bring the evidence home,
verify it without the server, and present a bounded cleanup plan. The local workspace
remains the source of truth. Completing a remote job no longer leaves evidence
custody as an undocumented manual step.

Both clients, the HTTP adapters and the app compose the same format and admission
owners. On the Mac, archive work calls the local Python client through a fixed
module, without a shell or network operation; an unavailable environment gives a
setup repair. This dependency is explicit: install the Python client environment
alongside the updated app for diagnostic transport. It does not load a model.

The app's Server Jobs panel opens **Inputs, evidence and cleanup** with the selected
job's originating client and current local workspace captured. Scientific workflows
also opens offline custody without a server. Input review, staging, execution review,
submission, import, verification and cleanup are separate explicit operations.
Changing app selection cannot retarget an open sheet. No submission retries itself.

## Public workflow

Both command lines accept `--root <workspace>` and `--json`. Python remote verbs
use `steerlab runner … --runner <url>`; Mac remote verbs use
`steerlab-cli remote … --site <id>` or `--url <url>`. Python authentication retains
the token-file convention; Mac credentials remain in the Keychain.

| Step | Both clients' verb (under the appropriate namespace) | HTTP owner |
|---|---|---|
| Discover exact local input files | `science input-plan <request.json>` | Workbench `POST /api/science/workspace/input-plan` |
| Capture reviewed inputs | `science package <request.json> --archive <new.tar.gz> --plan-sha256 <digest>` | Workbench `POST /api/science/workspace/package` |
| Upload using existing policy checks | Python `bundle upload`; Mac `remote upload`; or approved external transport | Existing bundle upload owner |
| Stage isolated execution copy | `runner/remote science-stage <server-archive-path> --sha256 <digest>` | `POST /api/science/stage` |
| Review and submit | Existing `science-plan` / `science-submit`, using the returned `request` object | Existing science plan/submit owners |
| Observe/reconnect/cancel | Existing jobs, logs, recovery and cancellation verbs | Existing durable job owners |
| Export completed evidence | `runner/remote science-export <job-id>` | `POST /api/science/jobs/{job_id}/export` |
| Download and import | `runner/remote science-fetch <job-id>` | Export plus existing policy-checked download; shared local import |
| Import externally transferred bytes | `science import <archive> --sha256 <exported-digest>` | Workbench `POST /api/science/workspace/import` |
| Verify offline | `science custody`; `science verify-custody <receipt-digest>` | Workbench `POST /api/science/workspace/custody` or `verify-custody` |
| Review removal | `runner/remote cleanup-plan <job-id> --receipt-sha256 <digest>` | `POST /api/science/jobs/{job_id}/cleanup-plan` |
| Apply reviewed removal | `runner/remote cleanup-apply <job-id> --receipt-sha256 <digest> --plan-sha256 <digest> --confirm-removal` | `POST /api/science/jobs/{job_id}/cleanup-apply` |

Workbench workspace bodies contain explicit `workspaceRoot` plus the action fields:
`requestFile`; package additionally `archivePath` and `planSHA256`; import
`archivePath` and `archiveSHA256` (optional `expectedContext`); verify
`receiptSHA256`; list no other fields. Unknown fields refuse. The serving workspace
must match. Runner deployments refuse these authoring operations by the existing
role census. Staging and exports are legitimate execution-copy operations.

Stage body is `{bundlePath, bundleSHA256}`. It returns
`request: {inputBundleSHA256: "…"}`; save that object as the request document for
existing plan/submit commands. Model acquisition/qualification is separate: weights
are not implicitly packaged. Batteries include referenced vector files, model
variant documents, adapter directories and neutral bases; stability includes its
source manifest and required concept/reference data. Unknown paths, links, missing
members or drift refuse. Queue admission rechecks the complete staged file inventory before model loading,
including adapters and neutral bases. Portable variant dependency references must
be relative; an absolute reference refuses with the originals retained. Author a
new portable variant through the normal immutable-publication workflow instead of
editing an existing artifact.

Plan and submit must use the same controller session. Its host and PID are bound
into the plan; reconnecting to a successor requires a fresh plan. These diagnostics
retain their scientific types and do not support checkpoint resume.

HTTP transfer is used only where the profile permits it. For external-only policies,
use the existing approved site transport into the declared run root, then stage;
export provides the exact path/digest for transfer back and local import. This slice
does not invent a new SSH/rsync scheduler or automate every site's external transfer.

## Custody and failure semantics

Input and evidence archives use distinct schema-1 kinds, exact ordinary-file
inventories and SHA-256 hashes. Readers reject path traversal, links, duplicates,
extra/missing members, malformed metadata and excessive file/byte counts. Import
retains the archive under `.steerlab/diagnostic-archives/`, publishes only a new
`runs/<id>` or `diagnostics/<id>` directory, then issues a content-addressed receipt
under `.steerlab/diagnostic-custody/`. Existing matching output is reused; differing
output refuses. Directory publication uses atomic no-replace primitives on Mac/Linux;
unsupported filesystems refuse instead of falling back to replacement.

Verification reads the receipt bytes, retained archive, and complete expanded file
inventory, including checks for unexpected files. Root and originating job/plan are
bound. Neither receipt nor transport adds a manifest revision or rewrites frozen
bytes. A receipt proves local custody, not scientific correctness or qualification.

Import is recoverable, not a filesystem-wide atomic transaction: interruption after
output publication can leave verified files/archive without a receipt. Retain those
files and rerun the same import; receipt issuance is idempotent and never replaces
output. A receipt is usable only after complete verification succeeds. Cleanup
interruption records an attention state and the surviving original/quarantine paths;
inspect its audit before any further removal. Never infer success from a timeout.

## Cleanup policy and exact boundary

A server administrator explicitly sets `STEERLAB_DIAGNOSTIC_CLEANUP_POLICY` to an
ordinary JSON file containing, for example:

```json
{
  "schemaVersion": 1,
  "allowDiagnosticOutputRemoval": true,
  "minimumRetentionHours": 24,
  "source": "Reviewed retention policy supplied by the workspace operator"
}
```

This example is not permission or a default policy. Obtain the actual rule from
the site documentation/operator; absent, malformed or disallowing policy refuses.
Policy bytes and declared minimum retention are bound into the reviewed plan.
No private profile or launch configuration is silently changed. The policy path
must be configured in the serving controller's environment by its administrator.

Only the output directory of a **successful standalone diagnostic in an isolated
staged execution copy** can be removed. Original workspace outputs, partial/failed
jobs, active/uncertain/checkpointed/resumable work, inputs, cached models, export
archives, job records and local files are outside this removal scope. The server
checks other job results and workspace experiment/model-variant declarations for
dependencies, blocking unreadable declarations. It reads all durable jobs under an
exclusive transaction; foreign-controller registrations and updates cannot race
admission/removal. Existing workspace admission also excludes local submissions.
Ordinary out-of-band filesystem writers are not a supported coordination mechanism.

These locks are held during export/tree/archive hashing and, on apply, quarantine
verification and deletion. Same-controller bookkeeping waits; other controllers'
SQLite writes may time out after 30 seconds. Even small outputs can be slow on a
shared filesystem. Before expanding artifact scope, measure critical-section time
and test competing controllers; retain fresh dependency and byte admission when
shortening locks. See the N1 acceptance criteria in
[the review follow-up](WORKFLOW-REVIEW-HARDENING-HANDOFF.md).

Every shipped cleanup adapter re-verifies the local receipt immediately before
plan/apply. The server cannot inspect a different machine's disk: it compares the
client's custody attestation to its own exact export context and members. The
operator remains responsible for storage outside the application and undeclared
external consumers. No claim is made to discover arbitrary external dependencies.

Apply requires explicit confirmation and the unchanged plan digest. It records
intent, moves the bounded target to quarantine without replacement, verifies its
members again, removes it and records removed/retained paths. No broad `runs/` purge,
cache eviction, automatic retention daemon or scientific artifact mutation is added.

## Advanced scientific access

The maintained method catalog now describes each existing HTTP action's ID, method,
path, service role and authority rationale. Both clients expose
`runner/remote science-call <operation> --action <id> --request <request.json>`.
The app's Scientific workflows view offers those same actions. A request has exactly:

```json
{"path": {}, "query": {}, "body": {}}
```

Fill path placeholders exactly with single components, query values with strings,
and body with the existing route's JSON object. GET requires an empty body. Read the
operation's guide and original request schema; this adapter does not invent a second
scientific schema or weaken the route's authority/authentication/preflight gates.
Large integer seeds survive the raw Mac request/response path. Mutations are sent
once; ambiguous failures require inspection before retry. The app currently exposes
a guided JSON action editor, not bespoke conceptual forms for every advanced method.

The gate compares catalog HTTP actions against the tested runtime route census and
engine command declarations. Reader, fine-tune, J-lens, Gemma Scope and
judgment-completion paths are reachable where already HTTP-supported. Style
rescoring, SAE engine tools and nine OptVec commands still use their named Python
engine CLI: they have no new managed HTTP execution owner in this slice. This is an
explicit remaining interface limitation, not proof that an implementation gap is
scientifically necessary. In particular J-space remains exploratory analysis.

## Review feedback incorporated

The previous review has no blocking correction. Its controller-session note is now
in the guides, agent contract and UI. Its private registry observation requires a
maintainer decision: either intentionally track `.authoring/` citation companions
in the private Sites repository or ignore them there. They are not credentials;
secrets still belong in the Keychain. Do not edit the live private registry as part
of this public implementation. Its `.steerlab/manifest-locks/` retains its own ignore.

## Validation and landing

See the [validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md) for actual
suite results. Tests exercise malicious/drifting archives, immutable imports,
offline verification, stale plans/policy, active and foreign-controller jobs,
closed HTTP/CLI requests, a real loopback fetch/cleanup journey, raw integer parity
and a Python-created archive imported and verified through the Mac adapter.
They do not establish live Slurm execution, site transport policy, GPU arithmetic,
interactive app usability or Linux filesystem qualification. No mechanical code
move is claimed; scientific computation owners remain unchanged from the base.

Before landing: read the complete diff, both suites under Xcode beta/Metal with
external scratch, resource/reference/role gates, scientific AST/boundary audits,
bridge gates normal/release, public vocabulary scan and whitespace checks. Recheck
main ancestry; only the maintainer's designated reviewing agent integrates, by
fast-forward when appropriate. Rebuild the app/CLI and serving engine from the
landed source before using the new paths. This branch performs no deployment.

Next qualification should follow a complete researcher journey through both an
agent and the app, including interruption/reconnect, external-only transport,
missing policy, local receipt drift and offline results. Remaining engine-only
operations need separately scoped managed execution owners if universal remote
execution parity is required; catalog coverage alone does not close that work.
