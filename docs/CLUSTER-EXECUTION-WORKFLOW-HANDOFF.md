# Cluster onboarding and remote scientific execution

Branch: `codex/cluster-execution-workflow`, based on main `73bfbd7`.
Review and landing remain with the maintainer's reviewing agent through the user.
Do not install, deploy or merge this branch as part of implementation validation.

## What changes for the researcher

The researcher supplies cluster documentation and answers the remaining account,
resource and policy questions. An agent uses the existing shipped interview and
review prompts to prepare the companion document. The app, CLI and Mac HTTP
adapter now accept that **same reviewed document**, retain its citations privately,
and hand the saved site to the existing connection/bootstrap/qualification flow.
No researcher has to extract a nested JSON object into a second file. A changed
companion requires a new review; acceptance never replaces a saved site.

For standalone capability batteries and extraction stability, the researcher can
use Server Jobs → Scientific diagnostic, or ask an agent to prepare a request.
The form and both remote clients call the same Python plan and submission owners.
The plan states the serving root, model commits, input digest, executor and
resources, with memory fit explicitly unqualified. Model preparation/qualification remains a prerequisite. The controller host/process and metadata root are also bound. Submission checks its external digest, and the queued child checks
both the recorded plan and its inputs again before loading weights. The seed is
canonical decimal text in this protocol, preserving UInt64 values through clients
whose generic JSON number representation is a double.

Jobs are durable and can be observed after the submitting terminal or sheet closes.
Keep the original endpoint and job ID. Streaming and client timeouts stop watching;
explicit cancellation stops execution. Batteries retain their output directory
before generation begins, including when a child is interrupted. Stability remains
a diagnostic under `diagnostics/`, and a standalone battery remains its own run.
Neither is relabelled as a study stage, qualification, or format-2 control.

## Public paths

| Action | Mac CLI | Python client | HTTP / app |
|---|---|---|---|
| Prepare/review a sourced site | `cluster sites guide`, `cluster sites review <draft.json>` | Uses a runner URL; site registry is a Mac workbench capability | Mac `GET /api/cluster/sites/guide`, `POST /api/cluster/sites/review`; setup wizard From documentation |
| Accept the reviewed companion | `cluster sites accept <draft.json> --draft-sha256 <digest>` | Same explicit Mac workbench restriction | Mac `POST /api/cluster/sites/accept`; Import reviewed profile |
| Plan a standalone diagnostic | `remote science-plan <request.json> --site <id>` | `runner science-plan <request.json> --runner <url>` | Python `POST /api/science/plan`; Server Jobs form |
| Submit that plan | `remote science-submit <request.json> --plan-sha256 <digest> --site <id>` | `runner science-submit <request.json> --plan-sha256 <digest> --runner <url>` | Python `POST /api/science/submit`; Submit reviewed diagnostic |
| Inspect/log/cancel | Existing `remote jobs`, `logs`, `cancel` | Existing `runner jobs [<job-id>] [--cancel]`, `runner logs` | Existing job API; Server Jobs and captured diagnostic sheet |
| Resume eligible checkpoints | Existing `remote resubmit <job-id>` | `runner resubmit <job-id> [--walltime <hh:mm:ss>]` | Existing resubmit API and Resume button |
| Reconcile retained child records and merges | `remote reconcile` | `runner reconcile` | Existing `POST /api/jobs/reconcile`; Reconcile jobs button |
| Review controller ownership | `remote recovery <job-id>` | `runner recovery <job-id>` | `GET /api/jobs/{job_id}/recovery`; Recovery review sheet |
| Explicitly recover an orphan | `remote recover <job-id> --review-token <value> --reason <text> --confirm-owner-exited` | Same flags under `runner recover` | `POST /api/jobs/{job_id}/recover`; existing ownership gate, with operator attestation |

All client commands accept `--json`. The new Mac remote commands use a saved site
or URL and resolve credentials from the Keychain; they accept no bearer-token flag.
Python uses the existing runner token-file mechanism. `--review-token` is a public
concurrency digest from the ownership report, not an authentication credential.

Mac profile review/accept bodies carry exact UTF-8 companion text as `draftText`;
acceptance also requires `draftSHA256`. Scientific submission carries
`{"request": <request object>, "planSHA256": "<review digest>"}`. The scientific
request itself has exactly `operation` and `parameters`:

```json
{
  "operation": "stability",
  "parameters": {
    "experiment": "example",
    "concept": "signal",
    "resamples": 32,
    "fraction": 0.5,
    "seed": "0",
    "orderShuffles": 8
  }
}
```

Battery parameters are `batteryFile` (runner-relative path), `agents` (array of
existing baseline/condition/artifact references), and optional `modelID`,
`revision`, `alphaUnits`. Both operations accept optional `dtype` and `device`.
Every remote model revision must be an immutable 40-character commit. Unknown
parameters refuse. Resources come from the runner's declared local/Slurm profile;
this is not an arbitrary-command submission endpoint.

## Boundaries and recovery semantics

- Inputs must already be staged under the runner workspace through its permitted
  transport. This slice does not add a standalone input bundle or automatic
  upload/import lifecycle. Local authoring remains authoritative.
- These two diagnostics have **no checkpoint resume**. Existing study jobs retain
  their checkpoint/resubmission gates. Reconciliation is not recomputation.
- A Slurm submission intention is stored before the one scheduler call. A timeout
  leaves an attention state and its unique `schedulerSubmissionName`; clients must
  inspect it before retrying. Child records can subsequently recover its allocation
  ID and output through reconciliation. No automatic duplicate submission occurs.
- Controller recovery records an explicit assertion about owner exit. It uses the
  existing external review token and rejects live or stale ownership. An inability
  to establish controller liveness is not permission to recover it.
- The profile companion archive is private configuration evidence outside scientific
  manifests. No frozen study bytes or existing run data are rewritten. A filesystem
  failure after evidence retention may leave review evidence without an accepted
  profile; the archive itself makes no claim of connectivity or acceptance.
- Acceptance does not connect, deploy, allocate, download or delete. Continue through
  the existing preview/authentication/bootstrap/connect/qualification owners, with
  their existing permissions. Managed cleanup remains the next separate slice.

## Review feedback and verification

The earlier review's changelog omission was fixed on main before this branch.
Main also contains the independent final-norm float32 hardening in `73bfbd7`;
this branch does not alter it. The requested catalog follow-up is implemented:
resource and client-reference gates now compare every non-null engine command
against the engine's declared verb tables/dispatch, including family help paths.
Negative controls reject invented verbs and families.

`scripts/ci/audit-stability-preflight.py` proves the extracted admission block and
numerical/output tail are unchanged against the base. The battery owner's only
execution addition is an optional output-created callback immediately after its
existing directory mint, before model loading, so interrupted output can be located.
Tests cover real diagnostic arithmetic with fake model capture, queue-time input
and recorded-plan drift, exact large seeds, controller role refusal, durable Slurm
intention/uncertainty, local child interruption, record reconciliation, public API
shapes, token-path boundaries, and the Mac review/accept path.

Final suite results and gate status are recorded in
[validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md).
Fixture tests and successful builds do not establish live model, scheduler,
site-policy, or deployed-app qualification. No live cluster operation, model
installation, app installation, or real study was performed for this slice.

## Maintainer handoff

Read the complete diff, this document, and the updated operation matrix. Run both
full suites with Xcode beta/Metal and external build scratch, plus the normal
reference/resource, boundary, bridge and hygiene gates and the new AST audit.
Resolve findings with the user; only the maintainer's reviewing agent lands the
branch, using the established fast-forward process when main remains an ancestor.
The app and engine need builds from the landed source to expose the new interfaces;
this branch does not perform those deployments.

After audit, the next product slice is verified remote output custody and bounded
cleanup, followed by live end-to-end qualification of the researcher journeys.
Keep those tasks distinct from execution status: a finished job is neither proof
that evidence is safely local nor permission to remove its remote files.
