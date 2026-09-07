# Managed scientific methods and shared authoring — audit handoff

This continuation is on `codex/remote-custody-cleanup`, after `e74a34a`, based on
main `6a94a8f`. The complete unlanded range includes the preceding diagnostic
custody/cleanup slice. Read this alongside
[that handoff](REMOTE-CUSTODY-CLEANUP-HANDOFF.md). No main merge or deployment is
performed here. The maintainer's designated reviewing/integration agent, through
the researcher, owns independent diff review, both full suites, AST checks and
any eventual fast-forward landing. Recheck current main before landing.

## What changes for a researcher

A researcher can choose a named scientific operation in Research methods, answer
its purpose/claim/control/selection questions, choose actual input files and
review the resulting request. The form shows method-specific fields, explanations
and defaults. Additional engine settings remain available as explicit JSON;
they cannot silently override form answers. File pins come from the selected
bytes. Unresolved required decisions and missing inputs refuse publication.

The app, both agent CLIs and the workbench HTTP interface call one portable
Python authoring owner. The shared source is
`WorkspaceSeed/prompts/method-guides/workflows.json`; the resource gate checks its
Python package and compiled Swift copies, operation coverage and category
agreement. The separate method guides retain coworker author and independent
review prompts. This improves agreement about what the researcher authorized;
it does not replace the researcher's scientific judgment.

`science draft` is read-only. `science publish` rechecks the external review hash
and atomically creates a new `requests/<name>/` containing `request.json` and
`review.json`. Existing destinations refuse. Nothing is added to content-hashed
study manifests. Publication does not execute, freeze or scientifically qualify
anything. The UI displays the exact serialized request; integer seeds stay text
in form answers and original advanced JSON until Python writes them as integers.

From that published request the app opens the existing input review, packaging,
permitted transfer, staging, execution planning and evidence workflow. CLI agents
use those same named operations. Native MLX numerical equivalence is not claimed:
these additional calculations execute through their existing Python owners.

## Public operations and authority

| Capability | Public path | Meaning |
|---|---|---|
| Method instructions | `science interview <operation> --json` | Same operation schema as the app |
| Review answers | `science draft <operation> --answers <file> --json` | Request, rationale, captured input bytes and external review hash |
| Publish request | `science publish <operation> --answers <file> --destination requests/<name> --plan-sha256 <hash> --json` | Create-only request and review together |
| Execute | Existing `science input-plan/package`, runner/remote stage and plan/submit | Same closed request, isolated input copy and durable job machinery |
| Inspect SAE roster or qualification | `science sae-check <path>` / `sae-show <path>` | Offline inspection; warnings remain visible |
| Review and pin SAE roster | `science sae-pin-plan <path> --experiment <draft>` then `sae-pin` with `--plan-sha256` | External roster/manifest byte precondition and existing draft-only pin owner |
| Coordinate campaign | `runner science-call optvec-campaign --action post-science-campaign` (Mac `remote`) | Exact route parameters and reviewed action, described below |

Use `steerlab` for the Python client and `steerlab-cli` for the Mac client. Add
`--json` and the appropriate workspace/server/site arguments. Workspace HTTP
uses `POST /api/science/workspace/{action}` on both workbench implementations;
runner role admission still refuses local authorship. Draft/publish bodies take
`workspaceRoot`, `operation`, `answersText` (original JSON text); publish adds
`destination` and `planSHA256`. SAE actions use `workspaceRoot`, `path`, plus
`experiment` for pin review/apply and `planSHA256` for apply.

The thirteen managed operations are `rescore-style`, `sae-family-report`,
`sae-qualification-record`, `optvec-train`, `optvec-eval`, `optvec-geometry`,
`optvec-fracture`, `optvec-interpret`, `optvec-family`, `optvec-gradient`,
`optvec-gradient-mint`, `optvec-campaign`, and **`jspace`**. Each request is
`{"operation": "…", "parameters": {"config": {…}}}`. Configs are validated by
the original owners in a root-scoped process before execution planning; model
weights are not loaded during that check. Effective configs include decoded
defaults. GPU operations require explicit model IDs and immutable revisions.
Campaign plans enumerate every model/revision introduced by grid overrides.
CPU operations can execute locally on a controller without unloading its model
registry; GPU work on a controller still requires Slurm.

The input closure includes referenced vector tensors and sidecars, data files,
readout/gradient/report directories, lens records and converted tensors. Optional
fracture item-index files and family-report qualification files are included when
they affect the existing owner's reading. Referenced qualification evidence must
exist in the managed workspace. Paths must remain within that workspace and
ordinary; hash drift, symlinks and unresolved inputs refuse. Failed managed jobs
retain a partial marker and output-root location in their durable result; their
partial output directories are not guessed or attributed across concurrent jobs.

## J-space is separate from OptVec

J-space analyzes propagated residual differences through a J-lens. It does not
optimize a vector, and a readout is not evidence of an output-level behavioral
effect. Its catalog category, interview and public operation are independent.
The historical engine spelling `optvec jspace` and run type remain unchanged.
The current batch owner requires OptVec metadata for the intervention layer and
absolute dose; accepting other artifact types would be a separate scientific
admission change, not a namespace rename. This restriction is visible in the
form and guidance.

The new input closure exposed an absolute-path problem in imported J-lens
records. A narrow resolver now recognizes the importer's canonical
`runs/jlens-lenses/<id>/jacobians.safetensors` location in a relocated workspace.
Lens records are not rewritten. Managed input review verifies the declared
converted hash and containment; qualification's existing once-per-readout hash
check and layer loading use the same resolved file. Noncanonical external files
remain unavailable for managed transport unless made workspace-relative through
a legitimate import. The test corrupts the original tensor after copying and
proves that the relocated lens loads and verifies its unchanged captured bytes.

## Campaigns are multiple jobs

The initial CPU job materializes only. Cell configs preserve the existing
campaign grid and scientific identities; scheduler scripts invoke a small
root-scoped child that rechecks configuration and input hashes before calling
`optvec_train.train`. Each cell writes new numerical outputs beneath its own
campaign directory. It writes the completion marker only after successful
training and result publication. Completed cells cannot be rerun in place.

The controller route is `POST /api/science/campaign/{job_id}/{action}`. Actions
`status` and `plan` take `{}`; `submit` and `cancel` require
`{"planSHA256":"…","confirmAction":true}`. The review binds static file pins,
current campaign state and observed cell status. Stale review refuses. A
serialized action invokes the existing campaign scheduler's one-cycle queue
top-up, retry budget and uncertain-submission adoption. There is no unattended
submission loop. Cancel requests resolve unrecorded IDs by exact per-cell scheduler name before
mutation, and report individual failures without hiding accepted cancellations.
Acceptance is not proof of termination. Read status before any further action. Existing generic
job cancellation does not cancel a completed materialization's independent cells.

`science-export/fetch` captures the campaign and nested outputs only after all
cells completed and scheduler termination is established. Partial campaigns stay
on the runner. Automatic cleanup is **not** extended to trained vectors, lens
readouts, qualification artifacts or campaign directories: the existing policy
covers only isolated successful battery and stability output copies. This avoids
silently expanding a site's removal authorization.

## Implementation ownership and review focus

- `experiment/managed_methods.py` names the fixed scientific owners; no arbitrary
  module/function name comes from a request.
- `experiment/managed_inputs.py` owns the portable input closure;
  `api/managed_validation.py` isolates engine config imports from the portable
  client and from the shared server process.
- `experiment/method_authoring.py` owns interview resolution and create-only
  request publication; `experiment/sae_authoring.py` owns offline roster adapters.
- `api/scientific_execution.py` extends the existing durable local/Slurm path.
  `managed_campaign.py` owns controller admission and bounded subprocess calls;
  `managed_campaign_engine.py` owns campaign adaptation and queued cell checks.
- `MethodAuthoringSheet`, `SAERosterSheet` and `DiagnosticLifecycleSheet` provide
  forms and coordination over those same owners with captured workspace/client
  targets. The generic action editor remains an advanced interface.

Read the actual diff, especially input-closure completeness, relocated lenses,
state preconditions, partial evidence, queued cell drift and export eligibility.
No numerical algorithm or scientific threshold is intentionally changed.
`scripts/ci/audit-managed-scientific-owners.py` compares eleven original owner ASTs
against `6a94a8f` and separately proves that three J-lens consumers differ only by
the declared path resolver substitution/import. Negative controls must fail. The historical `audit-python-boundaries.py` is
explicitly scoped to the original public-name migration commit; running it on
this later feature work reports expected AST differences and is not a gate for
this slice. Use the scoped audit above for the actual unchanged-body claim.
There is no new compatibility facade, alias convention or bridge.

## Validation and remaining qualification

Final full Python suite: **6,122 passed, 9 skipped, 8 warnings**. Final full
serial Xcode beta suite: **277 SteeringKit + 4,594 ExperimentKit passed**.
The separate app build, shared-resource/reference/bridge gates and scoped
scientific-owner AST audit pass. Commands, logs and limits are recorded in
[the validation ledger](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md).
The test scope includes actual CPU geometry/style/qualification owners, shared
request publication and refusal, UInt64 preservation through the Mac adapter,
real converted-lens relocation, campaign scheduler fixtures and queued input
checks. Scheduler fixtures are not a live allocation or accounting test.

The next stage is independent audit followed by complete researcher journeys:
agent and app authoring, real model/lens/SAE applicability, live local and Slurm
execution, interruption and uncertain submission, permitted external-only
transfer, offline custody and site-policy refusals. Test Linux locking and
publication semantics on Linux, and perform interactive app QA. No live cluster,
large model, scientific outcome, deployment or 1.0 qualification is claimed here.
Scope restrictions above are part of the product contract, not hidden omissions.
