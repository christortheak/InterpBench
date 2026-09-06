# Model preparation and study authoring equivalence

Implementation and independent-review handoff, 2026-09-06.
Branch `codex/model-study-equivalence` starts from landed main `55606d9`.
The preceding design/interview audit is included. This branch does not install
the app, deploy an engine, download a model, run a study or merge itself.

## Researcher outcome

An external agent can discover an existing agent, attach its reviewed artifact,
author a semantic panel, cast its seats, declare the execution pipeline and
prepare the selected engine's model cache. The resulting ordinary study can be
opened and edited by the Mac app. The app's panel and pipeline authorship uses
the same scientific compilation and declaration owners as the public adapters.

This closes routine preparation and assembly gaps. It does not replace the
researcher's decisions about interventions, controls, interaction structure or
outcomes. Study verification, freeze gates and substrate-specific scientific
qualification still determine whether a draft is ready to execute.

## Model preparation

| Target | Plan | Install | Observe/cancel |
|---|---|---|---|
| Mac local MLX cache | `steerlab-cli model plan <model> [--revision <ref>]` | `steerlab-cli model install <model> [--revision <ref>]` | Foreground completion; the app's existing local-model HTTP status/cancel operations address its observed request ID |
| Python engine, from Python | `steerlab model plan <model> --runner <url> [--revision <ref>]` | `steerlab model install <model> --runner <url> --plan-sha256 <digest> [--revision <ref>]` | `steerlab model status/cancel <job-id> --runner <original-url>`; `runner logs <job-id>` for logs |
| Python engine, from Mac | `steerlab-cli remote model-plan <model> --site <id> [--revision <ref>]` | `steerlab-cli remote model-install <model> --site <id> --plan-sha256 <digest> [--revision <ref>]` | `remote model-status/model-cancel <job-id> --site <original-id>`; `remote logs` for logs |

All commands support `--json`. Python uses the existing `--token-file`, timeout
and CA-bundle options. The new Mac verbs resolve credentials through the
Keychain, using the site's identity or an explicit URL's host/port key; they
accept no raw-token argument. A Python engine running locally uses the same `--runner` workflow as
a remote engine; the lightweight client does not acquire GPU dependencies to
download weights itself. Local MLX and Python HF caches remain distinct.

`GET /api/models/plan?model=<id>&revision=<ref>` is a runner-owned read. It
reports the actual cache root, requested model/revision, deployment profile,
compute-egress declaration and external `planSHA256`. It does not contact the
hub, look up credentials or load weights. Cache-file observations and unknown
size/memory/credential facts are separate from the reviewed request identity.
Presence in the cache is not proof of complete usable weights or scientific
qualification.

`POST /api/models/install` retains the existing durable-job implementation.
The new clients supply `planSHA256`; the engine checks it before submission.
Existing app/API callers can still use the explicit install request without
that optional transport precondition. Every caller is subject to the same
declared egress restriction. Installation remains the operation that permits
an online child; `HF_HUB_OFFLINE=1` continues to govern execution and is not
misinterpreted as proof that a model is cached.

The profile renderers now carry a known `computeEgress` value into
`STEERLAB_COMPUTE_EGRESS`. A declared `no` refuses installation at the service
boundary and points to transfer-host staging. An undeclared value remains
`unknown`, preserving the existing explicit-install policy; it is not a claim
that access was checked. Profile rematerialization is needed to deliver this
new declaration to an existing deployment. No existing configuration is edited
automatically by this implementation.

The app offers Plan beside its server Install controls. Replies are associated
with the captured connection; changing the selected connection prevents a
delayed plan or install response from becoming the new connection's state.
An install obtains and submits a plan on that same captured client. Completion
and cancellation remain ordinary Compute jobs. Cancelling observation does not
cancel the job, and partial cache files are retained. A submission timeout is
ambiguous: inspect jobs on the original endpoint before considering a retry.

## Study authorship

Use `steerlab` for Python and `steerlab-cli` for Mac. Python takes `--root`;
the Mac uses the selected workspace or `STEERLAB_WORKSPACE`.

- **Agents:** `agent list` and `agent inspect <runs-relative-path>` discover
  native and imported saved agents. `experiment attach-agent <study> --artifact
  <path> --artifact-sha256 <digest> --manifest-sha256 <digest>` replaces an arm
  with the same name/path and embeds the exact reviewed artifact. It requires
  a matching base model and never edits the artifact or its evidence.
- **Panel inputs:** author semantic JSON containing roles, materials, turns,
  routing, contracts and outcomes. Model/sampling bindings belong in the
  study; seat agents belong in the casting. Python `panel check <file>` validates
  the proposed semantic input and returns its `fileSHA256`. `panel import <file>
  --file-sha256 <digest>` on either client publishes a new immutable version.
  `panel inspect <workspace-relative-path>` returns its exact review and seats.
  The Mac's existing `panel check` remains the runnable-scenario check; its
  HTTP semantic check accepts source text as described below.
- **Casting:** `panel compile <workspace-relative-path> --experiment <study>
  --casting <file> --file-sha256 <panel-digest> --manifest-sha256 <study-digest>`
  names every seat explicitly. The file is `{"seats":{"first":null,"second":
  {"artifactPath":"runs/.../model-variant.json","artifactFileSHA256":"..."}}}`.
  Null means baseline. Extra/missing seats, foreign-model agents, changed
  reviewed bytes and frozen/stale studies refuse before compilation.
- **Pipeline:** `experiment set-pipeline <study> --file <json-file>
  --manifest-sha256 <digest>` replaces the complete declaration. The file holds
  the pipeline object, or JSON `null` to remove it. Validation uses the existing
  stage/gate rules. Unknown declarations are refused; no stage is submitted.
- **Casting expansion:** `design expand <design> --casting <file> --mode
  permutations|composition --file-sha256 <design-digest>` returns `result.batch`
  ready for `design batch --rows`. Permutations accepts an exact seat casting
  and emits distinct re-seatings. Composition accepts `{"agents":[<reviewed
  agent reference>]}` and emits all-baseline, each solo-treated seat, then
  all-treated. One-seat duplicates are suppressed. Expansion is read-only,
  bounded to 64 seats and 4,096 rows, and makes no study or submission.

The Mac retains its existing direct `panel compile --seat ...` form. That form
captures its inputs when invoked; unnamed seats mean baseline, as documented
before this branch. It now delegates publication to the shared reviewed-draft
owner. Agents carrying an earlier review should use the casting-file form;
mixing that form with direct seat/model-setting flags is refused.

The app's changed semantic panel saves create new immutable versions instead
of rewriting a file an older study may pin. An unchanged save reuses its
selected record. App seat saving checks the retained study review before
publishing compilation. Agent catalog paths normalize the operating system's
workspace-root aliases before becoming relative pins. Identical panel imports
report `changed: false`. Pipeline editing continues through the existing shared
owner; the raw transport adapter preserves the validated declaration rather
than lossy parsing it through UI fields.

## HTTP operations and authority

The Mac workbench exposes `POST /api/experiment/pipeline`, `/api/panel/inspect`,
`/api/panel/check`, `/api/panel/import`, `/api/panel/compile`, and
`/api/design/expand` over the same owners. Existing agent/list/inspect/attach
routes remain in use. These are authoring operations in the local workbench;
they do not move authoring authority to a cluster runner.

Every new study/panel HTTP body identifies `workspaceRoot`. Pipeline also takes
`name`, `manifestFileSHA256`, and `document` (an object or explicit null). Panel
inspection takes `path`. Semantic check/import takes `source`, a string holding
the original JSON text; import also takes the check's `fileSHA256`. Compilation
takes `name`, `manifestFileSHA256`, `path`, `fileSHA256` and `casting`. Design
expansion takes the existing design identity/review fields, `casting` and
`mode`. Unknown operation-specific fields refuse rather than disappearing.

## Review and validation map

- Python client owners: `study_agents`, `study_panels`, `study_pipeline`,
  `design_expansion`; thin `authoring_commands` and `model_commands` adapters.
- Model service: `api/model_preparation.py`, the existing install/job owners,
  both site environment renderers and their shared fixture outputs.
- Swift owners: `StudyPanelAuthoring`, `StudyPipelineAuthoring`,
  `StudyDesignExpansion`, `RemoteModelPreparation` and its presentation controller.
- The preceding audit's **N1** is covered by a real Python-cast agent condition
  decoded and re-encoded by the Mac, comparing the whole condition and retaining
  source artifact bytes. Standalone Python attachment is compared as well.
- Additional journeys cover Python panel/pipeline authorship, Mac HTTP edits,
  Python rereads and equivalent casting expansion rows. Failure tests cover
  stale reviews, malformed fields, egress denial and request/job identity.

Final suites pass: **6,036 Python tests** (9 skipped), **277 SteeringKit** and
**4,581 ExperimentKit** tests. Commands and logs are recorded in the
[validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md). Runtime scientific bodies are not moved in this
slice; authoring and policy changes are intentional semantic changes. Retain
the existing AST audits, bridge gates and generated-reference/contract checks.

The audit's **N2** remains a deployment requirement: rebuild/install the Mac app
and CLI from the integrated revision before opening newly Python-minted drafts
there. Deploy the engine and rematerialize approved profile configuration when
enabling the new remote preparation workflow. This branch performs none of
those deployment actions.

The maintainer, through the designated reviewing/integration agent, must read
the actual diff and approve landing after both full suites and applicable gates.
Recheck main ancestry and integrate any later fixes first. No site names,
study-case vocabulary or secrets belong in public files or commits; frozen
studies and runs remain immutable. External review digests stay outside the
scientific manifest's content-hashed fields.

Next work remains method guidance and advanced-operation coverage, the bounded
cluster onboarding/remote lifecycle work, and interactive/live scientific
qualification. Passing these source-level journeys does not qualify numerical
behavior, a live cluster, a real model download or interactive UI operation.
