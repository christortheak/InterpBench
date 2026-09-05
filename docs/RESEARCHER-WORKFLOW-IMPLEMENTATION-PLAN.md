# InterpBench: one research workflow across agents, CLI, API, and SwiftUI

**Review date:** 2026-09-05

**Document revision:** 4 — implementation handoff with identity, hygiene, and review constraints

**Reviewed revision:** `bfd13a5` on `main`

**Reviewed checkout:** InterpBench repository

**Scope:** Product workflow, agent instructions, CLI/API exposure, SwiftUI parity, cluster operations, scientific guidance, and implementation acceptance criteria.

This document consolidates a read-only source and documentation review. It does not report live cluster testing, interactive UI testing, or a new numerical validation of the scientific implementations. The findings describe the reviewed revision; check subsequent changes before assigning fixes. No code was changed for this assessment.

Revision 2 incorporated the independent team's **“Surface parity audit: agent contracts, CLIs, HTTP API, and the app”**, dated 2026-09-05 and reviewing the same source revision. Its useful additions are the incomplete remote command recipe, missing convenient agent operations for templates and model preparation, and the need to enforce workspace authority. Selected conclusions were checked against source before incorporation. This document does not adopt raw command-mention or endpoint-literal counts as measures of functional coverage.

Revision 3 incorporated the teams' subsequent agreement on runner/workbench enforcement, coverage classifications, and complementary use of the reviews. It added an implementation plan tied to **clarity, researcher accessibility, agent driveability, and equivalence between interfaces**. The source findings remain observations at `bfd13a5`; the plan and target behaviors are proposals, not claims that new functionality has landed.

Revision 4 adds explicit handoff constraints for stale-write protection, public-repository hygiene, scientific-artifact immutability, and maintainer-controlled review. The companion synthesis has now been read from the repository and is included beside this document: [Surface and workflow synthesis, revision 2](AGENT-SURFACE-AND-WORKFLOW-2026-09-05.md). Its release-scope recommendations are distinct from authorization to implement or defer work.

**Maintainer scope decision for this handoff:** WP-5 and WP-6 remain in scope. Remove estimation requests; do not turn them into preliminary estimation tasks or inherit a deferral from an older audit. The plan defines bounded implementations of these packages. This scope decision does not itself authorize starting implementation or landing changes.

**Reading guide:** Sections 1–2 explain the intended experience and what will change. Section 11 is the implementation plan; section 12 defines the acceptance journeys. Sections 4–10 retain the detailed findings and evidence for implementing agents. Section 13 explains coordination, validation, and handoff.

The independent audit reports an installed app at `0.9.5+b6949ffd`, older than the reviewed source. Checks against that installed binary must be distinguished from checks against a binary built from `bfd13a5`. Report the actual CLI/app/server build identities in subsequent validation.

Source references below use repository-relative paths and one-based line numbers at the reviewed revision, so the document remains usable outside the reviewer's filesystem. Locate the named symbol if lines have moved.

## 1. The intended product

The researcher should be able to work with an agent at a conceptual level: state the research question, discuss the construct and study design, make substantive decisions, and inspect the evidence without mastering extraction, steering, orchestration, or cluster administration.

The agent should be able to:

- Discover the available extraction and intervention techniques, their appropriate uses, and their limitations.
- Obtain the exact dataset formats and generation prompts for each technique, including instructions for independent coworking authors and reviewers.
- Author, verify, freeze, submit, monitor, and analyze studies.
- Conduct steering tests and multi-agent studies through supported operations.
- Turn institutional cluster documentation into a proposed site configuration, asking the researcher about unresolved facts.
- Execute remotely through a workflow comparable to local execution, including queue monitoring, reconnection, resumption, evidence download, and cleanup.
- Keep the local workspace authoritative and remote filesystems consistent with site policies.

The CLI and API should expose the necessary operations. SwiftUI should provide another interface to the same authoring, execution, and evidence services.

This does not require embedding an LLM in the app. A clear external-agent contract and shared application services can support the vision. Nor does minimal friction mean hiding scientific decisions: the researcher should make those decisions in conceptual terms, while the agent handles their implementation.

### 1.1 What the researcher should experience

A researcher can say, for example, “I want to test whether steering this model toward greater caution changes its decisions, while preserving general ability.” The agent explains suitable methods and asks about the construct, task, controls, and claims. It then handles dataset formats, pins, study assembly, model preparation, and execution through supported operations.

The researcher can open the resulting study in SwiftUI, inspect or revise the draft, and continue there. An agent can resume work from that same saved state. Neither has to recreate the other's setup. If the selected machine cannot execute the design, both interfaces explain the same restriction and present supported alternatives.

The researcher still chooses the substantive design and accepts consequential scientific tradeoffs. The system removes unnecessary technical decisions: which private helper to call, how to calculate a hash, which server owns a job, or where to copy a finished result.

### 1.2 What equivalence means

**The same declared operation against the same identified study and supported execution context must have equivalent meaning, validation, artifact effects, and recovery behavior regardless of how it is invoked.**

| Dimension | Required equivalence | Presentation may differ |
|---|---|---|
| Intent and target | Workspace, study/revision, site, and operation are explicit | A selected row can supply the target in SwiftUI; CLI/API requests name it |
| Authoring | The same declaration produces the same intended study semantics and pins | A form, a command, or a structured request can express it |
| Preflight | The same applicability rules, scientific gates, and policy checks apply | The app explains them visually; the agent gets structured reasons and repairs |
| Execution | Accepted requests use the same applicable execution service and retain durable job identity | The app shows progress; the agent observes structured status/logs |
| Evidence | Outputs retain the same provenance and are imported into the originating workspace | UI reports and agent-readable artifacts offer different views of the same evidence |
| Recovery | Stop watching, cancel, resume, retry, and import have consistent meanings | Buttons and commands need not have identical layouts or wording |

This is equivalence of supported behavior, not a demand for identical command names, every UI interaction to have its own CLI verb, or byte-identical MLX/CUDA computations. Preserve existing substrate-specific qualification, canonicalization, and artifact contracts. Purely presentational state need not become a public operation; every research-relevant action should be reachable programmatically and represented in the human workflow.

### 1.3 Architecture that supports the experience

Use the existing focused services as the starting point. SwiftUI, CLI handlers, and HTTP handlers should translate user intent into explicit requests and present results; they should not own separate scientific or authoring rules. An external agent drives those public interfaces and can discover their contracts.

Within each runtime, adapters should call the same operation owner. Across Swift and Python, use explicit interoperability contracts, producer-generated fixtures, and behavioral checks. The plan does not require forcing both languages into one implementation or creating a new monolithic dispatcher.

The authority model is authoring client versus execution hardware: the local workspace owns study definitions; a runner receives pinned inputs and produces artifacts; a workbench can intentionally serve the authored workspace. Credentials, service role, and current UI selection do not silently change that ownership.

## 2. Overall assessment

**InterpBench has much of the machinery this vision requires. The remaining gap is making it a dependable, coherent workflow that an agent can operate without reconstructing the system's conventions.**

An experienced agent with the checkout can already accomplish a great deal. A researcher describing a study conceptually still risks encountering technical assembly steps, inconsistent interfaces, and operational bugs.

| Capability | Assessment |
|---|---|
| Agent onboarding and command discovery | Strong foundation; advanced coverage remains uneven |
| Extraction methods and dataset authoring | Substantial implementation and documentation; incomplete method-specific guidance |
| Study authorship, freezing, and submission | Well developed; some workflows still require manual assembly |
| Steering tests and scientific controls | Extensive safeguards; production qualification remains unfinished |
| Multi-agent studies | Real authoring, casting, and execution support; coauthoring guidance is comparatively thin |
| Cluster configuration from institutional documentation | Rich configuration machinery; the document-to-profile workflow is missing |
| Local/remote execution and recovery | Significant infrastructure, with correctness and policy gaps |
| Workspace authority | Runner/workbench roles are documented and inventoried; their separation is not enforced at runtime |
| SwiftUI and agent/API parity | Broad overlap, but not yet interchangeable interfaces |

The most valuable next work is completing and proving the connections between capabilities, rather than expanding the technique inventory.

### 2.1 What will change and why it matters

| Current friction | Planned change | Contribution to the vision | Evidence of completion |
|---|---|---|---|
| The agent assembles knowledge from scattered references | Discoverable operation and method guides, complete recipes, and maintained capability coverage | **Clarity:** the available choices, prerequisites, and next steps are explicit | A shipped-instructions journey succeeds without reading private implementation code |
| The researcher finishes routine attachment and pinning by hand | Agent-accessible template, study-pack, attachment, and model-preparation operations | **Accessibility:** the researcher discusses the study rather than its plumbing | Agent assembles a verified draft that opens correctly in SwiftUI |
| UI, CLI, and API paths can apply different rules | Shared operation owners, explicit requests, and equivalent behavior checks | **Equivalence:** switching interfaces does not change what the study means | The same intent passes or refuses consistently across supported surfaces |
| Switching workspace or server can redirect work | Captured operation context and durable job/evidence identity | **Trust and driveability:** concurrent interaction and reconnection are dependable | Context-switch and restart tests preserve ownership |
| Runner deployments expose independent authoring | Enforced runner/workbench roles | **Clarity:** inputs and results have an unambiguous authority | Runner refuses authoring while workbench and derived computation still work |
| Cluster onboarding begins with a finished profile | A bounded documentation-to-profile authoring workflow using existing validation and previews | **Accessibility:** users provide institutional facts rather than learn the schema | Agent produces a source-supported profile and asks only unresolved questions |
| Remote work requires manual follow-up and filesystem judgment | Managed observation, resume, merge, verified import, and bounded cleanup | **Driveability:** remote execution forms a complete supported workflow | Interrupted study returns verified evidence and supports policy-compliant cleanup |
| Successful execution can be mistaken for scientific support | Consistent claim language and discoverable qualification results | **Scientific clarity:** users understand what their evidence does and does not establish | Reports and agents distinguish software success, evidence admissibility, and substantive inference |

### 2.2 Proposed release boundary

This plan recommends a **bounded implementation of the stated vision** for the release that presents InterpBench as conceptually accessible and agent-drivable. It does not require a built-in conversational agent, automatic discovery of every cluster type, or general-purpose autonomous filesystem administration.

The bounded targets are: external-agent coauthoring through supported operations; document-to-profile assistance through a prompt, schema, source evidence, questions, and previews; and cleanup limited to understood artifacts with verified local custody and explicit policy rules. The maintainer has kept WP-5 and WP-6 in scope for this handoff and removed estimation requests. Do not treat these packages as deferred or create preliminary estimation work. Release dates and broader release commitments are not set by this document.

Correctness and authority defects come first. If a capability is explicitly deferred, record the release limitation, the supported fallback, the affected acceptance journey, and the narrower product claim. Do not mark a journey passed after quietly removing the part that was missing.

## 3. Foundations to preserve

Several existing features directly support the intended product:

- **Workspace agent contracts.** Generated instructions explain lifecycle ordering, file formats, immutability rules, and repairs.
- **Structured command results.** Core study-path commands provide envelopes, typed refusals, repair actions, and next steps.
- **Generated reference sections.** Declarative command tables support consistency checks. This protects the generated surface; it does not prove full command coverage, narrative accuracy, or behavioral parity.
- **Versioned dataset-generation prompts.** Prompts carry hashes and separate author/reviewer instructions.
- **Scientific provenance.** Studies pin inputs, distinguish exploratory work from frozen studies, and record controls and provenance.
- **Verified evidence imports.** Imports check hashes and protect immutable results.
- **Shared cluster setup.** CLI and SwiftUI use common lifecycle and preview machinery.
- **Existing service-role design.** A runner/workbench distinction and a route-ownership census already provide a foundation for enforcing workspace authority.
- **Composite execution.** The cross-platform `steerlab run` packages, checks capabilities, submits, waits, and imports verified evidence.
- **Conceptual UI assistance.** Study types, data readiness, coauthoring prompts, study-pack import, and the primary Run button already reduce the amount of technical assembly the researcher must perform.

References:

- `Server/steerlab_server/client_cli.py:20` — client/runner separation and composite execution.
- `Sources/SteerLabApp/ExperimentsPanelView.swift:35` — study interface and readiness integration.
- `Sources/ExperimentKit/AgentContract.swift` — generated workspace contract and maintenance.
- `Sources/ExperimentKit/ClusterSitePreview.swift:36` — shared cluster preview.
- `docs/PORTABILITY-CONTRACTS.md:1525` — existing service-role design and census.

## 4. Confirmed code-path problems

These findings are supported by source inspection. Their live manifestations were not reproduced during this review. They should be distinguished from the product gaps in later sections.

### OPS-01 — Observing jobs can interfere with running work

**Priority: High.**

`steerlab-server jobs list` constructs a `JobManager`. Its constructor defaults to sweeping orphaned jobs. The sweep marks local running/pending jobs failed and can cancel children of a pending Slurm fan-out.

A second process opening the job database does not establish that the first process's jobs are orphaned. An agent following the reasonable instruction “check the queue” can therefore disturb work owned by a running daemon.

**Required outcome:** Observation must be read-only. Recovery must use explicit process ownership or liveness evidence. A short age grace period alone would not establish ownership correctly.

**Acceptance criteria:** Listing jobs while another process owns local work or is submitting shards leaves all job states and scheduler allocations unchanged. Genuine owner death remains recoverable through a deliberate recovery path.

References:

- `Server/steerlab_server/cli.py:1357` — `_jobs` constructs a manager for listing.
- `Server/steerlab_server/api/jobs.py:636` — constructor defaults and startup behavior.
- `Server/steerlab_server/api/jobs.py:698` — `_sweep_orphans`.

### OPS-02 — UI job actions are not consistently bound to the originating server

**Priority: High.**

Active and recent study-job records carry an ID, verb, and study, but no originating server identity. Cancellation and resumption receive the current connection. Sweep cancellation explicitly resolves the currently selected server.

Starting a job on server A, switching to B, and cancelling can send the request to B. Usually this means the intended job remains active; an ID collision could target unrelated work.

**Required outcome:** A durable job reference carries its site/server identity and workspace context through submission, monitoring, cancellation, resumption, reconnection, and import.

**Acceptance criteria:** Switching servers or restarting the app does not change the target of actions on an existing job. If the originating server is unavailable, the action reports that fact and identifies the required reconnection.

References:

- `Sources/ExperimentKit/StudyRemoteJobController.swift:31` — recent and active job records.
- `Sources/ExperimentKit/ExperimentPanel.swift:185` — sweep cancellation.
- `Sources/ExperimentKit/StudyRemoteJobActions.swift` — cancellation and resumption.

### OPS-03 — Evidence imports can resolve the wrong workspace

**Priority: High.**

The auto-import service captures a `workspaceRoot`, but its actual importer resolves staging and destination directories through global `ExperimentStore` state. Manual import has the same underlying dependency.

Switching workspaces during a download can change where the subsequent import lands. The auto-import ledger can still belong to the originally captured workspace.

Hash verification establishes the integrity of the bytes, not that they were imported into the correct workspace.

**Required outcome:** Pass an explicit captured destination through download, verification, publication, revision adoption, and ledger updates. Packaging should receive the same treatment where it still consults global workspace state.

**Acceptance criteria:** A workspace switch at any asynchronous boundary cannot redirect an in-flight import or split its evidence and ledger across workspaces. An operation either completes in its captured context or stops with a clear status.

References:

- `Sources/ExperimentKit/EvidenceAutoImport.swift:502` — `performImport`.
- `Sources/ExperimentKit/ClusterClient.swift:4648` — `EvidenceBundleImporter`.
- `Sources/ExperimentKit/ClusterClient.swift:4781` — global destination resolution for sibling evidence.
- `Sources/ExperimentKit/ExperimentPanel.swift:1718` — manual job-evidence import.

### OPS-04 — The ordinary bundle path does not enforce advertised transfer policy

**Priority: High for policy-constrained sites.**

The server advertises `httpTransfer` and `externalTransferRequired` from the configured transfer method. The app displays the external-transfer designation, but ordinary bundle upload/download methods still use HTTP. The corresponding server endpoints do not check that policy.

Separately, cluster run import uses rsync. These are distinct transfer paths rather than one abstraction consistently governed by the site profile.

**Required outcome:** Transfer restrictions must affect actual execution. Route through a supported policy-compliant transport, or refuse with an actionable explanation. Apply the rule on both client and server boundaries where relevant.

**Acceptance criteria:** A site requiring external transfer cannot silently move bundles through the ordinary HTTP path. The selected transfer mechanism and applicable policy are visible in the operation result.

References:

- `Server/steerlab_server/api/profile.py:385` — transfer capability advertisement.
- `Sources/ExperimentKit/ClusterClient.swift:2753` — HTTP bundle upload.
- `Sources/ExperimentKit/ClusterClient.swift:4176` — HTTP artifact download.
- `Server/steerlab_server/api/routes.py:3911` — upload endpoint.
- `Server/steerlab_server/api/routes.py:3997` — download endpoint.
- `Sources/ExperimentKit/WorkspaceRunImport.swift:909` — separate SSH/rsync import implementation.

### OPS-05 — Submission has a consequential implicit default

**Priority: Medium.**

`study submit` defaults to `run` when `--verb` is omitted. The CLI reference records an actual sweep-as-run incident.

**Required outcome:** Require an explicit operation, or derive it through a higher-level operation whose intent is unambiguous. Do not turn an omitted study-stage argument into a GPU allocation for a different stage.

**Acceptance criteria:** Omitting the verb from the low-level submission command produces a usage refusal before staging or allocation.

References:

- `Server/steerlab_server/cli.py:1555` — default verb.
- `docs/CLI-REFERENCE.md:5137` — documented incident and gap.

### OPS-06 — Runner/workbench workspace authority is documented but unenforced

**Priority: High.**

The architecture already distinguishes a **runner**, which executes bundles against a disposable working root, from a **workbench**, which serves a live authored workspace. The route census classifies current HTTP operations as runner, workbench, or both. That classification lives beside the tests and governs no runtime request: the current FastAPI application serves both surfaces.

Consequently, a deployment being used as a runner can still expose workbench authoring operations. This permits independent changes to study inputs or declarations in what the local-authority workflow treats as a remote cache. An authoring-divergence report on import detects some consequences; it does not prevent the split in authority.

**Required outcome:** Finish the existing service-role separation. In the intended researcher workflow, the local authoring workspace owns study definitions and the cluster runner consumes pinned inputs and produces execution artifacts. Runner deployments should refuse independent workbench authoring operations with typed, actionable repairs. Preserve legitimate workbench use rather than globally deleting authoring functionality.

Do not confuse this with authentication. Mutating `/api` requests are already privileged by default, with a small explicit exception list for non-writing operations. Authentication establishes who may make a request; service-role and workspace-authority checks establish where that request may author a study. Adding privileged prefixes alone would not fix this finding.

Also distinguish authoritative inputs and declarations from derived artifacts. Computing a vector, training an adapter, fitting a reader, or writing evaluation evidence on a GPU can be legitimate execution. The remedy must not prohibit those outputs merely because they write files. Review the census's `both` routes individually: their current callers explain real migration dependencies.

**Acceptance criteria:** A runner rejects workbench authoring even for an authenticated caller, while supported bundle execution, model preparation, evidence production, and recovery continue to work. A workbench retains its intentional authoring operations. Import does not silently adopt divergent study definitions. Any future adoption operation creates or updates an explicitly identified draft under a named rule and preserves frozen evidence.

References:

- `docs/PORTABILITY-CONTRACTS.md:1525` — service roles and route census.
- `docs/PORTABILITY-CONTRACTS.md:1590` — census is not runtime enforcement.
- `Server/tests/route_roles.py` — current role classifications and rationale.
- `Server/tests/test_route_roles.py` — census completeness and consistency checks.
- `Server/steerlab_server/api/app.py:161` — mutating-by-default privilege classification.

### UI-01 — Imported pipeline results remain tied to server UI state

**Priority: Medium.**

`StudyPipelinesView` displays server pipelines and “Imported / local” pipelines. Its parent only renders it when the compute target is the server.

Evidence already brought home should remain visible when the researcher switches to local compute.

**Required outcome:** Local evidence visibility depends on its presence in the workspace, not on the active compute target or connection.

**Acceptance criteria:** Imported pipeline evidence remains accessible after disconnecting the server and selecting local compute.

References:

- `Sources/SteerLabApp/ExperimentsPanelView.swift:283` — conditional display.
- `Sources/SteerLabApp/StudyPipelinesView.swift` — server and local pipeline rows.

## 5. Method-specific agent guidance is incomplete

The extraction reference usefully distinguishes recipes, inputs, normalization, orientation, and limitations. The remaining problem is getting the correct knowledge into the agent's workflow at the right moment.

The workspace contract lists four extraction methods, then describes the inputs as positive and negative files and states that everything extraction reads is those two files. That description does not cover grand-mean or designated-reference extraction correctly.

The standardized authoring registry currently covers contrastive pairs, choice prompts, validation sets, reader pairs, and capability batteries. Additional generation templates exist elsewhere, but the pieces do not constitute one complete method-specific contract.

**Recommended outcome:** For each technique, expose a coherent, retrievable guide containing:

| Contract element | What the agent needs |
|---|---|
| Research purpose | Questions the technique can answer and limitations on interpretation |
| Inputs | Exact schemas, required files, naming, pairing, and rendering constraints |
| Dataset production | Coworking-author prompt and independent-reviewer prompt |
| Split discipline | Training, orientation/selection, validation, and final-evaluation roles |
| Operations | Commands/API calls for authoring, execution, checking, and repair |
| Capabilities | Supported engines, model restrictions, and qualification requirements |
| Outputs | Artifact locations, provenance, diagnostics, and permissible claims |

Cover readers, optimized vectors, SAE-derived interventions, fine-tuning, and J-space as distinct workflows. Do not describe every technique as a paired-extraction variant.

References:

- `docs/AGENTS-WORKSPACE-DRAFT.md:119` — extraction attachment instructions.
- `docs/EXTRACTION-RECIPES.md` — recipe reference.
- `Sources/ExperimentKit/AuthoringPrompts.swift:152` — standardized prompt kinds.
- `WorkspaceSeed/prompts/generation/` — additional generation templates.

## 6. Coauthoring still hands technical assembly back to the researcher

The app's “Copy LLM Prompt” already implements a valuable conceptual collaboration workflow: interview the researcher and produce a study pack.

However, the prompt tells the coworking model to leave concepts and saved-agent conditions empty, then instruct the researcher to attach them in the app. Confirmation policies and scenario selection also require subsequent UI actions.

Avoiding fabricated hashes is correct. The desired endpoint is for the agent to invoke the real attachment and pinning operations after authoring the files. “Never invent a hash” should lead to “call the operation that computes it.”

Multi-agent studies have substantial scenario, seat, visibility, and casting machinery. Their coauthoring prompt is comparatively brief and largely asks which existing scenario and agents to use. It needs equivalent guidance for designing interactions, authoring materials, choosing controls, and specifying transcript outcomes.

**Recommended outcome:** A coworking agent can produce the study materials and assemble a verified draft through supported operations. The researcher is asked about substantive choices and unresolved facts rather than routine technical attachment steps.

References:

- `Sources/ExperimentKit/StudyAuthoring.swift:486` — `StudyCoauthoring`.
- `Sources/SteerLabApp/StudyTypeSection.swift:82` — coauthoring prompt UI.
- `docs/AGENTS-WORKSPACE-DRAFT.md:698` — panel casting workflow.

## 7. CLI, API, and SwiftUI are not yet one operation contract

### 7.1 Classify the gap before assigning a fix

The surface audit usefully inventories agent contracts, three CLIs, the Python HTTP API, and the app. Preserve that breadth, and explicitly distinguish the app's Swift HTTP server from the Python server. An operation absent from one is not necessarily unavailable through the other.

| Coverage category | Meaning | Appropriate response |
|---|---|---|
| UI-only | No supported programmatic operation was found | Expose the existing service through an agent interface |
| API-accessible, missing CLI | An agent can already call the endpoint, but lacks a convenient command | Add a CLI operation and discovery instructions over the existing service |
| File-authored | The supported input is an authored file, not an imperative command | Publish the schema, author/reviewer prompts, validation, and attachment workflow |
| Reachable but poorly documented | The operation exists; the agent's instructions do not explain it | Improve the contract and executable examples |
| Intentionally restricted | The operation belongs to a particular engine or service role | Publish the restriction, discoverable capabilities, and routing repair |
| Unverified | Presence or equivalent behavior has not been checked | Inspect and exercise the workflow before declaring parity |

Model installation has a privileged Python HTTP endpoint, so it is **API-accessible with missing CLI coverage**, not literally app-only. Local playground chat has a Swift HTTP endpoint; its missing CLI operation is a distinct issue from API availability.

The documented differences between the Swift authoring CLI and the Python client are not automatically defects. A typed Swift setter and a client protocol-field operation can provide equivalent behavior. Assess their effects rather than requiring identical command spellings.

### 7.2 Complete useful agent operations

The core lifecycle has a good machine interface. Important gaps remain:

- **Model preparation:** provide a supported model-install operation that uses the existing preflight and policy checks and exposes progress and failure through the agent contract. This is a routine study prerequisite, not an optional UI convenience.
- **Study design templates:** provide discoverable operations to list, inspect, instantiate, and save designs through the existing design-library service. Study duplication alone does not express the library workflow.
- **Study assembly:** artifact attachment exists in Swift application code without a corresponding Swift CLI verb. Study-pack coauthoring/import is exposed through the app without equivalent CLI/API exposure found in this review.
- **Advanced execution:** `rescore-style` is not an accepted submitted-study operation, and the standalone floor battery has no bundle submission route.
- **Machine coverage:** some advanced verbs remain outside the declarative help/envelope system.

The Swift HTTP API also includes selection-based operations: select a study, then mutate the selected study. That is harder to reason about during concurrent human/agent interaction than requests that explicitly identify their workspace and study.

**Recommended outcome:** Define supported application operations with explicit targets, capability requirements, results, refusals, and artifact effects. Expose them through agent interfaces and SwiftUI. Avoid implementing independent business rules in new CLI wrappers.

Parity means equivalent operations and artifacts with explicit restrictions. It does not require MLX and CUDA to support identical execution semantics. The agent should discover a restriction and route appropriately without making the researcher learn the engine architecture.

### 7.3 Separate generated-reference checks from workflow correctness

Passing generated-reference checks establishes consistency for declared command tables. It does not establish that every operation is declared, that narrative instructions are current, or that the documented preflight, defaults, and recovery behavior are correct.

The independent audit's literal-mention search also has false negatives: the workspace contract already documents `confirm` and includes a `remote resubmit` example. These should not become duplicate documentation tasks. Similarly, endpoint-literal scans are useful inventories, but runtime-composed paths and method distinctions need inspection. Reconcile coverage against the existing route census and actual callers before using counts as completion gates.

**Recommended maintained capability matrix:** For each research operation, record:

- Purpose and stable operation identity.
- CLI, API, and UI entry points, including documented equivalent spellings.
- Authoritative workspace and supported service roles.
- Engine/model capability requirements and preflight.
- Inputs, output artifacts, and scientific restrictions.
- Failure, cancellation, recovery, and import behavior.
- Verification evidence and any unverified surface.

References:

- `Server/steerlab_server/cli_help.py:262` — help coverage is not the complete surface.
- `Server/steerlab_server/api/submissions.py:34` — accepted submitted-study verbs.
- `Server/steerlab_server/api/routes.py:2105` — model-install endpoint.
- `Sources/ExperimentKit/StudyDesignLibrary.swift:6` — existing design-library owner.
- `Sources/ExperimentKit/ExperimentStore.swift:2137` — Swift artifact attachment.
- `Sources/ExperimentKit/WebServer.swift:376` — local chat endpoint.
- `Sources/ExperimentKit/WebServer.swift:756` — selection-based HTTP operations.
- `docs/AGENTS-WORKSPACE-DRAFT.md:534` — confirmation instructions already present.
- `docs/AGENTS-WORKSPACE-DRAFT.md:1093` — resubmission instructions already present.
- `docs/AGENTS-WORKSPACE-DRAFT.md:1002` — standalone floor-battery execution and lack of bundle route.
- `docs/PORTABILITY-CONTRACTS.md` — documented authoring-client differences and service-role contracts.

## 8. Cluster onboarding lacks the document-to-profile workflow

The configuration substrate is strong. Profiles describe scheduler resources, storage, environment setup, and policies. Shared previews expose rendered environment settings, scheduler headers, and unresolved facts. The editor, setup wizard, and CLI share important parts of this machinery.

The missing part is the front of the journey. Agent instructions largely tell the agent to obtain an existing profile; the wizard offers site editing and JSON import. This review did not find a shipped workflow that turns institutional documentation into a proposed profile.

**Recommended output from cluster coauthoring:**

1. A draft site profile.
2. Documentation sources supporting important site facts.
3. Unresolved questions for the researcher.
4. A rendered plan for inspection.
5. Validation and connection checks before deployment.

The existing unresolved-facts machinery is a good starting point. However, missing login-node policy currently produces an advisory explaining that the bootstrap guard will not refuse anything. An incomplete profile is not automatically a sufficiently understood site.

Scope must also be explicit: scheduler support currently means Slurm or no scheduler. Reading arbitrary cluster documentation cannot imply support for arbitrary schedulers.

References:

- `Sources/ExperimentKit/ClusterSitePreview.swift:36` — shared preview contract.
- `Sources/ExperimentKit/ClusterEnvironmentRenderer.swift:1072` — unresolved facts.
- `Sources/ExperimentKit/ClusterEnvironmentRenderer.swift:1133` — missing login-node policy behavior.
- `Sources/ExperimentKit/ClusterSiteProfile.swift:328` — scheduler variants.
- `Sources/SteerLabApp/ClusterSetupWizard.swift:270` — site selection/edit/import workflow.
- `prompts/fixtures/cluster-site-profile/README.md` — fictional worked profile and cross-engine rendering fixtures.

## 9. Complete the remote recipe, recovery, and cleanup

### 9.1 The agent contract omits essential links in submission

The contract names `remote submit-bundle` but does not supply the preceding `remote package` and `remote upload` sequence. Submission expects a server-side path returned by upload. The agent should not have to reconstruct that dependency from a large reference document.

Publish a complete operation recipe: identify the intended site and workspace, inspect capabilities and required readiness, package locally, transfer through a permitted mechanism, submit the returned server path, retain the job's identity, observe progress, recover after interruption, and import verified evidence. The example must distinguish stopping observation from cancelling computation and distinguish downloading an artifact from importing a run or complete chain.

Name `remote capabilities`, `remote logs`, `remote cancel`, and the appropriate import operations alongside submission. Preserve the existing resubmission guidance rather than treating it as absent. Explain when the composite Python-client `run` provides the full round trip and when the lower-level Swift remote or cluster operations are needed. Do not prescribe HTTP upload where site policy requires external transfer; OPS-04 must govern the recipe.

Also provide a compact index of advanced/server-side families and their purposes: optimization, SAE/Gemma Scope, fine-tuning, J-lens, impact assessment, bundle inspection, housekeeping, and diagnostics. An index should route the agent to focused instructions instead of copying every reference page into the workspace contract.

**Acceptance criteria:** An agent with the shipped instructions and supported command help can take an eligible study from local inputs to verified local evidence without inventing shell scripts or private calls. The instructions identify their prerequisites and provide a complete recovery path.

References:

- `docs/AGENTS-WORKSPACE-DRAFT.md:1037` — existing remote instructions.
- `docs/CLI-REFERENCE.md:2039` — package, upload, and related remote operations.
- `Server/steerlab_server/client_cli.py:20` — composite client/runner workflow.

### 9.2 Managed cleanup remains incomplete

Existing safeguards include node-scratch cleanup, housekeeping, purge-risk reporting, verified imports, and shard-family purge-eligibility checks.

The remaining product gap is a managed lifecycle for retained remote artifacts:

> Verify the local copy → establish that remote artifacts are no longer needed → produce a cleanup plan → perform authorized cleanup → record the result.

`cluster import` deliberately reports eligibility without deleting. That is a sensible import boundary, but another operation must finish the lifecycle if agents are to avoid improvised filesystem cleanup.

Shard merging also still depends on a running server reconciler. Queue submission alone does not guarantee that the merged artifact will become available without that service.

**Recommended outcome:** Provide explicit cleanup planning and execution over verified evidence and declared retention rules. Protect active jobs, resumable checkpoints, unmerged shard evidence, and artifacts lacking verified local custody. Make policy-required scheduler cleanup distinct from application-managed removal of retained artifacts.

References:

- `Sources/ExperimentKit/WorkspaceImportPolicy.swift:688` — shard-family purge eligibility.
- `Sources/ExperimentKit/ClusterCLIRunner.swift:851` — import summary and purge reporting.
- `Server/steerlab_server/api/housekeeping.py` — quota and purge-risk observation.
- `docs/CLI-REFERENCE.md:5149` — remaining shard lifecycle gaps.

## 10. Scientific guidance and qualification remain release concerns

### SCI-01 — Interpretation language overstates what a control establishes

The UI guidance says matched-norm random controls “prove” an effect is the concept's direction. That is too strong. Beating random controls does not rule out a systematic confound in the extracted direction.

The extraction reference correctly explains this limitation. Scientific guidance should agree across documentation, UI, agent prompts, and generated reports.

**Required outcome:** Distinguish successful execution, admissible evidence, and support for the substantive claim. Structural gates and dataset audits do not independently establish construct validity.

References:

- `Sources/ExperimentKit/StudyAuthoring.swift:293` — UI interpretation claim.
- `docs/EXTRACTION-RECIPES.md:387` — what arithmetic and provenance establish, and what they do not.

### SCI-02 — Production qualification is explicitly unfinished

The production validation matrix is marked unexecuted, has no result rows, and identifies missing production runners. This limits promises about dependable use across production model/runtime configurations.

This assessment does not reopen every previously fixed scientific implementation issue. It distinguishes code fixes and regression coverage from empirical qualification of the configurations researchers will use.

**Required outcome:** Complete and publish bounded qualification results with pinned configurations, predeclared tolerances, artifacts, and explicit unavailable cases. The agent should be able to discover qualification status before execution and describe its implications accurately.

Reference: `docs/VALIDATION-MATRIX.md:1`.

## 11. Implementation plan

The packages below describe implementation work within the handoff scope. Owner labels name responsibilities, not assigned people or new agent tasks. Each package should become bounded issues and reviewable changes in one shared backlog, with links to the finding IDs and acceptance journeys. This handoff requests no preliminary estimates and makes no calendar-duration claims. WP-5 and WP-6 remain in scope.

### 11.1 Delivery order and dependencies

| Package | Suggested responsibility | Dependency | Researcher-visible outcome |
|---|---|---|---|
| WP-0 — Baseline and operation inventory | Integration lead with interface owners | Start immediately; must not delay OPS-01 | Everyone implements the same research operations and release scope |
| WP-1 — Correct ownership and observation | Runtime, jobs, and evidence owners | Start immediately; coordinate context shape with WP-0 | Observing, switching contexts, and reconnecting cannot redirect or damage work |
| WP-2 — Shared services and service-role enforcement | Swift application and Python API owners | WP-0 contracts; integrate WP-1 context fixes | UI and agents act through equivalent rules; runner and workbench authority are explicit |
| WP-3 — Complete public operations | CLI/API/UI owners | Relevant WP-2 services | Templates, model preparation, and study assembly are agent-drivable |
| WP-4 — Research guidance and coauthoring | Method and scientific documentation owners | Early corrections now; executable guides follow WP-3 | Researcher supplies concepts and decisions; agent handles formats and assembly |
| WP-5 — Cluster profile coauthoring | Cluster configuration owner | WP-0 operation inventory; existing profile services | Researcher supplies cluster documentation and missing facts, not a hand-built config |
| WP-6 — Complete remote lifecycle | Cluster, jobs, transfer, and evidence owners | WP-1 ownership; WP-2 authority; WP-3 operations | A remote study returns verified evidence and supports bounded cleanup |
| WP-7 — Qualification and journey verification | Validation owner with independent reviewers | Begin harness work early; completion follows relevant packages | Published capabilities are demonstrated on supported configurations |

WP-1 is the first implementation priority. Correct documentation and harmless interpretation wording can proceed alongside it. WP-3 can land operation by operation; WP-4 need not wait for every interface to be complete. WP-5's prompt/schema work can proceed independently of unrelated study UI changes. WP-7 should design the acceptance fixtures early so the other packages know what they must satisfy.

### WP-0 — Establish the shared baseline and operation inventory

**Change:** Turn the audits into one implementation backlog and one maintained capability matrix, using the classification in section 7.1. Record the actual source revision and installed CLI/app/server build identities. Inspect the unverified matrix cells and reconcile HTTP coverage against registered method/path pairs and the existing role census.

For each operation, identify its current service owner, authority, supported targets, inputs, results, refusals, artifacts, and recovery behavior. Record documented equivalents between Swift and Python. Propose the minimum request context needed for workspace/study identity and revision, target site, and durable job identity; use existing identifiers where possible.

**Benefit:** Removes ambiguity for implementing agents and prevents two teams from independently inventing different operations for the same researcher action.

**Deliverables:** Maintained matrix, ordered issue list, agreed operation contracts for the first slices, and explicit release-scope decisions or proposed decisions awaiting the maintainer. No new dispatcher or service rewrite is required for this package.

**Gate:** Every in-scope research action has an owner and a coverage status; absence, intentional restriction, and unverified behavior are distinct. OPS-01 proceeds without waiting for the whole inventory.

### WP-1 — Make observation, jobs, and evidence context-safe

**Change:** Implement OPS-01 first, then OPS-02 through OPS-05 and UI-01 in bounded slices:

- Separate job reads from recovery. Recovery must establish ownership/liveness rather than treating every manager construction as a daemon restart.
- Carry originating site/server and workspace identity with job references through persistence, cancellation, resumption, and reconnection. Existing records lacking identity must not silently borrow the currently selected server.
- Pass captured workspace context through packaging and the complete import transaction, including revision adoption and ledger updates. Check all asynchronous boundaries where current global state can replace captured state.
- Require the operation on low-level submission paths. Inspect composite entry points and explicitly decide which defaults remain intentional; do not create conflicting interpretations across wrappers.
- Enforce transfer restrictions at client and server boundaries. A permitted alternative must be reachable or the refusal must identify the actual supported remedy.
- Show imported pipeline evidence whenever it exists locally, independent of compute selection.

**Benefit:** Researchers can use the app while an agent works, switch studies, and recover remote jobs without becoming responsible for hidden connection or workspace state.

**Deliverables:** Corrected code paths and focused regressions using disposable job stores, controlled transports, and deterministic context-switch boundaries.

**Gate:** Read operations change no job or scheduler state; actions retain their originating context; interrupted imports publish no incomplete durable evidence; forbidden transfers refuse before moving data; imported results remain visible offline. Invalid or unresolvable old job identity is reported explicitly.

### WP-2 — Finish shared authoring services and enforce runner/workbench roles

**Change:** Complete the panel-authoring migration around explicit operation requests and results. Route the app, Swift CLI, and Swift HTTP handlers through the same relevant owners. Replace selection-dependent writes with captured workspace/study targets; protect draft mutations against stale revisions so simultaneous UI and agent edits cannot silently overwrite each other.

**Identity constraint:** Do not add a concurrency revision, version, timestamp, or other stale-write field to the manifest document. Do not change manifest serialization or content/freeze hashing to implement this feature. A new field inside content-hashed bytes could re-identify existing studies. The precondition belongs outside the manifest: for example, an expected digest of the exact file bytes supplied in the operation request or HTTP precondition. Keep that concurrency digest distinct from the scientific study/content identity.

Compare the expected precondition with the current file and publish the write under one serialized or atomic update mechanism shared by all relevant writers; a hash check followed by an unprotected write leaves the original race. Missing-precondition behavior must be explicit for every supported adapter. Do not migrate or rewrite frozen manifests, canonical freeze documents, hash sidecars, or existing evidence for this feature.

In Python, turn the existing service-role census into an enforced deployment boundary. Keep authentication separate from service-role authorization. Review `both` operations with their existing callers before narrowing them; return typed repairs when an operation belongs to a workbench rather than a runner. Do not confuse runner/workbench service roles with controller/GPU-session execution topology.

Retire the four Swift compatibility bridges as callers move to the focused services. Preserve public operations that are intentional stable interfaces; do not replace the bridges with a new generic facade that hides the same mutable panel dependencies.

**Benefit:** A researcher and an agent can take turns on the same study. The interface used does not determine the study's rules, and the server cannot accidentally become a second authoring authority.

**Deliverables:** Explicit authoring transactions, equivalent interface adapters, draft-conflict handling, enforced service roles, and bridge removal under the existing retirement gate.

**Gate:** Equivalent supported declarations produce equivalent intended study semantics and pins across interfaces; stale writes refuse with a repair; two writers starting from the same file cannot silently overwrite one another. Adding stale-write protection alone leaves existing manifest bytes, scientific content/freeze hashes, and frozen-study verification unchanged. Runners refuse independent authoring while workbenches and legitimate derived computation remain functional. All bridge callers use their intended owners, and the existing bridge-release gate passes.

Reference: `docs/BRIDGE-RETIREMENT.md:1`.

### WP-3 — Expose the missing public research operations

**Change:** Add supported entry points over the existing services, starting with the operations that block complete journeys:

| Operation group | Minimum supported actions | Shared human/agent experience |
|---|---|---|
| Study designs | List, inspect, instantiate, and deliberately save a design/version | The same library can start a study from the app or an agent |
| Model preparation | Inspect readiness, plan installation, install, and observe completion/failure | The same resource and site-policy preflight applies |
| Study assembly | Import/validate a study pack; attach concept, reader, agent, and applicable artifact inputs; pin data through real services | Nobody fabricates hashes or repeats setup in the other interface |
| Advanced methods | Discover supported optimization, SAE, fine-tuning, J-lens/J-space, stability, rescoring, and battery operations and their restrictions | Agents see the same capabilities and limitations as researchers |
| Execution and recovery | Explicit stage/target submission, observation, cancellation, resumption, and verified import | Each interface uses consistent job and artifact meanings |

For advanced methods, close the inventory's missing entry points or record the exact supported operation mapping. Decide whether non-manifest jobs need a first-class job submission shape; do not force an unrelated artifact into a study manifest solely to reuse a route. Every in-scope capability must have a public programmatic path and an appropriate representation in the app, even where the execution belongs exclusively to one engine.

**Benefit:** The agent can finish the work that the researcher requested through supported operations. App controls become another way of reaching those same capabilities.

**Deliverables:** CLI/API adapters, UI bindings or cross-links, help and result schemas, capability declarations, and updated matrix rows. Specific new command names should be settled in the interface review, not inferred from prose examples in this plan.

**Gate:** Each completed operation has a positive case and meaningful refusal cases across its supported interfaces. No wrapper bypasses the owner's preflight, defaults, policy, or artifact validation. Unsupported engine cases give an actionable restriction rather than an unexplained missing verb.

### WP-4 — Make research instructions complete and usable by external agents

**Change:** Correct the extraction paragraph and scientific overclaims immediately. Build per-technique guides from the existing recipe, training, scope, and validation references. Expose them through shipped resources and discoverable operations/help; a released installation must not require an unstated source checkout.

Use progressive guidance: a short workspace contract explains the lifecycle and how to find method-specific depth. Each method guide supplies the schema, author/reviewer prompts, split roles, commands, outputs, limitations, and applicability listed in section 5. Include an advanced-family index and the complete remote recipe.

Extend the existing study coauthoring workflow so an external agent can execute assembly through WP-3. Separate scientific decisions from technical mechanics. Identify which decisions require researcher input, which missing facts require questions, which defaults are explicit and scientifically inert, and which unresolved declarations must block progress. Extend multi-agent guidance to scenario design, materials, visibility, casting, controls, and outcomes.

**Benefit:** Researchers can express the study in ordinary research terms. Agents can obtain the details on demand rather than guessing formats or asking the researcher to operate the app for them.

**Deliverables:** Consistent shipped guides and prompt resources, executable examples, independent-review instructions, and updated UI explanations using the same definitions.

**Gate:** An external agent using released instructions and public help can assemble the bounded acceptance studies without consulting private source, fabricating pins, or handing back routine attachment work. A second reviewer can determine what dataset checks were performed and which substantive scientific questions remain unresolved.

### WP-5 — Add bounded cluster-document coauthoring

**Change:** Reuse profile schemas, fictional examples, unresolved-fact reporting, and the shared preview. Provide a cluster coauthoring prompt/workflow that accepts institutional documentation supplied to the external agent and produces a proposed profile plus a source-supported fact record and unresolved questions.

Distinguish an institution's documented defaults from the researcher's actual account, allocation, permitted partitions, storage paths, and transfer constraints. Do not copy unknown values from the example profile. Declare unsupported scheduler types early. Keep human authentication as a bounded handoff, and expose the same profile validation/preview in the app and agent workflow.

**Benefit:** The researcher supplies their cluster documentation and personal allocation facts without learning the profile schema or scheduler script conventions.

**Deliverables:** Authoring and review prompt resources, a defined companion fact/question record, validation/preview integration, and a guide from approved profile to connection and qualification. No built-in general web-browsing agent is required.

**Gate:** Given a fictional documented site with deliberate gaps, the agent produces a correct reviewable profile, identifies those gaps, and asks only relevant unresolved questions. Unknown policy never becomes permissive execution through an unnoticed default. The app and CLI show the same rendered plan and unresolved facts.

### WP-6 — Complete remote execution, recovery, and cleanup

**Change:** Make the following sequence accessible as a documented composed workflow over existing operations:

> Capture study/workspace/site → check capabilities and policy → package → transfer through a permitted transport → submit → retain job identity → observe or reconnect → resume/merge as needed → verify/import → report local custody → plan and apply authorized cleanup.

Perform checks that can be decided locally before expensive transfer or allocation. Preserve the distinction between stopping observation and cancelling computation. Make merge/reconciliation prerequisites discoverable and provide a supported recovery path if the usual driver stops; observation itself must not become implicit destructive recovery.

For cleanup, begin with a constrained artifact class supported by existing eligibility evidence. A plan names exact targets, the applicable policy, local verification evidence, and blockers. Application rechecks eligibility and active/resumable dependencies before deletion; changed facts invalidate or narrow the plan rather than expanding its scope. Record what was removed and what was retained. Scheduler-managed node-scratch cleanup remains a separate policy mechanism.

**Benefit:** Remote work behaves like a study operation with additional execution logistics, rather than a collection of shell tasks the researcher must supervise. Local custody and permitted cleanup are visible in both the app and agent results.

**Deliverables:** Complete public recovery/import paths, consistent lifecycle results, bounded cleanup planning/execution, and UI views/actions over those services. A broad rewrite of all transports is not a prerequisite for enforcing their common policy contract.

**Gate:** Journeys D and E pass, including loss of connection, restart, changed workspace/server selection, interrupted transfer, incomplete merge, and cleanup refusal when custody or dependency checks fail. No hand-written scheduler script or improvised filesystem deletion is required for the declared supported journey.

### WP-7 — Demonstrate scientific and interface qualification

**Change:** Execute the bounded production validation matrix with pinned configurations and predeclared tolerances. Build the cross-interface journey harness and supplement it with interactive SwiftUI checks. Make qualification status discoverable before execution and inspectable with the results.

Treat the app-open Swift-suite hang as an investigation item: record build identities, environment, the blocked test, and a reproducible trigger before asserting a universal cause or operating rule. Capture useful diagnostics in disposable test environments.

**Benefit:** Researchers and agents can distinguish a supported, qualified workflow from one that is merely implemented or exploratory. Release claims rest on observed behavior.

**Deliverables:** Linked qualification artifacts, journey results, verified capability-matrix rows, UI QA evidence, and explicit unresolved limitations.

**Gate:** Applicable scientific matrix rows have results or explicit unavailable outcomes; all in-scope journeys pass on declared configurations; cross-interface handoffs do not require repairs outside public operations. Software tests, live workflow evidence, and scientific qualification are reported separately.

### 11.2 How to land the work without another large migration

Land vertical slices: one operation's service, interfaces, help, relevant tests, and user-facing behavior together. For a mechanical move, preserve behavior and use the existing AST audit; put semantic fixes in separately reviewable changes. Keep scientific behavior changes explicit and reviewed rather than hiding them inside ownership refactors.

Integrate current main regularly in isolated branches, coordinate ownership of shared files, and retain the running checkout/workspaces. **Implementation does not authorize an implementing agent to merge or land its own work. The maintainer owns the landing decision, acting through the maintainer's designated reviewing/integration agent.** Submit the actual diff and validation evidence to that reviewer; the implementing agent's own approval is insufficient.

Before an implementation change lands, the designated reviewer must read the diff, both the Python and Swift suites must pass for the proposed landing state, and every claimed mechanical move must pass the existing AST audit against its stated baseline. Run the bridge and other applicable release gates as well. Record which checks passed, on which revision and environment. A local branch commit is a review artifact, not approval to merge, push, install, or deploy. Only the maintainer, through that review process, can approve exceptions or landing.

On this maintainer's Mac, build/test with Xcode beta and the installed Metal toolchain configured as previously documented; put build scratch outside the file-provider-managed checkout. Record the actual toolchain and binary identities. The current working setup includes `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer`; resolve the installed Metal identifier for `TOOLCHAINS` if it has changed. Preserve the documented serial Xcode test and `CLANG_COVERAGE_MAPPING=NO` settings where applicable.

### 11.3 Validation gates shared by the packages

| Gate | What it establishes | What it does not establish |
|---|---|---|
| Targeted regression | The reproduced defect is fixed under its controlled trigger | Live production qualification |
| Existing Python and Swift suites | Covered implementation contracts remain intact | Complete surface parity or UI usability |
| Mechanical AST audit and bridge gate | Claimed moves preserve bodies; transitional bridges meet their release criterion | Scientific correctness of a semantic change |
| Operation-equivalence tests | Supported interfaces apply equivalent intent, preflight, artifact effects, and repairs | Byte identity across different engines or stochastic runs |
| Interactive UI checks | Researcher can author, run, inspect, and recover through the actual app | All numerical scientific contracts |
| End-to-end journeys | The researcher/agent workflow completes under declared conditions | Support for untested clusters and model configurations |
| Production scientific matrix | Named numerical and intervention contracts hold for qualified configurations | Construct validity of every future study |

For each operation-equivalence test, compare meaningful effects and refusal semantics, not presentation text. Include wrong-workspace, stale-draft, unsupported-engine, and unavailable-server cases where relevant. Use intentional interoperability rules for hashes and canonical documents rather than requiring different engines to produce identical bytes.

## 12. Suggested acceptance journeys

These are proposed product acceptance scenarios, not tests already performed by this review.

Each completed journey must retain the declared input fixture, source and binary identities, target configuration, public operations invoked, relevant UI observations, resulting artifacts, and any manual intervention. Record which scientific decisions the researcher supplied. A journey that needed private source inspection, invented pins, or undocumented repair does not meet the agent-driveability criterion merely because it eventually ran.

| Journey | Principal packages | Main product criterion |
|---|---|---|
| A — Conceptual steering study | WP-2, WP-3, WP-4, WP-7 | Researcher accessibility and scientific clarity |
| B — Multi-agent study | WP-2, WP-3, WP-4, WP-7 | Complete study authoring through supported operations |
| C — Cluster from documentation | WP-5, WP-6, WP-7 | Accessible, policy-aware onboarding |
| D — Interruption and recovery | WP-1, WP-6, WP-7 | Durable ownership and agent driveability |
| E — Evidence and cleanup | WP-1, WP-6, WP-7 | Local custody, offline visibility, and managed remote completion |
| F — Templates and models | WP-3, WP-7 | Agent access to routine preparation |
| G — Workspace authority | WP-2, WP-7 | Enforced runner/workbench separation |
| H — Interface handoff | WP-1, WP-2, WP-3, WP-7 | Equivalent meaning across app, CLI, and API |

### Journey A — Conceptual steering study

A researcher describes a construct and behavioral question. An agent explains the relevant method choices, obtains the required decisions, delegates dataset authorship and independent review, assembles a draft through supported operations, checks readiness, and executes the declared screening and confirmation workflow. The researcher can inspect the same study and evidence in SwiftUI.

Success requires no invented hashes, no manual manifest repairs, no undocumented private function calls, and no request for the researcher to perform routine attachment steps.

### Journey B — Multi-agent study

A researcher describes the interaction and outcome of interest. The agent authors or adapts the scenario, materials, visibility rules, casting, controls, and transcript evaluation. The app presents the resulting design and evidence without requiring a second independent setup process.

### Journey C — New cluster from documentation

The researcher supplies institutional documentation. The agent produces a source-supported profile, asks only unresolved questions, previews the rendered configuration, and completes supported setup and qualification. Unknown policies remain explicit. Human authentication remains a clear, bounded handoff.

### Journey D — Remote interruption and recovery

A study is submitted remotely. The app closes or the connection drops; the agent later reconnects, observes the correct job without altering it, resumes if appropriate, waits for complete evidence, and imports it into the originating workspace. Switching servers or workspaces does not change job or evidence ownership.

### Journey E — Verified evidence and policy-compliant cleanup

The agent verifies local custody of a completed study's required artifacts, produces a remote cleanup plan, applies authorized cleanup under the site's policy, and records the result. Unmerged, active, resumable, or unverified artifacts remain protected. Imported results remain visible in SwiftUI with no server connection.

### Journey F — Agent-driven template and model preparation

An agent lists and inspects available study designs, instantiates one into a named local workspace, and prepares its model on the intended execution target through the same preflight and policy checks the app uses. It reports progress, failures, and the resulting identities. The researcher opens the draft in SwiftUI and sees the same design and preparation state.

### Journey G — Enforced workspace authority

An authenticated agent attempts a workbench authoring operation against a runner. The runner returns a typed refusal and identifies the correct authoring context. The agent authors locally and submits pinned inputs successfully. Execution may produce vectors, adapters, readers, and evidence, but it cannot independently rewrite the authoritative study definition. The workbench retains its intended authoring capabilities, and the import path does not silently adopt divergent definitions.

### Journey H — Researcher and agent change interfaces without changing the study

Create and save a draft through one public authoring interface. Open it in SwiftUI and inspect its methods, data, arms, and readiness. Make a supported draft change in the app; have an agent inspect and continue it through the CLI or API. Attempt a stale write from the earlier revision and confirm a clear conflict rather than silent overwrite.

Submit through one supported interface, observe through another, and inspect the resulting imported evidence in both the app and a structured agent result. Compare the persisted scientific declarations, pins under their applicable interoperability rules, preflight decisions, job identity, and artifact provenance. Repeat equivalent invalid requests through supported adapters and verify that they name the same underlying problem and repair.

The researcher never recreates a configuration just to switch interfaces. Supported platform differences are explained as capability restrictions, not silently transformed settings.

## 13. Team handoff guidance

Treat the numbered findings as review-backed starting points, not authorization to execute changes or cleanup. Before implementing a finding, check whether a subsequent commit has resolved it and retain the existing scientific and artifact contracts.

### 13.1 Non-negotiable implementation constraints

- **Preserve scientific identity.** WP-2 stale-write preconditions stay outside manifest bytes and scientific content/freeze hashes. Use an external expected-file digest or equivalent precondition and serialize comparison plus publication across writers. Do not introduce a manifest revision field or rewrite existing studies as a concurrency migration.
- **Keep the public repository generic.** Do not add real site or institution names, hostnames, account/allocation identifiers, private storage paths, or study-case-specific vocabulary/data to source, fixtures, docs, commit messages, or review artifacts intended for the repository. Use neutral fictional examples where a schema or test requires values. Keep real site profiles and study material in their designated private stores/workspaces.
- **Keep credentials out of repository and study artifacts.** On this Mac, credentials belong in the Keychain; never copy them into JSON, environment files, command arguments, logs, commits, or documents. Preserve the existing non-Mac client contract for protected token-file references rather than breaking it with a Mac-only assumption. No credential value belongs in the handoff or test evidence.
- **Preserve immutable artifacts.** Do not hand-edit frozen manifests or historical files under `runs/`, re-pin old evidence to make a gate pass, or use live study workspaces for regression reproduction. Corrections create new drafts or new result artifacts through supported operations. Normal engine writes while producing an active run are not authorization to repair historical evidence in place.
- **Use disposable validation state.** Reproduce defects against test stores and controlled transports, never against live job stores. Until OPS-01 is fixed, query jobs through the running server rather than constructing the unsafe CLI job manager against a controller-owned store.
- **The maintainer's reviewing agent is the landing gate.** Work on a branch. Deliver the diff and evidence to the maintainer through the designated reviewing/integration agent. Both suites must pass before implementation lands; claimed mechanical moves additionally require the AST audit. Implementing agents do not self-approve merges, pushes, installations, or deployments.

These constraints are explicit here because a general reference to repository instructions does not communicate all of them. This document's preparation is not authorization to start any implementation package.

### 13.2 Backlog and evidence ownership

For each completed workstream, record the implementation revision, validation performed, supported configurations, and remaining limits. Do not use passing unit suites as a substitute for the end-to-end researcher journeys or the production scientific validation matrix.

Maintain one implementation backlog rather than treating the audits as competing instructions. Each issue should record: finding or product gap, proposed public behavior, owning service, affected surfaces, dependencies, acceptance criteria, verification evidence, and disposition. Preserve the existing finding IDs when closing or superseding them. Give new coverage gaps their own IDs rather than folding unrelated semantic changes into an operational fix.

Use this document for the product outcomes and implementation acceptance plan. Use the other team's synthesis as a complementary operational review and the explicit constraints above for this handoff. The maintainer's scope decision supersedes the companion's earlier proposed deferrals: WP-5 and WP-6 stay in scope, without preliminary estimation requests. Where any other instructions or release boundaries differ, reconcile them in the shared backlog before implementing the disputed change.

The existing Python alias cleanup and Swift bridge-release gate should be preserved. They are established maintainability work, not reasons to reopen the removed alias convention or duplicate the gate definitions here. Bridge removal belongs to WP-2; any newly found boundary regression should have a focused issue and test. Track the app-open test hang separately as an investigation until its cause is isolated.

Use the two audits together: the independent surface inventory helps locate missing entry points and instructions; this assessment connects those gaps to workspace authority, operational correctness, and complete researcher journeys. Verify uncertain matrix cells before promising parity. Preserve intentional engine and authoring-CLI differences when they provide equivalent supported behavior.

**Completion standard:** The researcher can understand the available methods and the meaning of the results; an agent can complete supported workflows from shipped instructions and public operations; and changing between SwiftUI, CLI, and API does not change the study's meaning, ownership, or safeguards. The implementation plan is complete only when that behavior is demonstrated for the declared release scope, not when every matrix cell merely names an endpoint.
