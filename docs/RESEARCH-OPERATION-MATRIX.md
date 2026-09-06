# Maintained research operation matrix

Baseline for this revision: `45655ae` on `codex/researcher-workflow-implementation`,
based on main `bfd13a5`. This is the working inventory for WP-0–WP-7, alongside
[the implementation plan](RESEARCHER-WORKFLOW-IMPLEMENTATION-PLAN.md) and
[the validation ledger](RESEARCHER-WORKFLOW-IMPLEMENTATION-STATUS.md). Historical
audit counts are not a coverage or release score. Update the affected rows with
each operation change; add a row when an operation has a different authority,
artifact, or recovery contract.

## How to read the matrix

- **Present:** source entry point and owner identified; this alone does not prove
  successful execution, semantic equivalence, or scientific qualification.
- **Gap:** an in-scope operation has no supported entry point on that surface.
- **Restricted:** an intentional engine/service-role restriction, requiring a
  discoverable explanation and routing repair.
- **File:** the supported input is an authored file. It still needs schema,
  prompts, validation and real attachment/pinning operations.
- **Unverified:** the particular path has not been inspected or exercised here;
  absence must not be inferred. A mixed cell identifies what remains unchecked.

Column abbreviations are product names, not interchangeable executable aliases:
**Swift** = `steerlab-cli`; **Client** = `steerlab`; **Engine** =
`steerlab-server`. **HTTP** explicitly distinguishes the Swift app server from
the Python engine. UI entries identify source affordances; interactive acceptance
is outstanding unless the evidence column says otherwise.

The owner column names implementation responsibility. It does not assign people
or authorize new agent tasks. The source groups below provide navigable owners.

## Study and input authorship

| ID / operation | Owner | Swift | Client / Engine | HTTP | UI | Evidence / next gate |
|---|---|---|---|---|---|---|
| AUTHOR-01 list/read a study | Manifest repository/store | Present: `experiment list`, `manifest` | Client list; Engine list | Python manifest GET; Swift named manifest read/state | Studies | Swift CLI/Python ETag and named Swift HTTP read tested; browser-state Keychain blocking remains |
| AUTHOR-02 create/duplicate/rename a draft | Store, study management | Present lifecycle authoring; rename mapping unverified | Client create/duplicate; rename unverified | Python create; Swift selection-based create | Studies | Remaining creation/rename stale-write audit |
| AUTHOR-03 save study setup | Protocol authoring/store | Named field setters; complete setup adapter gap | Client set-protocol; Engine restricted authoring | Python named protocol/manifest; Swift explicit protocol | Study Setup | Shared service and named Swift HTTP read/save/refusal paths tested; native fields untouched; browser end-to-end qualification remains |
| AUTHOR-04 attach/detach extraction concepts | Store | Present attach/detach | Client attach/detach; Engine authoring restriction | Python authoring attach/detach; Swift attach | Concepts in study | Preserve recipe-specific files, reading/rendering choice and scope invalidation; concurrent whole-writer audit |
| AUTHOR-05 attach an existing artifact/reader/agent | Store, variant owners | Artifact CLI gap; variant paths differ | Client attachment mapping present; Engine attach-artifact | Python authoring attachment; exact type coverage unverified | Study/agent controls | Record per-artifact restrictions; avoid inventing hashes or using unrelated manifest fields |
| AUTHOR-06 declare/remove conditions and controls | Manifest field policies | Present declare-condition/remove paths | Client condition operations | Python condition routes; Swift adapter inventory incomplete | Conditions/controls | Shared admission exists; complete condition-owner and explicit-target migration |
| AUTHOR-07 edit/import task prompt records | TaskPromptsAuthoring, Document/Import | File + pin-prompts; full editor/import adapter gap | File + client protocol pin mapping | Swift named prompts load/save with both file digests | Input Data; read-only web preview | Native and HTTP share immutable-version owner; exact read/save/refusal wire tests and late-preview response tests pass; CLI edit and raw-import migration remain |
| AUTHOR-08 pin rubric and judges | JudgeRubricStore, evaluation declaration | Present pin-rubric | Client protocol-field mapping | Python protocol; Swift setup | Judging | Root-bound pin and full-record judge tests pass; complete cross-surface request proof pending |
| AUTHOR-09 declare instruments, scope, parsers and exclusions | Manifest field policies | Present named setters | Client named/field equivalents | Python protocol; Swift coverage mixed | Study measurement controls | Preserve declared measurement rather than enabling from detected columns; full matrix of equivalent fields pending |
| AUTHOR-10 declare reasoning, sampling and model capabilities | Manifest field policies, capability records | Present sampling/model operations | Client protocol/capability; Engine probe | Python capability/probe and protocol; Swift setup | Model/sampling controls | Protocol service refuses invalid sampling before publication; qualification is model/template-specific |
| AUTHOR-11 declare sweep/promotion/confirmation policy | Store, confirmation owner | Present sweep setters/confirm | Client grid/protocol; Engine confirmation execution | Python authoring/submission; exact confirmation API mapping unverified | Studies/pipeline | `confirm` is already documented; audit defaults and do not create a duplicate documentation task |
| AUTHOR-12 verify/freeze a study | Store, freeze coordinator | Present verify/freeze | Client verify/freeze; Engine verify/execution gates | Python verification and manifest authority; Swift freeze adapter | Freeze controls | Shared manifest lock and preserved scientific identity tested; bridge and explicit adapter migration remain |
| DESIGN-01 list/inspect designs | StudyTemplateStore, design library | Gap | Gap | Gap | Templates | Expose existing library, not a second template format |
| DESIGN-02 instantiate/batch a design | StudyTemplateStore/Batch | Gap | Gap | Gap | Template instantiation | Include seat expansion, pins, lineage, collisions and rollback in public result contract |
| DESIGN-03 save/version a design from a study | StudyTemplateStore, management | Gap | Gap | Gap | Save as design/save back | Preserve prior studies and provenance; explicit destination/version operation needed |
| PACK-01 preview/import/export a study pack | StudyCoauthoring | Gap | Gap | Gap identified in audits | Copy LLM Prompt/import | Owner exists; public adapter must preserve file containment, differing-file refusal and rollback |
| DATA-01 author paired stimuli, stories and reference corpora | AuthoringPrompts, stimulus stores | File + authoring prompt | Client concept import/prompt; Engine data family | Python concept/story/neutral routes; Swift builder | Data | Guidance must distinguish paired, grand-mean and designated-reference datasets |
| DATA-02 author held-out probes, readers, choice items and batteries | AuthoringPrompts, input stores | File + prompt/reader/battery paths | Client prompts; Engine reader/battery paths | Python probe/reader/battery routes; Swift builder | Data/readers | Exact dataset/run compatibility and coauthoring examples remain WP-4 |
| PANEL-01 author/check semantic scenarios | PanelAuthoring, scenario store | Present panel family; File | Client gap; Engine list/check, authoring restrictions | Python scenario save; Swift coverage unverified | Panels | Multi-agent materials/visibility/turn/outcome guidance remains |
| PANEL-02 cast seats and compile a study | SeatCasting, PanelComposition | Present panel compile | Client gap; Engine authoring restricted | Swift setup through shared owner; Python equivalent unverified | Seats/Study Setup | Captured workspace and semantic-hash provenance tests pass; public full-operation parity pending |

## Model preparation, execution and scientific tools

| ID / operation | Owner | Swift | Client / Engine | HTTP | UI | Evidence / next gate |
|---|---|---|---|---|---|---|
| MODEL-01 inspect readiness/capabilities | Model capability/preflight owners | Present model/remote capabilities | Client capabilities; Engine model/profile | Python info/capabilities/preflight | Compute/model picker | Runtime roles differ by endpoint; availability is not numerical qualification |
| MODEL-02 plan/install a model and observe completion | LocalModelInstaller, Python model_install | CLI gap | CLI gap | Python privileged models/install exists | Compute/model install | API-accessible, not UI-only; reuse policy/disk/network preflight and durable job observation |
| MODEL-03 load/unload/cancel interactive residency | Chat/cluster model owners | Exact CLI mapping unverified | Client restricted; Engine serve/workbench | Python load/unload/cancel, Swift chat state | Playground | Workbench interactive slot, distinct from runner batch preparation |
| EXEC-01 extract/validate/sweep/run/evaluate/analyze/promote | Stage owners | Present experiment stages | Client delegates execution; Engine stages | Python submitted operations and workbench interactive paths | Studies | Unit/integration coverage exists; supported substrates and pinned live qualification must be stated per stage |
| EXEC-02 declare/run a pipeline | Pipeline owner, chain runner | Present pipeline | Client composite run; Engine pipeline | Python study submission/pipeline reads | Pipeline composer | Pipeline whole-document writer migration, offline UI QA and full interruption journey remain |
| EXEC-03 rescore style and complete judgments | Style/judgment stage owners | Present relevant experiment verbs | Engine rescore-style/judgment completion; Client mapping restricted | rescore-style not in submitted-study vocabulary | Coverage unverified | Add supported operation mapping; never silently submit as another stage |
| EXEC-04 extraction stability | Extraction stability owner | Restricted / no local counterpart identified | Engine extract-stability | Submitted/standalone mapping unverified | Unverified | Publish exact reachable operation, restrictions and interpretation |
| EXEC-05 standalone capability/floor battery | Battery runner | Battery mapping present; exact standalone parity unverified | Engine battery | No bundle submission route identified | Variant battery affordance | Give standalone jobs an intentional request contract rather than a fake study |
| METHOD-01 readers and reader scoring | Reader owners | Present reader family | Engine reader/data paths; Client file authoring | Python reader fit/score | Readers | Interpret a reader as an instrument; document training/evaluation split and model pins |
| METHOD-02 optimized vectors | OptVec owners | Restricted execution; artifact consumers | Engine optvec family | Submission coverage unverified by operation | Optimization | Index train/eval/geometry/family/interpret/fracture/campaign paths; identify envelope gaps |
| METHOD-03 fine-tune adapters | FineTuneStore, Python fine-tune owners | Authoring/artifact surface; execution restricted | Engine finetune plan/train/submit | Python plan/train/submit | Adapters | Workbench plan versus runner train/submit authority must remain explicit |
| METHOD-04 J-lens acquisition/qualification/readout | J-lens owners | Restricted execution, app rendering | Engine jlens family | Python J-lens workbench routes | J-lens | Publish all supported verbs, qualification and model/token/layer constraints |
| METHOD-05 J-space optimization | OptVec J-space owner | Restricted execution | Engine optvec jspace | Exact API mapping unverified | Exact UI mapping unverified | Separate from J-lens readout and ordinary paired extraction; add accessible discovery |
| METHOD-06 SAE/Gemma Scope analysis/import/qualification | SAE/Gemma Scope owners | Restricted execution; artifact attachment | Client pin-sae; Engine sae/Gemma Scope | Python Gemma Scope routes | SAE/adapter coverage mixed | Preserve convention/qualification stamps and distinguish feature-ID versus report-ranked imports |
| METHOD-07 vector parity/mirror/norm operations | Vector owners | Present vector family | Engine vectors; Client gap | Python vector operations | Exact controls unverified | Existing artifact writes need immutable-evidence admission audit; never relax it for parity |
| CHAT-01 local and remote exploratory generation | Chat/variant services | Remote chat present; local CLI gap | Client mapping restricted | Swift chat; Python generate/variant | Playground | API-accessible; explicitly distinguish exploration from frozen study execution |

## Cluster, remote custody and recovery

| ID / operation | Owner | Swift | Client / Engine | HTTP | UI | Evidence / next gate |
|---|---|---|---|---|---|---|
| SITE-01 validate/preview a profile | ClusterSiteProfile/Schema/Preview | Present cluster operations | Client runner URL; Engine profile/site operations | Python deployment profile checks | Cluster wizard | Exact common preview exists; profile facts must match supported scheduler/transfer policy |
| SITE-02 coauthor a profile from documentation | ClusterProfileCoauthoring; existing validation/preview | Present `cluster sites guide`, `sites review` | Python guided adapter gap; Mac configuration workflow | No browsing service required | Cluster wizard: From documentation | Source/value and policy guards, shared preview and real CLI fixtures tested; real-document interview and interactive review remain |
| SITE-03 connect/bootstrap/qualify/tunnel/disconnect | Cluster operation owners | Present cluster family | Client managed runner or runner URL; Engine serve | Session/deployment subset | Cluster wizard/Compute | Credentials stay in Keychain on Mac; source/profile/account distinction and complete instructions pending |
| REMOTE-01 inspect capabilities/transfer policy | ClusterClient, RunnerClient, service authority | Present remote capabilities | Client runner capabilities; Engine profile | Python capabilities/profile | Compute | HTTP transfer restrictions and role denial tested; show supported repair/transport choices |
| REMOTE-02 package/inspect/transfer a pinned bundle | Bundle/transfer owners | Present remote package/upload | Client bundle/runner operations; Engine bundle | Python upload/inspect/download | Submit controls | Policy preflight tested; complete source-byte capture and allowed external-transfer journey remain |
| REMOTE-03 explicitly submit a study/stage | Submission owners | Present submit-bundle/study path | Client runner submit/run; Engine study submit | Python submit/submit-bundle | Run/submit | Missing low-level operation refuses; verify intentional defaults of every composite adapter |
| REMOTE-04 list/watch/log jobs | Durable job store/ownership, job controllers | Present remote jobs/logs | Client runner jobs/logs; Engine jobs list | Python jobs/detail/stream | Recent/active jobs | Observation does not recover/cancel; owner-death and competing claims tested |
| REMOTE-05 cancel/resume/reconcile/reconnect | Origin-aware jobs/recovery owners | Present cancel/resubmit; full reconcile mapping unverified | Engine/client mapping varies | Python cancel/resubmit/reconcile | Job controls | Persisted job and Compute-list origin guards tested; delayed connection replies refused; finish remaining auxiliary/recovery paths |
| REMOTE-06 verify/import individual and chain evidence | Evidence import/chain/adoption owners | Present import/import-chain | Client bundle import/runner evidence/composite run | Python bundle evidence/download/import | Results/import | Captured root, adoption CAS, verified reuse, origin/version-scoped ledger and restart/refusal tests pass; durable archive/member receipts and full remote journey remain |
| REMOTE-07 browse imported results offline | Result/pipeline stores | Present results/import inspection paths | Client imported artifacts | Swift state; Python workbench catalog | Results/pipeline | Source UI gating fixed; interactive disconnected/local-target check outstanding |
| REMOTE-08 inspect quota/purge/maintenance | Housekeeping owners | Cluster/remote mapping unverified | Engine housekeeping | Python housekeeping status/refresh/maintenance | Compute | Existing information is not authorization to delete; policy provenance and public discovery needed |
| REMOTE-09 plan/apply bounded cleanup | Cleanup owner (WP-6), custody/dependency stores | Gap | Gap in managed local-custody workflow | Gap in managed workflow | Gap | Exact targets, verified local custody, active/resumable dependency checks, stale-plan refusal and audit result |
| QUAL-01 inspect/record qualification and run journeys | Qualification owners (WP-7) | Partial per-method operations | Engine method-specific qualification | Method-specific routes | Partial views | Live scientific matrix and A–H journey results remain outstanding; no universal tolerance invented |

## Owner source index

- Authorship: [protocol command](../Sources/ExperimentKit/StudyProtocolAuthoring.swift),
  [draft replacement](../Sources/ExperimentKit/DraftAuthoringTransaction.swift),
  [repository](../Sources/ExperimentKit/ExperimentRepository.swift),
  [field policies](../Sources/ExperimentKit/ManifestDraftEdits.swift),
  [Python authoring](../Server/steerlab_server/experiment/authoring.py).
- Designs and assembly: [study management](../Sources/ExperimentKit/StudyManagementController.swift),
  [design library](../Sources/ExperimentKit/StudyDesignLibrary.swift),
  [template store](../Sources/ExperimentKit/StudyTemplate.swift),
  [study packs/coauthoring](../Sources/ExperimentKit/StudyAuthoring.swift),
  [task records](../Sources/ExperimentKit/TaskPromptsImport.swift),
  [seat casting](../Sources/ExperimentKit/SeatCasting.swift).
- Interfaces: [Swift dispatcher](../Sources/ExperimentKit/ExperimentCLIRunner.swift),
  [Swift HTTP](../Sources/ExperimentKit/WebServer.swift),
  [Python client](../Server/steerlab_server/client_cli.py),
  [engine CLI](../Server/steerlab_server/cli.py),
  [runtime method/path role census](../Server/steerlab_server/api/route_roles.py).
  The census is authoritative for Python route roles and is tested against
  registered routes; the matrix does not replace it with endpoint counts.
- Models and clusters: [local installer](../Sources/ExperimentKit/LocalModelInstaller.swift),
  [engine install](../Server/steerlab_server/api/model_install.py),
  [profile schema](../Sources/ExperimentKit/ClusterSiteProfileSchema.swift),
  [shared preview](../Sources/ExperimentKit/ClusterSitePreview.swift),
  [cluster commands](../Sources/ExperimentKit/ClusterCLIRunner.swift),
  [housekeeping](../Server/steerlab_server/api/housekeeping.py).
- Execution/custody: [remote job owner](../Sources/ExperimentKit/StudyRemoteJobController.swift),
  [bundle submission](../Sources/ExperimentKit/StudyBundleSubmissionController.swift),
  [auto-import](../Sources/ExperimentKit/EvidenceAutoImport.swift),
  [chain import](../Sources/ExperimentKit/EvidenceChainImport.swift),
  [Python runner adapter](../Server/steerlab_server/client/runner.py),
  [Python ownership](../Server/steerlab_server/api/job_ownership.py).

## Common operation contracts

Authoring requests identify one workspace and study, reviewed exact file bytes or
external digest, and intentional changes. A stale refusal requires a new review,
not a hidden refresh/retry. Pin operations consume real bytes and compute their
hashes; generated hashes never come from a coworking model. Frozen manifests and
runs remain immutable. Design and data operations additionally need destination
identity and overwrite/adoption rules appropriate to their artifacts.

Execution requests identify stage, target capabilities, pinned inputs and runtime
requirements. Persisted job references bind server/site and workspace before an
asynchronous boundary. Stopping observation is distinct from cancelling work.
Import records the verified archive, output destination and originating job;
cleanup depends on verified local custody and current dependency facts, not on
age or a successful download message alone.

Python runtime authorization uses the existing runner/workbench role census:
runner allows runner/both operations; workbench retains its intended operations.
Authentication is separate. A derived execution artifact remains legitimate
runner output. Surface parity does not require MLX and CUDA to execute the same
technique, but a restriction must be discoverable and actionable.

## Verification identities and next work

The development Swift CLI reports `swift-app 0.9.5+45655ae1`. It is an unstamped
build product resolving resources from this checkout: 5/6 developer resource
families, with the packaging manifest intentionally absent. It is **not** an
installed-distribution qualification. The installed app's plist reports 0.9.5;
its source commit was not inferred from that version and it was not relaunched.
No live remote service, model download or scheduler allocation was exercised.

The latest checkpoint passes 277 SteeringKit, 4,418 ExperimentKit and 5,896 Python
tests (9 Python skips). Those counts qualify the changes recorded in the ledger;
they do not upgrade an unverified matrix cell. Production numerical qualification
and interactive UI journeys are still separate gates.

Priority remains: finish context/write safety and return truthful operation
results; expose designs, model preparation and study assembly over the real
owners; complete shipped method/coauthoring guides; implement profile coauthoring
and bounded cleanup; then close all journey and qualification gates. WP-5 and WP-6
remain in scope. Expanded advanced-method and per-field cells, executable journey
links, and exact missing adapter mappings are still WP-0/WP-7 work, not implied
complete by this initial inventory. Landing belongs to the maintainer through the
designated reviewing/integration agent after diff review and both suites.
