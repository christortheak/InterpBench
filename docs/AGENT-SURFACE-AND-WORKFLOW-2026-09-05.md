# One product, four surfaces: what agents and researchers can reach, what is broken, and what 1.0 requires

Date: 2026-09-05. Tree: `main` at `bfd13a5` (the landed maintainability
refactor on 0.9.5). Installed app: 0.9.5+b6949ffd.

**Maintainer handoff amendment, 2026-09-05:** This synthesis is committed
alongside the [researcher workflow implementation plan, revision 4](RESEARCHER-WORKFLOW-IMPLEMENTATION-PLAN.md).
That plan contains the explicit concurrency, repository-hygiene, immutable-
artifact, and maintainer-review constraints for implementation. The maintainer
has confirmed that **WP-5 and WP-6 remain in scope, with estimation requests
removed**. Earlier release-deferral recommendations below are historical
proposals and do not remove these packages from the handoff. This amendment
records the maintainer's clarification; it does not attribute new findings
to the original audit team or authorize implementation to begin.

Status: **synthesis for the agent team, revision 2. No code, contract, or
user doc was edited to produce it.** It replaces the surface-parity audit of
the same date and folds in the researcher-and-agent workflow assessment
(revision 2) written the same day by another agent. Where the two reviews
disagree, this document says so and says which view it takes. Revision 2
adopts that assessment's OPS-06 in place of this document's earlier §4.1
recommendation, corrects a matrix cell it caught, and takes its coverage
classification.

Build identities matter: the verb tables and doc checks below were read from
the source tree at `bfd13a5`; the installed app used for the app-side doc
check is 0.9.5+b6949ffd, one landing older. Nothing in the app column was
exercised against a binary built from `bfd13a5`.

Two kinds of statement appear below and are labelled throughout:

- **Verified** — the author of this document reproduced the count from the
  tree or read the cited source lines at `bfd13a5`.
- **Reported** — taken from the workflow assessment and read at the cited
  lines, but not reproduced live (no job was killed, no import misrouted).

Nothing in either review reproduced a defect against live cluster work.
Reproduce before fixing, against throwaway state, never against a live job
store.

## 0. The direction this document answers to

These are the maintainer's standing rules for the project, stated so the
priorities below are checkable against them rather than against taste.

1. **Pre-1.0 means no avoidable debt.** There is no launch, no blog post,
   no external user. Compatibility layers exist only for our own tests and
   unfinished work; they are removed before 1.0, not carried past it.
   `docs/BRIDGE-RETIREMENT.md` already states this for the Swift bridges.
2. **The authoring client owns the study; the running hardware executes
   it.** (Ruling of 2026-08-28; `PORTABILITY-CONTRACTS.md` §9.1 names the two
   service roles, runner and workbench.) In this study the authoring client is
   the Mac workspace and the engine is a runner and a cache. Evidence comes
   home verified; nothing under `runs/` is ever edited; frozen manifests are
   never hand-edited.
3. **The workspace `AGENTS.md` is the product's promise to an agent.** If an
   agent cannot complete a workflow from the contract plus `--help`, the
   contract is wrong, not the agent.
4. **The app is a human interface to the same services**, not a second
   product. An affordance that exists only in the app is either promoted to
   an operation or declared exploratory.
5. **Nothing lands on the word of the agent that wrote it.** Diffs are read,
   both suites are run, and the AST audit gates any move that claims to be
   mechanical.
6. **Science is the researcher's.** Method rulings (dose direction,
   tolerances, promotion) are made by the researcher in conceptual terms; the
   tooling implements them and never quietly defaults them.

## 1. Verdict

The machinery is largely there and the surfaces are mechanically consistent
where a generator or a twin test enforces it. Three things stand between this
tree and a 1.0 an agent can operate dependably:

1. **Seven source-verified defects, one of them dangerous** (§2). An agent
   following the contract's own verb table can fail a running fan-out.
2. **The agent contract stops short in exactly the places an agent needs it**
   (§3): the bundle submission sequence, the readiness gate, the server-only
   families, and the datasets it assumes are hand-authored.
3. **A ruling not enforced, and two decisions not made** (§2 OPS-06, §4).
   The runner/workbench separation is ruled and censused but no request is
   checked against it, so a runner deployment still serves workbench
   authoring. Separately: whether agents may install models and instantiate
   templates. Today the app can do both; the install endpoint exists and no
   CLI calls it; templates have no programmatic path at all.

The original synthesis proposed a narrower release boundary for other product
work. The maintainer's handoff amendment above supersedes that proposal for
WP-5 and WP-6; use the companion plan for their bounded implementation scope.

## 2. Defects (fix first)

| ID | Finding | Status | Priority |
|---|---|---|---|
| OPS-01 | `steerlab-server jobs list` constructs `JobManager()`, whose constructor defaults `sweep_orphans=True`; the sweep marks a pending Slurm parent failed and cancels its shards, from a second process that owns nothing. `jobs list` is in the declared agent-path verb table. | Verified: `cli.py` `_jobs`; `api/jobs.py` `__init__`, `_sweep_orphans` | **Highest.** Fix this week. |
| OPS-05 | `study submit` and `remote submit-bundle` both default `--verb run`; the reference records a sweep-submitted-as-run incident. | Verified: `cli.py` `verb = … or "run"`; `CLI-REFERENCE.md` §gaps | High. Refuse an omitted verb before staging. |
| OPS-03 | The evidence auto-import service captures a workspace root, but the importer resolves the global runs directory; a workspace switch mid-download can land evidence in the wrong workspace with the ledger in the other. | Verified: `EvidenceAutoImport.swift`, `ClusterClient.swift` importer | High. Pass the captured root through download, verify, publish, adopt, ledger. |
| OPS-02 | Recent and active job records carry id, verb, study, state — no server identity. Cancel and resume go to the current connection. | Verified: `StudyRemoteJobController.swift`; `ExperimentPanel.cancelSweep` | High. A job reference carries its site and workspace. |
| OPS-04 | The profile advertises `httpTransfer` / `externalTransferRequired`; the app decodes the flag; neither the upload and download routes nor the client enforce it. | Verified: `api/profile.py`; no reference in `routes.py`; `ClusterClient.swift` decodes only | High on policy-constrained sites; otherwise medium. |
| OPS-06 | The portability contract rules two service roles (runner, workbench) and censuses every route into one of them (`Server/tests/route_roles.py`); nothing at runtime reads the census, so one FastAPI app serves both and a runner deployment exposes workbench authoring. Privilege is authentication, not role: mutating routes are already privileged by default, which does not fix this. | Verified: `PORTABILITY-CONTRACTS.md` §9.1, §11; `route_roles.py`, `test_route_roles.py`; `api/app.py` mutating-by-default | High. A runner refuses workbench authoring with a typed repair; a workbench keeps it; derived GPU artifacts (vectors, adapters, readers, evidence) remain legitimate execution output; import never silently adopts a divergent definition. |
| UI-01 | Imported pipeline evidence is rendered only inside the compute-target-is-server branch, so it vanishes when the researcher selects local compute. | Verified: `ExperimentsPanelView.swift` | Medium. Visibility follows presence in the workspace. |
| SCI-01 | Two authoring strings say matched-norm random controls "prove" an effect. The extraction reference says correctly that they do not rule out a shared confound. | Verified: `StudyAuthoring.swift` lines 94 and 293 | Low effort, do now. One-line wording fix. |

Operational rule until OPS-01 is fixed: **query jobs through the running
server (`steerlab-cli remote jobs --site …`), never with
`steerlab-server jobs list` on a login node while a controller owns work.**

Also verified, not a defect: the Swift test suite hangs if `SteerLab.app` is
running at the same time. Which test blocks is unidentified. Worth a ticket
because it will catch anyone who tests with the app open.

## 3. The contract versus the surfaces (verified inventory)

Six surfaces were compared mechanically; the commands are in the appendix.

| Surface | Size |
|---|---|
| Swift CLI `steerlab-cli` | 73 family-verb entries in 14 generated tables |
| Python client `steerlab` | 30 verbs |
| Server CLI `steerlab-server` | about 50 verbs; 19 declared on the agent path |
| HTTP API | 129 routes; 22 privileged prefixes |
| App | 10 workbench sections; 65 distinct engine endpoints called |
| Contracts | workspace `AGENTS.md` template (1,602 lines), repository `AGENTS.md`, `ONBOARDING.md` |

### 3.1 What holds

- Both reference-doc generators pass on this tree; docs and parsers cannot
  drift.
- The server's 19 agent-path verbs are twin-tested against the Swift tables,
  server-only members declared by exception.
- The workspace contract names no verb that does not exist.
- The app calls no route that does not exist.
- The two authoring CLIs differ in shape (15 Swift-only, 4 client-only
  `experiment` verbs) and `PORTABILITY-CONTRACTS.md` §1.4 documents the
  mapping. Not a finding.

### 3.2 Where the contract fails an agent

1. **The submission sequence.** The contract names `remote submit-bundle`
   and `remote jobs`, never `remote package` or `remote upload`, and
   `submit-bundle` takes the path `upload` returns. `remote capabilities`,
   the readiness gate every chain should wait on, is unnamed, as are
   `remote logs`, `remote cancel`, `remote import`, `remote import-chain`.
   (`remote resubmit` and `experiment confirm` ARE documented; an earlier
   pass of this audit missed them, and they must not become duplicate
   tasks.) Twenty-three Swift verbs are unnamed in total; the `cluster`, `install`,
   and `docs` ones are reasonably delegated to ONBOARDING, the `remote` ones
   are not.
2. **Server-only families have no index.** The contracts name eight server
   verbs. OptVec, SAE, fine-tuning, Gemma Scope, the impact ledger, bundle
   inspection, housekeeping, and most of J-lens are reachable only through
   the reference's later sections.
3. **The extraction paragraph is wrong for two recipes.** The contract says
   "everything the extraction reads is these two files" (positive and
   negative). That is false for grand-mean and designated-reference
   extraction. (Reported by the workflow assessment; verified at
   `AgentContract.swift` line 524.)
4. **Concept datasets are file-authored, and the contract never says so.**
   The app's Data section creates concepts, generates stimuli through the
   prompt helpers, imports probes and neutral corpora. The CLI offers
   `authoring prompt` and expects files. The contract implies this and
   should state it with the layout.
5. **Advanced verbs sit outside the envelope system** (reported): some
   server families answer in prose rather than the declared envelope;
   `rescore-style` is not an accepted submitted-study verb; the standalone
   battery has no bundle route; artifact attachment exists in the Swift
   store and the Python client but not in the Swift CLI.

### 3.3 Capability matrix

**✓** reachable · **—** absent · **doc** absent by documented design ·
**(field)** reachable as a protocol field · **?** app coverage not verified
view by view.

| Capability | Swift CLI | Python client | Server CLI | HTTP API | App | Contract |
|---|---|---|---|---|---|---|
| Author a study (create, attach, declare, pin, set-*, freeze) | ✓ | ✓ (field) | — | ✓ authoring routes | ✓ Studies | ✓ |
| Execute stages (extract … promote) | ✓ | doc | ✓ | ✓ | ✓ | ✓ |
| `confirm`, `rescore-style`, `extract-stability` | ✓ / ✓ / — | — | ✓ / ✓ / ✓ | ✓ | ? | ✓ / — / — |
| Package → upload → submit | ✓ | ✓ | ✓ | ✓ priv. | ✓ | **submit only** |
| Jobs: watch, cancel, resume, import | ✓ | ✓ | ✓ (`jobs list` unsafe, §2) | ✓ | ✓ | **`jobs` only** |
| Readiness gate `remote capabilities` | ✓ | ✓ | — | ✓ | implicit | **—** |
| Cluster lifecycle (15 verbs) | ✓ | doc | — | — | ✓ wizard | 8 of 15 |
| Vector parity, mirror, norm backfill | ✓ | — | ✓ | ✓ | ? | `compare` only |
| Model capability records | ✓ | ✓ | ✓ | ✓ priv. probe | ? | probe only |
| **Model install** (API-accessible, no CLI) | **—** | **—** | **—** | ✓ priv. `/api/models/install` | ✓ Compute | — |
| **Templates (design library)** | **—** | **—** | **—** | — | ✓ Templates | — |
| **Concept / corpus authoring** | prompt only | prompt, `concept import` | — | ✓ (engine page) | ✓ Data | implied |
| Local playground chat (Swift HTTP endpoint, no CLI) | — | — | — | Swift server `/api/chat` | ✓ | — |
| `remote chat` (server variant) | ✓ | — | — | ✓ | ✓ | **—** |
| Agents: variants, OptVec | promote, variants | — | ✓ `optvec` | ✓ | ✓ | promote only |
| Multi-agent panels | ✓ | — | ✓ | ✓ scenario (engine page) | ✓ | ✓ |
| J-lens (9 verbs) | doc | — | ✓ | ✓ | renders | 4 of 9 |
| SAE, Gemma Scope, fine-tuning, ledger, housekeeping | doc | pin-sae only | ✓ | ✓ | ? / ✓ adapters | **—** |
| Results, import | `cluster import` | `bundle import` | — | ✓ | ✓ | ✓ |

## 4. Decisions the maintainer has to make

These are not defects. Each is a place where two surfaces embody different
answers and no document records which is intended. Recommendations are the
author's; the decision is the maintainer's.

### 4.1 Server-side authoring — resolved into OPS-06

An earlier revision of this document treated the engine's authoring routes
as an open design question and recommended retiring them. That was the wrong
frame. The ruling exists (`PORTABILITY-CONTRACTS.md` §9.1: runner versus
workbench) and the census exists (§11); what is missing is runtime
enforcement, which is defect OPS-06 in §2. Retiring authoring would have
removed a legitimate workbench capability to fix a runner problem. The
inventory that prompted the question stands as evidence for OPS-06: of the
engine's 129 routes, 13 experiment, 10 concept, 8 authoring, the scenario
save/run, neutral-corpus, reader-score, and model-variant routes are never
called by the app and serve the engine's own web page.

### 4.2 Templates and model installation

**Verified.** Model installation has a privileged HTTP endpoint and no CLI
on any product: API-accessible, missing a convenient operation. Templates have
no programmatic path at all: genuinely app-only. Rule 4 says each is promoted
to an operation or declared exploratory.

**Recommendation:** promote both. A `template` family (list, instantiate,
save-back) is small, and the design library is exactly the thing a
coauthoring agent should reach for. A `model install` verb should run the
same preflight the privileged route runs, so agents pre-stage models the
same way a person does instead of through the bootstrap flag alone.

### 4.3 Transfer policy

OPS-04 is a defect, but the shape of its fix is a decision: one transport
abstraction governed by the site profile, or two paths (HTTP bundles, rsync
imports) each checking the policy. **Recommendation:** enforce at both
boundaries now, unify later; the enforcement is small and the unification is
not.

## 5. Earlier release-scope proposals — superseded for WP-5 and WP-6

**Handoff instruction:** The deferrals in this historical section are not
authorization to omit WP-5 or WP-6. The maintainer has kept both packages in
scope and removed requests for estimates. Implement their bounded versions
when the implementation handoff is authorized, following the companion plan
and its acceptance criteria. Broader release claims remain the maintainer's
decision.

The workflow assessment's §1 describes a product in which an agent turns
institutional cluster documentation into a proposed site profile, assembles a
verified draft from a conceptual interview through supported operations, and
drives a managed remote-cleanup lifecycle. These are good goals. They are
not gaps in what the current tree promises, and treating them as 1.0 blockers
would put months of feature work ahead of the defects in §2.

What 1.0 does owe on each:

- **Coauthoring:** the contract must be accurate (§3.2) and the attach and
  pin operations must be callable, so an agent never has to invent a hash or
  hand a routine attachment step back to the researcher. Building the
  interview-to-draft assembly is post-1.0.
- **Cluster onboarding:** the contract says "bring your own profile" and the
  wizard edits or imports one. The document-to-profile workflow is
  post-1.0; the scope statement (Slurm or no scheduler) belongs in the
  contract now.
- **Remote cleanup:** `cluster import` reports purge eligibility and deletes
  nothing, by design. A cleanup-plan-and-execute operation is post-1.0; the
  rule that agents never improvise filesystem cleanup belongs in the contract
  now.
- **Method-specific guides** (per technique: purpose, inputs, prompts, split
  roles, operations, capabilities, outputs): the recipe, scope, training,
  and validation pages landed today are the raw material. Assembling them
  into per-technique contracts is a documentation slice for 1.0; the wrong
  extraction paragraph (§3.2 item 3) is fixed now.

## 6. Qualification

`docs/VALIDATION-MATRIX.md` is a pre-registration of fifteen checks with no
result rows, and it says so. The 2026-09-05 science fixes each carry
regression tests; that establishes covered behaviour going forward, not the
empirical qualification of the production configurations researchers will
run. The impact ledger verb exists to find already-produced artifacts those
fixes reach. Both reviews agree: publish bounded qualification results with
pinned configurations before 1.0, and make qualification status discoverable
before execution (the J-lens `qualify` record is the model to copy).

## 7. Sequence and exit conditions

| Order | Workstream | Exit condition |
|---|---|---|
| 1 | Defects, §2 | Listing jobs changes nothing; an omitted verb refuses before staging; a job reference carries site and workspace; an import lands in its captured workspace or stops; transfer policy is enforced at both boundaries; a runner refuses workbench authoring while a workbench keeps it; imported pipelines stay visible offline; no UI string says a control proves anything. Each reproduced first, on throwaway state. |
| 2 | Contract, §3.2 | An agent completes package → upload → gate → submit → watch → import from the contract plus `--help`. Server-only families indexed. The extraction paragraph correct for all four recipes. Dataset layout stated. |
| 3 | Decisions, §4 | Each recorded in the contract with its rule. Templates and model install promoted or declared exploratory. |
| 4 | Bridge retirement (existing gate) | The four Swift bridge files removed; the ratchet script has nothing to protect. `docs/BRIDGE-RETIREMENT.md`. |
| 5 | Qualification, §6 | Result rows in the validation matrix for the declared production configurations, with unavailable cases explicit. |
| 6 | Acceptance journeys, §8 and companion plan §12 | All in-scope journeys pass on declared local and remote configurations, including interruption, WP-5 document-driven onboarding, WP-6 cleanup, and cross-interface handoff. |

Workstreams 1 and 2 can proceed in parallel today. Workstream 3 needs the
maintainer's rulings first. Nothing in 5 or 6 is a substitute for the
suites, and the suites are not a substitute for 5 or 6.

## 8. Acceptance journeys (adopted from the workflow assessment, amended)

- **A. Conceptual steering study.** Researcher states construct and
  question; agent explains method choices, obtains decisions, authors or
  delegates datasets, assembles a verified draft through supported
  operations, runs screen and confirm. No invented hash, no manifest edit,
  no private function, no routine attachment handed back.
- **B. Multi-agent study.** Scenario, materials, visibility, casting,
  controls, transcript outcomes authored through supported operations; the
  app shows the same design and evidence.
- **C. Cluster from documentation.** Under the maintainer's scope decision,
  use the companion plan's WP-5 and Journey C: the agent produces a
  source-supported profile, asks unresolved questions, previews and validates
  it, then connects and qualifies the target. Human authentication remains a
  bounded handoff.
- **D. Interruption and recovery.** Submit remotely; app closes; agent
  reconnects, observes without altering, resumes if appropriate, waits for
  complete evidence, imports into the originating workspace. Switching
  servers or workspaces changes nothing about ownership.
- **E. Verified evidence and managed cleanup.** Imported results remain
  visible with no server connection. Cleanup remains in scope under WP-6:
  plan exact eligible targets, recheck local custody and dependencies, apply
  authorized cleanup, and record the result through supported operations.
- **F. Agent-driven template and model preparation.** (From the workflow
  assessment.) An agent lists and instantiates a design into a named
  workspace and prepares its model on the execution target through the same
  preflight the app uses; the researcher opens the draft in the app and sees
  the same state. Gated on the §4.2 decision.
- **G. Enforced workspace authority.** (From the workflow assessment.) An
  authenticated agent attempting workbench authoring against a runner gets a
  typed refusal naming the correct authoring context; it authors locally and
  submits pinned inputs; execution may produce vectors, adapters, readers,
  and evidence but cannot rewrite the study definition. This is OPS-06's
  acceptance test.

## 9. Rules for the agents picking this up

Read the companion plan's §13.1 before beginning. Stale-write protection must
not add fields to the manifest or alter scientific content/freeze hashing;
use external file-byte/hash preconditions with serialized comparison and
publication. The maintainer controls landing through the designated
reviewing/integration agent: diff read, both suites passing for the proposed
landing state, and the AST audit for every claimed mechanical move. Branch
commits are review artifacts and do not authorize merging or deployment.

- Reproduce a §2 finding on throwaway state before fixing it; never against
  a live job store or a workspace with studies in flight.
- Land through the standing method: a branch, the diff read by the
  maintainer's agent, both suites green, the AST audit for any move that
  claims to be mechanical.
- Carry regression assertions with any moved code; never restore a deleted
  implementation to resolve a conflict.
- No site names, hostnames, or study-case vocabulary in files or commit
  messages; this repository is public.
- Do not run `steerlab-server jobs list` on a login node while a controller
  is running, until OPS-01 is fixed.
- Passing suites certify covered contracts, not the journeys in §8 and not
  the qualification in §6.

## Appendix: reproducing the inventory

```bash
# Swift verb tables (generated regions) and both doc checks
grep -c "GENERATED:swift-" docs/CLI-REFERENCE.md
Server/.venv.nosync/bin/python -m steerlab_server.cli docs cli-reference --check
steerlab-cli --workspace <workspace> docs cli-reference --check
# Server declared agent-path table
Server/.venv.nosync/bin/python -c "from steerlab_server import cli_envelope as e; print(sorted((s.family, s.verb) for s in e.VERB_SPECS))"
# Python client verbs
Server/.venv.nosync/bin/steerlab --help
# HTTP routes
grep -rhoE '@router\.(get|post|put|delete|patch)\("(/[^"]+)"' Server/steerlab_server/api/*.py | sort -u
# App → engine endpoints (client files only; WebServer.swift is the app's own server)
grep -rhoE '"/api/[A-Za-z0-9_/{}.-]+' $(grep -rlE '"/api/' Sources | grep -v WebServer.swift) | sort -u
# Contract mentions
grep -oE "steerlab-(cli|server) [a-z]+ [a-z-]+" Sources/ExperimentKit/AgentContract.swift AGENTS.md docs/ONBOARDING.md | sort -u
# OPS-01 source
grep -n "def _jobs" -A 6 Server/steerlab_server/cli.py; grep -n "sweep_orphans" Server/steerlab_server/api/jobs.py
```
