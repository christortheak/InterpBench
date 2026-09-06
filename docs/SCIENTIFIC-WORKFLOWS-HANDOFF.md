# Scientific workflows and shared method guidance

Branch: `codex/scientific-workflows`, based on landed main `b369693`.
Scope: the advanced-operation and research-guidance slice of WP-3/WP-4.
Maintainer review and integration through the user remain the landing gate.

## What changes for researchers and agents

Both installed clients now expose `science list`, `science guide <method>` and
`science operation <operation>`, with normal help and JSON envelopes. A
researcher can choose a method conceptually; an agent retrieves its input
formats, scientific decisions, coworker author prompt and independent review
prompt without consulting a source checkout. Discovery performs no execution.

The Studies panel has a **Research methods and guides** sheet over the same
catalog and text, with links into Optimizations and study designs. It displays
public execution paths and restrictions alongside each operation. The Mac
workbench and Python service both expose `GET /api/science/catalog`,
`/api/science/guide/{method}` and `/api/science/operation/{operation}`. Python
runner-role deployments may serve these workspace-independent reads too.

The twelve guides cover extraction, readers, optimization, fine-tuning, J-lens,
J-space, SAE, stability, batteries, deferred judging, style rescoring and
multi-agent studies. They separate train/development/final-evaluation roles,
examples from validated datasets, and diagnostics from behavioral or causal
claims. Missing scientific decisions go back to the researcher; routine file
formats, pins and supported command discovery belong to the agent.

## Execution coverage and explicit boundaries

WP-3 allows either closing a missing entry point or recording its exact
supported operation mapping. This slice uses existing scientific owners and
records engine-only mappings instead of inventing generic study submissions:

| Operation | Public path and outcome |
|---|---|
| Extraction stability | Existing `steerlab-server experiment extract-stability`; activation capture followed by diagnostics outside runs. Python engine only. |
| Standalone capability battery | Existing `steerlab-server battery run`, including preflight `--dry-run`, plus generation-prompt/lint. Study-blind input and new battery report; format 3 is not a study control. |
| Style rescoring | Existing Mac `experiment rescore-style --run`; engine `experiment rescore-style --source` now has strict declared flags and a structured `runDirectory` result. Original generations are preserved. |
| Deferred evaluation completion | Existing `steerlab-server experiment complete-judgment` now has a full envelope, output identity, reuse state and typed errors; same original intake owner. |
| Deferred sweep completion | New `steerlab-server experiment complete-sweep-judgment <study> --awaiting-run <run> --judgments <file> --json`, over the existing intake owner. Can append/heal a draft recommendation. |
| Other advanced methods | Catalog maps reader, fine-tuning, J-lens, SAE/Gemma Scope and all nine OptVec verbs to their existing public interfaces and restrictions. |

Stability and standalone batteries have **no new HTTP, bundle or automatic
scheduler submission route**. This remains a boundary to consider in remote
lifecycle work. The catalog and UI state it explicitly. There is no generic
`science execute` verb and no transformation of diagnostics into fake study
manifests. J-space remains exploratory analysis, not training or causal proof.

The completion CLI accepts JSON lists, wrapped judgment lists and JSONL. It
preserves the issued packet and judge requirements and the owner's epoch and
coverage checks. Evaluation instruction-digest mismatch remains an owner
warning, not a newly invented hard gate. Sweep has no such claim field and
refuses it rather than discarding it. A completed evaluation is reused; a
completed sweep can heal its missing draft projection. The result reports
`runDirectory`, `awaitingRun`, `kind`, `reused` and `changed`.

The engine's three CPU adapters now use 64/65/66/70 for malformed input,
refusal, missing input and failure, including human mode. Old completion
callers that expected generic exit 1 should consume the envelope instead.
Unknown runtime failures remain failures; typed lifecycle gates retain their
identity. The scientific computation/publication bodies are unchanged.

## Maintained sources and review map

- `WorkspaceSeed/prompts/method-guides/`: catalog and method texts; edit here.
- `scripts/ci/check-science-resources.py --write`: copies package data and
  generates `ScienceResourceText.swift`; without `--write` it is the drift gate.
- `experiment/science_catalog.py` and `ScienceCatalog.swift`: read-only owners.
  The catalog and each guide return the hash of their exact shipped bytes.
- `client/science_commands.py`, `api/science_routes.py`, Swift CLI/WebServer:
  thin discovery adapters. New Python routes participate in the role census.
- `ScienceGuidesView.swift`: presentation over that same owner.
- `experiment/scientific_commands.py`: CPU CLI adapters. No new model loader,
  model concurrency mechanism, scheduler command or scientific arithmetic.
- `docs/AGENTS-WORKSPACE-DRAFT.md` and its shipping `AgentContract.swift`
  mirror: discoverability, input roles and the exact client/engine distinction.
- `test_science_workflows.py` and `ScienceCatalogTests.swift`: operation
  admission/publication journeys and real Python/Mac/HTTP byte parity.

The prior maintainer notes are documented in
[model/study equivalence](MODEL-STUDY-AUTHORING-EQUIVALENCE.md): panel compilation
intentionally changes the draft's kind and type to multi-agent; model plan
digests bind absolute cache/workspace roots and cannot move between deployment
mount views.

## Review and remaining qualification

Read the actual diff, run both full suites and the resource/reference/role and
bridge gates. Historical mechanical migrations retain their historical AST
audits; this slice intentionally changes transport/result behavior and does
not claim its CLI adapter changes are mechanical moves. Its extraction,
battery, analysis, sweep-judgment and evaluation-judgment owners are unchanged
from `b369693`.

Validation results and commands are recorded in
[the validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md). Tests cover
real CPU refusal/publication/reuse paths, example row schemas, strict discovery
flags, installed-resource access from outside the checkout, role census and
cross-client payload/text/digest equality. They do not establish live numerical
qualification, interactive UI usability or a live remote execution journey.

The validation pass also repairs a race in the existing local cancellation
test fixture: it now atomically publishes its complete JSON record before
the parent can cancel on path existence. Production cancellation code and
evidence assertions are unchanged. The validation history records the initial
failure, isolated rerun and final complete suite.

No installation or deployment occurs in this slice. Rebuild the app at the
maintainer's deployment step; deploy the engine after ongoing jobs, then
regenerate approved site configuration for the previously landed egress gate.
Do not modify frozen studies or runs, add revision preconditions inside hashed
manifest bytes, put secrets outside the prescribed credential stores, or add
site/study-case vocabulary. Recheck current main ancestry before integration;
only the designated maintainer reviewing agent lands the branch.

Next work is the bounded cluster onboarding and remote lifecycle/custody work,
followed by interactive and scientific qualification of the declared journeys.
