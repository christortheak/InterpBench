# Python reusable designs and study interviews

Implementation and independent-review handoff, 2026-09-06.
Branch `codex/python-design-interviews` starts from main `82da781`. It includes
main's reviewed assembly corrections and receipt/import gates. Main, installed
binaries, live services and study workspaces are unchanged by this branch.

## What changes for researchers

A researcher can discuss a study with an agent on either supported platform,
reuse an existing design, choose its agents or panel seats, and inspect an
ordinary draft in the Mac app. They do not need to reconstruct a manifest by
hand or repeat the study's measurement settings for each casting.

Both clients emit the same interview. It distinguishes concept derivation,
agent comparisons and panels; asks about the hypothesis, controls, held-out
inputs and measurement; and teaches real reviewed import/casting operations.
The interview is guidance for a collaborating agent, not an autonomous service:
it writes no files, chooses no scientific answers and makes no model calls.
The agent and researcher assemble the agreed pack, then preview and apply it.

Reusing a design carries the settings that determine what a result means.
Casting changes the selected agents while preserving task inputs, role text,
turn contracts and measurement declarations. Editing a design changes future
instances. Earlier studies and their original lineage remain intact.

## Command journey

Use `steerlab` for the lightweight Python client and `steerlab-cli` for the Mac
client. Supply `--root <workspace>` to Python, or set `STEERLAB_WORKSPACE` for
either. Keep research data outside the code checkout. All commands speak `--json`.

1. Emit `authoring study conceptStudy --json`, `agentComparison`, or `multiAgent`.
   Ask the researcher the relevant questions, inspect available inputs and use
   `authoring prompt <kind>` for the appropriate dataset-generation contract.
   Do not invent hashes or install unreviewed generated data.
2. Check `design list --json`, then `design inspect <name> --json`. Inspection
   returns the stored document, `designFileSHA256`, portable identity, and panel
   seat IDs when the pinned panel can be read. Catalog failures are reported.
3. Inspect each chosen artifact with `agent inspect <runs-relative-path> --json`.
   Use its `artifactFileSHA256`. Python adds inspection in this slice; the Mac
   also has agent discovery and standalone attachment. No command here installs
   a model or rewrites an agent artifact.
4. Prepare an explicit casting file. Comparisons use `{"agents": []}` for
   baseline, or `{"agents": [{"artifactPath": "runs/.../model-variant.json",
   "artifactFileSHA256": "<reviewed digest>"}]}`. Panels use
   `{"seats": {"first": null, "second": {"artifactPath": "runs/.../model-variant.json",
   "artifactFileSHA256": "<reviewed digest>"}}}` with the design's actual seat
   IDs, every seat present, and null explicitly selecting baseline.
5. Run `design instantiate <design> --casting <file> --file-sha256 <designFileSHA256>
   --study-name <new-name> --json`. Read the actual destination name; occupied
   names receive a suffix. Changed design/agent/prompt/panel pins refuse.
   Instrument scopes are derived again from the reviewed prompt bytes, and
   compiled panels use the study's model and sampling settings.
6. Read `experiment inspect <study> --json` on Python, or `experiment manifest
   <study> --json` on the Mac. Continue editing the ordinary draft through the
   existing authoring commands or app. Verification, freeze and execution gates
   still apply. A successful mint is not a scientific qualification.
7. Save reusable settings with `design save <study> --manifest-sha256 <digest>
   --name <design-name> --json`. Unchanged instances reuse their design; new or
   divergent sources create a separate entry. Save can read a frozen source;
   it never edits that source or its runs. Optional description/name choices
   apply only when a new entry is created.
8. Revise an existing design with `design update <design> --study <study>
   --manifest-sha256 <source-digest> --file-sha256 <design-digest> --json`.
   The source must name that design in its lineage. Both reviews must still
   match. Design name, description, creation time and parent lineage are retained.
   `design describe <design> --description <text> --file-sha256 <digest>` changes
   the note alone, leaving the design's scientific identity unchanged.

For a study without an existing design, assemble the interview's `{study,files}`
pack and use `pack preview` / `pack apply --review-sha256`. The app's Paste Study
JSON reaches the same Mac owners. Then save the reviewed study as a design for
subsequent use. Pack intake remains create-only and preserves existing inputs.

## Batch semantics

`design batch <design> --rows <file> --file-sha256 <digest> --json` accepts
`{"rows":[{"casting":{"agents":[]},"studyName":"baseline"}]}`. Shape errors
and a stale top-level review refuse before the batch begins. Each row is then
an independent publication, with a shared `batchGroup` provenance stamp.

The result lists actual minted names and each failed row's issue/repair. A
partial batch returns refusal or failure and reports whether it changed the
workspace. Keep successes and retry only repaired failed rows. Repeating the
whole batch creates additional studies. No submission occurs during minting.

## Design identity and historical evidence

The Mac's original design hash incorporates Swift's encoded manifest. Python
cannot use its own JSON encoder and honestly call that the same identity.
This checkpoint therefore adds an explicitly named **portable-v1** authoring
identity, exposed as `portableContentHash` and `portableHashAlgorithm` by both
clients. The Mac's existing `contentHash` and legacy minting rule remain intact.

A new Python-minted study records `templateProvenance.hashAlgorithm` alongside
its template name and portable hash. An absent algorithm retains the original
Swift interpretation. The app's lineage reader understands both, distinguishing
study divergence from later revision of the design. Unknown algorithms cannot
certify agreement. No existing design, frozen study, run or old stamp is migrated.

Use the updated Mac app/CLI alongside the new Python client before editing
newly Python-minted drafts. Older Mac builds do not interpret the portable
lineage stamp. The interchange tests use matching source builds; this task
does not install or replace the app.

This is provenance for a newly authored draft, **not a revision/precondition
inside the manifest**. Stale-write protection still uses external file SHA-256s
and the shared file-lock protocol. Freeze hashing itself is unchanged.

Portable identity covers the stripped study settings and semantic panel
path/hash, excluding design labels/history and instance-specific bindings.
It uses a length-framed JSON tree, UTF-8 key ordering, exact integers and binary
floating-point values, with the Mac manifest's decode defaults. Null validation
pins retain their distinct meaning; arbitrary JSON method blocks retain null
values. Scope and casting derivation still have their usual effect on a minted
study's settings and therefore on lineage agreement.

## Implementation and review map

- `client/design_commands.py`: declarative verbs and thin CLI dispatch.
- `client/design_files.py`: ordinary-path admission, catalog, exact reviews,
  atomic replacement and unique destinations.
- `client/study_designs.py`: source/design transactions, save/update/mint/batch,
  unchanged-source policy and scope derivation.
- `client/design_panels.py`, `client/panel_documents.py`: semantic derivation,
  explicit seat/agent reviews, shared document defaults and engine validation.
- `client/design_identity.py`, `PortableDesignIdentity.swift`: portable identity;
  `StudyDesign.swift` interprets the new optional lineage algorithm.
- `WorkspaceSeed/prompts/study-interviews/study-*.md`: maintained interviews.
  `check-study-interviews.py --write` regenerates packaged Python copies and
  compiled Swift text. A gate verifies both; wheel installs need no checkout.
- The existing model-variant decoder and panel validator now defer importing
  the GPU generator until execution. Their runtime/validation bodies are checked
  against `82da781` by `audit-design-lazy-imports.py`, including negative controls.
- `test_client_study_designs.py`: real CLI/service journeys, immutable sources,
  review drift, path refusal, partial batches and lightweight panel authoring.
- `PythonStudyDesignParityTests.swift`: real Python operations followed by the
  app's shared owners, Mac update and Python reuse, panel cast round trips,
  exact interview equality, and identity checks including large seeds.

The app already uses these shared owners; this slice adds no alternate SwiftUI
editor or new Python runner-authoring route. Tests of those owners establish
artifact interchange, not an interactive UI walkthrough or numerical parity.

Publication uses shared cooperating-writer locks and atomic file writes.
New panel inputs are immutable versions; failure or process interruption after
preparing an input can leave an unreferenced version. This is not a multi-file
crash transaction or protection against a hostile writer ignoring locks.

## Handoff gates and next work

Before landing, the maintainer through the designated reviewing/integration
agent must read the actual diff and approve it, with both full suites and the
AST gates passing. Local verification is not self-approval. Recheck main's
ancestry before integration and incorporate any later fixes first. No site
names or study-case vocabulary belongs in code/commits; secrets stay in Keychain
or the established non-Mac token-path mechanism; runs remain immutable.

Use Xcode beta, `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1`, external
`/private/tmp` scratch, serial Xcode testing and `CLANG_COVERAGE_MAPPING=NO`.
For real cross-client tests put `TEST_RUNNER_STEERLAB_TEST_PYTHON=<client-python>`
in the shell environment **before** `xcodebuild`, not after it as a build setting.
The [previous workflow](PYTHON-STUDY-ASSEMBLY-WORKFLOW.md) contains the complete
invocation. Results and logs belong in the
[validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md).

Next: model preparation and remaining scenario/casting/pipeline authoring,
then cluster-document coauthoring and managed remote lifecycle (WP-5/6), then
complete researcher/agent/UI journeys and scientific/GPU qualification (WP-7).
Automatic casting expansion and Python standalone agent discovery/attachment
remain explicit surface gaps; this slice supplies reviewed casting into designs.
