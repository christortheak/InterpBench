# Review — `codex/python-design-interviews` at d13b7fc (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`82da781` (current main) to `d13b7fc`, one commit, 39 files,
+2,370 / −271. Main is an ancestor; the branch fast-forwards. Worktree
`/private/tmp/interpbench-workflow-handoff` clean at the tip, no venv. No
edits were made to the branch.

## 1. Verdict

**Landable by fast-forward.** Both suites are green on d13b7fc (§2). The design
identity is the load-bearing piece of this slice and it is done
carefully on both sides; the Python design owners mirror the Mac's
admission rules; the lazy-import change is mechanically proven; the
shared interviews are generated from one maintained source and gated.
Two notes (§4), neither blocking.

## 2. Verified independently

| Check | Result |
|---|---|
| `scripts/ci/audit-design-lazy-imports.py` (runtime/validation bodies of the variant decoder and panel validator vs `82da781`, negative controls) | passes |
| `scripts/ci/check-study-interviews.py` (WorkspaceSeed sources equal the packaged Python copies and the compiled Swift text) | "match both packaged clients" |
| Client reference check, task-prompt parser audit, bridge gates normal and `--release` | all pass |
| Vocabulary in the diff and commit message; `git diff --check` | clean |
| Python `DEFAULTS` for the portable identity vs the Mac manifest's decode defaults | identical set: studyKind, multiAgentIncludeBaseline, concepts, conditions, variantConditions, seeds `[20260610]`, temperature 0, maxTokens 2048 from `decodeIfPresent ?? …`, plus `recordTokenIDs` as the one stored default. Optional fields (promptMode, reasoningEffort, modelRevision) are omitted on encode and dropped as null on the Python side; required ones are present on both. |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 6,007 passed, 9 skipped, 8 warnings, matching the branch's claim |
| Full Xcode beta suite from the venv-less worktree, `TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild` | `TEST SUCCEEDED`: 277 SteeringKit + 4,578 ExperimentKit, matching the branch's claim; the four parity tests ran the real Python client |

## 3. What the code does

**Portable identity (`design_identity.py`, `PortableDesignIdentity.swift`).**
Same material on both sides: `{schemaVersion, study, semanticScenario}`
with the study re-encoded, lifecycle fields removed, `createdAt` blanked,
`status` forced to draft, nulls dropped except `validationHash`, five
opaque JSON blocks passed through untouched. Same framing: length-prefixed
UTF-8 strings, integers as decimal text, non-integral doubles as the
big-endian IEEE bit pattern in hex, arrays and objects length-prefixed,
object keys sorted by UTF-8 bytes. Booleans are distinguished from
integers on both sides (Python checks `bool` before `int`; Swift checks the
CFBoolean type id before `objCType`), and 64-bit seeds survive because
Swift reads `stringValue` for integral `NSNumber`s rather than going
through `Double`. `-0.0` frames as `i0;` on both. The parity test hashes a
template containing `UInt64.max`, 2^53 + 1, a 1e-15 temperature, a
non-ASCII description, a pinned-absent validation hash and a `pipeline`
block with a null value, on both sides, and checks a label change leaves
the hash alone while a `maxTokens` change moves it.

**The new manifest key.** `TemplateProvenance.hashAlgorithm` is an
optional Swift field, omitted when nil, so every existing manifest encodes
byte-for-byte as before. Python-minted drafts set it to `portable-v1` and
stamp the portable hash; Mac-minted drafts keep the legacy stamp with no
algorithm. The lineage reader interprets both and returns
diverged-plus-revised for any algorithm it does not know. This is
provenance on a new draft, not a precondition inside content-hashed
bytes, and no frozen study, run or old stamp is rewritten. The Python
store round-trips the key because `Document` preserves unknown keys; the
old comment saying a server re-save drops it is correctly removed.

**Design owners (`design_files.py`, `study_designs.py`).** Ordinary-path
admission walks each component with `lstat` and refuses symlinks and
non-regular ancestors. Reads require name and schema agreement and a
decodable study body. `create` follows the Mac's order (source lock,
library lock, new destination lock), reuses an unchanged lineage match by
portable hash plus panel equality, otherwise publishes a new design with
`publish_new` (no replacement). `update` requires both reviews and that
the source's lineage names the destination; replacement is atomic via
`os.replace` on a temp file. `instantiate` strips lifecycle fields,
applies decode defaults, stamps provenance, rechecks the pinned prompt
bytes, re-derives instrument scope from those bytes, casts, and saves
under the create-only rule. `batch` validates the whole shape and the
top-level review first, then treats each row as an independent
publication with a shared group stamp, classifying each failure and
keeping successes visible; the CLI returns 65 with `changed` truthful.

**Panels (`design_panels.py`, `panel_documents.py`).** Semantic
derivation blanks model bindings and strips agent pins with warnings for
mixed models, no model, or unpinned seats. Casting requires exactly one
of `agents` or `seats`, every seat present, agents under `runs/` with a
matching digest and base model, decoded through the existing `ModelVariant`
model, and the bound panel validated by the engine's `multi_agent.validate`.
`panel_documents.normalized` applies the Mac document's decode defaults
without coercing malformed shapes. The GPU generator is now imported
inside the two functions that execute, so validation and decoding stay
importable without torch; the audit proves the bodies are otherwise
unchanged and the `multi_agent.generate` module attribute survives as a
thin wrapper.

**Interviews.** One maintained source per intent under
`WorkspaceSeed/prompts/study-interviews/`, copied into the Python package
data (declared in `pyproject.toml`) and compiled into
`StudyInterviewText.swift` by the generator; `StudyCoauthoring.prompt(for:)`
now returns those constants and the 208-line inline text is gone. The
parity test asserts byte equality of all three across the two clients.
`workspace init` seeds the three files for new workspaces.

**Surface.** Nine client verbs declared once and dispatched by family;
`confirmAgent` maps to `conceptStudy` exactly as the Mac's legacy alias
does; derivation warnings are emitted as proper `designDerivationWarning`
advisories (the vocabulary code added earlier), so the envelope contract
the last review flagged is respected here.

## 4. Notes, not blockers

### N1 — Cross-client coverage of a cast agent

The Swift parity tests mint with `agents: []` and `seats: {first: null}`;
a real agent artifact is cast only in the Python-side tests. The variant
condition Python writes (`name`, `artifactPath`, `artifactHash`, embedded
`artifact`) follows the Mac's shape, but no test asserts that the Mac
decodes and re-encodes a Python-cast agent condition unchanged. One
fixture artifact in the parity suite would close it.

### N2 — Older Mac builds and Python-minted drafts

The handoff says it plainly and it bears repeating for deployment: a Mac
build without this branch reads a Python-minted draft's lineage stamp as
the legacy algorithm and reports divergence. Rebuild and install the app
from the landed tip before authoring Python-minted drafts in a workspace
the app also opens. This is the same app-before-anything rule already in
force.

## 5. Landing instructions

1. Both suites are green on d13b7fc (§2).
2. Fast-forward main to the tip. Deploy consequence: the app must be
   rebuilt before Python-minted drafts are opened in it (N2); nothing
   changes for the engine or the cluster.
