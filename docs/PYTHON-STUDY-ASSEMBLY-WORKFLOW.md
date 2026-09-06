# Python study assembly: workflow and review handoff

This checkpoint starts at main `ef3dec8`, including `3ea802a`'s terminal-job
receipt gate and immutable sibling reimport. It adds a bounded part of WP-3:
pack interchange and reviewed prompt/vector intake in the lightweight `steerlab`
client. It is prepared for independent review through the maintainer's agent;
it does not authorize integration or deployment.

## What changes for the researcher

An agent on Linux can now take a proposed study pack, show which files it will
create, and apply the reviewed proposal to the local workspace. The Mac app can
open that ordinary draft and continue the study. Conversely, a pack exported
from the Mac can be imported by the Python client. No GPU or model download is
needed to author these inputs.

This reduces hand editing and makes the decisions inspectable. A successful
import means a draft was saved. `verificationIssues` identifies unresolved
inputs or declarations; it does not establish scientific validity or readiness
to run. The usual verification, freeze and execution gates still apply.

## The client journey

Use `steerlab`, the Python client, for these commands. Keep study files outside
the code checkout and supply `--root <workspace>` or `STEERLAB_WORKSPACE`.

1. Prepare a complete study manifest, or JSON with `study` holding that manifest
   and optional `files` mapping workspace-relative paths under `prompts/` to
   UTF-8 text. Discuss the scientific choices with the researcher and use
   `steerlab authoring prompt <kind> --json` for supported dataset instructions.
2. Run `steerlab pack preview <pack.json> --json`. Inspect `result.files`,
   `referencedInputs`, `advisories` and the intended workspace. Preview does not
   write workspace files. Existing differing inputs and existing study names
   refuse; identical inputs can be reused.
3. Apply with `steerlab pack apply <pack.json> --review-sha256 <reviewSHA256> --json`.
   Use the returned token from this client's preview in the same workspace.
   A changed pack, named input or destination requires a new review. Apply
   creates a draft, strips freeze metadata and pins named task prompts, rubric
   and battery from actual bytes when no hash was supplied. A failed automatic
   pin is reported with its cause and leaves that pin unset.
4. Read `result.study.document`, `manifestFileSHA256`, `verificationIssues` and
   `nextSteps`. Use `steerlab experiment inspect <study> --json` to obtain a fresh
   document and external file digest for subsequent edits. Never put this
   authoring precondition inside manifest content.
5. For a revised prompt set, run
   `steerlab experiment import-prompts <study> --file <records.jsonl> --manifest-sha256 <digest> --json`.
   Full records, including unknown metadata, are preserved. LF, CRLF and CR
   record separators are normalized to LF on both clients; blank lines and
   outer record whitespace are removed. Escaped JSON content stays intact.
   The actual resulting bytes are
   hashed. A new content-addressed file is pinned under
   `prompts/tasks/versions/`; previous inputs remain intact. Repeating the same
   import with the current review reports `changed: false`.
6. For an existing vector, inspect its workspace-relative path with
   `steerlab experiment inspect-artifact <path> --json`, then use
   `steerlab experiment attach-artifact <study> <concept> --artifact <path> --artifact-sha256 <digest> --sidecar-sha256 <digest> --manifest-sha256 <digest> --json`.
   Optional `--source-concept` and `--eval-run` carry provenance. Both file
   reviews and the draft review must still match. The existing scientific store
   decides model, substrate, norm and provenance admission. Inspection alone
   does not certify attachability. Artifacts under `runs/` can be read; they
   are never edited by this operation.
7. Continue with `steerlab experiment verify <study> --json`, resolve the
   reported issues, and freeze only after substantive review. Execute using
   the established `bundle`/`runner` or composite `run` workflow. Study assembly
   never submits work or contacts a runner.

`steerlab pack export <study> --json` returns `result.pack` and
`externalDependencies`. Save the **pack object**, not its CLI envelope, for
later import. Change its study name before importing into a workspace where
that name already exists. Export includes readable UTF-8 dependencies under
`prompts/`; vectors, models and other dependencies still require the execution
bundle workflow. Export is permitted for frozen studies without changing them.
Existing pinned hashes are preserved, so exporting drifted inputs does not
magically repair their pins.

## Equivalence and limits

| Operation | Mac CLI / app / Swift HTTP | Python client |
|---|---|---|
| Pack preview/apply/export | Existing shared Mac owner | Same pack format and create-only workflow |
| Read reviewed study | `experiment manifest`; named HTTP document | `experiment inspect`; document and file digest |
| Full JSONL import | Existing reviewed import | Same immutable-version policy; shared Python run parser |
| Vector review and attachment | Mac scientific admission | Python scientific admission; substrate restrictions remain |
| Study interview, design library | Existing Mac operations | Added in [the design/interview checkpoint](PYTHON-DESIGN-INTERVIEW-WORKFLOW.md) |
| Model preparation | Existing Mac operations | Still follow-up work |

Review tokens are opaque and may differ between implementations. Obtain a new
review when changing surfaces or workspaces. Input SHA-256 values describe
actual bytes and travel with the study. Freeze identities retain each engine's
existing rules. This is authoring interchange, not numerical parity.

These are local client services, not new Python HTTP routes or GPU-engine CLI
verbs. The runtime runner/workbench boundary remains enforced. Existing client
verbs keep their syntax; this checkpoint does not retrofit external reviews
onto every older authoring command.

Filesystem locks coordinate cooperating Swift/Python writers. New inputs are
staged and published with an atomic, no-replacement hardlink. A filesystem that
cannot provide this primitive refuses publication. Pack failure
rollback removes only newly created files whose bytes still match; it never
removes reused inputs. Empty directories may remain. A process crash can leave
prepared inputs, and a prompt publication followed by a manifest-write failure
can leave an unreferenced version. Neither is a multi-file crash transaction.
This is not protection against a hostile process changing filesystem ancestors
outside the lock protocol.

## Review map and gates

- `client/study_assembly.py`: declarative command specs and thin dispatch.
- `client/authoring_files.py`: containment, reviewed draft lock and file snapshots.
- `client/study_packs.py`: preview/apply/export and automatic pin diagnostics.
- `client/study_inputs.py`: immutable prompt intake and reviewed vector admission.
- `experiment/task_inputs.py`: the existing record parser extracted for reuse.
  `scripts/ci/audit-task-prompt-parser.py` compares its record loop and retained
  file/hash gates with `ef3dec8`, and rejects a deliberately altered ID rule.
- `test_client_study_assembly.py`: refusal, preservation, CLI and light-import
  regressions. `PythonStudyAssemblyParityTests.swift` executes real Python and
  Mac pack round trips against a shared fixture.
- `scripts/ci/check-client-assembly-reference.py`: exact CLI table check;
  `--write` regenerates its region without changing the other CLI tables.

Before landing, read the entire diff, run both full suites, the parser audit,
normal/release bridge checks, public scan and reference/contract checks. The
maintainer, through the designated reviewing/integration agent, owns approval.
Recheck main's ancestry before integration and bring in any later fixes first.
Use Xcode beta, the explicit Metal toolchain and external temporary scratch.
The real interchange test needs the Python client dependencies (including NumPy):
it uses `Server/.venv.nosync/bin/python` when available, otherwise `python3`.
For a worktree using a separate environment, set the variable in the **shell
environment before `xcodebuild`**, as in this complete invocation from the
worktree root (substitute the environment and scratch paths):

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON="<environment>/bin/python" \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath "/private/tmp/<scratch>/DerivedData" CLANG_COVERAGE_MAPPING=NO
```

Xcode forwards that environment variable as `STEERLAB_TEST_PYTHON`. Passing
`TEST_RUNNER_STEERLAB_TEST_PYTHON=...` **after** `xcodebuild` instead supplies an
Xcode build setting and does not configure the test process. Both forms were
checked without a worktree venv or symlink: the environment prefix passed; the
build setting failed with missing NumPy. The test still fails with a repair
instead of silently skipping when the selected environment is incomplete.

Audit follow-up to `06d1f78`: F1 is corrected by keeping verification issues
only in the result and checking advisory entries against the closed vocabulary.
F2 now has direct import-operation regressions for LF, CRLF, CR and mixed
separators, comparing actual files and hashes across clients. Reproduction
upgraded it from a coverage note: Swift previously refused valid multi-record
CRLF input. Normalization is confined to authoring intake; existing pins and
files are not rewritten. For F4, the environment-prefix/build-setting distinction
above was reproduced without a worktree venv; interpreter discovery is unchanged.

Validation results for this checkpoint are recorded in
[the validation history](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md). No live
model/GPU study, cluster, installed app or interactive UI qualification is
claimed by this change.

## Next slices

1. Python reusable designs and study interviews are implemented in the
   [next checkpoint](PYTHON-DESIGN-INTERVIEW-WORKFLOW.md); independent review is the landing gate.
2. Model preparation and the remaining scenario/casting/pipeline authoring
   paths, with explicit capability and execution-role boundaries.
3. Cluster-document coauthoring and managed remote monitoring/recovery/cleanup
   (WP-5/6), retaining the integrated receipt gates and immutable evidence.
4. Complete researcher, agent and UI journeys plus scientific/GPU qualification.

The operation matrix and implementation plan retain the wider vision. This
checkpoint closes the named assembly gap; it does not claim all surfaces or
all research workflows are equivalent yet.
