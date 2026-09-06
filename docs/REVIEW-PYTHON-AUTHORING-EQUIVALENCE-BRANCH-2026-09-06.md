# Review — `codex/python-authoring-equivalence` at 06d1f78 (2026-09-06)

Reviewer: Claude (maintainer's reviewing/integration agent). Range reviewed:
`ef3dec8` (current main) to `06d1f78`, one commit, 20 files,
+1,261 / −136. Main is an ancestor; the branch fast-forwards. Worktree
`/private/tmp/interpbench-workflow-handoff` clean at the tip. No edits were
made to the branch.

## 1. Verdict

**Landable after two small fixes** (F1, F4). F1: the client's `pack apply` and
`import-prompts` put the verification issues into the envelope's
`advisories` array as bare strings, which breaks the closed advisory
vocabulary the Python envelope enforces everywhere else and diverges from
the Mac, which reports the same issues only in `result.verificationIssues`.
Everything else holds: the four client modules do what the handoff says,
the parser extraction is mechanically proven, the Python/Mac pack
interchange is tested against real bytes, and the immutability rules are
respected. Two fixes are needed before landing (F1, F4); two notes follow (§4); neither note blocks.

## 2. Verified independently

| Check | Result |
|---|---|
| Parser audit `scripts/ci/audit-task-prompt-parser.py` (AST of the record loop vs `ef3dec8`, file/hash gates, negative control) | passes |
| Client reference check `scripts/ci/check-client-assembly-reference.py` | "matches declared flags" |
| Bridge gates, normal and `--release` | both PASS |
| Vocabulary in the diff and commit message; `git diff --check` | clean |
| Empirical run of the built client on the fixture in a scratch workspace | preview `ready` exit 0; apply exit 0, draft and prompt file created, lock sidecars under `.steerlab/manifest-locks/`; a second apply with the same token refuses `staleManifest` exit 65 ("already exists"). The apply envelope's `advisories` field contained `["no concepts or variants attached"]` — see F1. |
| Full Python suite on the worktree (main venv, cwd = worktree/Server) | 5,971 passed, 9 skipped, 8 warnings, matching the branch's claim (main: 5,938; the difference is this branch's tests) |
| Full Xcode beta suite on the worktree, plain (no venv in the worktree) | `TEST FAILED`, exactly one issue: the Python/Mac interchange test, `ModuleNotFoundError: numpy` from the system `python3` fallback (F4) |
| Full Xcode beta suite with `TEST_RUNNER_STEERLAB_TEST_PYTHON=<main venv>` | same single failure: the variable does not reach the test (F4) |
| Full Xcode beta suite with `Server/.venv.nosync` symlinked to the main checkout's venv (the condition of the branch's own runs) | `TEST SUCCEEDED`: 277 SteeringKit + 4,572 ExperimentKit, matching the branch's claim; the interchange test passes against real bytes |

## 3. What the code does, against the handoff

**Containment (`authoring_files.py`).** Relative paths only, no `..`
segments, no NUL, symlinks refused, resolved path must stay inside the
workspace (inside `prompts/` for pack files) and outside `runs/` unless the
read is explicitly read-only. `reviewed_draft` takes the shared manifest
lock, requires the external digest, and admits drafts only.
`publish_new` stages to a temp file and publishes by `os.link`, so the
final name appears atomically and an existing file is never replaced;
identical bytes are reused, differing bytes refuse. Correct.

**Packs (`study_packs.py`).** Same on-disk format as the Mac
(`{study, files}` or a bare manifest); the slug rule mirrors the Mac's;
freeze metadata is stripped on decode; `Manifest.from_dict` rejects
unreadable shapes while unmodeled keys survive. Preview is read-only and
refuses an existing study or a differing existing input. Apply re-previews,
requires the token, takes every touched path's lock in sorted order,
re-previews under the locks, publishes only new files, auto-pins named
inputs from real bytes and reports each pin failure with its cause, runs
`verify` before the manifest is saved, saves under the create-only rule,
and on failure unlinks only files whose bytes still match what it wrote.
Export runs under the manifest lock, works on frozen studies without
touching them, exports UTF-8 inputs under `prompts/`, and names everything
else as an external dependency. Correct, and the pin-failure reporting is
an improvement over the Mac's silent `try?`.

**Inputs (`study_inputs.py`).** `import_prompts` normalizes blank lines
and outer whitespace, validates every record with the same parser the run
loop uses, publishes a content-addressed version under
`prompts/tasks/versions/`, and saves the draft only if the document
changed, so a repeat import is `changed: false`. `attach_artifact`
inspects both files, locks them, re-inspects, requires both digests and
the draft review, then defers to `experiment_store.attach_artifact` for
substrate, norm and provenance admission. Vector and run bytes are never
written. Correct.

**Parser extraction (`task_inputs.py`).** `parse_prompts(text)` is the
old `load_prompts` record loop over a `StringIO` with the same
universal-newline policy; `load_prompts` keeps its file and hash gates and
ends by delegating. The audit proves both and rejects a changed identity
rule. Reproduced.

**Dispatch and surface.** `study_assembly.VERB_SPECS` declares the seven
verbs; `client_cli` routes `pack` as a family and the four `experiment`
verbs by name; the generated reference region is checked by a script that
the Python suite runs. Light-import test proves the journey pulls in no
torch, transformers, fastapi, uvicorn, peft or sae_lens.

**Interchange test.** `PythonStudyAssemblyParityTests` writes the shared
fixture through the Python client (preview, apply, export), applies the
exported pack on the Mac, checks status, freeze fields, condition names,
that the pinned prompt hash equals the bytes on disk with custom metadata
preserved, exports from the Mac, round-trips that through Python again,
and checks the hash and file text agree. Real bytes, both directions.

## 4. Findings

### F1 — Verification issues leak into `advisories` as bare strings (fix before landing)

`client/study_assembly.py` returns `CLIResult(..., advisories=issues)`
where `issues` is `result.verificationIssues`, a list of strings.
`Envelope.add_advisories` appends entries unchecked, so the wire document
carries `"advisories": ["no concepts or variants attached"]` instead of
`{code, detail}` objects from `ADVISORY_CODES`, the closed vocabulary the
`advisory()` constructor enforces and the contract test pins. Two
consequences: any agent that reads `advisories[].code` breaks on this
verb, and the state becomes `okWithAdvisories` for a condition the Mac
reports as `ready` with the issues in the result. The fix is to drop the
`advisories=` argument and leave the issues in `result.verificationIssues`
(matching the Mac), or, if an advisory is wanted, add one code to the
vocabulary and its contract list and wrap each issue with `advisory()`.
Either way, add an assertion in `test_cli_roundtrip_and_missing_or_extra_arguments`
that every advisory entry is an object with a vocabulary code.

### F2 — `import-prompts` byte parity across surfaces is asserted, not tested (note)

Python trims blank lines and outer whitespace and joins with LF; the Mac's
`TaskPromptsDocument.serialized()` writes each item's `rawLine` plus LF.
They look equivalent, but the interchange test covers only the pack path,
where both sides copy the supplied text verbatim. If the two importers ever
diverge, the same JSONL imported on each surface would yield different
version hashes, each self-consistent. One round-trip assertion on
`import-prompts` would close it. Not a defect.

### F4 — The documented repair for the interchange test does not work under xcodebuild (fix before landing)

`PythonStudyAssemblyParityTests` resolves the interpreter as
`$STEERLAB_TEST_PYTHON`, else `<repo-under-test>/Server/.venv.nosync/bin/python`
if present, else `python3`. The handoff, the workflow document and the
test's own error message all say to pass
`TEST_RUNNER_STEERLAB_TEST_PYTHON=<env>/bin/python` to `xcodebuild`
"which forwards it as `STEERLAB_TEST_PYTHON`". Measured on this machine
with the repository's canonical invocation (serial, Xcode beta, Metal
toolchain), on a linked worktree that has no venv:

| Invocation | Result |
|---|---|
| plain full suite | parity test fails: system `python3`, no NumPy |
| full suite with `TEST_RUNNER_STEERLAB_TEST_PYTHON=<main venv>` as a build setting | same failure |
| only that test with `STEERLAB_TEST_PYTHON=<main venv>` exported in the shell | same failure |
| only that test with the `TEST_RUNNER_` build setting | same failure |
| full suite with `Server/.venv.nosync` symlinked to the main checkout's venv | see §2 (this is the condition their own runs had, since their checkout carries the venv) |

So the variable never reaches the test process in this configuration, and
the only working path is a venv inside the checkout under test. Anyone
verifying a worktree per AGENTS.md gets a red suite and a repair that does
not repair. Fix before landing, any of:

- Resolve the interpreter through the git common directory for a linked
  worktree (`.git` is a file naming `gitdir:`; its parent's parent is the
  main checkout), and try `<main checkout>/Server/.venv.nosync/bin/python`
  before falling back.
- Or read the interpreter from a small dotfile or build setting that is
  actually delivered (verify it under the canonical invocation, not under
  `swift test`).
- And in every case correct the three places that document the variable,
  or delete the claim.

Failing loudly rather than skipping remains the right choice; the defect is
the false repair.

### F3 — Commit message has no body (note)

The handoff document carries the design; the commit carries one line.
Worth a body if the maintainer amends on landing.

## 5. Landing instructions

1. Fix F1 (a two-line change plus one assertion) and F4 (interpreter resolution for linked worktrees, and the three documents that describe the variable); rerun both suites, the Swift one from a worktree without a venv.
2. Confirm the two PENDING rows in §2 are green (this document will be
   updated when they are).
3. Fast-forward main to the tip. No deploy consequence: this is the
   Python client and its tests; the engine, the app and the rollout order
   from the earlier reviews are unchanged.

---

# Re-review — follow-up 99891ff (2026-09-06, later the same day)

One commit on top of 06d1f78 (`Fix study assembly advisory and prompt
import contracts`, 11 files, +199 / −22). Main `ef3dec8` is still an
ancestor; the worktree is clean at the tip and, for this re-review,
carries no venv.

## R1. Finding by finding

| Finding | What changed | Verified how |
|---|---|---|
| F1 advisories | `study_assembly.run` no longer passes `advisories=`; issues stay in `result.verificationIssues`; the CLI round-trip test now asserts `state == "ready"`, no `advisories` key, and that every advisory entry anywhere in an envelope is an object whose `code` is in `ADVISORY_CODES`. | Diff read. Matches the Mac's contract. |
| F2 import parity | Both intakes normalize CRLF and CR to LF before splitting (`TaskPromptsImport.normalizedLineEndings`, applied in `preview`, `looksLikeJSONL` and `importJSONL`; Python `prompt_bytes`). Reproducing the note found a real Swift defect: valid multi-record CRLF input was refused. New parametrized tests on both sides (LF, CRLF, CR, mixed) import the same text, compare the published bytes, hash and version path across clients, check escaped `\r\n` inside a JSON string survives, check the prior input file is untouched, and check a repeat is `changed: false`. | Diff read. The Swift-side parametrized parity test drives the real Python client through `client_cli.main`. |
| F4 test environment | Interpreter discovery unchanged. The documents and the test's error message now say to set `TEST_RUNNER_STEERLAB_TEST_PYTHON=<env>/bin/python` in the shell environment **before** `xcodebuild`, and state that the same name passed after the command is a build setting that does not reach the test. | That is the one form I had not measured (I tried the build-setting form and the un-prefixed exported form). Measured now from this worktree with no venv and no symlink, full suite: see R2. |

## R2. Verification on 99891ff

| Check | Result |
|---|---|
| Parser audit, client reference check, bridge gates normal and `--release` | all pass |
| Vocabulary in the diff and both commit messages; `git diff --check` | clean |
| Full Python suite on the worktree | 5,975 passed, 9 skipped, 8 warnings, matching the branch's claim |
| Full Xcode beta suite from the venv-less worktree, `TEST_RUNNER_STEERLAB_TEST_PYTHON=<main venv>` set in the shell environment before `xcodebuild` | `TEST SUCCEEDED`: 277 SteeringKit + 4,574 ExperimentKit, matching the branch's claim. The environment-prefix form does reach the test; my first-pass table (F4) had measured the build-setting form and the un-prefixed export, not this one. |

## R3. Verdict

**Landable by fast-forward.** Both suites are green on 99891ff, the Swift
one under exactly the worktree condition F4 described, so F4's
documentation correction is sufficient and the interpreter resolution can
stay as it is. F1 and F2 are closed in source with regressions, and F2's
reproduction fixed a real Swift defect (CRLF multi-record input was
refused). No deploy consequence: this is the Python client, one Swift
intake normalization, tests and documents.
