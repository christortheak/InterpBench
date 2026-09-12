# Probe contracts and library: P0/P1 implementation handoff

P2 is now added on this branch. Read [the combined review handoff](PROBE-TRAINING-IMPLEMENTATION-HANDOFF.md)
for current verification and the additional training/evaluation changes.

Branch: `codex/probe-artifact-contracts`, from main `22f08a9`.
Contract commit: `8f59b8d`. Implementation commit: `50764ba`.
The user resolved the J-lens/GPU discussion and authorized the phase-1 program.
This is its first reviewable slice, not completion of all interventions work.

## Researcher outcome

There is now a Probes section with a shared native/Python library. It identifies
legacy readers and new portable instruments, gives model/layer/method information,
reports unknown provenance and malformed artifacts, and previews stored
parameters on request, with a Reveal action for the complete file. Opening it loads no model and trains nothing. Work
runs through the existing local Python helper off the UI thread. Parameter JSON
is formatted off-main and its displayed preview is bounded to 64 KiB; the CLI
and HTTP inspection responses retain the full document. The existing
reader builder remains reachable through a clearly labelled link into Data.

The same inspection contract is reached by:

| Surface | Path |
|---|---|
| App | Probes → select a record |
| Native CLI | `steerlab-cli science probe-list` and `science probe-inspect <path>`, with `--workspace <root> --json` |
| Portable CLI | `steerlab science probe-list` and `science probe-inspect <path>`, with `--root <root> --json` |
| Mac workbench HTTP | `POST /api/science/workspace/probe-list` or `probe-inspect` |
| Python workbench HTTP | The same action paths and request bodies |
| Agent reference | `science operation probe-library` and `science guide readers` |

HTTP takes `workspaceRoot`, plus `path` for inspection. Runner-role access remains
restricted by the existing service-role gate. The portable envelope uses `result`;
the native workspace adapter uses `result.response`; HTTP returns the owner result
directly. The engine CLI has no new local library verb; its workbench HTTP and the
portable client are the relevant routes. No new execution route is advertised.

## Owners and scientific meaning

`experiment/probe_artifacts.py` validates the closed v1 document and provides a
binary64 CPU reference scorer. Preprocessing and affine/ReLU arithmetic are fully
specified, with independent linear and XOR fixtures. Strict threshold ties are
negative. No probability calibration is claimed. Input binding equality covers
model/revision, backend coordinates, precision, site, rendering, and population;
future execution adapters must derive that binding from actual computation.
Missing pins remain explicit limitations, rather than invented compatibility.

`experiment/probe_library.py` owns discovery and inspection for every new surface.
It reads exact bytes, hashes them, and retains legacy fields without migration.
Native records use `ReadingProbeArtifact`; Python records use `kind: readingProbe`
and selection accuracy. No legacy method is relabelled as logistic regression.
A validated portable artifact does not become an entry in the existing native
Playground picker, whose activation semantics do not yet satisfy the new contract.

The new Swift `ProbeLibrary` types are presentation records, not a replacement
artifact serializer. The Mac uses the same Python owner via `DiagnosticWorkspace`.
Cross-client tests re-encode a *new* file, prove semantic round trips and exact
large-seed strings, and prove that changed encoding receives a different byte hash.
Original artifacts and historical consumers are unchanged.

The shared observation/intervention contract is recorded alongside the artifact
contract: sites, positions, before/after order, state isolation, distinct action
units, and future tensor-provider requirements. No unused runtime framework or
compatibility bridge is introduced in this slice.

## Tests and limitations

Verification on this branch:

- Full Python suite: **6,481 passed, 9 skipped**, with 8 warnings.
- Full Xcode beta suite: **290 SteeringKit tests and 4,639 ExperimentKit tests**,
  including 5 skips; `TEST SUCCEEDED`. The three new native tests call the real
  Python client, rather than a substitute library.
- Source-built native and portable CLI smoke: `probe-list` and `probe-inspect`
  return identical owner payloads, including the exact file hash, against a
  synthetic workspace.
- Generated declarations and both CLI reference regions match. All maintained
  AST audits, negative controls, and normal/release bridge gates pass.
- Full source diff read, public scan clean, and `git diff --check` clean.

Test logs for this session are `/private/tmp/probe-contracts-python-verified.log`,
`/private/tmp/probe-contracts-native-final.log`, and
`/private/tmp/probe-contracts-gates-verified.log`. They are local verification
records, not committed study evidence. The contract accurately describes the
reference scorer's use of binary64 values and the interpreter's `sum`; it does
not promise bitwise reduction equality across interpreter versions or backends.

Reproduce with the existing test-capable Python environment as `TEST_PYTHON`:

```sh
cd Server
HF_HUB_OFFLINE=1 PYTHONPATH=. "$TEST_PYTHON" -m pytest -q
cd ..
HF_HUB_OFFLINE=1 \
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
TEST_RUNNER_STEERLAB_TEST_PYTHON="$TEST_PYTHON" \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-science-gpu-build \
  CLANG_COVERAGE_MAPPING=NO
"$TEST_PYTHON" scripts/ci/check-generated.py --audits \
  --cli /private/tmp/interpbench-science-gpu-build/Build/Products/Debug/steerlab-cli
"$TEST_PYTHON" scripts/ci/public_scan.py
git diff --check
```

Both suites run serially, with scratch outside the file provider. No new
historical AST baseline is needed: existing scientific owners did not move;
intentional new scoring arithmetic is covered by independent numerical fixtures.
Existing audits and negative controls remain intact.

The new tests cover independent arithmetic; malformed shapes, booleans, non-finite
values, and incompatible bindings; data-role declarations; exact original bytes;
legacy formats; invalid-file visibility; workspace confinement; real CLI and HTTP
owners; and native decoding through the actual Python adapter. These establish
artifact and library behavior, not classifier fitting quality or generation parity.
The library bounds each JSON inspection at 64 MiB, ignores unrelated filenames,
and reports matching files it cannot inspect. Symlinked run directories are
reported and not followed. The scan is not a filesystem transaction: another
process may change a file before later inspection, whose newly computed hash
identifies the bytes actually inspected. No reviewed execution depends on list data.

Interactive app layout acceptance remains with the reviewing/running agents after
their rebuild. Open Probes, switch selections and workspaces, refresh, inspect a
legacy Python record, and verify no earlier selection or workspace leaks into the
view. Check loading/error guidance and access to full stored JSON. Package tests
do not claim a live UI session.

## Next implementation slice

P2 supplies capture, actual mean-difference/regularized-linear/MLP fitting,
evaluation, protocol and data authoring, and a guided training UI on the same
owners. The legacy training UI and native-only Data inventory summary still exist;
move their responsibilities into Probes with a reviewed migration and mechanical
proof where bodies move. Do not create a second trainer merely to relocate a panel.
P3 adds study measurements and Results alignment under the shared runtime contract;
P4–P6 add intervention adapters, policies, and remote/multi-agent evidence. P7
qualifies runtime overhead and behavior. No model job or intervention is submitted
by the library commands in this branch.

## Integration

Independent reviewing/integration agents, through the user, read the full diff,
check both suites and the maintained gates, and decide landing. Python payload
bytes and the compiled identity change together; rebuild the app before pairing
it with this checkout. This worktree does not install the app or deploy an engine.
A remote workbench gains the new inspection actions only after its reviewed
update; running study controllers need no incidental restart for this work.

The main checkout still holds the originally untracked phase-1 plan at the same
path as this branch's now-committed plan. Compare and preserve it before landing;
the branch intentionally updates its hold notice with the user's authorization.
Do not discard any newer edits in that file as generic untracked cleanup.
