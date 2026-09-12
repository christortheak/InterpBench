# Probe library, training, and evaluation: combined review handoff

Branch: `codex/probe-artifact-contracts`, based on main `22f08a9`.
Review the complete branch, including P0/P1 commits `8f59b8d`, `50764ba`, and
`d25c9a2`; the user requested P2 before sending the combined work to auditors.
P2 implementation commit: `f034897`.
The implementation remains in `/private/tmp/interpbench-probe-artifact-contracts`.

## What changes for a researcher

Probes now owns the library and the entry points to capture, fit, and evaluate.
The app offers three guided requests, plus data-author and independent-review
instructions. Each form describes the inputs and effective settings and leads
into the existing explicit execution/evidence workflow. Opening a form does not
train, download a model, create data, or delegate to another agent.

`probe-capture` reads labeled text at one declared residual site on a prepared
Python model and saves separate role-specific activation datasets. `probe-train`
fits a mean-difference reader, regularized linear classifier, or small ReLU MLP
on CPU from those retained activations. `probe-evaluate` scores a pinned probe
against a matching dataset and saves a separate report. Retraining and evaluation
never overwrite existing runs or trained bytes. Capture can be reused across
fits; a fixed layer is supported, with manual comparisons using selection data.
No automatic layer/threshold search or calibration is claimed.

The legacy Playground reader controls are redesigned into a small component
opened from Probes. They call the unchanged original native/server owners.
Legacy server training still uploads local examples when present, as the existing
owner did; the UI now says this explicitly. Data retains dataset browsing and row
editing. The portable classifiers do not silently enter the old Playground picker.

## One implementation behind the surfaces

| Surface | Supported route |
|---|---|
| Mac app | Probes → Capture activations / Fit a probe / Evaluate a probe |
| Both CLIs | `science interview`, `science draft`, and `science publish` with `probe-capture`, `probe-train`, or `probe-evaluate` |
| Workbench HTTP | Existing `/api/science/workspace/interview`, `/draft`, and `/publish` actions |
| Engine HTTP | Existing `/api/science/plan` and `/api/science/submit`, with the same managed operation request |
| Transport / lifecycle | Existing input-plan/package, stage, plan, submit, job monitoring/cancellation, export/fetch/import, and custody paths |
| Agent contract | `science guide readers`; the same maintained interview fields and role definitions as the app |
| Library | P1's shared `science probe-list` / `probe-inspect` and Probes section |

Native forms capture their workspace root. Both clients use the same local Python
owner; a remote model is execution hardware, not the source of truth for authorship.
After collection, the fitted instrument is discoverable in the local library.
The engine CLI gains no parallel custom command family. Scientific reports remain
ordinary JSON evidence accessible through the existing Results/file viewer and
returned report paths; a dedicated report dashboard is not added in this slice.

## Code and scientific contracts to inspect

- `probe_data.py`: bounded pinned reads, dataset validation, whole-group role
  assignment, source identity/overlap, and create-only run publication.
- `probe_capture.py`: explicit decoder paths, pre/post hook capture, token/position
  alignment, offline prepared model loading, and recorded actual input binding.
- `probe_training.py`: fitting-only preprocessing, explicit binary recipes,
  deterministic local RNG, objective/gradient, fixed-step optimization, optional
  shuffled-label control, and portable artifact publication.
- `probe_evaluation.py`: vectorized reference-compatible scoring, confusion and
  tie-aware ROC metrics, denominator handling, and honest overlap status.
- `docs/techniques/operations/probe-*.json`: maintained declarations/interviews.
  Existing operation bindings and routes retain their semantics.
- `ProbesPanelView`, `LegacyProbeTrainingView`, and the existing shared authoring
  sheet: UI orchestration only. Original `ConceptBuilder` and Python legacy
  `probes.py` scientific bodies are unchanged.

[The recipe and data contract](PROBE-TRAINING-AND-EVALUATION.md) specifies the
normalization, objectives, initialization, update rule, class mapping, score
meaning, capture semantics, input schema, role hashing, and bounded storage.
It distinguishes factual overlap checks from scientific judgment. Known overlap
in evaluation is reported without blocking exploration. A reserved final-test
capture cannot accidentally become fitting or selection input without explicitly
preparing a new dataset and documenting the changed purpose.

## Verification

- Full Python suite: **6,508 passed, 9 skipped**, with 8 warnings.
- Full Xcode beta suite: **290 SteeringKit and 4,640 ExperimentKit tests**,
  including 5 skips; `TEST SUCCEEDED`. The new native test runs a real CPU fit.
- Native and portable CLI training drafts produce identical owner payloads and
  review hashes from the same answers and activation file.
- Generated declarations and both CLI reference regions match. All maintained
  AST audits, their mutation controls, and normal/release bridge gates pass.
- Full source diff read; `git diff --check` and the public scan are clean.

Final logs: `/private/tmp/probe-training-python-verified.log`,
`/private/tmp/probe-training-native-final.log`, and
`/private/tmp/probe-training-final-audits.log`. These are local verification
records, not committed research evidence. Tokenizer tests verify that fast
normalization/special-token changes affect identity and that a slow tokenizer
without full serialization remains explicitly unknown.

Targeted tests include independent gradient differences for linear/MLP objectives,
an independently calculated mean-difference example, nonlinear XOR separation,
reference-score agreement, RNG isolation, fitting-only normalization, fixed
weights under changed selection data, final-test isolation, known overlap,
undefined metrics and AUC ties, actual toy-model pre/post hooks, mask/position
alignment, hook teardown, output/RNG preservation, split stability, and byte caps.

The managed round trip uses real CPU fitting and evaluation in relocated staged
workspaces, then exports/imports results, verifies custody, and finds the imported
probe in the shared library. Every managed interview drafts through its real owner.
The new native test mints a real Python-fitted probe, inspects its exact bytes from
Swift, and drafts the shared training interview through the native adapter.

The one mechanical extraction, `probe_artifacts.validate_input`, has a dedicated
AST audit against `d25c9a2` with site and numeric mutation controls. It proves the
previous artifact validation and score bodies unchanged after inlining that
helper. The historical registration audit names the exact new probe preflight
extension; original bindings, roles, and owner bodies remain protected. Its
negative controls remain active. The legacy UI is a deliberate presentation
redesign, not a claimed mechanical move of scientific code.

Reproduce with the same Python interpreter for both suites, run serially:

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

## Limits and next work

This is binary classification at a fixed site. No native MLX capture/trainer or
GPU-accelerated classifier optimizer is advertised. The Mac drives the shared
Python operations, just as an app-free client does. Real-checkpoint CUDA/MPS
capture, hardware memory/timing, and interactive UI qualification remain for the
running/reviewing agents. Tiny decoder tests do not qualify any named model.

The 64 MiB JSON pilot budget deliberately bounds retained activations; capture
and fitting hold their selected data in memory. Repeated preflight/input hashing
remains part of the existing reviewed-input workflow; no receipt trust bypass is
introduced. There is no optimizer/activation checkpoint continuation. Cancellation
retains request/job evidence and requires a fresh run. Reports and original files
remain immutable; this branch adds no remote cleanup permission.

P3 next: frozen study measurement settings, a shared observation runtime, position
and stage alignment, multiple probes/agents, and Results integration. Conditional
policies and intervention adapters follow that foundation. Successful classifier
fitting is not evidence of causal use, live measurement parity, or steering benefit.

## Integration process

The user's reviewing/integration agents read the full diff, rerun both suites and
all maintained audits, and decide landing through the user. This branch installs
no app, merges no main, and deploys no engine. Rebuild the app when landing because
the Python payload and compiled identity change together. Running study controllers
are not restarted by this work.

Main still had the original untracked
`docs/PROBES-AND-INTERVENTIONS-PHASE-1-PLAN.md` when P1 was created. Preserve and
compare that file before landing this branch's authorized version; do not discard
newer work as generic untracked cleanup. See the P1 handoff for its original
verification record and artifact contract.
