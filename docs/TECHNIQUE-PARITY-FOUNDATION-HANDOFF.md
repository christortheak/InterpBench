# Technique/parity foundation handoff

Branch: `codex/technique-parity-foundation`, based on main `293888e`.
Date: 2026-09-07. Main and the installed app are not modified by this work.

## Scope delivered

This starts steps 1–2 of the ordered program: the technique guide, worked
example, source-checked capability inventory, scientific acceptance criteria and
exact sampling contract. Steps 3–6 remain implementation/measurement work; this
handoff does not claim seeded MLX execution or MPS qualification has landed.

- [ADDING-A-TECHNIQUE.md](ADDING-A-TECHNIQUE.md): scientific-effect classification,
  real owners/registries, interface journey, input/output identity, fixture
  obligations, generators and historical versus current audit scope.
- [ADDING-A-TECHNIQUE-EXAMPLE.md](ADDING-A-TECHNIQUE-EXAMPLE.md): a complete fictional
  CPU owner with registration/interview/catalog deltas and real-owner tests.
  `scripts/ci/qualify-technique-example.py` executes the actual document blocks in
  a temporary source copy; the fictional operation is never registered in the
  production tree. The copy and its outputs are removed on completion.
- [SUBSTRATES.md](SUBSTRATES.md): all 24 science catalog operations joined with
  explicit production/reading/backend profiles in `substrate-capabilities.json`.
  A separate core-lifecycle section prevents treating the catalog as every feature.
  `check-substrates.py` checks exact census, known states, evidence paths and the
  generated table. It runs through the science-resource check and in CI.
- [TECHNIQUE-PARITY-IMPLEMENTATION.md](TECHNIQUE-PARITY-IMPLEMENTATION.md): the
  six-step program, corrected seed policies, distinct parity claims and a
  measurement protocol. The old briefs link to this active refinement.
- Root `AGENTS.md` now records the researcher's guidance-first principle and
  points coding agents to these documents.

No runtime admission policy, scientific arithmetic, manifest schema, model
dependency, generated client identity or existing artifact is changed. The new
inventory is documentation tooling; it is not yet returned as structured backend
status by the app/CLI/API. Assertions in its checker govern documentation quality,
not a researcher's permission to execute an operation.

## Review of the newly landed UI work

Read the maintainer's UI review and inspected representative changes on main:
draft review initialization, errors at their originating controls, typed connection/
results state, freeze explanations, workspace controls and clipboard feedback.
These are useful improvements in clarity and recovery. The follow-up removes the
global Add Condition default action, supplies Reconnect, removes duplicate controls
and treats an untouched multi-agent template as clean.

This was a focused source review, not a second complete 139-file audit or a live
GUI qualification. The enforced window floor and split remount/divider behavior
still need a live pass, together with new-draft, reconnect and pristine-template
journeys. No UI regression was established in the inspected changes, and no UI
code was edited in this foundation. Main also already contains the N4 installer
version consolidation; it is not duplicated here.

## Validation

Interim commits: `c9efe13` contains the guide, executable example, inventory and
checks; the following documentation commit contains the refined program, brief
cross-references and this handoff. Both are on the branch for independent review.

| Check | Result on the foundation tree |
| --- | --- |
| Full Python suite | **6,175 passed, 9 skipped, 8 warnings**, 177.78 seconds. |
| Full Xcode beta suite | **277 SteeringKit + 4,602 ExperimentKit passed; TEST SUCCEEDED**. App target compiled. |
| Disposable worked-example qualification | **25 passed**; regenerated resources cover the temporary 25th operation, real-owner and isolated config validation pass, publications stay immutable and drift changes the input plan. |
| Inventory regression tests | **9 passed**; operation census, duplicate entries, unknown statuses and unsupported evidence assertions are exercised. |
| Shared science/bootstrap/interview resources, Python source identity, client reference | All pass; compiled identity unchanged. |
| Managed-owner AST audit, stability/preflight audit, parser and lazy-import audits | All pass, including their negative controls; no scientific body changed. |
| Bridge normal/release gates, public scan, diff whitespace | All pass. |

Python and Swift full suites ran sequentially. The example's script was also
rerun from its maintained entry point. No additional dependency was installed,
no hardware parity claim was made, and the installed app was not replaced.

Local validation logs are `technique-parity-python.log`,
`technique-parity-swift.log`, and `technique-example-repeatable.log` under
temporary storage; the commands below allow independent reproduction.

Commands from the branch root, using a test-capable Python environment:

```sh
PYTHONPATH=Server python scripts/ci/qualify-technique-example.py
PYTHONPATH=Server python -m pytest Server/tests/test_substrate_inventory.py -q
PYTHONPATH=Server HF_HUB_OFFLINE=1 python -m pytest Server/tests -q
python scripts/ci/check-science-resources.py
python scripts/ci/check-substrates.py
python scripts/ci/check-workspace-bootstrap.py
python scripts/ci/check-study-interviews.py
python scripts/ci/check-python-client-identity.py
python scripts/ci/check-client-assembly-reference.py
python scripts/ci/audit-managed-scientific-owners.py
python scripts/ci/audit-stability-preflight.py
python scripts/ci/audit-task-prompt-parser.py
python scripts/ci/audit-design-lazy-imports.py
python scripts/ci/check-swift-bridge-retirement.py
python scripts/ci/check-swift-bridge-retirement.py --release
python scripts/ci/public_scan.py
git diff --check
```

Swift uses Xcode beta and the installed Metal toolchain, a scratch directory under
temporary storage, and `TEST_RUNNER_STEERLAB_TEST_PYTHON` pointing to the test
environment in the shell before `xcodebuild`:

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1 \
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/interpbench-workflow-xcode CLANG_COVERAGE_MAPPING=NO
```

## Remaining work and review boundary

1. Independent fresh-agent guide trial, then its Python-only intervention trial.
   The scripted exercise proves the supplied example works; it does not prove
   a fresh agent can discover every step unaided.
2. Step 3: reader final-test separation first, then intervention descriptor/
   sidecar, native extraction stability, and adapter-scale/evidence-role stamps.
   These need scientific fixtures and independent review, not automatic AST re-pins.
3. Step 4: seeded MLX sampling with the existing per-path seed policies, scoped
   RNG and resume/interleaving tests; record measured repeatability limitations.
4. Step 5: actual MPS/CUDA comparisons with declared configurations and tolerances.
   No new hardware qualification records were produced in this foundation.
5. Step 6: simplify registration/regeneration from the exercised guide, preserving
   small owners and explicit input roles. No developer scaffold is shipped yet.

The inventory is deliberately conservative. Evidence-link checks establish that
referenced files exist, not that a claimed measurement is valid or covers all
operations sharing a profile. Review those claims and split profiles as necessary.
There is no new qualification allowlist. Existing immutable evidence and input
identity protections remain in force; changing an existing refusal is a separately
reviewed behavior change.

The researcher coordinates the independent coding/audit agent and landing. Read
the full diff, verify both suites and the applicable owner-preservation audits,
then integrate through that process. No merge or deployment is performed here.
