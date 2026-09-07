# Adding a technique to SteerLab

Implementation map checked against `293888e` on 2026-09-07. This guide is for
coding agents extending the instrument. Research inputs belong in a separate
workspace. Start with the repository's `AGENTS.md`; follow the linked scientific
contracts when a technique touches them. The older
[brief](ADDING-A-TECHNIQUE-BRIEF-2026-09-07.md) records the original proposal;
this guide and [the execution plan](TECHNIQUE-PARITY-IMPLEMENTATION.md) refine it.

## 1. Help the researcher explore honestly

Prefer guidance over refusal. Explain what a technique measures, offer a useful
default, show the effective configuration, and record limitations with results.
An unqualified backend or unconventional research design is not, by itself, a
reason to prevent execution. Qualification limits our claims about evidence.

Refuse when the requested computation cannot be performed as described, required
input bytes cannot be identified or interpreted, or proceeding would overwrite
immutable evidence or misrepresent the operation. Name the concrete problem and
the shortest repair or executable alternative. Do not add scientific permission
lists, hidden switches, repeated approval dialogs, or new frozen-manifest fields
to implement an advisory. Existing integrity contracts still apply; changing an
existing refusal is a separate, reviewed behavior change.

The app, CLIs and HTTP API are ways to drive owners, not independent scientific
implementations. A Mac can author and inspect a Python-executed technique without
an MLX implementation. Keep J-space analysis distinct from OptVec optimization,
even where the historical Python module or engine command contains `optvec`.

## 2. Classify by scientific effect and implementation cost

Classes can overlap. Class A is an integration shape, not an estimate of how long
inventing or validating a scientific method takes.

| Addition | Start here | Other owners to inspect | Identity obligation |
| --- | --- | --- | --- |
| A: managed report, analysis or training operation | `experiment/managed_methods.py` and a small owner module | input/output closure, interview, catalog, execution adapters | Define new artifact provenance and downstream use; do not assume a report or trained instrument is identity-neutral. |
| B: training objective or adapter family | `experiment/lora_train.py`, `lora_data.py` | model variant attachment; Swift `FineTuneStore` / `FineTuneTrainer` if supported | Training recipe, adapter scale/convention, data roles, model revision and produced artifact identity. |
| C: intervention mode, location or centering | `steering/intervention.py`, `plan.py`, `injector.py`, `ablator.py` | Python condition/variant owners and matching Swift types | Distinguish new semantics with a new descriptor/version; preserve old frozen bytes. |
| D: extraction recipe | `steering/vector_math.py`, `experiment/recipe_identity.py`, `extraction_workflow.py` | Swift extraction recipe, math, identity and builder | New recipe name/identity; legacy artifact hashes must stay unchanged. |

Python paths above are relative to `Server/steerlab_server/`. An existing generic
managed owner does not cover every potential class A operation: J-lens acquisition
and qualification, for example, have their own routes. Inspect before extending.

Write these answers before code:

- The research question and mathematical operation, with normalization, token,
  layer, precision and dose conventions.
- Inputs and their roles: fitting, model/parameter selection, final evaluation,
  nuisance controls, or exploratory analysis. State which choices use outcomes.
- Outputs, consumers, identity material and limitations. Distinguish arithmetic
  fidelity, predictive readout and causal evidence.
- Producer engine, numerical backend, artifact reader and authoring surfaces.
  Use [SUBSTRATES.md](SUBSTRATES.md); mark unmeasured combinations explicitly.
- One independently checkable numerical example and failure/degeneracy cases.
  A new scientific algorithm needs more than an integration smoke test.

For a researcher who has not decided these questions, present a proposed design
and uncertainties. Do not require a fully confirmatory design to explore.

## 3. Managed operation walkthrough

Use [the complete CPU example](ADDING-A-TECHNIQUE-EXAMPLE.md) in a disposable
checkout first. It exercises the integration shape without shipping a fictional
method. The current locations below are deliberately explicit; simplification
comes after the guide is tested.

1. **Owner.** Create a focused module under `experiment/`. Its config's
   `from_dict` validates shape, rejects unknown keys and reports concrete repairs;
   `to_dict` reports the effective values. Keep discovery and authoring importable
   without GPU packages. Put heavy imports inside execution where possible.
   `managed_methods.execute` passes `root` and `log` only if the entry function
   declares them. Return a result object, including the published output path
   expected by evidence export, rather than just printing a report.
2. **Execution registration.** Add a `Method(module, config_class, function,
   compute)` entry to `METHODS`. `cpu` means execution beside the controller;
   `gpu` requires `modelID` plus an immutable 40-hex `revision`. In a controller
   profile, model work requires its Slurm executor; elsewhere it follows the
   profile. This routing is in `api/scientific_execution.py`, not the registry.
   `SPECIAL` has explicit dispatch and validation branches: adding a name there
   alone does not implement it. Prefer the ordinary `METHODS` path.
3. **Input closure.** Inspect `experiment/managed_inputs.py`. `ARTIFACTS`,
   `ARTIFACT_LISTS`, `FILES`, `TREES`, and `LENSES` identify references recursively.
   An unfamiliar key is not automatically a captured dependency. Add an explicit
   role or owner-specific inventory when necessary; never infer a file from an
   arbitrary string. Include sidecars and discovered files that affect results.
   Prove the package is complete on an isolated runner, and that changed bytes
   between review and execution cannot silently change the job.
4. **Interview.** Add the operation to
   `WorkspaceSeed/prompts/method-guides/workflows.json`. Field IDs are the owner's
   config paths, including nesting: `datasets.targetTrain`, not `targetTrain`.
   Kinds are `text`, `integer`, `number`, `boolean`, `integers`, `numbers`,
   `artifact`, `artifacts`, `file`, `files`, `fileRef`, or `documentFile`.
   Defaults and answers are text; `fileRef` becomes `{path, sha256}` while
   `documentFile` loads JSON. Optional blank values use the interview default.
   `advanced` cannot override a form answer. Both clients use this resource.
5. **Catalog.** Add the operation under an existing method in `catalog.json`,
   or add a method and its guide. Record actual outputs, limitations, command
   references and HTTP actions. A route reference must exist in the census.
   Existing `/api/science/plan` and `/api/science/submit` usually need no new verb
   or route. Never copy another operation's output description without checking.
   Add the operation to `docs/substrate-capabilities.json`; its exact ID census
   is gated against the catalog. The latter is documentation metadata, not a
   runtime support switch.
6. **Adapters and evidence.** Inspect `api/managed_validation.py` (isolated config
   validation), `api/scientific_execution.py` (planning and child execution),
   `api/diagnostic_transport.py` (staging/export), and the local custody owner.
   Existing adapters should handle an ordinary managed operation. Verify this:
   successful publication alone is not evidence that execution/export works.
   New output classes do not inherit permission for remote cleanup.
7. **Regenerate and test.** Use sections 7–8. Add a minimal valid case to
   `Server/tests/test_interview_validation.py`: its explicit fixture census must
   equal `managed_methods.OPERATIONS`. Run the drafted config through the actual
   `managed_methods.validate`, not a mocked parser. Extend integration coverage
   for packaging, queue-time verification, output export/import and inspection.
8. **Document.** Update the method guide, capability inventory and CHANGELOG.
   Provide an author prompt with the exact data schema and a separate review
   prompt. Name undecided conceptual choices rather than inventing input hashes
   or model pins. Describe when missing evidence limits a claim, not exploration.

The generic app entry is Research methods → Author request → execution/evidence.
Its code includes `ScienceCatalog`, `MethodAuthoringSheet`, and the shared
`DiagnosticWorkspace` Python adapter. Locate callers with `rg` before editing:
the app need not gain a new hand-written form for an ordinary managed operation.

## 4. Training methods

Read [TRAINING-RECIPES.md](TRAINING-RECIPES.md). A new objective needs a new row
with the tokenizer/rendering, label masking, loss reduction, optimizer,
schedule, precision, initialization, scaling and data roles specified. A matching
method name on two backends does not establish equivalent training.

Inspect `experiment/model_variant.py`, the adapter sidecars, and Swift
`FineTuneStore.swift`, `FineTuneTrainer.swift`, `FineTuneTrainingData.swift`.
For a non-LoRA family, verify that serialization, loading and application can
represent it; do not squeeze it into LoRA fields. If only Python trains or reads
it, say so and offer that route from the Mac. Test an independently calculable
loss/update on a small fixture plus attachment and artifact round-trip behavior.

## 5. Intervention modes

Read [INTERVENTION-SCOPE.md](INTERVENTION-SCOPE.md). Inspect Python `PATHS` and
centering vocabulary in `steering/intervention.py`, the descriptors in
`steering/plan.py`, and arithmetic in `injector.py` / `ablator.py`. Search for
`add` / `ablate` dispatch in `experiment/condition_execution.py` and
`model_variant.py`. Swift counterparts live under
`Sources/SteeringKit/Injection/`; manifest/variant decoding is in ExperimentKit.
Find all enum switches and the app controls, not only the visible picker.

Test token position, layer location, chunked prefill, neutral centering,
normalization, composition order, zero vectors and dependent directions. Match
descriptor and sidecar semantics. The current Swift scope sidecar is still owed;
do not describe it as existing merely because the arithmetic is implemented.

If Swift cannot execute the new mode, retain the request and explain how to run
it with Python. Never silently omit its arithmetic. A named unsupported-operation
response is justified by inability to perform the requested computation; absence
of comparative measurements calls for an advisory instead.

## 6. Extraction recipes

Read [EXTRACTION-RECIPES.md](EXTRACTION-RECIPES.md). Python owns the functions in
`steering/vector_math.py`, names/fields in `experiment/recipe_identity.py`, and
dispatch in `experiment/extraction_workflow.py`. Swift counterparts are
`Sources/SteeringKit/Extraction/ExtractionRecipe.swift`, `SteeringVectorMath.swift`,
`Sources/ExperimentKit/RecipeIdentity.swift`, and `ConceptBuilder.swift`.

State the exact activation populations, pairing, reading position, centering,
denominator, normalization and degeneracy handling. Use independent hand-computed
fixtures and shared cross-language vectors; cross-language AST equality cannot
prove numerical equivalence. Keep old names and identity bytes stable. A new
recipe is a new name, and changes to old semantics need explicit versioning.
Reader techniques also need the split-role contract in
[REPE-IMPLEMENTATION-BRIEF.md](REPE-IMPLEMENTATION-BRIEF.md): fitting or selection
must not silently consume `finalTest` rows.

### When an addition really needs a verb or route

An ordinary managed operation reuses the existing science verbs. For a new
interaction, account for each affected surface explicitly:

1. Portable client: add the family `VERB_SPECS` entry (aggregated by
   `client_cli.CLIENT_VERB_SPECS`) and owner dispatch. Test synopsis, JSON
   envelope, actual invocation and repair text with `--root`.
2. Python engine CLI: update `cli.py` parsing/dispatch and
   `cli_envelope.VERB_SPECS`; declare the generated reference region in
   `cli_reference.py`. An engine verb is not automatically a portable-client verb.
3. Native CLI: update `ExperimentCLIFlags.swift`, owner dispatch and
   `CLIReferenceDocument.swift`; update the exact census in
   `ExperimentCLIEnvelopeTests.swift`. Native workspace selection is
   `--workspace`, not the portable client's `--root`.
4. HTTP: register the route and its role/rationale in `api/route_roles.CENSUS`.
   Runner versus workbench is an ownership boundary: local authoring must not
   accidentally become a mutation on execution hardware. Review
   `_PRIVILEGED_PREFIXES` / `_OPEN_MUTATING_PATHS` in `api/app.py` only if relevant;
   any authentication change needs an explicit reason and tests.
5. Catalog actions must name the actual method/path/service role. Update
   `ScienceCatalog` Codable fields if the shared catalog schema changes, so the
   Swift encoding does not silently drop metadata returned by Python.
6. The app calls the shared owner or route. Show outcomes and repairs where the
   action occurs; preserve authored values on failure. Regenerate each affected
   reference region and test the journey rather than just the new verb count.

## 7. Generators and current checks

Run commands from the checkout root with its supported Python environment.
`PYTHONPATH=Server` makes scripts use this tree. Generators edit committed
resources; review their diff. The order is significant: seed copies first,
workspace bootstrap next, Python source identity last. A scientific AST baseline
is not a generated resource and must never be refreshed by these commands.

```sh
python scripts/ci/check-science-resources.py --write
python scripts/ci/check-study-interviews.py --write
python scripts/ci/check-workspace-bootstrap.py --write
python scripts/ci/check-substrates.py --write
python scripts/ci/check-client-assembly-reference.py --write
python scripts/ci/check-python-client-identity.py --write
```

For a changed CLI synopsis also run the matching source-built executable:
`steerlab-cli docs cli-reference --write` (Swift regions) and
`steerlab-server docs cli-reference --write` (engine regions). The client assembly
regions use the Python script above. Do not use a stale installed executable.

| Check under `scripts/ci/` | What it establishes / baseline | Repair for a legitimate change |
| --- | --- | --- |
| `check-science-resources.py` | WorkspaceSeed ↔ Python seed ↔ `ScienceResourceText.swift`; registry/interview census and catalog routes | Edit maintained sources, then `--write`. |
| `check-study-interviews.py` | Shared study prompts ↔ Python seed ↔ `StudyInterviewText.swift` | Edit WorkspaceSeed, then `--write`. |
| `check-workspace-bootstrap.py` | Exact seed inventory, packaged copies, agent draft, compiled bootstrap | Update `client/resources/workspace.json` for new seed files; `--write`. |
| `check-python-client-identity.py` | Shipped Python/resource bytes ↔ `PythonClientIdentity.swift` | Run `--write` after all source/resource changes; rebuild the Mac before sharing changed Python with it. |
| `check-substrates.py` | Complete catalog ID coverage, explicit backend distinctions, evidence links, generated capability tables | Edit `docs/substrate-capabilities.json`, then `--write`; no invented qualification. |
| `check-client-assembly-reference.py` | Client verb table ↔ its CLI reference region | Edit family specs, then `--write`. |
| `science_cli_census.py` | Engine command/action references | Helper called by science/reference checks; repair the declaration, not an invented route. |
| `audit-managed-scientific-owners.py` | Eleven owner ASTs and three specific lens-consumer substitutions; `--base 6a94a8f` | Preserve the mechanical proof; intentional science changes need new behavioral evidence and reviewed audit scope. |
| `audit-stability-preflight.py` | Stability extraction and battery publication hook vs `73bfbd7` | Checkpoint proof; adapt only with explicit reviewed semantics, never change the pin just to make green. |
| `audit-task-prompt-parser.py` | Parser extraction vs `ef3dec8` | Same rule; preserve negative controls. |
| `audit-design-lazy-imports.py` | Variant/panel import extraction vs `82da781` | Same rule; keep portable validation independent of execution packages. |
| `audit-python-boundaries.py` | Historical refactor renames from `python-boundary-renames.json` | Historical candidate audit (`--candidate`), not a general gate for new algorithms. |
| `check-swift-bridge-retirement.py` | Current bridge/caller inventory; also run `--release` | Retire the migration; do not add compatibility bridges to pass tests. |
| `public_scan.py` | Public repository vocabulary and tracked content | Follow repository hygiene; do not add private data to exceptions. |
| `qualify-client-release.py` | Installed app-free client, optional `--repair` | Use a disposable release/environment when packaging or dependencies change. |
| `qualify-technique-example.py` | The guide's actual owner/test code in a disposable source copy | Update the example when interfaces change; use a test-capable Python. |
| `test-prompt-preview.mjs` | Prompt-preview renderer | Run when that renderer changes; not scientific qualification. |

Swift syntax audits (`audit-panel-owner-access.swift`, `audit-study-management.swift`,
`audit-exclusion-policy.swift`, `audit-authoring-http-results.swift`) preserve
specific historical migrations, not arbitrary future algorithm changes. They
take current/baseline checkout paths; the panel audit also accepts `design`.
Use the exact checkpoint and Xcode host-library compile commands in
[BRIDGE-RETIREMENT.md](BRIDGE-RETIREMENT.md) and the exclusion/HTTP entries in
[RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md](RESEARCHER-WORKFLOW-VALIDATION-HISTORY.md).
Do not point historical comparisons at today's changed tree and then weaken them.

## 8. Acceptance and handoff

For each affected surface exercise discovery, draft, actual owner validation,
review/publication, isolated execution, export/import and result inspection. Cover
one understandable invalid-input repair and one supported exploratory case that
does not require stronger scientific evidence than the user has claimed.

Use the existing `test_interview_validation.py`, `test_managed_methods.py`,
scientific execution/custody suites, `ScienceCatalogTests.swift`, and
`MethodAuthoringTests.swift` as integration starting points. Discover
the exact current test names with `rg --files`; missing tests are work to add.
For new mathematics add independent expectations, not a test that merely repeats
the implementation. For a mechanical move run its AST/syntax audit with negative
controls. Read the entire final diff and run both full suites before landing.

On the supported Mac build environment:

```sh
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
export TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.920.1
# Set this to the test-capable Python environment, even in a venv-less worktree.
export TEST_RUNNER_STEERLAB_TEST_PYTHON=/absolute/path/to/test-environment/bin/python
PYTHONPATH=Server "$TEST_RUNNER_STEERLAB_TEST_PYTHON" -m pytest Server/tests -q
xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -derivedDataPath /private/tmp/steerlab-technique-tests CLANG_COVERAGE_MAPPING=NO
git diff --check
```

This is a test-environment placeholder, not a path to store in a product artifact.
Keep scratch outside file-provider folders; do not alter a running agent's
checkout or install over the researcher's app to test a branch.

Independent acceptance: a fresh coding agent should implement the disposable
CPU example using this guide and its links, without undocumented verbal repairs.
Record where it stalled. Then repeat with a hypothetical Python-only intervention
that the Mac can discover and route honestly. These trials are qualification of
the guide, not yet a claim made by publishing it. The researcher arranges the
independent audit/integration agent; no branch lands without their review through
the researcher. Include commits, exact commands, counts and remaining limitations
in the handoff. Keep unrelated scientific fixes separate from mechanical work.
