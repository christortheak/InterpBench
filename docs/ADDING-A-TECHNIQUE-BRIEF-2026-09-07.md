# Brief: write the "adding a technique" guide, and make the additions easier

Written 2026-09-07 by the maintainer's integration agent for the refactoring
agents, on main `7284463`. Two deliverables: a guide a researcher's coding
agent can follow to add a technique, and a short set of changes that make the
guide shorter. Write the guide first from the code as it is; then propose the
simplifications as separate commits so the researcher can take or leave each.

## 1. Why this is needed

The recipe and training documents describe what exists "as executed". Nothing
walks an agent through which registry to touch, which enums exist on both
engines, which audits pin which files, and which generated resources must be
rebuilt. Today an agent learns this from failing tests, one at a time. The
guide should let it do the whole job in one pass.

## 2. The guide: required structure

File: `docs/ADDING-A-TECHNIQUE.md`. Every path below must be verified against
the tree when the guide is written, not copied from this brief.

### 2.1 First, classify the addition

Four classes, in increasing cost. The guide opens with a decision table.

| Class | Examples | Engines touched | Frozen identity touched |
|---|---|---|---|
| A. Managed analysis or method | J-lens training, SAE training, a new geometry, a new readout, a new campaign type | Python only | no |
| B. Fine-tuning method | a non-LoRA adapter family, a new objective | Python, plus Swift if MLX must train it | adapter sidecar stamps |
| C. Intervention mode | J-space projection ablation, clamping, a new centering | Python and Swift | intervention scope sidecar, manifests |
| D. Extraction recipe | a sixth recipe | Python and Swift | `recipeIdentityHash` and its fields |

State plainly that class A is a day's work for an agent and class D is the
most expensive because frozen artifacts carry its identity.

### 2.2 Class A walkthrough (the pluggable path)

1. **Owner module** under `Server/steerlab_server/experiment/` (or `jlens/`)
   with a config class exposing `from_dict` that refuses unknown keys, a
   `to_dict`, and one entry function whose signature may take `root` and
   `log`; the managed executor passes only what the signature declares.
2. **Registry line** in `experiment/managed_methods.py`: module, config
   class, function, and `cpu` or `gpu`. GPU means `modelID` and a 40-hex
   `revision` are required and the operation runs through Slurm on a
   controller; CPU means it runs locally beside the controller.
3. **Input closure**: if the config references files or artifacts by keys
   the walker in `experiment/managed_inputs.py` does not already know, add
   the key to the right role set there, so packaging and the queued-child
   re-verification see the file.
4. **Interview** entry in `WorkspaceSeed/prompts/method-guides/workflows.json`
   with fields whose ids are the owner's config keys (dotted for nesting),
   kinds from the closed kind list, defaults as text, a claim boundary and
   the three questions.
5. **Catalog and guide**: the operation in `catalog.json` under an existing
   method, or a new method with its guide file; actions are only the routes
   the census already knows.
6. **Regenerate**: `scripts/ci/check-science-resources.py --write`, then
   `scripts/ci/check-python-client-identity.py --write`.
7. **Tests that will fail until you do this**, and what each wants:
   `test_interview_validation.py` (add your operation's fixture branch; the
   case list must equal the registry), `check-science-resources.py`
   (interviews cover the registry), `test_python_client_identity.py`
   (regenerated constant), the CLI reference tests if you added a verb.
8. **Docs**: one paragraph in the method guide, a CHANGELOG bullet, and if
   the method is a new category, the science-resource gate's category check.

Show a complete worked example in the guide: add a fictional CPU operation
end to end, with the diff for each step, and delete it at the end.

### 2.3 Class B walkthrough

Owners `experiment/lora_train.py` and `lora_data.py`; the comparability
matrix in `docs/TRAINING-RECIPES.md` (a new method is a new row with every
column answered, or it may not be compared); adapter attachment through
model variants and the adapter sidecar stamps; the Swift `FineTuneStore` if
MLX must train it, otherwise a stated "server-only" line in the matrix.

### 2.4 Class C walkthrough

Python: `steering/intervention.py` (the `PATHS` and centering constants are
the vocabulary; extend, never rename), `steering/plan.py` descriptors,
`steering/injector.py` or `ablator.py` for the arithmetic,
`experiment/condition_execution.py` and `experiment/model_variant.py` where
the closed `add`/`ablate` mode is read. Swift: the matching mode enum in the
manifest and variant stores, `InterventionPlan`, and the UI picker. Docs:
`docs/INTERVENTION-SCOPE.md` gains the new path with its centering rule and
its sidecar stamp. Tests: the scope sidecar tests on both engines. Say in the
guide that a mode with no Swift arithmetic must stamp "server-only" and be
refused on the Mac by name, never silently skipped.

### 2.5 Class D walkthrough

Python: the function in `steering/vector_math.py` (documented 1:1 twin of
`SteeringVectorMath.swift`), the recipe name and identity fields in
`experiment/recipe_identity.py`, the workflow branch in
`experiment/extraction_workflow.py`. Swift: `ExtractionRecipe.swift`,
`SteeringVectorMath.swift`, `ConceptBuilder.swift`. Docs:
`docs/EXTRACTION-RECIPES.md` gets a numbered section in the same form as the
five existing ones, including what enters the identity hash. Tests: the
vector-math parity tests and the identity-hash tests on both engines. The
guide must say that an existing artifact never changes identity; a new recipe
is a new name.

### 2.6 Gates: the complete list, what each pins, and how to re-pin

A table. For each script under `scripts/ci/` and each generated file: what it
checks, which base commit or file it pins, the exact command to regenerate or
re-pin, and the rule that re-pinning happens in the same commit as the
legitimate change with the reason in the message. Cover at least:
`audit-managed-scientific-owners.py` (eleven owner ASTs and three lens
consumers against `--base`), `audit-stability-preflight.py`,
`audit-task-prompt-parser.py`, `audit-design-lazy-imports.py`,
`audit-python-boundaries.py` (scoped to one historical commit; not a gate for
new work, say so), `check-science-resources.py --write`,
`check-python-client-identity.py --write`, `check-study-interviews.py`,
`check-client-assembly-reference.py`, `check-swift-bridge-retirement.py`
normal and `--release`, `public_scan.py`, the Swift audits, and the generated
`docs/CLI-REFERENCE.md` with its Python and Swift region tests.

### 2.7 Adding a CLI verb or a route (a checklist, because these are lists)

Python client: `client_cli.CLIENT_VERB_SPECS` or the family's `VERB_SPECS`.
Mac: `ExperimentCLIFlags.swift` specs, `CLIReferenceDocument.swift` labels,
the expected list in `ExperimentCLIEnvelopeTests.swift`. Route:
`api/route_roles.CENSUS` with role and rationale; the privileged-prefix or
open-mutating lists in `api/app.py` only with a written reason. Catalog
actions must match the census exactly. Then regenerate the CLI reference.

### 2.8 Substrate rule

Every new technique states which engine produces it and which reads it, in
the guide's template and in the catalog restriction text. Python-only is an
acceptable answer; silence is not.

## 3. Making it easier: proposals to implement after the guide

Each is a separate commit with its own tests; none changes scientific
behaviour. Propose more if the guide reveals them.

1. **One registration point for class A.** Today a managed method is
   declared in three places (registry, interview, catalog) plus two
   regenerations. Let the registry carry the interview fields and catalog
   text, and have the resource script emit `workflows.json` and the catalog
   operation from it, so an agent edits one Python file and runs one script.
   Keep the generated JSON committed and gated so the app and clients still
   read files.
2. **One regeneration command.** A `scripts/ci/regenerate.sh` that runs the
   resource, identity and CLI-reference generators in the right order, and a
   `scripts/ci/gates.sh` that runs every check. The guide then says "run
   regenerate, then gates" instead of listing eight scripts.
3. **Audit base pins as data.** Move the `--base` defaults of the AST audits
   into one small file that names each pinned owner with its base commit and
   a reason line, so re-pinning is a visible one-line diff reviewers can read.
4. **A scaffold verb.** `steerlab-server science scaffold <operation-id>`
   that writes the owner module skeleton, the registry line, the interview
   stub with the owner's config keys, and the test fixture branch, all marked
   `TODO`, so the agent starts from a failing-but-complete shape.
5. **Closed-set enums with one source.** For intervention modes and recipe
   names, generate the Swift enum cases from the Python constants (or the
   reverse) with a gate that the two agree, the way the science resources are
   already gated. This turns "remember the Swift enum" into a failing test
   with the missing case named.
6. **Registry-driven tests.** Wherever a test enumerates operations by hand,
   derive the list from the registry and fail if a fixture is missing; the
   new interview-validation test already does this and is the model.

## 4. Acceptance

The guide is accepted when a fresh agent, given only the guide and a one-line
description of a fictional class A operation, lands it with both suites green
in one attempt without reading any other document. Test that literally,
record the transcript summary in the validation ledger, and fix the guide
where the agent stalled. Then repeat for class C with a fictional mode that is
refused on the Mac by name.

Public repository hygiene applies to the guide and to every example in it.
