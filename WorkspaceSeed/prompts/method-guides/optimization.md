# Optimize a steering vector

Train an intervention against a declared objective and evaluate it on separate data.

Ask the researcher to specify the objective, construct, counterfactual/control, model/revision, injection sites, dose convention and train/dev/test split. Optimization data and final evaluation data have different roles.

Author the choice-row bundle below, then run `steerlab-server data check optvec --dir <directory> --json`. The prompt emitter has no OptVec-specific kind; use the coworker brief in this guide. Preserve item IDs, option order, declared target labels and pinned split files. Do not infer target labels from option length, topic or a conveniently available answer column.

The engine's optvec family exposes train, eval, geometry, interpret, family, fracture, gradient, jspace and campaign. Use the exact operation references below and each verb's public help; their closed config validators are the authority. Existing app execution controls live under Agents / Optimizations. A CLI-only analysis or campaign is explicitly listed as such rather than replaced by a different API request.

Training produces a vector artifact and optimization record. Evaluate held-out behavior and capability costs; geometric similarity, a training objective improvement and a family plot answer different questions. Report checkpoint selection and the dose actually tested. Never tune a final test split or describe a null-direction comparison as a calibrated statistical test without a declared test design.

## Dataset bundle and author brief

A choice row declares the target explicitly, even though the loader has a default:

```json
{"id":"target-train-1","prompt":"A described situation. Choose A or B.","options":["A","B"],"target":"B"}
```

Author `target-train.jsonl`, `target-val.jsonl`, `target-test.jsonl`, `anchor-train.jsonl`, `anchor-val.jsonl`, `anchor-test.jsonl`, `capability-train.jsonl`, `capability-eval.jsonl` and `neutral-fluency.jsonl` (text rows). Keep IDs unique across all files, target A/B balance within 45–55% per choice file, and content/fact patterns disjoint across splits. Tokenizer checks still have to establish single-token options on the pinned model; a single character does not prove that.

Publish `bundle.json` with nonempty `targetIssue`, `shiftDirection`, `caseFamilies` and `anchorIssues`, and a `files` object whose entries each contain `path` and the actual file `sha256`. Include `REPORT.md` with leakage/nuisance checks and unresolved scientific questions. These are dataset records, not changes to an existing study manifest. Use the readiness result's hashes and the selected operation's closed config; never copy placeholder hashes into a training declaration.

Give the coworker the approved target direction, families of situations, invariant anchor judgments and capability domains. Ask for the complete bundle and separate QC report; give another reviewer the inputs and rationale before any training. A predeclared fixed-steps training configuration can omit target validation; report that choice and do not silently reuse the test set for checkpoint selection.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.

## Shared method authoring and managed execution

Use `science interview <operation> --json` to read the exact form fields used by the app. Answer the purpose, proposed claim, controls and selection questions; supply field values as strings (including integer seeds) and put reviewed extra engine settings in `advanced`. `science draft <operation> --answers <answers.json> --json` resolves file hashes and reports all captured inputs without execution. `science publish <operation> --answers <answers.json> --destination requests/<new-name> --plan-sha256 <reviewed-hash> --json` publishes the machine request and rationale together, refusing changed inputs or an existing destination. Both commands are available under each client executable.

Use the resulting request file with `science input-plan` and `science package`, transfer and stage through the declared site policy, then review and submit the engine plan. The shared form is an authoring aid; the engine still checks its complete scientific config before execution. A successful parser or custody check is not scientific qualification. The workbench API exposes the same local owner at `POST /api/science/workspace/{action}`; it takes `workspaceRoot`, `operation`, and for draft/publish `answersText` (the exact JSON text). Publish also requires `destination` and `planSHA256`. Runner services cannot author the local workspace.
