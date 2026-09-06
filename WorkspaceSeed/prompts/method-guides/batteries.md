# Measure capability and generation health

Keep the standalone floor battery distinct from a study control.

Ask which agents and model revisions are being compared, which capability domains matter, and how any equivalence claim will be assessed. Keep the floor battery study-blind and fixed before seeing the study's desired effect.

A study pins a batteryFormat:2 capability battery for per-condition control. A standalone batteryFormat:3 floor battery adds long-form generation health and cannot be pinned as that study control. Use authoring prompt battery and steerlab-server battery generation-prompt for the applicable schema/charter; do not convert a format-3 file to format 2 by deleting its header.

The standalone operation takes a battery file and named agent references, including baseline, an artifact, or a concept:layer:alpha condition. It does not take a study manifest. There is no standalone HTTP or bundle execution route. Inspect and stage immutable inputs on the compute host before calling the engine CLI. The engine's own preflight checks model identity, pins, rendering and regime support.

Outputs are battery.jsonl and battery-report.json in a new run directory. Graded accuracy and long-form health are different regimes. A format-2 standalone reading cannot measure the missing long-form regime and reports that limitation. Capability equivalence needs a declared design; a successful run or similar point estimates do not prove it. Cancellation of observation is not proof that computation stopped.

## Supported execution

Run `steerlab-server battery generation-prompt --help` to author a study-blind floor brief, then `steerlab-server battery lint <file> --json`. Run `steerlab-server battery run <file> --agent baseline --model <model> --revision <revision> --dry-run --json` to inspect the preflight. Remove `--dry-run` only after reviewing that result. Add `--agent` once per declared agent; use `battery run --help` for its exact artifact and condition syntax.

For a format-2 study control, use `authoring prompt battery --name <name> --json` and the normal reviewed `experiment pin-battery` workflow. The standalone format-3 generator owns its different item schema. Keep that schema attached to coworker/reviewer deliveries; the two formats have different outcome regimes.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.
