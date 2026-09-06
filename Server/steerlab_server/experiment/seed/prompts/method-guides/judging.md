# Complete deferred judgments

Maintain blinding, packet identity and the declared judge panel.

Decide the rubric, judge panel, evaluation sample and intended reliability claim before scoring. Use independent judgments rather than asking a coding agent to invent favorable labels.

Inspect experiment sweep/evaluate awaiting through the existing workbench/runner endpoints. Hand each coworker the pinned blinded packets and their instructions, not an unblinded arm map. Require the operation's exact judgment row schema, packet IDs, judge identities and instruction pins; never fabricate missing judgments or edit packet files to make coverage pass.

Complete through the existing sweep/evaluate complete-judgment routes or `steerlab-server experiment complete-sweep-judgment` (sweep) and `steerlab-server experiment complete-judgment` (evaluate). Both take `<study> --awaiting-run <run> --judgments <file> --json`. These owners verify coverage, packet hashes and epoch. A sweep completion can append the selected recommendation to a draft; evaluate completion aggregates judgment evidence. Preserve that distinction and the owners' lifecycle gates.

Outputs are judgment records, agreement/report data and, where applicable, the recommendation. Judge agreement alone does not establish validity. Missing/partial coverage, stale packets or a changed study must produce the owner's repair, not an automatic retry with new pins.

## Intake shape and blinding

Fetch `GET /api/experiment/{name}/sweep/awaiting` or `GET /api/experiment/{name}/evaluate/awaiting` on the selected engine. An example row shape is:

```json
{"packetID":"issued-packet-id","judge":"declared-judge-id","winner":"A","model":"declared-judge-model"}
```

IDs/model strings above are placeholders for the issued packet and pinned judge, never new identities. Winner is `A`, `B` or `tie`. CLI input accepts a JSON list, a `{"judgments":[...]}` object, or JSONL. HTTP bodies name `sweepRun` or `evaluateRun` plus `judgments`. Evaluation accepts the optional `instructionsSha256` claim; its owner records mismatches as warnings, not a hard gate. Sweep has no such claim field; its CLI refuses one rather than dropping it.

Supply the full packet × judge coverage required by the issued manifest, including the actual model/annotator declarations it requests. A coder should prepare delivery and check completeness without seeing the unblinding map; independent judging coworkers see only the assigned blinded packets and pinned instructions. Completion is idempotent: an existing result can be returned, and a sweep may heal a missing draft recommendation projection. It does not rewrite its source evidence.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.
