# Diagnose extraction stability

Check sensitivity to resampling without mistaking it for behavioral validation.

Select a concept already declared in a study, its supported meanDifference or lat recipe, model/revision, resample count, fraction and seed. LAT additionally has order-shuffle sensitivity because row order enters its arithmetic.

The stability operation captures the same extraction rows as the engine, then resamples them. It writes a diagnostic outside immutable runs and does not replace the pinned vector. Read stimulusDrift: the diagnostic can report live stimulus bytes that differ from the study's pins. Restore/review those inputs rather than interpreting the result as a reading of the old artifact.

Use `steerlab-server experiment extract-stability <study> <concept> --resamples 32 --fraction 0.8 --seed 1 --json` on the compute host. For a managed remote job, use `runner science-plan` / `runner science-submit` (Python client), `remote science-plan` / `remote science-submit` (Mac), or `POST /api/science/plan` and `/api/science/submit`. Supply operation `stability` with parameters `experiment`, `concept` and optional `resamples`, `fraction`, `seed`, `orderShuffles`, `dtype`, `device`. Submit the exact returned `planSHA256`. Use `science input-plan` and `science package` to review and capture local inputs. Transfer the archive under the declared policy, then `runner science-stage` (Mac `remote science-stage`) returns the isolated execution request. Plan and submit in the same controller session; restart requires a new plan. The child rechecks inputs after queueing, and the job retains the diagnostic path. It does not resume from a checkpoint. CPU resampling still requires a model activation capture first. Mac numerical parity is not claimed; a Mac user can run the Python engine through its captured endpoint.

Report per-layer cosine movement, sign flips and the resampling settings. Stability does not show behavioral efficacy, construct specificity or a safe dose. Follow with held-out validation and a controlled behavioral study; do not invent a universal acceptance threshold.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.

Remote requests accept the seed as an unsigned integer or decimal string; plans return decimal text to preserve all UInt64 digits across clients.

Completed diagnostics can travel home through `runner science-fetch` (Mac `remote science-fetch`) or permitted external transfer followed by `science import`. `science custody` and `science verify-custody` re-read retained archives and expanded files offline. A verified receipt proves byte custody, not scientific qualification. Managed removal requires a declared server retention policy, reviewed `cleanup-plan`, fresh local verification and explicit `cleanup-apply --confirm-removal`; only isolated successful diagnostic output copies are eligible. Inputs, export archives, job records and local evidence are retained.
