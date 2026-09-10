# Fitting corpus preparation and remaining automation

This branch starts from `abcb42a`. Maintainer agents review and integrate it;
main, the installed app, cluster deployments, and existing runs stay untouched.

## This implementation

Researchers choose existing text, not automatically generated study data.
One portable owner prepares local text, JSONL, CSV, and Parquet sources, or
explicit public Hugging Face dataset files pinned to a repository revision.
It selects records reproducibly, previews examples and sampling limitations,
and publishes fitting JSONL alongside a provenance receipt. A local tokenizer
can report truncation and unusably short rows without loading model weights.
The app, both CLIs, and the workbench API call this owner.

Preview captures a bounded candidate in workspace scratch. Publication verifies
those captured bytes and creates a new destination; it does not silently
resample changed sources. Fitting can pin and capture the preparation receipt.
Source data, old requests, frozen studies, and runs are never rewritten.

The UI should explain source selection, sampling, and separate assessment data
before exposing storage details. Agent instructions must ask the researcher
which text population and sampling policy they want; preparing text does not
authorize fitting or delegating dataset authorship.

## Follow-up review disposition

- N1: name changed review information as a possible reason to review again.
- N2: retain dependency versions in compatibility checks until measured
  continuation establishes a safe migration rule.
- N3: explain literal device-name differences and their repair. Do not assume
  `cuda` always means `cuda:0`; it means the runtime's current CUDA device.
- N4: legacy compatibility is fixture-tested, not exercised on live checkpoints.
- N5: retain the live qualification and operational work below.

## Remaining automation, in order

1. Run an explicitly authorized small CUDA pilot on the intended exact model.
   Measure finite outputs, peak GPU and CPU memory, seconds per usable row,
   checkpoint I/O, and dimension batches. The 27B hybrid-attention target is
   unverified; batch 1 remains the conservative default until measurement.
2. Turn measured pilots into resource and wall-time recommendations. Add
   within-row progress and deadline-aware checkpointing where the reference
   permits it. Exercise continuation into a fresh run after interruption.
3. Provide held-out readout assessment and comparisons across independent fits
   and corpus budgets. Distinguish scientific assessment from backend
   qualification and successful fitting.
4. Exercise fit → collect → verify custody → register → qualify → select in
   the library against the release-stage cluster deployment; then automate
   the resumable handoffs around the existing owners.
5. Design reviewed storage retention and cleanup for partial runs and scratch
   checkpoints. Do not delete either implicitly. Evaluate large-artifact
   transfer limits and cluster filesystem policies before larger fits.
6. Add multi-GPU placement only if measured model sizes require it. Any
   distributed fitting/merge needs disjoint corpus accounting, compatible
   identities, and correct weighting of usable prompts.

## Review gate

Read the complete diff, run both full suites serially with Xcode beta and
external build scratch, and run generated-resource and historical AST audits.
Do not describe the preparation addition as a mechanical move. Preserve the
fitting numerical loop and verify it with the existing AST gate. The maintainer's
coding/auditing agents review through the user before integration.
