# Fit a J-lens: implementation and acceptance plan

Start from landed main `2b67199`, in `codex/jlens-fitting`.

The researcher supplies a text corpus and a prepared Hugging Face checkpoint.
The Python engine fits average residual-stream Jacobians with the pinned jlens
reference kernel. This does not change model weights, train an SAE, or optimize
a steering vector. UI, both CLIs, and HTTP share one managed operation.

## Deliverables

1. A portable, strict configuration and corpus format: JSONL objects with unique
   `id` and nonempty `text`. Pin all corpus bytes, model revision, dtype,
   rendering, source layers, final target, and estimator settings. Plain text is
   tokenized as raw text; no hidden chat-template insertion. A small prompt limit
   provides a pilot. Corpus order is explicit, never silently shuffled.
2. A model adapter over the pinned reference package, with local prepared-model
   loading only. Preserve the reference estimator: sum cotangents over valid
   target positions, average valid source positions, then average prompts.
   Record short prompts skipped and surface all other numerical errors.
3. Append-only checkpoint snapshots with input/configuration identity and tensor
   hashes. A continuation copies a selected checkpoint into a fresh run; it
   never modifies the old run. Model, corpus, estimator, and geometry must match.
   Increasing the prompt budget is permitted; scheduler auto-resubmit is not.
   Cancellation retains completed snapshots. An explicit snapshot can travel as
   ordinary reviewed scientific input; partial jobs are not evidence exports.
4. A complete output run with preserved corpus bytes, effective configuration,
   timings, diagnostics, a final safetensors lens, and an artifact-import
   description. Normal scientific export/import returns that run, and the
   existing reviewed artifact importer registers the returned lens. Keep the
   fit run immutable and qualification in the separate lens library.
5. Shared catalog/interview/agent guide, a discoverable Fit a new lens button in
   the lens dialog, and the existing author/review/execute/collect interfaces.
   Explain corpus choice, pilot cost, incomplete layer coverage, and intended
   use. No invented corpus, automatic delegation, or unrequested downloads.
6. Independent analytic Jacobian fixtures, reference agreement, continuation
   equivalence, input drift and checkpoint corruption checks, portable closure
   and return tests, both suites, generated resources, and existing AST audits.

## Scientific and deployment boundaries

Fitting completion is not qualification or evidence of behavioral meaning.
Qualification and held-out assessment remain separate follow-up actions. A
particular architecture's backward path must be measured, not inferred from
whether inference works. The current target requested by the researcher uses
hybrid attention; do not claim large-model support from toy fixtures alone.
Use generic fixtures and identifiers in committed tests. CPU toy fitting is
acceptable for numerical tests; real cluster GPU qualification is handed to the
maintainer's agents. Do not deploy, install an app, or merge main here.
