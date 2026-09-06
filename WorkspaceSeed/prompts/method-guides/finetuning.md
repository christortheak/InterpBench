# Prepare and compare fine-tuned adapters

Separate training-recipe choices from subsequent study claims.

Choose the training objective, supervised output, dataset splits, base model/revision, target modules, scaling, seed, checkpoint-selection rule and comparison intervention with the researcher.

Use the LoRA row schemas and coworker brief below, then `steerlab-server data check lora <package-manifest-or-dir> --json`. There is no LoRA kind in the generic prompt emitter. Use the workbench finetune plan operation before train or submit; the Python engine CLI also has finetune plan/train/submit. The existing Adapters pane reaches these owners. A plan is not training success, and a scheduler receipt is not a completed adapter.

Keep labels, masks and split roles explicit. Compare training recipes on objective/masking, effective adapter scaling, optimization schedule, target layers, precision and checkpoint selection; shared rank or a similarly named alpha does not make two recipes matched. Preserve dataset hashes and the resolved model revision in the training artifact.

Inspect the produced adapter and its qualification/evaluation evidence before composing an agent. Attach the reviewed agent to an ordinary comparison study with baseline and held-out prompts. A fine-tuning improvement on its training objective is not evidence that a steering vector and adapter implement the same intervention.

## Dataset rows and training declaration

Evidence-grade document mode has one complete example per row:

```json
{"id":"document-1","text":"One independently sourced training document."}
```

Instruction-chat mode uses user/assistant text and optional system text:

```json
{"id":"instruction-1","system":"Follow the declared task.","user":"A training input.","assistant":"The approved target response."}
```

The loader also accepts `prompt`/`completion` aliases; normalized rows use `user`/`assistant`. Unknown row keys refuse. Do not combine examples into a token stream or silently truncate them. Declare the long-document split/refuse policy. Keep training and validation files independently authored; retain a separate final evaluation corpus.

The package manifest names each arm's `adapter`, `training` and `validation` references with actual `path`/`sha256` values and the package QC report. Run the data check before authoring a training configuration. Select `document` or `instruction_chat`, base model/revision, dataset pins and evidence-grade policy explicitly in the fine-tuning plan; the workbench exposes its current request schema at `/openapi.json`. The engine CLI takes `finetune plan --config <file> --json` before `train` or `submit` with that reviewed config. An exploratory inline recipe is not interchangeable with the pinned dataset recipe.

Ask the coworker to return the chosen row format, independent splits and an audit of source permissions, duplication, label correctness and systematic nuisance differences. Ask the reviewer to compare intended supervised tokens with the selected engine recipe before approving a causal comparison.

## Engine config spelling

The engine CLI config is **snake_case**, unlike the workbench HTTP body's camelCase wire. Use `base_model_id`, `revision`, `training_mode`, `train_paths`, `validation_paths`, `expected_hashes` (path to actual SHA-256), `dataset_root`, `dataset_manifest_path`, `dataset_manifest_hash`, `dataset_bundle_id`, `reserved_evaluation_hashes` and `evidence_grade`. Training choices include `rank`, `alpha`, `dropout`, `learning_rate`, `target_modules`, `dtype`, `seed`, `epochs`/`max_steps`, `batch_size`, `gradient_accumulation`, `warmup_steps`, `lr_schedule`, `max_grad_norm`, `weight_decay`, `max_sequence_tokens`, `long_document_policy`, `chunk_overlap_tokens`, `eval_interval_steps`, `checkpoint_interval_steps`, `selection_metric` and `control_arm`. Unknown fields refuse. Do not paste HTTP camelCase directly into the engine config.

The plan validates this declaration and reports the dataset, schedule and evidence refusals before training. Ask about substantive unset choices; a library default is not researcher approval. Keep the complete reviewed plan beside the proposed config in the handoff.

## Coworker author prompt

Help author the inputs for this method. First restate the researcher-approved construct, comparison, model/revision, input roles and proposed claim. List unresolved scientific choices as questions; do not choose them silently. Use the schema and public operations above and the selected authoring prompt. Return proposed files separately from an audit describing split independence, labels, nuisance balance, applicability and missing facts. Do not execute, invent pins, or overwrite evidence.

## Independent review prompt

Review the proposed inputs without assuming the author is correct. Check the declared method, schema, labels, split overlap, source/identity pins, baseline and controls, rendering/sampling settings and whether the requested claim follows from the planned measurements. Separate mechanical checks from scientific judgment. Name each blocker and its repair; passing a parser is not scientific validation.
