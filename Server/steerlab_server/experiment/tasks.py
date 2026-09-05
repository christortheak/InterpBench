"""Compatibility API for experiment execution.

Stage implementations live in focused workflow modules. Existing public entry
points and private import spellings remain available during migration. Inject
private dependencies at their owning module; pipeline stage entry points are
bound here at invocation time for existing API/CLI integrations.
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
import random
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable
import torch
from .. import memory_diagnostic
from ..steering import model_loader, vector_math as vm
from ..steering import residual_norm_convention as norm_convention
from ..steering.extractor import ExtractionOptions as CoreExtractionOptions, extract as core_extract
from ..steering.stimulus_set import StimulusSet, load_texts
from ..steering import vector_store
from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar, save as save_vectors
from . import judging_custody
from . import judicial, lifecycle_gates, paths, prompt_render, recipe_identity
from . import response_format
from . import resume as resume_mod
from . import run_epoch
from . import sharding as sharding_mod
from . import system_prompt as system_prompt_mod
from . import choice_deltas
from . import truncation_gate
from . import turn_endpoint
from .generate import CellInjection, generate
from . import manifest as manifest_mod
from .manifest import JudgeRef, Manifest, VariantCondition
from .run_config import write_run_config
from .run_status import RunStatus, heal_after_completion
from .scoring import MarkerRubric, distinct_bigram_ratio, word_count
from .sampling import (
    derive_seed,
    seeded_generation as _seeded_generation,
)
from .task_inputs import (
    missing_task_prompts_refusal,
    missing_task_prompts_repair,
    load_prompts as _load_prompts,
    check_response_formats as _check_response_formats,
    check_transcript_prompts as _check_transcript_prompts,
    resolve_ordinal_aggregation as _resolve_ordinal_aggregation,
)
from .run_reporting import (
    write_metrics_csv as _write_metrics_csv,
    write_summaries_csv as _write_summaries_csv,
    reasoning_style_summary as _reasoning_style_block,
    choice_readouts as _choice_readouts,
    write_report as _write_report,
)
from .pipeline_ledger import (
    pipeline_ledger_path as _pipeline_ledger_path,
    write_pipeline_ledger as _write_pipeline_ledger,
    read_pipeline_ledger as _read_pipeline_ledger,
)
from .run_artifacts import (
    _model_dtype,
    _actual_dtype,
    _write_config_snapshot,
    _model_capabilities_for_run,
    _advise_capabilities,
    _RENDERING_RUN_TYPES,
    _latest_run,
)
from .study_admission import (
    _verify_or_warn,
    _stamped_experiment_hash,
    _require_source_epoch,
    _advise_implicit_case_family,
)
from .analysis_endpoints import (
    _condition_modalities,
    _key_records_by_transcript,
    _transcript_level_diffs,
    _endpoint_values,
    _item_factor_levels,
    _stratification_families,
    _endpoint_sample_values,
    _stratified_effect_rows,
    _promotion_decisions,
)
from .analysis_workflow import (
    analyze,
    rescore_style,
)
from .cancellation import (
    _observe_cancel,
    TaskCancelled,
    _cancel_checkpoint,
)
from .judge_dispatch import (
    _judge_roster,
    _preflight_openrouter_judges,
    evaluate_fanout_judge_models,
    write_judge_fanout_request,
    read_judge_fanout_request,
)
from .judgment_evidence import (
    _verify_judgment_marker,
    _sweep_judging_manifest,
    _find_judgment_run,
    _evaluate_judging_manifest,
    _verify_evaluate_marker,
    _find_evaluate_judgment_run,
)
from .pipeline_evidence import (
    _restore_self_pinned_revision,
    _stamp_pipeline_drift,
    list_pipeline_runs,
    _expected_promotion_identity,
    _find_minted_agent,
    _minted_agent_matching,
)
from .pipeline_policy import (
    _pipeline_will_judge,
    _pipeline_inline_judging_preflight,
    _pipeline_needs_model,
)
from .judgment_evidence import JUDGMENT_MARKER_SCHEMA
from .pipeline_ledger import PIPELINE_LEDGER_SCHEMA
from .model_resources import (
    _effective_dtype,
    _load_model,
    _acquire_model,
    _assert_resident_dtype_matches,
    _pin_model_revision,
)
from .execution_reporting import (
    _sampling_metadata,
    _write_substrate,
    _preview_line,
    _advise_cross_substrate,
    _advise_dependency_lock_drift,
    _advise_system_prompt_divergence,
    _sha256_text,
)
from .vector_materialization import (
    ConceptVectorBundle,
    _extract_all,
    _sha256_file,
    _materialize_pinned_artifact,
    _extract_designated_reference,
    _extract_grand_mean_bundles,
    _persist_vectors,
)
from .condition_execution import (
    _residual_norm_at,
    _condition_injections,
    _warn_on_mean_aligned_ablation,
    RANDOM_VECTOR_ALGORITHM,
    _matched_norm_random,
    _check_option_lengths,
    _sae_latent_preflight,
    _advise_sweep_ignores_sae_latent,
    _materialize_sae_latent_conditions,
    _effective_sae_latent_condition,
    _intervention_state,
    _reader_scorers,
    _reader_scores,
    _PROMPT_META_KEYS,
    CHOICE_INSTRUMENTS,
    EffectiveCondition,
    _effective_ordinary_condition,
    _variant_intervention_state,
    _effective_variant_condition,
    _run_arm_system_prompts,
    _require_manifest_sampling_policy,
    _adapters_suspended,
    effective_sample_count,
    _execute_condition,
)
from .extraction_workflow import (
    extract,
    _advise_inert_declarations,
    _write_reading_position_diagnostics,
    LOGIT_LENS_VOCABULARY_TOP_K,
    _write_logit_lens_vocabulary,
)
from .validation_workflow import (
    validate,
    _autopin_capability_battery,
    _validate_impl,
    _extract_validation_controls,
    _undeclared_control_advisories,
    _battery_results,
)
from .layer_resolution import (
    _require_uniform_depth,
    _matrix_layers,
    _validation_layer_resolutions,
    _resolution_block,
    DEFAULT_SWEEP_LAYER_FRACTIONS,
    DEFAULT_SWEEP_ALPHAS,
    resolve_sweep_layers,
    concept_sweep_layers,
)
from .sweep_workflow import (
    sweep,
    _sweep_with_spec,
    _sweep_impl,
)
from .choice_scoring import (
    _score_choice,
    _battery_backends,
    _choice_target_logprobs,
    _mean_logprob_shift,
)
from .sweep_judging import (
    _judge_preference,
    _judge_preflight,
    _assert_study_model_judge_matches_held,
    _sweep_judge_panel,
    _emit_judging_packets,
    _write_deferred_judging,
)
from .sweep_evidence import (
    list_awaiting_judgment,
    _conditions_from_recommendations,
    _project_judgment_conditions,
    complete_sweep_judgment,
    _sweep_progress_path,
    DEV_GENERATIONS_FILE,
    DEV_GENERATION_TEXT_LIMIT,
    _dev_generations_path,
    _dev_generation_key,
    _load_dev_generation_keys,
    _append_dev_generation,
    _load_sweep_progress,
)
from .run_workflow import (
    run,
    _run_impl,
    _run_capability_battery,
)
from .run_preflight import (
    _token_preflight_or_warn,
    _instrument_preflight,
    _response_format_preflight,
    _panel_load_model,
    _artifact_preflight,
    _scenario_preflight_or_warn,
    _memory_preflight_or_stay_silent,
    _resume_admission,
    _panel_resume_admission,
)
from .panel_workflow import (
    _panel_transcript_directory,
    _PANEL_TRANSCRIPT_ARTIFACTS,
    panel_transcript_completeness,
    _advise_panel_transcripts,
    _panel_records_from,
    _park_panel_run,
    _run_multi_agent_study,
)
from .run_readouts import (
    _verified_identity_for,
    _require_verified_variant_identities,
    _open_jlens_trace,
)
from .rubric_inputs import (
    no_rubric_refusal,
    no_rubric_repair,
    missing_rubric_refusal,
    missing_rubric_repair,
    _resolve_rubric,
)
from .forward_resolution import (
    _verify_agent_concrete,
    _concrete_from_pin,
    _resolve_forward_variant,
    _resolve_manifest_forward_refs,
)
from .judge_resources import (
    ModelIdentity,
    study_model_identity,
    judge_model_identity,
    judge_models_still_needed,
    judge_slots_required,
    _identity_text,
    _release_models_for_judge,
    _judge_callable,
    _coder_callable,
    _local_judge_generation,
    _missing_external_credentials,
    judging_custody_plan,
    log_judging_custody,
)
from .evaluation_evidence import (
    _normalized_judge_entries,
    _judgment_key,
    _verify_judgment_provider,
    _verified_judgment_payload,
    _agreement_entries,
    _load_human_validation,
    _materialize_human_validation,
    JUDGING_CONTEXT_FILENAME,
    _judging_context,
    _load_resumable_judgments,
    _judgment_stamp_judge,
)
from .evaluation_workflow import (
    _evaluate_response_coding,
    evaluate,
)
from .deferred_evaluation import (
    _emit_evaluate_judging,
    judge_worker,
    read_judge_worker_artifact,
    list_awaiting_evaluate_judgment,
    _instructions_intake_stamp,
    complete_evaluate_judgment,
)
from .judge_dispatch import JUDGE_FANOUT_REQUEST_FILE
from .analysis_endpoints import _STRATUM_JOIN
from .analysis_endpoints import _PRIMARY_ENDPOINT_ORDER


def pipeline(name: str, root: str | None = None, dtype: str = "auto",
             device: str | None = None, *, model_provider=None,
             should_cancel: Callable[[], bool] | None = None, log=None,
             checkpoint: "resume_mod.CheckpointFlag | None" = None,
             pipeline_run_directory: str | None = None,
             on_pipeline_directory: Callable[[str], None] | None = None,
             model_release=None) -> str:
    """Execute the declared pipeline through explicit stage/model capabilities."""
    from .pipeline_workflow import PipelineModels, PipelineStages, pipeline as execute
    from . import promote as promote_lib
    return execute(
        name, root, dtype, device, model_provider=model_provider,
        should_cancel=should_cancel, log=log, checkpoint=checkpoint,
        pipeline_run_directory=pipeline_run_directory,
        on_pipeline_directory=on_pipeline_directory, model_release=model_release,
        stages=PipelineStages(extract=extract, validate=validate, sweep=sweep,
                              run=run, evaluate=evaluate, analyze=analyze,
                              promote=promote_lib.promote),
        models=PipelineModels(acquire=_acquire_model, pin_revision=_pin_model_revision))
