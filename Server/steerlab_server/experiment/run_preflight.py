"""Pre-model admission for prompts, instruments, artifacts, scenarios and resume identity.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import json
import os
from ..steering import model_loader
from . import lifecycle_gates, paths, sampling
from . import resume as resume_mod
from . import sharding as sharding_mod
from . import study_admission
from . import task_inputs


def token_preflight_or_warn(manifest, prompts_file, root, _log) -> None:
    """Refuse a study whose prompts cannot fit, naming EVERY offending item.

    Never drops or truncates: silently excluding items would change the
    measured sample without recording it. A preflight that cannot run (no
    cached tokenizer, unreadable config) warns and yields to the
    in-generation check rather than blocking the run."""
    from . import token_preflight
    try:
        prompts = task_inputs.load_prompts(manifest, prompts_file, root)
        report = token_preflight.preflight(
            prompts,
            model_id=manifest.model_id,
            revision=manifest.model_revision,
            prompt_mode=manifest.prompt_mode,
            system_prompt=manifest.system_prompt,
            qwen_thinking_enabled=manifest.qwen_thinking_enabled,
            max_tokens=manifest.max_tokens,
            reasoning_effort=manifest.reasoning_effort,
            reasoning_max_tokens=manifest.reasoning_max_tokens)
    except token_preflight.PreflightError as exc:
        _log(f"token preflight unavailable ({exc}) — the in-generation "
             "context check remains the backstop")
        return
    except Exception as exc:  # noqa: BLE001 — see the docstring
        _log(f"token preflight unavailable ({type(exc).__name__}: {exc}) — "
             "the in-generation context check remains the backstop")
        return
    refusal = token_preflight.refusal(report)
    if refusal:
        raise RuntimeError(refusal)
    if report.get("promptBudget"):
        worst = max((i["promptTokens"] for i in report["items"]), default=0)
        _log(f"token preflight: {report['itemCount']} prompts, longest "
             f"{worst} tokens, budget {report['promptBudget']} — all fit")


def instrument_preflight(manifest, prompts_file, root, _log) -> None:
    """Instrument/exclusion coherence BEFORE the model loads (2026-08-06).

    The same gates run again inside ``_run_impl`` (the backstop for callers
    that reach it directly), but on a cluster the difference is the queue
    wait plus a multi-minute 27B load: a declaration that can never fire —
    an option-consuming instrument declared when no in-scope item carries
    ``options`` — silently produced zero records while the sampled arm
    burned the whole GPU allocation. Like ``artifact_preflight`` this is a
    pure file read, decidable now and unable to change mid-run, so a miss
    REFUSES rather than warns. Also logs the ladder-window advisories: a
    declared outOfRange keep-window whose bounds cannot bind the scale the
    items' options imply (min 0 / max 100 on a 1–7 ladder) is legal but
    inert, and inert-by-declaration is worth a loud line before compute."""
    from . import exclusions as _exclusions
    prompts = task_inputs.load_prompts(manifest, prompts_file, root)
    task_inputs.check_response_formats(manifest, prompts)
    _exclusions.preflight(manifest.raw, prompts)
    for warning in _exclusions.ladder_warnings(
            list(manifest.raw.get("exclusionRules") or []), prompts):
        _log(f"warning: {warning}")


def response_format_preflight(manifest, prompts_file, root) -> None:
    """The response-format/scope-drift gate BEFORE the model loads.

    The identical ``_check_response_formats`` still runs at run start inside
    ``_run_impl`` — this is the same rule moved ahead of the expensive part,
    not a second rule. Field incident 2026-08-06: a 4-shard Slurm run of a
    scope-drifted study staged and loaded gemma-3-27b-it (~2.5 minutes per
    shard, all four wasted) before the run-stage gate refused. The refusal
    was correct; the ordering was not. Like the artifact preflight — and
    unlike the token preflight — nothing here depends on the node's caches:
    the inputs are the manifest and the prompt file, read through the exact
    loader the run uses, so any error raised now (drifted scope, unreadable
    instrument, or a prompt file the loader refuses) is one the run would
    raise after the load. Refusing is strictly earlier, never new."""
    prompts = task_inputs.load_prompts(manifest, prompts_file, root)
    task_inputs.check_response_formats(manifest, prompts)


def panel_load_model(manifest, root, _log):
    """Which model the CLI path should LOAD for a panel, before acquiring it.

    Every turn runs on its seat's own base model; the manifest's ``modelID``
    is the default for seats that name none, and no turn otherwise consults
    it. Loading it regardless is at best wasted and at worst fatal — a
    manifest left at 27B while every seat said 4B spent the load fetching a
    27B nobody would use, and surfaced as a huggingface_hub traceback.

    Returns a manifest whose ``model_id`` is what to load. It NEVER refuses:
    seats that disagree are a legal mixed-model panel, and the API path serves
    them from the registry. On the CLI, where only one model can be resident,
    the majority seat model is loaded and ``run_scenario`` reports the first
    turn that needs another — a capability limit, surfaced where it bites.
    """
    from collections import Counter

    from . import multi_agent
    spath = manifest.multi_agent_scenario_path
    if not spath:
        return manifest
    try:
        if not os.path.isabs(spath):
            spath = os.path.join(paths.project_root() if root is None else root, spath)
        scenario, _ = multi_agent.load_scenario(spath)
    except (OSError, ValueError, KeyError):
        return manifest  # the runner's own load reports this properly
    # Weight by TURNS, not seats: the model most turns need is the one worth
    # holding resident.
    by_turn = Counter()
    seats = {a.id: a.base_model_id for a in scenario.agents}
    for turn in scenario.turns:
        model = seats.get(turn.speaker_agent_id) or manifest.model_id
        if model:
            by_turn[model] += 1
    if not by_turn:
        return manifest
    chosen = by_turn.most_common(1)[0][0]
    if len(by_turn) > 1:
        _log("mixed-model panel: seats name "
             + ", ".join(f"{m} ({n} turn{'s' if n != 1 else ''})"
                         for m, n in by_turn.most_common())
             + f". Loading {chosen}; the server's registry serves the rest, "
               "and each turn records the model it actually ran on.")
    if chosen != manifest.model_id:
        _log(f"panel runs on '{chosen}'; the study's declared default is "
             f"'{manifest.model_id}'. The declared value is kept for "
             "provenance — no turn consults it.")
        import copy
        manifest = copy.copy(manifest)
        manifest.model_id = chosen
    return manifest


def artifact_preflight(manifest, root, _log) -> None:
    """Refuse BEFORE the model loads when a steering-artifact reference does
    not resolve on this host — naming EVERY dangling reference, not the
    first.

    Unlike the token preflights, a miss here REFUSES rather than warns: the
    check is a file stat through the exact resolution generation uses
    (``model_variant.missing_artifacts`` → ``paths.resolve_artifact``), so a
    reference that does not resolve now cannot resolve mid-run. Observed
    live 2026-08-04: six app-promoted agents carried absolute Mac paths; the
    panel run allocated a GPU, loaded 27B weights to 51 GiB, and died on
    turn 2's `fear.json` — and the agentComparison twin burned its whole
    baseline arm before erroring every variant condition. Seconds versus
    GPU-hours.

    The per-condition error-record machinery in the run loop remains the
    backstop for failures existence cannot predict (unreadable tensors,
    foreign substrate, missing norms)."""
    from . import model_variant, multi_agent
    problems: list[str] = []

    def check(label: str, variant) -> None:
        for miss in model_variant.missing_artifacts(variant, root):
            problems.append(
                f"{label}: {miss['kind']} reference "
                f"'{miss['reference']}' does not resolve")

    for vc in manifest.variant_conditions:
        try:
            variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                       if vc.artifact else
                       model_variant.ModelVariant.from_file(
                           paths.resolve_artifact(vc.artifact_path, root)))
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"variant condition '{vc.name}': cannot read "
                            f"variant artifact ({exc})")
            continue
        check(f"variant condition '{vc.name}'", variant)

    spath = manifest.multi_agent_scenario_path
    if manifest.study_kind == "multiAgent" and spath:
        scenario = None
        try:
            if not os.path.isabs(spath):
                spath = os.path.join(
                    paths.project_root() if root is None else root, spath)
            scenario, _ = multi_agent.load_scenario(spath)
        except (OSError, ValueError, KeyError):
            pass  # the runner's own load reports this properly
        for agent in (scenario.agents if scenario is not None else []):
            if not agent.variant_artifact_path:
                continue
            try:
                variant = model_variant.ModelVariant.from_file(
                    paths.resolve_artifact(agent.variant_artifact_path))
            except (OSError, KeyError, ValueError) as exc:
                problems.append(
                    f"agent '{agent.name}': cannot read variant artifact "
                    f"'{agent.variant_artifact_path}' ({exc})")
                continue
            check(f"agent '{agent.name}'", variant)

    if problems:
        raise RuntimeError(
            f"artifact preflight: {len(problems)} steering-artifact "
            "reference(s) do not resolve on this host — refusing before the "
            "model loads:\n  - " + "\n  - ".join(problems)
            + "\nA reference recorded on another machine rebases "
            "automatically when the artifact exists under this workspace's "
            "runs/, experiments/, adapters/, or prompts/ — these do not, so "
            "the artifacts themselves are absent (or the references are "
            "wrong). Re-promote/re-import the agents or fix the paths, then "
            "resubmit.")


def span_policy_preflight(manifest, prompts_file, root, _log) -> None:
    """Refuse a span policy the study cannot run, BEFORE the model loads.

    A span policy (``positions: "span"``) is declared once for the whole study
    and resolved on every task prompt. The declared characters are checked
    here with ``prompt_span.declared_user_range``, which needs no tokenizer:
    a missing or ambiguous anchor in any item refuses the run with the
    resolver's own code and repair, naming every offending item. Scripted
    transcripts and panel seats are refused too, since phase 1 resolves
    single-prompt renders only. The token-level resolution runs again, with
    the loaded tokenizer, before the first generation
    (``policy_spans.run_preflight``). An agent that cannot be read is left
    to ``artifact_preflight`` and the run loop."""
    from . import model_variant, multi_agent, policy_spans

    def agent(reference, inline=None):
        try:
            return (model_variant.ModelVariant.from_dict(inline) if inline else
                    model_variant.ModelVariant.from_file(paths.resolve_artifact(reference, root)))
        except (OSError, KeyError, ValueError, TypeError):
            return None

    if manifest.study_kind == "multiAgent":
        spath = manifest.multi_agent_scenario_path
        if not spath:
            return
        try:
            if not os.path.isabs(spath):
                spath = os.path.join(paths.project_root() if root is None else root, spath)
            scenario, _ = multi_agent.load_scenario(spath)
        except (OSError, ValueError, KeyError):
            return  # the runner's own load reports this properly
        seats = [(a.name, agent(a.variant_artifact_path)) for a in scenario.agents if a.variant_artifact_path]
        policy_spans.refuse(policy_spans.panel_problems([(n, v) for n, v in seats if v is not None]),
                            when="in this panel")
        return
    arms = [(vc.name, agent(vc.artifact_path, vc.artifact)) for vc in manifest.variant_conditions]
    arms = [(name, variant) for name, variant in arms if variant is not None]
    if not any(policy_spans.policy_artifacts.span_policies(v.intervention_policies) for _, v in arms):
        return
    policy_spans.declared_preflight(arms, task_inputs.load_prompts(manifest, prompts_file, root))


def scenario_preflight_or_warn(manifest, root, _log) -> None:
    """Panel twin of ``token_preflight_or_warn`` (plan A4).

    Refuses only on a turn whose FLOOR — its own template plus shared
    materials, before any deliberation accumulates — already exceeds the
    budget, because that cannot fit whatever the model writes. A worst-case
    projection over the accumulating context warns instead: it charges every
    prior routed turn its full Max tokens, so it is a bound rather than a
    prediction, and blocking on it would refuse runs that fit.

    Same escape hatch as its twin: a preflight that cannot RUN never blocks a
    run — the in-generation context check remains the backstop."""
    from . import multi_agent, scenario_preflight
    spath = manifest.multi_agent_scenario_path
    if not spath:
        return
    try:
        if not os.path.isabs(spath):
            spath = os.path.join(paths.project_root() if root is None else root, spath)
        scenario, _ = multi_agent.load_scenario(spath)
        report = scenario_preflight.preflight(
            scenario, model_id=manifest.model_id, revision=manifest.model_revision)
    except scenario_preflight.token_preflight.PreflightError as exc:
        _log(f"token preflight unavailable ({exc}) — the in-generation "
             "context check remains the backstop")
        return
    except Exception as exc:  # noqa: BLE001 — see the docstring
        _log(f"token preflight unavailable ({type(exc).__name__}: {exc}) — "
             "the in-generation context check remains the backstop")
        return
    refusal = scenario_preflight.refusal(report, scenario)
    if refusal:
        raise RuntimeError(refusal)
    advisory = scenario_preflight.advisory(report, scenario)
    if advisory:
        _log(advisory)
    elif report.get("contextWindow"):
        worst = max((t["projectedPromptTokens"] for t in report["turns"]), default=0)
        _log(f"token preflight: {report['turnCount']} turns, worst-case "
             f"projection {worst} tokens on a {report['contextWindow']}-token "
             "window — all fit")
    _memory_preflight_or_stay_silent(report, manifest, _log)


def _memory_preflight_or_stay_silent(scenario_report, manifest, _log) -> None:
    """Peak-memory advisory beside the token one — MPS only, never blocking.

    The context window was never the binding constraint on a Mac: a panel
    passed token preflight at 14,731/131,072 and died at ~turn 21 on memory.
    Same escape hatch as every preflight here: an estimator that cannot run
    never blocks a run."""
    try:
        from ..steering import model_loader
        from . import memory_preflight

        device = model_loader.resolve_device(None)
        if not device.startswith("mps"):
            return  # CUDA/CPU: see memory_preflight's module docstring
        from transformers import AutoConfig
        kwargs = {"revision": manifest.model_revision} if manifest.model_revision else {}
        config = AutoConfig.from_pretrained(manifest.model_id, **kwargs)
        size = model_loader.snapshot_size_bytes(
            manifest.model_id, manifest.model_revision)
        model = memory_preflight.model_from_config(
            manifest.model_id, config,
            weights_gib=(size / memory_preflight.GIB) if size else None)
        from . import generate as generate_mod
        mem = memory_preflight.report(
            scenario_report, model, device=device,
            budget=memory_preflight.budget_gib(),
            # Which turns will chunk their prefill — without this the
            # estimate assumes single-pass and over-warns on exactly the
            # long turns chunking exists to save.
            prefill_chunk_for=lambda n: generate_mod.prefill_chunk_size(device, n))
        line = memory_preflight.advisory(mem) or memory_preflight.summary(mem)
        if line:
            _log(line)
    except Exception as exc:  # noqa: BLE001 — advisory-only, see docstring
        _log(f"memory preflight unavailable ({type(exc).__name__}: {exc}) — "
             "the run proceeds without a peak-memory estimate")


def resume_admission(run_directory, *, name, manifest, shard,
                      check_experiment_hash: bool = True,
                      reading=None) -> None:
    """Admit (or refuse) a supplied run directory for a standard study run.

    Pure file I/O on the run directory — ``require_resumable`` reads the
    directory's completion/park state, the epoch stamp is a JSON read, and the
    shard stamp is another. That is why ``run`` calls it BEFORE acquiring the
    model (open-issues §16): the observed failure order on the cluster was
    stage + 51 GiB weight load + device copy (~4 minutes of a GPU allocation),
    and only THEN the shard-identity ``ResumeError``. Four allocations died
    that way on 2026-08-18.

    ``check_experiment_hash=False`` defers only the epoch comparison — the one
    input that is not stable across the model load, because
    ``_pin_model_revision`` writes a resolved revision into an unpinned DRAFT
    manifest and changes its content hash. ``_run_impl`` runs the full gate
    again afterwards, so nothing is skipped, only ordered.
    """
    resume_mod.require_resumable(run_directory, verb="run")
    if check_experiment_hash:
        stamped = study_admission.stamped_experiment_hash(run_directory)
        if stamped is not None and stamped != manifest.content_hash():
            raise resume_mod.ResumeError(
                f"run directory {run_directory} was checkpointed by experiment "
                f"content hash {stamped[:12]}…, but '{name}' now hashes "
                f"{manifest.content_hash()[:12]}… — refusing to mix")
    # Shard identity is part of a partial's resume identity: resuming a
    # shard partial without its exact --shard k/K (or vice versa) would
    # silently generate the wrong record subset into the same file.
    shard_stamp = sharding_mod.read_shard_stamp(run_directory)
    if shard is None and shard_stamp is not None:
        raise resume_mod.ResumeError(
            f"run directory {run_directory} is shard "
            f"{shard_stamp.get('shardIndex')}/{shard_stamp.get('shardCount')} "
            "of a sharded run — resume it with the same --shard k/K")
    if shard is not None:
        if shard_stamp is None:
            raise resume_mod.ResumeError(
                f"run directory {run_directory} is not a shard partial "
                "(no shard.json) — it cannot be resumed under --shard")
        if (int(shard_stamp.get("shardIndex", -1)),
                int(shard_stamp.get("shardCount", -1))) \
                != (shard.index, shard.count):
            raise resume_mod.ResumeError(
                f"run directory {run_directory} was checkpointed as shard "
                f"{shard_stamp.get('shardIndex')}/"
                f"{shard_stamp.get('shardCount')}, not {shard.label} — "
                "refusing to mix shard ranges")
    seed_rule_admission(run_directory, name=name, manifest=manifest,
                        shard_stamp=shard_stamp)
    if reading is not None:
        # Before the model loads, only what the parked run's own record can
        # settle; the exact comparison over the rendered prompts runs again
        # once the tokenizer is in hand (``_run_impl``).
        prompt_reading_admission(run_directory, name=name, reading=reading,
                                 shard_stamp=shard_stamp, definite_only=True)


def seed_rule_admission(run_directory, *, name, manifest,
                        shard_stamp=None) -> None:
    """Refuse to resume a run whose sampled records follow a different seed
    rule than this engine applies to the same manifest.

    The content hash cannot catch it: the manifest is unchanged, and only the
    engine's reading of it moved. Before a declared derivedSHA256 was honored
    at one sample per item (issue #3), such a study ran each item once per
    listed seed under manifestSeeds. Resumed now, its finished cells would
    keep those records while the remaining cells got one derivedSHA256 record
    each, and a shard partial's ``expectedKeys`` would name sample indices
    the run no longer generates. Pure file I/O, so ``run`` refuses before
    acquiring the model, like the other resume gates."""
    expected = (sampling.DERIVED_SHA256 if sampling.manifest_derives_seeds(manifest)
                else sampling.MANIFEST_SEEDS)
    count = sampling.sample_count(manifest)
    found, sampled, widest_records, widest_plan = set(), 0, 0, 0
    path = os.path.join(run_directory, resume_mod.GENERATIONS_FILENAME)
    try:
        with open(path, encoding="utf-8") as handle:
            lines = list(handle)
    except OSError:
        lines = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a torn final line; the writer's own resume handles it
        if (not isinstance(record, dict) or "instrument" in record
                or "error" in record or "output" not in record):
            continue
        sampled += 1
        if record.get("seedPolicy"):
            found.add(record["seedPolicy"])
        if isinstance(record.get("sampleIndex"), int):
            widest_records = max(widest_records, record["sampleIndex"] + 1)
    for key in (shard_stamp or {}).get("expectedKeys") or []:
        if (isinstance(key, list) and len(key) == 5
                and key[4] == resume_mod.KIND_SAMPLED
                and isinstance(key[3], int)):
            widest_plan = max(widest_plan, key[3] + 1)
    if found <= {expected} and max(widest_records, widest_plan) <= count:
        return
    # What the parked directory holds, said from what was actually read: its
    # records, and (for a shard partial) the plan its shard.json fixed.
    if not sampled:
        held = "it has no sampled records yet"
    elif found:
        held = (f"its sampled records use seedPolicy "
                f"{', '.join(sorted(found))} with up to {widest_records} "
                "sample(s) per item")
    else:
        held = (f"its sampled records state no seed policy and reach up to "
                f"{widest_records} sample(s) per item")
    if widest_plan > count:
        held += (f", and its shard plan (shard.json) expects up to "
                 f"{widest_plan} sample(s) per item")
    if shard_stamp is not None:
        fresh = (f"steerlab-server study submit {name} --verb run --executor "
                 f"slurm --parallel {shard_stamp.get('shardCount')}  (a fresh "
                 "fan-out under the "
                 "current rule; or finish these partials on the engine release "
                 "that started them)")
    else:
        fresh = (f"steerlab-server experiment run {name}  (a fresh run under "
                 "the current rule, without --resume; or finish this run on "
                 "the engine release that started it)")
    raise resume_mod.ResumeRefusal(
        f"run directory {run_directory} was checkpointed under a different "
        f"seed rule: {held}, but this engine runs '{name}' with seedPolicy "
        f"{expected} and {count} sample(s) per item (a declared derivedSHA256 "
        "is now honored at one sample per item) — refusing to mix two seed "
        "rules in one run",
        gate=lifecycle_gates.MANIFEST_EPOCH, repair=fresh)


def study_prompt_reading(model, manifest, prompts, root, _log):
    """``(reading, counts, record)`` for a study run on the loaded model.

    ``reading`` is the rule the run reads its prompts by, resolved from the
    study's declaration, the submission's plan, the environment override,
    or the device default (``prompt_reading.resolve_prompt_reading``). A
    declaration the engine cannot read refuses the run before any
    generation. ``counts`` are the token counts of every task prompt under
    every arm's rendering (``prompt_reading.rendered_token_counts``), or None
    when they cannot be measured, which is said in the log. ``record`` is the
    run's ``notes.promptReading`` object."""
    from . import condition_execution, prompt_reading
    from . import generate as generate_mod
    device = generate_mod._model_device(model)
    try:
        reading = prompt_reading.for_manifest(manifest, device, log=_log)
    except prompt_reading.PromptReadingError as exc:
        raise lifecycle_gates.refusing(
            lifecycle_gates.STUDY_DECLARATION,
            f"the study's promptReading declaration cannot be read: {exc}",
            repair=exc.repair_action) from exc
    counts = None
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        try:
            settings = condition_execution.run_arm_render_settings(manifest, root)
            counts = prompt_reading.rendered_token_counts(
                tokenizer, manifest.model_id, prompts, settings)
        except Exception as exc:  # noqa: BLE001 - the record says what it lacks
            _log(f"prompt reading: the task prompts' token counts could not be "
                 f"measured ({type(exc).__name__}: {exc}); the run records its "
                 "rule without them")
    if counts is not None:
        longest = max((n for row in counts for n in row), default=0)
        chunked = sum(1 for row in counts if any(reading.chunk_size(n) for n in row))
        _log(f"prompt reading: {reading.describe()} (source {reading.source}); "
             f"{chunked} of {len(counts)} task prompt(s) read in chunks, the "
             f"longest {longest:,} tokens")
    else:
        _log(f"prompt reading: {reading.describe()} (source {reading.source})")
    return reading, counts, prompt_reading.record(reading, device=device,
                                                  counts=counts)


def _has_records(run_directory) -> bool:
    try:
        return os.path.getsize(
            os.path.join(run_directory, resume_mod.GENERATIONS_FILENAME)) > 0
    except OSError:
        return False


def _prompt_reading_repair(name, shard_stamp, run_directory=None) -> str:
    if shard_stamp is not None:
        return (f"steerlab-server study submit {name} --verb run --executor "
                f"slurm --parallel {shard_stamp.get('shardCount')}  (a fresh "
                "fan-out that reads every prompt one way; or finish these "
                "partials on the engine release that started them)")
    # Resuming through a submission is the repair that keeps the run: the
    # submission carries the parked run's recorded reading (or its release's
    # rule) as the plan. A rendered script replayed as it was (remote
    # resubmit) carries only what it was first given, so it is not offered.
    parked = (f"runs/{os.path.basename(os.path.normpath(run_directory))}"
              if run_directory else "<run-dir>")
    return (f"steerlab-server study submit {name} --verb run --resume {parked}  "
            "(resume through a submission, which reads the prompts as this run "
            "recorded; or finish it on the engine release that started it; or "
            f"start afresh with steerlab-server experiment run {name}, without "
            "--resume)")


def prompt_reading_admission(run_directory, *, name, reading, counts=None,
                             shard_stamp=None, log=None,
                             definite_only: bool = False) -> None:
    """Refuse to resume a run whose prompts were read differently from how
    this engine reads them now.

    Chunked and single-pass readings do not give token-identical texts, so a
    run that continued under a different reading would mix two readings
    whose records cannot be told apart. The comparison is of EFFECTIVE
    behaviour, not of the rule's text or its source: two rules agree when
    every prompt the study renders is read the same way under both, so a
    declared chunked 4096 above 8192 equals the CUDA default, and a study
    whose prompts are all at or below both thresholds reads identically
    under either. ``counts`` (this run's rendered prompt lengths, the same
    prompts the parked run read, since the experiment hash matched) make the
    comparison exact. Without them the parked run's own record decides from
    its longest prompt, where it can (``prompt_reading.agree_on_longest``).

    A run made before 0.9.20 records no reading and is taken as read by that
    release's rule for its device (``prompt_reading.run_directory_reading``).

    ``definite_only`` (the pre-model call in ``run``) refuses only what is
    certain without the prompts' lengths, and leaves the rest to the call
    after the model loads, which has them. Pure file I/O otherwise."""
    from . import prompt_reading
    then, entry, note = prompt_reading.run_directory_reading(run_directory)
    if then is None and entry is None and not _has_records(run_directory):
        # Made before runs recorded their reading, naming no device, and
        # nothing generated yet: no prompt has been read, so nothing can mix.
        return
    candidates = ([then] if then is not None else
                  [prompt_reading.historic_reading("mps"),
                   prompt_reading.historic_reading("cpu")])
    longest = (entry.get("longestPromptTokens")
               if isinstance(entry, dict)
               and isinstance(entry.get("longestPromptTokens"), int) else None)
    differing: list[int] = []
    verdict: bool | None = True
    for candidate in candidates:
        if counts is not None:
            found = prompt_reading.differing_counts(candidate, reading, counts)
            differing = differing or found
            agree = not found
        else:
            agree = prompt_reading.agree_on_longest(candidate, reading, longest)
        if agree is False:
            verdict = False
            break
        if agree is None:
            verdict = None
    if verdict is True:
        if note and log is not None and then is not None and then.historic \
                and counts is not None and any(
                    then.chunk_size(n) for row in counts for n in row):
            log(f"note: {os.path.basename(run_directory)}: {note}")
        return
    if verdict is None and definite_only:
        return
    held = (f"read its prompts as {then.describe()}" if then is not None
            else "cannot show how it read its prompts")
    if note:
        held += f" ({note})"
    now = f"{reading.describe()}, {prompt_reading.describe_source(reading)}"
    if differing:
        why = (prompt_reading.describe_difference(differing)
               + ", and chunked and single-pass readings do not give "
                 "token-identical texts")
    elif verdict is False:
        why = ("its longest prompt would be read differently, and chunked "
               "and single-pass readings do not give token-identical texts")
    else:
        why = ("neither its record nor this engine can show that every prompt "
               "would be read the same way under both")
    raise resume_mod.ResumeRefusal(
        f"run directory {run_directory} {held}, but this engine reads them as "
        f"{now}: {why} — refusing to mix two prompt readings in one run",
        gate=lifecycle_gates.MANIFEST_EPOCH,
        repair=_prompt_reading_repair(name, shard_stamp, run_directory))


def panel_resume_admission(run_directory, *, manifest, shard,
                            check_experiment_hash: bool = True) -> None:
    """The panel path's twin of :func:`resume_admission`.

    Same admissions, same order, different sentences (a panel refusal names
    the directory's basename and speaks of turns) — the wording is the
    contract a field operator reads, so the two stay separate rather than
    being merged into one parameterised message.
    """
    resume_mod.require_resumable(run_directory, verb="run")
    if check_experiment_hash:
        stamped = study_admission.stamped_experiment_hash(run_directory)
        if stamped is not None and stamped != manifest.content_hash():
            raise resume_mod.ResumeError(
                f"refusing to resume {os.path.basename(run_directory)}: it was "
                f"produced under experiment hash {stamped[:12]}… but the live "
                f"manifest is {manifest.content_hash()[:12]}… — appending turns "
                "across a manifest epoch would mix two experiments into one "
                "artifact. Duplicate the study to iterate.")
    existing_stamp = sharding_mod.read_shard_stamp(run_directory)
    if shard is None and existing_stamp is not None:
        raise resume_mod.ResumeError(
            f"{os.path.basename(run_directory)} is shard "
            f"{existing_stamp.get('shardIndex')}/"
            f"{existing_stamp.get('shardCount')} of a sharded panel run — "
            "resume it with the same --shard k/K")
    if shard is not None:
        if existing_stamp is None:
            # The case the first guard missed: an unsharded partial
            # resumed under --shard was accepted, restamped as a shard,
            # and "completed" holding only that shard's subset.
            raise resume_mod.ResumeError(
                f"{os.path.basename(run_directory)} is not a shard partial "
                "(no shard.json) — it cannot be resumed under --shard")
        if (int(existing_stamp.get("shardIndex", -1)),
                int(existing_stamp.get("shardCount", -1))) != (shard.index,
                                                               shard.count):
            raise resume_mod.ResumeError(
                f"{os.path.basename(run_directory)} was checkpointed as "
                f"shard {existing_stamp.get('shardIndex')}/"
                f"{existing_stamp.get('shardCount')}, not {shard.label} — "
                "refusing to mix shard ranges")
