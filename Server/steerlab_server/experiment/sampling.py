"""Per-record seed identity and scoped RNG state. Importable without torch.

The seed algorithm and RNG draw order are scientific contracts. Runtime RNG
state is touched only when entering a non-greedy generation scope.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
from threading import RLock


# torch's default generators are process-global. Forking alone does not
# serialize overlapping scopes on different worker threads.
_generation_lock = RLock()


def derive_seed(experiment_hash: str, condition: str, prompt_id: str,
                sample_index: int) -> int:
    """Deterministic per-record seed for samplesPerItem runs (seedPolicy
    'derivedSHA256'): any (condition, prompt, sampleIndex) cell reproduces its
    exact sample stream on this substrate without a seeds table.

    Policy (asserted by tests, do not change silently): the derivation
    includes CONDITION identity, so baseline and every saved-agent/steered
    condition draw DISTINCT random streams for the same (prompt,
    sampleIndex) — the design is paired at the prompt level, not a
    common-random-numbers design across conditions. Anything that PAIRS
    records across conditions must therefore join on (promptID,
    sampleIndex), never the seed (``paired_judge._pair_generations``)."""
    blob = f"{experiment_hash}|{condition}|{prompt_id}|{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2 ** 63)


DERIVED_SHA256 = "derivedSHA256"
MANIFEST_SEEDS = "manifestSeeds"


def derives_seeds(temperature, samples_per_item, seed_policy) -> bool:
    """The seed rule of an ordinary study's sampled records — the ONE place
    it is decided (Swift twin ``StudySampling.derivesSeeds``).

    Records derive per-record seeds (:func:`derive_seed`) when decoding is
    stochastic and the study either declares ``derivedSHA256`` or samples an
    item more than once; ``samplesPerItem`` then sets the per-item count, at
    any value including 1. Otherwise they enumerate the manifest's fixed seed
    list, one generation per listed seed — the behaviour of every undeclared
    and ``manifestSeeds`` study, unchanged. At temperature 0 decoding is
    greedy and no seed is drawn, so the declaration has nothing to decide.

    Before 2026-10-31 only ``samplesPerItem > 1`` selected derived seeds, so a
    study declaring ``derivedSHA256`` with one sample per item reused the
    fixed list for every item (public issue #3). Its records said so: they
    stamp ``seedPolicy`` per record."""
    if not temperature or temperature <= 0:
        return False
    return (samples_per_item or 1) > 1 or seed_policy == DERIVED_SHA256


#: The engine capability that says a declared derivedSHA256 decides the seed
#: rule at one sample per item (``remoteStudy.declaredSeedPolicy`` in the
#: capability snapshot, ``api/profile.py``). Both clients refuse to submit a
#: study whose seeds an engine without it would choose differently.
DECLARED_SEED_POLICY_CAPABILITY = "declaredSeedPolicy"


def declared_policy_changes_seeds(temperature, samples_per_item, seed_policy,
                                  study_kind=None) -> bool:
    """Whether an engine from before issue #3 would seed this study's
    sampled records differently: a declared derivedSHA256 at one sample per
    item above temperature 0. Panels never read the policy. Swift twin
    ``StudySampling.declaredPolicyChangesSeeds``."""
    return (study_kind != "multiAgent" and seed_policy == DERIVED_SHA256
            and (samples_per_item or 1) <= 1
            and bool(temperature) and temperature > 0)


def remote_honors_declared_seed_policy(capabilities) -> bool:
    """Whether a capability snapshot advertises
    :data:`DECLARED_SEED_POLICY_CAPABILITY`, in the nested or the dotted
    feature form. A nested value, when present, wins over the dotted one,
    as in the Swift twin ``ClusterCapabilities.supportsDeclaredSeedPolicy``."""
    if not isinstance(capabilities, dict):
        return False
    remote = capabilities.get("remoteStudy")
    nested = (remote.get(DECLARED_SEED_POLICY_CAPABILITY)
              if isinstance(remote, dict) else None)
    if nested is not None:
        return nested is True
    features = capabilities.get("features")
    return (isinstance(features, dict) and features.get(
        "remoteStudy." + DECLARED_SEED_POLICY_CAPABILITY) is True)


#: The verbs that generate a study's sampled records from its seed rule.
SEED_RULE_VERBS = ("run", "pipeline")
#: The client refusal's code when the engine predates the capability.
SEED_POLICY_UNSUPPORTED_CODE = "declaredSeedPolicyUnsupported"


def submission_needs_declared_seed_policy(manifest, verb) -> bool:
    """Whether submitting ``verb`` for this study needs an engine with
    :data:`DECLARED_SEED_POLICY_CAPABILITY`: a step that generates the
    study's sampled text, from a shape an older engine seeds differently.
    Swift twin: the guard in
    ``SubstrateRouting.declaredSeedPolicySubmissionRefusal``."""
    from . import execution_plan
    return (verb in SEED_RULE_VERBS
            and execution_plan.resolve(manifest.outcome_instruments)
            .generates_sampled_text
            and declared_policy_changes_seeds(
                manifest.temperature, manifest.samples_per_item,
                manifest.seed_policy, manifest.study_kind))


def outdated_engine_reason(where: str) -> str:
    """Why a client refuses to submit, in the same words on both clients
    (Swift twin ``SubstrateRouting.declaredSeedPolicySubmissionRefusal``)."""
    return (f"The engine at {where} predates the declared seed policy at one "
            "sample per item: its capabilities do not list "
            f"remoteStudy.{DECLARED_SEED_POLICY_CAPABILITY}, so it would run "
            "this study's derivedSHA256 at samplesPerItem 1 with the fixed "
            "seed list, the same seeds for every item. Nothing was submitted.")


def outdated_engine_repair() -> str:
    return ("Bring that engine up to date (for a cluster site, steerlab-cli "
            "cluster push --site <id>; for a local runner, restart it from "
            "this release), then submit again.")


#: The engine capability that says the temperature-0 rules (one seed, one
#: sample per item) refuse only a study whose run reads them, one that
#: generates sampled text (review round 3 of public issue #9). An engine
#: from before it refused every ordinary study at temperature 0 with several
#: seeds or several samples per item, readout-only ones included, after the
#: queue wait and the model load. ``remoteStudy.greedyRulesSampledTextOnly``
#: in the capability snapshot (``api/profile.py``); both clients refuse to
#: submit such a study to an engine without it.
GREEDY_RULES_CAPABILITY = "greedyRulesSampledTextOnly"
#: The client refusal's code when the engine predates the capability.
GREEDY_RULES_UNSUPPORTED_CODE = "greedyRulesUnsupported"


def remote_has_greedy_rules(capabilities) -> bool:
    """Whether a capability snapshot advertises
    :data:`GREEDY_RULES_CAPABILITY`, nested or dotted, the nested value
    winning (Swift twin ``ClusterCapabilities.supportsGreedyRulesSampledTextOnly``)."""
    if not isinstance(capabilities, dict):
        return False
    remote = capabilities.get("remoteStudy")
    nested = (remote.get(GREEDY_RULES_CAPABILITY)
              if isinstance(remote, dict) else None)
    if nested is not None:
        return nested is True
    features = capabilities.get("features")
    return (isinstance(features, dict) and features.get(
        "remoteStudy." + GREEDY_RULES_CAPABILITY) is True)


def submission_needs_greedy_rules(manifest, verb) -> bool:
    """Whether submitting ``verb`` needs an engine with
    :data:`GREEDY_RULES_CAPABILITY`: a step that runs an ordinary study at
    temperature 0 that lists several seeds or several samples per item.
    (One that generates sampled text is refused on the client anyway, by
    :func:`submission_greedy_seed_list_problem` or at verify; what this
    catches is a readout-only study, which this release runs and an older
    engine refused.) Swift twin:
    ``SubstrateRouting.greedyRulesSubmissionRefusal``."""
    if manifest.study_kind == "multiAgent" or not _submission_runs_the_study(
            manifest, verb):
        return False
    if manifest.temperature and manifest.temperature > 0:
        return False
    return len(manifest.seeds or []) > 1 or (manifest.samples_per_item or 1) > 1


def greedy_rules_engine_reason(where: str) -> str:
    """Why a client refuses to submit, in the same words on both clients
    (Swift twin ``SubstrateRouting.greedyRulesSubmissionRefusal``)."""
    return (f"The engine at {where} predates the fix for public issue #9 "
            "that lets an answer-token readout study run at temperature 0 "
            "with several seeds or several samples per item: its capabilities "
            f"do not list remoteStudy.{GREEDY_RULES_CAPABILITY}, so it would "
            "refuse this study after the queue wait and the model load. "
            "Nothing was submitted.")


def greedy_rules_submission_refusal(manifest, where: str) -> tuple[str, str, str]:
    """``(code, reason, repair)`` for a study
    :func:`submission_needs_greedy_rules` names, sent to an engine without
    :data:`GREEDY_RULES_CAPABILITY`, chosen by the study (round 4). One that
    generates sampled text is refused by every engine, this release's
    included, so it gets the rule's own sentence and repair: updating the
    engine would not help. A readout-only study is refused only by an older
    engine, so it gets the old-engine sentence. Either way it is refused
    before the queue. Swift twin:
    ``SubstrateRouting.greedyRulesSubmissionRefusal``."""
    from . import lifecycle_gates
    frozen = manifest.status != "draft"
    problem = manifest_greedy_seed_list_problem(manifest)
    if problem:
        return (lifecycle_gates.STUDY_DECLARATION, problem,
                greedy_seed_list_repair(manifest.name, frozen=frozen))
    problem = greedy_samples_problem(
        generating_temperature(manifest), manifest.samples_per_item,
        manifest.study_kind, manifest.outcome_instruments)
    if problem:
        return (lifecycle_gates.STUDY_DECLARATION, problem,
                greedy_samples_repair(manifest.name, frozen=frozen))
    return (GREEDY_RULES_UNSUPPORTED_CODE, greedy_rules_engine_reason(where),
            greedy_rules_engine_repair(manifest.name, frozen=frozen))


def greedy_rules_engine_repair(name: str, *, frozen: bool = False) -> str:
    """Update the engine, or give the study one seed and one sample per
    item, on a duplicate when it is frozen."""
    from . import command_vocabulary as vocabulary
    target = f"{name}-v2" if frozen else name
    one = vocabulary.protocol_field(
        target, f"experiment set-sampling {target} --seeds <n> "
        "--samples-per-item 1", "seeds='[<n>]'", "samplesPerItem=1")
    steps = ((f"experiment duplicate {name} {target}", one) if frozen
             else (one,))
    return ("Bring that engine up to date (for a cluster site, steerlab-cli "
            "cluster push --site <id>, then restart its controller; for a "
            "local runner, restart it from this release), or set one seed "
            "and one sample per item: " + vocabulary.authoring(*steps))


def manifest_derives_seeds(manifest, temperature=None) -> bool:
    """:func:`derives_seeds` for a loaded manifest, at ``temperature`` when
    given (a condition's effective temperature) and the study's otherwise."""
    return derives_seeds(
        manifest.temperature if temperature is None else temperature,
        manifest.samples_per_item, manifest.seed_policy)


def sample_count(manifest) -> int:
    """Generations per (condition, prompt) under the seed rule above: the
    declared ``samplesPerItem`` when seeds are derived, else one per listed
    seed."""
    if manifest_derives_seeds(manifest):
        return max(1, manifest.samples_per_item)
    return max(1, len(manifest.seeds))


def generating_temperature(manifest, root: str | None = None) -> float:
    """The highest temperature any generation of this study runs at: the
    study's own, or, for a panel, the highest of its seats' effective
    temperatures (a seat's own declaration, else the study's; issue #15).
    Every rule that asks "is this run greedy?" reads it, so a greedy study
    with one warm seat is not mistaken for a greedy run. Swift twin:
    ``StudySampling.generatingTemperature``."""
    temperature = float(getattr(manifest, "temperature", 0.0) or 0.0)
    if getattr(manifest, "study_kind", None) != "multiAgent":
        return temperature
    from . import multi_agent
    seats = multi_agent.panel_seat_temperatures(manifest, root)
    return max(seats) if seats else temperature


def run_seed_policy(manifest) -> str:
    """The seed policy a generating run's ``config.json`` stamps: the
    OPERATIVE one, not the declaration. For an ordinary study it is the
    policy its sampled records stamp too. A panel's turns derive their seeds
    whenever the temperature is above 0 (``multi_agent``); at temperature 0
    its turns stamp ``greedy``, a word the closed config vocabulary does not
    have, so the config says ``manifestSeeds``: no seed is drawn either way
    (the results export reads the two as agreeing). Swift twin: the
    ``seedPolicy`` argument ``ExperimentTasks`` passes to
    ``RunMetadata.write`` (``StudySampling.policy``, and the panel rule)."""
    if manifest.study_kind == "multiAgent":
        # A panel seeds the turns of every seat above temperature 0, so a
        # greedy study with a warm seat derives seeds for that seat's turns.
        derived = generating_temperature(manifest) > 0
    else:
        derived = manifest_derives_seeds(manifest)
    return DERIVED_SHA256 if derived else MANIFEST_SEEDS


def record_seeds(manifest, *, experiment_hash: str, condition: str,
                 prompt_id: str, temperature=None) -> tuple[str, list]:
    """The seed policy in effect and the ``(sampleIndex, seed)`` pairs one
    (condition, prompt) cell generates, in emission order."""
    if manifest_derives_seeds(manifest, temperature):
        return DERIVED_SHA256, [
            (index, derive_seed(experiment_hash, condition, prompt_id, index))
            for index in range(max(1, manifest.samples_per_item))]
    return MANIFEST_SEEDS, list(enumerate(manifest.seeds))


#: How :func:`greedy_seed_list_problem`'s sentence opens, on both engines.
#: ``verification_refusal`` matches it to name the repair verb, and it says
#: none of the words a drift sentence uses, so it is never read as drift.
GREEDY_SEED_LIST_PREFIX = ("at temperature 0 decoding is greedy, so every "
                           "seed gives the same response")


def greedy_seed_list_problem(temperature, seeds, study_kind=None,
                             instruments=None) -> str | None:
    """Why an ordinary study at temperature 0 that generates sampled text
    must list exactly one seed (public issue #9), or None.

    Under the fixed seed list each item's sampled text runs once per listed
    seed. At temperature 0 decoding is greedy and draws no random number, so
    every seed gives the same response: N seeds would write N identical
    records per item, and analysis would count them as N observations. Every
    arm runs at the study's temperature (``condition_execution.
    require_manifest_sampling_policy``), so the rule is the study's.

    Only sampled text loops over the seeds. An answer-token readout reads
    each item once whatever the seeds (``condition_execution.
    execute_condition``), so a study that measures readouts alone
    (``instruments`` resolving to no sampled text, ``execution_plan``) is
    never refused here; its unused list is an advisory
    (:func:`unused_seed_settings`). ``instruments`` absent means the default,
    sampled text. A panel (``multiAgent``) never reads the list: its turns
    take their seeds by their own rule and stamp ``greedy`` at temperature 0
    (``multi_agent``).

    One sentence everywhere: a ``verify`` violation on a draft (so ``freeze``
    refuses), the engine's submission preflight, both engines' run start, and
    both clients before they submit. Swift twin
    ``StudySampling.greedySeedListProblem``; the shared fixture
    ``Tests/Fixtures/cross-engine/study-sampling-seeds.json`` pins both."""
    if study_kind == "multiAgent" or (temperature and temperature > 0):
        return None
    count = len(seeds or [])
    if count <= 1:
        return None
    from . import execution_plan
    if not execution_plan.resolve(instruments).generates_sampled_text:
        return None
    return (f"{GREEDY_SEED_LIST_PREFIX}, and the study's {count} seeds would "
            f"write {count} identical records per item. Set one seed, or "
            "raise the temperature above 0 to sample")


def manifest_greedy_seed_list_problem(manifest) -> str | None:
    """:func:`greedy_seed_list_problem` for a loaded manifest."""
    return greedy_seed_list_problem(manifest.temperature, manifest.seeds,
                                    study_kind=manifest.study_kind,
                                    instruments=manifest.outcome_instruments)


def _submission_runs_the_study(manifest, verb) -> bool:
    """Whether a submission of ``verb`` runs the study: ``run``, or a
    ``pipeline`` whose declared stages include it. A pipeline with no block
    (or one that does not resolve) is left to the engine, which refuses the
    block itself."""
    if verb == "run":
        return True
    if verb != "pipeline":
        return False
    from . import pipeline_spec
    block = manifest.raw.get("pipeline")
    try:
        stages = (pipeline_spec.resolve_pipeline(block).stages
                  if block is not None else ())
    except ValueError:
        stages = ()
    return "run" in stages


def submission_greedy_seed_list_problem(manifest, verb) -> str | None:
    """:func:`greedy_seed_list_problem` for a submission of ``verb``: one
    that runs the study (``run``, or a ``pipeline`` whose declared stages
    include it). Both clients refuse with it before they submit, as they do
    for :func:`submission_needs_declared_seed_policy`, so an engine from
    before the rule is never reached. A pipeline with no block (or one that
    does not resolve) is left to the engine, which refuses the block itself.
    Swift twin: ``SubstrateRouting.greedySeedListSubmissionRefusal``."""
    if not _submission_runs_the_study(manifest, verb):
        return None
    return manifest_greedy_seed_list_problem(manifest)


def frozen_greedy_seed_list_advisory(manifest) -> str | None:
    """What ``verify`` says of a study frozen under an older release that the
    seed rule refuses, or None: it still verifies (its runs stay readable),
    so the reader learns here, not at submission, that running it needs a
    duplicate with one seed. Advisory code ``refusedWhenRun``. A study whose
    declared chain has no run stage is told too, worded for a run submitted
    on its own (``--verb run``), which is refused the same way (round 4).
    Swift twin ``StudySampling.frozenGreedySeedListAdvisory``; the shared
    fixture pins its text as the Mac command line spells the repair, and
    each client spells the repair for itself."""
    if manifest.status == "draft":
        return None
    problem = manifest_greedy_seed_list_problem(manifest)
    if problem is None:
        return None
    when = "when its run is submitted or started"
    block = manifest.raw.get("pipeline")
    if block is not None:
        from . import pipeline_spec
        try:
            stages = pipeline_spec.resolve_pipeline(block).stages
        except ValueError:
            stages = ()
        if "run" not in stages:
            when = ("when a run of it is submitted or started (its declared "
                    "pipeline does not run it)")
    return (f"this study will be refused {when}, because {problem}. To run "
            "it: " + greedy_seed_list_repair(manifest.name, frozen=True))


def greedy_seed_list_repair(name: str, *, frozen: bool = False) -> str:
    """The repair for :func:`greedy_seed_list_problem`, as commands the reader
    can run (``command_vocabulary``): one seed, through the Mac command line's
    ``set-sampling --seeds`` or the cross-platform client's ``set-protocol``.
    A frozen study is never edited, so it is duplicated first. Swift twin
    ``StudySampling.greedySeedListRepair``."""
    from . import command_vocabulary as vocabulary
    target = f"{name}-v2" if frozen else name
    one_seed = vocabulary.protocol_field(
        target, f"experiment set-sampling {target} --seeds <n>",
        "seeds='[<n>]'")
    steps = ((f"experiment duplicate {name} {target}", one_seed) if frozen
             else (one_seed,))
    return (vocabulary.authoring(*steps)
            + "  (one seed; or raise the temperature above 0 to sample)")


def _samples_are_read(study_kind, instruments) -> bool:
    """Whether ``samplesPerItem`` decides anything a run writes: a panel's
    replicates always do (``panel_workflow``), and an ordinary study's
    sampled text does. An answer-token readout reads each item once
    whatever the count (``condition_execution.execute_condition``), so a
    readout-only study's ``samplesPerItem`` is inert (the run's config.json
    still stamps the declared value, as provenance;
    ``execution_plan.inert_sampling_advisory`` says it is inert)."""
    if study_kind == "multiAgent":
        return True
    from . import execution_plan
    return execution_plan.resolve(instruments).generates_sampled_text


def greedy_samples_problem(temperature, samples_per_item, study_kind=None,
                           instruments=None) -> str | None:
    """``verify``'s sentence for several samples per item at temperature 0,
    or None. Also the run's: the submission preflight and run start refuse
    with it (``study_admission.settings_refusal``). Never for a study whose
    run does not read the count (:func:`_samples_are_read`; review round 3
    of public issue #9). Swift twin: ``StudySampling.samplesPerItemProblems``."""
    if ((samples_per_item or 1) > 1 and (not temperature or temperature <= 0)
            and _samples_are_read(study_kind, instruments)):
        return ("samplesPerItem > 1 requires temperature > 0 "
                "(greedy decoding makes every sample identical)")
    return None


def samples_per_item_problems(temperature, samples_per_item, seed_policy,
                              study_kind=None, instruments=None) -> list[str]:
    """``verify``'s joint rules for several samples per item, in order: a
    temperature above 0 (:func:`greedy_samples_problem`) and the derived
    seed policy. Neither applies to a study whose run does not read the
    count, which gets ``execution_plan.inert_sampling_advisory`` instead.
    Byte-identical to the Swift twin ``StudySampling.samplesPerItemProblems``;
    the shared fixture ``Tests/Fixtures/cross-engine/study-sampling-seeds.json``
    pins both."""
    if (samples_per_item or 1) <= 1 or not _samples_are_read(study_kind,
                                                             instruments):
        return []
    problems = []
    greedy = greedy_samples_problem(temperature, samples_per_item, study_kind,
                                    instruments)
    if greedy:
        problems.append(greedy)
    if seed_policy != DERIVED_SHA256:
        problems.append(
            "samplesPerItem > 1 requires seedPolicy 'derivedSHA256' "
            "(per-record seeds derived from condition/prompt/sampleIndex)")
    return problems


def greedy_samples_repair(name: str, *, frozen: bool = False) -> str:
    """The repair for :func:`greedy_samples_problem`: sample once per item, or
    raise the temperature. A frozen study is duplicated first."""
    from . import command_vocabulary as vocabulary
    target = f"{name}-v2" if frozen else name
    once = vocabulary.protocol_field(
        target, f"experiment set-sampling {target} --samples-per-item 1",
        "samplesPerItem=1")
    steps = ((f"experiment duplicate {name} {target}", once) if frozen
             else (once,))
    return (vocabulary.authoring(*steps)
            + "  (one sample per item; or raise the temperature above 0 to "
            "sample several)")


def unused_seed_settings(temperature, samples_per_item, seed_policy, seeds,
                         study_kind=None, instruments=None) -> list[str]:
    """Declared seed settings an ordinary study's run will not read — said at
    ``verify`` and at the sampling write, never refused (advisory code
    ``seedSettingNotUsed``). A panel (``multiAgent``) says nothing here: its
    turns take seeds by their own rule and never read the policy or the list.
    A temperature-0 study with several seeds that generates sampled text says
    nothing here either: it is refused (:func:`greedy_seed_list_problem`),
    and a setting a study is refused for is never called one with "no
    effect". A study that measures answer-token readouts alone is told its
    list of several seeds is not used: a readout reads each item once, at any
    temperature. Byte-identical to the Swift twin
    ``StudySampling.unusedSeedSettings``; the shared fixture
    ``Tests/Fixtures/cross-engine/study-sampling-seeds.json`` pins both."""
    if study_kind == "multiAgent":
        return []
    count = len(seeds or [])
    from . import execution_plan
    plan = execution_plan.resolve(instruments)
    if not plan.generates_sampled_text and plan.scores_directly and count > 1:
        return [f"the seed list ({count} seeds) is not used: this study "
                "records answer-token readouts only, and a readout reads each "
                "item once and draws no seed at any temperature. To generate "
                "responses that use the seeds, add sampledText to "
                "outcomeInstruments."]
    if seed_policy != DERIVED_SHA256:
        return []
    if not temperature or temperature <= 0:
        if greedy_seed_list_problem(temperature, seeds, study_kind,
                                    instruments):
            return []
        return ["seedPolicy derivedSHA256 has no effect at temperature 0: "
                "decoding is greedy and no seed is used. Seeds are derived "
                "per record only when the temperature is above 0."]
    if not plan.generates_sampled_text:
        return []
    if count > 1:
        runs = max(1, samples_per_item or 1)
        return [f"the seed list ({count} seeds) is not used under seedPolicy "
                f"derivedSHA256: each item runs samplesPerItem ({runs}) "
                f"time{'' if runs == 1 else 's'}, each with its own seed "
                "derived from the study hash, condition, prompt ID, and "
                "sample index. To generate more responses per item, raise "
                "samplesPerItem."]
    return []


def manifest_unused_seed_settings(manifest) -> list[str]:
    """:func:`unused_seed_settings` for a loaded manifest."""
    return unused_seed_settings(manifest.temperature, manifest.samples_per_item,
                                manifest.seed_policy, manifest.seeds,
                                study_kind=manifest.study_kind,
                                instruments=manifest.outcome_instruments)


@contextmanager
def seeded_generation(temperature: float, seed: int):
    """Generation-local per-record seeding (RNG isolation, 2026-07-13).

    ``torch.manual_seed`` mutates PROCESS-GLOBAL RNG state, so two concurrent
    seeded studies interleaving records would corrupt each other's sample
    streams. Serialize overlapping scoped records and save the global RNG
    state (CPU, every visible CUDA device, and MPS) for the duration of one
    record's generation and seed INSIDE the scope: the
    record draws exactly the stream its seed names, and the global state is
    restored on exit — interleaved seeded records draw identically to serial
    ones. Unrelated code that bypasses this scope must not mutate torch's
    generators concurrently. Greedy records (``temperature <= 0``) never touch the RNG at all
    (preserves the resume byte-equality contract unchanged).
    """
    if not temperature or temperature <= 0:
        yield
        return
    import torch

    with _generation_lock:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        # fork_rng defaults to CUDA and does not restore MPS, although
        # manual_seed seeds MPS too. Restore it explicitly on every exit.
        mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
