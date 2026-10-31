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
    feature form (Swift twin ``ClusterCapabilities.supportsDeclaredSeedPolicy``)."""
    if not isinstance(capabilities, dict):
        return False
    remote = capabilities.get("remoteStudy")
    if isinstance(remote, dict) and remote.get(DECLARED_SEED_POLICY_CAPABILITY) is True:
        return True
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


def run_seed_policy(manifest) -> str:
    """The seed policy a generating run's ``config.json`` stamps: the
    OPERATIVE one, which its sampled records stamp too, not the declaration.
    A panel's turns derive their seeds whenever the temperature is above 0
    (``multi_agent``); every other study follows :func:`derives_seeds`. Swift
    twin: the ``seedPolicy`` argument ``ExperimentTasks`` passes to
    ``RunMetadata.write`` (``StudySampling.policy``, and the panel rule)."""
    if manifest.study_kind == "multiAgent":
        derived = bool(manifest.temperature) and manifest.temperature > 0
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


def unused_seed_settings(temperature, samples_per_item, seed_policy, seeds,
                         study_kind=None) -> list[str]:
    """Declared seed settings an ordinary study's run will not read — said at
    ``verify`` and at the sampling write, never refused (advisory code
    ``seedSettingNotUsed``). A panel (``multiAgent``) says nothing here: its
    turns take seeds by their own rule and never read the policy or the list.
    Byte-identical to the Swift twin ``StudySampling.unusedSeedSettings``; the
    shared fixture ``Tests/Fixtures/cross-engine/study-sampling-seeds.json``
    pins both."""
    if seed_policy != DERIVED_SHA256 or study_kind == "multiAgent":
        return []
    if not temperature or temperature <= 0:
        return ["seedPolicy derivedSHA256 has no effect at temperature 0: "
                "decoding is greedy and no seed is used. Seeds are derived "
                "per record only when the temperature is above 0."]
    count = len(seeds or [])
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
                                study_kind=manifest.study_kind)


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
