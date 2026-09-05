"""Shared study verification and source-epoch admission policy.

Retains existing draft/frozen and measurement-drift semantics. Callers own the
choice of verb and must retain the guard's returned provenance in their output.
"""
from __future__ import annotations
import os
from . import lifecycle_gates, run_epoch
from .manifest import Manifest


def _verify_or_warn(manifest: Manifest, root: str | None) -> None:
    """Warn on any violation; RAISE only for a frozen manifest.

    KNOWN CROSS-ENGINE DIVERGENCE (characterised 2026-07-26, deliberately
    unresolved). Swift's ``loadVerified`` throws unconditionally, so the same
    DRAFT — stimuli drifted from their pins, a variant artifact missing, an
    instrument declared in a configuration that cannot read it — refuses on
    the Mac and runs on the server.

    One of the two engines is wrong and it is not obvious which. Against
    changing this: a draft run is exploratory by construction (freeze is what
    makes a result citable), and refusing every draft whose pins have drifted
    would block iteration that is legitimately messy. For changing it: a
    drifted pin means the run's stamped provenance is false even when the
    numbers are only being eyeballed.

    Measured cost of switching the server to Swift's rule: 50 tests across 11
    files currently depend on the permissive behaviour — nearly all through
    fixture shortcuts (variant artifacts never written to disk, placeholder
    stimulus hashes, studies with neither concepts nor variants). That is
    mechanical to fix, but the BEHAVIOURAL change would also refuse live
    draft runs on the cluster that succeed today, so it is a policy decision
    for the researcher rather than a cleanup.
    """
    violations = manifest.verify(root)
    for v in violations:
        print(f"VERIFY WARNING: {v}")
    if violations and manifest.status == "frozen":
        # WP0 step 8: typed `pinDrift`. The prose is byte-identical to what
        # this site has always raised; the gate id and the runnable repair are
        # strictly additive, and `LifecycleError` IS a `RuntimeError` so every
        # existing catch still catches.
        raise lifecycle_gates.refusing(
            lifecycle_gates.PIN_DRIFT,
            f"experiment '{manifest.name}' is frozen but failed verification: {violations}",
            repair=(f"steerlab-server experiment verify {manifest.name} "
                    "(names every drifted pin) ; then restore the named files "
                    "to their pinned bytes — a frozen pin is never re-pinned: "
                    f"duplicate {manifest.name} on the Mac to change it"))



def _stamped_experiment_hash(run_directory: str) -> str | None:
    """The manifest-epoch stamp of a run directory. Delegates to the shared
    :mod:`run_epoch` reader so ``promote``'s epoch guard and this one cannot
    drift apart."""
    return run_epoch.stamped_experiment_hash(run_directory)



def _require_source_epoch(verb: str, name: str, manifest: Manifest,
                          run_dir: str, *, allow_unverified_epoch: bool
                          ) -> tuple[bool, str | None]:
    """Epoch guard for evaluate/analyze source runs (2026-07-13): a source run
    is eligible ONLY if its stamped experiment hash equals the LIVE manifest's
    content hash — otherwise a pre-edit draft run could be judged/analyzed
    under a frozen manifest and stamped with the frozen hash. Legacy runs with
    NO stamp refuse unless ``allow_unverified_epoch`` explicitly accepts them;
    the caller must then stamp ``epochUnverified: true`` into its output.

    Returns ``(unverified, measurement_drift)``. Every caller of this guard
    is a MEASUREMENT verb (evaluate/analyze/rescore-style), so drift confined
    to measurement-side fields (``run_epoch.MEASUREMENT_FIELDS`` — judges,
    evaluation, pipeline) is tolerated rather than refused: those fields
    cannot have affected a byte of the source run's generations, and refusing
    them forced a full GPU re-run to swap a judge whose model had died at its
    provider (2026-08-05). The caller must LOG the returned drift and stamp
    it (``measurementDrift``) into its output — tolerated is never silent.
    The rule itself lives in :mod:`run_epoch`, shared with ``promote`` (which
    stays strict: a judge swap changes what a judged sweep's evidence
    means)."""
    # A source run that is NOT THERE is a path fault, and it gets its own
    # typed refusal with its own repair (ledger 2026-08-21). Before this split
    # the missing directory fell through to "carries no experiment-hash stamp
    # … or pass allowUnverifiedEpoch" — the same sentence a genuinely legacy
    # run gets, on a run that was correctly stamped all along, with a repair
    # that invites the operator to switch the epoch firewall off. The two are
    # different failures and must read as different failures.
    unreadable = run_epoch.unreadable_source_refusal(verb, run_dir)
    if unreadable:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE, unreadable,
            repair=run_epoch.unreadable_source_repair(verb, name))
    refusal, unverified, drift = run_epoch.epoch_refusal(
        verb, name, manifest.content_hash(), run_dir,
        allow_unverified=allow_unverified_epoch, live_manifest=manifest,
        tolerate_measurement_drift=True,
        # The whole family reads the source run's RECORDS, so a run from the
        # other engine is refused here rather than measured into a result
        # this engine's pairing keys cannot have produced (WP0 dry run #2,
        # P0 — found on the Swift side, identical hole here).
        refuse_foreign_substrate=True)
    if refusal:
        # WP0 step 8: typed `manifestEpoch`. Same prose, same exit code; the
        # gate id is what tells an agent this is a REFUSAL against a healthy
        # system rather than the bare `RuntimeError` a genuine defect raises.
        #
        # A foreign run's repair is not "re-run" and is certainly not
        # ``--allow-unverified-epoch`` (which forgives a missing stamp, and
        # would leave this run just as unreadable) — it is the same verb on
        # the engine that wrote the records.
        #
        # THE ASSUMPTION, stated so a third engine cannot inherit it silently
        # (2026-08-18, WP0 residual (d)): there are exactly TWO substrates
        # (CLAUDE.md), so "foreign" means ``swift-mlx``, whose CLI is
        # ``steerlab-cli``; and every verb that reaches this guard —
        # ``analyze``, ``evaluate``, ``rescore-style`` — is TWINNED, i.e. it
        # exists on that CLI under the same spelling. Both halves are pinned
        # by ``test_foreign_substrate_repair_names_an_engine_that_has_the_verb``
        # (the Mac verb roster it checks against is the generated
        # ``swift-*`` region of ``docs/CLI-REFERENCE.md``). This is
        # deliberately a pinned assumption and not engine-capability
        # negotiation: a third substrate must come back here and decide what
        # it can compose, and the test is what will stop it from not
        # noticing.
        if run_epoch.foreign_substrate(run_dir) is not None:
            repair = (f"steerlab-cli experiment {verb} {name}  (on the engine "
                      "that produced the run; this engine reads its results, "
                      "it does not re-measure them)")
        else:
            repair = (f"steerlab-server experiment run {name}  (a run of the "
                      "CURRENT manifest), or re-read the older run with "
                      f"steerlab-server experiment {verb} {name} "
                      "--allow-unverified-epoch")
        raise lifecycle_gates.refusing(
            lifecycle_gates.MANIFEST_EPOCH, refusal, repair=repair)
    return unverified, drift



def _advise_implicit_case_family(fires: bool, run_directory: str | None, _log,
                                 *, write_file: bool) -> None:
    """The deprecated ``caseFamily == "sentencing"`` endpoint selection, said
    out loud wherever it actually fires (2026-08-18).

    Same shape as :func:`_advise_dependency_lock_drift` — logged at the verb's
    START, appended to the run directory's ``advisories.txt``, and NEVER a
    refusal. Compatibility is the point: a manifest that already depends on the
    trigger must keep producing the same numbers, so this changes nothing about
    the run except that the run says how its endpoint was chosen.

    ``fires`` is computed by the CALLER rather than re-derived here, because
    the sites do not share one predicate: the record-parse and analyze-rescue
    paths let a declared ``numericParser`` win
    (:func:`manifest.implicit_case_family_endpoint`), while the multi-agent
    panel-effects endpoint reads ``case_family`` alone. Deriving one predicate
    for all three would make the advisory lie at one of them.
    """
    if not fires:
        return
    from .manifest import IMPLICIT_CASE_FAMILY_ADVISORY
    _log(f"ADVISORY: {IMPLICIT_CASE_FAMILY_ADVISORY}")
    if not write_file or not run_directory:
        return
    # Append, like the lock-drift advisory: the cross-substrate advisory may
    # already own this file for this run.
    try:
        with open(os.path.join(run_directory, "advisories.txt"), "a",
                  encoding="utf-8") as handle:
            handle.write(IMPLICIT_CASE_FAMILY_ADVISORY + "\n")
    except OSError:  # the advisory must never sink a run
        pass
