"""The epoch guard must name the SUBSTRATE when the run came from the other
engine.

Epoch hashes are per-engine by design (the two engines canonicalize the
manifest differently), so a foreign-substrate run always mismatches. Blaming
"a different manifest epoch" sends the researcher hunting for a manifest
change that never happened — observed live on the Swift side 2026-07-26 with
an imported sweep, and symmetric here for a swift-mlx run imported into a
server workspace.

Twin of Swift's `RunEpoch` coverage in `CrossEngineLifecycleTests`.
"""

import json
import os

from steerlab_server.experiment.run_epoch import epoch_refusal, foreign_substrate


def _run(tmp_path, substrate, *, stamp="aaaa"):
    directory = tmp_path / "20260726T000000000-exp-s-sweep"
    directory.mkdir()
    config = {"runType": "sweep", "experimentHash": stamp}
    if substrate is not None:
        config["substrate"] = substrate
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def test_foreign_substrate_is_named(tmp_path):
    directory = _run(tmp_path, "swift-mlx")
    assert foreign_substrate(directory) == "swift-mlx"
    refusal, unverified, _ = epoch_refusal(
        "promote", "s", "bbbb", directory, allow_unverified=False)
    assert refusal is not None
    assert "swift-mlx" in refusal
    assert "do not cross substrates" in refusal
    # The misleading explanation must not appear for a foreign run.
    assert "different manifest epoch" not in refusal
    assert unverified is False


def test_same_substrate_still_reports_the_epoch_mismatch(tmp_path):
    directory = _run(tmp_path, "python-hf-transformers")
    assert foreign_substrate(directory) is None
    refusal, _, _ = epoch_refusal(
        "promote", "s", "bbbb", directory, allow_unverified=False)
    assert refusal is not None
    assert "different manifest epoch" in refusal


def test_legacy_run_without_substrate_key_is_not_foreign(tmp_path):
    directory = _run(tmp_path, None)
    assert foreign_substrate(directory) is None
    refusal, _, _ = epoch_refusal(
        "promote", "s", "bbbb", directory, allow_unverified=False)
    assert "different manifest epoch" in refusal


def test_matching_epoch_is_accepted_regardless(tmp_path):
    directory = _run(tmp_path, "python-hf-transformers")
    assert epoch_refusal(
        "promote", "s", "aaaa", directory,
        allow_unverified=False) == (None, False, None)


def _manifest(**overrides):
    from steerlab_server.experiment.manifest import Manifest
    raw = {"name": "s", "modelID": "org/m", "concepts": [], "conditions": []}
    raw.update(overrides)
    return Manifest.from_dict(raw)


def _run_with_snapshot(tmp_path, substrate, snapshot_manifest, name="r"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(
        {"runType": "sweep", "substrate": substrate,
         "experimentHash": "engine-local-hash-that-never-matches"}),
        encoding="utf-8")
    (directory / "experiment.json").write_text(
        json.dumps(snapshot_manifest.raw), encoding="utf-8")
    return str(directory)


def test_foreign_run_whose_snapshot_matches_is_ACCEPTED(tmp_path):
    """The core fix: a foreign stamp is not evidence of a different epoch, it
    is no evidence at all. The snapshot IS comparable — re-hashed here, it
    proves the epochs agree, and the run is eligible."""
    live = _manifest()
    directory = _run_with_snapshot(tmp_path, "swift-mlx", _manifest())
    refusal, unverified, _ = epoch_refusal(
        "promote", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live)
    assert refusal is None, refusal
    assert unverified is False


def test_foreign_run_whose_snapshot_differs_names_the_fields(tmp_path):
    live = _manifest(modelRevision="a" * 40)
    directory = _run_with_snapshot(tmp_path, "swift-mlx", _manifest())
    refusal, _, _ = epoch_refusal(
        "promote", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live)
    assert refusal is not None
    assert "modelRevision" in refusal
    assert "a" * 40 in refusal
    # The unactionable framing is gone.
    assert "epoch stamps do not cross substrates" not in refusal


def test_volatile_freeze_stamps_are_not_reported_as_changes(tmp_path):
    """The diff must ignore exactly what the hash ignores. Reporting `status`
    or `frozenAt` would name a field the gate never checked."""
    from steerlab_server.experiment import manifest_diff
    a = _manifest(status="draft")
    b = _manifest(status="frozen", frozenAt="2026-07-26T00:00:00Z",
                  freezeHash="abc", gitCommit="def")
    assert manifest_diff.differences(a, b) == []


# --- measurement-side drift tolerance (2026-08-05) ------------------------------
# A pinned judge model died at its provider between run and evaluate; swapping
# the judge changed the manifest hash and the guard demanded a full GPU re-run
# to judge generations the edit could not have affected. Drift confined to
# MEASUREMENT_FIELDS (judges, evaluation, pipeline) is tolerable for
# measurement verbs — loudly, and stamped — while generation-side drift and
# the promote/jlens guards stay strict.

_ENGINE = "python-hf-transformers"


def _judged(judge_model, **overrides):
    return _manifest(
        judges=[{"name": "j1", "kind": "openrouter", "model": judge_model,
                 "provider": "deepinfra"}],
        pipeline={"stages": ["run", "evaluate", "analyze"]},
        **overrides)


def test_judge_swap_is_tolerated_only_when_asked_and_reports_the_drift(tmp_path):
    live = _judged("google/gemini-3.6-flash")
    directory = _run_with_snapshot(
        tmp_path, _ENGINE, _judged("deepseek/dead-model"))
    # Strict (the default, and what promote uses): refuse.
    refusal, _, drift = epoch_refusal(
        "promote", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live)
    assert refusal is not None
    assert drift is None
    # Measurement verbs: tolerated, and the drift text names the fields.
    refusal, unverified, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert unverified is False
    assert drift is not None and "judges" in drift


def test_pipeline_block_removal_is_tolerated(tmp_path):
    live = _manifest(judges=[{"name": "j1", "kind": "openrouter",
                              "model": "m/x", "provider": "deepinfra"}])
    directory = _run_with_snapshot(tmp_path, _ENGINE, _judged("m/x"))
    refusal, _, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert drift is not None and "pipeline" in drift


def test_a_renamed_duplicate_is_tolerated_on_measurement_verbs(tmp_path):
    # The duplicate-never-edit path (2026-08-28): duplicate a study, pin a
    # new rubric and panel onto the DUPLICATE, evaluate against the
    # original's run. The duplicate's name necessarily differs — and a name
    # cannot have affected a byte of the source run's generations, so the
    # tolerance must not refuse on the one field duplication itself changes.
    live = _judged("m", name="s-calibration",
                   judgeRubricFile="prompts/rubrics/coding-cf-v1.md",
                   judgeRubricHash="a" * 64)
    directory = _run_with_snapshot(
        tmp_path, _ENGINE,
        _judged("m", judgeRubricFile="prompts/rubrics/paired-cf-v1.md",
                judgeRubricHash="b" * 64))
    refusal, unverified, drift = epoch_refusal(
        "evaluate", "s-calibration", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert unverified is False
    # The rename is SAID in the tolerated-drift stamp, not slipped through.
    assert drift is not None and "name" in drift
    # promote still refuses a renamed manifest: identity is not tolerated
    # where evidence meaning is at stake.
    refusal, _, drift = epoch_refusal(
        "promote", "s-calibration", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live)
    assert refusal is not None and drift is None


def test_a_rename_alone_is_tolerated_with_no_measurement_edit(tmp_path):
    # Rename with NOTHING else changed: still the duplicate path (the new
    # rubric may be pinned in a later write), still tolerable.
    live = _manifest(name="s2")
    directory = _run_with_snapshot(tmp_path, _ENGINE, _manifest())
    refusal, _, drift = epoch_refusal(
        "evaluate", "s2", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert drift is not None and "name" in drift


def test_generation_side_drift_still_refuses_despite_tolerance(tmp_path):
    # A judge swap RIDING ALONG with a generation-side edit must not slip
    # through: the tolerance is field-scoped, not a bypass.
    live = _judged("google/gemini-3.6-flash", temperature=0.9)
    directory = _run_with_snapshot(
        tmp_path, _ENGINE, _judged("deepseek/dead-model"))
    refusal, _, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is not None
    assert drift is None
    assert "temperature" in refusal


def test_foreign_run_gets_the_same_measurement_tolerance(tmp_path):
    live = _judged("google/gemini-3.6-flash")
    directory = _run_with_snapshot(
        tmp_path, "swift-mlx", _judged("deepseek/dead-model"))
    refusal, _, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert drift is not None and "judges" in drift


def test_run_without_snapshot_cannot_prove_measurement_only_drift(tmp_path):
    # No snapshot = no way to bound the drift to measurement fields: refuse.
    directory = _run(tmp_path, _ENGINE, stamp="cccc")
    live = _judged("google/gemini-3.6-flash")
    refusal, _, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is not None
    assert drift is None


def test_rubric_swap_is_measurement_drift(tmp_path):
    # The motivating workflow (2026-08-05 review): re-pin a study from the
    # withdrawn paired K&Z rubric to the perResponseCoding rubric and
    # re-evaluate the SAME healthy source run. The rubric is text handed to
    # a judge at evaluate time — it cannot have touched a generation.
    live = _judged("m", judgeRubricFile="prompts/rubrics/coding-cf-v1.md",
                   judgeRubricHash="a" * 64)
    directory = _run_with_snapshot(
        tmp_path, _ENGINE,
        _judged("m", judgeRubricFile="prompts/rubrics/paired-cf-v1.md",
                judgeRubricHash="b" * 64))
    refusal, unverified, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert unverified is False
    assert drift is not None and "judgeRubric" in drift
    # Strict verbs still refuse the same swap.
    refusal, _, drift = epoch_refusal(
        "promote", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live)
    assert refusal is not None and drift is None


def test_human_validation_pin_swap_is_measurement_drift(tmp_path):
    live = _judged("m", humanValidation={"path": "prompts/h.jsonl",
                                         "hash": "c" * 64})
    directory = _run_with_snapshot(tmp_path, _ENGINE, _judged("m"))
    refusal, unverified, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is None, refusal
    assert drift is not None and "humanValidation" in drift


def test_clobbered_revision_pin_is_same_epoch(tmp_path):
    """The run pinned the resolved revision (snapshot carries it); a bundle
    re-import then overwrote the live manifest with the unpinned copy. Same
    manifest, one write behind its own machinery — accepted, strict verbs
    included (2026-08-05: a completed 2-hour pipeline refused its own
    continuation over this)."""
    live = _manifest()                                  # no modelRevision
    directory = _run_with_snapshot(
        tmp_path, _ENGINE, _manifest(modelRevision="a" * 40))
    refusal, unverified, drift = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live)
    assert refusal is None, refusal
    assert (unverified, drift) == (False, None)


def test_a_genuinely_different_revision_still_refuses(tmp_path):
    live = _manifest(modelRevision="b" * 40)
    directory = _run_with_snapshot(
        tmp_path, _ENGINE, _manifest(modelRevision="a" * 40))
    refusal, _, _ = epoch_refusal(
        "evaluate", "s", live.content_hash(), directory,
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True)
    assert refusal is not None
    assert "modelRevision" in refusal


# --- the foreign-substrate REFUSAL for measurement verbs (WP0 dry run #2) ----
#
# The snapshot rescue above answers "do the SETTINGS match?". For a
# measurement verb that is the wrong question: what fails is READING the
# records, whose schema and pairing keys are per-engine (this engine pairs on
# (promptID, sampleIndex), Swift on (seed, promptID)). Dry run #2 measured the
# consequence on the Swift side — analyze on a server run exited 0, wrote a
# durable empty analysis, and advised that a run with two conditions and 24
# records "has no non-baseline condition". The hole was identical here.


def test_a_measurement_verb_refuses_a_foreign_run_whose_snapshot_matches(
        tmp_path):
    from steerlab_server.experiment.run_epoch import foreign_substrate_refusal
    live = _manifest()
    directory = _run_with_snapshot(tmp_path, "swift-mlx", live)
    for verb in ("analyze", "evaluate", "rescore-style"):
        refusal, unverified, drift = epoch_refusal(
            verb, "s", live.content_hash(), directory, allow_unverified=False,
            live_manifest=live, tolerate_measurement_drift=True,
            refuse_foreign_substrate=True)
        assert refusal == foreign_substrate_refusal(
            verb, os.path.basename(directory), "swift-mlx")
        assert f"Run {verb} on the engine that produced the run" in refusal
        assert (unverified, drift) == (False, None)
    # promote is the OTHER family and is deliberately untouched: it reads
    # engine-neutral selection metadata, and a cluster workspace's every
    # sweep is foreign.
    refusal, _, _ = epoch_refusal(
        "promote", "s", live.content_hash(), directory, allow_unverified=False,
        live_manifest=live)
    assert refusal is None, refusal


def test_allow_unverified_does_not_excuse_a_foreign_substrate(tmp_path):
    """The flag forgives a MISSING stamp, never a substrate this engine
    cannot read — the run stays just as unreadable."""
    live = _manifest()
    directory = tmp_path / "unstamped"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"runType": "run", "substrate": "swift-mlx"}),
        encoding="utf-8")
    refusal, unverified, _ = epoch_refusal(
        "analyze", "s", live.content_hash(), str(directory),
        allow_unverified=True, live_manifest=live,
        tolerate_measurement_drift=True, refuse_foreign_substrate=True)
    assert refusal is not None
    assert "was produced on swift-mlx" in refusal
    assert unverified is False


def test_a_native_run_is_unaffected_by_the_foreign_refusal(tmp_path):
    """The guard must not have become stricter for the engine that wrote the
    run."""
    live = _manifest()
    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(
        {"runType": "run", "substrate": _ENGINE,
         "experimentHash": live.content_hash()}), encoding="utf-8")
    refusal, unverified, drift = epoch_refusal(
        "analyze", "s", live.content_hash(), str(directory),
        allow_unverified=False, live_manifest=live,
        tolerate_measurement_drift=True, refuse_foreign_substrate=True)
    assert (refusal, unverified, drift) == (None, False, None)


def test_the_foreign_refusal_sentence_is_the_cross_engine_literal():
    """Byte-identical to Swift's `RunEpoch.foreignSubstrateRefusal`. The two
    engines' agents read the same sentence, so it is pinned on both sides."""
    from steerlab_server.experiment.run_epoch import foreign_substrate_refusal
    assert foreign_substrate_refusal("analyze", "r", "swift-mlx") == (
        "analyze: source run 'r' was produced on swift-mlx, and analyze "
        "reads that run's own records — record schemas and pairing keys are "
        "per-engine, so measuring them here reports an empty result as a "
        "success. Run analyze on the engine that produced the run")
